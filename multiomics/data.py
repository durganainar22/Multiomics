"""Build the aligned multi-omics dataset from the raw UCSC Xena files.

Run once:  python -m multiomics.data
Reads data/raw/, writes data/processed/.

Only label-free, per-patient steps happen here (sample choice, library-size
normalisation, mutation calling rules). Anything learned from the data
(variance filtering, imputation, scaling, PCA) happens inside the model
pipeline so it is fit on training folds only - see multiomics/features.py.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

XENA_GDC = "https://gdc-hub.s3.us-east-1.amazonaws.com/download"
XENA_LEGACY = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"
RAW_FILES = {
    "rna": ("TCGA-BRCA.star_counts.tsv.gz", f"{XENA_GDC}/TCGA-BRCA.star_counts.tsv.gz"),
    "meth": ("TCGA-BRCA.methylation450.tsv.gz", f"{XENA_GDC}/TCGA-BRCA.methylation450.tsv.gz"),
    "mut": ("TCGA-BRCA.somaticmutation_wxs.tsv.gz", f"{XENA_GDC}/TCGA-BRCA.somaticmutation_wxs.tsv.gz"),
    "clinical": ("BRCA_clinicalMatrix.tsv", f"{XENA_LEGACY}/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix"),
}

# Variant consequences that change the protein; everything else (synonymous,
# intronic, UTR, up/downstream, ...) is not counted as a mutation.
NON_SILENT = {
    "missense_variant", "stop_gained", "frameshift_variant", "splice_acceptor_variant",
    "splice_donor_variant", "inframe_deletion", "inframe_insertion", "stop_lost",
    "start_lost", "protein_altering_variant", "incomplete_terminal_codon_variant",
}

STATUS = {"Positive": 1.0, "Negative": 0.0}


def patient_id(barcode: str) -> str:
    return barcode[:12]


def primary_tumour_samples(barcodes) -> dict:
    """Map patient -> one primary-tumour sample barcode.

    TCGA sample type 01 = primary solid tumour (11 = normal tissue,
    06 = metastasis). If a patient has several (01A, 01B), take vial A first.
    """
    chosen = {}
    for b in sorted(barcodes):
        if b[13:15] == "01" and patient_id(b) not in chosen:
            chosen[patient_id(b)] = b
    return chosen


def read_header(path: Path) -> list:
    with gzip.open(path, "rt") as f:
        return f.readline().rstrip("\n").split("\t")


def load_clinical(path: Path) -> pd.DataFrame:
    """PAM50 label + ER/PR/HER2 status for primary tumours with a PAM50 call.

    Uses the full-cohort receptor columns (breast_carcinoma_*), not the
    *_nature2012 ones, which only cover the 2012 subset.
    HER2: IHC Positive/Negative; if IHC is equivocal or missing, use FISH.
    Missing values stay NaN - they are imputed inside the model pipeline.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["sampleID"].str[13:15] == "01"]
    df = df[df["PAM50Call_RNAseq"].notna()].copy()
    df["patient"] = df["sampleID"].map(patient_id)
    df = df.drop_duplicates("patient").set_index("patient")

    her2_ihc = df["lab_proc_her2_neu_immunohistochemistry_receptor_status"].map(STATUS)
    her2_fish = df["lab_procedure_her2_neu_in_situ_hybrid_outcome_type"].map(STATUS)
    return pd.DataFrame({
        "PAM50": df["PAM50Call_RNAseq"],
        "ER": df["breast_carcinoma_estrogen_receptor_status"].map(STATUS),
        "PR": df["breast_carcinoma_progesterone_receptor_status"].map(STATUS),
        "HER2": her2_ihc.fillna(her2_fish),
    })


def log_cpm(log2_counts: pd.DataFrame) -> pd.DataFrame:
    """Xena log2(count+1) (genes x samples) -> log2(CPM+1).

    Library sizes differ ~6x between samples, so raw counts would make
    sequencing depth look like biology.
    """
    counts = np.expm1(log2_counts * np.log(2))
    cpm = counts / counts.sum(axis=0) * 1e6
    return np.log2(cpm + 1)


def load_rna(path: Path, samples: dict) -> pd.DataFrame:
    cols = list(samples.values())
    df = pd.read_csv(path, sep="\t", index_col=0, usecols=["Ensembl_ID", *cols])
    df = log_cpm(df[cols]).astype(np.float32)
    df.columns = list(samples.keys())
    return df.T  # patients x genes


def load_methylation(path: Path, samples: dict) -> pd.DataFrame:
    """Beta values, CpG probes only. ~485k probes, read in chunks."""
    cols = list(samples.values())
    ref = "Composite Element REF"
    chunks = []
    reader = pd.read_csv(path, sep="\t", index_col=0, usecols=[ref, *cols],
                         dtype={c: np.float32 for c in cols}, chunksize=50_000)
    for chunk in reader:
        chunks.append(chunk.loc[chunk.index.str.startswith("cg"), cols])
        print(f"  methylation: {sum(len(c) for c in chunks):,} probes read", flush=True)
    df = pd.concat(chunks)
    df.columns = list(samples.keys())
    return df.T  # patients x probes


def load_mutations(path: Path) -> pd.DataFrame:
    """Non-silent mutation calls for primary tumours: one row per (patient, gene)."""
    df = pd.read_csv(path, sep="\t", usecols=["sample", "gene", "effect"])
    samples = primary_tumour_samples(df["sample"].unique())
    df = df[df["sample"].isin(set(samples.values()))]
    non_silent = df["effect"].fillna("").str.split(";").map(lambda terms: bool(NON_SILENT & set(terms)))
    df = df[non_silent].assign(patient=lambda d: d["sample"].map(patient_id))
    return df[["patient", "gene"]].drop_duplicates(), set(samples)


def mutation_matrix(calls: pd.DataFrame, patients: list) -> pd.DataFrame:
    """Binary patients x genes matrix (1 = gene carries a non-silent mutation)."""
    calls = calls[calls["patient"].isin(patients)]
    m = pd.crosstab(calls["patient"], calls["gene"]).clip(upper=1)
    return m.reindex(patients, fill_value=0).astype(np.float32)


def download_raw() -> None:
    import urllib.request
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in RAW_FILES.values():
        if not (RAW / name).exists():
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, RAW / name)


def build() -> None:
    download_raw()
    files = {k: RAW / name for k, (name, _) in RAW_FILES.items()}

    clinical = load_clinical(files["clinical"])
    rna_samples = primary_tumour_samples(read_header(files["rna"])[1:])
    meth_samples = primary_tumour_samples(read_header(files["meth"])[1:])
    mut_calls, mut_patients = load_mutations(files["mut"])

    # patients with all three omics layers AND a PAM50 label; patients missing
    # from the mutation file were not sequenced (not "zero mutations")
    cohort = sorted(set(clinical.index) & set(rna_samples) & set(meth_samples) & mut_patients)
    print(f"cohort: {len(cohort)} patients (RNA {len(rna_samples)}, methylation "
          f"{len(meth_samples)}, mutation {len(mut_patients)}, PAM50 {len(clinical)})")

    blocks = {
        "rna": load_rna(files["rna"], {p: rna_samples[p] for p in cohort}),
        "meth": load_methylation(files["meth"], {p: meth_samples[p] for p in cohort}),
    }
    mut = mutation_matrix(mut_calls, cohort)
    blocks["mut"] = mut
    # burden = number of genes with a non-silent mutation (all genes, not just
    # the recurrent ones kept for modelling)
    blocks["burden"] = mut.sum(axis=1).to_frame("n_mutated_genes")
    blocks["clin"] = clinical.loc[cohort, ["ER", "PR", "HER2"]].astype(np.float32)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    names = {}
    for name, df in blocks.items():
        assert list(df.index) == cohort
        np.save(PROCESSED / f"{name}.npy", np.ascontiguousarray(df.to_numpy(np.float32)))
        names[name] = [str(c) for c in df.columns]
        print(f"  {name}: {df.shape[1]:,} features")
    (PROCESSED / "features.json").write_text(json.dumps(names))
    clinical.loc[cohort, ["PAM50"]].to_csv(PROCESSED / "labels.csv")
    print("labels:", clinical.loc[cohort, "PAM50"].value_counts().to_dict())


def load_dataset(processed: Path = PROCESSED):
    """Returns X (float32 array, patients x all features), y (PAM50 labels,
    indexed by patient), columns (feature names 'block:feature') and blocks
    (block name -> slice of X columns).

    X is a plain array, not a DataFrame: sklearn inspects DataFrame columns one
    by one, which is slow with ~480k methylation probes."""
    names = json.loads((processed / "features.json").read_text())
    labels = pd.read_csv(processed / "labels.csv", index_col=0)["PAM50"]
    arrays, columns, blocks = [], [], {}
    for block, feats in names.items():
        arrays.append(np.load(processed / f"{block}.npy"))
        blocks[block] = slice(len(columns), len(columns) + len(feats))
        columns += [f"{block}:{f}" for f in feats]
    return np.hstack(arrays), labels, columns, blocks


if __name__ == "__main__":
    build()
