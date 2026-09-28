# Multi-Omics BRCA Cancer Subtype Prediction

A machine learning pipeline that integrates RNA-seq, DNA methylation, somatic mutations and receptor status from TCGA-BRCA to classify breast cancer PAM50 molecular subtypes, and measures how much each omics layer contributes.

---

## Biological Question

Breast cancer has five PAM50 molecular subtypes (Luminal A, Luminal B, HER2-enriched, Basal-like, Normal-like) with different prognoses and treatments. Two questions:

1. How well can the subtype be predicted when all preprocessing is kept strictly inside cross-validation?
2. **Does adding methylation, mutation and clinical data help beyond expression, and can the non-expression layers recover the subtype on their own?**

PAM50 calls are themselves derived from RNA-seq expression, so an RNA-based model is partly re-learning the label definition. That is why the per-layer ablation below is the main result, not a single accuracy number.

---

## Results

489 primary tumours with all three omics layers and a PAM50 call (LumA 249, LumB 117, Basal 78, Her2 30, Normal 15). Repeated stratified 5-fold CV (5 repeats = 25 folds), mean ± SD. Balanced accuracy is the headline metric because the classes are imbalanced.

| Omics layers | Logistic regression: balanced acc. | Logistic regression: accuracy | XGBoost: balanced acc. | XGBoost: accuracy |
|---|---|---|---|---|
| RNA-seq only | **0.765 ± 0.069** | **0.885 ± 0.022** | 0.684 ± 0.063 | 0.820 ± 0.037 |
| Methylation only | 0.694 ± 0.065 | 0.791 ± 0.030 | 0.586 ± 0.068 | 0.739 ± 0.028 |
| Mutation only | 0.382 ± 0.024 | 0.608 ± 0.024 | 0.388 ± 0.062 | 0.536 ± 0.035 |
| Receptor status (ER/PR/HER2) only | 0.431 ± 0.049 | 0.634 ± 0.031 | 0.428 ± 0.048 | 0.642 ± 0.033 |
| Methylation + mutation | 0.685 ± 0.066 | 0.773 ± 0.029 | 0.589 ± 0.072 | 0.742 ± 0.026 |
| RNA + methylation + mutation | 0.748 ± 0.090 | 0.855 ± 0.034 | 0.673 ± 0.066 | 0.804 ± 0.028 |
| All layers + receptor status | 0.757 ± 0.066 | 0.860 ± 0.027 | 0.683 ± 0.055 | 0.807 ± 0.036 |

Random forest on all layers: balanced accuracy 0.564 ± 0.064, accuracy 0.760 ± 0.031. Full table with macro-F1: [`results/cv_results.md`](results/cv_results.md).

Final model (all layers, logistic regression), out-of-fold predictions for every patient:

| Subtype | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Basal | 0.962 | 0.962 | 0.962 | 78 |
| Her2 | 0.852 | 0.767 | 0.807 | 30 |
| LumA | 0.878 | 0.928 | 0.902 | 249 |
| LumB | 0.814 | 0.786 | 0.800 | 117 |
| Normal | 0.625 | 0.333 | 0.435 | 15 |

### Key Findings
- **Expression carries almost all of the signal.** Adding methylation, mutation and receptor status to RNA-seq did not improve on RNA alone; the differences are within one SD. That's expected, because PAM50 is defined from expression.
- **Methylation alone recovers the subtype reasonably well** (79% accuracy, balanced 0.69) without any expression data, so methylation patterns track the subtypes independently.
- **Mutations and receptor status alone are weak** (balanced accuracy ~0.4) and add nothing on top of methylation.
- **Logistic regression beats XGBoost and random forest in every combination.** With ~390 training patients and ~110 PCA features, the simpler linear model generalises better.
- **Basal is the most separable subtype; Normal-like is the hardest** (15 patients, recall 0.33). LumA vs LumB confusion is expected because they sit on a proliferation continuum.

---

## Pipeline

```
UCSC Xena (GDC TCGA-BRCA)                       multiomics/data.py  (label-free, per patient)
  STAR counts ─────────► log2(CPM+1)  (library-size normalised)
  450k methylation ───► beta values, CpG probes
  somatic mutations ──► non-silent calls -> binary gene matrix + mutated-gene count
  clinical matrix ────► PAM50 label; ER/PR/HER2 (HER2: IHC, FISH if equivocal)
        │  primary tumour samples only (TCGA -01), patients with all layers
        ▼
Model pipeline, fit on training folds only        multiomics/features.py
  RNA-seq      top-5000 variance -> impute -> scale -> PCA(50)
  Methylation  top-5000 variance -> impute -> scale -> PCA(50)
  Mutation     genes mutated in >= 5% of training patients + mutated-gene count
  Receptors    most-frequent imputation + was-missing indicators
        ▼
  concatenate -> classifier (logistic regression / XGBoost / random forest)
        ▼
train.py    repeated 5x5 CV for every omics combination, then fits and saves the final model
predict.py  loads the saved model and predicts subtype + probabilities
```

### What changed from the first version (the notebook)
`project.ipynb` is the original exploratory analysis. It reported 81.6% test accuracy, but it had problems that this version fixes:
- **Label leakage:** missing ER/PR/HER2 values were filled with the most common value *of the patient's own PAM50 subtype*, so the features encoded the answer. Receptor status is now imputed without labels, inside each training fold.
- **Preprocessing leakage:** variance filtering, imputation, scaling and PCA were fit on all patients before the train/test split. They are now steps of one sklearn `Pipeline`, fit on training folds only.
- **Normal tissue mixed in:** samples were matched to patients by ID and duplicates dropped arbitrarily, so some "tumour" profiles were adjacent normal tissue. Only primary tumour samples (TCGA code 01) are used now.
- **No library-size normalisation:** raw counts were log-transformed directly (library sizes differ ~6x). Counts are now converted to CPM first.
- **Single tiny test split:** 98 test patients, with Her2 n=5 and Normal n=3. This is replaced by 25-fold repeated CV reporting mean ± SD and balanced accuracy.
- **Mutations:** silent variants are no longer counted; binary genes are used directly instead of PCA on 18 columns; the mutated-gene count uses all genes.
- **Receptor status:** ER/PR now come from the full-cohort clinical columns instead of the 2012 subset (in this cohort ER is missing for 30 patients instead of 57, PR for 33 instead of 58). HER2 comes from IHC with FISH as fallback (missing for 75 vs 63 with the 2012 column).
- **Ablation:** the claim that methylation adds complementary signal was never tested before. The table above tests it, and it doesn't hold beyond expression.

---

## Dataset

- **Source:** TCGA-BRCA via the [UCSC Xena GDC hub](https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Breast%20Cancer%20(BRCA)): STAR counts, Illumina 450k methylation beta values, WXS somatic mutations (GDC data release 40)
- **PAM50 labels and receptor status:** UCSC Xena legacy `BRCA_clinicalMatrix` (`PAM50Call_RNAseq`)
- **Cohort:** 489 patients with a primary tumour profiled on all three platforms and a PAM50 call

---

## Repository Structure

```
Multiomics/
├── multiomics/
│   ├── data.py          ← download raw files, build aligned dataset
│   └── features.py      ← per-omics preprocessing + classifier pipeline
├── train.py             ← cross-validation, ablation, final model
├── predict.py           ← predict subtypes with the saved model
├── tests/               ← pytest: leakage and preprocessing checks
├── results/             ← CV table, classification report, confusion matrix
├── project.ipynb        ← original exploratory notebook (v1, superseded)
├── requirements.txt
├── data/                ← raw + processed data (not tracked)
└── models/              ← trained model (not tracked)
```

---

## Reproducing the Analysis

```bash
conda create -n multiomics python=3.12
conda activate multiomics
pip install -r requirements.txt

python -m multiomics.data   # downloads ~3.1 GB from Xena, builds data/processed/ (~3 min)
python train.py             # 5x5 CV for all omics combinations + final model (~40 min)
python train.py --skip-cv   # refit the final model only
python -m pytest tests
```

The build needs ~8 GB of RAM (the methylation matrix is ~480k probes).

### Predicting new patients

```bash
python predict.py patients.csv
```

`patients.csv` has one row per patient: the first column is the patient ID, and the other columns are features named as in training (`rna:ENSG…`, `meth:cg…`, `mut:TP53`, `burden:n_mutated_genes`, `clin:ER`/`clin:PR`/`clin:HER2`). Only the ~10k features listed in `models/metadata.json` → `required_features` are needed. Receptor status may be left out, since it's imputed.

---

## Author

Durga Gomathi Arumuganainar
MS Bioinformatics, Northeastern University
