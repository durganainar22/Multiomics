import numpy as np
import pandas as pd
import pytest

from multiomics import features
from multiomics.data import log_cpm, primary_tumour_samples
from multiomics.features import build_pipeline, required_features


def test_primary_tumour_only_and_vial_a_first():
    barcodes = ["TCGA-AA-0001-11A", "TCGA-AA-0001-01B", "TCGA-AA-0001-01A",
                "TCGA-AA-0002-06A", "TCGA-AA-0003-11A"]
    assert primary_tumour_samples(barcodes) == {"TCGA-AA-0001": "TCGA-AA-0001-01A"}


def test_log_cpm_normalises_library_size():
    counts = pd.DataFrame({"deep": [100.0, 300.0], "shallow": [10.0, 30.0]})
    out = log_cpm(np.log2(counts + 1))
    assert np.allclose(out["deep"], out["shallow"])   # same composition, same CPM


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(features, "N_TOP_FEATURES", 20)
    monkeypatch.setattr(features, "N_PCS", 5)
    rng = np.random.default_rng(0)
    n = 60
    y = np.repeat([0, 1, 2], n // 3)
    rna = rng.normal(size=(n, 40)) + y[:, None] * 0.5
    meth = rng.uniform(size=(n, 50))
    meth[rng.uniform(size=meth.shape) < 0.05] = np.nan
    mut = (rng.uniform(size=(n, 10)) < 0.2).astype(float)
    clin = rng.choice([0.0, 1.0, np.nan], size=(n, 3))
    cols = ([f"rna:g{i}" for i in range(40)] + [f"meth:cg{i}" for i in range(50)]
            + [f"mut:G{i}" for i in range(10)] + ["burden:n_mutated_genes"]
            + ["clin:ER", "clin:PR", "clin:HER2"])
    X = np.hstack([rna, meth, mut, mut.sum(1, keepdims=True), clin]).astype(np.float32)
    blocks = {"rna": slice(0, 40), "meth": slice(40, 90), "mut": slice(90, 100),
              "burden": slice(100, 101), "clin": slice(101, 104)}
    return X, y, cols, blocks


def test_preprocessing_never_uses_labels(toy):
    """Same X with shuffled labels must give identical features: only the
    classifier may see y (the old notebook imputed ER/PR/HER2 from the label)."""
    X, y, _, blocks = toy
    prep = lambda labels: (build_pipeline("all", blocks, "lr")
                           .named_steps["preprocess"].fit(X, labels).transform(X))
    shuffled = np.random.default_rng(1).permutation(y)
    assert np.allclose(prep(y), prep(shuffled))


def test_predict_needs_only_required_features(toy, tmp_path):
    import joblib, json
    from predict import SubtypeModel
    X, y, cols, blocks = toy
    pipe = build_pipeline("all", blocks, "xgb").fit(X, y)
    joblib.dump({"pipeline": pipe, "columns": cols}, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"classes": ["A", "B", "C"], "required_features": required_features(pipe, cols)}))
    model = SubtypeModel(tmp_path)
    df = pd.DataFrame(X, columns=cols)
    slim = df[[c for c in model.required if not c.startswith("clin:")]].iloc[:5]
    # same patients with every column present must give the same answer
    assert np.allclose(model.predict(slim).iloc[:, 1:],
                       model.predict(df.iloc[:5].drop(columns=["clin:ER", "clin:PR", "clin:HER2"])).iloc[:, 1:])
    out = model.predict(slim)
    assert list(out.columns) == ["subtype", "A", "B", "C"]
    assert np.allclose(out[["A", "B", "C"]].sum(axis=1), 1)
    with pytest.raises(ValueError):
        model.predict(slim.drop(columns=slim.columns[0]))
