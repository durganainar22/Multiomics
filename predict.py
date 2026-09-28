"""Predict PAM50 subtypes with the saved model.

    python predict.py patients.csv          # rows = patients, columns = features
    python predict.py patients.parquet --out predictions.csv

Feature columns use the training names ('rna:ENSG...', 'meth:cg...',
'mut:TP53', 'burden:n_mutated_genes', 'clin:ER', ...). Only the columns in
models/metadata.json "required_features" are needed; others are ignored.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS = Path(__file__).resolve().parent / "models"


class SubtypeModel:
    def __init__(self, model_dir=MODELS):
        saved = joblib.load(model_dir / "model.joblib")
        self.pipeline, self.columns = saved["pipeline"], saved["columns"]
        meta = json.loads((model_dir / "metadata.json").read_text())
        self.classes = meta["classes"]
        self.required = meta["required_features"]

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """features: one row per patient. Returns subtype + class probabilities."""
        missing = sorted(set(self.required) - set(features.columns))
        # clinical receptor status may be unknown - the model imputes it
        missing = [c for c in missing if not c.startswith("clin:")]
        if missing:
            raise ValueError(f"{len(missing)} required features missing, e.g. {missing[:5]}")
        # put columns in training order; unused ones become NaN and are dropped
        X = features.reindex(columns=self.columns).to_numpy(np.float32)
        proba = self.pipeline.predict_proba(X)
        out = pd.DataFrame(proba, index=features.index, columns=self.classes)
        out.insert(0, "subtype", np.array(self.classes)[proba.argmax(axis=1)])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out")
    args = ap.parse_args()
    read = pd.read_parquet if args.input.endswith(".parquet") else pd.read_csv
    features = read(args.input)
    model = SubtypeModel()
    if features.columns[0] not in model.columns:
        features = features.set_index(features.columns[0])   # first column = patient ID
    result = model.predict(features)
    if args.out:
        result.to_csv(args.out)
    print(result.round(3).to_string())


if __name__ == "__main__":
    main()
