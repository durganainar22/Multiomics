"""Evaluate every omics combination with repeated cross-validation, then fit
the final model on all patients and save it.

    python -m multiomics.data     # once: build data/processed/
    python train.py               # 5x5 repeated CV + final model
    python train.py --repeats 1   # quicker check
    python train.py --skip-cv     # refit final model only

Writes results/ (CV table, confusion matrix, report) and models/ (model.joblib
+ metadata.json used by predict.py).
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.preprocessing import LabelEncoder

from multiomics.data import load_dataset
from multiomics.features import CONFIGS, build_pipeline, required_features

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

SCORING = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
           "macro_f1": "f1_macro"}
# ablation: every omics combination with a linear and a tree model;
# random forest only on the full feature set, for comparison
EXPERIMENTS = [(c, m) for c in CONFIGS for m in ("lr", "xgb")] + [("all", "rf")]


def run_cv(X, y, blocks, repeats):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)
    rows = []
    for config, model in EXPERIMENTS:
        t = time.time()
        scores = cross_validate(build_pipeline(config, blocks, model), X, y,
                                cv=cv, scoring=SCORING)
        row = {"config": config, "model": model, "n_folds": len(scores["fit_time"])}
        for metric in SCORING:
            s = scores[f"test_{metric}"]
            row[f"{metric}_mean"], row[f"{metric}_sd"] = s.mean(), s.std()
        rows.append(row)
        print(f"{config:>13} {model:>4}  bal.acc {row['balanced_accuracy_mean']:.3f} "
              f"+/- {row['balanced_accuracy_sd']:.3f}  acc {row['accuracy_mean']:.3f}  "
              f"({time.time() - t:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def write_summary(results, classes, counts):
    lines = ["# Cross-validation results", "",
             f"Repeated stratified 5-fold CV. {sum(counts.values())} patients: "
             + ", ".join(f"{c} {counts[c]}" for c in classes) + ".", "",
             "| Omics | Model | Balanced accuracy | Accuracy | Macro-F1 |",
             "|---|---|---|---|---|"]
    for r in results.itertuples():
        lines.append(f"| {r.config} | {r.model} | {r.balanced_accuracy_mean:.3f} ± "
                     f"{r.balanced_accuracy_sd:.3f} | {r.accuracy_mean:.3f} ± {r.accuracy_sd:.3f} "
                     f"| {r.macro_f1_mean:.3f} ± {r.macro_f1_sd:.3f} |")
    (RESULTS / "cv_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def out_of_fold_report(pipeline, X, y, le):
    """Every patient predicted by a model that never saw them (one 5-fold pass)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(pipeline, X, y, cv=cv)
    report = classification_report(y, pred, target_names=le.classes_, digits=3)
    (RESULTS / "classification_report.txt").write_text(report)
    print(report)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(le.inverse_transform(y), le.inverse_transform(pred),
                                            labels=le.classes_, cmap="Blues", ax=ax)
    ax.set_title("Out-of-fold predictions (5-fold CV)")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--final-config", default="all", choices=list(CONFIGS))
    # logistic regression beat XGBoost and RF in every omics combination
    ap.add_argument("--final-model", default="lr", choices=["lr", "rf", "xgb"])
    ap.add_argument("--skip-cv", action="store_true",
                    help="reuse results/cv_results.csv, only refit the final model")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    X, labels, columns, blocks = load_dataset()
    le = LabelEncoder()
    y = le.fit_transform(labels)   # XGBoost needs integer classes
    counts = labels.value_counts().to_dict()
    print(f"{X.shape[0]} patients x {X.shape[1]:,} input features; {counts}")

    if args.skip_cv:
        results = pd.read_csv(RESULTS / "cv_results.csv")
    else:
        results = run_cv(X, y, blocks, args.repeats)
        results.to_csv(RESULTS / "cv_results.csv", index=False)
        write_summary(results, le.classes_, counts)

    final = build_pipeline(args.final_config, blocks, args.final_model)
    out_of_fold_report(final, X, y, le)

    final.fit(X, y)
    # the pipeline works on a plain array, so the column names travel with it
    joblib.dump({"pipeline": final, "columns": columns}, MODELS / "model.joblib", compress=3)
    best = results[(results.config == args.final_config) & (results.model == args.final_model)]
    (MODELS / "metadata.json").write_text(json.dumps({
        "config": args.final_config,
        "model": args.final_model,
        "classes": le.classes_.tolist(),
        "n_patients": int(X.shape[0]),
        "required_features": required_features(final, columns),
        "cv": best.drop(columns=["config", "model"]).iloc[0].to_dict(),
    }, indent=1))
    print(f"saved {MODELS / 'model.joblib'}")


if __name__ == "__main__":
    main()
