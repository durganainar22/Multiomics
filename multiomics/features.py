"""Model pipeline: per-omics preprocessing + classifier, as one sklearn object.

Everything that learns from data (which genes/probes are most variable, mean
values for imputation, scaling, PCA axes) is a pipeline step, so during
cross-validation it is fit on the training folds only and the held-out fold
never influences it. No step ever sees the labels except the classifier.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data
from xgboost import XGBClassifier

# which omics blocks each experiment uses (the ablation)
CONFIGS = {
    "rna": ["rna"],
    "meth": ["meth"],
    "mut": ["mut", "burden"],
    "clin": ["clin"],
    "meth+mut": ["meth", "mut", "burden"],
    "rna+meth+mut": ["rna", "meth", "mut", "burden"],
    "all": ["rna", "meth", "mut", "burden", "clin"],
}

N_TOP_FEATURES = 5000   # most variable genes / probes kept per fold
N_PCS = 50              # PCs per expression / methylation block
MIN_MUT_FRAC = 0.05     # keep genes mutated in >= 5% of training patients


class _NanSelector(SelectorMixin, BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


class TopVarianceSelector(_NanSelector):
    """Keep the k features with the highest variance (missing values ignored)."""

    def __init__(self, k=N_TOP_FEATURES):
        self.k = k

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=np.float32)
        with np.errstate(all="ignore"):
            var = np.nanvar(X, axis=0)
        var = np.nan_to_num(var, nan=-np.inf)   # all-missing features rank last
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[np.argsort(var)[::-1][: self.k]] = True
        return self


class MinFrequencySelector(_NanSelector):
    """Keep binary features that are 1 in at least `min_frac` of rows."""

    def __init__(self, min_frac=MIN_MUT_FRAC):
        self.min_frac = min_frac

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan", dtype=np.float32)
        self.support_ = np.nanmean(X, axis=0) >= self.min_frac
        return self


def _omics_block():
    return Pipeline([
        ("top_variance", TopVarianceSelector(k=N_TOP_FEATURES)),
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=N_PCS, random_state=42)),
    ])


def _block_transformer(block):
    if block in ("rna", "meth"):
        return _omics_block()
    if block == "mut":
        return MinFrequencySelector(min_frac=MIN_MUT_FRAC)
    if block == "burden":
        return "passthrough"
    if block == "clin":
        # label-free: most common value in the training folds, plus a
        # was-missing flag per receptor
        return SimpleImputer(strategy="most_frequent", add_indicator=True)
    raise ValueError(f"unknown block {block!r}")


def make_classifier(name):
    if name == "lr":
        return LogisticRegression(max_iter=5000, random_state=42)
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    if name == "xgb":
        return XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6,
                             random_state=42, n_jobs=-1, eval_metric="mlogloss")
    raise ValueError(f"unknown model {name!r}")


def build_pipeline(config, blocks, model="xgb"):
    """config: key of CONFIGS; blocks: block name -> slice of X columns."""
    preprocess = ColumnTransformer(
        [(b, _block_transformer(b), blocks[b]) for b in CONFIGS[config]],
        remainder="drop",
    )
    steps = [("preprocess", preprocess)]
    if model == "lr":   # PCs, binary genes and counts are on different scales
        steps.append(("scale", StandardScaler()))
    steps.append(("model", make_classifier(model)))
    return Pipeline(steps)


def required_features(fitted_pipeline, columns):
    """Names of the input columns the fitted model actually uses (after
    selection). Everything else in the input is dropped by the selectors."""
    columns = np.asarray(columns)
    used = []
    for name, step, cols in fitted_pipeline.named_steps["preprocess"].transformers_:
        if name == "remainder":
            continue
        names = columns[cols]
        selector = step.steps[0][1] if isinstance(step, Pipeline) else step
        if isinstance(selector, _NanSelector):
            names = names[selector.get_support()]
        used += names.tolist()
    return used
