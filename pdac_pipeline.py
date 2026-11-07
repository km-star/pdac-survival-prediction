#!/usr/bin/env python3
"""
PDAC multi-omic vital-status prediction -- reproducibility pipeline (single file).

Implements what the manuscript's methods section describes, with every table
and figure generated from saved per-patient predictions:

  audit        data/column audit and label-version table (seconds)
  baseline     Step 2: the seven Molecular Twin classifiers, MOESM3 labels
  pipelines    Step 3: fold-internal pipelines P1-P6, MOESM3 labels
  ablation     A0-A4 on every analyte
  alpha_sweep  P1 at each fixed L1 alpha (the alpha-sensitivity figure)
  oct2021      seven classifiers + P1-P6 on the reconstructed "October-2021"
               labels (five longest-surviving deceased set to alive)  (opt-in)
  lvs_sim      label-version sensitivity under random label flips   (opt-in)
  nulltest     every method re-run on shuffled labels: the null
               distribution for "above chance" and "Step 3 > Step 2" (opt-in)
  report       metrics, statistics, LaTeX tables, figures

Step 1 (reproduction) is the seven classifiers exactly as released on Zenodo
(record 8423595, MTPilot_multiomic_model_training.py), run with fixed seeds and
compared with the published Table 1 confusion matrices (table2_reproduction).

Design rules
  * Leave-one-patient-out outer loop. Everything that looks at data
    (imputation, scaling, feature selection, resampling, hyper-parameter
    tuning, model selection) is fitted on the n-1 training patients only.
  * Hyper-parameters are tuned by inner stratified 3-fold CV inside each
    outer fold. "Best model" columns (BASE_AUTO, PIPE_AUTO) pick, per outer
    fold, the method with the highest inner-CV balanced accuracy on the
    training patients -- never by looking at the held-out predictions.
  * Balanced accuracy means (sensitivity + specificity) / 2. Plain accuracy is
    reported next to it under its own name.
  * Runs are cached per (labels, analyte, method, seed); re-running resumes.
  * --cv loo is the manuscript's protocol; --cv rskf (repeated stratified
    k-fold, separate --outdir) is the lower-variance robustness run.

Usage
  python pdac_pipeline.py --data 43018_2023_697_MOESM3_ESM.xlsx --outdir results --parts audit
  python pdac_pipeline.py --data ... --outdir results --parts all --n-jobs 32
  python pdac_pipeline.py --data ... --outdir results --smoke          # ~minutes
See README.md for HPC and Colab instructions.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFECV, SelectorMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

__version__ = "3.0.0"

# =============================================================================
# 1. CONSTANTS
# =============================================================================

TARGET, DAYS, PATIENT_ID = "label_deceased", "label_days_to_death", "Biobank_Number"

ANALYTE_PREFIXES = {  # column prefixes in MOESM3
    "Clinical": "clinical_", "Plasma_Protein": "plasma_protein_",
    "Tissue_Protein": "tissue_protein_", "Plasma_Lipid": "plasma_lipid_",
    "Pathology": "pathology_NF", "CNV": "CNV_", "SNV": "freebayes_SNV_",
    "INDEL": "pindel_INDEL_", "RNA": "rna_expr_", "RNA_Fusion": "AF4_",
}
GENOMIC = {"SNV", "CNV", "INDEL"}          # missing values mean "no call" -> 0
ANALYTES = ["Clinical", "SNV", "INDEL", "CNV", "RNA", "RNA_Fusion", "Pathology",
            "Plasma_Protein", "Plasma_Lipid", "Tissue_Protein", "Multiomic"]

# Osipov et al., Nat. Cancer 2024, Table 1: published LOOCV confusion matrices.
# Positive class = deceased. Accuracy, balanced accuracy and the number of
# deceased patients behind each published row all follow from these counts.
PUBLISHED_T1 = {
    "Multiomic":      dict(TP=26, FP=4,  TN=7,  FN=2),
    "Plasma_Protein": dict(TP=32, FP=8,  TN=6,  FN=5),
    "RNA_Fusion":     dict(TP=35, FP=12, TN=8,  FN=2),
    "Tissue_Protein": dict(TP=32, FP=10, TN=4,  FN=3),
    "Plasma_Lipid":   dict(TP=34, FP=12, TN=2,  FN=3),
    "Clinical":       dict(TP=47, FP=19, TN=5,  FN=3),
    "RNA":            dict(TP=33, FP=14, TN=6,  FN=4),
    "Pathology":      dict(TP=34, FP=11, TN=13, FN=13),
    "CNV":            dict(TP=43, FP=20, TN=4,  FN=5),
    "INDEL":          dict(TP=39, FP=17, TN=7,  FN=9),
    "SNV":            dict(TP=45, FP=23, TN=1,  FN=3),
}

BASE_MODELS = ["SVM", "L1_SVM", "L1_RF", "PCA_LR", "DNN", "RFE_LR", "RFE_RF"]
PIPELINES = ["P1", "P2", "P3", "P4", "P5", "P6"]
ABLATION = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "P1", "A4": "RFE_RF"}  # config -> run
PRETTY = {
    "P1": "P1 L1+SMOTE+RF", "P2": "P2 RFECV+XGB", "P3": "P3 Stacking",
    "P4": "P4 t-test+SMOTE+RF", "P5": "P5 L1+NearMiss+RF", "P6": "P6 LGBM-DART",
    "A0": "A0 RF only", "A1": "A1 L1+RF", "A2": "A2 SMOTE+RF",
    "A3": "A3 L1+SMOTE+RF", "A4": "A4 RFECV+RF",
    "BASE_AUTO": "Step 2 (nested best of 7)", "PIPE_AUTO": "Step 3 (nested best of 6)",
}

# The seven classifiers follow the released Zenodo script line by line
# (StandardScaler first in every pipeline; RandomForest defaults = 100 trees):
#   SVM     SGDClassifier()                      (hinge, L2, alpha 1e-4)
#   L1_SVM  SGDClassifier(alpha=0.05, penalty="l1")
#   L1_RF   SelectFromModel(SGD L1 0.05) + RandomForestClassifier()
#   PCA_LR  PCA(20) + LogisticRegression(max_iter=1000)
#   DNN     SelectFromModel(SGD L1 0.05) + MLP(512, 256, 128, 64, 32)
#   RFE_LR  RFECV(LogisticRegression(max_iter=1000), step=0.2)   (5-fold inside)
#   RFE_RF  RFECV(RandomForestClassifier(), step=0.2)            (5-fold inside)
# The released script fixes no random seeds; this one does (--seeds).

SUSPICIOUS = re.compile(
    r"(deceas|death|dead|surviv|vital|follow|recur|relaps|progress|last_contact|"
    r"days_to|outcome|status_at)", re.I)


@dataclass
class Cfg:
    rf_trees: int = 500            # P1/P4/P5/A*, L1_RF
    released_rf_trees: int = 100   # RandomForestClassifier() default, as in the released code
    boost_trees: int = 300         # XGBoost / LightGBM
    alphas: tuple = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
    k_grid: tuple = (50, 100, 200, 500)
    nm_ratios: tuple = (1.0, 1.5, 2.0)   # deceased kept per alive patient
    inner_cv: int = 3
    stack_cv: int = 3
    lgbm_min_child: int = 5        # LightGBM min_child_samples (see P6)
    n_boot: int = 1000
    n_flip: int = 5
    cv: str = "loo"                # "loo" (the manuscript) or "rskf" (repeated stratified k-fold)
    cv_folds: int = 5
    cv_repeats: int = 10


# =============================================================================
# 2. DATA
# =============================================================================

def _sanitize(cols):
    seen, out = set(), []
    for c in cols:
        nc = re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z_]", "_", str(c))).strip("_") or "feat"
        base, i = nc, 1
        while nc in seen:
            nc, i = f"{base}_{i}", i + 1
        seen.add(nc)
        out.append(nc)
    return out


@dataclass
class View:
    name: str
    X: np.ndarray            # patients x features, NaN preserved
    patients: list
    cols: list
    zero_mask: np.ndarray    # True where a missing value should become 0
    nan_cols: np.ndarray     # True where the column has any NaN among these patients

    @property
    def n(self):
        return len(self.patients)


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    days: pd.Series | None
    blocks: dict
    audit: pd.DataFrame
    _views: dict = field(default_factory=dict)

    def view(self, name) -> View:
        if name not in self._views:
            if name == "Multiomic":
                cols = [c for a in ANALYTE_PREFIXES for c in self.blocks[a]]
                present = np.ones(len(self.X), bool)
                for a in ANALYTE_PREFIXES:
                    present &= self.X[self.blocks[a]].notna().any(axis=1).to_numpy()
                owner = {c: a for a in ANALYTE_PREFIXES for c in self.blocks[a]}
            else:
                cols = list(self.blocks[name])
                present = self.X[cols].notna().any(axis=1).to_numpy()
                owner = {c: name for c in cols}
            sub = self.X.loc[present, cols]
            Xv = sub.to_numpy(dtype=float)
            self._views[name] = View(
                name, Xv, list(sub.index), cols,
                np.array([owner[c] in GENOMIC for c in cols]), np.isnan(Xv).any(axis=0))
        return self._views[name]


def load_dataset(path, exclude_regex=None) -> Dataset:
    path = Path(path)
    raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path, sheet_name=0)
    if TARGET not in raw.columns:
        raise SystemExit(f"'{TARGET}' column not found in {path.name}")
    raw = raw[raw[TARGET].notna()].reset_index(drop=True)
    ids = (raw[PATIENT_ID].astype(str) if PATIENT_ID in raw.columns
           else pd.Series([f"P{i:03d}" for i in range(len(raw))]))
    if ids.duplicated().any():
        ids = ids + "_" + ids.groupby(ids).cumcount().astype(str)
    raw.index = ids.to_numpy()
    y = pd.to_numeric(raw[TARGET]).astype(int)
    days = pd.to_numeric(raw[DAYS], errors="coerce") if DAYS in raw.columns else None

    feats = raw.drop(columns=[c for c in (TARGET, DAYS, PATIENT_ID) if c in raw.columns])
    numeric = feats.apply(pd.to_numeric, errors="coerce")
    coerced = (feats.notna() & numeric.isna()).sum()
    clean = _sanitize(numeric.columns)
    orig = dict(zip(clean, map(str, numeric.columns)))
    numeric.columns = clean
    coerced.index = clean

    excl = re.compile(exclude_regex, re.I) if exclude_regex else None
    blocks, status = {}, {}
    for name, prefix in ANALYTE_PREFIXES.items():
        cols = []
        for c in clean:
            if c in status or not c.startswith(prefix):
                continue
            if excl is not None and excl.search(orig[c]):
                status[c] = (name, "excluded_by_regex")
            else:
                status[c] = (name, "used")
                cols.append(c)
        if not cols:
            raise SystemExit(f"No columns with prefix '{prefix}' ({name}). "
                             f"First columns: {list(feats.columns[:20])}")
        blocks[name] = cols
    rows = []
    for c in clean:
        a, st = status.get(c, ("", "outcome_column_excluded" if orig[c].lower().startswith("label_")
                                else "unassigned_excluded"))
        rows.append(dict(column=orig[c], analyte=a, status=st,
                         n_missing=int(numeric[c].isna().sum()),
                         n_non_numeric_lost=int(coerced[c]),
                         suspicious_name=bool(SUSPICIOUS.search(orig[c]))))
    return Dataset(numeric, y, days, blocks, pd.DataFrame(rows))


def reconstruct_oct2021(y, days, n_flip=5):
    """The manuscript's rule: the n_flip deceased patients with the longest
    days-to-death are set back to alive."""
    if days is None:
        raise SystemExit(f"'{DAYS}' column is required to reconstruct October-2021 labels")
    d = days[y == 1].dropna().sort_values(ascending=False, kind="stable")
    flipped = list(d.index[:n_flip])
    y_old = y.copy()
    y_old.loc[flipped] = 0
    return y_old, flipped


def published(name):
    c = PUBLISHED_T1[name]
    n_dead, n_alive = c["TP"] + c["FN"], c["TN"] + c["FP"]
    sens, spec = c["TP"] / n_dead, c["TN"] / n_alive
    return dict(n=n_dead + n_alive, n_dead=n_dead, n_alive=n_alive,
                ACC=(c["TP"] + c["TN"]) / (n_dead + n_alive), BA=(sens + spec) / 2,
                Sens=sens, Spec=spec, **c)


# =============================================================================
# 3. FOLD-INTERNAL BUILDING BLOCKS
# =============================================================================

class SafeSKF:
    """Shuffled stratified k-fold that lowers k when a class is too small."""

    def __init__(self, n_splits=3, seed=0):
        self.n_splits, self.seed = n_splits, seed

    def _k(self, y):
        return max(2, min(self.n_splits, int(np.bincount(np.asarray(y).astype(int)).min())))

    def split(self, X, y, groups=None):
        yield from StratifiedKFold(self._k(y), shuffle=True, random_state=self.seed).split(X, y)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._k(y) if y is not None else self.n_splits


class HalfMedianImputer(TransformerMixin, BaseEstimator):
    """Missing -> half the training-fold median (0 where zero_mask is set)."""

    def __init__(self, zero_mask=None):
        self.zero_mask = zero_mask

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        fill = np.where(np.isnan(med), 0.0, med / 2.0)
        if self.zero_mask is not None:
            fill = np.where(np.asarray(self.zero_mask, bool), 0.0, fill)
        self.fill_, self.n_features_in_ = fill, X.shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        r, c = np.where(np.isnan(X))
        X[r, c] = self.fill_[c]
        return X


class SafePCA(PCA):
    """PCA(20) as released, with n_components capped at what the training fold allows."""

    def _fit(self, X, *args, **kwargs):          # shared by fit and fit_transform
        self.n_components = min(self.n_components, X.shape[0], X.shape[1])
        return super()._fit(X, *args, **kwargs)


class L1HingeSelector(SelectorMixin, BaseEstimator):
    """Keep features with a non-zero coefficient in an L1-penalised hinge-loss
    linear model (SGD) -- SelectFromModel's rule for an L1 penalty (|coef| > 1e-5).
    If the penalty removes everything, the min_features largest coefficients are
    kept so that downstream steps always get input."""

    def __init__(self, alpha=0.005, max_iter=1000, random_state=None, min_features=1,
                 threshold=1e-5):
        self.alpha, self.max_iter = alpha, max_iter
        self.random_state, self.min_features, self.threshold = random_state, min_features, threshold

    def fit(self, X, y):
        est = SGDClassifier(loss="hinge", penalty="l1", alpha=self.alpha,
                            max_iter=self.max_iter, random_state=self.random_state)
        coef = np.abs(est.fit(X, y).coef_).ravel()
        mask = coef > self.threshold
        self.fallback_ = bool(mask.sum() < self.min_features)
        if self.fallback_:
            mask = np.zeros(coef.size, bool)
            mask[np.argsort(-coef, kind="stable")[: self.min_features]] = True
        self.mask_, self.n_features_in_ = mask, X.shape[1]
        return self

    def _get_support_mask(self):
        return self.mask_


class WelchTSelector(SelectorMixin, BaseEstimator):
    """Keep the k features with the largest |Welch t| between the two classes."""

    def __init__(self, k=100):
        self.k = k

    def fit(self, X, y):
        y = np.asarray(y)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = stats.ttest_ind(X[y == 1], X[y == 0], axis=0, equal_var=False).statistic
        t = np.nan_to_num(np.abs(np.asarray(t, dtype=float)), nan=0.0, posinf=0.0)
        mask = np.zeros(X.shape[1], bool)
        mask[np.argsort(-t, kind="stable")[: min(self.k, X.shape[1])]] = True
        self.mask_, self.n_features_in_ = mask, X.shape[1]
        return self

    def _get_support_mask(self):
        return self.mask_


class Resampled(ClassifierMixin, BaseEstimator):
    """Resample the training fold, then fit the classifier. Never touches test data.

    sampler='smote'     minority oversampled to parity, k = min(k_neighbors, n_min - 1)
    sampler='nearmiss3' majority undersampled to round(ratio * n_min) with NearMiss-3
    """

    def __init__(self, estimator, sampler="smote", ratio=1.0, k_neighbors=5, random_state=None):
        self.estimator, self.sampler, self.ratio = estimator, sampler, ratio
        self.k_neighbors, self.random_state = k_neighbors, random_state

    def _resample(self, X, y):
        counts = np.bincount(y, minlength=2)
        minority, majority = int(np.argmin(counts)), int(np.argmax(counts))
        n_min, n_maj = int(counts[minority]), int(counts[majority])
        if n_min < 2 or n_min == n_maj:
            return X, y
        if self.sampler == "smote":
            from imblearn.over_sampling import SMOTE
            s = SMOTE(k_neighbors=min(self.k_neighbors, n_min - 1), random_state=self.random_state)
        elif self.sampler == "nearmiss3":
            from imblearn.under_sampling import NearMiss
            keep = int(round(self.ratio * n_min))
            if keep >= n_maj:
                return X, y
            k = min(3, n_min)
            s = NearMiss(version=3, sampling_strategy={majority: keep},
                         n_neighbors=k, n_neighbors_ver3=k)
        else:
            raise ValueError(self.sampler)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return s.fit_resample(X, y)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y).astype(int)
        Xr, yr = self._resample(X, y)
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_, self.n_fit_ = self.estimator_.classes_, len(yr)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class PosWeighted(ClassifierMixin, BaseEstimator):
    """Set scale_pos_weight = n_alive / n_deceased from the training fold.
    Deceased is the positive class, so this down-weights the majority."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        self.scale_pos_weight_ = n_neg / n_pos if n_pos else 1.0
        self.estimator_ = clone(self.estimator).set_params(
            scale_pos_weight=self.scale_pos_weight_).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# =============================================================================
# 4. METHODS
# =============================================================================

@dataclass
class Ctx:
    seed: int
    p: int
    cfg: Cfg
    zero_mask: np.ndarray | None


def parse_method(method):
    """'P1@alpha=0.01' -> ('P1', ('alpha', 0.01)); 'P1' -> ('P1', None)."""
    fixed = None
    if "@" in method:
        method, kv = method.split("@")
        k, v = kv.split("=")
        fixed = (k, float(v))
    return method, fixed


def build_method(method, ctx: Ctx):
    """Return (estimator, param_grid or None). The estimator takes raw features.

    Every estimator starts with fold-internal imputation (a no-op on MOESM3, which
    has no missing cell inside any analyte block) and standardisation."""
    base, fixed = parse_method(method)
    rs, cfg = ctx.seed, ctx.cfg
    pre = [("impute", HalfMedianImputer(ctx.zero_mask)), ("scale", StandardScaler())]

    def rf(trees=cfg.rf_trees):
        return RandomForestClassifier(n_estimators=trees, max_features="sqrt",
                                      random_state=rs, n_jobs=1)

    def l1_released():                      # SelectFromModel(SGDClassifier(alpha=0.05, penalty="l1"))
        return L1HingeSelector(alpha=0.05, random_state=rs)

    def rfecv_released(est):                # RFECV(est, step=0.2): 5-fold, accuracy, as released
        return RFECV(est, step=0.2, cv=SafeSKF(5, rs), n_jobs=1)

    def pipe(*steps):
        return Pipeline(pre + list(steps))

    # ---- seven Molecular Twin classifiers, as released on Zenodo ---------------
    if base == "SVM":
        return pipe(("clf", SGDClassifier(random_state=rs))), None
    if base == "L1_SVM":
        return pipe(("clf", SGDClassifier(alpha=0.05, penalty="l1", random_state=rs))), None
    if base == "L1_RF":
        return pipe(("select", l1_released()), ("clf", rf(cfg.released_rf_trees))), None
    if base == "PCA_LR":
        return pipe(("pca", SafePCA(n_components=20, random_state=rs)),
                    ("clf", LogisticRegression(max_iter=1000, random_state=rs))), None
    if base == "DNN":
        return pipe(("select", l1_released()),
                    ("clf", MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64, 32),
                                          max_iter=1000, random_state=rs))), None
    if base == "RFE_LR":
        return pipe(("clf", rfecv_released(LogisticRegression(max_iter=1000, random_state=rs)))), None
    if base == "RFE_RF":
        return pipe(("clf", rfecv_released(rf(cfg.released_rf_trees)))), None

    # ---- proposed pipelines ---------------------------------------------------
    smote_rf = Resampled(rf(), sampler="smote", random_state=rs)
    if base in ("P1", "A3"):
        est = pipe(("select", L1HingeSelector(random_state=rs)), ("clf", smote_rf))
        if fixed:                                    # alpha sweep
            return est.set_params(select__alpha=fixed[1]), None
        return est, {"select__alpha": list(cfg.alphas)}
    if base == "P2":
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=cfg.boost_trees, max_depth=3, learning_rate=0.05,
                            reg_lambda=2.0, eval_metric="logloss", tree_method="hist",
                            random_state=rs, n_jobs=1, verbosity=0)
        sel = RFECV(LogisticRegression(max_iter=500, random_state=rs), step=0.2,
                    cv=SafeSKF(3, rs), scoring="balanced_accuracy", n_jobs=1)
        return pipe(("select", sel), ("clf", PosWeighted(xgb))), None
    if base == "P3":
        members = [(m, build_method(m, ctx)[0]) for m in BASE_MODELS]
        return StackingClassifier(
            members, final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=rs),
            cv=SafeSKF(cfg.stack_cv, rs), stack_method="auto", n_jobs=1), None
    if base == "P4":
        ks = sorted({min(k, ctx.p) for k in cfg.k_grid})
        return pipe(("select", WelchTSelector()), ("clf", smote_rf)), {"select__k": ks}
    if base == "P5":
        return (pipe(("select", L1HingeSelector(random_state=rs)),
                     ("clf", Resampled(rf(), sampler="nearmiss3", random_state=rs))),
                {"select__alpha": list(cfg.alphas), "clf__ratio": list(cfg.nm_ratios)})
    if base == "P6":
        from lightgbm import LGBMClassifier
        # min_child_samples: LightGBM's default of 20 allows no split at all when
        # fewer than 40 training patients are available (the multi-omic analyte).
        lgbm = LGBMClassifier(boosting_type="dart", n_estimators=cfg.boost_trees, num_leaves=15,
                              learning_rate=0.05, drop_rate=0.1, reg_alpha=2.0, reg_lambda=2.0,
                              min_child_samples=cfg.lgbm_min_child, random_state=rs, n_jobs=1,
                              verbose=-1)
        return pipe(("clf", PosWeighted(lgbm))), None

    # ---- ablation ---------------------------------------------------------------
    if base == "A0":
        return pipe(("clf", rf())), None
    if base == "A1":
        return (pipe(("select", L1HingeSelector(random_state=rs)), ("clf", rf())),
                {"select__alpha": list(cfg.alphas)})
    if base == "A2":
        return pipe(("clf", smote_rf)), None
    raise ValueError(f"unknown method '{method}'")


# =============================================================================
# 5. NESTED LEAVE-ONE-OUT
# =============================================================================

def _score_of(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1].astype(float), "proba"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        return np.ravel(model.decision_function(X)).astype(float), "decision"
    return model.predict(X).astype(float), "label"


def _support_of(model):
    steps = model.steps if isinstance(model, Pipeline) else []
    for _, step in steps:
        if hasattr(step, "get_support"):
            return np.asarray(step.get_support(), bool)
    return None


def outer_splits(y, cfg, seed):
    """[(repeat, test_indices)]. Every patient is tested exactly once per repeat."""
    if cfg.cv == "loo":
        return [(0, np.array([i])) for i in range(len(y))]
    out = []
    for rep in range(cfg.cv_repeats):
        skf = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=1000 * seed + rep)
        out += [(rep, test) for _, test in skf.split(np.zeros(len(y)), y)]
    return out


def _fit_fold(est, grid, X, y, test, seed, inner_cv, want_inner):
    """One outer fold. Only the training rows are used for anything that is fitted."""
    warnings.simplefilter("ignore")
    train = np.ones(len(y), bool)
    train[test] = False
    Xtr, ytr, Xte = X[train], y[train], X[test]
    t0 = time.perf_counter()
    inner, params = np.nan, ""
    if grid:          # tune on the training patients; the tuning score doubles as inner score
        search = GridSearchCV(clone(est), grid, scoring="balanced_accuracy",
                              cv=SafeSKF(inner_cv, seed), refit=True, n_jobs=1,
                              error_score="raise").fit(Xtr, ytr)
        model, inner, params = search.best_estimator_, float(search.best_score_), \
            json.dumps(search.best_params_)
    else:
        if want_inner:  # same inner folds as the tuned methods
            inner = float(cross_val_score(clone(est), Xtr, ytr, scoring="balanced_accuracy",
                                          cv=SafeSKF(inner_cv, seed), n_jobs=1).mean())
        model = clone(est).fit(Xtr, ytr)
    score, kind = _score_of(model, Xte)
    return dict(pred=model.predict(Xte).astype(int), score=score, score_kind=kind,
                inner_score=inner, seconds=time.perf_counter() - t0, params=params,
                support=_support_of(model))


class Runner:
    def __init__(self, ds: Dataset, outdir, cfg: Cfg, n_jobs=1, analytes=None, methods=None):
        self.ds, self.cfg, self.n_jobs, self.methods = ds, cfg, n_jobs, methods
        self.out = Path(outdir)
        self.analytes = analytes or ANALYTES
        self.labels = {"moesm3": ds.y}
        self.flipped = []
        if ds.days is not None:
            self.labels["oct2021"], self.flipped = reconstruct_oct2021(ds.y, ds.days, cfg.n_flip)
        self.failures = []
        scheme = dict(cv=cfg.cv, folds=cfg.cv_folds if cfg.cv != "loo" else None,
                      repeats=cfg.cv_repeats if cfg.cv != "loo" else None)
        f = self.out / "cv_scheme.json"
        if f.exists() and json.loads(f.read_text()) != scheme:
            raise SystemExit(f"{self.out} holds results from {f.read_text()}; "
                             f"use a different --outdir for {scheme}")
        self.out.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(scheme))

    def only(self, methods):
        return [m for m in methods if self.methods is None or m in self.methods]

    def need_oct2021(self):
        if "oct2021" not in self.labels:
            raise SystemExit(f"'{DAYS}' column is required for the October-2021 reconstruction")

    def path(self, labels, analyte, method, seed):
        return self.out / "preds" / labels / analyte / f"{method}__s{seed}.csv"

    def run(self, analyte, labels, method, seed, inner=False):
        v = self.ds.view(analyte)
        y = self.labels[labels].loc[v.patients].to_numpy().astype(int)
        splits = outer_splits(y, self.cfg, seed)
        reps = 1 + max(r for r, _ in splits)
        path = self.path(labels, analyte, method, seed)
        if path.exists():
            old = pd.read_csv(path)
            if (len(old) == reps * v.n and (old["y"].to_numpy() == np.tile(y, reps)).all()
                    and (not inner or old["inner_score"].notna().all())):
                return old
        X = v.X
        ctx = Ctx(seed, X.shape[1], self.cfg, v.zero_mask)
        t0 = time.perf_counter()
        try:
            est, grid = build_method(method, ctx)
            folds = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_fold)(est, grid, X, y, test, seed, self.cfg.inner_cv, inner)
                for _, test in splits)
        except Exception as e:                      # record, keep going, fail at the end
            self.failures.append((labels, analyte, method, seed, repr(e)))
            print(f"  !! {labels}/{analyte}/{method} failed: {e!r}", flush=True)
            self.out.mkdir(parents=True, exist_ok=True)
            with (self.out / "failures.log").open("a") as fh:
                fh.write(f"{labels}/{analyte}/{method}/s{seed}\n{traceback.format_exc()}\n")
            return None
        rows = []
        for (rep, test), f in zip(splits, folds):
            n_sel = int(f["support"].sum()) if f["support"] is not None else X.shape[1]
            for j, i in enumerate(test):
                rows.append(dict(rep=rep, pos=int(i), patient=v.patients[i], y=int(y[i]),
                                 pred=int(f["pred"][j]), score=float(f["score"][j]),
                                 score_kind=f["score_kind"], inner_score=f["inner_score"],
                                 seconds=f["seconds"] / len(test), params=f["params"],
                                 n_selected=n_sel))
        df = pd.DataFrame(rows).sort_values(["rep", "pos"]).drop(columns="pos").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")              # write, then rename: no half-written cache
        df.to_csv(tmp, index=False)
        tmp.replace(path)
        if folds[0]["support"] is not None:
            np.savez_compressed(path.with_suffix(".support.npz"),
                                mask=np.vstack([f["support"] for f in folds]),
                                features=np.array(v.cols, dtype=str))
        m = frame_metrics(df)
        print(f"  {labels:<9}{analyte:<15}{method:<28} BA={m['BA']:.3f} ACC={m['ACC']:.3f} "
              f"TN={m['TN']:>4.3g}/{m['n_alive']:<3.3g} [{time.perf_counter() - t0:6.0f}s]", flush=True)
        return df


# =============================================================================
# 6. EXPERIMENT PARTS
# =============================================================================

def part_baseline(R, seeds):
    for s in seeds:
        for a in R.analytes:
            for m in R.only(BASE_MODELS):
                R.run(a, "moesm3", m, s, inner=True)


def part_pipelines(R, seeds):
    for s in seeds:
        for a in R.analytes:
            for m in R.only(PIPELINES):
                R.run(a, "moesm3", m, s, inner=True)


def part_ablation(R, seeds):
    for s in seeds:
        for a in R.analytes:
            for m in sorted(set(ABLATION.values())):
                R.run(a, "moesm3", m, s, inner=m in PIPELINES + BASE_MODELS)


def part_oct2021(R, seed):
    R.need_oct2021()
    for a in R.analytes:
        for m in R.only(BASE_MODELS + PIPELINES):
            R.run(a, "oct2021", m, seed)


def part_alpha_sweep(R, seed, analytes):
    for a in analytes:
        for alpha in R.cfg.alphas:
            R.run(a, "moesm3", f"P1@alpha={alpha}", seed)


def part_lvs_sim(R, seed, analytes, methods, flips, reps):
    """Set k randomly chosen deceased patients of the analyte back to alive and re-run."""
    for a in analytes:
        v = R.ds.view(a)
        dead = [p for p in v.patients if R.ds.y[p] == 1]
        for k in flips:
            for r in range(reps):
                rng = np.random.default_rng([seed, ANALYTES.index(a), k, r])
                y_old = R.ds.y.copy()
                y_old.loc[rng.choice(dead, size=k, replace=False)] = 0
                R.labels[f"flip{k}_r{r}"] = y_old
                for m in methods:
                    R.run(a, f"flip{k}_r{r}", m, seed)


def part_nulltest(R, seed, methods, perms):
    """Shuffle labels among the analyte's own patients (class counts unchanged) and re-run."""
    for r in range(perms):
        for a in R.analytes:
            v = R.ds.view(a)
            rng = np.random.default_rng([seed, ANALYTES.index(a), r])
            y_perm = R.ds.y.copy()
            y_perm.loc[v.patients] = rng.permutation(R.ds.y.loc[v.patients].to_numpy())
            R.labels[f"perm_r{r}"] = y_perm
            for m in methods:
                R.run(a, f"perm_r{r}", m, seed, inner=True)


# =============================================================================
# 7. METRICS AND STATISTICS
# =============================================================================

def metrics(y, pred, score=None):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp, fn = int(((y == 1) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    auc = np.nan
    if score is not None and np.isfinite(score).all() and len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, score))
    return dict(n=len(y), n_dead=tp + fn, n_alive=tn + fp, TP=tp, FP=fp, TN=tn, FN=fn,
                ACC=(tp + tn) / len(y), BA=(sens + spec) / 2, Sens=sens, Spec=spec,
                PPV=tp / (tp + fp) if tp + fp else np.nan,
                NPV=tn / (tn + fn) if tn + fn else np.nan,
                MCC=(tp * tn - fp * fn) / den if den else 0.0, AUC=auc,
                fisher_p=float(stats.fisher_exact([[tp, fn], [fp, tn]], alternative="greater")[1]))


def frame_metrics(df):
    """Metrics for one run. With repeated CV, counts are per-repeat means and
    BA/AUC are means over repeats (pooled BA equals the mean per-repeat BA)."""
    reps = df["rep"].nunique() if "rep" in df else 1
    m = metrics(df["y"].to_numpy(), df["pred"].to_numpy(),
                df["score"].to_numpy(dtype=float) if reps == 1 else None)
    m["repeats"], m["BA_sd_repeats"] = reps, np.nan
    if reps > 1:
        per = [metrics(g["y"].to_numpy(), g["pred"].to_numpy(), g["score"].to_numpy(dtype=float))
               for _, g in df.groupby("rep")]
        for k in ("n", "n_dead", "n_alive", "TP", "FP", "TN", "FN"):
            m[k] = m[k] / reps
        aucs = [p["AUC"] for p in per]
        m["AUC"] = float(np.mean(aucs)) if np.isfinite(aucs).all() else np.nan
        m["BA_sd_repeats"] = float(np.std([p["BA"] for p in per], ddof=1))
        m["fisher_p"] = np.nan          # counts are pooled over repeats: not a valid test
    return m


def as2d(df, col):
    """(repeats x patients) array; rows are stored repeat by repeat in patient order."""
    reps = df["rep"].nunique() if "rep" in df else 1
    return df[col].to_numpy().reshape(reps, -1)


def _strat_idx(y, B, rng):
    """Bootstrap patient indices resampled within each class: deceased first, then alive."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return (pos[rng.integers(0, len(pos), (B, len(pos)))],
            neg[rng.integers(0, len(neg), (B, len(neg)))])


def boot_ba(df, B, seed=0):
    Y, P = as2d(df, "y"), as2d(df, "pred")
    ip, ineg = _strat_idx(Y[0], B, np.random.default_rng(seed))
    ok = (P == Y).astype(float)
    return (ok[:, ip].mean(axis=2) + ok[:, ineg].mean(axis=2)).mean(axis=0) / 2


def boot_auc(df, B, seed=0):
    Y, S = as2d(df, "y"), as2d(df, "score").astype(float)
    ip, ineg = _strat_idx(Y[0], B, np.random.default_rng(seed))
    n1, n0 = ip.shape[1], ineg.shape[1]
    out = np.zeros(B)
    for s in S:
        ranks = stats.rankdata(np.hstack([s[ip], s[ineg]]), axis=1)
        out += (ranks[:, :n1].sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out / len(S)


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def mcnemar_exact(y, pa, pb):
    """b = A wrong & B right, c = A right & B wrong; exact two-sided binomial p."""
    b = int(((pa != y) & (pb == y)).sum())
    c = int(((pa == y) & (pb != y)).sum())
    p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, p


def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out


def build_auto(members: dict):
    """Per outer fold, use the member with the best inner-CV score (ties: first)."""
    names = list(members)
    inner = np.column_stack([members[m]["inner_score"].to_numpy() for m in names])
    if np.isnan(inner).any():
        return None
    pick = inner.argmax(axis=1)
    rows = np.arange(len(pick))
    first = members[names[0]]
    kinds = {k for m in names for k in members[m]["score_kind"].unique()}
    score = np.column_stack([members[m]["score"].to_numpy() for m in names])[rows, pick]
    return pd.DataFrame(dict(
        rep=first["rep"], patient=first["patient"], y=first["y"],
        pred=np.column_stack([members[m]["pred"].to_numpy() for m in names])[rows, pick],
        score=score if kinds == {"proba"} else np.nan,        # mixed scales: no AUC
        score_kind="proba" if kinds == {"proba"} else "mixed",
        inner_score=inner.max(axis=1),
        seconds=sum(members[m]["seconds"].to_numpy() for m in names),
        params=[names[i] for i in pick], n_selected=np.nan))


# =============================================================================
# 8. REPORT
# =============================================================================

def _latex(df, path, caption):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "--" if not np.isfinite(v) else f"{v:.3f}"
        return str(v).replace("_", r"\_").replace("%", r"\%")
    n_text = sum(not pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
             r"\begin{tabular}{" + "l" * n_text + "r" * (df.shape[1] - n_text) + "}", r"\hline",
             " & ".join(cell(c) for c in df.columns) + r" \\", r"\hline"]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    Path(path).write_text("\n".join(lines))


def _save(df, out, name, caption=None):
    df.to_csv(out / f"{name}.csv", index=False)
    if caption:
        obj = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        _latex(df[obj + [c for c in df.columns if c not in obj]], out / f"{name}.tex", caption)


def load_preds(out):
    rows = {}
    for f in sorted((out / "preds").glob("*/*/*.csv")):
        method, seed = f.stem.rsplit("__s", 1)
        rows[(f.parts[-3], f.parts[-2], method, int(seed))] = pd.read_csv(f)
    return rows


def part_audit(R, out):
    out.mkdir(parents=True, exist_ok=True)
    R.ds.audit.to_csv(out / "column_audit.csv", index=False)
    a = R.ds.audit
    print(f"\nColumns: {len(a)}  used: {(a.status == 'used').sum()}  "
          f"not used: {(a.status != 'used').sum()}")
    lost = a[(a.status == "used") & (a.n_non_numeric_lost > 0)]
    if len(lost):
        print(f"  !! {len(lost)} used columns contain non-numeric values that became missing "
              f"(see column_audit.csv), e.g. {list(lost.column[:5])}")
    sus = a[(a.status == "used") & a.suspicious_name]
    if len(sus):
        print(f"  !! {len(sus)} FEATURE columns have outcome-like names. Check each one; "
              f"drop real outcome/follow-up variables with --exclude-regex:")
        for c in sus.column[:40]:
            print(f"       {c}")
    rows = []
    for an in ANALYTES:
        v = R.ds.view(an)
        pub = published(an)
        row = dict(Analyte=an, p=len(v.cols), n=v.n,
                   n_dead_moesm3=int(R.labels["moesm3"].loc[v.patients].sum()),
                   n_dead_published=pub["n_dead"], n_published=pub["n"],
                   cells_missing=int(np.isnan(v.X).sum()), cols_with_missing=int(v.nan_cols.sum()))
        if "oct2021" in R.labels:
            row["n_dead_oct2021"] = int(R.labels["oct2021"].loc[v.patients].sum())
            row["flipped_in_subset"] = len(set(R.flipped) & set(v.patients))
            row["published_matches"] = ("moesm3" if pub["n_dead"] == row["n_dead_moesm3"] else
                                        "oct2021" if pub["n_dead"] == row["n_dead_oct2021"]
                                        else "neither")
        rows.append(row)
    lv = pd.DataFrame(rows)
    _save(lv, out, "label_versions")
    print("\nDeceased counts per analyte: released file vs reconstruction vs the published "
          "Table 1 confusion matrices (TP+FN)")
    print(lv.drop(columns=["cells_missing", "cols_with_missing"]).to_string(index=False))
    if R.flipped:
        print(f"Patients flipped to alive for 'oct2021': {R.flipped}")
    return lv


def part_report(R, seeds, out_root):
    out, cfg, seed0 = out_root / "report", R.cfg, seeds[0]
    out.mkdir(parents=True, exist_ok=True)
    lv = part_audit(R, out)
    preds = load_preds(out_root)
    if not preds:
        print("No predictions yet.")
        return

    # nested model selection columns
    for (labels, seed) in sorted({(k[0], k[3]) for k in preds}):
        for an in ANALYTES:
            for name, members in (("BASE_AUTO", BASE_MODELS), ("PIPE_AUTO", PIPELINES)):
                have = {m: preds[(labels, an, m, seed)] for m in members
                        if (labels, an, m, seed) in preds}
                auto = build_auto(have) if len(have) >= 2 else None
                if auto is not None:
                    auto["members"] = "+".join(have)
                    if len(have) < len(members) and labels == "moesm3":
                        print(f"  note: {name} for {an} uses only {'+'.join(have)}")
                    preds[(labels, an, name, seed)] = auto
                    d = out / "nested_selection"
                    d.mkdir(exist_ok=True)
                    auto.to_csv(d / f"{labels}__{an}__{name}__s{seed}.csv", index=False)

    # ---- every run, every metric, with bootstrap intervals -------------------
    rows = []
    for (labels, an, method, seed), df in preds.items():
        m = frame_metrics(df)
        m["BA_lo"], m["BA_hi"] = ci(boot_ba(df, cfg.n_boot, seed))
        m["AUC_lo"], m["AUC_hi"] = (ci(boot_auc(df, cfg.n_boot, seed))
                                    if np.isfinite(m["AUC"]) else (np.nan, np.nan))
        rows.append(dict(labels=labels, Analyte=an, Method=method, seed=seed, **m,
                         inner_BA=df["inner_score"].mean(), n_selected=df["n_selected"].mean(),
                         cpu_seconds=df["seconds"].sum()))
    M = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(ANALYTES)}
    M = M.sort_values(["labels", "Analyte", "Method", "seed"],
                      key=lambda s: s.map(order) if s.name == "Analyte" else s)
    _save(M, out, "metrics_long")
    main = M[(M.labels == "moesm3") & (M.seed == seed0)].set_index(["Analyte", "Method"])

    def get(an, method, col="BA", labels="moesm3"):
        r = M[(M.labels == labels) & (M.seed == seed0) & (M.Analyte == an) & (M.Method == method)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    # ---- Table I: dataset ---------------------------------------------------------
    t1 = pd.DataFrame([dict(
        Analyte=a, p=int(r.p), n=int(r.n), n_dead=int(r.n_dead_moesm3),
        n_alive=int(r.n - r.n_dead_moesm3), Published_ACC=published(a)["ACC"],
        Published_BA=published(a)["BA"]) for a, r in lv.set_index("Analyte").iterrows()])
    _save(t1, out, "table1_dataset", "Dataset summary. Published accuracy and balanced "
          "accuracy are computed from the published confusion matrices.")

    # ---- Table II: reproduction of the released models vs the published Table 1 ------
    rep_rows, rep_sum = [], []
    for an in ANALYTES:
        pub = published(an)
        have = [m for m in BASE_MODELS if (an, m) in main.index]
        for m in have:
            r = main.loc[(an, m)]
            over = M[(M.labels == "moesm3") & (M.Analyte == an) & (M.Method == m)]
            cm = [int(r.TP), int(r.FP), int(r.TN), int(r.FN)]
            pub_cm = [pub["TP"], pub["FP"], pub["TN"], pub["FN"]]
            rep_rows.append(dict(
                Analyte=an, Method=m, ACC=r.ACC, BA=r.BA, TP=cm[0], FP=cm[1], TN=cm[2], FN=cm[3],
                pub_ACC=pub["ACC"], pub_BA=pub["BA"], pub_TP=pub_cm[0], pub_FP=pub_cm[1],
                pub_TN=pub_cm[2], pub_FN=pub_cm[3], ACC_minus_pub=r.ACC - pub["ACC"],
                cm_L1_distance=int(sum(abs(a - b) for a, b in zip(cm, pub_cm))),
                same_confusion_matrix=cm == pub_cm, n_seeds=len(over),
                ACC_seed_min=over.ACC.min(), ACC_seed_max=over.ACC.max(),
                BA_seed_min=over.BA.min(), BA_seed_max=over.BA.max()))
        if have:
            g = pd.DataFrame([x for x in rep_rows if x["Analyte"] == an]).set_index("Method")
            rep_sum.append(dict(
                Analyte=an, n=int(r.n), n_dead=int(r.n_dead), majority_ACC=r.n_dead / r.n,
                pub_ACC=pub["ACC"], pub_BA=pub["BA"],
                best_of7_ACC_optimistic=g.ACC.max(), best_of7_model=g.ACC.idxmax(),
                best_of7_BA_at_that_model=g.loc[g.ACC.idxmax(), "BA"],
                closest_model=g.cm_L1_distance.idxmin(),
                closest_cm_L1_distance=int(g.cm_L1_distance.min()),
                any_exact_match=bool(g.same_confusion_matrix.any()),
                Step2_nested_ACC=get(an, "BASE_AUTO", "ACC"), Step2_nested_BA=get(an, "BASE_AUTO")))
    if rep_rows:
        _save(pd.DataFrame(rep_rows), out, "table2_reproduction_by_model")
        _save(pd.DataFrame(rep_sum), out, "table2_reproduction",
              "Re-execution of the seven released classifiers on MOESM3 against the "
              "published Table 1 confusion matrices. Published values are plain accuracy.")

    # ---- Table III + paired tests: Step 2 vs Step 3 ---------------------------------
    cmp_rows, t3 = [], []
    for an in ANALYTES:
        base = preds.get(("moesm3", an, "BASE_AUTO", seed0))
        row = dict(Analyte=an, Published_BA=published(an)["BA"],
                   Step2=get(an, "BASE_AUTO"), **{p: get(an, p) for p in PIPELINES},
                   Step3=get(an, "PIPE_AUTO"))
        if base is not None:
            y, pb = base["y"].to_numpy(), base["pred"].to_numpy()
            for method in PIPELINES + ["PIPE_AUTO"]:
                other = preds.get(("moesm3", an, method, seed0))
                if other is None:
                    continue
                pa = other["pred"].to_numpy()
                d = boot_ba(other, cfg.n_boot, seed0) - boot_ba(base, cfg.n_boot, seed0)
                b, c, p = (mcnemar_exact(y, pb, pa) if cfg.cv == "loo"
                           else (np.nan, np.nan, np.nan))   # needs one prediction per patient
                ba_a, ba_b = get(an, method), get(an, "BASE_AUTO")
                cmp_rows.append(dict(Analyte=an, Method=method, BA=ba_a, BA_step2=ba_b,
                                     delta=ba_a - ba_b, delta_lo=ci(d)[0], delta_hi=ci(d)[1],
                                     cohens_h=cohens_h(ba_a, ba_b), step2_wrong_method_right=b,
                                     step2_right_method_wrong=c, mcnemar_p=p))
        t3.append(row)
    t3 = pd.DataFrame(t3)
    if cmp_rows:
        C = pd.DataFrame(cmp_rows)
        C["mcnemar_p_holm"] = C.groupby("Method")["mcnemar_p"].transform(holm)
        _save(C, out, "stats_pairwise_vs_step2")
        pa = C[C.Method == "PIPE_AUTO"].set_index("Analyte")
        for col in ("delta", "delta_lo", "delta_hi", "mcnemar_p"):
            t3[col] = t3.Analyte.map(pa[col])
        port = []
        for method, g in C.groupby("Method"):
            d = g["delta"].to_numpy()
            rng = np.random.default_rng(seed0)
            bm = d[rng.integers(0, len(d), (cfg.n_boot, len(d)))].mean(axis=1)
            try:
                wil = float(stats.wilcoxon(d, zero_method="pratt").pvalue)
            except ValueError:
                wil = 1.0
            w, l = int((d > 1e-12).sum()), int((d < -1e-12).sum())
            port.append(dict(Method=method, analytes=len(d), mean_delta=d.mean(),
                             mean_lo=ci(bm)[0], mean_hi=ci(bm)[1], wins=w, ties=len(d) - w - l,
                             losses=l, wilcoxon_p=wil, mean_cohens_h=g["cohens_h"].mean(),
                             sign_p=float(stats.binomtest(w, w + l, 0.5).pvalue) if w + l else 1.0))
        P = pd.DataFrame(port)
        P["wilcoxon_p_holm_over_P1_P6"] = np.nan
        six = P.Method.isin(PIPELINES)
        P.loc[six, "wilcoxon_p_holm_over_P1_P6"] = holm(P.loc[six, "wilcoxon_p"])
        _save(P, out, "stats_portfolio")
    done = t3[t3.Step2.notna() | t3.Step3.notna()]
    t3.loc[len(t3)] = [f"Mean ({len(done)} analytes)"] + list(done.drop(columns="Analyte").mean())
    if cmp_rows:      # interval for the mean gap: bootstrap over analytes, not a mean of intervals
        pa = P[P.Method == "PIPE_AUTO"]
        t3.loc[len(t3) - 1, ["delta_lo", "delta_hi", "mcnemar_p"]] = (
            [float(pa.mean_lo.iloc[0]), float(pa.mean_hi.iloc[0]), np.nan] if len(pa)
            else [np.nan] * 3)
    _save(t3, out, "table3_step2_vs_step3", "Balanced accuracy on MOESM3 labels. Step 2 and "
          "Step 3 use nested model selection; delta is Step 3 minus Step 2 with a paired "
          "bootstrap 95\\% interval and exact McNemar p.")

    fr = []
    for metric in ("BA", "ACC"):
        treat = ["BASE_AUTO"] + PIPELINES
        tab = main[metric].unstack("Method").reindex(columns=treat).dropna()
        if len(tab) >= 3:
            st = stats.friedmanchisquare(*[tab[t] for t in treat])
            ranks = tab.rank(axis=1, ascending=False).mean()
            fr.append(dict(metric=metric, blocks=len(tab), treatments=len(treat),
                           chi2=st.statistic, df=len(treat) - 1, p=st.pvalue,
                           kendall_W=st.statistic / (len(tab) * (len(treat) - 1)),
                           **{f"mean_rank_{t}": ranks[t] for t in treat}))
    if fr:
        _save(pd.DataFrame(fr), out, "stats_friedman")

    # ---- Table IV: discrimination -----------------------------------------------------
    cols = ["AUC", "AUC_lo", "AUC_hi", "Sens", "Spec", "PPV", "NPV", "BA", "BA_lo", "BA_hi",
            "ACC", "MCC", "TP", "FP", "TN", "FN", "fisher_p"]
    t4 = main.reset_index()
    t4 = t4[t4.Method.isin(PIPELINES + BASE_MODELS + ["BASE_AUTO", "PIPE_AUTO"])]
    _save(t4[["Analyte", "Method"] + cols], out, "table4_discrimination_all")
    mo = t4[(t4.Analyte == "Multiomic") & t4.Method.isin(PIPELINES + ["RFE_RF"])]
    if len(mo):
        _save(mo[["Method", "AUC", "AUC_lo", "AUC_hi", "Sens", "Spec", "PPV", "NPV", "BA"]],
              out, "table4_multiomic", "Discrimination on the multi-omic analyte (MOESM3 "
              "labels), with bootstrap 95\\% intervals for AUC.")

    # ---- Table V: ablation ---------------------------------------------------------------
    ab = []
    for an in ANALYTES:
        for config, run_id in ABLATION.items():
            if (an, run_id) in main.index:
                r = main.loc[(an, run_id)]
                ab.append(dict(Analyte=an, Config=PRETTY[config], BA=r.BA, ACC=r.ACC, Sens=r.Sens,
                               Spec=r.Spec, TN=round(float(r.TN), 1), n_alive=int(r.n_alive),
                               all_deceased=bool(r.TN == 0 and r.FN == 0)))
    if ab:
        _save(pd.DataFrame(ab), out, "table5_ablation", "Ablation on every analyte.")

    # ---- Table VI: label versions -----------------------------------------------------------
    lvs = []
    for an in ANALYTES:
        for method in BASE_MODELS + PIPELINES:
            old, new = (get(an, method, "BA", "oct2021"), get(an, method, "BA"))
            if np.isnan(old) or np.isnan(new):
                continue
            d = get(an, method, "n_dead") - get(an, method, "n_dead", "oct2021")
            lvs.append(dict(Analyte=an, Method=method, deaths_added=int(d), BA_oct2021=old,
                            BA_moesm3=new, LVS_BA=(new - old) / d if d else np.nan,
                            ACC_oct2021=get(an, method, "ACC", "oct2021"),
                            ACC_moesm3=get(an, method, "ACC"),
                            LVS_ACC=(get(an, method, "ACC") - get(an, method, "ACC", "oct2021")) / d
                            if d else np.nan))
    if lvs:
        _save(pd.DataFrame(lvs), out, "table6_label_versions",
              "Label-version sensitivity (change per added death).")
    sim = M[M.labels.str.startswith("flip")].copy()
    if len(sim):
        sim["k"] = sim.labels.str.extract(r"flip(\d+)_")[0].astype(int)
        sim["BA_moesm3"] = [get(a, m) for a, m in zip(sim.Analyte, sim.Method)]
        sim["LVS_BA"] = (sim.BA_moesm3 - sim.BA) / sim.k
        g = sim.groupby(["Analyte", "Method", "k"])["LVS_BA"]
        _save(g.agg(reps="size", mean="mean", sd="std", lo=lambda s: s.quantile(0.025),
                    hi=lambda s: s.quantile(0.975)).reset_index(), out, "lvs_simulation")
    null = M[M.labels.str.startswith("perm")]
    if len(null):     # whole pipeline re-run on shuffled labels: the honest "is it above chance" test
        nc = []
        for (an, method), g in null.groupby(["Analyte", "Method"], sort=False):
            obs = get(an, method)
            nc.append(dict(Analyte=an, Method=method, permutations=len(g), null_mean_BA=g.BA.mean(),
                           null_sd=g.BA.std(), null_max=g.BA.max(), observed_BA=obs,
                           p_perm=(1 + (g.BA >= obs - 1e-12).sum()) / (1 + len(g))
                           if np.isfinite(obs) else np.nan))
        gap = null[null.Method.isin(["BASE_AUTO", "PIPE_AUTO"])].pivot_table(
            index=["Analyte", "labels"], columns="Method", values="BA")
        if {"BASE_AUTO", "PIPE_AUTO"} <= set(gap.columns):
            gap["delta"] = gap["PIPE_AUTO"] - gap["BASE_AUTO"]
            for an, g in gap.dropna().groupby("Analyte", sort=False):
                obs = get(an, "PIPE_AUTO") - get(an, "BASE_AUTO")
                nc.append(dict(Analyte=an, Method="PIPE_AUTO minus BASE_AUTO", permutations=len(g),
                               null_mean_BA=g.delta.mean(), null_sd=g.delta.std(),
                               null_max=g.delta.max(), observed_BA=obs,
                               p_perm=(1 + (g.delta >= obs - 1e-12).sum()) / (1 + len(g))
                               if np.isfinite(obs) else np.nan))
        _save(pd.DataFrame(nc), out, "null_check")

    # ---- seeds, runtime, tuned values, feature stability -----------------------------------
    ms = M[M.labels == "moesm3"]
    if ms.seed.nunique() > 1:
        _save(ms.groupby(["Analyte", "Method"], sort=False)["BA"].agg(
            seeds="size", mean="mean", sd="std", min="min", max="max").reset_index(),
            out, "seed_sensitivity")
    rt = main.reset_index().groupby("Method")["cpu_seconds"].sum().div(3600).rename("cpu_hours")
    _save(rt.reset_index(), out, "runtime_by_method")
    tuned, stab = [], []
    for (labels, an, method, seed), df in preds.items():
        if labels != "moesm3" or seed != seed0:
            continue
        if df["params"].astype(str).str.startswith("{").all():
            for k, s in pd.DataFrame([json.loads(p) for p in df["params"]]).items():
                for val, cnt in s.value_counts().items():
                    tuned.append(dict(Analyte=an, Method=method, parameter=k, value=val,
                                      test_rows=cnt))
        f = R.path(labels, an, method, seed).with_suffix(".support.npz")
        if f.exists() and method in BASE_MODELS + PIPELINES + ["A1"]:
            z = np.load(f, allow_pickle=False)
            mask, freq = z["mask"], z["mask"].mean(axis=0)
            inter = mask.astype(int) @ mask.astype(int).T
            sizes = mask.sum(axis=1)
            union = sizes[:, None] + sizes[None, :] - inter
            iu = np.triu_indices(len(mask), 1)
            jac = (inter[iu] / np.maximum(union[iu], 1)).mean() if len(mask) > 1 else np.nan
            top = np.argsort(-freq, kind="stable")[:15]
            stab.append(dict(Analyte=an, Method=method, mean_selected=sizes.mean(),
                             mean_pairwise_jaccard=jac, in_every_fold=int((freq == 1).sum()),
                             top_features="; ".join(f"{z['features'][i]} ({freq[i]:.2f})"
                                                    for i in top if freq[i] > 0)))
    if tuned:
        _save(pd.DataFrame(tuned), out, "tuned_values_by_fold")
    if stab:
        _save(pd.DataFrame(stab), out, "feature_stability")

    info = dict(version=__version__, python=sys.version.split()[0], platform=platform.platform(),
                cpus=os.cpu_count(), n_jobs=R.n_jobs, seeds=list(seeds), cfg=vars(cfg),
                packages={})
    for pkg in ("numpy", "pandas", "scipy", "sklearn", "imblearn", "xgboost", "lightgbm"):
        try:
            info["packages"][pkg] = __import__(pkg).__version__
        except Exception:
            info["packages"][pkg] = None
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    make_figures(M, seed0, out, "LOOCV" if cfg.cv == "loo"
                 else f"{cfg.cv_folds}-fold x {cfg.cv_repeats}")
    print(f"\nReport written to {out}/")
    show = t3[["Analyte", "Published_BA", "Step2", "Step3"]
              + [c for c in ("delta", "delta_lo", "delta_hi") if c in t3.columns]]
    print(show[show.Step2.notna() | show.Step3.notna()].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


# =============================================================================
# 9. FIGURES
# =============================================================================

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def make_figures(M, seed0, out, cv_label="LOOCV"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight"})
    main = M[(M.labels == "moesm3") & (M.seed == seed0)]
    piv = main.pivot_table(index="Method", columns="Analyte", values="BA")
    ans = [a for a in ANALYTES if a in piv.columns]
    short = [a.replace("_", "\n") for a in ans]

    def chance(ax, horizontal=True):
        (ax.axhline if horizontal else ax.axvline)(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)

    # Fig A: published vs Step 2 vs Step 3 (dots with bootstrap intervals) ---------
    if {"BASE_AUTO", "PIPE_AUTO"} <= set(piv.index):
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        x, off = np.arange(len(ans)), 0.24
        series = [("Published (from Table 1 counts)", [published(a)["BA"] for a in ans], C1, "o", None),
                  ("Step 2: seven classifiers", piv.loc["BASE_AUTO", ans], C2, "s", "BASE_AUTO"),
                  ("Step 3: fold-internal pipelines", piv.loc["PIPE_AUTO", ans], C3, "^", "PIPE_AUTO")]
        for j, (lab, vals, col, marker, key) in enumerate(series):
            xs = x + (j - 1) * off
            if key:
                r = main[main.Method == key].set_index("Analyte").reindex(ans)
                ax.errorbar(xs, r.BA, yerr=[np.clip(r.BA - r.BA_lo, 0, None),
                                            np.clip(r.BA_hi - r.BA, 0, None)],
                            fmt="none", ecolor=col, elinewidth=1.3, capsize=0, zorder=2)
            ax.scatter(xs, vals, s=30, color=col, marker=marker, edgecolor="white",
                       linewidth=0.7, zorder=3, label=lab)
        chance(ax)
        ax.text(1.005, 0.5, "chance", transform=ax.get_yaxis_transform(), color=MUTED,
                va="center", ha="left", fontsize=7)
        for xi in x[:-1]:
            ax.axvline(xi + 0.5, color=GRID, lw=0.5, zorder=0)
        ax.set_xticks(x, short, fontsize=6.5)
        ax.set_xlim(-0.5, len(ans) - 0.5)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(f"Balanced accuracy ({cv_label})")
        ax.set_ylim(0.2, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=7)
        fig.savefig(out / "fig_step_comparison.png")
        plt.close(fig)

    # Fig B: heat map, diverging around chance -----------------------------------------
    rows = [m for m in ["BASE_AUTO"] + BASE_MODELS + ["PIPE_AUTO"] + PIPELINES if m in piv.index]
    if rows:
        cmap = LinearSegmentedColormap.from_list("div", ["#c0392b", "#f0efec", "#1c5cab"])
        fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(rows) + 1.0))
        data = piv.loc[rows, ans].to_numpy()
        im = ax.imshow(data, cmap=cmap, norm=TwoSlopeNorm(0.5, 0.2, 0.9), aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(data[i, j] - 0.5) > 0.27 else INK)
        ax.set_xticks(range(len(ans)), short, fontsize=6.5)
        ax.set_yticks(range(len(rows)), [PRETTY.get(m, m) for m in rows], fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
        cb.set_label("Balanced accuracy (0.5 = chance)", fontsize=7)
        cb.outline.set_visible(False)
        fig.savefig(out / "fig_heatmap.png")
        plt.close(fig)

    # Fig C: ablation on the multi-omic analyte (or the first analyte that was run) ----------
    focus = "Multiomic" if "Multiomic" in ans else (ans[0] if ans else None)
    mo = main[main.Analyte == focus].set_index("Method")
    if all(r in mo.index for r in ABLATION.values()):
        labs = [PRETTY[c].split(" ", 1)[1].replace("+", "\n+ ") for c in ABLATION]
        r = mo.loc[list(ABLATION.values())]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
        axes[0].bar(labs, r.BA, 0.6, color=C1, zorder=2)
        axes[0].errorbar(labs, r.BA, yerr=[np.clip(r.BA - r.BA_lo, 0, None),
                                           np.clip(r.BA_hi - r.BA, 0, None)], fmt="none",
                         ecolor=INK, elinewidth=0.6, capsize=2, zorder=3)
        chance(axes[0])
        axes[0].set_ylim(0, 1.0)
        axes[0].set_ylabel(f"Balanced accuracy ({focus.replace('_', ' ')})")
        axes[1].bar(labs, r.TN, 0.6, color=C1, zorder=2)
        n_alive = int(r.n_alive.iloc[0])
        axes[1].axhline(n_alive, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        axes[1].text(len(labs) - 0.55, n_alive, f"all {n_alive} alive", color=MUTED,
                     va="bottom", ha="right", fontsize=7)
        for i, v in enumerate(r.TN):
            axes[1].text(i, v + 0.15, f"{v:.3g}", ha="center", va="bottom", color=INK, fontsize=7.5)
        axes[1].set_ylim(0, n_alive * 1.18)
        axes[1].set_ylabel("Alive patients correctly identified")
        for ax in axes:
            ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
            ax.tick_params(axis="x", labelsize=6.5, length=0)
        fig.tight_layout(w_pad=2.5)
        fig.savefig(out / "fig_ablation_multiomic.png")
        plt.close(fig)

    # Fig D: alpha sweep and label versions ---------------------------------------------------
    sweeps = main[main.Method.str.startswith("P1@alpha=")]
    sw_an = ("Multiomic" if (sweeps.Analyte == "Multiomic").any()
             else (sweeps.Analyte.iloc[0] if len(sweeps) else None))
    sw = sweeps[sweeps.Analyte == sw_an].set_index("Method")
    old = M[(M.labels == "oct2021") & (M.seed == seed0) & (M.Analyte == focus)].set_index("Method")
    ms_ = [m for m in ["RFE_RF"] + PIPELINES if m in old.index and m in mo.index]
    panels = (["sweep"] if len(sw) else []) + (["labels"] if ms_ else [])
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.6), squeeze=False)
        for ax, kind in zip(axes[0], panels):
            if kind == "sweep":
                sw = sw.assign(alpha=sw.index.str.split("=").str[1].astype(float)).sort_values("alpha")
                xs = np.arange(len(sw))
                ax.plot(xs, sw.BA, color=C1, lw=2, marker="o", ms=5, zorder=3)
                for xi, (ba, tn) in enumerate(zip(sw.BA, sw.TN)):
                    ax.annotate(f"TN={tn:.3g}", (xi, ba), xytext=(0, 8), textcoords="offset points",
                                ha="center", fontsize=6.5, color=INK, zorder=4,
                                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
                ax.set_xticks(xs, [f"{a:g}" for a in sw.alpha])
                ax.set_xlabel(f"Fixed L1 penalty α (P1, {sw_an.replace('_', ' ')})")
                ax.set_ylim(0.2, 1.0)
            else:
                x = np.arange(len(ms_))
                ax.bar(x - 0.2, old.loc[ms_, "BA"], 0.37, color=C1, zorder=2,
                       label=f"Reconstructed Oct-2021 ({old.n_dead.iloc[0]:.0f} deceased)")
                ax.bar(x + 0.2, mo.loc[ms_, "BA"], 0.37, color=C2, hatch="////", edgecolor="white",
                       linewidth=0.4, zorder=2, label=f"MOESM3 ({mo.n_dead.iloc[0]:.0f} deceased)")
                ax.set_xticks(x, [m.replace("RFE_RF", "RFECV\n+RF") for m in ms_], fontsize=6.5)
                ax.set_xlabel(focus.replace("_", " "))
                ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
                ax.set_ylim(0, 1.0)
            chance(ax)
            ax.set_ylabel("Balanced accuracy")
            ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        fig.tight_layout(w_pad=2.5)
        fig.savefig(out / "fig_alpha_and_labels.png")
        plt.close(fig)


# =============================================================================
# 10. COMMAND LINE
# =============================================================================

ALL_PARTS = ["audit", "baseline", "pipelines", "ablation", "alpha_sweep", "report"]
OPT_PARTS = ["oct2021", "lvs_sim", "nulltest"]        # not part of "all"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="MOESM3 .xlsx (or a .csv export of it)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--parts", nargs="+", default=["all"],
                    choices=ALL_PARTS + OPT_PARTS + ["all"])
    ap.add_argument("--analytes", nargs="+", default=None, choices=ANALYTES)
    ap.add_argument("--methods", nargs="+", default=None, choices=BASE_MODELS + PIPELINES,
                    help="restrict baseline/pipelines/oct2021 to these (e.g. leave out P3)")
    ap.add_argument("--rf-trees", type=int, default=500)
    ap.add_argument("--boost-trees", type=int, default=300)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="first seed is the primary one; all are run for baseline/pipelines/ablation")
    ap.add_argument("--n-jobs", type=int, default=-1, help="outer folds fitted in parallel")
    ap.add_argument("--exclude-regex", default=None,
                    help="drop feature columns whose name matches (outcome/follow-up variables)")
    ap.add_argument("--n-flip", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--sweep-analytes", nargs="+", default=["Multiomic"], choices=ANALYTES)
    ap.add_argument("--sim-analytes", nargs="+", default=["Multiomic"], choices=ANALYTES)
    ap.add_argument("--sim-methods", nargs="+", default=["RFE_RF", "P5"])
    ap.add_argument("--sim-flips", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--sim-reps", type=int, default=20)
    ap.add_argument("--null-methods", nargs="+", default=BASE_MODELS + PIPELINES)
    ap.add_argument("--null-perms", type=int, default=20)
    ap.add_argument("--cv", choices=["loo", "rskf"], default="loo",
                    help="outer scheme: leave-one-out (manuscript) or repeated stratified k-fold")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="quick end-to-end check: two small analytes, small models")
    a = ap.parse_args(argv)

    cfg = Cfg(n_boot=a.n_boot, n_flip=a.n_flip, rf_trees=a.rf_trees, boost_trees=a.boost_trees,
              cv=a.cv, cv_folds=a.cv_folds, cv_repeats=a.cv_repeats)
    analytes = a.analytes
    if a.smoke:
        cfg = replace(cfg, rf_trees=50, released_rf_trees=20, boost_trees=30, n_boot=200)
        analytes = analytes or ["RNA_Fusion", "INDEL"]
        a.sweep_analytes = [analytes[0]]
    parts = [p for p in ALL_PARTS + OPT_PARTS
             if p in a.parts or ("all" in a.parts and p in ALL_PARTS)]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"pdac_pipeline {__version__} | parts: {' '.join(parts)} | n_jobs={a.n_jobs}")
    ds = load_dataset(a.data, a.exclude_regex)
    print(f"Loaded {a.data}: {len(ds.y)} patients, {int(ds.y.sum())} deceased, "
          f"{int((ds.y == 0).sum())} alive")
    R = Runner(ds, out, cfg, a.n_jobs, analytes, a.methods)
    seed0 = a.seeds[0]
    if "audit" in parts and "report" not in parts:
        part_audit(R, out / "report")
    if "baseline" in parts:
        part_baseline(R, a.seeds)
    if "pipelines" in parts:
        part_pipelines(R, a.seeds)
    if "ablation" in parts:
        part_ablation(R, a.seeds)
    if "oct2021" in parts:
        part_oct2021(R, seed0)
    if "alpha_sweep" in parts:
        part_alpha_sweep(R, seed0, [s for s in a.sweep_analytes if s in R.analytes])
    if "lvs_sim" in parts:
        part_lvs_sim(R, seed0, a.sim_analytes, a.sim_methods, a.sim_flips, a.sim_reps)
    if "nulltest" in parts:
        part_nulltest(R, seed0, a.null_methods, a.null_perms)
    if "report" in parts or any(p in parts for p in OPT_PARTS):
        part_report(R, a.seeds, out)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Output: {out}/")
    if R.failures:
        print(f"{len(R.failures)} run(s) FAILED - see {out}/failures.log:")
        for f in R.failures:
            print("   ", *f[:4], "|", f[4][:120])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
