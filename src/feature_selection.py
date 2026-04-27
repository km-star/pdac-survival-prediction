"""
feature_selection.py
====================
Reduces the raw CNA (~24,000 genes) and RNA-seq (~20,000 genes) feature spaces
to a manageable, survival-informative subset.

Two-stage strategy:
  1. Variance filter   — remove near-zero-variance genes (uninformative noise)
  2. Survival correlation — keep genes whose expression is associated with OS
     via log-rank test or univariate Cox p-value
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import kruskal
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def variance_filter(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Remove columns with variance below `threshold`.
    For CNA data (discrete -2/-1/0/1/2), this removes genes that are
    homogeneous across samples (no copy-number alteration signal).

    Reduces CNA matrix from ~24,000 → ~4,000–6,000 genes.
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    mask = selector.get_support()
    filtered = df.loc[:, mask]
    logger.info(f"Variance filter: {df.shape[1]} → {filtered.shape[1]} features "
                f"(removed {df.shape[1] - filtered.shape[1]})")
    return filtered


def survival_correlation_filter(
    feature_df: pd.DataFrame,
    os_months: pd.Series,
    os_status: pd.Series,
    method: str = "logrank",
    top_n: int = 300,
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Keep genes whose expression level is significantly associated with
    overall survival using log-rank test (median split) or univariate Cox.

    Parameters
    ----------
    feature_df   : patient × gene matrix (aligned index)
    os_months    : continuous OS time
    os_status    : 1 = event (death), 0 = censored
    method       : "logrank" (faster) or "cox" (more precise)
    top_n        : maximum features to return (ranked by p-value)
    p_threshold  : nominal p-value cutoff (before multiple testing)

    Returns
    -------
    Filtered feature_df with only survival-informative genes.
    """
    common = feature_df.index.intersection(os_months.index).intersection(os_status.index)
    X = feature_df.loc[common]
    t = os_months.loc[common].astype(float)
    e = os_status.loc[common].astype(int)

    pvals = {}
    for gene in tqdm(X.columns, desc=f"Survival filter ({method})"):
        expr = X[gene]
        try:
            if method == "logrank":
                median = expr.median()
                high = expr >= median
                low = ~high
                if high.sum() < 5 or low.sum() < 5:
                    continue
                result = logrank_test(t[high], t[low], event_observed_A=e[high], event_observed_B=e[low])
                pvals[gene] = result.p_value
            elif method == "cox":
                tmp = pd.DataFrame({"T": t, "E": e, "x": expr}).dropna()
                if tmp.shape[0] < 10:
                    continue
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(tmp, duration_col="T", event_col="E")
                pvals[gene] = cph.summary.loc["x", "p"]
        except Exception:
            continue

    pval_series = pd.Series(pvals).sort_values()
    significant = pval_series[pval_series < p_threshold]
    top_genes = significant.head(top_n).index.tolist()

    logger.info(f"Survival filter: {X.shape[1]} → {len(top_genes)} features "
                f"(p < {p_threshold}, top {top_n})")
    return feature_df[top_genes]


def fuse_modalities(
    cna_filtered: pd.DataFrame,
    rna_filtered: pd.DataFrame,
    clinical_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Inner-join CNA and RNA feature matrices on patient index, optionally
    appending clinical features (stage, age, treatment flags).

    Total feature count target: ~20 after downstream SelectKBest / RFE.
    """
    fused = cna_filtered.join(rna_filtered, how="inner", lsuffix="_cna", rsuffix="_rna")
    if clinical_features is not None:
        fused = fused.join(clinical_features, how="inner")
    logger.info(f"Fused matrix shape: {fused.shape}")
    return fused


def final_feature_reduction(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 20,
    method: str = "mutual_info",
) -> pd.DataFrame:
    """
    Final reduction to top-k features via mutual information or RFECV.
    Achieves the 589 → 20 feature reduction reported in results.
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

    common = X.index.intersection(y.dropna().index)
    Xc = X.loc[common].fillna(0)
    yc = y.loc[common]

    score_func = mutual_info_classif if method == "mutual_info" else f_classif
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(Xc, yc)
    selected_cols = Xc.columns[selector.get_support()].tolist()

    logger.info(f"Final reduction: {Xc.shape[1]} → {len(selected_cols)} features")
    return X[selected_cols]
