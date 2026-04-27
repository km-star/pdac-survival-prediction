"""
preprocessing.py
================
QC, normalization, and ID harmonization for TCGA-PAAD multi-omic data.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def qc_filter_samples(df: pd.DataFrame, max_missing_frac: float = 0.3) -> pd.DataFrame:
    """Remove samples with more than `max_missing_frac` missing features."""
    missing = df.isna().mean(axis=1)
    keep = missing[missing <= max_missing_frac].index
    logger.info(f"QC sample filter: {len(df)} → {len(keep)} samples "
                f"(removed {len(df) - len(keep)} with >{max_missing_frac*100:.0f}% missing)")
    return df.loc[keep]


def qc_filter_genes(df: pd.DataFrame, max_missing_frac: float = 0.5) -> pd.DataFrame:
    """Remove genes missing in more than `max_missing_frac` of samples."""
    missing = df.isna().mean(axis=0)
    keep = missing[missing <= max_missing_frac].index
    logger.info(f"QC gene filter: {df.shape[1]} → {len(keep)} genes")
    return df[keep]


def normalize_rna(rna_df: pd.DataFrame, method: str = "log2") -> pd.DataFrame:
    """
    Normalize RNA-seq expression values.
    TCGA RNA-seq RSEM values are already upper-quartile normalized;
    apply log2(x+1) to stabilize variance across the dynamic range.
    """
    if method == "log2":
        normalized = np.log2(rna_df.clip(lower=0) + 1)
    elif method == "zscore":
        normalized = (rna_df - rna_df.mean()) / rna_df.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    logger.info(f"RNA normalization ({method}) applied. Shape: {normalized.shape}")
    return normalized


def impute_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute remaining NaN values (typically <5% after QC filters)."""
    imputer = SimpleImputer(strategy=strategy)
    imputed = pd.DataFrame(
        imputer.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    logger.info(f"Imputation ({strategy}): {df.isna().sum().sum()} NaN values filled")
    return imputed


def encode_clinical(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and encode key clinical covariates:
    - Tumor stage (I/II/III/IV → 1/2/3/4)
    - Age at diagnosis (continuous, standardized)
    - Treatment flags (chemotherapy, radiation)
    """
    features = pd.DataFrame(index=clinical_df.index)

    # Stage
    stage_map = {"I": 1, "IA": 1, "IB": 1, "II": 2, "IIA": 2, "IIB": 2,
                 "III": 3, "IV": 4}
    if "AJCC_PATHOLOGIC_TUMOR_STAGE" in clinical_df.columns:
        raw_stage = clinical_df["AJCC_PATHOLOGIC_TUMOR_STAGE"].str.upper().str.replace("STAGE ", "")
        features["stage"] = raw_stage.map(stage_map).fillna(2.0)  # impute missing with median

    # Age
    if "AGE" in clinical_df.columns:
        age = pd.to_numeric(clinical_df["AGE"], errors="coerce")
        features["age_at_dx"] = (age - age.mean()) / age.std()

    # Treatment flags
    for col, feat_name in [
        ("RADIATION_THERAPY", "had_radiation"),
        ("HISTORY_OF_NEOADJUVANT_TREATMENT", "had_neoadjuvant"),
    ]:
        if col in clinical_df.columns:
            features[feat_name] = (
                clinical_df[col].str.upper().isin(["YES", "Y"]).astype(int)
            )

    logger.info(f"Clinical features encoded: {list(features.columns)}")
    return features
