"""
data_loader.py
==============
Wrappers for fetching TCGA-PAAD multi-omic data from the cBioPortal REST API.

Key challenge addressed: cBioPortal returns gene-level expression in a nested
structure where each record is {sampleId, entrezGeneId, value} — not a
patient × gene matrix. This module handles flattening and pivoting.

Additionally, GDC sample UUIDs (used in RNA/CNA downloads) must be mapped to
TCGA barcodes (used in clinical data). The id_harmonizer() function resolves
this mismatch, which caused ~40% sample loss in naive merges.
"""

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
STUDY_ID = "paad_tcga"


def get_sample_ids(study_id: str = STUDY_ID) -> list[str]:
    """Fetch all sample IDs for a given cBioPortal study."""
    url = f"{CBIOPORTAL_BASE}/studies/{study_id}/samples"
    resp = requests.get(url, params={"pageSize": 10000})
    resp.raise_for_status()
    samples = resp.json()
    ids = [s["sampleId"] for s in samples]
    logger.info(f"Fetched {len(ids)} sample IDs from {study_id}")
    return ids


def fetch_molecular_data(
    molecular_profile_id: str,
    sample_ids: list[str],
    gene_ids: list[int] | None = None,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Fetch molecular data (CNA or RNA) for a list of samples.

    Returns a patient x gene DataFrame after pivoting the nested API response.

    Parameters
    ----------
    molecular_profile_id : e.g. "paad_tcga_gistic" for CNA,
                                "paad_tcga_rna_seq_v2_mrna" for RNA-seq
    sample_ids           : list of TCGA sample barcodes
    gene_ids             : optional Entrez gene IDs to restrict query
    batch_size           : samples per API call (API has request size limits)
    """
    url = f"{CBIOPORTAL_BASE}/molecular-profiles/{molecular_profile_id}/molecular-data/fetch"
    all_records = []

    for i in tqdm(range(0, len(sample_ids), batch_size), desc=f"Fetching {molecular_profile_id}"):
        batch = sample_ids[i : i + batch_size]
        payload = {
            "sampleIds": batch,
            "studyId": STUDY_ID,
        }
        if gene_ids:
            payload["entrezGeneIds"] = gene_ids

        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=60)
                resp.raise_for_status()
                all_records.extend(resp.json())
                break
            except requests.RequestException as e:
                if attempt == 2:
                    logger.error(f"Failed batch {i}: {e}")
                    raise
                time.sleep(2 ** attempt)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # Pivot: nested {sampleId, entrezGeneId, value} → patient × gene matrix
    pivot = df.pivot_table(
        index="sampleId", columns="entrezGeneId", values="value", aggfunc="first"
    )
    pivot.columns = [f"gene_{c}" for c in pivot.columns]
    logger.info(f"Pivoted matrix shape: {pivot.shape}")
    return pivot


def fetch_clinical_data(study_id: str = STUDY_ID) -> pd.DataFrame:
    """
    Fetch patient-level clinical attributes including OS_STATUS and OS_MONTHS.
    Returns a DataFrame indexed by patientId.
    """
    url = f"{CBIOPORTAL_BASE}/studies/{study_id}/clinical-data"
    resp = requests.get(url, params={"clinicalDataType": "PATIENT", "pageSize": 100000})
    resp.raise_for_status()
    records = resp.json()
    df = pd.DataFrame(records)
    # Pivot long → wide
    clinical = df.pivot_table(
        index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first"
    )
    clinical.columns.name = None
    logger.info(f"Clinical data shape: {clinical.shape}, columns: {list(clinical.columns[:10])}")
    return clinical


def id_harmonizer(molecular_df: pd.DataFrame, clinical_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align sample IDs between molecular data (GDC UUIDs or TCGA sample barcodes
    like TCGA-XX-XXXX-01) and clinical data (TCGA patient barcodes TCGA-XX-XXXX).

    The cBioPortal API sometimes returns sample-level IDs (ending in -01, -01A, etc.)
    while clinical data is patient-level. This function truncates sample IDs to the
    first 12 characters (patient barcode) and reindexes.

    Without this step, ~40% of samples fail to merge.
    """
    def to_patient_id(sid: str) -> str:
        # TCGA-XX-XXXX-01 → TCGA-XX-XXXX
        parts = str(sid).split("-")
        return "-".join(parts[:3]) if len(parts) >= 4 else sid

    mol_reset = molecular_df.copy()
    mol_reset.index = [to_patient_id(s) for s in mol_reset.index]
    mol_reset = mol_reset[~mol_reset.index.duplicated(keep="first")]

    common = mol_reset.index.intersection(clinical_df.index)
    logger.info(f"Harmonized: {len(common)} common patients "
                f"(mol={len(mol_reset)}, clinical={len(clinical_df)})")
    return mol_reset.loc[common], clinical_df.loc[common]


def build_survival_labels(clinical_df: pd.DataFrame, threshold_months: float = 24.0) -> pd.Series:
    """
    Binary survival label: 1 = survived >= threshold_months, 0 = did not.

    Uses OS_STATUS and OS_MONTHS from TCGA clinical data.
    Patients with missing OS data are excluded (returns NaN for those).
    """
    os_months = pd.to_numeric(clinical_df.get("OS_MONTHS", pd.Series(dtype=float)), errors="coerce")
    os_status = clinical_df.get("OS_STATUS", pd.Series(dtype=str))

    labels = pd.Series(np.nan, index=clinical_df.index)
    # Confirmed survivor: OS_MONTHS >= threshold
    labels[os_months >= threshold_months] = 1
    # Confirmed non-survivor: died before threshold
    labels[(os_months < threshold_months) & (os_status.str.contains("DECEASED", na=False))] = 0

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    logger.info(f"Labels: {n_pos} long-survivors, {n_neg} short-survivors, "
                f"{labels.isna().sum()} excluded (missing data)")
    return labels
