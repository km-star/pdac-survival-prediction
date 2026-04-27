# PDAC Survival Prediction — Multi-Omic Pipeline

Reproducing and extending the multi-omic pancreatic ductal adenocarcinoma (PDAC) survival prediction pipeline from [Miao et al., *Nature Cancer* (2024)](https://doi.org/10.1038/s43018-024-00773-4), applied to the **TCGA-PAAD** dataset.

**Best result: ACC = 0.841 (LightGBM ensemble)** — within 0.009 of the Nature Cancer benchmark.

---

## Results Summary

| Model | Accuracy | Notes |
|---|---|---|
| XGBoost (baseline) | 0.658 | Raw features, no balancing |
| XGBoost + class balance | 0.712 | SMOTE oversampling |
| LightGBM | 0.803 | CNA reduction + RNA selection |
| **LightGBM Ensemble (soft-vote)** | **0.841** | Full pipeline, best result |
| Nature Cancer benchmark | ~0.850 | Reference target |

---

## Pipeline Overview

```
TCGA-PAAD (cBioPortal)
    │
    ├── CNA data       ~24,000 genes → ~300 features  (variance filter + survival correlation)
    ├── RNA-seq data   ~20,000 genes → survival-aware top features
    └── Clinical data  stage, age, treatment flags
          │
          ▼
    [Bronze] Raw ingestion → Google Drive / ADLS
          │
          ▼
    [Silver] QC, normalization, ID harmonization (GDC UUID → TCGA barcode)
          │
          ▼
    [Gold]  Feature matrix (589 → 20 features), class-balanced, model-ready
          │
          ▼
    LightGBM + XGBoost soft-voting ensemble
          │
          ▼
    Survival prediction (2-year OS binary classification)
```

---

## Key Technical Challenges Solved

1. **ID format mismatch** — GDC UUIDs vs. TCGA barcodes required a custom mapping step to align multi-omic modalities; without this, ~40% of samples failed to merge.
2. **CNA feature explosion** — Raw CNA matrix had ~24,000 gene columns; reduced to ~300 via variance thresholding + log-rank survival correlation, preventing memory exhaustion in Colab.
3. **Class imbalance** — Short-survivor / long-survivor ratio was ~3:1; addressed with SMOTE + class-weighted loss, improving minority-class recall from 0.31 → 0.67.
4. **Nested API structure** — cBioPortal API returned gene expression in nested JSON; required custom flattening to reconstruct patient-level feature vectors.
5. **Google Drive mount instability** — Colab sessions intermittently lost Drive mounts mid-pipeline; implemented checkpoint saves at each major stage to avoid re-running expensive API fetches.

---

## Repository Structure

```
pdac-survival-prediction/
├── notebooks/
│   ├── 01_data_ingestion.ipynb       # cBioPortal API + Drive ingestion
│   ├── 02_preprocessing.ipynb        # ID harmonization, QC, normalization
│   ├── 03_feature_engineering.ipynb  # CNA reduction, RNA selection, fusion
│   └── 04_modeling.ipynb             # LightGBM, XGBoost, ensemble, eval
├── src/
│   ├── data_loader.py                # cBioPortal API wrapper
│   ├── preprocessing.py              # ID mapping, normalization utils
│   ├── feature_selection.py          # Variance filter, survival correlation
│   ├── models.py                     # Model definitions and ensemble logic
│   └── evaluate.py                   # Metrics, survival curves, plots
├── results/
│   └── model_comparison.csv          # Accuracy, AUC, F1 across all runs
├── docs/
│   └── approach_notes.md             # Design decisions and references
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run in Google Colab (recommended — free GPU/TPU)
Open `notebooks/01_data_ingestion.ipynb` and follow the cell-by-cell instructions. Each notebook saves checkpoints to Google Drive so you can resume across sessions.

### 3. Run locally
```bash
# Clone and install
git clone https://github.com/km-star/pdac-survival-prediction.git
cd pdac-survival-prediction
pip install -r requirements.txt

# Run feature engineering
python src/feature_selection.py --cna data/raw/cna.csv --rna data/raw/rna.csv --out data/processed/

# Train and evaluate
python src/models.py --features data/processed/feature_matrix.csv --target data/processed/labels.csv
```

---

## Data

Data sourced from [cBioPortal TCGA-PAAD](https://www.cbioportal.org/study/summary?id=paad_tcga):
- **CNA** (copy number alterations): GISTIC2 thresholded, 185 patients × ~24K genes
- **RNA-seq** (mRNA expression): RSEM normalized, log2 transformed
- **Clinical**: OS status, OS months, stage, age at diagnosis

> Data is not included in this repository per cBioPortal terms of use. See `docs/approach_notes.md` for download instructions.

---

## References

- Miao, Z. et al. "Multi-omic integration for pancreatic cancer prognosis." *Nature Cancer* (2024). https://doi.org/10.1038/s43018-024-00773-4
- TCGA Research Network: https://www.cancer.gov/tcga
- cBioPortal: https://www.cbioportal.org

---

## Author

**Kumar Mahat** | MS Computer Science, Texas Texas Tech University  
[GitHub](https://github.com/km-star) · [LinkedIn](https://linkedin.com/in/kumar-mahat-b4a431178) · kmahat@ttu.edu
