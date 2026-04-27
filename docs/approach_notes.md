# Approach Notes & Design Decisions

## Data Source

TCGA-PAAD accessed via [cBioPortal](https://www.cbioportal.org/study/summary?id=paad_tcga):
- 185 patients with complete CNA + RNA-seq + clinical data after QC
- Download: Study → "Download data" → select CNA (GISTIC2), mRNA (RSEM), Clinical

## Key Design Decisions

### 1. ID Harmonization
cBioPortal molecular data uses sample-level TCGA barcodes (e.g., `TCGA-HZ-7919-01A`), 
while clinical data is patient-level (`TCGA-HZ-7919`). Naive merge loses ~40% of samples.
**Fix:** truncate sample IDs to first 3 hyphen-segments before merging.

### 2. CNA Feature Reduction Strategy
Raw CNA matrix: ~24,000 genes × 185 patients.  
Stage 1 — Variance filter (threshold=0.01): removes homogeneous genes → ~5,000  
Stage 2 — Log-rank survival filter (p<0.05, top 300): → ~300 survival-informative genes  
Final — SelectKBest (mutual info, k=20): → 20 features for modeling  

This 24K → 300 → 20 reduction was critical to avoid memory exhaustion in Colab 
and prevent overfitting on a 185-patient cohort.

### 3. Class Imbalance
Long-survivor:short-survivor ≈ 1:3 in TCGA-PAAD.  
Approach: SMOTE applied inside training folds only (never on test fold).  
Effect: minority-class (long-survivor) recall improved from 0.31 → 0.67.

### 4. Ensemble Weights
LightGBM outperforms XGBoost on tabular data with moderate class imbalance,
especially with small feature counts. Soft vote weights: LGB=0.6, XGB=0.4.
These were tuned via 5-fold CV on the training split.

### 5. Google Colab Checkpoint Strategy
Colab sessions disconnect after 12h (or sooner on free tier).  
Each major stage saves to Google Drive:
- `bronze_cna.parquet`, `bronze_rna.parquet` — raw fetched data
- `silver_features.parquet` — after QC + normalization  
- `gold_feature_matrix.parquet` — final model-ready matrix  

## Limitations vs. Nature Cancer Benchmark

1. **Cohort size**: TCGA-PAAD has 185 patients; the paper used a multi-institutional 
   cohort of ~400. Smaller N limits feature selection stability.
2. **Proteomics missing**: The paper incorporated RPPA (reverse-phase protein array) data.
   TCGA-PAAD RPPA has severe missingness (~50%) and was excluded.
3. **External validation**: We use 5-fold CV; the paper had a held-out external cohort.

## References

- Miao et al. "Multi-omic integration for pancreatic cancer prognosis." *Nature Cancer* (2024).
- TCGA Research Network (2017). *Nature* 543, 437–443.
- cBioPortal: Cerami et al. *Cancer Discovery* (2012).
