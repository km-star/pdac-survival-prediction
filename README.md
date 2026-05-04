# Multi-Omic ML — Pancreatic Cancer Survival Prediction

Reproducing and improving the **Nature Cancer (2024)** PDAC Molecular Twin survival prediction model.  
Original paper: *Chowell et al., Memorial Sloan Kettering Cancer Center*

---

## 📊 Results

| Metric | Published Benchmark | This Work |
|--------|-------------------|-----------|
| Accuracy (ACC) | 0.850 | **0.878** |
| Positive Predictive Value (PPV) | — | **0.918** |
| AUC-ROC | — | **0.903** |

> Exceeded the published benchmark by **+0.028 ACC** using tuned XGBoost/LightGBM ensemble.

---

## 🔬 Methodology

### Dataset
- **TCGA-PAAD** dataset from cBioPortal
- 10 biological data types: mRNA expression, copy number alterations (CNA), protein expression, methylation, somatic mutations, clinical variables, and more

### Feature Engineering Pipeline
```
Raw features: 589 multi-omic molecular features
    ↓ Survival-aware RNA selection
    ↓ CNA feature reduction (~24K → ~300 features)
    ↓ Class balancing (SMOTE)
    ↓ Cross-validated feature importance ranking
Final model: 20-feature parsimonious model (97% complexity reduction)
```

### Models Evaluated
- XGBoost (baseline)
- LightGBM
- Random Forest
- Logistic Regression
- Soft-voting Ensemble (XGBoost + LightGBM) — **best results**

### Key Challenges Solved
- GDC UUID vs TCGA barcode ID format mismatch in raw data
- Nested API gene column structures in expression data
- Data leakage prevention across cross-validation folds
- Class imbalance in survival outcomes

---

## 🛠 Tech Stack

```
Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · TensorFlow
Matplotlib · Seaborn · Google Colab · cBioPortal API · GDC API
```

---

## 📁 Repository Structure

```
pdac-survival-prediction/
├── data/               # Data loading and preprocessing scripts
├── notebooks/          # Jupyter notebooks for EDA and modeling
├── src/
│   ├── preprocessing.py    # Feature engineering pipeline
│   ├── models.py           # Model training and evaluation
│   └── utils.py            # Helper functions
├── results/            # Model performance metrics and plots
└── README.md
```

---

## 🔗 References

- Chowell et al. (2024). *Molecular Twin artificial intelligence platform for patient-level oncology data*. Nature Cancer.
- TCGA-PAAD dataset: [cBioPortal](https://www.cbioportal.org/)
- GDC Data Portal: [portal.gdc.cancer.gov](https://portal.gdc.cancer.gov/)

---

## 👤 Author

**Kumar Mahat** — MS CS, Texas Tech University  
[LinkedIn](https://linkedin.com/in/kumar-mahat-b4a431178) | kmahat@ttu.edu

*Research conducted as part of MS Computer Science program at Texas Tech University.*
