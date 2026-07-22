# Reproducibility and Sparse Feature Selection for Multi-Omic PDAC Survival Prediction

Code accompanying **"Reproducibility and Sparse Feature Selection for Small Imbalanced Multi-Omic Pancreatic Ductal Adenocarcinoma Survival Prediction"** — under review, IEEE BIBM 2026.

Kumar Mahat, Rohit Soni, Uma Chinta, Xiuzhen Huang, Victor S. Sheng
Texas Tech University · Cedars-Sinai Medical Center

---

## What this is

A reproducibility audit of the [Molecular Twin](https://www.nature.com/articles/s43018-024-00775-4) PDAC survival platform (Osipov et al., *Nature Cancer*, 2024), plus a set of leakage-free preprocessing pipelines evaluated against a same-dataset baseline.

Three findings:

1. **The released code does not match the published methods.** Five concrete discrepancies. Correcting them raised mean balanced accuracy across 11 analyte modalities from **0.705 → 0.910**.
2. **L1 sparse selection is what prevents majority-class collapse** at p ≫ n — not the resampling step. SMOTE without prior feature selection provides no benefit whatsoever (TN = 0, unchanged).
3. **The published multi-omic result is not reproducible from the released dataset**, and the reason is a label-version difference, not an implementation failure. We introduce a metric to quantify this.

---

## The five discrepancies

Found by executing `MTPilot_training.py` from [Zenodo record 8423595](https://zenodo.org/records/8423595) and comparing against the paper's methods section:

| # | Published method | Released implementation |
|---|---|---|
| 1 | `SVC(kernel='rbf')` | linear `SGDClassifier` |
| 2 | MLP layers (256, 128, 64, 32, 16) | (512, 256, 128, 64, 32) |
| 3 | Fixed-threshold RFE | `RFECV(step=0.2)` |
| 4 | Missing values imputed | Missing values dropped |
| 5 | `class_weight` applied | No `class_weight` argument passed |

Gains after correction were uniform across all 11 analytes (+0.087 to +0.332, mean +0.205, bootstrap CI [+0.163, +0.247]) — consistent with genuine fixes rather than selective tuning.

---

## Three-step comparison protocol

The released MOESM3 dataset reflects a later follow-up cutoff than the published analysis (28 deceased / 11 alive vs. 23 / 16 in the multi-omic subset). Comparing directly against the paper's Table 1 would therefore be invalid. We separate the two effects:

| Step | What it measures | Result |
|---|---|---|
| **1. Reproduction** | Corrected implementation, October 2021 labels | 11/11 exceed paper baseline · mean 0.910 |
| **2. Fair baseline** | Paper's own models, MOESM3 labels | mean 0.702 — **the primary comparison baseline** |
| **3. Novel pipelines** | Our pipelines, MOESM3 labels | 8/11 exceed Step 2 · mean 0.721 |

Friedman χ² = 8.43, **p = 0.038** across 11 analytes. Bootstrap 95% CI [+0.002, +0.031].

Step 2 is the honest baseline. Comparisons against Step 1 or against the paper's published table are reported for completeness but are not the claim.

---

## Pipelines

All components refit **inside every LOO fold** — scaling, selection, resampling. No exceptions.

| | Pipeline |
|---|---|
| **P1** | L1 sparse selection → SMOTE → Random Forest |
| **P2** | RFECV → XGBoost |
| **P3** | Stacking meta-learner over 7 base classifiers |
| **P4** | Welch t-test filter → SMOTE → Random Forest |
| **P5** | L1 selection → NearMiss-3 undersampling → Random Forest |
| **P6** | LightGBM DART |

Best discrimination on the multi-omic analyte: **P6, AUC 0.857, sensitivity 0.929**.

---

## Ablation: L1 selection is the critical component

Multi-omic analyte, n = 39, p = 6,355. TN = alive patients correctly identified; TN = 0 means the classifier predicted every patient deceased.

| Config | Bal. Acc | TN | |
|---|---|---|---|
| A0 · RF only | 0.718 | 0 | majority-class collapse |
| A1 · L1 + RF | 0.744 | 3 | **collapse broken** |
| A2 · SMOTE + RF | 0.718 | 0 | no gain — SMOTE alone does nothing |
| A3 · L1 + SMOTE + RF | 0.744 | 3 | no gain over A1 |

Ordering matters: **feature selection must precede resampling** when p ≫ n. In a 6,355-dimensional space with 38 training patients per fold, synthetic minority samples are indistinguishable from noise.

---

## Label-version sensitivity

```
LVS(f) = Δ Balanced Accuracy / Δ n_deceased
```

Five additional deaths recorded after the original cutoff shifted the majority-class prior from 59.0% → 71.8%:

| Model | Oct-2021 labels | MOESM3 labels | LVS |
|---|---|---|---|
| RFECV + RF | 0.974 | 0.718 | −0.051 / death |
| P5 | 0.974 | 0.744 | −0.046 / death |

On matched labels, P5 outperforms the original method by +0.026 and is less sensitive to follow-up updates.

**Takeaway for the field:** survival ML studies should report the outcome-label cutoff date and class distribution alongside performance metrics. A faithful reimplementation will still produce different numbers if the labels have since been updated.

---

## Data

MOESM3, supplementary file from Osipov et al. (2024). 74 resected stage I/II PDAC patients, 10 analyte modalities, 6,363 features. Multi-omic subset: n = 39 (28 deceased, 11 alive), p = 6,355.

Not redistributed here — obtain from the paper's supplementary materials.

---

## Stack

Python · scikit-learn · imbalanced-learn · XGBoost · LightGBM · pandas · NumPy
Compute: TTU HPCC Matador partition, NVIDIA Tesla V100-PCIE-32GB

---

## Limitations

Single-institution cohort. LOOCV at n < 50 is high-variance, so per-analyte pipeline rankings are hypothesis-generating rather than conclusive. External validation on independent PDAC cohorts is required before any per-analyte finding guides method selection in practice.

---

## Citation

```bibtex
@inproceedings{mahat2026pdac,
  title     = {Reproducibility and Sparse Feature Selection for Small Imbalanced
               Multi-Omic Pancreatic Ductal Adenocarcinoma Survival Prediction},
  author    = {Mahat, Kumar and Soni, Rohit and Chinta, Uma and
               Huang, Xiuzhen and Sheng, Victor S.},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026},
  note      = {Under review}
}
```
