# PDAC Multi-Omic Survival Prediction

**Novel ML Methods for Pancreatic Ductal Adenocarcinoma Survival Prediction on Small Imbalanced Multi-Omic Cohorts**

Kumar Mahat · Rohit Soni · Victor S. Sheng  
Texas Tech University · IEEE BIBM 2026 (under review)

---

## Overview

This repository reproduces and extends the [Molecular Twin platform](https://doi.org/10.1038/s43018-023-00697-7) (Osipov et al., *Nature Cancer* 2024) for PDAC post-resection survival prediction. We propose six novel ML method families designed specifically for small (n<100), highly imbalanced multi-omic cohorts, and provide the first mechanistic explanation of the majority-class collapse phenomenon in this setting.

**Key results (MOESM3 dataset, 74 patients, 11 analyte modalities, LOOCV):**

| Result | Value |
|--------|-------|
| Novel methods beating Paper Table 1 (fair comparison) | **8 / 11 analytes** |
| Best reproduction ACC (Multi-omic, Step 1) | **0.949** vs paper 0.850 |
| LightGBM DART AUC (Multi-omic) | **0.857** |
| LightGBM DART Sensitivity (Multi-omic) | **0.929** |
| Implementation corrections identified from Zenodo source | **5** |

---

## Dataset

**MOESM3** — `43018_2023_697_MOESM3_ESM.xlsx`  
Supplementary dataset from Osipov et al., *Nature Cancer* 2024.  
Download from [Springer](https://doi.org/10.1038/s43018-023-00697-7) → Supplementary Information → Source Data Fig. 1.

| Analyte | Features | Patients |
|---------|----------|----------|
| Clinical | 331 | 74 |
| SNV | 610 | 72 |
| INDEL | 125 | 72 |
| CNV | 648 | 72 |
| RNA Expression | 2,000 | 57 |
| RNA Fusion | 29 | 57 |
| Computational Pathology | 820 | 71 |
| Plasma Protein | 257 | 51 |
| Plasma Lipid | 406 | 51 |
| Tissue Protein | 1,130 | 49 |
| **Multi-omic** | **6,355** | **39** |

---

## Novel Methods (M1–M6)

| ID | Method | Key Idea |
|----|--------|----------|
| M1 | SMOTE + L1 + RF | L1 sparse selection inside LOO, then SMOTE on selected features only |
| M2 | RFECV + XGBoost | Cross-validated recursive feature elimination driving XGBoost with class weighting |
| M3 | Stacking Meta-Learner | 7 paper models as level-0 learners; logistic regression meta-learner |
| M4 | Ttest + SMOTE + RF | Welch t-test survival discriminant filter; more powerful than L1 for RNA |
| M5 | NearMiss + L1 + RF | Boundary-focused majority undersampling; uses real patients, not synthetic |
| M6 | LightGBM DART | Dropout regularization for small cohorts; best AUC=0.857 |

---

## Three-Step Fair Comparison Protocol

```
Step 1 — Reproduce on Oct-2021 labels (45 deceased)
         Verifies implementation correctness vs published Table 1

Step 2 — Paper models on MOESM3 (50 deceased)
         Establishes fair same-dataset baseline

Step 3 — Novel methods on MOESM3 (50 deceased)
         Valid comparison: Step 3 vs Step 2, same dataset
         Result: 8/11 analytes beat Paper Table 1
```

---

## Setup

```bash
# Clone
git clone https://github.com/km-star/pdac-survival-prediction.git
cd pdac-survival-prediction

# Create conda environment
conda create -n pdac python=3.11 -y
conda activate pdac

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Full pipeline (Steps 1-6, ~6-10 hours on GPU)
python pdac_comprehensive_final.py \
    --data  ./43018_2023_697_MOESM3_ESM.xlsx \
    --outdir ./results \
    --part  all

# Individual parts
python pdac_comprehensive_final.py --data ./MOESM3.xlsx --outdir ./results --part step1
python pdac_comprehensive_final.py --data ./MOESM3.xlsx --outdir ./results --part step3
python pdac_comprehensive_final.py --data ./MOESM3.xlsx --outdir ./results --part ablation
python pdac_comprehensive_final.py --data ./MOESM3.xlsx --outdir ./results --part figures
```

**CLI options:**

| Flag | Description |
|------|-------------|
| `--data` | Path to MOESM3 Excel file |
| `--outdir` | Output directory for CSVs and figures |
| `--part` | `all` / `step1` / `step2` / `step3` / `ablation` / `stats` / `figures` |

**Resume support:** If the job is interrupted, rerun with the same `--outdir` — completed analytes are skipped automatically.

---

## HPCC (TTU Matador Partition)

```bash
# Submit Slurm job (Tesla V100-PCIE-32GB)
sbatch run_comprehensive_final.sh

# Monitor
squeue -u kmahat
tail -f pdac_final_JOBID.log
```

Expected runtimes on V100:

| Part | Time |
|------|------|
| Step 1 Reproduction | ~25–40 min |
| Step 2 Fair baseline | ~25–40 min |
| Step 3 Novel methods | ~3–5 hours |
| Step 4 Ablation | ~1–2 hours |
| Step 5 McNemar tests | ~1–2 hours |
| **Total** | **~6–10 hours** |

---

## Outputs

| File | Description |
|------|-------------|
| `FINAL_SUMMARY.csv` | Best ACC per analyte across all steps |
| `step1_reproduction.csv` | Step 1 full results (7 models × 11 analytes) |
| `step2_fair_baseline.csv` | Step 2 fair baseline results |
| `step3_novel_methods.csv` | Step 3 novel methods results with 95% CI |
| `ablation_results.csv` | Ablation: RF / L1+RF / SMOTE+RF / L1+SMOTE+RF / RFECV+RF |
| `mcnemar_tests.csv` | McNemar χ² and Bonferroni-corrected p-values |
| `fig1_accuracy_comparison.png` | Per-analyte ACC bar chart |
| `fig2_ablation_multiomic.png` | Ablation study — ACC and TN |
| `fig3_three_step_summary.png` | Three-step protocol summary |
| `fig4_l1_alpha_sensitivity.png` | L1 alpha sweep — Multi-omic |
| `fig5_delta_heatmap.png` | Full Δ heatmap — all methods × analytes |
| `fig6_lgbm_dart_clinical.png` | LGBM DART AUC and sensitivity |

---

## Implementation Corrections from Zenodo Source

Five discrepancies identified between the paper description and `MTPilot_multiomic_model_training.py`:

| # | Component | Paper Description | Zenodo Actual | Impact |
|---|-----------|-----------------|---------------|--------|
| 1 | SVM model | SVC(kernel='rbf') | SGDClassifier() — linear L2 | High |
| 2 | MLP layers | (256,128,64,32,16) | (512,256,128,64,32) | Medium |
| 3 | Feature elim. | RFE(n_feat=0.5) | RFECV(step=0.2, cv=LOO) | High |
| 4 | Null handling | fillna(median) | dropna() per analyte | High |
| 5 | Class weight | class_weight='balanced' | No class weighting | Medium |

---

## Citation

```bibtex
@inproceedings{mahat2026pdac,
  title     = {Novel ML Methods for PDAC Survival Prediction on Small
               Imbalanced Multi-Omic Cohorts},
  author    = {Mahat, Kumar and Soni, Rohit and Sheng, Victor S.},
  booktitle = {IEEE International Conference on Bioinformatics and
               Biomedicine (BIBM)},
  year      = {2026},
  address   = {Dallas, TX, USA}
}
```

---

## Acknowledgments

Computational resources: TTU HPCC Matador partition, NVIDIA Tesla V100-PCIE-32GB.  
Dataset: Osipov et al. MOESM3 ([Zenodo 8423595](https://zenodo.org/records/8423595)).  
Advisor: Dr. Victor S. Sheng, Texas Tech University.
