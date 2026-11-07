# Reproducibility audit of multi-omic PDAC survival prediction

Code, per-patient predictions and tables for a reproducibility audit of the
Molecular Twin platform (Osipov et al., *Nature Cancer* 2024) and a
leakage-free comparison of fold-internal preprocessing pipelines on its released
dataset.

- **`FINDINGS.md`** — what the results show and what the paper can claim.
- **`RESULTS.md`** — every table, generated from the saved predictions
  (leave-one-out); **`RESULTS_rskf.md`** — the same for repeated 5-fold.
- **`results/loo_5seeds/`** — the leave-one-out run behind those tables, seeds
  42–46, plus the 20-shuffle permutation null on three analytes (`report/`
  tables and figures, `preds/` per-patient predictions, `run.log`); `results/rskf_5x10/` is the repeated stratified 5-fold run (10
  repeats); `results/loo_seed42/` is the original single-seed run.
- **`pdac_pipeline.py`** — the whole pipeline in one file.

## Reproduce

```bash
pip install -r requirements.txt
pytest -q tests                                        # 18 tests, about a minute
# put 43018_2023_697_MOESM3_ESM.xlsx in data/ (see data/README.md)
python pdac_pipeline.py --data data/43018_2023_697_MOESM3_ESM.xlsx --outdir results/loo_seed42 --parts audit
python pdac_pipeline.py --data data/43018_2023_697_MOESM3_ESM.xlsx --outdir results/loo_seed42 --parts all --n-jobs 10
python scripts/make_results_md.py results/loo_seed42/report RESULTS.md
```

The full leave-one-out run with one seed took 93 minutes on a 10-core laptop.
Re-running skips cached predictions, so an interrupted run resumes. `--parts
report` rebuilds tables and figures from whatever exists. HPC: `run_hpc.slurm`
(`RUN=loo|rskf|null`). Colab: `pdac_colab.ipynb`.

## What the pipeline does

| Part | Content | Output in `report/` |
|---|---|---|
| `audit` | Column check; deceased counts in MOESM3 vs the published confusion matrices | `table1_dataset`, `label_versions`, `column_audit` |
| `baseline` | The seven released classifiers, seeded, as in Zenodo 8423595 | `table2_reproduction*`, `table3_*`, `table4_*` |
| `pipelines` | P1–P6 fold-internal pipelines | `table3_*`, `table4_*`, `stats_*` |
| `ablation` | RF / L1 / SMOTE / L1+SMOTE / RFECV on every analyte | `table5_ablation`, `fig_ablation_multiomic.png` |
| `alpha_sweep` | P1 at fixed L1 penalties | `fig_alpha_and_labels.png` |
| `oct2021`, `lvs_sim`, `nulltest` | Opt-in: label-flip sensitivity and permutation null | `table6_*`, `lvs_simulation`, `null_check` |

Design rules: leave-one-patient-out outer loop (or `--cv rskf`); every fitted
step, including hyper-parameter tuning and model selection, sees only the
training fold; balanced accuracy = (sensitivity + specificity) / 2, with
accuracy reported alongside; runs cached per (labels, analyte, method, seed).

## The seven released classifiers

Taken line by line from `MTPilot_multiomic_model_training.py` (Zenodo 8423595),
each after `StandardScaler`:

| Name | Released definition |
|---|---|
| SVM | `SGDClassifier()` (hinge, L2, α = 1e-4) |
| L1_SVM | `SGDClassifier(alpha=0.05, penalty="l1")` |
| L1_RF | `SelectFromModel(SGDClassifier(alpha=0.05, penalty="l1"))` + `RandomForestClassifier()` (100 trees) |
| PCA_LR | `PCA(20)` + `LogisticRegression(max_iter=1000)` |
| DNN | `SelectFromModel(SGD L1)` + `MLPClassifier((512, 256, 128, 64, 32), max_iter=1000)` |
| RFE_LR | `RFECV(LogisticRegression(max_iter=1000), step=0.2)` |
| RFE_RF | `RFECV(RandomForestClassifier(), step=0.2)` |

The released code fixes no random seeds and scores each leave-one-out fold
with the `balanced_accuracy` scorer on one patient, which averages to plain
accuracy. `tests/test_pipeline.py` pins these definitions.

## Layout

```
pdac_pipeline.py          pipeline (data, methods, nested CV, statistics, tables, figures)
scripts/make_results_md.py  report folder -> RESULTS.md
tests/                    unit, leakage, planted-signal and model-definition tests (synthetic data)
results/loo_5seeds/       leave-one-out, seeds 42-46: report/, preds/, run.log
results/rskf_5x10/        repeated stratified 5-fold x 10, seed 42
results/loo_seed42/       the original single-seed run
data/README.md            where to get MOESM3 (not redistributed)
run_hpc.slurm             SLURM template; pdac_colab.ipynb  Colab wrapper
```

## Data and code sources

- Osipov, A. et al. The Molecular Twin artificial-intelligence platform integrates multi-omic data to predict outcomes for pancreatic adenocarcinoma patients. *Nat. Cancer* 5, 299–314 (2024). Supplementary Data 3 = MOESM3.
- Osipov, A. et al. Molecular Twin Pilot Source Code. Zenodo, 2023. https://zenodo.org/records/8423595
