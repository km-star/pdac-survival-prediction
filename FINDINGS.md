# Findings for the journal manuscript

Written for the co-author drafting the paper. Every number here is in
`RESULTS.md` and traceable to `results/loo_seed42/report/*.csv` and the
per-patient predictions in `results/loo_5seeds/preds/`. The run is
leave-one-patient-out cross-validation with five seeds (42–46); seed 42 is the
primary seed behind the headline tables and `results/loo_seed42/` is the
original single-seed run kept for reference. Table 6 of RESULTS.md gives the
spread over seeds.

## 1. What the study is now

A reproducibility audit of the Molecular Twin platform (Osipov et al., Nature
Cancer 2024) using its released code (Zenodo 8423595) and released dataset
(Supplementary Data 3, "MOESM3"), followed by a leakage-free comparison of six
alternative pipelines under balanced accuracy. The framing of our earlier
internal draft (five implementation corrections, a +0.205 gain, a label-version
explanation of the multi-omic gap, LightGBM AUC 0.857) is not supported and
must not be reused. Section 6 lists what to drop and why.

## 2. Facts established from the original sources

1. **The published Table 1 was computed on the released labels.** For all 11
   analytes, TP + FN in the published confusion matrix equals the number of
   deceased patients in MOESM3 (`label_versions.csv`). The article's text says
   45 of 74 patients were deceased as of 21 October 2021; its own Table 1 and
   the released file give 50. This is an internal inconsistency of the article
   worth one sentence, not a basis for reanalysis.
2. **The published metric is plain accuracy.** The released script calls
   scikit-learn's `balanced_accuracy` scorer inside leave-one-out
   cross-validation. With one test patient per fold the score is 1 or 0, and
   the mean over folds is accuracy. The article calls it accuracy. Because the
   cohort is 68–72% deceased, predicting every patient deceased already scores
   0.65–0.72.
3. **The released code matches the article's description.** The seven
   classifiers are SGD hinge-loss SVM (L2), SGD SVM with L1 penalty, L1
   selection + 100-tree random forest, PCA(20) + logistic regression, L1
   selection + a five-layer MLP (512-256-128-64-32), RFECV(step 0.2) + logistic
   regression, RFECV(step 0.2) + 100-tree random forest, each after
   standardisation. Neither the article nor the released hyper-parameter
   document mentions an RBF kernel, a smaller MLP or class weights. The only
   reproducibility defect in the released code itself is the absence of random
   seeds. MOESM3 has no missing cell inside any analyte block, so imputation
   choices cannot change a result; patients without an analyte are excluded
   from it, as in the released code.
4. **Multi-omic cohort.** 39 patients have all ten analytes: 28 deceased, 11
   alive, 6,355 features.

## 3. Reproduction (Table 2 in RESULTS.md)

Running the seven released models with a fixed seed on MOESM3:

- Two published confusion matrices are reproduced exactly: RNA Fusion with the
  L1 SVM (35/12/8/2, accuracy 0.754) and Plasma Lipid with RFE logistic
  regression (accuracy 0.706).
- For 10 of 11 analytes, the best-of-seven accuracy is within 0.06 of the
  published value (range of gaps −0.028 to +0.056).
- The multi-omic analyte is the exception: best-of-seven accuracy 0.744
  against the published 0.846, a difference of four patients out of 39. No
  released model comes closer than eight patients (L1 distance between
  confusion matrices) to the published 26/4/7/2. The random-forest and MLP
  models are unseeded in the released code, so seed variation was checked: over
  five seeds and seven models (35 runs) the multi-omic accuracy ranges from
  0.538 to 0.769, with the best-of-seven per seed at 0.718–0.769. The published
  0.846 is outside this range by three patients at best.
- Seed spread elsewhere is small: best-of-seven accuracy moves by at most 0.07
  across seeds on any analyte, and the published value lies inside or within
  0.03 of the five-seed range for the other ten analytes.
- The published values, re-expressed as balanced accuracy, lie between 0.49
  (SNV, 1 of 24 alive patients identified) and 0.67 (RNA Fusion), with the
  multi-omic row at 0.78. Table 1 of RESULTS.md gives all of them.

Suggested framing: the released code and data reproduce the published
single-analyte results; the headline multi-omic result does not reproduce
within four patients; and the published metric does not distinguish the models
from a majority-class predictor.

## 4. Fair comparison of pipelines (Tables 3 and 4)

Protocol: leave-one-patient-out; imputation, scaling, feature selection,
resampling, hyper-parameter tuning and model selection all fitted inside each
training fold; balanced accuracy as the primary metric. "Step 2" is the seven
released models with per-fold nested selection; "Step 3" is the six pipelines
P1–P6 with the same nested selection.

| Quantity | Value |
|---|---|
| Step 2 mean balanced accuracy, 11 analytes | 0.531 |
| Step 3 mean balanced accuracy | 0.559 |
| Step 3 − Step 2, bootstrap 95% interval over analytes | +0.027 [−0.038, +0.091] |
| Friedman test, 7 treatments × 11 analytes, balanced accuracy | χ² = 7.58, p = 0.27 |
| Best single pipeline (P5, L1 + NearMiss-3 + RF): wins/losses vs Step 2 | 8/3, sign test p = 0.23, Wilcoxon p = 0.18 |
| Per-analyte McNemar p below 0.05 | INDEL (Step 3 worse, 0.31 vs 0.53) and Pathology (Step 3 worse, 0.57 vs 0.66) |

Interpretation: no pipeline improves on the released models at a level that
survives the portfolio test; both steps are near chance on most analytes.
Across the five seeds the mean Step 3 − Step 2 gap is −0.009 to +0.036, never
significant; the between-analyte standard deviation of the gap (about 0.11) is
four times its mean. One analyte stands out consistently: Tissue Protein, where
Step 3 scores 0.74–0.84 over the five seeds (mean 0.78) against 0.51–0.67 for
Step 2, with P1 and P5 agreeing. Plasma Protein (0.53–0.66 over seeds) does not
hold up. Tissue Protein should be reported as a candidate finding; the
permutation test (`nulltest`, running) decides whether it exceeds chance.

Multi-omic discrimination (Table 4): P5 has the highest AUC, 0.72 [0.54, 0.89];
the released RFECV + RF has AUC 0.71 but identifies zero alive patients
(balanced accuracy 0.48); LightGBM DART has AUC 0.53 and predicts every patient
deceased on SNV, INDEL and RNA Fusion. Recommendation: drop P6 from the
pipeline set or report it only as a negative result.

## 5. Ablation (Table 5)

On the multi-omic analyte every configuration is at or below chance: RF alone
0.500 (all deceased), L1 + RF 0.464 (0 alive found), SMOTE + RF 0.510, L1 +
SMOTE + RF 0.528 (1 alive found), RFECV + RF 0.482 (0 alive found). The claim
that L1 selection prevents majority-class collapse is not supported there.

On lower-dimensional analytes the combination matters: on Clinical, L1 + SMOTE
+ RF finds 7 of 24 alive patients (balanced accuracy 0.586) where RF alone
finds none; on RNA it finds 11 of 20 (0.626) against 3 for RF alone. A defensible
statement is that resampling after sparse selection raises specificity on some
single-analyte problems, with no evidence of benefit at p = 6,355, n = 39.

## 6. Claims from the earlier internal draft that must go, with the reason

| Claim | Status |
|---|---|
| Five discrepancies between methods and released code; correcting them raised mean balanced accuracy 0.705 → 0.910 | Three of the five are not in the article or hyper-parameter document; the other two cannot change a result. The earlier Step 1 and Step 2 ran identical models; the gain came from the five flipped labels and plain accuracy reported as balanced accuracy. |
| "October 2021 labels" (23/16) explain the multi-omic gap; LVS metric | The published confusion matrices use the MOESM3 labels (28/11) for every analyte. |
| RFECV + RF reaches 0.974 on the multi-omic analyte | It reaches 0.482 balanced accuracy, 0.692 accuracy, zero alive patients identified. |
| Friedman χ² = 8.43, p = 0.038 (Step 3 > Step 2) | p = 0.27 on balanced accuracy with the correct seven-treatment design. |
| L1 selection is the critical component preventing collapse | Not on the multi-omic analyte (0 alive found with or without L1). |
| LightGBM DART AUC 0.857, sensitivity 0.929 | AUC 0.53; collapses on three analytes. |
| Balanced accuracy values throughout | Were plain accuracy. The published 0.850 is 33/39 correct. |

## 7. Methods details the paper must state

| Item | Setting |
|---|---|
| Outcome | Vital status at the MOESM3 cutoff (binary); no time-to-event modelling |
| Cohort per analyte | Patients with that analyte present (74, 72, 72, 72, 57, 57, 71, 51, 51, 49); multi-omic 39 |
| Preprocessing | Standardisation per training fold; half-median imputation is in the code as a safeguard but never triggers |
| Step 2 | Seven released models as above, seeded; model chosen per outer fold by inner stratified 3-fold balanced accuracy |
| P1 | L1 hinge selection, α ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5} by inner CV; SMOTE k = min(5, n_alive − 1) to parity; RF 500 trees |
| P2 | RFECV (logistic regression, step 0.2, balanced accuracy) + XGBoost (300 trees, depth 3, η 0.05, λ 2, scale_pos_weight = n_alive / n_deceased) |
| P3 | Stacking of the seven released models, out-of-fold meta-features from stratified 3-fold inside each training fold, logistic-regression meta-learner |
| P4 | Welch t-test top-K, K ∈ {50, 100, 200, 500} by inner CV; SMOTE; RF 500 |
| P5 | L1 hinge selection (α as P1) + NearMiss-3 keeping r × n_alive deceased, r ∈ {1, 1.5, 2} by inner CV; RF 500 |
| P6 | LightGBM DART, 300 rounds, 15 leaves, η 0.05, drop rate 0.1, λ1 = λ2 = 2, min_child_samples 5, scale_pos_weight as P2 |
| Statistics | Friedman over 7 treatments; per-pipeline Wilcoxon and sign tests with Holm; paired stratified bootstrap (1,000); exact McNemar; permutation null when run |
| Software | Python 3.14, scikit-learn 1.8.0, xgboost 3.2.0, lightgbm 4.6.0, imbalanced-learn 0.14.2 |

## 8. Suggested structure

1. Introduction: small-cohort multi-omic prognosis; why reproducibility audits
   matter; the Molecular Twin study as the only public ten-analyte PDAC cohort.
2. Materials: MOESM3, Zenodo code, the label-count inconsistency, the metric.
3. Reproduction: Table 2 and Figure `fig_step_comparison.png`.
4. Leakage-free comparison: protocol, Tables 3–4, heat map.
5. Ablation: Table 5 and `fig_ablation_multiomic.png`.
6. Discussion: accuracy versus balanced accuracy under 70% imbalance; what the
   reproduction does and does not establish; LOO variance; recommendations
   (report confusion matrices, seeds, cutoff dates, and majority-class baselines).
7. Limitations: single cohort; one label version available; LOO variance;
   per-analyte findings are hypothesis-generating.

## 9. Seed sensitivity (Table 6)

Standard deviation of balanced accuracy over five seeds, averaged over
analytes: 0.01 (PCA + LR, no randomness beyond the inner folds) to 0.07 (P5);
the nested Step 2 and Step 3 selections sit at 0.05–0.06. Leave-one-out
estimates on these cohorts therefore move by about ±0.05 from the random seed
alone, which is the same size as most of the between-method differences. The
paper should report the seed range, not a single seed.

## 10. Repeated stratified 5-fold cross-validation (RESULTS_rskf.md)

The same protocol with stratified 5-fold cross-validation repeated 10 times
(every patient tested 10 times; `results/rskf_5x10/`). This is the
lower-variance check; its numbers are means over the 10 repeats, with the
between-repeat standard deviation in `metrics_long.csv` (0.03–0.09).

| Quantity | Leave-one-out (seed 42) | Repeated 5-fold |
|---|---|---|
| Step 2 mean balanced accuracy | 0.531 | 0.533 |
| Step 3 mean balanced accuracy | 0.559 | 0.557 |
| Step 3 − Step 2, 95% interval over analytes | +0.027 [−0.038, +0.091] | +0.024 [−0.008, +0.072] |
| Friedman p (balanced accuracy) | 0.27 | 0.35 |
| Best-of-7 accuracy within 0.05 of published | 10 of 11 analytes | 10 of 11 analytes |
| Multi-omic best-of-7 accuracy vs published 0.846 | 0.744 | 0.728 |
| Tissue Protein: Step 2 / Step 3 | 0.63 / 0.81 | 0.54 / 0.79 |
| Tissue Protein: Step 3 − Step 2, 95% interval | +0.18 [−0.01, +0.36] | +0.25 [+0.14, +0.35] |
| Multi-omic P5 AUC | 0.72 | 0.72 |

The two schemes agree on every conclusion. Tissue Protein is the single analyte
where the fold-internal pipelines separate from the released models: P1 (L1 +
SMOTE + RF) 0.81 and P5 (L1 + NearMiss + RF) 0.74 under repeated k-fold, with
between-repeat standard deviations of 0.04–0.06, against 0.54 for the released
models. On the multi-omic analyte the picture is unchanged: everything near
chance, P5 the best discriminator at AUC 0.72. Stacking (P3) is the one
pipeline that is consistently below Step 2 under repeated k-fold (3 wins, 8
losses, mean −0.023 [−0.046, −0.001]).

## 11. Permutation null (Table 7 of RESULTS.md)

For Tissue Protein, Plasma Protein and the multi-omic analyte, every method
except stacking was re-run end to end on 20 label shuffles (class counts
unchanged; 720 leave-one-out runs). `p_perm` = (1 + number of shuffles scoring
at or above the observed value) / 21, so the smallest attainable p is 0.048,
reached only when no shuffle matches the observed result.

| Analyte | Method | Observed BA | Null mean (sd) | Null max | p_perm |
|---|---|---|---|---|---|
| Tissue Protein | Step 3 nested | 0.807 | 0.505 (0.107) | 0.629 | 0.048 |
| Tissue Protein | P1 L1+SMOTE+RF | 0.779 | 0.515 (0.090) | 0.693 | 0.048 |
| Tissue Protein | P5 L1+NearMiss+RF | 0.771 | 0.514 (0.115) | 0.671 | 0.048 |
| Tissue Protein | Step 3 − Step 2 | +0.179 | 0.018 (0.082) | 0.164 | 0.048 |
| Tissue Protein | Step 2 (released models) | 0.629 | 0.486 (0.088) | 0.643 | 0.095 |
| Plasma Protein | P1 | 0.638 | 0.503 (0.087) | 0.633 | 0.048 |
| Plasma Protein | Step 3 nested | 0.651 | 0.491 (0.087) | 0.713 | 0.095 |
| Multi-omic | best of any method (SVM, DNN) | 0.648, 0.628 | 0.48–0.49 (0.08–0.11) | 0.66–0.70 | 0.095 |
| Multi-omic | P5 | 0.630 | 0.500 (0.130) | 0.756 | 0.143 |
| Multi-omic | P1 | 0.528 | 0.506 (0.119) | 0.802 | 0.381 |

Reading:

- **Tissue Protein is the one result that beats every shuffle.** The two sparse
  resampling pipelines and the nested Step 3 selection all exceed the best of
  20 shuffled-label runs, and so does their margin over the released models.
  The released models on the same analyte do not (p = 0.095–0.24). This is
  consistent across leave-one-out, five seeds and repeated 5-fold (Sections
  4, 9, 10). Report it as a candidate finding that survives a 20-shuffle
  permutation test, with the caveats that 20 shuffles bound p at 0.048, that
  14 methods × 3 analytes were examined, and that n = 49 with 14 alive patients.
- **Plasma Protein is weak**: only P1 reaches p = 0.048; the nested selection
  does not. Do not present it as a signal.
- **The multi-omic analyte shows no method above chance.** The permutation
  distributions are wide (sd 0.08–0.13) and shuffled labels produced balanced
  accuracies up to 0.80 (P1) and 0.76 (P5, P6). This is the number to quote
  when discussing why single leave-one-out values on 39 patients cannot
  support claims like the earlier draft's 0.974 or 0.857.

## 12. Status

All planned runs are complete: leave-one-out with five seeds, repeated
stratified 5-fold with ten repeats, and the permutation null on three
analytes. Nothing further is running. Additional shuffles (to tighten the
0.048 floor) or the null on the remaining eight analytes could be added with
`--parts nulltest`; each shuffle of all analytes costs about 15 CPU-hours.
