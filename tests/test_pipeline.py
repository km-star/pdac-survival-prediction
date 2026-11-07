"""
Tests for pdac_pipeline.py. They use synthetic data only.

    pip install pytest
    pytest -q tests

Tests that need imbalanced-learn, xgboost or lightgbm are skipped when the
package is missing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdac_pipeline as pp  # noqa: E402
from make_synthetic_moesm3 import make  # noqa: E402

SMALL = pp.Cfg(rf_trees=15, released_rf_trees=8, boost_trees=8, n_boot=100)


@pytest.fixture(scope="module")
def noise_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    make(signal=0.0, seed=1, scale=0.5).to_csv(path, index=False)
    return path


# ---- published table and metrics ---------------------------------------------------

def test_published_counts_are_consistent():
    assert pp.published("Multiomic")["n_dead"] == 28 and pp.published("Multiomic")["n"] == 39
    assert pp.published("Clinical")["n_dead"] == 50 and pp.published("Clinical")["n"] == 74
    assert round(pp.published("Multiomic")["ACC"], 2) == 0.85
    # accuracy and balanced accuracy are different numbers
    assert round(pp.published("SNV")["ACC"], 2) == 0.64
    assert round(pp.published("SNV")["BA"], 2) == 0.49


def test_all_deceased_prediction_has_balanced_accuracy_one_half():
    y = np.r_[np.ones(28), np.zeros(11)].astype(int)
    m = pp.metrics(y, np.ones(39, int))
    assert m["BA"] == 0.5 and m["Sens"] == 1.0 and m["Spec"] == 0.0 and m["TN"] == 0
    assert round(m["ACC"], 3) == 0.718


def test_metrics_follow_from_one_confusion_matrix():
    y = np.r_[np.ones(28), np.zeros(11)].astype(int)
    pred = y.copy()
    pred[:2] = 0      # 2 false negatives
    pred[28:35] = 1   # 7 false positives
    m = pp.metrics(y, pred)
    assert (m["TP"], m["FN"], m["TN"], m["FP"]) == (26, 2, 4, 7)
    assert m["PPV"] == pytest.approx(26 / 33) and m["NPV"] == pytest.approx(4 / 6)
    assert m["BA"] == pytest.approx((26 / 28 + 4 / 11) / 2)


def test_mcnemar_and_holm():
    y = np.ones(10, int)
    a, b = y.copy(), y.copy()
    a[:6] = 0                                   # A wrong on six, B right on all
    assert pp.mcnemar_exact(y, a, b) == (6, 0, pytest.approx(0.03125))
    assert pp.mcnemar_exact(y, a, a)[2] == 1.0  # identical predictions
    assert list(pp.holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


# ---- fold-internal building blocks ---------------------------------------------------

def test_imputer_uses_training_rows_only():
    train = np.array([[2.0, np.nan], [4.0, 8.0], [np.nan, 4.0]])
    imp = pp.HalfMedianImputer(zero_mask=np.array([False, False])).fit(train)
    assert list(imp.fill_) == [1.5, 3.0]
    out = imp.transform(np.array([[np.nan, np.nan]]))
    assert list(out[0]) == [1.5, 3.0]
    assert list(pp.HalfMedianImputer(np.array([True, False])).fit(train).fill_) == [0.0, 3.0]


def test_l1_selector_never_returns_zero_features():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(30, 20)), rng.integers(0, 2, 30)
    sel = pp.L1HingeSelector(alpha=50.0, random_state=0).fit(X, y)
    assert sel.fallback_ and sel.get_support().sum() == 1
    assert sel.transform(X).shape == (30, 1)


def test_welch_selector_finds_planted_features():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(20)].astype(int)
    X = rng.normal(size=(40, 50))
    X[:, [3, 7]] += 3 * y[:, None]
    assert set(np.flatnonzero(pp.WelchTSelector(k=2).fit(X, y).get_support())) == {3, 7}


def test_resampling_touches_training_data_only():
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 5)), np.r_[np.ones(30), np.zeros(10)].astype(int)
    rf = pp.RandomForestClassifier(n_estimators=5, random_state=0)
    assert pp.Resampled(rf, "smote", random_state=0).fit(X, y).n_fit_ == 60          # 30 + 30
    assert pp.Resampled(rf, "nearmiss3", ratio=1.5).fit(X, y).n_fit_ <= 25            # 10 + 15
    assert pp.Resampled(rf, "nearmiss3", ratio=5.0).fit(X, y).n_fit_ == 40            # nothing to drop


def test_pos_weight_is_alive_over_deceased():
    class Dummy(pp.ClassifierMixin, pp.BaseEstimator):
        def __init__(self, scale_pos_weight=1.0):
            self.scale_pos_weight = scale_pos_weight

        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self
    y = np.r_[np.ones(28), np.zeros(11)].astype(int)
    w = pp.PosWeighted(Dummy()).fit(np.zeros((39, 1)), y)
    assert w.scale_pos_weight_ == pytest.approx(11 / 28)


# ---- data handling ------------------------------------------------------------------------

def test_loader_and_label_reconstruction(noise_csv):
    ds = pp.load_dataset(noise_csv)
    assert len(ds.y) == 74 and int(ds.y.sum()) == 50
    counts = {a: ds.view(a).n for a in pp.ANALYTES}
    assert counts == {"Clinical": 74, "SNV": 72, "INDEL": 72, "CNV": 72, "RNA": 57,
                      "RNA_Fusion": 57, "Pathology": 71, "Plasma_Protein": 51,
                      "Plasma_Lipid": 51, "Tissue_Protein": 49, "Multiomic": 39}
    used = set(ds.view("Multiomic").cols)
    assert not any(c.startswith("label") or c.startswith("notes") for c in used)
    audit = ds.audit.set_index("column")
    assert audit.loc["label_recurrence", "status"] == "outcome_column_excluded"
    assert audit.loc["notes_misc", "status"] == "unassigned_excluded"
    assert audit.loc["clinical_days_to_last_followup", "suspicious_name"]
    assert audit.loc["clinical_sex_text", "n_non_numeric_lost"] == 74

    y_old, flipped = pp.reconstruct_oct2021(ds.y, ds.days, 5)
    assert int(y_old.sum()) == 45 and len(flipped) == 5
    longest = ds.days[ds.y == 1].sort_values(ascending=False).index[:5]
    assert set(flipped) == set(longest)

    ds2 = pp.load_dataset(noise_csv, exclude_regex="followup|sex_text")
    assert "clinical_days_to_last_followup" not in ds2.view("Clinical").cols


def test_released_models_match_zenodo_definitions():
    """MTPilot_multiomic_model_training.py (Zenodo 8423595), model by model."""
    ctx = pp.Ctx(seed=0, p=50, cfg=pp.Cfg(), zero_mask=np.zeros(50, bool))
    est = {m: pp.build_method(m, ctx)[0] for m in pp.BASE_MODELS}
    assert all(e.steps[1][0] == "scale" for e in est.values())
    assert est["SVM"].steps[-1][1].get_params()["penalty"] == "l2"
    assert est["L1_SVM"].steps[-1][1].get_params()["alpha"] == 0.05
    assert est["L1_RF"].steps[-2][0] == "select" and est["L1_RF"].steps[-1][1].n_estimators == 100
    assert est["PCA_LR"].steps[-2][1].n_components == 20
    assert est["DNN"].steps[-2][0] == "select"
    assert est["DNN"].steps[-1][1].hidden_layer_sizes == (512, 256, 128, 64, 32)
    assert est["RFE_LR"].steps[-1][1].step == 0.2 and est["RFE_RF"].steps[-1][1].step == 0.2
    assert est["RFE_RF"].steps[-1][1].estimator.n_estimators == 100


def test_safe_pca_caps_components():
    X = np.random.default_rng(0).normal(size=(12, 30))
    assert pp.SafePCA(n_components=20).fit(X).n_components_ == 12


# ---- nested selection ------------------------------------------------------------------------

def test_auto_selection_uses_inner_scores_only():
    def frame(pred, inner):
        return pd.DataFrame(dict(rep=0, patient=list("abc"), y=[1, 0, 1], pred=pred,
                                 score=[.5, .5, .5],
                                 score_kind="proba", inner_score=inner, seconds=1.0, params="",
                                 n_selected=1))
    auto = pp.build_auto({"m1": frame([1, 1, 1], [0.9, 0.1, 0.5]),
                          "m2": frame([0, 0, 0], [0.2, 0.8, 0.5])})
    assert list(auto["params"]) == ["m1", "m2", "m1"] and list(auto["pred"]) == [1, 0, 1]


# ---- end to end: leakage and signal --------------------------------------------------------

def _run(tmp_path, df, methods, analyte="Tissue_Protein"):
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    R = pp.Runner(pp.load_dataset(path), tmp_path / "out", SMALL, n_jobs=1)
    return R, {m: pp.frame_metrics(R.run(analyte, "moesm3", m, 0, inner=True))["BA"]
               for m in methods}


def test_no_optimistic_leak_when_p_far_exceeds_n(tmp_path):
    """2,400 pure-noise features, 49 patients. Feature selection done outside the
    cross-validation loop scores about 0.9 here; done inside it must stay near 0.5."""
    pytest.importorskip("imblearn")
    df = make(signal=0.0, seed=3, scale=30)
    R, ba = _run(tmp_path, df, ["P4", "L1_RF", "A1"])
    assert all(v < 0.70 for v in ba.values()), ba
    # the same data with selection outside the loop, to show the test has teeth
    v = R.ds.view("Tissue_Protein")
    y = R.labels["moesm3"].loc[v.patients].to_numpy()
    X = pp.StandardScaler().fit_transform(pp.HalfMedianImputer().fit_transform(v.X))
    Xs = pp.WelchTSelector(k=50).fit(X, y).transform(X)              # leaks: sees all labels
    pred = np.array([pp.LogisticRegression(max_iter=500).fit(
        np.delete(Xs, i, 0), np.delete(y, i)).predict(Xs[i:i + 1])[0] for i in range(len(y))])
    assert pp.metrics(y, pred)["BA"] > 0.85


def test_planted_signal_is_found(tmp_path):
    pytest.importorskip("imblearn")
    _, ba = _run(tmp_path, make(signal=2.5, seed=4, scale=1.0), ["P4", "P1", "L1_RF"])
    assert ba["P4"] > 0.8 and ba["P1"] > 0.75, ba


def test_runs_are_cached(tmp_path):
    df = make(signal=0.0, seed=5, scale=0.3)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    R = pp.Runner(pp.load_dataset(path), tmp_path / "out", SMALL, n_jobs=1)
    first = R.run("RNA_Fusion", "moesm3", "PCA_LR", 0)
    stamp = R.path("moesm3", "RNA_Fusion", "PCA_LR", 0).stat().st_mtime_ns
    again = R.run("RNA_Fusion", "moesm3", "PCA_LR", 0)
    assert R.path("moesm3", "RNA_Fusion", "PCA_LR", 0).stat().st_mtime_ns == stamp
    assert (first["pred"].to_numpy() == again["pred"].to_numpy()).all()


def test_every_method_builds_and_predicts(tmp_path):
    pytest.importorskip("imblearn")
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(36, 12)), np.r_[np.ones(24), np.zeros(12)].astype(int)
    X[rng.random(X.shape) < 0.05] = np.nan
    ctx = pp.Ctx(seed=0, p=12, cfg=SMALL, zero_mask=np.zeros(12, bool))
    for method in pp.BASE_MODELS + pp.PIPELINES + ["A0", "A1", "A2", "P1@alpha=0.01"]:
        est, grid = pp.build_method(method, ctx)
        fold = pp._fit_fold(est, grid, X, y, np.array([0, 30]), 0, 3, True)
        assert set(fold["pred"]) <= {0, 1} and np.isfinite(fold["inner_score"]), method
        assert len(fold["score"]) == 2


def test_repeated_kfold_tests_every_patient_once_per_repeat(tmp_path):
    df = make(signal=0.0, seed=6, scale=0.3)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    cfg = pp.replace(SMALL, cv="rskf", cv_folds=5, cv_repeats=3)
    R = pp.Runner(pp.load_dataset(path), tmp_path / "out", cfg, n_jobs=1)
    out = R.run("RNA_Fusion", "moesm3", "PCA_LR", 0)
    assert len(out) == 3 * 57 and (out.groupby("rep")["patient"].nunique() == 57).all()
    m = pp.frame_metrics(out)
    per_rep = [pp.metrics(g["y"], g["pred"])["BA"] for _, g in out.groupby("rep")]
    assert m["BA"] == pytest.approx(np.mean(per_rep)) and m["repeats"] == 3
    assert pp.boot_ba(out, 50).shape == (50,)
    with pytest.raises(SystemExit):                      # no mixing of schemes in one folder
        pp.Runner(pp.load_dataset(path), tmp_path / "out", SMALL, n_jobs=1)
