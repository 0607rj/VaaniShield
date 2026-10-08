import numpy as np
import pandas as pd
import pytest

from ml.evaluation.metrics import brier, det_curve, ece, eer, group_bootstrap, nll, rates_at_threshold, roc_auc


def _brute_force_eer(scores, labels):
    """Independent implementation: thresholds at every score and +inf, decide spoof iff s >= t."""
    s, y = np.asarray(scores, float), np.asarray(labels)
    best = None
    for t in np.concatenate((np.unique(s), [np.inf])):
        fa = np.mean(s[y == 0] >= t)
        miss = np.mean(s[y == 1] < t)
        cand = (abs(fa - miss), (fa + miss) / 2)
        if best is None or cand[0] < best[0]:
            best = cand
    return best[1]


def test_eer_perfect_separation_is_zero():
    assert eer([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])[0] == 0.0


def test_eer_inverted_scores_is_one():
    assert eer([0.8, 0.9, 0.1, 0.2], [0, 0, 1, 1])[0] == 1.0


def test_eer_hand_computed_case():
    # bonafide 0.1,0.2,0.3,0.6 ; spoof 0.4,0.7,0.8,0.9 -> at t in (0.4,0.6]: FA=1/4 (0.6), miss=1/4 (0.4)
    s = [0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    assert eer(s, y)[0] == pytest.approx(0.25)


def test_eer_matches_brute_force_on_continuous_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.normal(loc=y * 1.5, scale=1.0)
    assert eer(s, y)[0] == pytest.approx(_brute_force_eer(s, y), abs=1e-12)


def test_eer_threshold_reproduces_operating_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = rng.normal(loc=y * 2.0, scale=1.0)
    e, thr = eer(s, y)
    r = rates_at_threshold(s, y, thr)
    assert (r["false_alarm"] + r["miss"]) / 2 == pytest.approx(e)


def test_eer_identical_distributions_near_half():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1], 5000)
    s = rng.normal(size=y.size)
    assert eer(s, y)[0] == pytest.approx(0.5, abs=0.02)


def test_eer_invariant_to_monotone_transform():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    s = rng.normal(loc=y, scale=1.0)
    assert eer(s, y)[0] == pytest.approx(eer(3 * s + 7, y)[0])
    assert eer(s, y)[0] == pytest.approx(eer(1 / (1 + np.exp(-s)), y)[0])


def test_det_curve_endpoints():
    fa, miss, _ = det_curve([0.1, 0.9, 0.5], [0, 1, 1])
    assert {fa[0], miss[0]} == {0.0, 1.0} and {fa[-1], miss[-1]} == {0.0, 1.0}


def test_roc_auc_matches_sklearn_with_ties():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(loc=y, scale=1.0), 1)  # many ties
    assert roc_auc(s, y) == pytest.approx(sklearn.roc_auc_score(y, s))


def test_rates_at_threshold_counts():
    r = rates_at_threshold([0.1, 0.6, 0.4, 0.9], [0, 0, 1, 1], 0.5)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (1, 1, 1, 1)
    assert r["false_alarm"] == 0.5 and r["miss"] == 0.5 and r["hter"] == 0.5 and r["f1"] == 0.5


@pytest.mark.parametrize("scheme", ["equal_mass", "equal_width"])
def test_ece_constant_wrong_prediction(scheme):
    assert ece(np.full(100, 0.9), np.zeros(100, int), 15, scheme) == pytest.approx(0.9)


@pytest.mark.parametrize("scheme", ["equal_mass", "equal_width"])
def test_ece_small_for_calibrated_probs(scheme):
    rng = np.random.default_rng(5)
    p = rng.uniform(size=50000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert ece(p, y, 15, scheme) < 0.02


def test_nll_and_brier_values():
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert nll([0.5, 0.5], [1, 0]) == pytest.approx(np.log(2))


@pytest.mark.parametrize(
    "scores,labels",
    [([0.1, 0.2], [0, 0]), ([0.1, np.nan], [0, 1]), ([0.1, 0.2], [0, 2]), ([0.1], [0, 1]), ([], [])],
)
def test_input_validation(scores, labels):
    with pytest.raises(ValueError):
        eer(scores, labels)


def test_probability_range_validation():
    with pytest.raises(ValueError):
        ece([1.2, 0.1], [1, 0])


def _bootstrap_frame():
    rng = np.random.default_rng(6)
    rows = []
    for g in range(30):
        for _ in range(10):
            y = int(rng.integers(0, 2))
            rows.append({"speaker_group": f"g{g}", "label": y, "score": rng.normal(loc=y * 1.5)})
    return pd.DataFrame(rows)


def test_group_bootstrap_deterministic_and_brackets_point():
    df = _bootstrap_frame()
    fn = lambda d: eer(d["score"].to_numpy(), d["label"].to_numpy())[0]  # noqa: E731
    a = group_bootstrap(df, fn, n=200, seed=0)
    b = group_bootstrap(df, fn, n=200, seed=0)
    assert a == b
    assert a["lo"] <= a["point"] <= a["hi"]
    assert a["n_groups"] == 30 and a["n_used"] + a["n_skipped"] == 200


def test_group_bootstrap_skips_undefined_resamples():
    df = pd.DataFrame({"speaker_group": ["a", "b"], "label": [0, 1], "score": [0.0, 1.0]})
    fn = lambda d: eer(d["score"].to_numpy(), d["label"].to_numpy())[0]  # noqa: E731
    out = group_bootstrap(df, fn, n=100, seed=0)
    assert out["n_skipped"] > 0  # resamples drawing the same group twice have one class only


def test_group_bootstrap_needs_two_groups():
    df = pd.DataFrame({"speaker_group": ["a", "a"], "label": [0, 1], "score": [0.0, 1.0]})
    with pytest.raises(ValueError):
        group_bootstrap(df, lambda d: 0.0, n=10)
