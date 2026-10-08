"""Evaluation metrics (docs/architecture.md M15, docs/experiment_protocol.md §5).

Conventions (DEC-1):
    labels: 0 = bonafide, 1 = spoof
    score : spoof logit, higher = more likely spoof; decide "spoof" iff score >= threshold
    false alarm = bonafide decided spoof; miss = spoof decided bonafide

The EER follows the det-curve algorithm of the official ASVspoof 2019 evaluation script (bona fide =
"target", higher = target). We convert by negating scores. Parity against the official script on real
score files is test P3.6 (Phase 3); until then the implementation is only verified on hand-computed cases.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import rankdata

_EPS = 1e-12


def _as_scores_labels(scores: Any, labels: Any, need_both: bool = True) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if s.ndim != 1 or y.ndim != 1 or s.shape != y.shape:
        raise ValueError(f"scores and labels must be 1-D and equal length, got {s.shape} and {y.shape}")
    if s.size == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain NaN or inf")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (bonafide) or 1 (spoof)")
    y = y.astype(np.int64)
    if need_both and (y.min() == y.max()):
        raise ValueError("both classes (bonafide and spoof) must be present")
    return s, y


def det_curve(scores: Any, labels: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (false_alarm, miss, thresholds) over all candidate thresholds.

    Mirrors ASVspoof's ``compute_det_curve`` with target = bonafide and target score = -spoof score.
    ``thresholds`` are in spoof-score space: decide spoof iff score >= threshold.
    """
    s, y = _as_scores_labels(scores, labels)
    target = -s[y == 0]  # bonafide, official convention: higher = bonafide
    nontarget = -s[y == 1]  # spoof
    n = target.size + nontarget.size
    all_scores = np.concatenate((target, nontarget))
    is_target = np.concatenate((np.ones(target.size), np.zeros(nontarget.size)))
    order = np.argsort(all_scores, kind="mergesort")
    is_target = is_target[order]
    tar_sums = np.cumsum(is_target)
    nontar_sums = nontarget.size - (np.arange(1, n + 1) - tar_sums)
    frr = np.concatenate(([0.0], tar_sums / target.size))  # bonafide rejected  = false alarm
    far = np.concatenate(([1.0], nontar_sums / nontarget.size))  # spoof accepted = miss
    thr_official = np.concatenate(([all_scores[order[0]] - 0.001], all_scores[order]))
    # Official: accept as bonafide iff target_score > thr  <=>  spoof_score < -thr  <=>  spoof iff score >= -thr
    return frr, far, -thr_official


def eer(scores: Any, labels: Any) -> tuple[float, float]:
    """Equal error rate (fraction, not %) and the spoof-score threshold where it occurs."""
    fa, miss, thr = det_curve(scores, labels)
    i = int(np.argmin(np.abs(fa - miss)))
    return float((fa[i] + miss[i]) / 2.0), float(thr[i])


def roc_auc(scores: Any, labels: Any) -> float:
    """Probability that a random spoof scores higher than a random bonafide (ties count 1/2)."""
    s, y = _as_scores_labels(scores, labels)
    ranks = rankdata(s)  # average ranks for ties
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def rates_at_threshold(scores: Any, labels: Any, threshold: float) -> dict[str, float | int]:
    """Operating-point metrics at a fixed (dev-chosen) threshold. Positive class = spoof."""
    s, y = _as_scores_labels(scores, labels)
    pred = (s >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fa = fp / (fp + tn)
    miss = fn / (fn + tp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "false_alarm": fa,
        "miss": miss,
        "hter": (fa + miss) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _check_probs(probs: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    p, y = _as_scores_labels(probs, labels, need_both=False)
    if (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return p, y


def ece(
    probs: Any,
    labels: Any,
    n_bins: int = 15,
    scheme: str = "equal_mass",
) -> float:
    """Expected calibration error of p_spoof against the spoof label (positive-class reliability).

    ECE = sum_b (n_b / N) * |mean(label in b) - mean(p in b)|.
    ``equal_mass``: bins of (near-)equal count over sorted probabilities (primary, protocol §5).
    ``equal_width``: bins [0,1/n), ..., [(n-1)/n, 1].
    """
    p, y = _check_probs(probs, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    n = p.size
    if scheme == "equal_mass":
        order = np.argsort(p, kind="mergesort")
        bins = [b for b in np.array_split(order, min(n_bins, n)) if b.size]
    elif scheme == "equal_width":
        idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
        bins = [np.flatnonzero(idx == b) for b in range(n_bins)]
        bins = [b for b in bins if b.size]
    else:
        raise ValueError("scheme must be 'equal_mass' or 'equal_width'")
    return float(sum(b.size / n * abs(y[b].mean() - p[b].mean()) for b in bins))


def nll(probs: Any, labels: Any) -> float:
    p, y = _check_probs(probs, labels)
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(probs: Any, labels: Any) -> float:
    p, y = _check_probs(probs, labels)
    return float(np.mean((p - y) ** 2))


def group_bootstrap(
    df: Any,
    metric_fn: Callable[[Any], float],
    group_col: str = "speaker_group",
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Percentile CI by resampling whole groups (speakers) with replacement (protocol §5).

    For paired system comparisons pass a frame holding both systems' scores per clip and a
    ``metric_fn`` returning the difference, so both systems see the same resampled groups.
    Resamples where the metric is undefined (e.g. one class only) are skipped and counted.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    point = float(metric_fn(df))
    group_rows = list(df.groupby(group_col, sort=True).indices.values())
    if len(group_rows) < 2:
        raise ValueError("need at least 2 groups to bootstrap")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    skipped = 0
    for _ in range(n):
        pick = rng.integers(0, len(group_rows), size=len(group_rows))
        idx = np.concatenate([group_rows[i] for i in pick])
        try:
            values.append(float(metric_fn(df.iloc[idx])))
        except ValueError:
            skipped += 1
    if not values:
        raise ValueError("metric undefined on every bootstrap resample")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_used": len(values),
        "n_skipped": skipped,
        "n_groups": len(group_rows),
        "alpha": alpha,
        "seed": seed,
    }
