"""Robustness report: slices, metrics table, plots and banner are produced from a score table (no audio, no models)."""

import json

import numpy as np
import pandas as pd

from ml.evaluation.robustness import BANNER, RobustCfg, report, slices


def fake_run(seed=0):
    rng = np.random.default_rng(seed)
    meta, scores = [], []
    for lang in ("en", "hi", "hi-en"):
        for i in range(6):
            lab = i % 2
            cid = f"{lang}_{i}"
            for tr in ("clean", "sim:opus_voip_8k", "sim:aac_24k"):
                vid = cid if tr == "clean" else f"{cid}__{tr[4:]}"
                meta.append({"clip_id": vid, "language": lang, "transform": tr, "speaker_group": f"{lang}_spk{i // 2}",
                             "source_dataset": "s", "parent_clip_id": None if tr == "clean" else cid})
                for m, sep in (("good", 3.0), ("random", 0.0)):
                    logit = sep * (lab - 0.5) + rng.normal(0, 1)
                    scores.append({"model": m, "clip_id": vid, "label": lab, "spoof_logit": logit,
                                   "p_spoof": 1 / (1 + np.exp(-logit)), "threshold_p_spoof": 0.5})
    return pd.DataFrame(scores), pd.DataFrame(meta)


def test_slices_cover_conditions_and_languages():
    _, meta = fake_run()
    s = slices(meta)
    assert s["clean (all languages)"].sum() == 18 and s["codec (all sim, pooled)"].sum() == 36
    assert s["hi-en codec"].sum() == 12 and s["sim:aac_24k"].sum() == 18


def test_report_writes_metrics_plots_and_banner(tmp_path):
    scores, meta = fake_run()
    cfg = RobustCfg(run_id="t", n_per_class=3, pools=[], subset_manifest="x", codec_config="y",
                    models=[], bootstrap_n=20, output_dir=str(tmp_path))
    res = report(cfg, scores, meta, tmp_path)
    for f in ("metrics.csv", "metrics.json", "eer_heatmap.png", "eer_by_codec.png", "roc_by_language.png", "codec_score_shift.png"):
        assert (tmp_path / f).stat().st_size > 0, f
    assert json.loads((tmp_path / "metrics.json").read_text())["label"] == BANNER
    good = res[(res.model == "good") & (res.slice == "clean (all languages)")].iloc[0]
    rand = res[(res.model == "random") & (res.slice == "clean (all languages)")].iloc[0]
    assert good.eer < rand.eer and good.auc > 0.8  # metrics rank a separating model above a random one
    assert good.eer_ci95_lo <= good.eer <= good.eer_ci95_hi
