import pandas as pd
import pytest

from ml.evaluation import evaluate as ev


def test_evaluate_frame():
    df = pd.DataFrame(
        {
            "clip_id": ["a", "b", "c", "d"],
            "label": ["bonafide", "bonafide", "spoof", "spoof"],
            "spoof_logit": [-2.0, -1.0, 1.0, 2.0],
            "p_spoof": [0.1, 0.2, 0.8, 0.9],
        }
    )
    out = ev.evaluate_frame(df, threshold=0.0)
    assert out["eer"] == 0.0 and out["auc"] == 1.0
    assert out["at_threshold"]["false_alarm"] == 0.0
    assert out["calibration"]["brier"] == pytest.approx(0.025)


def test_evaluate_frame_rejects_duplicates_and_bad_labels():
    with pytest.raises(ValueError):
        ev.evaluate_frame(pd.DataFrame({"clip_id": ["a", "a"], "label": [0, 1], "spoof_logit": [0.0, 1.0]}))
    with pytest.raises(ValueError):
        ev.evaluate_frame(
            pd.DataFrame({"clip_id": ["a", "b"], "label": ["real", "fake"], "spoof_logit": [0.0, 1.0]})
        )
