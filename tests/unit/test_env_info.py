import json
from pathlib import Path

import importlib.util

_spec = importlib.util.spec_from_file_location("env_info", Path(__file__).resolve().parents[2] / "scripts" / "env_info.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

REPO = Path(__file__).resolve().parents[2]

FFMPEG_ENCODERS_SAMPLE = """Encoders:
 V..... = Video
 A..... = Audio
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
 A....D libmp3lame           libmp3lame MP3 (MPEG audio layer 3) (codec mp3)
 A....D libopencore_amrnb    OpenCORE AMR-NB (Adaptive Multi-Rate NarrowBand) (codec amr_nb)
 A....D libopus              libopus Opus (codec opus)
"""


def test_parse_ffmpeg_version():
    line = "ffmpeg version 7.1-full_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg developers"
    assert E.parse_ffmpeg_version(line) == "7.1-full_build-www.gyan.dev"
    assert E.parse_ffmpeg_version("garbage") is None


def test_parse_ffmpeg_encoders_skips_legend():
    names = E.parse_ffmpeg_encoders(FFMPEG_ENCODERS_SAMPLE)
    assert names == {"libx264", "aac", "libmp3lame", "libopencore_amrnb", "libopus"}
    assert "=" not in names


def test_check_encoders():
    ok = E.check_encoders({"libopus", "libmp3lame"})
    assert ok["ok"] and ok["optional"]["libopencore_amrnb"] is False
    bad = E.check_encoders({"libopus"})
    assert not bad["ok"] and bad["missing_required"] == ["libmp3lame"]


def test_bytes_to_gib():
    assert E.bytes_to_gib(16 * 1024**3) == 16.0
    assert E.bytes_to_gib(None) is None


def test_package_versions_missing_is_none():
    out = E.package_versions(["numpy", "surely-not-a-real-package-xyz"])
    assert out["numpy"] and out["surely-not-a-real-package-xyz"] is None


def _env(**over):
    env = {
        "python": {"version": "3.12.13", "in_virtualenv": True},
        "torch": {"installed": True, "cuda_available": True},
        "ffmpeg": {
            "found": True,
            "encoders": E.check_encoders({"libopus", "libmp3lame", "libopencore_amrnb"}),
        },
        "packages": {"audio": {"numpy": "2.0"}, "watch_should_be_absent": {"torchaudio": None}},
    }
    for k, v in over.items():
        env[k] = v
    return env


def test_summarize_issues_clean_env():
    assert E.summarize_issues(_env()) == []


def test_summarize_issues_flags_each_problem():
    env = _env(
        python={"version": "3.14.7", "in_virtualenv": False},
        torch={"installed": True, "cuda_available": False},
        ffmpeg={"found": False, "encoders": E.check_encoders(set())},
        packages={"audio": {"soundfile": None}, "watch_should_be_absent": {"torchaudio": "2.11.0"}},
    )
    text = " | ".join(E.summarize_issues(env))
    for needle in [
        "3.12",
        "virtual environment",
        "CUDA not available",
        "ffmpeg not found",
        "soundfile",
        "torchaudio",
    ]:
        assert needle in text


def test_collect_environment_never_raises_and_is_json():
    env = E.collect_environment()
    json.dumps(env)  # must be serialisable
    assert {"python", "os", "cpu", "ram", "torch", "system_gpus", "ffmpeg", "packages"} <= env.keys()


def test_environment_check_script_runs(tmp_path):
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "environment_check.py"), "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "Python" in r.stdout and "CUDA available" in r.stdout
    saved = list(tmp_path.glob("env_*.json"))
    assert len(saved) == 1 and "environment" in json.loads(saved[0].read_text())
