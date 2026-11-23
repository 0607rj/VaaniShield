"""Build a Hindi real-vs-AI set: FLEURS Hindi (real) vs Meta MMS-TTS Hindi (fake), SAME sentences.

Real : google/fleurs hi_in test split (CC-BY 4.0), 418 recordings of read Hindi speech.
Fake : facebook/mms-tts-hin (VITS, CC-BY-NC 4.0) speaking the same transcriptions; 2 variants per sentence
       (different random seeds / speaking rates) so the fake class is not one fixed rendering.
Split: by sentence id (60/20/20, seeded), so a sentence never appears in two splits.

Output: data/raw/fleurs_hi/{real,fake}/*.wav and data/manifests/hindi_fleurs_mms.csv
Usage : python scripts/build_hindi_set.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.data.audio import load_audio  # noqa: E402

PARQUET = ("google/fleurs", "parquet-data/hi_in/test-00000-of-00001.parquet")
OUT = ROOT / "data/raw/fleurs_hi"
MANIFEST = ROOT / "data/manifests/hindi_fleurs_mms.csv"
SEED = 2026


def main() -> int:
    from huggingface_hub import hf_hub_download

    df = pd.read_parquet(hf_hub_download(*PARQUET, repo_type="dataset"))
    (OUT / "real").mkdir(parents=True, exist_ok=True)
    (OUT / "fake").mkdir(parents=True, exist_ok=True)

    sentences = sorted(df["id"].unique())
    rng = np.random.default_rng(SEED)
    order = rng.permutation(sentences)
    n = len(order)
    split_of = {s: ("train" if i < 0.6 * n else "dev" if i < 0.8 * n else "test") for i, s in enumerate(order)}

    rows = []
    for i, r in df.iterrows():
        path = OUT / "real" / f"fleurs_hi_{r['id']}_{i:03d}.wav"
        if not path.exists():
            w, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            sf.write(path, w, sr)
        rows.append({"clip_id": path.stem, "path": str(path.relative_to(ROOT / "data")).replace("\\", "/"),
                     "label": "bonafide", "source_dataset": "fleurs_hi", "generator": "none", "language": "hi",
                     "sentence_id": int(r["id"]), "gender": r["gender"], "split": split_of[r["id"]],
                     "licence": "CC-BY-4.0"})
    print(f"real: {len(rows)} clips", flush=True)

    from transformers import AutoTokenizer, VitsModel

    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
    tts = VitsModel.from_pretrained("facebook/mms-tts-hin").eval()
    texts = df.drop_duplicates("id").set_index("id")["raw_transcription"]
    made = 0
    for sid, text in texts.items():
        for v, (rate, noise) in enumerate(((1.0, 0.667), (0.9, 0.8))):
            path = OUT / "fake" / f"mmstts_hi_{sid}_v{v}.wav"
            if not path.exists():
                torch.manual_seed(SEED + int(sid) * 10 + v)
                tts.speaking_rate, tts.noise_scale = rate, noise
                with torch.no_grad():
                    wav = tts(**tok(text, return_tensors="pt")).waveform[0].numpy()
                sf.write(path, wav, tts.config.sampling_rate)
                made += 1
            rows.append({"clip_id": path.stem, "path": str(path.relative_to(ROOT / "data")).replace("\\", "/"),
                         "label": "spoof", "source_dataset": "mms_tts_hi", "generator": "mms_tts_hin", "language": "hi",
                         "sentence_id": int(sid), "gender": "", "split": split_of[sid], "licence": "CC-BY-NC-4.0"})
        if made and made % 50 == 0:
            print(f"  generated {made} fake clips", flush=True)
    m = pd.DataFrame(rows)
    m["duration_s"] = [round(load_audio(ROOT / "data" / p).size / 16000, 2) for p in m.path]
    m.to_csv(MANIFEST, index=False)
    print(m.groupby(["split", "label"]).size().to_string())
    print(f"saved {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
