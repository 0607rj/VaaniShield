# Hindi dataset repair report: v2 pilot

**Date:** 2026-10-08
**Result: the gate verdict is FAIL. The repaired pilot is NOT ready for training or reported results.**
- 9 of 10 gate checks pass: all leakage checks, format identity, preprocessing identity and class balance.
- The waveform/spectrum shortcut probes still fail the proposed thresholds.
- The remaining causes are identified (§5), and a next step is proposed for team decision (§8).

**Reproduce (laptop, ~25 min total, mostly network):**
```bash
python scripts/acquire_kathbath.py --transcripts --accept-terms
python scripts/acquire_kathbath.py --fetch-ids data/manifests/hindi_v2_reference_ids.txt --tars clean/train_audio.tar --allow-train --extra-per-speaker 10 --accept-terms
python -m ml.data.build_hindi_v2 --config ml/configs/hindi_v2_pilot.yaml
python -m ml.data.hindi_gate --manifest data/manifests/hindi_v2_pilot.csv                       # after
python -m ml.data.hindi_gate --manifest data/manifests/vaanishield_v1.csv --language hi          # before
python scripts/hindi_v2_diagnostics.py                                                            # §4-§5 numbers
```
**Old data untouched:** `vaanishield_v1.csv`, `hinglish_pilot_v1.csv` and every file they reference verify unchanged
(`scripts/verify_data.py`: 2,134/2,134). The v2 audio is new: `data/processed/hindi_v2/pilot/` (111 MB).

---

## 1. The old dataset problem (research/hindi_data_leakage_audit.md)
- **Two different recording chains:**
  - real = Kathbath *test-known* phone recordings (AAC m4a, 44.1 kHz, raw level);
  - fake = IndicSynth output (PCM WAV, 24 kHz, peak-normalised; FreeVC adds DC offset).
- **Mostly different speakers:** 4 of 19 cloned speakers had any real clip; 0 of the 323 recordings the fakes were
  made from were in the real pool.
- **No sentence check:** IndicSynth rows had no transcript IDs, so sentence leakage could not be checked.
- **Shortcut strength:** a metadata-only classifier had EER 0.000, and a waveform-statistics probe 0.17.

## 2. Repair procedure (what was done)
| Step | What | Evidence |
|---|---|---|
| 1. Find the exact sources | IndicSynth metadata names each fake's **target reference** (voice) and, for FreeVC, **source reference** (words). We listed the 323 distinct recordings (`data/manifests/hindi_v2_reference_ids.txt`). Kathbath transcripts showed all 323 are in Kathbath's **train** split (93,024 Hindi files) | `--transcripts`: 4 files, 18 MB |
| 2. Fetch them exactly | New `--fetch-ids` mode walks the Hindi section of the 85 GB train tar header-by-header with 48 parallel chunk walkers: 96k range requests, 223 MB read, 16 min. Verified on the test-known tar: 3,130/3,130 members found, no extras. **323/323 found**, plus 230 other recordings of the same 24 speakers (553 files, 55 MB) | provenance `kathbath_hi.json` (mode `fetch_ids`, tar ETag recorded); full listing `raw/kathbath_hi/_listings/` |
| 3. Pair and balance | Real pool = the fakes' own reference recordings first, then same-speaker recordings. Per speaker group, real count = spoof count | 314 references + 6 same-speaker; 313/320 fakes have their target reference in the manifest; 135/160 FreeVC fakes have their source reference |
| 4. Speaker/source-disjoint splits | Speaker graph (target ↔ VC source) → 11 groups. Groups containing the held-out family (FreeVC) → test only. Other groups by clip share 60/20/20 | `docs/data_flow.md` rules 3–4 |
| 5. Sentence-disjoint | Transcript ID for every row: real = Kathbath transcript; XTTS = TTS prompt; FreeVC = its source recording's transcript. Test sentences are removed from train/dev | 0 rows needed dropping |
| 6. Identical preprocessing (v2) | **Both classes:** decode → mono → soxr HQ 16 kHz → **40 Hz high-pass (DC removal)** → trim silence (−40 dB) → **loudness normalisation to −26 dBFS active RMS**, peak limit 0.95 → PCM_16 WAV. One recorded parameter set (`PREPROC_V2`, hash `ea2fe7a7998a`) | `ml/data/audio.py` |
| 7. Matched codec history | Fakes pass through AAC-LC 128 kb/s @ 44.1 kHz (Kathbath's measured capture codec) before v2, so both classes share the same last lossy stage | metadata `chain` per file |
| 8. Provenance | Per file: original path + sha256, original container/codec/rate/bitrate, Kathbath split, reference IDs, the full chain, gain, peak-limit flag, post-processing level/DC/peak | `data/manifests/hindi_v2_pilot_metadata.jsonl` (+ header with config, tool versions, group table) |

Generator metadata is preserved (generator, family, target and source speaker, reference IDs).
The licence column keeps each source's status (Kathbath: CONFLICT, as in the registry).

## 3. New dataset composition (`data/manifests/hindi_v2_pilot.csv`, 640 clips)
| Split | Real | Spoof | Speaker groups | Spoof families |
|---|---|---|---|---|
| train | 100 | 100 | 4 | xtts 100 |
| dev | 25 | 25 | **1** | xtts 25 |
| test | 195 | 195 | 6 | freevc 160 (**held out**) + xtts 35 (seen) |

- **Speakers:** 23 real; 19 cloned targets, all 19 with real audio (v1: 4). 11 groups, all with both classes
  (v1: 4 of 27).
- **Stored format:** identical for both classes (WAV, PCM_16, 16 kHz, mono).
- **Duration (10th / 50th / 90th percentile):** real 3.82 / 5.31 / 7.90 s; spoof 3.60 / 5.06 / 8.22 s.

## 4. Before / after shortcut metrics
Gate report: `runs/gate/{vaanishield_v1_hi,hindi_v2_pilot_hi}/gate_report.md`. EER: 0.5 = chance, 0 = perfect
shortcut. "Worst" = the lowest EER over logistic regression and random forest, speaker-grouped CV and train→test.

| Gate check | v1 Hindi (before) | v2 pilot (after) |
|---|---|---|
| G1 metadata-only classifier, worst EER (proposed min 0.40) | **0.000 FAIL** | **0.467 PASS** |
| G2 identical stored format | FAIL (AAC/MOV/44.1 kHz vs PCM/WAV/24 kHz) | PASS (PCM_16/WAV/16 kHz both) |
| G3 speaker leakage | PASS | PASS (0 overlaps) |
| G4 source leakage (fake ↔ its reference recordings, VC sources) | PASS (but references absent) | PASS (0 problems) |
| G5 duplicate leakage | PASS | PASS |
| G6 generator protocol (held-out family test-only; dev ⊆ train families) | FAIL (FreeVC in train/dev) | PASS |
| G7 sentence leakage (+ every row has a transcript ID) | FAIL (400 rows without ID) | PASS (0 / 0) |
| G8 identical preprocessing (same recorded chain; DC/loudness/peak verified on stored audio) | FAIL (no record) | PASS (0 rows failing) |
| G9 class balance, spoof/real per split (proposed 0.5–2) | FAIL (2.14 / 0.26 / 1.80) | PASS (1.0 / 1.0 / 1.0) |
| G10 waveform probe worst EER (proposed ≥ 0.35) | 0.166 | **0.241 FAIL** |
| G10 40-band spectrum probe worst EER (proposed ≥ 0.40) | 0.210 | **0.256 FAIL** |
| G10 strongest single feature, sep AUC (proposed ≤ 0.80) | 0.88 (peak level) | 0.71 (silent-frame fraction) |
| **Verdict** | **FAIL** (7 checks) | **FAIL** (1 check) |

## 5. Remaining possible confounds (`runs/audit/hindi_v2/diagnostics.json`)
Feature-family ablation on the v2 pilot (worst EER; higher = weaker shortcut):

| Feature set | v2 as built | −30 dBFS target | + −60 dBFS noise both | **pauses removed both (VAD)** | VAD + 6 kHz low-pass both |
|---|---|---|---|---|---|
| all non-semantic | 0.241 | 0.231 | 0.253 | **0.341** | 0.372 |
| silence only | 0.282 | 0.287 | 0.306 | **0.441** | 0.466 |
| level only | 0.338 | 0.313 | 0.344 | 0.395 | 0.426 |
| without silence and level | 0.372 | 0.372 | 0.374 | 0.364 | 0.379 |
| spectral band ratios only | 0.364 | 0.364 | 0.359 | 0.364 | 0.385 |
| 40-band spectrum (LTAS) | 0.256 | 0.244 | 0.287 | 0.291 | 0.281 |

The last two candidate columns were tested **in memory only**; nothing was adopted. What remains:
1. **Pause structure.** Fakes have more and cleaner pauses: silent-frame fraction 0.14 vs 0.05; noise floor −43 vs
   −35 dB below the loudest frame.
   - It is the strongest remaining cue. A shared noise floor does not remove it; removing internal pauses from both
     classes does (silence-only EER 0.28 → 0.44).
2. **Level via peak limiting.** 9.4 % of real clips (30/320; phone transients) hit the 0.95 peak limit and end up
   below −26 dBFS; no fake does. A −30 dBFS target halves this (4.7 %) but does not change the probe.
3. **Spectral envelope.**
   - Fakes carry less 4–8 kHz energy (6–8 kHz band share 0.0003 vs 0.0011).
   - The 40-band probe stays at 0.24–0.29 even after removing pauses and low-passing both classes at 6 kHz, so the
     difference is not only above 6 kHz.
   - Within the same speakers, after matched codec and preprocessing, this is **plausibly a genuine synthesis
     artefact** (vocoder / voice-cloning spectral envelope). That is what a detector *should* learn. It may also
     partly reflect how the generators were trained (e.g. on band-limited or resampled audio), which we cannot
     verify: Needs verification from the IndicSynth generation pipeline.
   - Consequence: the spectrum probe is not a clean "non-semantic" probe. It behaves like a weak detector.
4. **Source name.** Real = Kathbath, fake = IndicSynth by definition. The model never sees the name, and the stored
   formats are now identical, but `check_manifest` still flags it. It cannot be removed without new real or fake
   sources.
5. **Codec history is matched, not identical.**
   - Real: one AAC pass at capture.
   - Fake: generated from AAC-decoded references, then one matched AAC pass.
   - The audit found AAC at 128 kb/s nearly invisible after 16 kHz decoding (EER 0.43–0.46).

## 6. Limitations
- **Few speaker groups (11).** FreeVC links merge speakers into large groups, and all of them go to test. Train has
  4 groups and **dev only 1**.
  - Dev-based model selection on one group is fragile.
  - Probe cross-validation over 11 groups has wide uncertainty; differences below ~0.05 EER are noise.
- **Pilot scale:** 640 clips from the 320 fakes already downloaded. A full build needs more IndicSynth Hindi shards
  (whose speakers also come from Kathbath train) and the same reference fetch.
- **Test-heavy split:** test holds 390 of 640 clips (FreeVC holdout groups plus one XTTS group). This follows the
  protocol, but leaves little training data at pilot scale.
- **Generator references are seen by the generator.** Real clips are the very recordings the generators were
  conditioned on. That is the right pairing for "same speaker, same session", but such pairs may be easier or harder
  than independent real speech. Results on independent real speech still need the old test-known pool (kept,
  unpaired).
- **Probe coverage:** trivial probes are a floor test. Passing them does not prove the absence of confounds a deep
  model could exploit.
- **Licence:** Kathbath remains "Needs verification from current dataset terms." (research/handoff_licence_audit.md).
  The new train-split recordings and `processed/hindi_v2/` must **not** be copied to other team members until that is
  resolved; they can be re-fetched with the commands above.

## 7. Pass/fail decision
| Question | Answer |
|---|---|
| Is the repaired pilot READY? | **No. Gate verdict FAIL** (G10). |
| Did the repair fix what it set out to fix? | Yes: format, sample rate, codec, DC, loudness target, speaker pairing, source pairing, sentence disjointness, generator protocol and balance all pass. |
| Is the remaining failure a data defect or a gate-definition question? | Both, and they must be separated. Pause structure (§5.1) and peak limiting (§5.2) are dataset cues. The spectral envelope (§5.3) is plausibly a real synthesis artefact that the current G10 spectrum criterion penalises. |
| Thresholds approved? | **No.** Every numeric threshold in `ml/configs/hindi_gate.yaml` is a PROPOSAL (`approved: false`). The project documents define none. Even a full pass would only reach PASS_PENDING_APPROVAL. |

## 8. Proposed next step (requires team decision; nothing applied)
- **D-H4: pause handling.** Adopt preprocessing **v2.1** = v2 + removal of internal pauses (keep 25 ms frames within
  40 dB of the loudest, 10 ms cross-fades) for **all** languages and classes, at training **and** inference.
  - Trade-off: it changes speech rhythm, and the deployed system must apply it too.
  - Alternative: keep v2, and report pause-based probe floors next to every Hindi result.
- **D-H5: peak limiting.** Keep −26 dBFS and accept the 9 % peak-limited real clips (documented), or switch to a
  limiter that does not change active RMS. A different target alone did not help.
- **D-H6: re-scope G10.** Split it into:
  - (a) a **non-speech confound probe** (silence, level and duration features) that must be near chance — proposed
    EER ≥ 0.35;
  - (b) a **spectrum probe that is reported, not gated**, since it cannot distinguish confounds from genuine
    artefacts.

  Under that definition, the in-memory VAD variant gives (a) = 0.341: just below 0.35, so it would still need review.
- **D-H3 (from the audit): approve or adjust all proposed thresholds** before any Hindi detector is trained, and
  never after seeing detector results.
- **Then:** build v2.1 as a new version (`processed/hindi_v2_1/`), re-run the gate, and only then plan the full
  Hindi build on the GPU machine.
