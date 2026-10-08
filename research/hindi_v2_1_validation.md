# Hindi v2.1 validation report

**Date:** 2026-10-08
**Dataset:** `data/manifests/hindi_v2_1_pilot.csv`; audio in `data/processed/hindi_v2_1/pilot/` (220 MB, float32 WAV).
Pause-removed diagnostic copy: `hindi_v2_1_pilot_pauses_removed.csv` (diagnostic only, never primary data).
**Recommendation: READY_WITH_LIMITATIONS**, for pilot / development training only (§9). Not for research claims.

All numeric acceptance limits are **provisional engineering acceptance criteria** (team decision D-H3). The project
documents define none of them, and none is a scientifically established threshold. "EER" below is the most
pessimistic value over logistic regression and random forest, speaker-grouped cross-validation and train→test
(0.5 = chance).

**Reproduce (all CPU; v1, v2 and v2.1 inputs untouched):**
```bash
python -m ml.data.build_hindi_v2 --config ml/configs/hindi_v2_1_pilot.yaml                  # ~4 min
python -m ml.data.hindi_gate --manifest data/manifests/hindi_v2_1_pilot.csv                  # acceptance + diagnostics
python -m ml.data.hindi_gate --manifest data/manifests/hindi_v2_pilot.csv                    # same gate on v2
python -m ml.data.hindi_gate --manifest data/manifests/vaanishield_v1.csv --language hi      # same gate on v1
python scripts/hindi_v2_1_analysis.py                                                        # peaks, level, floor, spectrum
python scripts/pause_controlled_diagnostic.py                                                # D-H4 experiment (~13 min)
```
Outputs:
- `runs/gate/{hindi_v2_1_pilot_hi,hindi_v2_pilot_hi,vaanishield_v1_hi}/gate_report.{md,json}`
- `runs/audit/hindi_v2_1/analysis.json`
- `runs/diag/pause_controlled/summary.{md,json}`

---

## 1. Changes from v2 to v2.1
| Item | v2 | v2.1 | Decision |
|---|---|---|---|
| Peak limiter | hard limit at 0.95 → touched 9.4 % of real clips, 0 % of fakes | **removed**; one linear loudness gain per clip, same rule for both classes | D-H5 |
| Storage | PCM_16 WAV | **32-bit float WAV**: values above 1.0 are stored exactly, so no sample is ever clipped | D-H5 (class-neutral alternative to a limiter) |
| Pauses | kept | **kept** in the primary data; a separate pause-removed copy is built only for the diagnostic | D-H4 |
| Gate | 10 checks, spectrum probe could fail the dataset | **A. acceptance gate** (13 checks) + **B. diagnostic probes** (5, never pass/fail) | D-H6 |
| Thresholds | "PROPOSED" | "PROVISIONAL ENGINEERING ACCEPTANCE CRITERIA" | D-H3 |
| Everything else | — | unchanged (verified): same 640 clip IDs, pairing, speaker groups, splits, transcripts, matched AAC history for fakes, 40 Hz high-pass, silence trim, −26 dBFS target | — |

- Preprocessing hash: v2 `ea2fe7a7998a`, v2.1 `587d94c32a05` (parameters in `ml/data/audio.py` `PREPROC_V2_1`).
- v2 still rebuilds with its original hash.
- **Why float storage is class-neutral:** both classes get the same linear gain rule and no non-linear step. Peaks
  above full scale are kept exactly, rather than limited (v2) or clipped (PCM).

## 2. Composition (identical to v2)
| Split | Real | Spoof | Speaker groups | Spoof families |
|---|---|---|---|---|
| train | 100 | 100 | 4 | xtts |
| dev | 25 | 25 | 1 | xtts |
| test | 195 | 195 | 6 | freevc (held out) + xtts |

- **Real clips:** 314 of them are the exact Kathbath recordings the fakes were generated from; 6 more are
  recordings of the same speakers.
- **Pairing:** 313/320 fakes have their target-voice reference in the dataset; 135/160 FreeVC fakes have their
  source-speech reference.
- **Splits:** speaker-group disjoint and sentence-disjoint; FreeVC is in test only.

## 3. Acceptance gate results (same gate applied to all three versions)
| Check (A) | v1 Hindi | v2 | **v2.1** | Key numbers for v2.1 |
|---|---|---|---|---|
| A1 file-metadata classifier (≥ 0.40) | FAIL (0.000) | PASS | **PASS** | EER 0.463 |
| A2 format / sample rate / codec / bitrate identical | FAIL | PASS | **PASS** | WAV, pcm_f32le, 16 kHz, mono, 512 kb/s, both classes |
| A3 duration (≥ 0.40) | FAIL | PASS | **PASS** | EER 0.466; 10/50/90th pct real 3.82/5.31/7.90 s, spoof 3.60/5.06/8.22 s |
| A4 edge + digital silence (≥ 0.40) | FAIL | PASS | **PASS** | EER 0.531; leading/trailing silence and zero-sample fraction medians 0 in both classes |
| A5 overall level (≥ 0.40; all rows within 0.5 dB of target) | FAIL | PASS | **PASS** | EER 0.491; 0 rows off target |
| A6 peak/clipping processing class-neutral | FAIL | **FAIL** (limiter 9.4 % vs 0 %) | **PASS** | limiter 0 % / 0 %; no clipped samples |
| A7 speaker leakage | PASS | PASS | **PASS** | 0 overlaps; both classes in every split |
| A8 source leakage (fake ↔ its reference recordings) | PASS | PASS | **PASS** | 0 problems |
| A9 duplicate leakage | PASS | PASS | **PASS** | 0 cross-split duplicates |
| A10 sentence leakage | FAIL (400 rows without ID) | PASS | **PASS** | 0 test sentences in train/dev; 0 missing IDs |
| A11 generator protocol | FAIL | PASS | **PASS** | train/dev xtts; test xtts + freevc |
| A12 identical preprocessing | FAIL (no record) | PASS | **PASS** | one hash for both classes; DC ≤ 0.001 in all rows |
| A13 class balance (0.5–2) | FAIL | PASS | **PASS** | 1.0 / 1.0 / 1.0 |
| **Verdict** | ACCEPTANCE_FAIL | ACCEPTANCE_FAIL | **ACCEPTANCE_PASS_PROVISIONAL** | |

`check_manifest` on v2.1: all 6 hard checks PASS. Its "source_dataset separates the classes" warning is inherent
(real = Kathbath, fake = IndicSynth by definition), and the model never sees that name.

## 4. Diagnostic probes (B; reported, not pass/fail)
| Probe | v1 Hindi | v2 | **v2.1** | v2.1, pauses removed (diag copy) | Reading (v2.1) | Cue type |
|---|---|---|---|---|---|---|
| B1 metadata-only | 0.000 | 0.467 | **0.463** | 0.390 ¹ | no usable separation | **nuisance**: eliminated |
| B2 non-speech / nuisance-only | 0.221 | 0.372 | **0.341** | 0.409 | moderate | **nuisance**, but driven by pause/dynamics (§6), not by processing |
| B3 pause-only | 0.388 | 0.428 | **0.419** | 0.384 ² | weak | **mixed / unresolved** |
| B4 waveform statistics | 0.166 | 0.241 | **0.221** | 0.309 | moderate | **mixed** |
| B5 spectral (band ratios + 40-band spectrum) | 0.230 | 0.259 | **0.234** | 0.253 | moderate | **potentially legitimate synthesis cue (not proven)** |

¹ Fakes contain more pauses, so their pause-removed copies are relatively shorter. This is an artefact of the
diagnostic copy only.
² With pauses removed, "pause" features describe only the few remaining short gaps.

**Interpretation of each probe:**
- **B1 metadata:** the repair removed the file-level shortcut (v1: perfect separation → near chance).
- **B2 nuisance-only:**
  - The strongest feature is active RMS **inside the 4.04 s model crop** (sep AUC 0.69). File-level loudness is
    identical (A5, AUC 0.50).
  - The crop difference arises because fake crops contain more low-energy frames (86 % vs 95 % active frames).
    Removing pauses weakens B2 to 0.409.
  - It is therefore a consequence of pause/dynamics structure, not of level processing.
- **B3 pause-only:** pause count, length, fraction and pause noise floor separate only weakly (0.419). Pause
  *structure* itself is a weak cue.
- **B4 waveform statistics:** moderate (0.221). It combines the noise floor of the quietest frames, pause fraction and
  crop level, and weakens to 0.309 once pauses are removed.
- **B5 spectral:** moderate (0.234), and **unchanged by pause removal** (0.253). §8 explains why this is plausibly a
  synthesis property.

## 5. Pause analysis (D-H4: pauses kept; dependence quantified)
**Pause-controlled detector diagnostic**:
- **Setup:** LFCC-CNN (14k parameters), CPU, 200 training clips, 3 seeds; dev-fixed thresholds; same test clips with
  and without pauses.
- **Label: DIAGNOSTIC — NOT FINAL RESULTS.**

| Train | Test | EER all (mean ± sd) | Seen XTTS | Held-out FreeVC |
|---|---|---|---|---|
| pauses kept | pauses kept | 0.265 ± 0.012 | 0.257 ± 0.052 | 0.263 ± 0.011 |
| pauses kept | pauses removed | 0.279 ± 0.013 | 0.237 ± 0.040 | 0.289 ± 0.007 |
| pauses removed | pauses kept | 0.262 ± 0.027 | 0.276 ± 0.016 | 0.262 ± 0.032 |
| pauses removed | pauses removed | 0.286 ± 0.021 | 0.258 ± 0.003 | 0.291 ± 0.029 |

- **Pause dependence** (test without pauses minus test with pauses, model trained with pauses): **+0.014 EER
  overall** (about one seed standard deviation), −0.020 on XTTS, +0.026 on FreeVC.
- A model trained without pauses does as well on normal test audio (0.262 vs 0.265).
- **Reading:** for this small spectral-feature detector, performance depends only slightly on pause structure,
  somewhat more for FreeVC.
- **Limit of this evidence:** a 14k-parameter LFCC model on 200 clips with one dev group. A raw-waveform or SSL model
  may exploit pauses more, so **the same 2×2 diagnostic must be repeated for every model trained on Hindi**. The
  configs and the script exist for that.

**Pause statistics (v2.1):**
- Fake crops have more low-energy frames (active-frame fraction 0.86 vs 0.95).
- The quiet-frame noise floor depends on the generator: real −40.5 dB, XTTS −38.6 dB (*noisier* than real), FreeVC
  −47.4 dB (much cleaner).
- So "fakes have clean pauses" is true for FreeVC only, which points to generator behaviour rather than a uniform
  recording-condition shortcut.

## 6. Peak / clipping analysis (D-H5)
| | Real | Spoof |
|---|---|---|
| v2: peak-limited clips | 30 / 320 (9.4 %) | 0 / 320 |
| v2.1: peak-limited clips | 0 | 0 |
| v2.1: clips with peak ≥ 1.0 (stored exactly as float) | **29 (9.1 %)**, max 3.24 | 0, max 0.50 |
| v2.1 peak, 50/90/99th percentile | 0.38 / 0.88 / 2.90 | 0.30 / 0.40 / 0.48 |
| Crest factor (median) | 17.6 dB | 15.6 dB (sep AUC 0.69) |
| Raw files containing full-scale samples | 16 (capture clipping) | **320** (all IndicSynth output is peak-normalised to 1.0) |

- **Removing the limiter** raised the gain of the 30 formerly limited clips by a median 3.96 dB, back to the common
  −26 dBFS target.
- **The crest-factor difference is a property of the signals, not of our processing:** phone recordings of real
  speech have sharper transients, and the generators' output is peak-normalised upstream. It stays visible as a
  diagnostic, not as an acceptance failure.
- **Downstream requirement (risk):** 29 real clips exceed ±1.0. Any later conversion to integer PCM or any codec
  round-trip (`ml/data/codec.py` converts to 16-bit) would clip **only real clips**, re-creating class-dependent
  distortion. Before codec augmentation or export, apply one fixed attenuation to both classes (e.g. −12 dB, which
  covers the max 3.24 = +10.2 dB), or add an equivalent guard. **Not yet implemented.**

## 7. Before/after statistics (v1 → v2 → v2.1)
| Statistic | v1 Hindi | v2 | v2.1 |
|---|---|---|---|
| Stored format by class | AAC/MOV 44.1 kHz vs PCM/WAV 24 kHz | PCM_16 16 kHz (both) | float32 16 kHz (both) |
| Speakers with both classes / speaker groups with both | 4 / 4 of 27 | 19 / 11 of 11 | 19 / 11 of 11 |
| Clips per split real:spoof | 105:225, 96:25, 39:70 | 100:100, 25:25, 195:195 | same as v2 |
| File-level active RMS | raw (peak-normalised fakes, +5 dB) | −26 dBFS (limited clips lower) | **−26.0 dBFS, every clip** |
| Peak-limited real / fake | n/a | 9.4 % / 0 % | **0 % / 0 %** |
| Metadata probe EER | 0.000 | 0.467 | 0.463 |
| Waveform probe EER | 0.166 | 0.241 | 0.221 |
| Spectral probe EER | 0.230 | 0.259 | 0.234 |

The waveform probe is slightly *lower* in v2.1 than in v2 (0.221 vs 0.241). Removing the limiter restores the real
clips' natural crest factor and slightly increases crop-level differences. This is an honest cost of removing a
class-dependent operation, and a sign that v2's limiter was partly *masking* a signal property.

## 8. Spectrum analysis
- **Class means:** relative to real, fakes have more energy at 0.1–3.3 kHz (+0.7 to +3.2 dB) and less at 4.9–7.5 kHz
  (−1.4 to −4.3 dB).
- **By generator:** both show the high-band deficit. XTTS: −0.2 to −3.3 dB above 4.9 kHz; FreeVC: −1.4 to −5.3 dB.
- **Paired by speaker** (same voice, real vs its clones; 19 speakers): fakes have less 4–8 kHz energy in **15 of 19
  speakers**. Median paired difference at 6.5 / 7.3 kHz: −3.4 / −4.1 dB.
- **Pause-independent:** the spectral probe stays at 0.25 after pause removal.

**Interpretation:**
- The difference is consistent within speakers, present for both generator families and independent of pauses. It
  survives matched codec history and identical preprocessing.
- That makes a **genuine synthesis property** (vocoder high-band deficit / voice-cloning spectral envelope) the most
  plausible explanation, i.e. legitimate evidence a detector should use.
- **Not proven:** the generators may have been trained on band-limited (16 kHz) data, and we have not verified the
  IndicSynth training pipeline. If so, the cue would be specific to this dataset's generators and may not transfer to
  other TTS systems. This is exactly what the held-out-family and cross-dataset evaluations must test.

## 9. Remaining risks
1. **Tiny group structure:** 11 speaker groups; **dev = 1 group**. Dev-based checkpoint and threshold selection is
   fragile, and probe CVs over 11 groups are noisy (differences < ~0.05 EER are not meaningful).
2. **Pause / dynamics cue** (B2, B4 moderate). Its effect on an LFCC detector is small (+0.014 EER). Its effect on
   stronger models is unknown, so the diagnostic must be repeated per model.
3. **Headroom:** 9.1 % of real clips exceed ±1.0. Any integer or codec step needs a fixed, class-neutral attenuation
   first (§6). Not yet implemented.
4. **Spectral cue attribution** is plausible, not proven (§8). Results on the held-out FreeVC family are the first
   check.
5. **Generator-dependent noise floor** (FreeVC very clean, XTTS slightly noisy): per-generator reporting is required.
6. **Pilot scale and test-heavy split:** 640 clips (train 200). The protocol puts every FreeVC-linked group in test.
7. **Licence:** Kathbath is still "Needs verification from current dataset terms." v2.1 audio (Kathbath-derived)
   must not be copied to teammates; it can be rebuilt from the documented commands.
8. **The provisional criteria themselves** (D-H3) are engineering choices and could be too lenient. Passing them is
   necessary, not sufficient.

## 10. Recommendation for training use: **READY_WITH_LIMITATIONS**
Evidence:
- **For use:**
  - all 13 acceptance checks pass, every leakage check at zero, and preprocessing is provably identical with no
    class-dependent peak processing;
  - nuisance cues are near chance (metadata, duration, edge silence, file level) or traced to pause/dynamics;
  - the dependence of a trained detector on pauses was measured and found small (+0.014 EER);
  - the main remaining separability (spectral) behaves like a legitimate synthesis cue.
- **Against READY_FOR_PILOT_TRAINING:**
  - a single dev group;
  - the pause/dynamics cue is moderate in probes and untested on stronger models;
  - the headroom guard for codec augmentation is missing.
- **Against NOT_READY:** none of the remaining issues is leakage or a class-dependent processing defect. They are
  scale and attribution limits that pilot training can safely be used to study.

**Allowed now:** pilot / development training on v2.1 (e.g. B0, B1, frozen XLS-R), labelled DIAGNOSTIC, under these
conditions:
1. Report every Hindi number with the B-probe floors (B2–B5) and the pause-controlled 2×2 for that model.
2. Do not use dev-only model selection as evidence. Also report the per-epoch dev curve, and treat thresholds as
   provisional.
3. No codec augmentation or PCM export of v2.1 until the fixed-attenuation guard exists.
4. Report per generator (seen XTTS vs held-out FreeVC).

**Not allowed yet:** confirmatory results or research claims on Hindi. Those need a full build with more speaker
groups (≥ 2 in dev), the headroom guard, and re-running this gate and diagnostic on the full build.
