# Dataset Card — VaaniShield canonical manifest v1

> Manifest: `data/manifests/vaanishield_v1.csv` (1,934 clips, 2.47 h). Built 2026-10-08 with
> `python -m ml.data.build_canonical_manifest`; checked with `python -m ml.data.check_manifest`
> (report: `data/manifests/vaanishield_v1_report.{md,json}`). Audio is NOT in Git (`data/raw/`, `data/processed/`).
> This is a **manageable subset** for pipeline work, not a training corpus for research claims.

## Composition
| Source | Clips | Label | Language | Codec as stored | Split rule | Licence (status) |
|---|---|---|---|---|---|---|
| ASVspoof 2019 LA | 1,074 (536 bona fide / 538 spoof) | both | en | FLAC 16 kHz | official partitions: train → train, dev → dev, eval → test | ODC-By 1.0 (verified) |
| IndicSynth Hindi | 320 (150 xtts_v2 + 150 freevc24 + 20 pilot) | spoof | hi | PCM_16 WAV 24 kHz | linked speaker groups (see below) | CC BY-NC 4.0 (verified) |
| Kathbath Hindi (IndicSUPERB) | 240 (20 speakers, test-known tar) | bona fide | hi | AAC (m4a) 44.1 kHz | linked speaker groups | **conflict**: CC0 (GitHub) vs CC-BY-4.0 (HF mirror); we follow CC-BY-4.0 + no redistribution |
| OpenSLR 104 Hindi-English | 300 segments (≤10 per recording, 30 speakers) | bona fide | hi-en | PCM_16 WAV 16 kHz (segments cut losslessly) | **test only** | CC BY-SA 4.0 (verified) |

## Splits (from the check report)
| Split | Bona fide | Spoof | Hours | Speaker groups | en / hi / hi-en |
|---|---|---|---|---|---|
| train | 421 | 543 | 1.16 | 36 | 634 / 330 / 0 |
| dev | 196 | 125 | 0.39 | 27 | 200 / 121 / 0 |
| test | 459 | 190 | 0.92 | 96 | 240 / 109 / 300 |

Spoof generators: train/dev = ASVspoof A01–A06 + xtts + freevc (dev has no xtts); test = ASVspoof A07–A19 (**unseen
in train by design**, official eval attacks) + xtts + freevc.

## How splits are made
- **ASVspoof**: official partitions (speaker-disjoint by the dataset's design).
- **Hindi (Kathbath + IndicSynth)**: IndicSynth reference audio comes from Kathbath, and its speaker IDs ARE Kathbath
  speaker IDs. Each fake clip links its cloned target voice and (for FreeVC) its conversion source speaker; linked
  speakers form one group, and whole groups go to one split (seeded, 70/15/15, both classes required per split).
  3 speakers (977, 1052, 3) have both real and fake audio in the subset; each is entirely in one split.
- **OpenSLR 104**: test only, because there is no Hinglish spoof yet; a single-class source must never be trained on.
  Speaker IDs come from the corpus' `transcripts/utt2spk` (30 speakers). Each segment records its recording as
  `parent_clip_id`; all segments of a recording stay in one split.

## Checks (2026-10-08): all hard checks PASS
required fields · split complete with both classes · no duplicates (file sha256 and decoded-audio hash) · no speaker
overlap · no derived clip in a different split than its parent / sibling segments · no generator-input leakage (a
fake's cloned or source speaker has no real audio in another split; no TTS sentence crosses splits).

## Known biases and shortcut risks (WARNINGS from the report)
1. **Hindi pool: codec, sample rate and source each perfectly separate the classes.** Every Hindi real clip is
   Kathbath AAC 44.1 kHz; every Hindi fake is IndicSynth PCM 24 kHz. A model can learn "AAC ⇒ real".
   **The Hindi data must not be used for training or for claims until this is addressed.** Full diagnosis, probe
   results and the proposed repair (paired Kathbath references, matched preprocessing, acceptance gate):
   `research/hindi_data_leakage_audit.md` (2026-10-08). Options (team decision):
   (a) pass both classes through the same lossy round-trip (e.g. AAC at Kathbath's settings) and the same resampling;
   (b) obtain Kathbath via the gated HF mirror if it stores a different format; (c) add other real Hindi sources
   with lossless audio. Whatever is chosen must be re-checked with `check_manifest`.
2. **Single-class sources**: IndicSynth (spoof only) and Kathbath (bona fide only) are used for training, so source
   identity equals label in the Hindi pool (same problem as 1).
3. **OpenSLR 104** is lecture-style technical speech; Kathbath is read speech; ASVspoof is VCTK read speech.
4. **Small speaker counts**: Kathbath 20 speakers; IndicSynth 18 target voices; test speakers per source are few,
   so per-slice results will have wide confidence intervals.
5. Exact-duplicate detection only; two recordings of the same sentence are not detected (Kathbath transcripts are
   not downloaded).

## Labelling rule
Label by provenance, never by listening: human-recorded = bonafide; TTS / voice conversion = spoof.

## Consent and privacy
No team recordings yet. All audio comes from public datasets under the licences above; nothing is redistributed.

---

# Hinglish pilot v1: `data/manifests/hinglish_pilot_v1.csv` (200 clips, 2026-10-08)

**Purpose:** verify the Hindi-English (code-switched) dataset and protocol on a laptop. **Not** a performance benchmark.
**Rebuild:** `python -m ml.data.build_hinglish_pilot plan` → `python scripts/acquire_csfleurs.py --files data/manifests/hinglish_pilot_plan.csv --accept-terms` → `python -m ml.data.build_hinglish_pilot build` → `python -m ml.data.check_manifest --manifest data/manifests/hinglish_pilot_v1.csv`.
Selection is deterministic (seed 2026). The plan CSV and manifest are in Git; the audio (~84 MB CS-FLEURS + 50 OpenSLR segments) is not.

## Sources (all language `hi-en`, all 16 kHz)
| Source | Role | Licence | What it is |
|---|---|---|---|
| OpenSLR 104 (MUCS Hindi-English, test archive) | real | CC BY-SA 4.0 | lecture speech, Kaldi segments of 2–15 s, cut losslessly |
| CS-FLEURS `read/test` (`audio/hin`) | real | CC BY-NC 4.0 | human read code-switched FLEURS sentences, **1 speaker (AD)** |
| CS-FLEURS `xtts/train` (`cs_hin_eng_n1_resample`) | spoof, family `xtts` | CC BY-NC 4.0 | XTTS synthesis, voices hin-spk0/1/2 |
| CS-FLEURS `mms/test` (`cs_hin_eng_..._vconcat`) | spoof, family `concatenative` | CC BY-NC 4.0 | MMS-TTS segments concatenated per language span |

The dataset guide's pilot planned real team recordings and a Hinglish XTTS model. Neither is used: team recordings
do not exist yet (consent pending), and the Hinglish-XTTS licence is unverified. **No speech was synthesised locally.**

## Composition and splits
| split | real | spoof | condition |
|---|---|---|---|
| train | 30 OpenSLR (15 speakers × 2) | 30 XTTS voice hin-spk0 | |
| dev | 10 OpenSLR (5 other speakers) | 10 XTTS voice hin-spk1 | threshold + epoch selection only |
| test T1 | 10 OpenSLR (5 other speakers) | 10 XTTS voice hin-spk2 | seen generator family, unseen voice and speakers |
| test T2 | 50 CS-FLEURS read (speaker AD) | 50 MMS concatenative | **held-out generator family** + unseen real source |

- **Speaker-disjoint:** 15 / 5 / 5 OpenSLR speakers; one XTTS voice per split; AD and the MMS voice are test-only.
- **Generator-disjoint holdout:** the `concatenative` family appears in test only.
- **Sentence-disjoint:** no FLEURS sentence id crosses splits.
- **Transcripts:** stored only as hashes.

## Checks (`hinglish_pilot_v1_report.md`): all 6 hard checks PASS
Required fields; split completeness; duplicates; speaker leakage; derived rows (segments of one recording share a
split); generator / transcript leakage.

## Shortcut risks (must be read before any number from this pilot)
1. **Train-split source confound:** train real = OpenSLR only, train spoof = CS-FLEURS only. `check_manifest` now
   checks the train split separately and flags this.
2. **Trivial-feature floor** (`runs/hinglish_pilot_audio_stats.csv`): a logistic regression on duration, peak level,
   RMS and leading silence, trained on the 60 train clips, gets **dev EER 0.20, T1 EER 0.00, T2 EER 0.40**.
   - XTTS clips are peak-normalised to 1.0; real clips peak lower.
   - OpenSLR segments are shorter, so they are repeat-padded more.
   - **T1 cannot show detector skill**: it is solved without listening to spoofing artefacts.
3. **Codec / sample rate:** not confounded (all 16 kHz; train/dev all PCM-16). MMS files are 32-bit PCM, but decoding
   to float removes this.
4. **Tiny n:** 20 dev and 20 T1 clips; one real speaker in T2. Any EER here has a confidence interval of tens of points.
5. **Domain:** OpenSLR is lecture speech; CS-FLEURS is read Wikipedia-style text.

## Consent
All audio comes from public datasets under the licences above. CS-FLEURS is non-commercial; this is research use only.

---

# Hindi v2 pilot (repaired): `data/manifests/hindi_v2_pilot.csv` (640 clips, 2026-10-08)

**Status: gate FAIL. Not for training or reported results.** Full report: `research/hindi_repair_report.md`.
Gate: `python -m ml.data.hindi_gate --manifest data/manifests/hindi_v2_pilot.csv` (9/10 checks pass; G10 waveform and
spectrum probes fail the PROPOSED thresholds, which are not yet team-approved).

- **What changed vs v1 Hindi** (v1 is kept unchanged for audit and reproducibility):
  - Real clips are the **exact Kathbath recordings each fake was generated from**: 314 references + 6 same-speaker
    recordings, from Kathbath *train*, fetched by `scripts/acquire_kathbath.py --fetch-ids`.
  - Fakes are the same 320 IndicSynth clips (160 XTTS-v2, 160 FreeVC24), source files unchanged.
  - Both classes: identical preprocessing v2 (16 kHz, 40 Hz high-pass, silence trim, −26 dBFS loudness, peak 0.95,
    PCM_16 WAV). Fakes additionally get Kathbath's capture codec (AAC 128 kb/s @ 44.1 kHz) so both share the last
    lossy stage.
  - Audio is in `data/processed/hindi_v2/pilot/`, per-file provenance in `hindi_v2_pilot_metadata.jsonl`.
- **Splits:** speaker-group disjoint; held-out family FreeVC in test only; sentence-disjoint.

  | split | real | spoof | groups | families |
  |---|---|---|---|---|
  | train | 100 | 100 | 4 | xtts |
  | dev | 25 | 25 | 1 | xtts |
  | test | 195 | 195 | 6 | freevc (held out) + xtts |
- **Shortcut status:**
  - metadata-only classifier EER 0.000 → 0.467;
  - waveform probe 0.17 → 0.24;
  - the strongest remaining cue is pause structure (fakes have more, cleaner pauses);
  - 9.4 % of real clips are peak-limited, versus none of the fakes;
  - the spectrum-envelope difference may be a genuine synthesis artefact.
- **Limitations:** 11 speaker groups only (dev = 1 group); Kathbath licence "Needs verification from current dataset
  terms." → do not transfer v2 audio.

---

# Hindi v2.1 pilot: `data/manifests/hindi_v2_1_pilot.csv` (640 clips, 2026-10-08), the current Hindi version

**Status: READY_WITH_LIMITATIONS, for pilot / development training only** (`research/hindi_v2_1_validation.md`).
Acceptance gate: ACCEPTANCE_PASS_PROVISIONAL (13/13; limits are provisional engineering acceptance criteria, D-H3).

- **Same clips, pairing, groups, splits and sentences as v2.** Only the preprocessing changed (D-H5):
  - no peak limiter (v2 limited 9.4 % of real and 0 % of fake clips);
  - one linear loudness gain to −26 dBFS active RMS per clip;
  - stored as **float32 WAV** so nothing clips; 29 real clips exceed ±1.0 and are kept exactly.
  - Preprocessing hash `587d94c32a05` for both classes.
- **Pauses kept** (D-H4). The diagnostic copy with pauses removed is `hindi_v2_1_pilot_pauses_removed.csv`
  (transform `diag:pauses_removed`, never primary data).
- **Diagnostic probes (EER):**
  - metadata 0.463 (nuisance, eliminated);
  - nuisance-only 0.341 (from pause/dynamics in the crop);
  - pause-only 0.419;
  - waveform statistics 0.221;
  - spectral 0.234 (pause-independent; high-band deficit of fakes in 15/19 paired speakers: plausibly a synthesis
    cue).
- **Pause dependence of a small LFCC detector:** +0.014 EER (diagnostic, 3 seeds).
- **Limitations:**
  - 11 speaker groups, dev = 1 group;
  - add a fixed class-neutral attenuation before any PCM/codec step;
  - Kathbath licence "Needs verification from current dataset terms." → do not transfer the audio.
- v1 and v2 are kept unchanged for audit and reproducibility.
