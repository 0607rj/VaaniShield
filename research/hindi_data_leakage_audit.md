# Hindi data leakage audit: can REAL vs SPOOF be told apart without hearing spoofing?

**Date:** 2026-10-08  **Scope:** Hindi pool of `data/manifests/vaanishield_v1.csv`. Controls: ASVspoof (English) and the
Hinglish pilot.
**Data:** 240 real Kathbath clips; 320 spoof IndicSynth clips (160 XTTS-v2, 160 FreeVC24).
**Reproduce:** `python scripts/audit_hindi_confound.py` → `runs/audit/hindi/features.csv`, `runs/audit/hindi/results.json`
(about 4 min CPU). No file under `data/` was modified; every candidate fix was applied in memory only.
**Status:** the Hindi pool stays **blocked for training** until the remediation in §G passes the gate in §G.5.

Reading guide:
- **EER 0.5** = chance; **EER 0** = a perfect shortcut.
- **"sep AUC"** = how well a single feature separates the classes (0.5 = not at all, 1.0 = perfectly).
- Classifiers are evaluated two ways:
  - "cv" = 5-fold cross-validation **grouped by speaker group** (no speaker group in both the training and the test
    fold);
  - "test" = train on the manifest's train split, score its (speaker-disjoint) test split.
- n is small (27 speaker groups), so treat differences below ~0.05 EER as noise.

---

## A. Exact cause of the confound

The two classes come from **two different releases with two different audio chains**. Our download scripts write the
upstream bytes unchanged (`acquire_indicsynth.py`: "audio bytes are written exactly as stored (no re-encoding)"), so
the differences are inherited, not created by us.

| | REAL: Kathbath (AI4Bharat) | SPOOF: IndicSynth Hindi |
|---|---|---|
| How it was made | Crowd-sourced read speech recorded on phones (Karya platform); released **only** as m4a ("The audio files are stored in m4a format", IndicSUPERB README) | XTTS-v2 (fine-tuned on IndicSUPERB) and FreeVC24 outputs, generated from **decoded** Kathbath reference recordings (IndicSynth paper §3; metadata names the reference files `…-<speaker>-<g>.wav`) |
| Container / codec | MP4/M4A (`mov`), **AAC-LC**, float planar | **WAV, PCM 16-bit** |
| Sample rate | **44.1 kHz** | **24 kHz** (native rate of both generators) |
| Bitrate | 127–128 kb/s | 384 kb/s (uncompressed) |
| Container tags | 8 tags, handler `SoundHandle` | none |
| Level | not normalised (median peak 0.63) | **peak-normalised** (median peak 0.999) |
| Silence | short pauses with room/phone noise | near-digital silence in pauses; XTTS appends ~0.3 s trailing silence |
| DC offset | ≈ 0 | FreeVC24 output carries a small DC offset (median −0.004) |
| Speakers in **our** pool | 20 speakers from Kathbath *test-known* | 19 target voices + 6 VC source speakers, drawn mainly from other Kathbath splits |

**Root causes, in order of how dangerous they are for a detector** (the model sees decoded 16 kHz audio, never the
container):
1. **Level normalisation asymmetry.** IndicSynth output is peak-normalised and 5 dB louder; Kathbath is raw phone level.
2. **Speaker composition.** Only **4 of 19** IndicSynth target speakers have *any* real clip in our pool. **0 of the
   323** Kathbath reference recordings that our 320 fakes were generated from are in our pool. Real and fake are mostly
   *different people*, so speaker/voice characteristics differ by class.
3. **Recording condition vs. synthetic silence.** Real pauses contain phone/room noise; synthetic pauses are much
   cleaner (noise floor −30 vs −35 dB relative to the loudest frame, and −31 vs −43 dB once FreeVC's DC offset is
   removed).
4. **Generator housekeeping artefacts** that are not "speech-generation artefacts" in the research sense: FreeVC DC
   offset; XTTS trailing silence (already removed by our silence trimming).
5. **File format** (AAC 44.1 kHz vs PCM 24 kHz). This is a *perfect* separator in metadata. After our decode → 16 kHz,
   however, its trace is weak for simple features (§D.4).

The selection procedure made 2 worse: we took Kathbath clips from the ungated *test-known* tar (small, Hindi
reachable by range requests), independently of which speakers IndicSynth cloned.

## B. Which datasets and splits are affected

| Pool | Affected? | Evidence |
|---|---|---|
| Hindi (Kathbath + IndicSynth), **all splits**: train 105 real / 225 spoof; dev 96 / 25; test 39 / 70 | **Yes**: metadata perfectly separable; level, speaker and silence shortcuts | §C, §D, §E |
| Hindi-English pilot (OpenSLR / CS-FLEURS) | Separate, already documented (train-split source confound, peak-normalised XTTS) | DATASET_CARD § Hinglish pilot; research_gap §8 |
| English (ASVspoof 2019 LA) | Known silence/noise shortcut (Müller et al. 2021); weaker here (§E control) | §E |
| Codec variants of Hindi clips (`processed/codec_sim/*/kathbath_hi`, `…/indicsynth_hi`) | Inherit the Hindi confound (same parents) | `codec_pilot_v1.csv`, `robust_smoke_codec.csv` |

**No model has been trained on Hindi** (all training configs use `include_sources: [asvspoof2019_la]` or the Hinglish
pilot). The Hindi rows were used only as *test* clips in the robustness smoke run, whose numbers are labelled
non-final.

## C. Evidence from the metadata (`results.json` → `metadata`)

| Property | REAL (n=240) | SPOOF (n=320) |
|---|---|---|
| container / codec | mov / aac (240) | wav / pcm_s16le (320) |
| sample rate | 44,100 (240) | 24,000 (320) |
| sample format | fltp | s16 |
| bitrate kb/s (min / median / max) | 127.4 / 128.0 / 128.2 | 384 / 384 / 384 |
| file size KB (median) | 91.7 | 262.1 |
| tags (handler) | 8 tags (`SoundHandle`) | 0 |
| native bandwidth, −50 dB rule (median; min–max) | 16,968 Hz (10,293–19,811) | 11,063 Hz (3,961–12,000) |
| native duration s (median; min–max) | 5.68 (2.88–14.72) | 5.59 (3.52–12.46) |
| speakers | 20 | 19 targets (+6 VC sources) |
| speakers with both classes | 4 (groups: 4 of 27) | |
| reference recordings of the fakes present as real clips | — | **0 of 323** |
| gender (from file names) | 125 m / 115 f | balanced by design (IndicSynth) |

Every one of the first six rows separates the classes perfectly by itself. Duration does **not** differ.

## D. Evidence from simple statistical tests

Mann–Whitney U, real vs spoof, on features of what the detector actually sees ("model view" = decode → 16 kHz →
trim silence → centre crop / repeat-pad to 4.04 s, i.e. `ml/data/audio.py` exactly).

### D.1 Current pipeline (M0), strongest single features
| Feature | REAL median | SPOOF median | sep AUC | p |
|---|---|---|---|---|
| peak amplitude | 0.594 | 0.996 | **0.88** | 1e-53 |
| active RMS (dBFS) | −21.3 | −16.4 | **0.81** | 1e-36 |
| noise floor (10th pct frame energy, dB rel. max) | −29.9 | −35.3 | 0.68 | 1e-12 |
| DC offset | 0.0000 | −0.0005 | 0.67 | 5e-12 |
| clipped-sample fraction | | | 0.65 | 2e-16 |
| silent-frame fraction | 0.019 | 0.044 | 0.62 | 1e-6 |

### D.2 Before trimming (file as stored, decoded to 16 kHz)
peak (sep AUC 0.93), leading silence 0.025 s vs 0.000 s (0.88), active RMS (0.82). Trailing silence 0.025 s vs 0.325 s
(XTTS). Our existing silence trimming removes most of the leading/trailing-silence signal; it does not remove level.

### D.3 Generator breakdown (M0 medians)
| | REAL | XTTS-v2 | FreeVC24 |
|---|---|---|---|
| noise floor dB | −29.9 | −37.5 | −34.4 |
| silent-frame fraction | 0.019 | 0.075 | 0.031 |
| DC offset | 0.0000 | 0.0000 | −0.0039 |
| peak | 0.59 | 0.995 | 0.997 |

### D.4 Does the AAC format itself leave a trace after decoding to 16 kHz?
Test: the **same** 320 spoof clips, original vs. passed through AAC-LC 128 kb/s at 44.1 kHz (Kathbath's chain), then
our pipeline. A probe has to tell "AAC" from "not AAC" with speaker-grouped CV.

| Probe | EER | AUC |
|---|---|---|
| 40-band long-term spectrum | 0.434 | 0.58 |
| 6–8 kHz bands only | 0.444 | 0.59 |
| waveform statistics | 0.463 | 0.54 |

Median spectral change in the 6–8 kHz bands: < 0.1 dB, except −0.6 dB in the top 200 Hz band.

**Conclusion:** at 128 kb/s, AAC is close to transparent once decoded to 16 kHz, so simple features barely detect it.
A deep detector *might* still learn subtle quantisation traces; this audit cannot rule that out (§G.3 handles it).

## E. Can a trivial classifier identify REAL/SPOOF from non-semantic information alone?

Features never describe speech content or synthesis artefacts:
- duration, silence, level, DC, noise floor, clipping;
- coarse band energies, roll-off, flatness of pauses;
- separately, a 40-band long-term spectrum (LTAS).

Models: logistic regression and random forest. EER (lower = stronger shortcut):

| Input | logreg cv / test | forest cv / test |
|---|---|---|
| **E1: container metadata** (sample rate, codec, bitrate, file size, duration) | **0.000 / 0.000** | **0.000 / 0.000** |
| File as stored, decoded, untrimmed: waveform stats | 0.113 / 0.155 | **0.034 / 0.054** |
| **Current pipeline (M0)**: waveform stats | 0.166 / 0.257 | 0.184 / 0.257 |
| M0, level-free stats (peak/RMS/crest/clipping removed) | 0.296 / 0.331 | 0.287 / 0.331 |
| M0, LTAS spectrum | 0.330 / 0.210 | 0.312 / 0.331 |
| *Control:* English (ASVspoof, 560 clips), M0 waveform stats | 0.337 / 0.411 | 0.327 / 0.395 |
| *Control:* English, M0 LTAS | 0.245 / 0.348 | 0.218 / 0.310 |
| *Control:* Hinglish pilot, M0 waveform stats | 0.590 / 0.600 | 0.400 / 0.617 |

**Answer:**
- **Yes.** From container metadata alone, REAL/SPOOF is identified perfectly.
- From the stored waveforms (no speech-content features), a random forest reaches EER 0.03–0.05.
- Through our current pipeline, simple non-semantic features still reach EER 0.17–0.26. That is clearly stronger than
  the same probe on ASVspoof (0.33–0.41), mainly because of **level** (peak/RMS).
- A CNN given raw audio can be expected to exploit these at least as well.

(The Hinglish-pilot controls are above 0.5 in cross-validation because its sources differ between train and test;
its separate shortcut is documented in research_gap §8.)

## F. Can the problem be corrected? Results of candidate fixes (all in memory, `results.json` → `classifiers`)

| Variant (both classes unless stated; all through trim + crop) | Waveform stats, forest cv / test | Level-free, forest cv / test | LTAS, forest cv / test | Strongest remaining single feature (sep AUC) |
|---|---|---|---|---|
| M0 current pipeline | 0.184 / 0.257 | 0.287 / 0.331 | 0.312 / 0.331 | peak 0.88 |
| M1 + loudness normalisation (−26 dBFS active RMS, peak-safe) | 0.268 / 0.304 | 0.309 / 0.331 | 0.312 / 0.331 | noise floor 0.68 |
| M2 + spoof through matched AAC (44.1 kHz, 128 kb/s) | 0.275 / 0.284 | 0.304 / 0.311 | 0.316 / 0.311 | DC offset 0.68 |
| M3 + both classes through the same AAC | 0.266 / 0.284 | 0.287 / 0.331 | 0.316 / 0.311 | DC offset 0.67 |
| M4 = M2 + 40 Hz high-pass (DC removal) | 0.250 / 0.311 | 0.280 / 0.358 | 0.346 / 0.304 | silent-frame fraction 0.79, noise floor 0.78 |
| M5 = M4, **only the 4 speaker groups that have both classes** (47 real / 145 spoof) | 0.339 / 0.411 | — | **0.556 / 0.527** | noise floor 0.86, silent-frame fraction 0.86 |

What each fix achieves:
- **Format unification** (decode → 16 kHz PCM, which the pipeline already does) removes the *metadata* shortcut
  completely, because the model never sees the container.
- **Matched AAC transcoding** (M2/M3) changes almost nothing measurable (consistent with §D.4). It is cheap insurance
  against subtle codec traces, not the main fix.
- **Loudness normalisation** (M1) removes the strongest shortcut (peak/RMS; EER 0.18 → 0.27).
- **DC removal** (M4) removes the FreeVC DC cue but *reveals* the synthetic-silence cue more clearly (silent-frame
  fraction and noise floor sep AUC 0.79).
- **Speaker matching** (M5) is the decisive test for the spectrum: within the same speaker groups the LTAS probe
  falls to chance (0.53–0.59). So the class difference in spectral shape comes from **speaker composition**, not
  from codec or synthesis.
- **Remaining after all fixes:** cleaner pauses in synthetic speech (noise floor, silent-frame fraction). These
  cannot be removed without treating the classes asymmetrically: adding noise only to fakes would create a new
  artefact. Part of this difference is arguably a genuine property of synthetic speech; it is also exactly the kind
  of cue that disappears in real attacks once audio is re-recorded or transmitted.
- Duration matching is not needed: durations already match (median 5.68 vs 5.59 s), and the fixed 4.04 s crop
  removes duration from the model view.

**So: the format/codec part is correctable (and largely already neutralised by decoding). Level and DC are
correctable by matched preprocessing. Speaker composition is correctable by re-sampling real data. The
recording-condition difference is only mitigable, and must be measured and reported.**

## G. Recommended correction (proposed; nothing applied yet)

### G.1 Repair the real class: speaker- and recording-matched pairs (data step, ~30–40 MB)
- Fetch the **323 exact Kathbath reference recordings** named in IndicSynth's metadata ("Target Reference Audio" and
  "Source Reference Audio"). Add other utterances of the same 19 target and 6 source speakers, up to a fixed number
  per speaker.
- Route: HF mirror `ai4bharat/Kathbath` (gated; each person accepts the terms with their own account), or the
  objectstore train/valid tars by range request.
- Balance per speaker group: equal real/spoof clips per group (cap the larger side), at least 2 groups per split.
- Splits keep using the existing linked speaker graph (`link_groups`), so a VC source speaker and its cloned targets
  stay in one split.
- Keep the current 240 test-known Kathbath clips, labelled **unpaired real**. They are useful for measuring the bona
  fide false-alarm rate, but are not used to compare classes.

### G.2 Matched preprocessing for both classes (pipeline v2; versioned, reversible)
Order: decode → mono → 16 kHz (unchanged) → **40 Hz high-pass (DC removal)** → trim silence (unchanged) →
**loudness normalisation to −26 dBFS active RMS with peak limit 0.95** → crop.
- Applies to **every** language and class.
- ASVspoof baselines must then be re-run under v2, or v1 kept for comparability with the upstream AASIST recipe.
  **Team decision D-H2.**

### G.3 Codec: make the format irrelevant rather than "fixed"
- Decoding to 16 kHz PCM already removes the container cue.
- Add a **matched-codec ablation**: spoof through AAC-LC 128 kb/s at 44.1 kHz (Kathbath's chain). If detector EER on
  Hindi changes materially between "original spoof" and "AAC-matched spoof", the detector is using codec traces.
- Training uses the planned **symmetric** codec/noise augmentation (both classes, same distribution), so neither
  class "owns" a codec.

### G.4 Residual synthetic-silence cue
- Do **not** add noise to fakes only.
- Report, next to every Hindi result, the trivial-feature probe EER on the *same* test clips (this audit script).
- Report results on the codec/noise conditions where both classes share a channel.

### G.5 Acceptance gate (re-run `scripts/audit_hindi_confound.py` on the repaired manifest)
Proposed values; the team must confirm them **before** any detector is trained on Hindi:
1. Model-view trivial probe (waveform stats, forest): cv **and** train→test EER ≥ 0.35. This is close to the
   ASVspoof control, which is the floor accepted in the field today.
2. No single non-semantic feature with sep AUC > 0.80.
3. LTAS probe EER ≥ 0.40.
4. `check_manifest`: all hard checks PASS; train-split source warning acknowledged in DATASET_CARD.

If the gate fails, Hindi is used for **evaluation only**, and every Hindi number is reported with its probe floor.

## H. Remove or repair?

**Repair, do not remove.** Hindi is central to RQ1 (language shift), and the problem is a sampling and processing
mismatch, not corrupted data:
- **IndicSynth clips:** keep all 320 unchanged on disk. Fixes happen in preprocessing and pairing, not by rewriting
  files.
- **Kathbath test-known clips:** keep, re-labelled "unpaired real" (§G.1).
- **Real clips:** add the speaker-matched reference recordings.
- **Manifests:** do not delete or overwrite `vaanishield_v1.csv`. Build `vaanishield_v2` (paired Hindi) as a new
  file, and keep v1 for provenance.
- **Codec variants of Hindi clips:** regenerate from the v2 manifest (deterministic, cheap).

## I. What this means for the research paper

1. **No Hindi detection result from the v1 pool may be reported.** Under v1, a near-perfect Hindi EER would be
   expected from shortcuts alone (E1 metadata 0.00; waveform probe 0.03–0.26).
2. RQ1 / H1 / H2 (English → Hindi → Hinglish shift) depend on the repaired, speaker-matched Hindi pool and the
   acceptance gate. Until then, Hindi results are **pending**.
3. The audit belongs in the paper's dataset section as **methodology**:
   - pairing IndicSynth with independently sampled Kathbath creates format, level and speaker shortcuts;
   - we measured them with trivial probes and fixed them by pairing and matched preprocessing.

   Do **not** claim this audit approach as novel: shortcut audits exist (e.g. Müller et al. 2021 on ASVspoof
   silence). We have **not** checked how other papers using IndicSynth handled their real-class pairing, so we must
   not claim that they are affected.
4. Every results table should carry a **"trivial-probe EER" column** per test condition (already recommended in
   research_gap §8). A detector result is only meaningful by how far it beats that floor.
5. The remaining synthetic-silence cue is a stated limitation. The codec / real-app conditions (RQ2) are where it is
   expected to disappear, which makes RQ2 more important, not less.

## Open items
- **D-H1:** approve fetching the 323 paired references (+ same-speaker utterances); route A or B (B needs each
  person's own HF acceptance).
- **D-H2:** adopt preprocessing v2 for all languages, or Hindi-only with a documented exception.
- **D-H3:** confirm the acceptance-gate thresholds in §G.5 before any Hindi training.
