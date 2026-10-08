# VaaniShield — Phase 0 Plan (v0)

Date: 2026-10-07. Status: **plan only — nothing implemented, no experiments run.**

Provenance tags used throughout:

| Tag | Meaning |
|---|---|
| **[DOC:syn]** | Verified from `vaanishield-synopsis.pdf` (page numbers are printed page numbers) |
| **[DOC:pg]** | Verified from `vaanishield-project-guide.pdf` |
| **[DOC:dg]** | Verified from `vaanishield-dataset-guide.pdf` |
| **[DOC:p1]** | Verified from `paper1.pdf` (HAV-DF, arXiv:2411.15457) |
| **[DOC:p2]** | Verified from `paper2.pdf` (IndicSynth, ACL 2025) |
| **[EXT]** | Checked against an external web source on 2026-10-07 (URL given in §H) |
| **[REC]** | Engineering recommendation (ours) |
| **[HYP]** | Hypothesis / assumption — untested |

---

## A. Document inspection

### A.1 What each document is

| File | What it is | Role for us |
|---|---|---|
| Synopsis (26 pp, 23 Aug 2026) | Formal proposal: AKGEC / AKTU, Group-20, supervisor Ms. Swati Tomar | Authoritative scope, RQs, objectives, architecture, timeline |
| Project guide v1.0 (21 pp, 24 Sep 2026) | Implementation guide: model ladder B0/B1/P1/P2, experiment IDs E0–E3, A1–A5, manifest schema, eval protocol | Authoritative engineering protocol |
| Dataset guide v1.0 (8 pp, 25 Sep 2026) | Hinglish data sources + 200-clip pilot | Authoritative for the Hinglish real/fake sourcing |
| paper1 — HAV-DF | Hindi **audio-video** deepfake dataset: 508 videos (200 real, 308 fake), YouTube-sourced, voice cloning via RVC; benchmarks only **video** detectors (accuracy %, not EER); data "available on reasonable request" | Related work only. Supports "Hindi is under-served", **not** usable as audio training data |
| paper2 — IndicSynth | ~4,000 h synthetic speech, 12 Indic languages, 989 target speakers, generated from IndicSUPERB bona fide speech with XTTS-v2 / FreeVC24 / VITS; CC BY-NC 4.0 | Primary Hindi **fake** source; reveals the matched **real** source (IndicSUPERB/Kathbath) |

### A.2 Key verified facts that shape the design

1. **IndicSynth Hindi has only two generators, both "Diversity" subsets** [DOC:p2, Table 1]: XTTS-v2 (53 F / 48 M target speakers, 101,202 clips, 171.36 h) and FreeVC24 (53 F / 48 M, 104,736 clips, 167.77 h). Paper states "the mimicry subsets for Hindi and Sanskrit are absent" [DOC:p2 §7].
   → Leave-one-generator-out on Hindi = only **2 folds** (XTTS ↔ FreeVC24).
2. **IndicSynth fakes are built from IndicSUPERB speakers and transcripts** [DOC:p2 §3]; "speaker IDs used in IndicSynth are the same as those used in IndicSuperb" [DOC:p2 App. A]. VC metadata records **both source and target speaker** IDs.
   → Speaker-disjoint splits must group target *and* VC-source speakers together.
3. IndicSUPERB is CC0 [DOC:p2 §8]. Hindi Kathbath = 150.2 h, CC0, on HF `ai4bharat/Kathbath` [EXT].
4. ASVspoof-2019-trained AASIST / AASIST-L / RawNet2 degrade heavily on IndicSynth Hindi without adaptation [DOC:p2 Table 2]. (These are *their* numbers. We will not reuse them as our results.)
5. AASIST has 297,866 params, AASIST-L 85,306 [DOC:p2 App. E]; both official code is MIT [DOC:p2 App. G].
6. Hindi IndicSuperb clips average **2.65 s** [DOC:p2 App. E] — shorter than our planned 4 s training crop (padding policy matters; see risk R7).
7. OpenSLR 104 Hindi-English: CC BY-SA 4.0, 16 kHz/16-bit, 89.86 h train + 5.18 h test, spoken technical tutorials [DOC:dg][EXT]; archives 7.3 GB + 443 MB [EXT]. **The OpenSLR page gives no speaker information** [EXT].
8. Hinglish XTTS-v2 (`harrrshall/xtts-v2-hinglish-synthetic`): CPML-derived, research/non-commercial, **4 fixed voices**, trained on 8.56 h fully synthetic audio, fine-tuned from XTTS-v2 [EXT; consistent with DOC:dg].

### A.3 Discrepancies and gaps between the documents

| # | Topic | Synopsis | Project guide | Dataset guide | Resolution [REC] |
|---|---|---|---|---|---|
| D1 | Real Hindi source | Not named (IndicSynth + "selected multilingual data") | IndicVoices-R or another licence-approved Hindi corpus | OpenSLR 104 (Hinglish) + team recordings | Use **Kathbath-Hindi** (the IndicSynth source corpus) as the primary Hindi real class to avoid a corpus shortcut; OpenSLR 104 + team recordings for Hinglish real; IndicVoices-R optional |
| D2 | Label vocabulary | bona fide / synthetic | `bonafide` / `spoof` | real / fake | Canonical `bonafide` / `spoof` in manifests; UI may say "real/fake" |
| D3 | Manifest columns | not specified | 8 columns | 14 columns | Single superset schema (§D.4) |
| D4 | Folder layout | `vaanishield/ml/...` | `ml/`, `data/{raw,processed,manifests}` | `dataset/audio/original|whatsapp/...` | One layout (§F); data lives **outside** Git |
| D5 | Input duration | 4 s crops | 2–20 s accepted by demo, 4 s crops | – | Training: 4 s; demo: accept 2–20 s, windowed |
| D6 | Generator for "language-mix" | Hindi+English segments of **same class** at pauses | "Join consented Hindi and English segments with logged boundary" | – | Same-class splice, **symmetric** across classes (see R9) |
| D7 | min t-DCF | Listed as a dependent variable | "Report only if implemented with correct protocol" | – | Only on ASVspoof 2019 LA, using its official ASV scores |
| D8 | Web stack | React/Vite + Express + MongoDB + FastAPI | Gradio/FastAPI first; Express/Mongo optional | – | Follow guide: FastAPI + Gradio first; React later; no MongoDB in MVP |
| D9 | "Indian English" | Abstract claims Hindi, Hinglish **and Indian English** | Only Hindi + Hinglish | – | No Indian-English corpus is identified anywhere. Either add one or drop the claim. **Needs team decision** |
| D10 | CS-FLEURS MMS-Test | – | – | "Concatenative TTS" | Unverified; MMS-TTS is generally described as VITS-based. Verify on the dataset card before using the label |
| D11 | Unseen-generator design | Leave one TTS/VC out | Reserve one generator | Hold out one engine | Hindi has only 2 generators, and the Hinglish sources are XTTS-family. Needs a **generator-family** column (R4) |
| D12 | Timeline | Sep 2026 = env setup; Oct = ASVspoof/AASIST | 12–16-week plan | – | Today is 7 Oct 2026, so we are ~1 month behind the synopsis Gantt. Plan in §G compresses it |
| D13 | HAV-DF role | Cited as a dataset in lit review | "related work only" | – | Related work only (audio-video, small, on request) |

---

## B. Architecture proposal

The documented architecture is kept as is [DOC:syn 4.2.1, DOC:pg §05/§08]. We only add interfaces and controls.

```
waveform (any fmt) ─► decode+QC ─► mono 16 kHz ─► VAD-trim (logged) ─► peak-safe norm
   │                                                                     │
   │  TRAIN ONLY: codec aug (MP3/Opus/AMR-NB/resample/low-pass) ─ RawBoost ─ language-mix splice
   ▼                                                                     ▼
 4 s crop (train) / overlapping 4 s windows, hop 2 s (inference)
   ▼
 XLS-R-300M (frozen; later top-k blocks unfrozen) ─► 25 hidden states (CNN-out + 24 blocks)
   ▼
 softmax-weighted layer sum  H = Σ α_l H_l          (α learnable; logged per run)
   ▼
 linear/conv adapter (1024 → 128)
   ▼
 Head: P1 = attentive-stats pooling + MLP      P2 = AASIST-lite graph head
   ▼
 logit (spoof) ─► temperature T (fitted on dev only) ─► p_spoof per window
   ▼
 clip score = documented aggregation (mean logit), segment timeline, quality warnings
```

Model ladder [DOC:pg §08]: **B0** LFCC-LCNN → **B1** AASIST (official) → **P1** XLS-R + light head → **P2** XLS-R + AASIST-lite.

Design choices [REC]:
- **Score convention.** Internally `score = spoof logit`, so higher means more likely spoof. ASVspoof tools expect bona fide to score higher, so the conversion happens once, in `metrics/`, and a unit test checks it.
- **Stable interface** `score(waveform, sample_rate) -> {p_spoof, windows[], warnings[]}` [DOC:pg §19] shared by all models, so evaluation and demo code never branch per model.
- **Compute.** No local NVIDIA GPU was detected on the team laptop. Run XLS-R frozen in fp16 on a Colab/Kaggle T4 [DOC:syn 1.5.1]. Augmentation is online and on CPU workers. Do not cache 25-layer features (~10 MB per 4 s clip in fp16), which is impractical.
- **Compactness.** If the learned α concentrates on lower or middle layers [HYP], truncate XLS-R above that layer for the demo. This cuts CPU latency and is measured, not assumed.

**Not novel (prior work exists).** XLS-R/wav2vec2 + AASIST (Tak et al. 2022, synopsis ref [10]) and learned multi-layer XLS-R weighting (SLS, Zhang et al., ACM MM 2024) [EXT]. Our architecture is a **re-implementation baseline**, not the contribution.

---

## C. Research-gap interpretation

### C.1 What prior work already covers [EXT unless noted]
- **Indic audio deepfake detection:** IndicSynth [DOC:p2], Indic-CodecFake/SATYAM (ACL Findings 2026; 12 Indic languages, neural-codec fakes, real speech from IndicSUPERB, CC BY 4.0), SEA-Spoof (includes Hindi; CC BY-NC-ND 4.0), IndicTTS Deepfake Challenge, and a 2026 preprint on pre-trained encoders across 22 Indic languages (arXiv 2608.12536).
- **Channel/compression robustness:** ADD-C (codecs + packet loss, arXiv 2504.12423), "Measuring robustness under real-world corruption" (arXiv 2503.17577), RTCFake (~600 h with real-platform transmission paired with offline audio, e.g. Zoom; arXiv 2604.23742), Codecfake / CodecFake+.
- **Multilingual SSL front ends:** Phukan et al. NAACL 2024 (synopsis [4]), SLS.
- **Code-switching deepfakes:** ArEnAV (Arabic-English, audio-visual). **No Hindi-English code-switched audio-deepfake benchmark was found in our searches.** This is search-limited evidence, not proof of absence.

### C.2 Defensible gap [DOC:pg §02 wording, refined]
> A speaker-disjoint evaluation of Hindi-English **code-switched** audio-deepfake detection under **paired** clean / simulated-codec / **real messaging-app** conditions, with **unseen-generator(-family)** testing and **calibration** reporting, using a compact multilingual detector.

Use the word "underexplored", never "first" [DOC:pg §02], until a systematic review is done (Phase 1 task).

### C.3 Candidate contributions, ranked by feasibility × defensibility [HYP — novelty unchecked]
| ID | Candidate | Why it may be new | Known overlapping prior work |
|---|---|---|---|
| C1 | **HinglishFake-Test**: consented, speaker-disjoint, with generator-family holdout | No Hinglish ADD benchmark found | IndicSynth, SEA-Spoof (monolingual) |
| C2 | **Paired sim-to-real channel-gap measurement**: the same source clip under clean, simulated (exact params) and real app round-trip (app/version/date), reporting ΔEER **and ΔECE** | RTCFake does paired real transmission, but not for Hinglish and not WhatsApp-style voice notes as far as we found | RTCFake, ADD-C |
| C3 | **Symmetric code-switch splice augmentation** with shortcut controls (splice applied equally to both classes; a "splice-only" probe shows the splice itself is not a cue) | No code-switch augmentation for ADD found | Partial-spoof / splicing literature |
| C4 | **Channel-conditioned calibration**: estimate the channel/quality condition from the audio and select a temperature/threshold per condition, fitted on *simulated* dev conditions and tested on *real-app* | Calibration under compression is rarely reported in ADD | Quality-aware calibration/fusion exists in biometrics generally; must search |
| C5 | Switch-point-aware segment analysis: does suspicion concentrate at language-switch boundaries? | Analysis contribution | – |

**Patent-sensitive:** C4 (and C4 + C5 combined as a system: channel estimation → conditional calibration → segment timeline with quality warnings). See PROJECT_STATE.md §Patent.

---

## D. Dataset strategy

### D.1 Roles
| Role | Source | Label | Licence (status) | Size / note |
|---|---|---|---|---|
| English baseline | ASVspoof 2019 LA | bonafide + spoof (A01–A19) | **Licence text not visible on DataShare page — verify** [EXT] | LA.zip 7.12 GB [EXT] |
| Hindi real | Kathbath Hindi (IndicSUPERB) | bonafide | CC0 [EXT][DOC:p2] | 150.2 h [EXT]; use a speaker-balanced subset |
| Hindi fake | IndicSynth Hindi (XTTS-v2, FreeVC24) | spoof | CC BY-NC 4.0 [DOC:p2][EXT] | 206k rows on HF viewer [EXT]; subset |
| Hinglish real | OpenSLR 104 | bonafide | CC BY-SA 4.0 [DOC:dg][EXT] | 7.3 GB train archive; lecture domain; **speaker IDs unknown** |
| Hinglish real (2) | Team/participant recordings (consented) | bonafide | Own consent terms | Natural phone speech [DOC:dg p7] |
| Hinglish real (opt.) | SwitchLingua audio (gated) | bonafide | CC BY-NC 4.0 [DOC:dg] | Confirm Hindi-English subset |
| Hinglish fake | Hinglish XTTS-v2 (4 voices) | spoof | CPML-derived, non-commercial [EXT] | XTTS family |
| Hinglish fake (2) | CS-FLEURS XTTS/MMS subsets; non-XTTS generator TBD | spoof | Verify per subset | Needed for a true unseen-family test |
| External realism | In-the-Wild | both | Apache-2.0 [EXT] | 20.8 h bona fide + 17.2 h spoof, 58 speakers [EXT]; English; **never tuned on** |
| Optional | MLAAD, IndicVoices-R, SEA-Spoof | – | verify | later only |

### D.2 Acquisition order [DOC:pg §17, adapted]
ASVspoof LA → OpenSLR 104 **test** archive (443 MB; inspect 20 clips first [DOC:dg]) → Kathbath-Hi subset → IndicSynth-Hi subset → Hinglish XTTS generation → consented recordings → real-app round-trips → In-the-Wild (only after the pipeline is frozen).

### D.3 Split policy [REC]
1. Build a **speaker graph**: nodes = speaker IDs; edges = (VC source ↔ target), (clone reference ↔ output voice), (same physical person across corpora, e.g. a team member who is also a clone reference). Split by **connected component**. This enforces the dataset guide rule that "the reference speaker must also stay within one split" [DOC:dg p7].
2. Split by component, then create crops, augmentations and app variants **inside** the split [DOC:pg §04; DOC:syn 4.2.3].
3. **Transcript disjointness** for Hindi test: no test sentence text in train (IndicSynth reuses Kathbath transcripts).
4. **Generator-family holdout:** `generator_family` ∈ {xtts, freevc, vits, mms, …}. Every "unseen" claim is at family level, because XTTS appears in IndicSynth, Hinglish-XTTS and CS-FLEURS.
5. **OpenSLR 104:** if no speaker IDs exist, group by recording/tutorial ID derived from utterance IDs, and document this as a weaker guarantee.
6. **Test freeze:** test manifests are hashed and committed before any model sees them; any change creates a new version.

### D.4 Canonical manifest schema (superset of DOC:pg §04 + DOC:dg p7)
`clip_id, path (dataset-relative), label{bonafide,spoof}, source_dataset, subset, speaker_id, speaker_group, gender, language{hi,en,hi-en}, script, transcript_id, generator, generator_family, generator_version, parent_clip_id, transform{clean|sim:<codec>:<params>|app:<app>:<ver>:<date>:<mode>}, sample_rate_orig, duration_s, sha256, licence, consent_status, split, notes`

### D.5 HinglishFake-Test (project benchmark) [DOC:syn 4.2.2, DOC:dg p6–7]
- Pilot: 100 real (50 OpenSLR + 50 team) + 100 fake (50 Hinglish-XTTS + 50 second engine) plus compressed copies [DOC:dg].
- Full target [HYP]: ≥ 20 consented speakers, ≥ 3 generator families, ≥ 2 apps. Size is set by speaker count, not hours.
- Consent covers recording, research retention, optional voice cloning for the **fake** class, and withdrawal. No cloning without that specific consent [DOC:pg §16].
- Real-app protocol [DOC:pg §09], with one addition [REC]: **verify that re-encoding actually occurred** by comparing input and output codec and hash. Sending a file "as document" may pass bytes through unchanged. Voice-note recording through a mic is *re-recording* and must be labelled as a separate condition.

---

## E. Experiment strategy

### E.1 Training compositions
- `T_en`: ASVspoof19 LA train
- `T_hi`: Kathbath-Hi + IndicSynth-Hi (one family held out per fold)
- `T_en+hi`: union
- `±aug`: codec / RawBoost / language-mix toggles

### E.2 Test conditions
ASVspoof19 LA eval | Hindi test (seen/unseen family) | HinglishFake-Test × {clean, sim-MP3, sim-Opus, sim-AMR-NB(-like), resample, real-app} | In-the-Wild.

### E.3 Matrix (IDs from DOC:pg §11)
| ID | Run | Answers |
|---|---|---|
| E0 | B0 LFCC-LCNN, all conditions | baseline floor; shortcut check |
| E1 | B1 AASIST | specialist baseline |
| E2 | P1 frozen XLS-R + light head | RQ3 |
| E3 | P2 XLS-R + AASIST-lite, full recipe | primary RQ |
| A1 | − codec aug | RQ2 |
| A2 | − language-mix aug | RQ1 / code-switch |
| A3 | XLS-R vs monolingual wav2vec2 (and final-layer only vs weighted) | RQ3 |
| A4 | sim-only vs + real-app in training | sim-to-real gap (A4 requires a separate real-app **training** pool from different speakers) |
| A5 | seen vs leave-one-family-out | RQ4 |
| C2/C4 | channel-gap table; channel-conditioned calibration | candidate contribution |

Each cell: ≥ 3 seeds [DOC:syn 4.2.4]; EER (primary), AUC, F1/FPR/FNR at a dev-fixed threshold, ECE (15 bins) before/after temperature scaling, plus NLL/Brier [REC]; latency median/p95 at batch size 1 on named hardware; 95% bootstrap CIs **resampled by speaker group** [DOC:pg §10].

### E.4 Sanity and shortcut controls [DOC:pg §08/§09 + REC]
Overfit 32 clips; label-swap test; silence/duration-bucket analysis (leading/trailing silence is a known ASVspoof shortcut); **corpus-probe**, i.e. can a classifier separate bona fide OpenSLR from bona fide Kathbath? If so, report per-source metrics; splice-only probe for C3.

### E.5 Compute budget [HYP]
T4 at fp16, frozen XLS-R, 4 s crops: roughly tens of clips/s forward. With 20k-clip epochs × ~20 epochs × 3 seeds × ~12 configurations, this fits within Kaggle's weekly GPU quota spread over ~6 weeks **only if** training subsets stay in the low tens of thousands. Measure throughput in Phase 4 before committing to sizes.

---

## F. Repository architecture

Merges DOC:syn 4.4.1 and DOC:pg §13.

```
VaaniShield/                      (Git repo root; source PDFs stay out of Git or in docs/sources/ via LFS — decide)
├── PROJECT_STATE.md              persistent source of truth
├── README.md  MODEL_CARD.md  DATASET_CARD.md  LICENSES_REGISTER.md  CONSENT_PROTOCOL.md
├── pyproject.toml  requirements-lock.txt  .gitignore  .pre-commit-config.yaml
├── docs/                         plans, decisions (ADR-style), literature matrix
├── ml/
│   ├── configs/                  one YAML per experiment (pydantic-validated)
│   ├── data_scripts/             download notes, manifest builders, QC, split, real-app pairing
│   ├── src/vaanishield/
│   │   ├── io/                   decode, resample, QC
│   │   ├── data/                 manifest schema, validator, leakage checks, Dataset classes
│   │   ├── augment/              codec (ffmpeg), rawboost, langmix
│   │   ├── features/             LFCC
│   │   ├── models/               lcnn, aasist (vendored @ pinned commit + LICENSE), xlsr_frontend, heads
│   │   ├── train/  eval/         trainer, metrics (EER, AUC, ECE, t-DCF, bootstrap), calibration
│   │   └── infer/                score(waveform, sr) + windowing
│   ├── experiments/<run_id>/     config snapshot, git commit, manifest hash, metrics.json, scores.csv (no weights)
│   └── notebooks/                exploration only; Colab launcher notebook
├── ml-service/app/               FastAPI + Gradio demo
├── client/                       React/Vite (later, optional)
├── paper/{figures,tables}/       tables generated from experiments/, "not measured" by default
└── tests/                        unit + integration (manifest, leakage, metrics, score direction, API)

Outside Git:  VAANI_DATA_ROOT=…/vaanishield-data/{raw,processed,manifests-private,consent}
              VAANI_CKPT_ROOT=…/checkpoints   (hash recorded in run manifest)
```

Tracking [REC]: the file-based run registry above is the source of truth. Weights & Biases or MLflow is an optional mirror, with credentials only in `.env`, which is gitignored.

Environment [REC]: Python **3.12** venv. The laptop default is 3.14, which may not have wheels for torch/torchaudio and other audio libraries yet. Also needed: ffmpeg with libopus/libmp3lame/libopencore-amrnb, which is **not installed** on the laptop; Colab/Kaggle images include ffmpeg, but AMR support must be checked.

---

## G. Phased implementation plan

Each phase ends with the 8-item report [user rule] and a PROJECT_STATE.md update. Do not cross a gate until it passes.

| Phase | Window (2026–27) | Work | Exit gate |
|---|---|---|---|
| **0** Plan | Oct 7 | This document, PROJECT_STATE.md | Team review of discrepancies D1–D13 and decisions Q1–Q8 |
| **1** Foundation | Oct 8–17 | git init, env lock, .gitignore, pre-commit, CI (tests only), licence register, consent form draft, literature matrix incl. SLS/RTCFake/ADD-C/ICF | Fresh clone installs and passes tests |
| **2** Manifest + leakage | Oct 13–24 | Schema, validator, speaker-graph splitter, leakage assertions, ASVspoof LA manifest | Validator reports 0 broken paths and 0 leakage on LA + synthetic toy cases |
| **3** Metrics + preprocessing | Oct 20–31 | Preprocessing contract, EER/AUC/F1/ECE/min-t-DCF/bootstrap with tests against the official ASVspoof eval script on the same scores | Metric parity test passes |
| **4** Baselines | Nov 1–21 | B0 LFCC-LCNN train on LA; B1 AASIST official checkpoint inference, then own training; throughput measurement | Results on LA reproduce across 2 seeds; score direction verified; failures documented |
| **5** Minimal demo | Nov 15–25 | FastAPI + Gradio over B1; response contract [DOC:pg §12]; no retention | Fresh clone runs demo; invalid audio fails safely |
| **6** Indic data | Nov 10–Dec 5 | OpenSLR 104 inspection (20 clips), Kathbath-Hi + IndicSynth-Hi subsets, speaker graph, 200-clip pilot | Dataset card + listening audit; leakage report clean |
| **7** HinglishFake-Test | Nov 20–Dec 20 (parallel) | Prompts, consented recordings, generation, sim codecs, real-app round-trips, **test freeze** | Frozen, hashed test manifest; consent records complete |
| **8** Proposed model | Dec 1–Jan 5 | P1, P2, codec/RawBoost/language-mix aug, temperature scaling, layer-weight logging | Best checkpoint chosen by dev EER only |
| **9** Experiments | Jan 5–Feb 7 | E0–E3, A1–A5, 3 seeds, calibration, latency, C2 channel-gap table | Every predefined cell is measured or marked unavailable |
| **10** Contribution + analysis | Jan 20–Feb 20 | C4 channel-conditioned calibration, C5 switch-point analysis, error analysis, compact model | Pre-registered comparison done on frozen tests |
| **11** Release + writing | Feb–Mar | Paper, model/dataset cards, demo video, **IPR-cell disclosure before any public posting** | Independent teammate reproduces one key table |

Team split for 4 members [REC]:
- **M1** data and governance: licences, consent, manifests, leakage.
- **M2** baselines and models.
- **M3** augmentation and real-app channel protocol.
- **M4** evaluation, demo and writing infrastructure.

Weekly rhythm per DOC:pg §14.

---

## H. External sources checked (2026-10-07)
- Indic-CodecFake meets SATYAM — https://aclanthology.org/2026.findings-acl.2159/ , https://arxiv.org/abs/2604.19949
- SLS (XLS-R + layer selection) — https://dl.acm.org/doi/10.1145/3664647.3681345
- ADD-C — https://arxiv.org/abs/2504.12423 ; Real-world corruption — https://arxiv.org/abs/2503.17577
- RTCFake — https://arxiv.org/abs/2604.23742
- SEA-Spoof — https://arxiv.org/abs/2509.19865
- Indic encoders (22 languages) — https://arxiv.org/abs/2608.12536
- Code-switch AV (ArEnAV) — https://arxiv.org/abs/2505.22581
- XTTS-v2 (CPML) — https://huggingface.co/coqui/XTTS-v2 ; Hinglish XTTS — https://huggingface.co/harrrshall/xtts-v2-hinglish-synthetic
- ASVspoof 2019 — https://datashare.ed.ac.uk/handle/10283/3336
- OpenSLR 104 — https://www.openslr.org/104/
- In-the-Wild — https://deepfake-total.com/in_the_wild
- IndicSynth HF — https://huggingface.co/datasets/vdivyasharma/IndicSynth ; IndicSUPERB — https://github.com/AI4Bharat/IndicSUPERB
