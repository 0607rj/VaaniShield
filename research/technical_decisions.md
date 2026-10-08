# Technical decisions derived from the research audit

Audit date 2026-10-07. Each decision has an ID, the evidence behind it, and a status:
- **proposed**: awaiting team review
- **adopted**: agreed
- **open**: needs data or a team decision

Decisions DEC-1…8 in `PROJECT_STATE.md` still stand. This file adds TD-1…TD-20.

## A. Model

| ID | Decision | Evidence | Status |
|---|---|---|---|
| TD-1 | The architecture (XLS-R-300M + weighted layers + AASIST-lite + RawBoost) is a **re-implemented baseline**, never a claimed contribution | Tak 2022 [R12], SLS [R14], AASIST [R10] | proposed |
| TD-2 | Keep the **frozen** encoder as the main configuration (synopsis 4.2.4). Partial unfreezing (top 2–4 blocks) is an ablation. Tak 2022 fine-tuned the whole of XLS-R (batch 14, lr 1e-6, 100 epochs, RTX 3090), which is impractical on a T4 | R12 FT; R15 (frozen SSL generalises); DOC:pg §08 | proposed |
| TD-3 | Use 64,600-sample (~4.04 s) crops for comparability with AASIST/Tak/SLS code. Hindi IndicSuperb clips average 2.65 s [R20], so padding must be identical for both classes; repeat-padding as in the AASIST code is preferred to zero-padding (to verify in code) | R12 FT; R20 FT | proposed |
| TD-4 | Layer-weight logging and a final-layer-only ablation (H10). The compact demo model is chosen by learned weights (H13). Cite Serrano [R17] for precedent | R14, R17 | proposed |
| TD-5 | ≥3 seeds per cell, mean and best both reported. AASIST documents EER varying 1.19–2.06 % across seeds for a baseline | R10 FT | adopted (matches DOC:syn) |
| TD-6 | Report parameter count, peak GPU memory and CPU latency (median/p95) for every model. SATYAM reports ~6.5–8.2 s per clip on an A100, so a latency comparison is a fair secondary axis | R22 FT; DOC:pg §08 | proposed |

## B. Data and protocol

| ID | Decision | Evidence | Status |
|---|---|---|---|
| TD-7 | Primary Hindi bona fide = Kathbath/IndicSUPERB, the same source corpus as IndicSynth (and ICF). IndicSynth metadata gives target and VC-source speaker IDs, which feed the speaker-graph split | R20 FT; R22 FT | proposed |
| TD-8 | Keep IndicSUPERB's official test-known/test-unknown structure available, so Hindi results can later be aligned with ICF's protocol. Note that IndicSynth fakes were generated from randomly chosen IndicSuperb clips, so we must re-derive speaker-disjoint splits rather than rely on IndicSUPERB's splits | R22 FT ("official IndicSUPERB splits"); R20 FT | open |
| TD-9 | **CS-FLEURS cannot supply real Hinglish.** Per the dataset card, Hindi appears only in the synthetic XTTS-Test2 subset (CC BY 4.0 NC). Real Hinglish must come from OpenSLR 104 and consented recordings | R31 | proposed (corrects the dataset guide's tentative Read-Test idea; re-confirm on download) |
| TD-10 | `generator_family` is mandatory. XTTS appears in IndicSynth-Hi, Hinglish-XTTS, CS-FLEURS XTTS-Test2 and SEA-Spoof. "Unseen" claims require a family absent from training, e.g. FreeVC (IndicSynth), F5-TTS, VITS-MMS, or a commercial system in SEA-Spoof (licence permitting) | R20, R23, R31 | proposed |
| TD-11 | Consistent VAD/silence policy for both classes, plus a silence-ablated evaluation | R50 | adopted |
| TD-12 | Language-mix splice design must be **symmetric**: both classes spliced cross-voice with the same cross-fade/loudness pipeline. A splice-only probe must pass (H3/H4) before any A2 result is interpreted | R52 | proposed |

## C. Channel / codec

| ID | Decision | Evidence | Status |
|---|---|---|---|
| TD-13 | Training codec augmentation uses Opus (several speech bitrates), MP3 (low–mid), AMR-NB or an honestly labelled "AMR-like" narrowband, and resampling. Codec list aligned with ADD-C/ASVspoof 5 where tooling allows (AMR-WB, Speex-WB, Opus). EVS/IVAS are optional only if a reference encoder is available | R44 FT, R03 | proposed |
| TD-14 | Hold out at least one simulated codec/bitrate from training for a held-out-codec condition [DOC:pg §09]. Real-app conditions are **never** used in training except in A4, and then only from speakers outside all test sets | DOC:pg §09/§11 | proposed |
| TD-15 | The real-app protocol distinguishes send modes (in-app voice note recorded from playback, audio file shared as media, file shared as document). Re-encoding is verified per file with `ffprobe` and hashes. The WhatsApp codec/bitrate is **measured**, not assumed: only blog sources were found | references.md (non-authoritative) | proposed |
| TD-16 | Keep the distinction between a neural codec as *generator* and a conventional codec as *channel* explicit. Never label codec-processed bona fide as spoof. Optionally add a small ICF Hindi subset as an extra *spoof* family test (CC BY 4.0) | R22, R40–R43 | proposed |
| TD-17 | Optional extra external test sets, after the pipeline is frozen: ASVspoof 2021 LA/DF (real telephony / compression), and RTCFake eval (real platforms, EN/ZH; licence to verify). Never tune on them | R02, R46 | open |

## D. Evaluation and calibration

| ID | Decision | Evidence | Status |
|---|---|---|---|
| TD-18 | EER stays primary [DOC], but every table also reports **false-alarm and miss rates at the dev-fixed threshold**, HTER, ECE (equal-mass bins plus 15-bin), NLL and Brier. EER cannot reveal calibration failures because it is invariant to monotone transforms | R53 | proposed |
| TD-19 | Calibration methods compared: none; global temperature (clean dev); global temperature (dev incl. simulated codecs); channel-conditioned temperature/threshold (C4). The channel estimator is trained on simulated conditions only | R53, R03, R15 | proposed |
| TD-20 | Bootstrap CIs resample **speaker groups**; paired bootstrap for system comparisons; pre-register H1–H13 margins in `docs/` before Phase 9 | DOC:pg §10; research_gap.md §5 | proposed |

## E. Items needing a team decision

- **Q-A:** Add SEA-Spoof Hindi as a training/eval source? It has commercial generators, which are valuable for unseen-family tests, but the licence is reportedly CC BY-NC-ND (verify; ND may forbid derived augmented copies).
- **Q-B:** Add an ICF Hindi subset as a neural-codec spoof family (TD-16)?
- **Q-C:** Is RTCFake worth downloading for an external real-channel test (size and licence unverified)?
- **Q-D:** Are the H1–H13 pre-registration margins acceptable?
