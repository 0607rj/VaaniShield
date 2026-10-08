# Research gap, novelty classification, RQs and hypotheses

Audit date 2026-10-07. Evidence comes from `literature_matrix.md` (paper #) and `references.md` (R##).

> **Standing rule.** Not finding a paper is **not** evidence of novelty. Our searches were targeted web searches plus full reading of ~9 papers, not a systematic review. Every "underexplored" or "potentially novel" label below is provisional. Each must be re-checked by a structured search before the paper is submitted and before any patent disclosure: Google Scholar, ACL Anthology, ISCA Archive, IEEE Xplore and arXiv, with the search strings in §7.

---

## 1. What the field already covers

| Area | Established by | Status |
|---|---|---|
| English-trained detectors fail on Indic languages | IndicSynth #15, ICF/SATYAM #17, SEA-Spoof #18, Indic encoders #19 | **solved / well documented** |
| Indic synthetic-speech datasets (TTS/VC; NAC resynthesis) | IndicSynth #15, ICF #17, SEA-Spoof #18 | **solved** |
| Cross-lingual and language-family transfer within Indic | ICF #17 | **solved** for NAC fakes |
| Unseen-generator / unseen-codec evaluation | ASVspoof 2019/2021 #1–2, ICF #17 | **solved** as a protocol |
| XLS-R + AASIST + RawBoost recipe | Tak 2022 #6, RTCFake #27 uses it as standard | **solved** |
| Learned weighting or selection of SSL layers | SLS #8, Serrano #11 | **solved** |
| Multilingual PTMs beat monolingual for ADD | Phukan #10, Pascu #9 | **solved** (English benchmarks) |
| Codec/compression hurts ADD; codec augmentation helps | ASVspoof 2021/5 #2–3, ADD-C #25, corruption study #26, RADAR #28 | **solved** (English, simulated) |
| Evaluation under *real* transmission | ASVspoof 2021 LA (real VoIP/PSTN) #2, RTCFake (7 real platforms, paired) #27, ML-ITW (found platform audio) #20 | **partially solved** |
| Calibration / threshold transfer under domain shift | ASVspoof 5 #3, Pascu #9, Zhou & Wang #31 | **partially solved** |
| Segment-level spoof scoring | PartialSpoof #29 | **solved** (as a technique) |
| Splice artefacts as shortcuts | Negroni #30, silence shortcut #13 | **known risk** |
| Code-switched deepfake datasets | ArEnAV (Arabic-English, AV) #21 | **partially solved** (other language pair / modality) |

## 2. Classification of each proposed VaaniShield contribution

Sources of the proposals: synopsis §1.4/§5.1, project guide §02, and our C1–C5 from `docs/PLAN_v0.md`.

| # | Proposed contribution | Classification | Evidence / reason |
|---|---|---|---|
| P-a | Audio deepfake detection for Hindi / Indic languages | **already solved** | #15, #17, #18 |
| P-b | Showing English-trained detectors degrade on Hindi | **already solved** | #15 Table 2, #17, #18 |
| P-c | XLS-R-300M front end + AASIST(-lite) back end | **already solved** | #6 (fine-tuned), #27 |
| P-d | Learned softmax-weighted layer aggregation | **already solved** | #8, #11 |
| P-e | RawBoost augmentation | **already solved** | #5, #6 |
| P-f | Opus/AMR/MP3 codec augmentation for robustness | **already solved** (English, simulated) | #25, #26, #3 |
| P-g | Temperature scaling for calibrated probabilities | **already solved** (technique) | #3, #9, #31 |
| P-h | Segment-level suspicion timeline in a demo | **already solved** (technique) | #29 |
| P-i | Leave-one-generator-out testing | **already solved** (protocol) | #1, #17 |
| P-j | Speaker-disjoint splits, leakage checks | standard practice, **not a contribution** | – |
| P-k | Compact detector via layer truncation / small head | **partially solved** | #4 (AASIST-L), #11 (80 % parameter reduction) |
| P-l | Evaluation under real transmission channels | **partially solved** | #2 (real telephony), #27 (real RTC platforms, paired), #20 |
| P-m | Calibration under channel shift | **partially solved** | #31 (threshold transfer), #3 |
| **C1** | Speaker-disjoint, consented **Hindi-English code-switched** ADD benchmark (HinglishFake-Test) with generator-family holdout | **underexplored** | Code-switched deepfakes exist for Arabic-English AV (#21). Hindi appears in multilingual sets (#15, #18) but monolingual. CS-FLEURS has Hindi-X **TTS only** (#22). No Hinglish audio-ADD benchmark found. |
| **C2** | **Paired** clean / simulated-codec / **real messaging-app store-and-forward** comparison on Hinglish, measuring whether simulation is a valid proxy (ΔEER, Δfalse-alarm, ΔECE) | **underexplored** (in this exact form). The broader idea is **partially solved** | #27 pairs offline/online for real-time calls on 7 platforms (English/Chinese); #2 uses real telephony. Asynchronous voice-note/file re-encoding (WhatsApp-style), Hinglish, and an explicit *simulation-validity* test were not found. |
| **C3** | **Symmetric** code-switch splice augmentation with shortcut controls | **unclear** | No code-switch augmentation for ADD found, but splice artefacts are a known shortcut (#30). Its value depends entirely on our results. |
| **C4** | **Channel-conditioned calibration**: estimate the channel condition from audio, then apply condition-specific temperature and threshold (fit on simulated, test on real-app) | **unclear** | #31 audits global/unlabeled calibration under shift; quality-aware calibration is common in biometrics (not searched). "Conditioned on estimated messaging channel" was not found, but prior art likely exists in adjacent fields. |
| **C5** | Segment scores analysed against **language-switch points** | **underexplored** (as an analysis) | No ADD work found relating segment scores to code-switch boundaries. Exploratory only. |
| **C-combo** | The **combined** setting: Hinglish code-switching × real messaging re-encoding × unseen generator family × calibration, with a compact multilingual detector, under one reproducible speaker-disjoint protocol | **potentially novel** (as an evaluation setting) | Each factor is covered somewhere; their intersection on Hindi-English was not found. This is the claim the project guide recommends [DOC:pg §02]. |

No item is classified as "potentially novel" as a *method*. That is intentional.

## 3. What VaaniShield must NOT claim as novel

1. "First Indic / Hindi audio deepfake detection work or dataset." (IndicSynth, ICF, SEA-Spoof, HAV-DF exist.)
2. "We show detectors trained on English fail on Hindi." (Already shown; we may *replicate* it, clearly labelled.)
3. "Novel XLS-R + AASIST architecture", "novel learned layer weighting", "novel AASIST-lite". (Tak 2022, SLS, AASIST-L.)
4. "Novel use of RawBoost / codec augmentation to improve compression robustness." (RawBoost, ADD-C, corruption study, ASVspoof 5.)
5. "First evaluation of deepfake detectors on real communication channels / real platforms." (ASVspoof 2021 LA real telephony; RTCFake real platforms incl. Telegram.)
6. "First paired offline-vs-transmitted deepfake dataset." (RTCFake.)
7. "First calibrated deepfake detector" or "first to study threshold transfer". (ASVspoof 5, Pascu 2024, Zhou & Wang 2026.)
8. "Novel segment-level localisation / explanation." (PartialSpoof family.) The timeline is a UI aid, not an explanation [DOC:syn 4.2.4].
9. "First code-switched deepfake dataset." (ArEnAV is Arabic-English.) The most we could say is "to our knowledge, the first *Hindi-English* code-switched *audio* ADD benchmark", and only after the systematic search.
10. "Unseen-generator generalisation" when the held-out generator shares a family with training (XTTS in IndicSynth, Hinglish-XTTS and CS-FLEURS).
11. "Robust" or "works on WhatsApp" in general. Only the specific app/version/date/send-mode conditions we measured can be claimed.
12. Any accuracy claim that is not our own measured number. Any "forensic", "courtroom" or "100 %" claim.

## 4. Research questions (final, replaces the 5-RQ draft)

These merge synopsis SRQ1–5 and guide RQ1–4 into four.

- **RQ1 — Language / code-switch shift.** How much does detection performance degrade from English → Hindi → Hindi-English code-switched speech for detectors trained on English and on English+Hindi data? Does *symmetric* code-switch splice augmentation reduce the Hinglish gap without creating a splice shortcut?
- **RQ2 — Messaging channel.** How closely do simulated codec transforms (Opus, AMR-NB, MP3, resampling) reproduce the effect of *real* messaging-app store-and-forward re-encoding on (a) discrimination and (b) bona fide false alarms? Does codec augmentation transfer from simulated to real-app conditions?
- **RQ3 — Representation and generator generalisation.** Under the same protocol, does a multilingual SSL front end (XLS-R-300M with learned layer weighting) generalise better than LFCC-LCNN, AASIST and a monolingual SSL front end to *unseen generator families* on Hindi and Hinglish?
- **RQ4 — Deployment reliability.** Under language and channel shift, how far do calibration (ECE) and dev-fixed thresholds (false-alarm/miss rates, HTER) degrade? Does channel-conditioned calibration reduce that degradation relative to a single global temperature, at acceptable CPU latency?

## 5. Hypotheses

General test procedure for every hypothesis:
- Splits and thresholds are frozen before testing.
- Report the mean over ≥3 seeds.
- Compute 95 % CIs by **bootstrap over speaker groups**, paired when two systems are compared on the same test clips.
- "Supported" means the CI of the stated difference excludes 0 in the stated direction. Otherwise the result is "not supported" and is reported anyway.
- Pre-register the margins below in `docs/` before Phase 9.
- Score convention: positive class = spoof.
  - "**False alarm**" = bona fide classified as spoof.
  - "**Miss**" = spoof classified as bona fide.

| ID | RQ | Hypothesis (falsifiable) | Experiment / comparison | Primary measure |
|---|---|---|---|---|
| H1 | RQ1 | For ASVspoof19-trained B1 (AASIST) and P2, EER(Hinglish-test, clean) > EER(Hindi-test, clean) > EER(LA eval). | E1, E3 trained on `T_en`; three test sets | EER, paired speaker-bootstrap differences |
| H2 | RQ1 | Adding Hindi training data (`T_en+hi`) lowers Hinglish-test EER vs `T_en`, for P2. | E3 `T_en` vs `T_en+hi` | ΔEER (Hinglish) |
| H3 | RQ1 | Symmetric language-mix augmentation lowers Hinglish-test EER vs no language-mix (A2), **and** a splice-only probe on the augmented training data has AUC within 0.50 ± 0.05. | A2; splice-only probe classifier (e.g. boundary-energy features → logistic regression) | ΔEER; probe AUC |
| H4 | RQ1 (control) | An *asymmetric* splice design (bona fide cross-speaker, spoof same-voice) gives a probe AUC above 0.60, i.e. it would create a shortcut. | Probe-only experiment, no detector training needed | probe AUC |
| H5 | RQ2 | For models trained **without** codec augmentation, the bona fide false-alarm rate at the dev-fixed threshold is higher on real-app re-encoded bona fide clips than on the same clips clean. | E1/E3-A1 on paired HinglishFake-Test real-app subset | Δfalse-alarm rate (paired) |
| H6 | RQ2 | Simulated codecs are a valid proxy for real-app re-encoding: across models/configurations, the real-app EER is closer to the matched simulated-codec EER than to the clean EER (|EER_real − EER_sim| < |EER_real − EER_clean|), and Spearman ρ(EER_sim, EER_real) ≥ 0.7. | All E/A configurations on paired subsets | gap difference; ρ with bootstrap CI |
| H7 | RQ2 | Codec augmentation lowers real-app EER vs no codec augmentation (A1), with clean-test EER non-inferior within +1.0 pp absolute. | A1 | ΔEER(real-app); non-inferiority on clean |
| H8 | RQ3 | P1/P2 (XLS-R) achieve lower EER than B0 (LFCC-LCNN) and B1 (AASIST) on the unseen-generator-family fold, on Hindi and Hinglish tests. | E0–E3, A5 leave-one-family-out | ΔEER per fold |
| H9 | RQ3 | Multilingual XLS-R gives lower Hindi/Hinglish EER than an English-only wav2vec 2.0 front end of comparable size with an identical head and recipe. | A3 | ΔEER |
| H10 | RQ3 | Learned layer weighting gives EER ≤ final-layer-only features (non-inferior within +0.5 pp) on Hindi/Hinglish. | A3 variant | ΔEER (non-inferiority) |
| H11 | RQ4 | With a single temperature fitted on clean dev, ECE and HTER at the dev threshold are higher on real-app conditions than on clean. | P2 calibrated on clean dev | ΔECE, ΔHTER |
| H12 | RQ4 | Channel-conditioned calibration (fitted on *simulated* dev conditions only) lowers real-app ECE and HTER vs a single global temperature fitted on the same dev pool. EER is unchanged by construction [R53], so it is not used here. | C4 vs global temperature scaling | ΔECE, ΔHTER, Δfalse-alarm |
| H13 | RQ4 | A layer-truncated XLS-R model (cut above the highest-weight layer) has p95 CPU latency ≤ 50 % of the full model for a 10 s clip on the named demo CPU, with Hinglish EER non-inferior within +1.0 pp. | Compact-model ablation | latency p95; ΔEER |
| E1 (exploratory) | RQ1 | Segment spoof scores differ between windows that contain a language-switch point and windows that don't. | C5; LID-based switch detection | effect size + CI. **Reported as exploratory, no confirmatory claim.** |

Notes:
- Thresholds 0.60 / ±0.05 / 0.7 / +1.0 pp / 50 % are **proposed pre-registration values**. The team should confirm or change them before Phase 9 and never after seeing test results.
- H6 needs enough model configurations (≥8) to make ρ meaningful. If fewer exist, report the gap difference only.
- HinglishFake-Test size limits power. If CIs are wide, report "inconclusive"; do not reduce the claim to a point estimate.

## 6. Novelty Boundary

**Question: what exactly are we adding beyond IndicSynth, Indic-CodecFake/SATYAM, AASIST, XLS-R-based detectors and codec-robustness studies?**

| Prior work | What it already gives | What VaaniShield adds (if experiments succeed) | What VaaniShield does **not** add |
|---|---|---|---|
| **IndicSynth** [#15] | Monolingual Indic TTS/VC fakes, incl. Hindi (XTTS-v2, FreeVC24); English-trained detectors fail on them | Uses IndicSynth as *training* data (they report no detector trained on it); evaluates transfer from monolingual Hindi to **code-switched Hinglish**; adds channel conditions and calibration | A new Indic synthetic corpus; the "English detectors fail" finding |
| **Indic-CodecFake / SATYAM** [#17] | Indic **neural-codec resynthesis** fakes; unseen-codec, cross-lingual, language-family evaluation; heavy ALM detector | (a) **Code-switching** within an utterance (ICF is monolingual per language). (b) **Traditional codecs as a *channel* applied to both classes**, not neural codecs as the *attack*; we test whether detectors confuse the two (H5). (c) Real messaging round-trips. (d) Calibration and CPU-latency reporting for a compact model | A better Indic detector than SATYAM on ICF (we don't evaluate on ICF unless added later); NAC-fake detection |
| **AASIST / AASIST-L** [#4] | Graph-attention back end; lightweight variant | Used as baseline (B1) and head (AASIST-lite); evaluated under language and channel shift | Any architectural change claimed as novel |
| **XLS-R-based detectors** (Tak 2022, SLS, Pascu, Phukan, Serrano) [#6, #8–11] | XLS-R+AASIST+RawBoost recipe; layer weighting; frozen-SSL generalisation; multilingual PTMs help | Measures whether these English-benchmark findings **hold on Hindi-English code-switched and app-re-encoded speech** (RQ3, H8–H10). Reports layer-weight profiles under Hinglish training as analysis | A new SSL front end or pooling method |
| **Codec / channel robustness** (ASVspoof 2021/5, ADD-C, corruption study, RTCFake, RADAR) [#2, #3, #25–28] | Simulated codec benchmarks and augmentation; real telephony; real RTC-platform paired data (EN/ZH) | (a) **Store-and-forward messaging re-encoding** (voice note vs file vs document send modes, with verified re-encoding). (b) **Explicit test of whether simulation is a valid proxy** for real-app processing, on paired clips (H6). (c) **Bona fide false-alarm** and **calibration** under real-app conditions (H5, H11, H12). (d) For **Hinglish** | "Codec augmentation helps"; "real platforms hurt detectors"; a paired real-platform dataset as such |
| **CS-FLEURS** (byan/cs-fleurs, CC BY-NC 4.0; built for code-switched ASR) | Real *and* synthetic (XTTS, MMS concatenative) **Hindi-English code-switched** speech on FLEURS text | Uses it as a **spoof-detection** resource (real vs synthetic, generator-family holdout); checked 2026-10-08 that its Hindi-English read subset is human speech | "First synthetic Hinglish speech"; whether CS-FLEURS was already used for deepfake detection is **not yet searched** (§7) |
| **Code-switched deepfakes** (ArEnAV) [#21] | Arabic-English AV deepfakes | Hindi-English, **audio-only**, consented, speaker-disjoint, with channel conditions | "First code-switched deepfake dataset" |
| **Calibration / threshold transfer** [#31, #3, #9] | Global and unlabeled calibration under shift audited; EER invariant to monotone transforms | **Condition-conditioned** calibration using an estimated channel label, fitted on simulated conditions and tested on real-app ones (H12) | "Calibrated ADD" as a new idea |

**One-sentence boundary (for the paper):**
> VaaniShield does not propose a new detector architecture. It contributes a consented, speaker-disjoint Hindi-English code-switched evaluation resource and protocol that jointly varies language mixing, generator family, and *paired* clean / simulated / real messaging-app channels. On it, we measure whether established multilingual-SSL detectors and standard codec augmentation transfer. We also test whether channel-conditioned calibration reduces deployment errors that EER hides.

## 7. Required follow-up before any novelty wording (Phase 1 task)

Systematic search, logged in `research/search_log.md` (to create): databases, date and hit counts for
- `("code-switch*" OR "code-mix*" OR Hinglish) AND (spoof* OR deepfake OR "anti-spoofing")`
- `(WhatsApp OR "messaging app" OR "voice note" OR "voice message") AND (deepfake OR spoof*) AND (detect*)`
- `(calibration OR "threshold transfer") AND (spoof* OR deepfake) AND (codec OR channel OR "domain shift")`
- `("quality-aware" OR "quality-dependent") AND calibration AND (biometric OR "speaker verification")` (prior art for C4)
- Plus forward citations of IndicSynth, ICF/SATYAM, RTCFake, ADD-C, **CS-FLEURS** (has it been used for spoof detection?).

Also check patent databases (InPASS / Google Patents) for C4-style "channel estimation → conditional calibration" claims before any IPR discussion.

## 8. Evidence from the 200-clip Hinglish pilot (2026-10-08): protocol lessons, not results

The pilot (`DATASET_CARD.md` § Hinglish pilot) was run to test the protocol. Its numbers come from 60 training clips
and test sets of 20 and 100 clips. **They are not results and must not be quoted as findings.** What they show about
the *design*:

1. **Shortcut floors are needed per test condition.**
   - Trivial features (duration, peak, RMS, leading silence) reach EER 0.00 on the seen-generator condition (T1).
   - A detector that scores 0.00 there (the pilot LFCC-CNN did) has shown nothing.
   - Rule: every confirmatory comparison (H1–H10) must report the trivial-feature probe EER on the same test clips. A
     condition where the probe is near 0 cannot support a claim. This extends the H3/H4 probe idea from splicing to
     level, duration and source.
2. **Generator-family holdout is informative even at pilot scale.**
   - The LFCC-CNN separated XTTS from both real sources (T1 and cross-condition EER 0.00).
   - On the held-out MMS-concatenative family it ranked fakes as *more* real than real speech (T2 EER 0.72, AUC 0.24).
   - This is the failure mode H8 asks about. The pilot only shows the protocol can expose it; H8 itself is untested.
3. **Single-source-per-class training data is the main risk for RQ1.**
   - In the pilot (train real = OpenSLR, train fake = CS-FLEURS XTTS) and in the Hindi pool (Kathbath vs
     IndicSynth), source identity equals the label in train.
   - Real and fake from the *same* recording conditions (team recordings + clones of the same consented speakers, as
     planned) remain necessary to answer RQ1 without this confound.
4. **Level normalisation is a protocol decision, not a tuning knob.**
   - XTTS output is peak-normalised.
   - Whether to loudness-normalise both classes before training must be fixed on dev/probe evidence before Phase 9,
     never chosen from test results.
