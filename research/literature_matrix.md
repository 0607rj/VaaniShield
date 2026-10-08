# Literature matrix — VaaniShield

Audit date 2026-10-07. Reference IDs (R##) and verification levels (FT/ABS/SEC/DOC) are in `references.md`.

Rules for this file:
- Numbers are **the authors' reported results**, quoted for context only. They are never our results.
- "Not verified" means we did not read that detail; do not fill it in from memory.
- "Overlap" rates how much the paper pre-empts a VaaniShield contribution:
  - **High**: we must not claim this.
  - **Medium**: we must position against it.
  - **Low**: background only.
- C1–C5 are defined in `research_gap.md` §2.

## Quick index

| # | Paper | Year | Theme | Overlap |
|---|---|---|---|---|
| 1 | ASVspoof 2019 LA [R01] | 2020 | benchmark | Low |
| 2 | ASVspoof 2021 [R02] | 2023 | benchmark, real telephony channel | **Medium** (C2) |
| 3 | ASVspoof 5 [R03] | 2026 | benchmark, codecs, calibration | **Medium** (C2, C4) |
| 4 | AASIST [R10] | 2022 | model | **High** (architecture) |
| 5 | RawBoost [R11] | 2022 | augmentation | **High** (augmentation) |
| 6 | Tak et al. wav2vec2/XLS-R+AASIST [R12] | 2022 | model | **High** (architecture) |
| 7 | XLS-R [R13] | 2022 | SSL front end | Low |
| 8 | SLS [R14] | 2024 | layer weighting | **High** (weighted layers) |
| 9 | Pascu et al. [R15] | 2024 | SSL + calibration | **Medium** (C4) |
| 10 | Phukan et al. [R16] | 2024 | multilingual PTMs | **High** (RQ3 premise) |
| 11 | Serrano et al. [R17] | 2025 | layer selection / compact | **Medium** (compact model) |
| 12 | In-the-Wild [R04] | 2022 | real-world eval | Low |
| 13 | Silence shortcut [R50] | 2021 | shortcut | Low (control) |
| 14 | MLAAD [R05] | 2024 | multilingual data | Medium (RQ1) |
| 15 | IndicSynth [R20] | 2025 | Indic data | **High** (Indic ADD) |
| 16 | HAV-DF [R21] | 2024 | Hindi AV data | Low |
| 17 | Indic-CodecFake / SATYAM [R22] | 2026 | Indic codec-fakes | **High** (Indic ADD, unseen codec) |
| 18 | SEA-Spoof [R23] | 2025 | multilingual incl. Hindi | **Medium** (Hindi ADD) |
| 19 | Indic encoders study [R24] | 2026 | Indic SSL encoders | Medium (RQ3) |
| 20 | ML-ITW [R07] | 2026 | multilingual in-the-wild | Medium (realism) |
| 21 | ArEnAV [R30] | 2025 | code-switched AV deepfakes | **Medium** (C1, other language) |
| 22 | CS-FLEURS [R31] | – | code-switched speech/TTS | Low (resource) |
| 23 | CodecFake family [R40–R42] | 2024–25 | neural-codec fakes | Medium (conceptual) |
| 24 | Proxy-to-wild gap [R43] | 2026 | codec-fake generalisation | Low |
| 25 | ADD-C [R44] | 2025 | simulated comm. channel | **High** (codec aug.) |
| 26 | Real-world corruption [R45] | 2025 | corruption robustness | **High** (codec aug.) |
| 27 | RTCFake [R46] | 2026 | real platform transmission | **High** (C2 partially) |
| 28 | RADAR 2026 [R06] | 2026 | media transformations | Medium (C2) |
| 29 | PartialSpoof [R51] | 2021–23 | segment-level | Medium (C5, timeline) |
| 30 | Splicing artifacts [R52] | 2024 | splice shortcut | **Medium** (C3 risk) |
| 31 | EER hides deployment failure [R53] | 2026 | threshold transfer, calibration | **High** (C4) |

---

## 1. ASVspoof 2019 LA [R01] — DOC
- **Year/venue:** 2020, Computer Speech & Language 64.
- **Problem:** standard logical-access spoof detection benchmark.
- **Dataset:** VCTK-derived bona fide speech; TTS/VC attacks. Per R12 (FT), the training data covers six attack algorithms; the eval set covers 13 attacks [R10 FT].
- **Model:** baselines (LFCC-GMM etc.); details not verified here.
- **Training/eval:** fixed train/dev/eval protocols; EER and min t-DCF.
- **Contribution:** reproducible English benchmark.
- **Limitation:** English, clean, fixed attacks. Silence-duration bias between classes [R50].
- **Relevance:** baseline training/eval (E0/E1), metric-parity tests.
- **Overlap:** none. Used as infrastructure.

## 2. ASVspoof 2021 [R02] — SEC
- **Year/venue:** 2023, IEEE/ACM TASLP 31.
- **Problem:** spoof/deepfake detection under channel and compression variability.
- **Dataset:** LA eval transmitted through **real telephony systems (VoIP and PSTN)**, with codecs incl. a-law, µ-law, G.722 and 8/16 kHz bandwidths. DF eval: ~600K utterances processed with commercial audio codecs, from VCTK plus VCC 2018/2020 sources (R12 FT confirms the DF sources and ">100 algorithms").
- **Model:** challenge baselines.
- **Training setup:** trained on ASVspoof 2019 LA only, so channel conditions are unseen at training time.
- **Evaluation:** EER, min t-DCF.
- **Contribution:** first large ASVspoof edition with real transmission channels.
- **Limitation:** English. Telephony, not messaging-app voice notes. No code-switching.
- **Relevance:** the precedent for "real channel" evaluation; an optional extra test set.
- **Overlap:** **Medium with C2.** "Evaluating under real transmission" is not new. Our difference is messaging-app store-and-forward re-encoding, Hinglish, and *paired* simulated-vs-real comparison.

## 3. ASVspoof 5 [R03] — ABS+SEC
- **Year/venue:** IEEE TASLP (accepted; arXiv Jan 2026); resources paper in CSL 95 (2026).
- **Problem:** spoofing, deepfake and adversarial attack detection with crowdsourced speech.
- **Dataset:** crowdsourced English data. Per secondary sources, the eval set includes ~11 codec conditions (opus, amr, speex, mp3, m4a, Encodec, bandwidth reduction); exact list not verified.
- **Model:** 53 team submissions.
- **Training setup:** challenge protocol. Calibrated scores/LLRs were encouraged and **a calibration study is reported** (ABS).
- **Evaluation:** EER, min DCF/t-DCF variants (not verified), calibration.
- **Contribution:** codec-aware large benchmark with a calibration focus.
- **Limitation:** English; simulated codecs rather than messaging-app round-trips (not verified).
- **Relevance:** codec list for our simulated conditions; calibration methodology.
- **Overlap:** **Medium with C2/C4.** Codec robustness and calibration are both studied, just not for Hinglish or messaging apps.

## 4. AASIST [R10] — FT
- **Year/venue:** 2022, ICASSP.
- **Problem:** a single efficient model to catch spectral and temporal artefacts without ensembles.
- **Dataset:** ASVspoof 2019 LA.
- **Model:** RawNet2-style sinc front end, then a heterogeneous stacking graph attention layer (HS-GAL), max graph operation and stack node. AASIST-L variant has 85K parameters (332 KB).
- **Training setup:** Adam, lr 1e-4, cosine annealing. AASIST-L found by population-based training. Results averaged over 3 seeds.
- **Evaluation:** LA eval pooled EER 1.13 (best seed 0.83) %, min t-DCF 0.0347 (0.0275). AASIST-L EER 0.99 % (best single).
- **Contribution:** strong single-system graph-attention countermeasure; documents seed variance (RawGAT-ST 1.19–2.06 % across seeds).
- **Limitation:** English lab data only; no channel or language shift tested.
- **Relevance:** B1 baseline; AASIST-lite head design; motivation for ≥3 seeds.
- **Overlap:** **High.** AASIST/AASIST-L are not our contribution.

## 5. RawBoost [R11] — ABS (+R12 FT)
- **Year/venue:** 2022, ICASSP.
- **Problem:** data augmentation for channel/nuisance variability without external data.
- **Dataset:** trained on ASVspoof 2019 LA, evaluated on ASVspoof 2021 LA.
- **Model:** raw end-to-end baseline.
- **Training setup:** three augmentations: (i) linear and non-linear convolutive noise, (ii) impulsive signal-dependent additive noise, (iii) stationary signal-independent additive noise.
- **Evaluation:** 27 % relative improvement over the baseline on 2021 LA (ABS). Per R12, configs (i)+(ii) suit LA/telephony and (iii) suits DF/compression.
- **Contribution:** a cheap, standard augmentation.
- **Limitation:** not designed around specific modern codecs or messaging apps.
- **Relevance:** part of our augmentation recipe. R12 shows the best configuration depends on the target channel, which is relevant to A1.
- **Overlap:** **High.** RawBoost usage is not novel.

## 6. Tak et al., wav2vec 2.0 + AASIST + RawBoost [R12] — FT
- **Year/venue:** 2022, Odyssey.
- **Problem:** generalisation to unseen attacks and channels.
- **Dataset:** ASVspoof 2019 LA train/dev; eval on ASVspoof 2021 LA and DF.
- **Model:** **XLS-R 0.3B** front end, **fine-tuned jointly**, FC to 128 dims, then self-attentive aggregation, then AASIST.
- **Training setup:** ~4 s (64,600 samples) inputs; batch 14; lr 1e-6; 100 epochs; 3 seeds; RTX 3090; weighted cross-entropy.
- **Evaluation:** 2021 LA EER 0.82 % (best) / 1.00 % (avg); 2021 DF EER 2.85 % (3.69 %). A simplified back end gives 1.19 % / 4.38 %.
- **Contribution:** SSL front end plus augmentation is complementary; the strongest open recipe.
- **Limitation (authors):** "massively more complex"; whether such footprints suit practical applications "remains to be seen". English only.
- **Relevance:** **our P2 is essentially this recipe** (with a frozen encoder and learned layer weights). It is the main reference implementation.
- **Overlap:** **High.** XLS-R + AASIST + RawBoost cannot be claimed.

## 7. XLS-R [R13] — ABS (+R12 FT)
- **Year/venue:** arXiv 2021; Interspeech 2022.
- **Problem:** cross-lingual SSL speech representations.
- **Dataset:** ~0.5M hours, 128 languages. Sources per R12: VoxPopuli, MLS, CommonVoice, VoxLingua107, BABEL.
- **Model:** wav2vec 2.0 at 0.3B / 1B / 2B parameters.
- **Training setup:** self-supervised pre-training.
- **Evaluation:** ASR, speech translation, LID benchmarks.
- **Contribution:** multilingual front end.
- **Limitation:** not built for spoof detection. **Hindi coverage in pre-training not verified** (to check).
- **Relevance:** P1/P2 front end; RQ3.
- **Overlap:** Low (we use it, we don't claim it).

## 8. SLS — XLS-R + Sensitive Layer Selection [R14] — SEC
- **Year/venue:** 2024, ACM MM.
- **Problem:** exploiting multi-layer XLS-R features.
- **Dataset:** trained on ASVspoof 2019 LA (per third-party card).
- **Model:** XLS-R 300M with **learned weighting of all 24 transformer layers** (per-layer gates), plus a light classifier.
- **Training setup:** a third-party benchmark card says the encoder is frozen with 64,600-sample inputs. **The original paper's setting is not verified** (ACM page returned 403).
- **Evaluation:** numbers seen only on a third-party card; not quoted.
- **Contribution:** layer-weighted XLS-R features for ADD.
- **Limitation:** English training; language and channel shift not studied (not verified).
- **Relevance:** direct precedent for our "learned weighted layer aggregation".
- **Overlap:** **High.** Weighted layer aggregation is not novel.

## 9. Pascu et al. [R15] — ABS
- **Year/venue:** 2024, Interspeech.
- **Problem:** generalisable *and calibrated* ADD with SSL features.
- **Dataset:** benchmark of eight deepfake datasets.
- **Model:** frozen large SSL representations plus logistic regression (<2k trainable parameters); 10 representation types compared.
- **Training setup:** frozen encoder.
- **Evaluation:** average EER 30.9 % (RawNet2) vs 8.8 % (theirs), as reported.
- **Contribution:** frozen SSL + linear head generalises; wav2vec2/WavLM families are best; calibration is addressed (details not verified).
- **Limitation:** Hindi/code-switch/messaging not covered (not verified).
- **Relevance:** supports our frozen-encoder P1; a calibration precedent.
- **Overlap:** **Medium with C4** (calibrated ADD exists).

## 10. Phukan et al., multilingual PTMs [R16] — ABS (synopsis ref [4])
- **Year/venue:** 2024, Findings of NAACL.
- **Problem:** which pre-trained model representations suit ADD.
- **Dataset:** ASVspoof 2019, In-the-Wild, DECRO.
- **Model:** monolingual, multilingual, speaker- and emotion-PTMs with simple downstream networks, plus the MiO fusion.
- **Training setup:** frozen representations with downstream networks.
- **Evaluation:** multilingual PTMs perform best (as reported).
- **Contribution:** evidence that multilingual PTMs help ADD.
- **Limitation:** English/Chinese benchmarks; no Hindi, code-switching or messaging channel.
- **Relevance:** RQ3 premise.
- **Overlap:** **High** for "multilingual SSL beats monolingual" in general. Only the Hindi/Hinglish + channel version is open.

## 11. Serrano et al., layer selection and fusion [R17] — ABS
- **Year/venue:** 2025, arXiv.
- **Problem:** out-of-domain ADD with SSL countermeasures.
- **Dataset:** four out-of-domain test corpora (names not verified).
- **Model:** six SSL encoders; per-layer selection vs MHFA pooling; score fusion.
- **Evaluation:** "selecting the best layer gave very good results, while reducing system parameters by up to 80%."
- **Contribution:** layer selection for compactness and generalisation.
- **Limitation:** English-centric (not verified).
- **Relevance:** precedent for our "truncate XLS-R by learned layer weights" compact demo idea.
- **Overlap:** **Medium.** A compact, layer-truncated detector is not novel by itself.

## 12. In-the-Wild [R04] — ABS
- **Year/venue:** 2022, Interspeech.
- **Problem:** does lab-trained ADD generalise?
- **Dataset:** 37.9 h of found celebrity/politician audio (17.2 h fake); Apache-2.0; 58 speakers.
- **Model:** several re-implemented architectures.
- **Training setup:** trained on ASVspoof 2019.
- **Evaluation:** "performance degradation of up to one thousand percent".
- **Contribution:** real-world generalisation benchmark.
- **Limitation:** English public figures; channel not controlled.
- **Relevance:** external held-out realism check; never tuned on.
- **Overlap:** Low.

## 13. Silence shortcut [R50] — ABS+SEC
- **Year/venue:** 2021, ASVspoof Workshop.
- **Problem:** what ASVspoof-trained models really learn.
- **Dataset:** ASVspoof 2019/2021.
- **Finding:** bona fide clips have longer leading/trailing silences. A model trained on leading-silence length alone gets high accuracy (secondary sources: up to 85 %).
- **Limitation:** English.
- **Relevance:** mandatory control: consistent VAD for both classes and a silence-ablated evaluation.
- **Overlap:** Low (we adopt the control).

## 14. MLAAD [R05] — ABS
- **Year/venue:** 2024, IJCNN; continuously updated.
- **Problem:** multilingual ADD training data.
- **Dataset:** latest arXiv revision: 54 languages, 205 TTS models, 1,153.6 h of synthetic audio (version-dependent). Real speech from M-AILABS (not verified). **Hindi coverage not verified.**
- **Model:** three detectors.
- **Evaluation:** across eight datasets, MLAAD- and ASVspoof19-trained models "alternately outperformed each other, each excelling on four".
- **Contribution:** multilingual multi-generator training data.
- **Limitation:** no controlled code-switching; no messaging channel.
- **Relevance:** optional extra training/eval data; precedent for multilingual training (RQ1).
- **Overlap:** Medium.

## 15. IndicSynth [R20] — FT (`paper2.pdf`)
- **Year/venue:** 2025, ACL (long).
- **Problem:** linguistic bias in ADD; no Indic synthetic data.
- **Dataset:** ~4,000 h synthetic speech in 12 Indic languages; 989 target speakers; built from IndicSUPERB (CC0); CC BY-NC 4.0. **Hindi: XTTS-v2 Diversity (101,202 clips, 171.36 h) + FreeVC24 Diversity (104,736 clips, 167.77 h); no Hindi mimicry subset.** Average Hindi clip 2.65 s.
- **Model:** benchmarks AASIST, AASIST-L and RawNet2 trained on ASVspoof19 LA. No IndicSynth-trained detector is reported.
- **Training setup:** off-the-shelf English-trained models, no adaptation.
- **Evaluation:** EER per language and generator on test sets of 4k+4k bona fide (IndicSUPERB) and 4k+4k synthetic. Hindi EERs (AASIST/AASIST-L/RawNet2): XTTS 42.0/45.4/14.5 %; FreeVC24 81.8/81.9/48.5 %. Also LID and speaker-verification impersonation tests.
- **Contribution:** first large Indic synthetic corpus with speaker/gender metadata.
- **Limitation:** no detector trained on it; 2 generators for Hindi; no code-switching; no channel conditions; no user study.
- **Relevance:** primary Hindi spoof source. Its bona fide counterpart is IndicSUPERB/Kathbath. Metadata supports speaker-graph splits.
- **Overlap:** **High** for "English-trained detectors fail on Hindi/Indic".

## 16. HAV-DF [R21] — FT (`paper1.pdf`)
- **Year/venue:** 2024, arXiv.
- **Problem:** no Hindi audio-video deepfake dataset.
- **Dataset:** 508 videos (200 real, 308 fake), 15 s, 1080p, from YouTube. Fakes made with FSGAN/FaceSwap/DeepFaceLab, VideoReTalking lip-sync and RVC voice cloning. Available "on reasonable request".
- **Model:** **video** detectors only (HeadPose, Xception, MesoNet, FWA…); accuracy %.
- **Training setup:** 408 train / 100 test videos.
- **Evaluation:** video detectors' accuracy is lower on HAV-DF than on English datasets (as reported). **No audio detector is evaluated.**
- **Contribution:** Hindi AV deepfake resource.
- **Limitation:** small; video-centric; no audio-ADD benchmark; unclear speaker splits; ethics of sourcing YouTube individuals not detailed.
- **Relevance:** related work only. Its limitations section explicitly names multilingual/code-mixed voice synthesis as future work.
- **Overlap:** Low.

## 17. Indic-CodecFake meets SATYAM [R22] — FT
- **Year/venue:** 2026, Findings of ACL.
- **Problem:** neural-audio-codec (NAC) synthesised deepfakes in Indic languages.
- **Dataset:** ICF, built by **resynthesising IndicSUPERB speech** (12 languages) through DAC, Encodec, SoundStream, SpeechTokenizer, FunCodec, AudioDec, SNAC, MIMI. One-to-one real↔fake pairs; official IndicSUPERB splits kept. *Seen* uses test-known; *unseen* uses test-unknown with held-out NACs (FunCodec, AudioDec, MIMI); a noisy unseen split is included. CC BY 4.0.
- **Model:** SATYAM, a hyperbolic ALM fusing frozen Whisper (semantic) and TRILLsson (prosodic) features via Bhattacharya distance in hyperbolic space, plus prompt alignment, with a Qwen2-family decoder. AASIST and ALMs are baselines.
- **Training setup:** AdamW, lr 1e-4, batch 32, 5 epochs; only light modules trained (~3.75M params).
- **Evaluation:** EER. Reported examples (theirs): unseen-codec clean 5.23 % vs AASIST 14.38 %; cross-lingual random splits 6.34/7.09 % vs AASIST 26.74/31.11 %. Also language-family transfer and cross-benchmark with CodecFake. Inference ~6.5–8.2 s per clip on A100 (as reported).
- **Contribution:** first Indic NAC-fake benchmark; ALM-based detector.
- **Limitation:** single LLM decoder family (authors). Codec fakes come from *resynthesis*, not TTS/VC. **No code-switching, no messaging channel, no calibration.** Heavy inference.
- **Relevance:** the most recent Indic SOTA comparator. Shows "Indic ADD", "unseen-codec generalisation" and "cross-lingual Indic transfer" are already studied.
  - Conceptual risk for us: in ICF, *neural-codec processing of real speech = fake*. In VaaniShield, *Opus/AMR processing of real speech = still bona fide*. A detector that learns "codec artefact ⇒ fake" will false-alarm on app-compressed bona fide speech. This tension is testable (H5).
- **Overlap:** **High** for Indic ADD and unseen-codec claims; **Low** for code-switching and messaging channel.

## 18. SEA-Spoof [R23] — FT (partial)
- **Year/venue:** 2025, arXiv.
- **Problem:** ADD for SEA languages.
- **Dataset:** **includes Hindi** (and Tamil, Thai, Indonesian, Malay, Vietnamese). 711 h total (356.3 h spoof / 354.8 h bona fide). Hindi real from Common Voice and Indic Speech Corpora. Generators include open-source systems (e.g., XTTS, VITS-MMS, F5-TTS) and commercial ones (e.g., ElevenLabs). Licence CC BY-NC-ND 4.0 per search snippet (not re-verified).
- **Model:** AASIST, AASIST3, MoLEx.
- **Evaluation:** MoLEx 1.25 % EER on ASVspoof5 collapses to 43.8 % on SEA-Spoof; fine-tuning restores performance (as reported).
- **Contribution:** paired multilingual real/spoof including Hindi with commercial generators.
- **Limitation:** monolingual utterances; no code-switching or messaging channel (not found).
- **Relevance:** a second Hindi spoof source with commercial generators, if the licence permits. Useful for an unseen-family test.
- **Overlap:** **Medium.** Hindi ADD with modern generators exists.

## 19. Indic pre-trained encoders study [R24] — ABS
- **Year/venue:** 2026, arXiv.
- **Problem:** spontaneous-speech detection and out-of-domain synthetic-speech generalisation across 22 Indic languages.
- **Model:** frozen AST, Vaani-FastConformer, Wav2vec2, Whisper, BEATs.
- **Evaluation:** multi-system TTS generalisation (four TTS models).
- **Contribution:** reports that out-of-domain success is "predicted by a training system's proximity to unseen TTS embeddings".
- **Limitation:** no code-switching or compression (ABS).
- **Relevance:** RQ3 comparator. Suggests an analysis of embedding proximity between our train and test generators.
- **Overlap:** Medium.

## 20. ML-ITW [R07] — ABS
- **Year/venue:** 2026, arXiv.
- **Problem:** multilingual in-the-wild ADD.
- **Dataset:** 28.39 h, 180 public figures, 14 languages, 7 social media platforms. **Hindi not verified.**
- **Model:** end-to-end, SSL-based and Audio-LLM detectors.
- **Evaluation:** "significant performance degradation across diverse languages and real-world acoustic conditions".
- **Contribution:** multilingual found-audio benchmark.
- **Limitation:** found audio, so the channel is uncontrolled and unpaired.
- **Relevance:** possible extra external test; realism precedent.
- **Overlap:** Medium for "real platform audio" in general.

## 21. ArEnAV — "Tell me Habibi, is it real or fake?" [R30] — ABS
- **Year/venue:** 2025, arXiv (cs.CV).
- **Problem:** code-switched audio-visual deepfakes.
- **Dataset:** 387k videos, 765+ h, **Arabic-English intra-utterance code-switching** plus dialects; 4 TTS + 2 lip-sync models.
- **Model:** SOTA deepfake detectors plus human evaluation.
- **Contribution:** described as the first large code-switched AV deepfake dataset.
- **Limitation:** Arabic-English; audio-visual focus; no messaging channel (not verified).
- **Relevance:** **closest code-switching precedent.** Shows code-switched deepfake data is an active idea.
- **Overlap:** **Medium with C1.** "Code-switched deepfake dataset" is not new in general; Hindi-English audio-only is our space.

## 22. CS-FLEURS [R31] — dataset card
- **Problem:** code-switched speech resource.
- **Dataset (per card summary):** Read-Test = read speech for 14 X-English pairs; XTTS-Train/Test1 = generative TTS for 16 X-English pairs; **XTTS-Test2 = 60 {Arabic, Chinese, Hindi, Spanish}-X pairs (only subset with Hindi)**; MMS-Test = "concatenative TTS", 45 X-English pairs. Licence "CC BY 4.0 for non-commercial use".
- **Relevance:** the dataset guide hoped Read-Test could give real Hinglish. **Per the card, Hindi-English is not in Read-Test** (to re-confirm when downloading). Hindi content is synthetic XTTS only, i.e. XTTS family again.
- **Overlap:** Low (resource).

## 23. CodecFake family [R40, R41, R42] — SEC
- **Years/venues:** CodecFake (Wu et al., Interspeech 2024); Codecfake (Xie et al., TASLP 2025; English + Chinese, >1M samples, CSAM training strategy); CodecFake+ (arXiv 2025).
- **Problem:** fakes from codec-based speech generation (ALMs).
- **Dataset:** speech resynthesised through many neural codecs.
- **Model:** mainly AASIST-based detectors.
- **Contribution:** models trained on TTS/VC data miss codec-based fakes; codec-resynthesised data helps.
- **Limitation:** English/Chinese; neural codecs as *attack*, not messaging-app compression of bona fide.
- **Relevance:** conceptual distinction we must make explicit: a neural codec as generator vs a traditional codec as channel.
- **Overlap:** Medium (conceptual), Low (empirical).

## 24. Proxy-to-wild domain gap [R43] — ABS
- **Year/venue:** 2026, arXiv.
- **Problem:** detectors trained on codec-resynthesised proxies fail on real codec-based deepfakes.
- **Model:** post-trained SSL backbone plus Domain-Shift Feature Augmentation (DSFA).
- **Evaluation:** CoSG ExtEval with "40 unseen generative models and long-form audio".
- **Relevance:** warns that a *proxy* (simulation) ≠ wild. Same logic as our sim-vs-real channel question, but on the generator side.
- **Overlap:** Low.

## 25. ADD-C [R44] — FT
- **Year/venue:** 2025 (v4 Jun 2026), arXiv.
- **Problem:** ADD robustness under communication codecs and packet loss.
- **Dataset:** test set from FoR, WaveFake+LJSpeech, MLAAD+M-AILABS, ASVspoof (English). Clean condition C0 (500 real + 500 fake) plus **simulated** AMR-WB, EVS, IVAS, Opus, Speex-WB and SILK codecs × packet-loss rates of 0/1/5/10/20 %.
- **Model:** GMM (CQCC), LCNN (LFCC), AASIST, plus their proposed models.
- **Training setup:** proposed DA = codec + packet-loss simulation over six equal subsets → 1.83M-utterance augmented set; 5 epochs, batch 256.
- **Evaluation:** EER, AUC, F1. Baselines degrade by 5.30 % EER on average from C0 to C1; augmentation-trained models stay stable (as reported).
- **Contribution:** simulated communication-channel benchmark plus codec/PLR augmentation.
- **Limitation:** all simulated; English; no real app; no calibration.
- **Relevance:** precedent and codec list for our simulated conditions and codec augmentation.
- **Overlap:** **High** for "codec augmentation improves codec robustness". We must not claim this.

## 26. Real-world corruption robustness [R45] — FT (partial)
- **Year/venue:** 2025 (v2 Jul 2026), arXiv (ACM DOI listed).
- **Problem:** systematic robustness of 10 detectors to 18 corruptions.
- **Dataset:** WaveFake (LJSpeech-derived), split 0.7/0.1/0.2.
- **Model:** traditional and foundation-model detectors (e.g., wav2vec/HuBERT/CLAP-based).
- **Corruptions:** noise, modifications (pitch, time-stretch, filtering…), compression (MP3, Opus, quantisation, EnCodec, DAC, FACodec, AudioDec).
- **Findings (authors):** (1) robust to noise, vulnerable to modification and compression, especially neural codecs; (2) foundation models more robust; (3) size helps with diminishing returns; (4) targeted augmentation or speech enhancement improves robustness. "For Opus, even at a high bitrate of 256 kbps, detection performance remains limited."
- **Limitation:** single English dataset; simulated corruptions; random (not speaker-disjoint) split.
- **Relevance:** supports H5/H7. Shows Opus hurts even at high bitrate.
- **Overlap:** **High** for "compression hurts and augmentation helps".

## 27. RTCFake [R46] — FT
- **Year/venue:** 2026, arXiv.
- **Problem:** ADD under **real** real-time-communication transmission (black-box enhancement + codecs + packet loss).
- **Dataset:** ~600 h, 307 speakers, English (LibriHeavy) + Chinese (Chinese-Lips). 7 TTS + 3 VC systems. Transmitted live through **WeChat, Zoom, QQ, DingTalk, Lark, VooV and Telegram** (virtual audio device in, OBS capture out). **Paired offline↔online** utterances; eval includes unseen platforms and unseen noise. On HF.
- **Model:** XLS-R + AASIST with RawBoost, plus phoneme-guided consistency learning (PCL) between offline and online pairs.
- **Training setup:** Adam lr 1e-6, wd 1e-4, ≤100 epochs, early stop 10; RTX 4090.
- **Evaluation:** EER. Open-source-trained models fail on both offline and online sets; PCL improves cross-platform generalisation (as reported).
- **Contribution:** first large paired real-platform ADD dataset; consistency learning.
- **Limitation (authors):** terminal hardware and user behaviour not covered; residual gap under extreme distortion. Also: real-time *calls*, not asynchronous voice notes or file sharing; no WhatsApp; no Hindi/code-switching; no calibration analysis (not found).
- **Relevance:** **closest prior work to C2.** Paired real-platform transmission is done; our setting differs (store-and-forward messaging, Hinglish, sim-vs-real proxy validity, calibration).
- **Overlap:** **High for "paired real-platform channel dataset"; Medium for C2 as we frame it.**

## 28. RADAR Challenge 2026 [R06] — ABS
- **Year/venue:** 2026, arXiv (challenge overview).
- **Problem:** ADD under media transformations ("compression, resampling, noise, and reverberation").
- **Dataset:** English dev; multilingual eval (English, Singapore English, Mandarin, Taiwanese Mandarin, Japanese, Vietnamese); 100k+ utterances.
- **Evaluation:** EER; 22 teams in the final evaluation.
- **Limitation:** whether real platform re-encoding was used is not verified; no Hindi.
- **Relevance:** media-transformation robustness is a 2026 challenge topic, so generic claims will look derivative.
- **Overlap:** Medium.

## 29. PartialSpoof [R51] — SEC
- **Problem:** spoofed segments embedded in bona fide utterances; segment-level labels.
- **Dataset:** built from ASVspoof 2019 LA by VAD segmentation and concatenation.
- **Model:** SSL-based countermeasures with utterance- and segment-level multi-task learning.
- **Relevance:** precedent for segment-level scoring/timelines. Our demo timeline is not novel.
- **Overlap:** Medium (C5 timeline aspect).

## 30. Splicing artifacts [R52] — ABS
- **Year/venue:** 2024, ASVspoof 5 Workshop.
- **Finding:** splicing artefacts alone give EER 6.16 % (PartialSpoof) and 7.36 % (HAD) "without needing to train any detector".
- **Relevance:** **direct risk for our language-mix splice augmentation (C3).** Splice boundaries are detectable, so splices must be applied symmetrically and controlled with a splice-only probe.
- **Overlap:** Medium (it constrains C3 rather than solving it).

## 31. "When EER hides deployment failure" [R53] — ABS
- **Year/venue:** 2026 (19 Jun), arXiv.
- **Problem:** EER uses an oracle threshold. Deployment needs a fixed threshold on unlabeled target data.
- **Dataset:** train ASVspoof 2019 LA; target In-the-Wild, ASVspoof 2021 DF.
- **Model:** SSL-AASIST.
- **Methods audited:** z-norm, temperature/shift calibration, embedding mean alignment, AS-norm with an unlabeled cohort, pseudo-label calibration.
- **Findings (authors):** transferring the LA-calibrated threshold to In-the-Wild gives HTER 39.5 % with 78.7 % of bona fide rejected, despite 11.2 % EER. Any strictly increasing score transform cannot change EER; "no audited correction reduces EER by more than 1% relative".
- **Relevance:** directly constrains C4. Calibration and threshold transfer under shift are already being audited. We must evaluate C4 by HTER/FPR at a fixed threshold and by ECE, never by EER. It also confirms that bona fide false rejection under shift is the deployment failure mode, which supports H5.
- **Overlap:** **High with C4 as a general idea.** Channel-*conditioned* selection from estimated condition, for Hinglish messaging, is not covered by its abstract.
