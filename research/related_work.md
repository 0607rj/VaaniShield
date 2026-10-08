# Related work — draft for the paper (v0, 2026-10-07)

> Draft prose for the paper's Related Work section. Citations use reference IDs (R##) from `references.md`. Convert them to BibTeX later.
> Re-verify every factual sentence against the final published version before submission. Sentences marked **[verify]** rest only on secondary sources (SEC).
> Numbers are other authors' results and are attributed to them.

## 2.1 Spoofing countermeasures and self-supervised front ends

The ASVspoof challenges established the standard logical-access benchmark and the EER / t-DCF evaluation practice [R01, R02, R03]. AASIST models spectral and temporal artefacts jointly with heterogeneous stacking graph attention, and its 85K-parameter variant AASIST-L remains competitive [R10]. Replacing AASIST's sinc front end with a fine-tuned wav2vec 2.0 XLS-R (0.3B) model [R13] and adding RawBoost augmentation [R11] produced large gains on the ASVspoof 2021 LA and DF evaluation sets. Those gains came with a front end that the authors describe as "massively more complex" than earlier countermeasures [R12].

Later work exploits intermediate SSL layers:
- learned weighting of all XLS-R transformer layers [R14] **[verify]**;
- single-layer selection that cuts parameters by up to 80 % [R17];
- frozen SSL representations with a linear classifier, which generalise across eight deepfake datasets [R15].

Multilingual pre-trained models outperform monolingual ones for audio deepfake detection on English-centric benchmarks [R16].

**Our position.** We adopt the XLS-R + AASIST-style recipe as a re-implemented baseline. We do not propose a new architecture. We test whether these findings transfer to Hindi-English code-switched and messaging-app re-encoded speech.

## 2.2 Generalisation and dataset shortcuts

Detectors trained on ASVspoof degrade sharply on found real-world audio [R04]. Part of their in-domain accuracy can come from dataset artefacts such as the leading/trailing silence imbalance in ASVspoof [R50]. Splicing artefacts in partially fake speech are detectable without training a detector [R52]. Both findings shape our controls: consistent silence handling for both classes, and a splice-only probe for our code-switch augmentation.

## 2.3 Multilingual and Indic audio deepfakes

Multilingual training resources such as MLAAD [R05] broadened language coverage. For Indian languages:
- **IndicSynth** provides ~4,000 h of TTS/VC speech in 12 languages, generated from IndicSUPERB [R25]. ASVspoof-trained AASIST, AASIST-L and RawNet2 show high EERs on it [R20]. For Hindi it contains two generator subsets (XTTS-v2 and FreeVC24) and no mimicry subset [R20].
- **Indic-CodecFake** resynthesises IndicSUPERB through neural audio codecs. It evaluates unseen-codec, cross-lingual and language-family transfer, and proposes SATYAM, a hyperbolic audio-LLM detector [R22].
- **SEA-Spoof** pairs real and spoofed speech in six languages including Hindi, using open-source and commercial generators [R23].
- A 2026 study compares frozen encoders for synthetic-speech generalisation across 22 Indic languages [R24].
- **HAV-DF** is a small Hindi audio-video deepfake set evaluated with video detectors [R21].
- ML-ITW collects multilingual in-the-wild deepfakes from social platforms [R07].

**Our position.** These resources are monolingual per utterance. None of those we examined targets Hindi-English intra-utterance code-switching or messaging-app channels.

## 2.4 Code-switched speech and deepfakes

Code-switched speech resources include the MUCS 2021 Hindi-English corpus [R32] and CS-FLEURS. In CS-FLEURS, Hindi appears only in a generative-TTS test subset [R31]. For deepfakes, ArEnAV introduced Arabic-English intra-utterance code-switched audio-visual deepfakes [R30].

**Our position.** We found no audio-only Hindi-English code-switched spoof-detection benchmark. That is search-limited evidence, and a systematic search is pending.

## 2.5 Codec, channel and platform robustness

Channel effects have been studied from several directions:
- ASVspoof 2021 transmitted LA data through real VoIP and PSTN systems and applied commercial compression to DF data [R02] **[verify details]**.
- ASVspoof 5 adds DSP and neural codec conditions and reports a calibration study [R03] **[verify codec list]**.
- ADD-C simulates six communication codecs with packet loss and shows codec/packet-loss augmentation stabilises performance [R44].
- A corruption study of 10 detectors finds compression, especially neural codecs, more damaging than noise; Opus degrades detection even at high bitrates [R45].
- RADAR 2026 benchmarks robustness to media transformations [R06].
- Closest to our setting, RTCFake transmits English and Chinese speech live through seven real-time communication platforms. It provides paired offline/online data and proposes phoneme-guided consistency learning [R46].

A separate line treats neural codecs as the *generator* of fakes [R40, R41, R42, R43]. That differs from conventional codecs acting as a *channel* on both bona fide and spoofed speech.

**Our position.** We study asynchronous store-and-forward messaging re-encoding (voice notes and audio files) of Hindi-English speech. We explicitly test whether laboratory codec simulation is a valid proxy for it.

## 2.6 Calibration and threshold transfer

EER is computed at an oracle threshold and is invariant to monotone score transforms. A 2026 audit shows that transferring a calibrated threshold from ASVspoof to In-the-Wild can reject most bona fide speech even at moderate EER, and that common unlabeled corrections barely change EER [R53]. Calibrated scores were encouraged in ASVspoof 5 [R03], and SSL-based detectors have been studied for calibration [R15].

**Our position.** We report ECE and fixed-threshold error rates alongside EER. We test whether calibration conditioned on an estimated channel condition reduces deployment error under real messaging-app processing.

## 2.7 Segment-level scoring

PartialSpoof introduced segment-level labels and countermeasures for partially spoofed utterances [R51] **[verify]**. Our segment timeline is a user-interface localisation aid, not a contribution. Our only research use of segment scores is an exploratory analysis relative to language-switch points.
