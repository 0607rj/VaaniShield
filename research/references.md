# References — VaaniShield research audit

Audit date: **2026-10-07**. Every entry has a verification level so readers know how far to trust the summarised details.

| Level | Meaning |
|---|---|
| **FT** | Full text read (supplied PDF, or PDF downloaded and extracted during this audit) |
| **ABS** | Abstract or official landing page read |
| **SEC** | Only secondary summaries/search snippets; re-verify before citing details |
| **DOC** | Cited inside our own project documents; not independently re-read |

Bibliographic details must be re-checked against the final published version before the paper is submitted [DOC:pg p19].

## Benchmarks and challenges
| ID | Reference | Level | URL |
|---|---|---|---|
| R01 | X. Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech," *Computer Speech & Language* 64, 2020 | DOC | https://datashare.ed.ac.uk/handle/10283/3336 |
| R02 | X. Liu et al., "ASVspoof 2021: Towards spoofed and deepfake speech detection in the wild," *IEEE/ACM TASLP* 31:2507–2522, 2023 | SEC | https://arxiv.org/abs/2210.02437 |
| R03 | "ASVspoof 5: Evaluation of spoofing, deepfake, and adversarial attack detection using crowdsourced speech," IEEE TASLP (accepted; arXiv Jan 2026); companion resources paper in *Computer Speech & Language* 95, 2026 | ABS+SEC | https://arxiv.org/abs/2601.03944 |
| R04 | N. Müller et al., "Does audio deepfake detection generalize?" *Interspeech* 2022 (In-the-Wild) | ABS | https://arxiv.org/abs/2203.16263 |
| R05 | N. M. Müller et al., "MLAAD: The multi-language audio anti-spoofing dataset," *IJCNN* 2024 (continuously updated) | ABS | https://arxiv.org/abs/2401.09512 |
| R06 | RADAR Challenge 2026: Robust audio deepfake recognition under media transformations (arXiv 2026) | ABS | https://arxiv.org/abs/2605.09568 |
| R07 | ML-ITW: A multilingual in-the-wild benchmark for speech deepfake detection (arXiv 2026) | ABS | https://arxiv.org/abs/2603.05852 |

## Model architectures and front ends
| ID | Reference | Level | URL |
|---|---|---|---|
| R10 | J.-w. Jung et al., "AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks," *ICASSP* 2022 | FT | https://arxiv.org/abs/2110.01200 |
| R11 | H. Tak et al., "RawBoost: A raw data boosting and augmentation method applied to automatic speaker verification anti-spoofing," *ICASSP* 2022 | ABS (+ described in R12 FT) | https://arxiv.org/abs/2111.04433 |
| R12 | H. Tak et al., "Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation," *Odyssey* 2022 | FT | https://arxiv.org/abs/2202.12233 |
| R13 | A. Babu et al., "XLS-R: Self-supervised cross-lingual speech representation learning at scale," *Interspeech* 2022 (arXiv 2021) | ABS | https://arxiv.org/abs/2111.09296 |
| R14 | Q. Zhang, S. Wen, T. Hu, "Audio deepfake detection with self-supervised XLS-R and SLS classifier," *ACM MM* 2024, pp. 6765–6773 | SEC (ACM page returned 403; third-party model card read) | https://dl.acm.org/doi/10.1145/3664647.3681345 |
| R15 | O. Pascu et al., "Towards generalisable and calibrated audio deepfake detection with self-supervised representations," *Interspeech* 2024, pp. 4828–4832 | ABS | https://www.isca-archive.org/interspeech_2024/pascu24_interspeech.html |
| R16 | O. C. Phukan et al., "Heterogeneity over homogeneity: Investigating multilingual speech pre-trained models for detecting audio deepfake," *Findings of NAACL* 2024 | ABS | https://aclanthology.org/2024.findings-naacl.160/ |
| R17 | P. Serrano et al., "Improving out-of-domain audio deepfake detection via layer selection and fusion of SSL-based countermeasures," arXiv 2025 | ABS | https://arxiv.org/abs/2509.12003 |

## Indic and multilingual resources
| ID | Reference | Level | URL |
|---|---|---|---|
| R20 | D. V. Sharma, V. Ekbote, A. Gupta, "IndicSynth: A large-scale multilingual synthetic speech dataset for low-resource Indian languages," *ACL* 2025, pp. 22037–22060 (= `paper2.pdf`) | FT | https://aclanthology.org/2025.acl-long.1070/ |
| R21 | S. Kaur et al., "Hindi audio-video-Deepfake (HAV-DF)," arXiv:2411.15457, 2024 (= `paper1.pdf`) | FT | https://arxiv.org/abs/2411.15457 |
| R22 | Girish, M. M. Akhtar, O. C. Phukan, A. B. Buduru, "Indic-CodecFake meets SATYAM: Towards detecting neural audio codec synthesized speech deepfakes in Indic languages," *Findings of ACL* 2026 | FT | https://arxiv.org/abs/2604.19949 |
| R23 | J. Wu et al., "SEA-Spoof: Bridging the gap in multilingual audio deepfake detection for South-East Asia," arXiv 2025 | FT (partial) | https://arxiv.org/abs/2509.19865 |
| R24 | "Evaluating pre-trained speech encoders for spontaneous speech detection and out-of-domain synthetic speech generalisation in Indic languages," arXiv 2026 | ABS | https://arxiv.org/abs/2608.12536 |
| R25 | T. Javed et al., "IndicSUPERB," *AAAI* 2023 (Kathbath corpus, CC0) | ABS+DOC | https://github.com/AI4Bharat/IndicSUPERB |

## Code-switching
| ID | Reference | Level | URL |
|---|---|---|---|
| R30 | "Tell me Habibi, is it real or fake?" (ArEnAV, Arabic-English code-switched audio-visual deepfakes), arXiv 2025 | ABS | https://arxiv.org/abs/2505.22581 |
| R31 | CS-FLEURS dataset card (byan/cs-fleurs) | dataset card | https://huggingface.co/datasets/byan/cs-fleurs |
| R32 | OpenSLR SLR104, MUCS 2021 Hindi-English | landing page | https://www.openslr.org/104/ |

## Codec, channel and real-world robustness
| ID | Reference | Level | URL |
|---|---|---|---|
| R40 | H. Wu, Y. Tseng, H.-y. Lee, "CodecFake: Enhancing anti-spoofing models against deepfake audios from codec-based speech synthesis systems," *Interspeech* 2024, pp. 1770–1774 | SEC | https://arxiv.org/abs/2406.07237 |
| R41 | Y. Xie et al., "The Codecfake dataset and countermeasures for the universally detection of deepfake audio," *IEEE TASLP* 2025 | SEC | https://arxiv.org/abs/2405.04880 |
| R42 | "CodecFake+: Codec-based resynthesized data as a proxy for detecting CodecFake speech," arXiv 2025 | SEC | https://arxiv.org/abs/2501.08238 |
| R43 | X. Chen et al., "Mitigating proxy-to-wild domain gap in deepfake speech," arXiv 2026 | ABS | https://arxiv.org/abs/2606.07494 |
| R44 | H. Shi et al., "Benchmarking audio deepfake detection robustness in real-world communication scenarios" (ADD-C), arXiv 2025 (v4 Jun 2026) | FT | https://arxiv.org/abs/2504.12423 |
| R45 | X. Li, P.-Y. Chen, W. Wei, "Measuring the robustness of audio deepfake detection under real-world corruption," arXiv 2025 (v2 Jul 2026; ACM DOI 10.1145/3800506.3803494) | FT (partial) | https://arxiv.org/abs/2503.17577 |
| R46 | J. Xue et al., "RTCFake: Speech deepfake detection in real-time communication," arXiv 2026 | FT | https://arxiv.org/abs/2604.23742 |

## Shortcuts, partial spoofs, calibration
| ID | Reference | Level | URL |
|---|---|---|---|
| R50 | N. M. Müller et al., "Speech is silver, silence is golden: What do ASVspoof-trained models really learn?" *ASVspoof 2021 Workshop* | ABS+SEC | https://arxiv.org/abs/2106.12914 |
| R51 | L. Zhang, X. Wang, E. Cooper, N. Evans, J. Yamagishi, "The PartialSpoof database and countermeasures for the detection of short generated audio segments embedded in a speech utterance" (2021–2023) | SEC | https://github.com/nii-yamagishilab/partialspoof |
| R52 | V. Negroni, D. Salvi, P. Bestagini, S. Tubaro, "Analyzing the impact of splicing artifacts in partially fake speech signals," *ASVspoof 5 Workshop* 2024 | ABS | https://arxiv.org/abs/2408.13784 |
| R53 | J. Zhou, M. Wang, "When EER hides deployment failure: Auditing threshold transfer and unlabeled score calibration for speech deepfake detectors," arXiv Jun 2026 | ABS | https://arxiv.org/abs/2606.21584 |

## Resource and licence pages (checked 2026-10-07)
- XTTS-v2 (CPML): https://huggingface.co/coqui/XTTS-v2
- Hinglish XTTS-v2 (CPML-derived, non-commercial, 4 voices, 8.56 h synthetic training data): https://huggingface.co/harrrshall/xtts-v2-hinglish-synthetic
- IndicSynth on HF (CC BY-NC 4.0): https://huggingface.co/datasets/vdivyasharma/IndicSynth
- In-the-Wild (Apache-2.0; 20.8 h bona fide + 17.2 h spoof; 58 speakers): https://deepfake-total.com/in_the_wild
- RTCFake data: https://huggingface.co/datasets/JunXueTech/RTCFake (licence not verified)
- WhatsApp voice-note codec: only **non-authoritative blog sources** were found (Opus in OGG; the bitrates quoted vary). Treat as unverified; measure empirically with ffprobe in the real-app protocol.
