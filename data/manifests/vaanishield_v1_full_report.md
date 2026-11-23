# Manifest report: `data\manifests\vaanishield_v1_full.csv`

| Check | Result |
|---|---|
| required_fields_present | PASS |
| split_complete_and_both_classes | PASS |
| no_duplicate_leakage | PASS |
| no_speaker_leakage | PASS |
| no_derived_rows_crossing_splits | PASS |
| generator_leakage | PASS |

## Shortcut warnings
- source indicsynth_hi is single-class (spoof); USED FOR TRAINING
- source kathbath_hi is single-class (bonafide); USED FOR TRAINING
- source openslr104_hien is single-class (bonafide); declared eval-only
- language 'hi': codec perfectly separates the classes (bonafide={'aac': 240}, spoof={'pcm_s16le': 320})
- language 'hi': sample_rate_orig perfectly separates the classes (bonafide={'44100': 240}, spoof={'24000': 320})
- language 'hi': source_dataset perfectly separates the classes (bonafide={'kathbath_hi': 240}, spoof={'indicsynth_hi': 320})
- language 'hi' train split: codec perfectly separates the classes (bonafide={'aac': 105}, spoof={'pcm_s16le': 225})
- language 'hi' train split: sample_rate_orig perfectly separates the classes (bonafide={'44100': 105}, spoof={'24000': 225})
- language 'hi' train split: source_dataset perfectly separates the classes (bonafide={'kathbath_hi': 105}, spoof={'indicsynth_hi': 225})

## Class distribution by split
| split | bonafide | spoof | hours | speaker groups |
|---|---|---|---|---|
| train | 2685 | 23025 | 24.71 | 36 |
| dev | 2644 | 22321 | 24.2 | 27 |
| test | 7694 | 63952 | 62.21 | 101 |

## Language distribution by split
| split | en | hi | hi-en |
|---|---|---|---|
| train | 25380 | 330 | 0 |
| dev | 24844 | 121 | 0 |
| test | 71237 | 109 | 300 |

## Source × class
| source | bonafide | spoof | codec |
|---|---|---|---|
| asvspoof2019_la | 12483 | 108978 | {'flac': 121461} |
| indicsynth_hi | 0 | 320 | {'pcm_s16le': 320} |
| kathbath_hi | 240 | 0 | {'aac': 240} |
| openslr104_hien | 300 | 0 | {'pcm_s16le': 300} |

## Spoof generator families by split
- train: {'asvspoof19_A04': 3800, 'asvspoof19_A01': 3800, 'asvspoof19_A03': 3800, 'asvspoof19_A05': 3800, 'asvspoof19_A02': 3800, 'asvspoof19_A06': 3800, 'freevc': 100, 'xtts': 125}
- dev: {'asvspoof19_A01': 3716, 'asvspoof19_A05': 3716, 'asvspoof19_A06': 3716, 'asvspoof19_A04': 3716, 'asvspoof19_A02': 3716, 'asvspoof19_A03': 3716, 'freevc': 25}
- test: {'asvspoof19_A10': 4914, 'asvspoof19_A15': 4914, 'asvspoof19_A11': 4914, 'asvspoof19_A09': 4914, 'asvspoof19_A14': 4914, 'asvspoof19_A17': 4914, 'asvspoof19_A12': 4914, 'asvspoof19_A13': 4914, 'asvspoof19_A18': 4914, 'asvspoof19_A08': 4914, 'asvspoof19_A19': 4914, 'asvspoof19_A07': 4914, 'asvspoof19_A16': 4914, 'freevc': 35, 'xtts': 35}
- unseen in train: {'dev': [], 'test': ['asvspoof19_A07', 'asvspoof19_A08', 'asvspoof19_A09', 'asvspoof19_A10', 'asvspoof19_A11', 'asvspoof19_A12', 'asvspoof19_A13', 'asvspoof19_A14', 'asvspoof19_A15', 'asvspoof19_A16', 'asvspoof19_A17', 'asvspoof19_A18', 'asvspoof19_A19']}
