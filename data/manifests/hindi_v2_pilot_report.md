# Manifest report: `data\manifests\hindi_v2_pilot.csv`

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
- language 'hi': source_dataset perfectly separates the classes (bonafide={'kathbath_hi': 320}, spoof={'indicsynth_hi': 320})
- language 'hi' train split: source_dataset perfectly separates the classes (bonafide={'kathbath_hi': 100}, spoof={'indicsynth_hi': 100})

## Class distribution by split
| split | bonafide | spoof | hours | speaker groups |
|---|---|---|---|---|
| train | 100 | 100 | 0.29 | 4 |
| dev | 25 | 25 | 0.09 | 1 |
| test | 195 | 195 | 0.62 | 6 |

## Language distribution by split
| split | en | hi | hi-en |
|---|---|---|---|
| train | 0 | 200 | 0 |
| dev | 0 | 50 | 0 |
| test | 0 | 390 | 0 |

## Source × class
| source | bonafide | spoof | codec |
|---|---|---|---|
| indicsynth_hi | 0 | 320 | {'pcm_s16le': 320} |
| kathbath_hi | 320 | 0 | {'pcm_s16le': 320} |

## Spoof generator families by split
- train: {'xtts': 100}
- dev: {'xtts': 25}
- test: {'freevc': 160, 'xtts': 35}
- unseen in train: {'dev': [], 'test': ['freevc']}
