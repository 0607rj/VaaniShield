# Manifest report: `data\manifests\hinglish_pilot_v1.csv`

| Check | Result |
|---|---|
| required_fields_present | PASS |
| split_complete_and_both_classes | PASS |
| no_duplicate_leakage | PASS |
| no_speaker_leakage | PASS |
| no_derived_rows_crossing_splits | PASS |
| generator_leakage | PASS |

## Shortcut warnings
- source openslr104_hien is single-class (bonafide); USED FOR TRAINING
- language 'hi-en' train split: source_dataset perfectly separates the classes (bonafide={'openslr104_hien': 30}, spoof={'cs_fleurs': 30})

## Class distribution by split
| split | bonafide | spoof | hours | speaker groups |
|---|---|---|---|---|
| train | 30 | 30 | 0.14 | 16 |
| dev | 10 | 10 | 0.05 | 6 |
| test | 60 | 60 | 0.4 | 8 |

## Language distribution by split
| split | en | hi | hi-en |
|---|---|---|---|
| train | 0 | 0 | 60 |
| dev | 0 | 0 | 20 |
| test | 0 | 0 | 120 |

## Source × class
| source | bonafide | spoof | codec |
|---|---|---|---|
| cs_fleurs | 50 | 100 | {'pcm_s16le': 100, 'pcm_32': 50} |
| openslr104_hien | 50 | 0 | {'pcm_s16le': 50} |

## Spoof generator families by split
- train: {'xtts': 30}
- dev: {'xtts': 10}
- test: {'concatenative': 50, 'xtts': 10}
- unseen in train: {'dev': [], 'test': ['concatenative']}
