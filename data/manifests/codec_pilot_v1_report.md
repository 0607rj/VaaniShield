# Manifest report: `data\manifests\codec_pilot_v1.csv`

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
- language 'hi-en' train split: source_dataset perfectly separates the classes (bonafide={'openslr104_hien': 18}, spoof={'cs_fleurs': 18})

## Class distribution by split
| split | bonafide | spoof | hours | speaker groups |
|---|---|---|---|---|
| train | 18 | 18 | 0.09 | 3 |
| dev | 24 | 24 | 0.15 | 3 |
| test | 24 | 24 | 0.15 | 2 |

## Language distribution by split
| split | en | hi | hi-en |
|---|---|---|---|
| train | 0 | 0 | 36 |
| dev | 0 | 0 | 48 |
| test | 0 | 0 | 48 |

## Source × class
| source | bonafide | spoof | codec |
|---|---|---|---|
| cs_fleurs | 24 | 66 | {'pcm_s16le': 88, 'pcm_32': 2} |
| openslr104_hien | 42 | 0 | {'pcm_s16le': 42} |

## Spoof generator families by split
- train: {'xtts': 18}
- dev: {'xtts': 24}
- test: {'concatenative': 24}
- unseen in train: {'dev': [], 'test': ['concatenative']}
