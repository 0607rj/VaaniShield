# Model Card — VaaniShield detector

> **Status: no model has been trained.** Every metric below is "not measured". This card is a template, filled in Phase 8 (P8.5) from `ml/experiments/*/metrics.json` by `paper/tools` (M22). Do not edit numbers by hand.

## Model details
| Field | Value |
|---|---|
| Name / version | not released |
| Architecture | planned: XLS-R-300M (frozen) → learned layer weights → AASIST-lite head → temperature calibration (docs/architecture.md) |
| Baselines | LFCC-LCNN (B0), AASIST (B1) |
| Input | mono 16 kHz, 2–20 s; scored in 4.04 s windows with 50 % overlap |
| Output | calibrated spoof probability, label at a dev-fixed threshold, segment timeline, quality warnings |
| Checkpoint sha256 | not released |

## Intended use
Decision support for researchers and educators. It estimates whether a Hindi / Hindi-English / English speech clip contains synthetic-speech artefacts.

## Out-of-scope uses
- Forensic, legal or courtroom evidence; punitive decisions about a person.
- Speaker identification or attribution of the generator.
- Any claim of "100 % accuracy".
- Use on languages, channels or generators outside the documented evaluation slices without re-evaluation.

## Training data
not yet trained (see DATASET_CARD.md for the planned sources and licences)

## Evaluation (all "not measured")
| Slice | EER | AUC | False alarm @thr | Miss @thr | ECE |
|---|---|---|---|---|---|
| ASVspoof 2019 LA eval | not measured | not measured | not measured | not measured | not measured |
| Hindi (seen / unseen family) | not measured | not measured | not measured | not measured | not measured |
| Hinglish clean | not measured | not measured | not measured | not measured | not measured |
| Hinglish simulated codecs | not measured | not measured | not measured | not measured | not measured |
| Hinglish real messaging apps | not measured | not measured | not measured | not measured | not measured |
| In-the-Wild | not measured | not measured | not measured | not measured | not measured |

## Limitations and ethical considerations
- Performance may degrade on unseen generators, languages, devices and app versions. The detector can be wrong in both directions. False accusations are a foreseeable harm (DOC:pg §16).
- Training data include non-commercial-licensed resources, so the model inherits those restrictions.
- The segment timeline is a localisation aid, not an explanation of which artefact caused the decision.
