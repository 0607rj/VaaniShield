# VaaniShield — Experiment Protocol (v1)

Design only. This protocol operationalises RQ1–RQ4 and H1–H13 (`research/research_gap.md`). Any deviation is logged in `PROJECT_STATE.md` §5 **before** the affected run.

## 1. Roles of data partitions

| Partition | Used for | Never used for |
|---|---|---|
| train | gradient updates | – |
| dev (clean + simulated codecs) | early stopping, checkpoint selection, temperature/threshold fitting, channel-estimator fitting, hyperparameter choice | – |
| test_* (locked) | one final scoring per pre-registered run | any choice of any kind |

If a test result is used to change anything (threshold, crop length, augmentation, checkpoint), that test set is **burned**. It must be relabelled as dev, and a fresh test set constructed or the limitation declared (DOC:pg §06).

## 2. Training compositions

| ID | Train pool | Dev pool |
|---|---|---|
| `T_en` | ASVspoof 2019 LA train | LA dev |
| `T_hi` | Kathbath-Hi + IndicSynth-Hi (seen families only) | held-out groups, seen families |
| `T_en+hi` | union, class- and source-balanced sampler | union of devs |
| `T_en+hi+cs` | `T_en+hi` + Hinglish bona fide train pool (OpenSLR 104 train groups, consented train speakers) + Hinglish spoof train pool (generator families not held out) | union of devs incl. Hinglish dev |

## 3. Experiment matrix

**Default recipe:** frozen XLS-R, weighted layers, AASIST-lite, RawBoost (algorithm combination chosen on dev), codec augmentation, symmetric language-mix, global-temperature calibration on dev (clean + simulated).

| Exp | Model | Train | Aug | Calib | Tests | Hypotheses |
|---|---|---|---|---|---|---|
| E0 | B0 LFCC-LCNN | T_en; T_en+hi+cs | none; full | global | all | H1, H8 |
| E1 | B1 AASIST (official) | T_en; T_en+hi+cs | none; RawBoost; full | global | all | H1, H5, H8 |
| E2 | P1 XLS-R + AttnStats | T_en; T_en+hi+cs | full | global | all | H8 |
| E3 | P2 XLS-R + AASIST-lite | T_en; T_en+hi; T_en+hi+cs | full | global | all | H1, H2, H8 |
| A1 | P2 | T_en+hi+cs | full − codec | global | sim + app | H5, H7 |
| A2 | P2 | T_en+hi+cs | full − langmix | global | Hinglish | H3 |
| A2p | splice-only probe (no detector) | symmetric vs asymmetric splice sets | – | – | probe AUC | H3, H4 |
| A3a | English wav2vec2 + AASIST-lite | T_en+hi+cs | full | global | Hi, Hinglish | H9 |
| A3b | P2 with `last` / `fixed_layer` aggregation | T_en+hi+cs | full | global | Hi, Hinglish | H10 |
| A4 | P2 | T_en+hi+cs + `train_realapp` | full | global | app | sim-to-real gap (secondary) |
| A5 | P2 (and E1) | leave-one-family-out folds | full | global | unseen-family tests | H8 |
| A6 | P2 unfreeze top-4 blocks | T_en+hi+cs | full | global | all | DOC:syn 4.2.4 |
| C4a | P2 best | – | – | none / global(clean) / global(clean+sim) / conditional | app + sim | H11, H12 |
| K1 | P2 truncated at layer k* (from learned α) | T_en+hi+cs | full | global | Hinglish + latency | H13 |
| X1 | exploratory segment analysis on best P2 | – | – | – | Hinglish with switch annotations | E1 (exploratory) |

**Seeds:** 2026, 2027, 2028 for every trained cell (TD-5). Report mean ± SD and best seed. Paired bootstrap uses the per-seed mean of logits per clip.

**Budget gate:** if measured T4 throughput (Phase 4) shows the matrix exceeds available GPU hours, cut in this order: A6 → A3a → A4 → T_hi-only variants. Record every cut in PROJECT_STATE as "not measured — budget".

## 4. Model selection and calibration rules

1. **Checkpoint selection:** the lowest dev EER on the **clean + simulated-codec** dev pool, pooled. Ties go to the earlier epoch.
2. **RawBoost algorithm, augmentation probabilities, learning rate:** a small grid (≤ 6 configurations) evaluated on dev with seed 2026 only. The chosen configuration is frozen in the YAML before the 3-seed runs.
3. **Temperature and threshold:** fitted on dev logits only. Thresholds are set at the dev EER point (default) **and** at a dev false-alarm rate of 1 % (secondary operating point). Both are reported.
4. **Channel estimator (C4):** trained on dev/train *simulated* condition labels. Real-app audio is never used to fit it.

## 5. Metrics (exact definitions)

- **EER:** decide "spoof" when `spoof_logit ≥ θ`. Sweep θ over all unique scores and compute, for each θ:
  - false-alarm rate = fraction of bona fide with score ≥ θ;
  - miss rate = fraction of spoof with score < θ.

  Following the official ASVspoof 2019 det-curve algorithm, EER = mean(false-alarm, miss) at the θ that minimises |false-alarm − miss|. There is no interpolation; this is required for parity with the official script.
  - Implemented in `vaanishield.metrics.core.eer`.
  - Checked against an independent brute-force implementation in unit tests.
  - Parity with the official script on real score files is test P3.6. Must match the official ASVspoof 2019 script within 1e-6 (§M15 parity test).
- **ROC-AUC:** `sklearn.metrics.roc_auc_score(labels, spoof_logit)`.
- **At fixed threshold (dev-chosen):** false-alarm rate, miss rate, HTER = (FA + miss)/2, precision/recall/F1 for the spoof class, confusion matrix.
- **ECE:** 15 equal-mass bins (primary) and 15 equal-width bins (secondary), on p_spoof against labels. NLL and Brier also reported. All computed before and after calibration.
- **min t-DCF:** only for `test_la19`, using the official ASV scores and cost parameters of ASVspoof 2019.
- **Latency:** median and p95 wall-clock of `Scorer.score` for 4 s and 10 s clips. 100 runs after 10 warm-ups, batch size 1, on named hardware (CPU model and thread count; GPU name). Parameter count and peak memory are also reported.
- **CIs:** 1,000 bootstrap resamples over `speaker_group` (paired across systems on the same clips); 95 % percentile intervals.

## 6. Required slices (every test run)

- language: hi / en / hi-en
- transform: clean / sim:<codec> / app:<app>:<mode>
- generator_family: seen / unseen
- source_dataset
- duration bucket: <2 s, 2–4 s, 4–8 s, >8 s
- gender, where known
- QC flags: narrowband, clipping

Slices with fewer than 30 clips per class are reported with an "n<30" marker and no CI.

## 7. Pre-registration

Before Phase 7 starts, commit `docs/preregistration.md`. It contains:
- H1–H13 with the margins from `research_gap.md` §5 (or team-amended values);
- the exact run IDs/configs that will test each hypothesis;
- the test sets used;
- the decision rule.

After that commit, margins and test sets cannot change. Any extra analysis is labelled "exploratory".

## 8. Sanity and shortcut checks (must pass before matrix runs)

| Check | Pass criterion |
|---|---|
| Overfit 32 clips | train loss < 0.05 within 200 steps |
| Label swap | dev AUC < 0.5 when labels are swapped at training |
| Score direction | mean spoof_logit(spoof) > mean spoof_logit(bonafide) on dev |
| Silence probe | logistic regression on [leading silence, trailing silence, duration] gives dev AUC ≤ 0.60. Otherwise fix the VAD policy |
| Corpus probe | classifier separating bona fide sources (e.g. OpenSLR vs Kathbath) is reported. High separability triggers per-source result reporting (R2) |
| Splice probe (A2p) | symmetric design probe AUC within 0.50 ± 0.05 (H3) |
| Determinism | the same config and seed run twice for 1 epoch on GPU gives dev EER within 0.1 pp (cuDNN non-determinism tolerance documented) |

## 9. Run lifecycle

`config committed → train (RunContext) → dev scores → calibrate → bundle → [optional: test scoring with --confirm-final] → metrics.json → status=completed`

- Failed runs keep `status=failed` and a `NOTES.md` entry. They are never deleted (DOC:syn 6.2.2).
- Test-split scoring requires all of the following:
  1. a clean git tree;
  2. the run's `status=completed` on dev;
  3. a `test_lock.yaml` hash match;
  4. the `--confirm-final` flag;
  5. an appended entry in `test_access_log.csv`.
- A second test scoring of the same bundle is allowed only for crash recovery and is logged.
