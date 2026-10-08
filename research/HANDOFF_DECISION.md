# Handoff decision

## Decision: **READY_WITH_CONDITIONS**

The teammate can safely run the **English (ASVspoof) baselines**, the codec pipeline and the robustness tooling. The
handoff is **not** ready for any Hindi experiment, and **not** ready for sending the full local data folder as
previously proposed. All conditions below must be met before the transfer.

Date: 2026-10-08. Evidence:
- `research/hindi_data_leakage_audit.md`;
- `research/handoff_licence_audit.md`;
- `PROJECT_STATE.md` §0.14–0.19.

---

## Update 2026-10-08 (after Hindi v2.1): decision unchanged, **READY_WITH_CONDITIONS**
- **Hindi:** the repaired **v2.1** pilot (`research/hindi_v2_1_validation.md`) is READY_WITH_LIMITATIONS for *pilot*
  training only. The old v1 Hindi pool stays excluded from all training and reporting.
- **Configs:** Hindi was removed from `robustness_full_gpu.yaml` and `codec_full_gpu.yaml` before any results existed.
  v1 Hindi is confounded, and v2.1 needs a class-neutral headroom guard before codec steps. **Condition C3 is
  therefore resolved.**
- **What the teammate does with Hindi:** only the optional step H in `docs/GPU_HANDOFF.md`, an independent rebuild of
  v2.1 from the original sources, plus the gate and the pause diagnostic. This is a reproducibility check, not Hindi
  training.
- **Kathbath audio** (including `processed/hindi_v2/`, `processed/hindi_v2_1/` and the fetched train recordings) is
  still not copied; the licence status is unchanged.
- **Tests:** 96 pass. **Still blocking:** C1, because the repository has no commit and no remote yet.

## 1. Hindi-data status: **v1 REJECTED; v2.1 approved for pilot training only (not for the GPU run, not for reported results)**
(Corrected 2026-10-08 to match the actual decision; the bullets below describe why v1 was rejected. v2.1 status:
`research/hindi_v2_1_validation.md` and the update section above.)
- **Cause, confirmed:** real Hindi = Kathbath phone recordings (AAC 44.1 kHz, raw level). Fake Hindi = IndicSynth
  output (PCM 24 kHz, peak-normalised). They are drawn from mostly different speakers: 4 of 19 target speakers
  overlap, and 0 of the 323 reference recordings behind our fakes are in the real pool.
- **Shortcut strength:**
  - container metadata alone separates the classes perfectly (EER 0.000);
  - simple waveform statistics through our current pipeline reach EER 0.17–0.26, versus 0.33–0.41 on ASVspoof.
- **Fixes tested in memory:**
  - loudness normalisation removes the main cue;
  - AAC matching has almost no measurable effect (the format trace is weak after decoding to 16 kHz);
  - spectral differences vanish within speaker-matched groups;
  - a "synthetic silence" cue remains and must be reported.
- **Repair (done):** each fake is paired with its own Kathbath reference recordings. v2.1 = matched preprocessing (DC
  high-pass, linear loudness normalisation, no limiter, float32 storage). It passes the acceptance gate (13/13,
  provisional criteria) and is READY_WITH_LIMITATIONS for pilot training only.
- **No model has been trained on Hindi**, and no Hindi number has been reported as a result.
- **Impact on the teammate:** none for the English work. Hindi has been removed from `robustness_full_gpu.yaml` and
  `codec_full_gpu.yaml` (C3 resolved). The full manifest still *contains* v1 Hindi rows, but no GPU config selects
  them.

## 2. Licence status: **partly verified**
| Group | Status | Handoff route |
|---|---|---|
| ASVspoof 2019 LA (ODC-By 1.0) | verified; redistribution with attribution | copy pilot files; full set downloaded by the teammate |
| IndicSynth Hindi (CC BY-NC 4.0, ungated) | verified; non-commercial redistribution with attribution | copy |
| OpenSLR 104 + our segments (CC BY-SA 4.0) | verified; share-alike | copy |
| Kathbath Hindi | **Needs verification from current dataset terms.** (CC0 vs CC-BY-4.0 conflict; third-party text; gated HF mirror) | **do not copy**; teammate downloads from the original source |
| CS-FLEURS subset | **Needs verification from current dataset terms.** (tag "CC BY-NC 4.0" vs card text "CC BY 4.0 for Non-Commercial use") | **do not copy**; teammate downloads |
| In-the-Wild `meta.csv` | **Needs verification from current dataset terms.** (CC BY-SA 4.0 vs Apache-2.0) | not needed; do not send |
| Codec variants (`processed/codec_sim`) | mixed parents | do not copy; regenerate (bit-identical) |
| XLS-R weights (Apache-2.0), full ASVspoof | public downloads | teammate downloads |
| Repository (our code + MIT-licensed vendored AASIST + manifests) | internal transfer fine; Kathbath/CS-FLEURS/ITW manifest rows carry the same "Needs verification" note | Git |

**Correction:** the earlier statement in `docs/GPU_HANDOFF.md` that "the licences allow this sharing" was unsupported
and is withdrawn (fixed in that file; see the licence audit).

## 3. Reproducibility status: **good, with one blocker**
- **Verified:**
  - fresh-clone portability test (§0.14);
  - the pilot audio matches every manifest hash (`scripts/verify_data.py`: 2,698/2,698);
  - codec variants are bit-identical on re-encode;
  - the robustness report is identical on re-run.
- **Verified today:** re-downloading Kathbath (`acquire_kathbath.py --subset 300`) and CS-FLEURS
  (`acquire_csfleurs.py --metadata` then `--files data/manifests/hinglish_pilot_plan.csv`) from the original sources
  gives **bit-identical** files (240/240 and 150/150), so they do not need to be copied.
- **Blocker:** the Git repository has **no commit and no remote yet**.

## 4. What the teammate can safely receive
1. **The Git repository** (after commit + push to a private repo).
2. **A restricted data archive** (tested: 794 MB, 1,802 entries). From the repository root:
   ```powershell
   tar -cf vaanishield-data.tar --exclude "raw/openslr104_hien/archives" -C data raw/asvspoof2019_la raw/indicsynth_hi raw/openslr104_hien processed/openslr104_hien
   ```
   Include a note: non-commercial research use only (IndicSynth), share-alike (OpenSLR), attribution per
   `LICENSES_REGISTER.md`, no onward redistribution.
3. **Downloaded by the teammate themselves**, from the repository root with `VAANI_DATA_ROOT` set:
   ```bash
   python scripts/acquire_kathbath.py --subset 300 --accept-terms
   python scripts/acquire_csfleurs.py --metadata --accept-terms
   python scripts/acquire_csfleurs.py --files data/manifests/hinglish_pilot_plan.csv --accept-terms
   python -m ml.data.build_codec_variants --config ml/configs/codec_pilot.yaml
   python -m ml.data.build_codec_variants --config ml/configs/codec_robust_smoke.yaml
   python scripts/verify_data.py          # must end with: OK: all audio present and unchanged
   ```
   Then the full ASVspoof download and the rest of `docs/GPU_HANDOFF.md`.

## 5. Conditions (all required before transfer)
- **C1:** commit + push the repository to a private remote; add the teammate as a collaborator.
- **C2:** send **only** the restricted archive above. Do **not** send `raw/kathbath_hi`, `raw/cs_fleurs`,
  `raw/in_the_wild` or `processed/codec_sim`.
- **C3 (resolved 2026-10-08: Hindi removed from both configs before any results):** Hindi stays out of every reported number.
  - Before the teammate runs step 9, the team decides to either remove the `hi` pool from
    `robustness_full_gpu.yaml` (allowed: no results exist yet), or keep it and treat those rows strictly as audit
    output.
  - The Hindi dev/test parents in `codec_full_gpu.yaml` are harmless (eval-only variants), but their results are
    covered by the same rule.
- **C4:** the teammate follows `docs/GPU_HANDOFF.md` (step 0 updated with C2 and §4.3), and the integrity rules there.

## 6. What still needs fixing (not blocking the English GPU work)
1. **Hindi v2.1 → research use** (the repair itself is done):
   - add a class-neutral headroom guard before any codec/PCM step (29 real clips exceed ±1.0);
   - build a larger Hindi set so dev and test each contain enough speaker groups (pilot: 11 groups, dev = 1);
   - re-run the acceptance gate, the diagnostic probes and the per-model pause diagnostic on that build.
2. **Licence verification:** Kathbath (ask AI4Bharat; read the HF gate terms), CS-FLEURS (ask the authors),
   In-the-Wild. Record dated answers in `licence_registry.yaml` and update `LICENSES_REGISTER.md` (the register was
   brought in line with the registry on 2026-10-08).
3. **Hinglish pilot** issues (train-split source confound; real speech from the same conditions as the fakes;
   D-P1..D-P3).
4. **Codec decisions:** AMR availability on the GPU machine; freeze bitrates (D-C1, D-C2).
