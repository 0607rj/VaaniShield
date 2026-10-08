# Handoff licence audit: may each file group be given to another team member?

**Date:** 2026-10-08. Evidence was re-checked today from the sources named in each row (HF API and dataset cards at
the pinned revisions, the OpenSLR page, the ASVspoof README saved in `data/manifests/licence_texts/`) and from
`data/manifests/licence_registry.yaml`. This is not legal advice. Where the evidence does not settle a question, the
cell says exactly: "Needs verification from current dataset terms."

**Correction of an earlier statement.** On 2026-10-08, `docs/GPU_HANDOFF.md` and my chat answer said "the licences
allow this sharing within the project". That was stated **without evidence** for every group, and it contradicted our
own registry (Kathbath and In-the-Wild: "redistribution: No (until conflict resolved)"). It is withdrawn; this file
replaces it.

"Team-member transfer" means giving a copy of the files to a teammate. Legally that is redistribution, so it follows
the redistribution column, plus any access terms the teammate would otherwise accept themselves.

## 1. Audio and derived audio in the proposed data bundle (`data/raw`, `data/processed`)

| File group (bundle path) | Size / files | Source (where our copy came from) | Licence (evidence) | Redistribution allowed? | Team-member transfer allowed? | Only original download permitted? | Current terms uncertain? |
|---|---|---|---|---|---|---|---|
| `raw/asvspoof2019_la` (pilot FLAC + CM protocols + README/LICENSE) | 69 MB, 1,078 | Edinburgh DataShare, LA.zip, by range requests | **ODC-By 1.0**. README §COPYING: "You are free to use this database under Open Data Commons Attribution License (ODC-By)" (saved copy) | Yes, with attribution (ODC-By permits sharing with attribution) | Yes, with attribution | No | No |
| `raw/indicsynth_hi` (320 WAV + 2 metadata CSV) | 90 MB, 322 | HF `vdivyasharma/IndicSynth` @ c0a10386, **not gated** | **CC BY-NC 4.0** (card tag + card text: "released under the CC BY-NC 4.0 License. It is intended for non-commercial, academic research only.") | Yes, for **non-commercial** purposes, with attribution and licence notice | Yes, for non-commercial academic research, with attribution | No | No (note: the voices are clones of real Kathbath speakers; use only for detection research) |
| `raw/kathbath_hi` (240 m4a, Hindi test-known) | 24 MB, 240 | AI4Bharat objectstore `clean/testkn_audio.tar` (ungated route A) | **CONFLICT:** CC0 for the "packaging" (IndicSUPERB GitHub README and HF card text) vs **CC-BY-4.0** (HF card licence tag). Card: "We do not own any of the raw text used in creating this dataset" (IndicCorp web crawl). HF mirror is **gated** (auto-approval) | Needs verification from current dataset terms. | Needs verification from current dataset terms. | Needs verification from current dataset terms. (The ungated objectstore download is available to anyone, so the teammate can download it themselves.) | **Yes** |
| `raw/openslr104_hien` (30 lecture WAVs + Kaldi transcripts) | 551 MB, 36 (+444 MB archive, excluded) | OpenSLR SLR104 mirrors (elda) | **CC BY-SA 4.0** (OpenSLR page today: "License: CC BY-SA 4.0") | Yes, with attribution; derivatives must be shared under CC BY-SA 4.0 | Yes, with attribution and share-alike notice | No | No |
| `raw/cs_fleurs` (150 WAV + 3 metadata JSONL incl. transcripts) | 102 MB, 153 | HF `byan/cs-fleurs` @ 0cdbf166, not gated | HF licence tag **cc-by-nc-4.0**, but card text says "licensed under CC BY 4.0 for Non-Commercial use" (non-standard, inconsistent with the tag). Upstream components (FLEURS text, XTTS and MMS synthesis models) carry their own terms, not reviewed | Needs verification from current dataset terms. | Needs verification from current dataset terms. | Needs verification from current dataset terms. (Ungated, so the teammate can download it themselves with the pinned script.) | **Yes** |
| `raw/in_the_wild` (`meta.csv` labels + 1 text file only; no audio) | 1 MB, 2 | HF `mueller91/In-The-Wild` @ eee168f9, not gated | **CONFLICT:** CC BY-SA 4.0 (HF card tag) vs Apache-2.0 (project page per registry) | Needs verification from current dataset terms. | Needs verification from current dataset terms. | Needs verification from current dataset terms. | **Yes** (not needed by the teammate in any case) |
| `processed/openslr104_hien/segments` (347 cut WAVs) | 68 MB | derived from OpenSLR by `cut_openslr_segment` (lossless) | CC BY-SA 4.0 (share-alike derivative) | Yes, under CC BY-SA 4.0 with attribution | Yes | No | No (also regenerable with `python -m ml.data.build_canonical_manifest`) |
| `processed/codec_sim/*/asvspoof2019_la` | 132 files | derived (codec round-trips) | ODC-By 1.0 (parent) | Yes, with attribution | Yes | No | No (regenerable, bit-identical) |
| `processed/codec_sim/*/openslr104_hien` | 38 files | derived | CC BY-SA 4.0 (parent) | Yes, share-alike | Yes | No | No (regenerable) |
| `processed/codec_sim/*/indicsynth_hi` | 66 files | derived | CC BY-NC 4.0 (parent) | Yes, non-commercial, attribution | Yes, non-commercial | No | No (regenerable) |
| `processed/codec_sim/*/kathbath_hi` | 66 files | derived | inherits the Kathbath conflict | Needs verification from current dataset terms. | Needs verification from current dataset terms. | Needs verification from current dataset terms. | **Yes** |
| `processed/codec_sim/*/cs_fleurs` | 214 files | derived | inherits the CS-FLEURS inconsistency | Needs verification from current dataset terms. | Needs verification from current dataset terms. | Needs verification from current dataset terms. | **Yes** |

## 2. Items in the Git repository (what a clone contains)

| Item | Licence (evidence) | Redistribution / transfer | Uncertain? |
|---|---|---|---|
| Our code, configs, docs, tests | Project code licence still undecided (PROJECT_STATE Q-L). Sharing within the team is the authors' own decision | Within the team: yes (authors' own work) | No for internal transfer; the public licence is open |
| `ml/models/vendored/aasist/*` (AASIST code, NAVER) | **MIT** (LICENSE + NOTICE included in the folder) | Yes, keep the copyright notice (already included) | No |
| `data/manifests/licence_texts/asvspoof2019_la/` (README, LICENSE_text) | ODC-By 1.0 dataset documentation | Yes, with attribution | No |
| Manifests/plans/reports for ASVspoof, OpenSLR, IndicSynth rows (paths, speaker IDs, sha256, *hashed* transcripts; no audio, no text) | Derived metadata; parent licences permit redistribution of derivatives (ODC-By, CC BY-SA, CC BY-NC) | Yes, under the parent licences' conditions | No |
| Manifest rows for Kathbath, CS-FLEURS, In-the-Wild (identifiers and hashes only; no audio, no transcript text) | Parent terms unresolved (above) | Needs verification from current dataset terms. | **Yes** |
| `data/manifests/provenance/*.json` (URLs, sha256, file lists) | Our own records of public URLs and hashes | Yes (contains no dataset content) | No |

## 3. Not in the bundle; the teammate downloads these directly (allowed for anyone)
| Item | Licence (evidence) | Note |
|---|---|---|
| Full ASVspoof 2019 LA (7.1 GB) | ODC-By 1.0 | `scripts/acquire_asvspoof.py --full --confirm-large --accept-terms` |
| XLS-R 300M weights (`facebook/wav2vec2-xls-r-300m` @ 1a640f32) | **Apache-2.0** (HF model licence tag, checked today; not gated) | downloaded automatically into `HF_HOME` |
| Python packages | their own licences, pinned in `environment/lock/` | installed by `uv` |

## 4. Consequence for the handoff
- **May be copied** (evidence of permission, with attribution / NC / SA conditions):
  - `raw/asvspoof2019_la`, `raw/indicsynth_hi`, `raw/openslr104_hien` (without the archive);
  - `processed/openslr104_hien`;
  - the `codec_sim` variants of ASVspoof, OpenSLR and IndicSynth.
- **Do not copy**; the teammate downloads from the original source with the pinned scripts:
  - `raw/kathbath_hi`;
  - `raw/cs_fleurs`;
  - `codec_sim/*/{kathbath_hi,cs_fleurs}`.

  These are reproducible: same files, verified by `scripts/verify_data.py`.
- **Not needed:** `raw/in_the_wild`.
- **Conditions travelling with the copy:**
  - include `LICENSES_REGISTER.md` (attribution);
  - non-commercial use only (IndicSynth);
  - share-alike for OpenSLR derivatives;
  - no onward redistribution outside the project.
- **To resolve the "Needs verification" rows**, someone on the team must:
  - **Kathbath:** read the current HF gate terms with their own account and ask AI4Bharat which licence (CC0 vs
    CC-BY-4.0) applies to the audio.
  - **CS-FLEURS:** ask the authors to confirm "CC BY-NC 4.0" (the tag) versus the card text.
  - **In-the-Wild:** confirm which licence applies.

  Record each answer, dated, in `licence_registry.yaml`.
