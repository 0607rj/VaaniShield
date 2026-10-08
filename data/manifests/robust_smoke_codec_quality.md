# Codec variant quality: CODEC_robust_smoke_v1

36 parents; tools {'pyav': '19.0.1', 'libavcodec': '63.1.102', 'soxr': '1.1.0', 'numpy': '2.5.3'}

| condition | n | role | kbps_requested | kbps_actual_mean | delay_ms_median | snr_db_median | lsd_db_median | bw_out_hz_median | bw_ref_hz_median | rms_change_db_median | clipped_frac_max | n_unavailable | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aac_24k | 36.0 | eval_heldout | 24.0 | 26.4 | 64.0 | 16.905 | 10.76 | 7531.0 | 8000.0 | -0.04 | 0.00019265038770890526 |  |  |
| aac_48k | 36.0 | eval_heldout | 48.0 | 52.7 | 64.0 | 24.73 | 4.8149999999999995 | 8000.0 | 8000.0 | -0.05 | 0.00048416975412596833 |  |  |
| amr_nb_12k2 |  | train_aug |  |  |  |  |  |  |  |  |  | 36.0 | amr_nb_12k2: no system ffmpeg binary on PATH |
| amr_wb_12k65 |  | eval_heldout |  |  |  |  |  |  |  |  |  | 36.0 | amr_wb_12k65: no system ffmpeg binary on PATH |
| g711_mulaw | 36.0 | train_aug |  | 64.2 | 0.0 | 21.53 | 31.645 | 3844.0 | 8000.0 | -0.025 | 5.1614235206069835e-05 |  |  |
| g722 | 36.0 | eval_heldout |  | 64.2 | 1.4 | 30.015 | 6.039999999999999 | 8000.0 | 8000.0 | 0.0 | 0.00011611091759291513 |  |  |
| mp3_16k | 36.0 | train_aug | 16.0 | 16.7 | 0.0 | 17.29 | 29.605 | 5500.0 | 8000.0 | -0.44 | 8.444430370393827e-05 |  |  |
| mp3_32k | 36.0 | train_aug | 32.0 | 33.0 | 0.0 | 20.83 | 14.405000000000001 | 7250.0 | 8000.0 | -0.44 | 0.0 |  |  |
| opus_voip_16k | 36.0 | train_aug | 16.0 | 15.6 | -0.1 | 10.59 | 8.925 | 7562.0 | 8000.0 | -0.225 | 9.703085581214826e-05 |  |  |
| opus_voip_32k | 36.0 | train_aug | 32.0 | 30.0 | -0.0 | 11.08 | 7.8 | 7562.0 | 8000.0 | -0.07500000000000001 | 6.468723720809884e-05 |  |  |
| opus_voip_8k | 36.0 | train_aug | 8.0 | 8.1 | 0.1 | 8.89 | 26.765 | 4562.5 | 8000.0 | -0.45999999999999996 | 0.00023222183518583026 |  |  |
| resample_22k | 36.0 | train_aug |  |  | 0.0 | 45.31 | 6.045 | 7594.0 | 8000.0 | -0.0 | 4.8162596927226314e-05 |  |  |
| resample_8k | 36.0 | train_aug |  |  | 0.0 | 21.64 | 31.520000000000003 | 3844.0 | 8000.0 | -0.03 | 5.4291763939410394e-05 |  |  |

SNR/LSD are measured against the 16 kHz parent after delay compensation. Low SNR for codecs is expected (perceptual codecs do not preserve waveform); LSD and bandwidth are the more meaningful indicators.
