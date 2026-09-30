# Verification of the final report's numbers (2026-09-28)

Pass run while `research/FINAL_REPORT.md` was refreshed on 2026-09-28. Coverage: every 2026-09-28 number the report
states, and every 2026-09-27 number in the scorecard, the executive summary and the section verdicts, was read back
from the file it cites; the cells of the 2026-09-27 per-task tables (§1.1-1.2, §2-7) that are not in those places
were checked by that day's report pass and here only by spot checks (six E2E cells, the ASR, speaker and turn rows). The check is
mechanical: a claim list (section, statement, file, key, stated value, scale) is evaluated against the JSON and a
claim passes when the stored value, times the scale (x100 for rates quoted in %), rounds to the stated digits.
Where the stated value came from a research note and the JSON disagrees, the note was corrected (never the JSON) and
the case is listed in section 2. "scratch" = `/Volumes/ExternalSSD/nvidia-audio-models/scratch` (the external SSD);
three tables of the day live only there and are named as such in the report. No model was loaded for this pass
(reading JSON only; one `pytest --collect-only` through `scripts/dev/gate.sh` to count tests).

Result: **551 claims checked, 543 match, 8 do not**; the 8 are the two DIARIZATION_FIX.md tables in section 2
(items 1 and 2). Six further disagreements that are not a single stored number (items 3-8) were found by reading.

## 1. Claim table

Legend: section numbers follow the report (1 ASR ... 7 product); letters mark 2026-09-28 additions (D diarization
fix and distillation, P performance, T TS-VAD turn path, N negative results, F latency / FC frontier, S speaker head,
R frontier pilots). "(x-100, sign flipped)" = the JSON stores Silero − ours; the report states ours − Silero.

| # | section | claim | file:key | value read | stated | ok |
|---|---|---|---|---|---|---|
| 1 | 1.1 ASR Libri | served WER % | `runs/final_asr.json`:`results/served/libri/wer_normalize_text/wer` | 0.0229 (x100) | 2.29 | yes |
| 2 | 1.1 ASR Libri | parakeet WER % | `runs/final_asr.json`:`results/parakeet/libri/wer_normalize_text/wer` | 0.0168 (x100) | 1.68 | yes |
| 3 | 1.1 ASR Libri | whisper_small WER % | `runs/final_asr.json`:`results/whisper_small/libri/wer_normalize_text/wer` | 0.0242 (x100) | 2.42 | yes |
| 4 | 1.1 ASR Libri | whisper_turbo WER % | `runs/final_asr.json`:`results/whisper_turbo/libri/wer_normalize_text/wer` | 0.0147 (x100) | 1.47 | yes |
| 5 | 1.1 ASR Libri | served RTF | `runs/final_asr.json`:`results/served/libri/rtf_cpu2` | 0.0154 | 0.015 | yes |
| 6 | 1.1 ASR Libri | parakeet RTF | `runs/final_asr.json`:`results/parakeet/libri/rtf_cpu2` | 0.0402 | 0.040 | yes |
| 7 | 1.1 ASR Libri | whisper_small RTF | `runs/final_asr.json`:`results/whisper_small/libri/rtf_cpu2` | 0.1834 | 0.183 | yes |
| 8 | 1.1 ASR Libri | whisper_turbo RTF | `runs/final_asr.json`:`results/whisper_turbo/libri/rtf_cpu2` | 0.8819 | 0.882 | yes |
| 9 | 1.1 ASR Libri | served - whisper_small delta | `runs/final_asr.json`:`paired_served_minus/whisper_small/libri/delta` | -0.0013 (x100) | -0.13 | yes |
| 10 | 1.1 ASR Libri | served - whisper_small CI lo | `runs/final_asr.json`:`paired_served_minus/whisper_small/libri/ci95[0]` | -0.0056 (x100) | -0.56 | yes |
| 11 | 1.1 ASR Libri | served - whisper_small CI hi | `runs/final_asr.json`:`paired_served_minus/whisper_small/libri/ci95[1]` | 0.003 (x100) | 0.30 | yes |
| 12 | 1.1 ASR Libri | served - turbo delta | `runs/final_asr.json`:`paired_served_minus/whisper_turbo/libri/delta` | 0.0082 (x100) | 0.82 | yes |
| 13 | 1.1 ASR Libri | served - parakeet delta | `runs/final_asr.json`:`paired_served_minus/parakeet/libri/delta` | 0.006 (x100) | 0.60 | yes |
| 14 | 1.1 ASR Libri | TDT v3 WER % | `runs/hybrid_asr.json`:`english/results/tdt_v3/libri/wer_normalize_text/wer` | 0.0203 (x100) | 2.03 | yes |
| 15 | 1.1 ASR Libri | TDT v3 - served delta | `runs/hybrid_asr.json`:`english/paired/tdt_v3 - served_la1 / libri / normalize_text/delta` | -0.0026 (x100) | -0.26 | yes |
| 16 | 1.2 ASR AMI | served WER % | `runs/final_asr.json`:`results/served/ami/wer_normalize_text/wer` | 0.2443 (x100) | 24.43 | yes |
| 17 | 1.2 ASR AMI | served WER % Whisper norm | `runs/final_asr.json`:`results/served/ami/wer_whisper_norm/wer` | 0.2063 (x100) | 20.63 | yes |
| 18 | 1.2 ASR AMI | parakeet WER % | `runs/final_asr.json`:`results/parakeet/ami/wer_normalize_text/wer` | 0.1932 (x100) | 19.32 | yes |
| 19 | 1.2 ASR AMI | parakeet WER % Whisper norm | `runs/final_asr.json`:`results/parakeet/ami/wer_whisper_norm/wer` | 0.1414 (x100) | 14.14 | yes |
| 20 | 1.2 ASR AMI | whisper_small WER % | `runs/final_asr.json`:`results/whisper_small/ami/wer_normalize_text/wer` | 0.2116 (x100) | 21.16 | yes |
| 21 | 1.2 ASR AMI | whisper_small WER % Whisper norm | `runs/final_asr.json`:`results/whisper_small/ami/wer_whisper_norm/wer` | 0.1443 (x100) | 14.43 | yes |
| 22 | 1.2 ASR AMI | whisper_turbo WER % | `runs/final_asr.json`:`results/whisper_turbo/ami/wer_normalize_text/wer` | 0.1963 (x100) | 19.63 | yes |
| 23 | 1.2 ASR AMI | whisper_turbo WER % Whisper norm | `runs/final_asr.json`:`results/whisper_turbo/ami/wer_whisper_norm/wer` | 0.1294 (x100) | 12.94 | yes |
| 24 | 1.2 ASR AMI | served RTF | `runs/final_asr.json`:`results/served/ami/rtf_cpu2` | 0.0223 | 0.022 | yes |
| 25 | 1.2 ASR AMI | parakeet RTF | `runs/final_asr.json`:`results/parakeet/ami/rtf_cpu2` | 0.049 | 0.049 | yes |
| 26 | 1.2 ASR AMI | whisper_small RTF | `runs/final_asr.json`:`results/whisper_small/ami/rtf_cpu2` | 0.2801 | 0.280 | yes |
| 27 | 1.2 ASR AMI | whisper_turbo RTF | `runs/final_asr.json`:`results/whisper_turbo/ami/rtf_cpu2` | 1.4694 | 1.469 | yes |
| 28 | 1.2 ASR AMI | served - parakeet delta | `runs/final_asr.json`:`paired_served_minus/parakeet/ami/delta` | 0.0511 (x100) | 5.11 | yes |
| 29 | 1.2 ASR AMI | served - parakeet CI lo | `runs/final_asr.json`:`paired_served_minus/parakeet/ami/ci95[0]` | 0.0338 (x100) | 3.38 | yes |
| 30 | 1.2 ASR AMI | served - parakeet CI hi | `runs/final_asr.json`:`paired_served_minus/parakeet/ami/ci95[1]` | 0.0714 (x100) | 7.14 | yes |
| 31 | 1.2 ASR AMI | served - whisper_small delta | `runs/final_asr.json`:`paired_served_minus/whisper_small/ami/delta` | 0.0327 (x100) | 3.27 | yes |
| 32 | 1.2 ASR AMI | served - whisper_small CI lo | `runs/final_asr.json`:`paired_served_minus/whisper_small/ami/ci95[0]` | 0.0122 (x100) | 1.22 | yes |
| 33 | 1.2 ASR AMI | served - whisper_small CI hi | `runs/final_asr.json`:`paired_served_minus/whisper_small/ami/ci95[1]` | 0.0585 (x100) | 5.85 | yes |
| 34 | 1.2 ASR AMI | served - whisper_turbo delta | `runs/final_asr.json`:`paired_served_minus/whisper_turbo/ami/delta` | 0.048 (x100) | 4.80 | yes |
| 35 | 1.2 ASR AMI | served - whisper_turbo CI lo | `runs/final_asr.json`:`paired_served_minus/whisper_turbo/ami/ci95[0]` | 0.0253 (x100) | 2.53 | yes |
| 36 | 1.2 ASR AMI | served - whisper_turbo CI hi | `runs/final_asr.json`:`paired_served_minus/whisper_turbo/ami/ci95[1]` | 0.0741 (x100) | 7.41 | yes |
| 37 | 1.2 ASR AMI | TDT v3 WER % | `runs/hybrid_asr.json`:`english/results/tdt_v3/ami/wer_normalize_text/wer` | 0.0972 (x100) | 9.72 | yes |
| 38 | 1.2 ASR AMI | TDT v3 WER % Whisper norm | `runs/hybrid_asr.json`:`english/results/tdt_v3/ami/wer_whisper_norm/wer` | 0.095 (x100) | 9.50 | yes |
| 39 | 1.2 ASR AMI | TDT v3 RTF | `runs/hybrid_asr.json`:`english/results/tdt_v3/ami/rtf_logged` | 0.0646 | 0.065 | yes |
| 40 | 1.2 ASR AMI | TDT v3 - served delta | `runs/hybrid_asr.json`:`english/paired/tdt_v3 - served_la1 / ami / normalize_text/delta` | -0.147 (x100) | -14.70 | yes |
| 41 | 1.2 ASR AMI | TDT v3 - served CI lo | `runs/hybrid_asr.json`:`english/paired/tdt_v3 - served_la1 / ami / normalize_text/ci95[0]` | -0.1702 (x100) | -17.02 | yes |
| 42 | 1.2 ASR AMI | TDT v3 - served CI hi | `runs/hybrid_asr.json`:`english/paired/tdt_v3 - served_la1 / ami / normalize_text/ci95[1]` | -0.1271 (x100) | -12.71 | yes |
| 43 | 1.2 ASR AMI | card AMI test (P) | `runs/hybrid_asr.json`:`english/card/ami_test` | 11.31 | 11.31 | yes |
| 44 | 1.3 dual lookahead | [70,13] ami WER % | `runs/hybrid_asr.json`:`english/results/served_la13/ami/wer_normalize_text/wer` | 0.2302 (x100) | 23.02 | yes |
| 45 | 1.3 dual lookahead | [70,13] icsi WER % | `runs/hybrid_asr.json`:`english/results/served_la13/icsi/wer_normalize_text/wer` | 0.2478 (x100) | 24.78 | yes |
| 46 | 1.3 dual lookahead | [70,13] libri WER % | `runs/hybrid_asr.json`:`english/results/served_la13/libri/wer_normalize_text/wer` | 0.0192 (x100) | 1.92 | yes |
| 47 | 1.3 dual lookahead | [70,1] ICSI WER % | `runs/hybrid_asr.json`:`english/results/served_la1/icsi/wer_normalize_text/wer` | 0.2729 (x100) | 27.29 | yes |
| 48 | 1.3 dual lookahead | [70,13]-[70,1] ami | `runs/hybrid_asr.json`:`english/paired/served_la13 - served_la1 / ami / normalize_text/delta` | -0.0141 (x100) | -1.41 | yes |
| 49 | 1.3 dual lookahead | [70,13]-[70,1] ami CI lo | `runs/hybrid_asr.json`:`english/paired/served_la13 - served_la1 / ami / normalize_text/ci95[0]` | -0.0248 (x100) | -2.48 | yes |
| 50 | 1.3 dual lookahead | [70,13]-[70,1] ami CI hi | `runs/hybrid_asr.json`:`english/paired/served_la13 - served_la1 / ami / normalize_text/ci95[1]` | -0.0032 (x100) | -0.32 | yes |
| 51 | 1.3 dual lookahead | [70,13]-[70,1] icsi | `runs/hybrid_asr.json`:`english/paired/served_la13 - served_la1 / icsi / normalize_text/delta` | -0.025 (x100) | -2.50 | yes |
| 52 | 1.3 dual lookahead | [70,13]-[70,1] icsi CI lo | `runs/hybrid_asr.json`:`english/paired/served_la13 - served_la1 / icsi / normalize_text/ci95[0]` | -0.038 (x100) | -3.80 | yes |
| 53 | 1.3 dual lookahead | [70,13]-[70,1] icsi CI hi | `runs/hybrid_asr.json`:`english/paired/served_la13 - served_la1 / icsi / normalize_text/ci95[1]` | -0.0132 (x100) | -1.32 | yes |
| 54 | 1.3 dual lookahead | [70,13]-[70,1] libri | `runs/hybrid_asr.json`:`english/paired/served_la13 - served_la1 / libri / normalize_text/delta` | -0.0037 (x100) | -0.37 | yes |
| 55 | 1.3 dual lookahead | [70,13]-[70,1] libri CI lo | `runs/hybrid_asr.json`:`english/paired/served_la13 - served_la1 / libri / normalize_text/ci95[0]` | -0.0064 (x100) | -0.64 | yes |
| 56 | 1.3 dual lookahead | [70,13]-[70,1] libri CI hi | `runs/hybrid_asr.json`:`english/paired/served_la13 - served_la1 / libri / normalize_text/ci95[1]` | -0.0011 (x100) | -0.11 | yes |
| 57 | 1.4 decoder adapt | AMI adapted WER % | `runs/hybrid_asr.json`:`adapt/_adapt/sets/ami/normalize_text/adapted/wer` | 0.225 (x100) | 22.50 | yes |
| 58 | 1.4 decoder adapt | ami adapted-served | `runs/hybrid_asr.json`:`adapt/_adapt/sets/ami/normalize_text/adapted - served/delta` | -0.0193 (x100) | -1.93 | yes |
| 59 | 1.4 decoder adapt | ami CI lo | `runs/hybrid_asr.json`:`adapt/_adapt/sets/ami/normalize_text/adapted - served/ci95[0]` | -0.0277 (x100) | -2.77 | yes |
| 60 | 1.4 decoder adapt | ami CI hi | `runs/hybrid_asr.json`:`adapt/_adapt/sets/ami/normalize_text/adapted - served/ci95[1]` | -0.0124 (x100) | -1.24 | yes |
| 61 | 1.4 decoder adapt | icsi adapted-served | `runs/hybrid_asr.json`:`adapt/_adapt/sets/icsi/normalize_text/adapted - served/delta` | -0.0012 (x100) | -0.12 | yes |
| 62 | 1.4 decoder adapt | icsi CI lo | `runs/hybrid_asr.json`:`adapt/_adapt/sets/icsi/normalize_text/adapted - served/ci95[0]` | -0.0078 (x100) | -0.78 | yes |
| 63 | 1.4 decoder adapt | icsi CI hi | `runs/hybrid_asr.json`:`adapt/_adapt/sets/icsi/normalize_text/adapted - served/ci95[1]` | 0.0056 (x100) | 0.56 | yes |
| 64 | 1.4 decoder adapt | libri adapted-served | `runs/hybrid_asr.json`:`adapt/_adapt/sets/libri/normalize_text/adapted - served/delta` | 0.0004 (x100) | 0.04 | yes |
| 65 | 1.4 decoder adapt | libri CI lo | `runs/hybrid_asr.json`:`adapt/_adapt/sets/libri/normalize_text/adapted - served/ci95[0]` | -0.0009 (x100) | -0.09 | yes |
| 66 | 1.4 decoder adapt | libri CI hi | `runs/hybrid_asr.json`:`adapt/_adapt/sets/libri/normalize_text/adapted - served/ci95[1]` | 0.0018 (x100) | 0.18 | yes |
| 67 | 6 transcript LID | full-utterance supported acc | `runs/hybrid_asr.json`:`lid_text/full/acc_supported` | 1 | 1.0 | yes |
| 68 | 6 transcript LID | full-utterance n | `runs/hybrid_asr.json`:`lid_text/full/n_supported` | 1260 | 1260 | yes |
| 69 | 6 transcript LID | 2 s supported acc % | `runs/hybrid_asr.json`:`lid_text/2s/acc_supported` | 0.9166 (x100) | 91.7 | yes |
| 70 | 6 transcript LID | 2 s n | `runs/hybrid_asr.json`:`lid_text/2s/n_supported` | 300 | 300 | yes |
| 71 | 6 transcript LID | AmberNet same utts full % | `runs/hybrid_asr.json`:`lid_text/ambernet_same_utts_restricted_to_supported/full` | 0.9993 (x100) | 99.9 | yes |
| 72 | 6 transcript LID | AmberNet same utts 2 s % | `runs/hybrid_asr.json`:`lid_text/ambernet_same_utts_restricted_to_supported/2s` | 0.9513 (x100) | 95.1 | yes |
| 73 | 0.6B core | ami la1 115M WER | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la1/normalize_text/115m/wer` | 0.2443 (x100) | 24.43 | yes |
| 74 | 0.6B core | ami la1 0.6B WER | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la1/normalize_text/0p6b/wer` | 0.1116 (x100) | 11.16 | yes |
| 75 | 0.6B core | ami la1 delta | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la1/normalize_text/0p6b - 115m/delta` | -0.1327 (x100) | -13.27 | yes |
| 76 | 0.6B core | ami la1 CI lo | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la1/normalize_text/0p6b - 115m/ci95[0]` | -0.1548 (x100) | -15.48 | yes |
| 77 | 0.6B core | ami la1 CI hi | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la1/normalize_text/0p6b - 115m/ci95[1]` | -0.1123 (x100) | -11.23 | yes |
| 78 | 0.6B core | ami la13 115M WER | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la13/normalize_text/115m/wer` | 0.2302 (x100) | 23.02 | yes |
| 79 | 0.6B core | ami la13 0.6B WER | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la13/normalize_text/0p6b/wer` | 0.0981 (x100) | 9.81 | yes |
| 80 | 0.6B core | ami la13 delta | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la13/normalize_text/0p6b - 115m/delta` | -0.1321 (x100) | -13.21 | yes |
| 81 | 0.6B core | ami la13 CI lo | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la13/normalize_text/0p6b - 115m/ci95[0]` | -0.1548 (x100) | -15.48 | yes |
| 82 | 0.6B core | ami la13 CI hi | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la13/normalize_text/0p6b - 115m/ci95[1]` | -0.1119 (x100) | -11.19 | yes |
| 83 | 0.6B core | icsi la1 115M WER | `runs/hybrid_asr.json`:`core_0p6b/sets/icsi/la1/normalize_text/115m/wer` | 0.2729 (x100) | 27.29 | yes |
| 84 | 0.6B core | icsi la1 0.6B WER | `runs/hybrid_asr.json`:`core_0p6b/sets/icsi/la1/normalize_text/0p6b/wer` | 0.1435 (x100) | 14.35 | yes |
| 85 | 0.6B core | icsi la1 delta | `runs/hybrid_asr.json`:`core_0p6b/sets/icsi/la1/normalize_text/0p6b - 115m/delta` | -0.1294 (x100) | -12.94 | yes |
| 86 | 0.6B core | icsi la1 CI lo | `runs/hybrid_asr.json`:`core_0p6b/sets/icsi/la1/normalize_text/0p6b - 115m/ci95[0]` | -0.1511 (x100) | -15.11 | yes |
| 87 | 0.6B core | icsi la1 CI hi | `runs/hybrid_asr.json`:`core_0p6b/sets/icsi/la1/normalize_text/0p6b - 115m/ci95[1]` | -0.109 (x100) | -10.90 | yes |
| 88 | 0.6B core | icsi la13 115M WER | `runs/hybrid_asr.json`:`core_0p6b/sets/icsi/la13/normalize_text/115m/wer` | 0.2478 (x100) | 24.78 | yes |
| 89 | 0.6B core | icsi la13 0.6B WER | `runs/hybrid_asr.json`:`core_0p6b/sets/icsi/la13/normalize_text/0p6b/wer` | 0.1239 (x100) | 12.39 | yes |
| 90 | 0.6B core | icsi la13 delta | `runs/hybrid_asr.json`:`core_0p6b/sets/icsi/la13/normalize_text/0p6b - 115m/delta` | -0.1239 (x100) | -12.39 | yes |
| 91 | 0.6B core | icsi la13 CI lo | `runs/hybrid_asr.json`:`core_0p6b/sets/icsi/la13/normalize_text/0p6b - 115m/ci95[0]` | -0.1495 (x100) | -14.95 | yes |
| 92 | 0.6B core | icsi la13 CI hi | `runs/hybrid_asr.json`:`core_0p6b/sets/icsi/la13/normalize_text/0p6b - 115m/ci95[1]` | -0.1013 (x100) | -10.13 | yes |
| 93 | 0.6B core | libri la1 115M WER | `runs/hybrid_asr.json`:`core_0p6b/sets/libri/la1/normalize_text/115m/wer` | 0.0229 (x100) | 2.29 | yes |
| 94 | 0.6B core | libri la1 0.6B WER | `runs/hybrid_asr.json`:`core_0p6b/sets/libri/la1/normalize_text/0p6b/wer` | 0.022 (x100) | 2.20 | yes |
| 95 | 0.6B core | libri la1 delta | `runs/hybrid_asr.json`:`core_0p6b/sets/libri/la1/normalize_text/0p6b - 115m/delta` | -0.0009 (x100) | -0.09 | yes |
| 96 | 0.6B core | libri la1 CI lo | `runs/hybrid_asr.json`:`core_0p6b/sets/libri/la1/normalize_text/0p6b - 115m/ci95[0]` | -0.0046 (x100) | -0.46 | yes |
| 97 | 0.6B core | libri la1 CI hi | `runs/hybrid_asr.json`:`core_0p6b/sets/libri/la1/normalize_text/0p6b - 115m/ci95[1]` | 0.0028 (x100) | 0.28 | yes |
| 98 | 0.6B core | libri la13 115M WER | `runs/hybrid_asr.json`:`core_0p6b/sets/libri/la13/normalize_text/115m/wer` | 0.0192 (x100) | 1.92 | yes |
| 99 | 0.6B core | libri la13 0.6B WER | `runs/hybrid_asr.json`:`core_0p6b/sets/libri/la13/normalize_text/0p6b/wer` | 0.0203 (x100) | 2.03 | yes |
| 100 | 0.6B core | libri la13 delta | `runs/hybrid_asr.json`:`core_0p6b/sets/libri/la13/normalize_text/0p6b - 115m/delta` | 0.0011 (x100) | 0.11 | yes |
| 101 | 0.6B core | libri la13 CI lo | `runs/hybrid_asr.json`:`core_0p6b/sets/libri/la13/normalize_text/0p6b - 115m/ci95[0]` | -0.0023 (x100) | -0.23 | yes |
| 102 | 0.6B core | libri la13 CI hi | `runs/hybrid_asr.json`:`core_0p6b/sets/libri/la13/normalize_text/0p6b - 115m/ci95[1]` | 0.0047 (x100) | 0.47 | yes |
| 103 | 0.6B core | AMI la1 Whisper-norm 115M | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la1/whisper_norm/115m/wer` | 0.2063 (x100) | 20.6 | yes |
| 104 | 0.6B core | AMI la1 Whisper-norm 0.6B | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la1/whisper_norm/0p6b/wer` | 0.1039 (x100) | 10.4 | yes |
| 105 | 0.6B core | AMI la1 Whisper-norm delta | `runs/hybrid_asr.json`:`core_0p6b/sets/ami/la1/whisper_norm/0p6b - 115m/delta` | -0.1024 (x100) | -10.2 | yes |
| 106 | 0.6B core | first partial AMI median 0.6B s | `runs/hybrid_asr.json`:`core_0p6b/first_partial/ami/median_s/0p6b` | 1.12 | 1.12 | yes |
| 107 | 0.6B core | first partial AMI delta CI lo | `runs/hybrid_asr.json`:`core_0p6b/first_partial/ami/ci95[0]` | -0.16 | -0.16 | yes |
| 108 | 0.6B core | first partial AMI delta CI hi | `runs/hybrid_asr.json`:`core_0p6b/first_partial/ami/ci95[1]` | 0.16 | 0.16 | yes |
| 109 | 0.6B core | first partial ICSI delta CI hi | `runs/hybrid_asr.json`:`core_0p6b/first_partial/icsi/ci95[1]` | 0.16 | 0.16 | yes |
| 110 | 0.6B core | CPU ms/chunk p50 0.6B | `runs/hybrid_asr.json`:`core_0p6b/rtf/cpu/0p6b/ms_per_160ms_chunk_p50` | 266.1 | 266.1 | yes |
| 111 | 0.6B core | CPU stream RTF 0.6B | `runs/hybrid_asr.json`:`core_0p6b/rtf/cpu/0p6b/stream_rtf_160ms` | 1.677 | 1.68 | yes |
| 112 | 0.6B core | CPU stream RTF 115M | `runs/hybrid_asr.json`:`core_0p6b/rtf/cpu/115m/stream_rtf_160ms` | 0.524 | 0.52 | yes |
| 113 | 0.6B core | CPU ms/chunk p50 115M | `runs/hybrid_asr.json`:`core_0p6b/rtf/cpu/115m/ms_per_160ms_chunk_p50` | 83.6 | 83.6 | yes |
| 114 | 0.6B core | MPS stream RTF 0.6B | `runs/hybrid_asr.json`:`core_0p6b/rtf/mps/0p6b/stream_rtf_160ms` | 0.239 | 0.24 | yes |
| 115 | 0.6B core | MPS allocated MB 0.6B | `runs/hybrid_asr.json`:`core_0p6b/rtf/mps/0p6b/mps_allocated_mb` | 3204.7 | 3204.7 | yes |
| 116 | 0.6B core | CPU RSS MB 0.6B | `runs/hybrid_asr.json`:`core_0p6b/rtf/cpu/0p6b/rss_process_mb` | 3404.5 | 3404.5 | yes |
| 117 | 0.6B core | VAD F1 AMI 115M (scratch) | `scratch/hybrid_asr/core_0p6b_rtf.json`:`vad/115m/ami_dev/f1` | 0.9329 | 0.9329 | yes |
| 118 | 0.6B core | VAD F1 AMI 0.6B (scratch) | `scratch/hybrid_asr/core_0p6b_rtf.json`:`vad/0p6b/ami_dev/f1` | 0.9398 | 0.9398 | yes |
| 119 | 0.6B core | VAD F1 ICSI 115M (scratch) | `scratch/hybrid_asr/core_0p6b_rtf.json`:`vad/115m/icsi_dev/f1` | 0.9039 | 0.9039 | yes |
| 120 | 0.6B core | VAD F1 ICSI 0.6B (scratch) | `scratch/hybrid_asr/core_0p6b_rtf.json`:`vad/0p6b/icsi_dev/f1` | 0.9079 | 0.9079 | yes |
| 121 | 0.6B core | VAD paired AMI delta | `scratch/hybrid_asr/core_0p6b_rtf.json`:`vad/paired_f1/ami_dev/delta` | 0.0069 | 0.0069 | yes |
| 122 | 0.6B core | VAD paired AMI CI lo | `scratch/hybrid_asr/core_0p6b_rtf.json`:`vad/paired_f1/ami_dev/ci95[0]` | 0.0018 | 0.0018 | yes |
| 123 | 0.6B core | VAD paired ICSI delta | `scratch/hybrid_asr/core_0p6b_rtf.json`:`vad/paired_f1/icsi_dev/delta` | 0.004 | 0.0040 | yes |
| 124 | 0.6B core | VAD paired ICSI CI lo | `scratch/hybrid_asr/core_0p6b_rtf.json`:`vad/paired_f1/icsi_dev/ci95[0]` | 0.0004 | 0.0004 | yes |
| 125 | 0.6B core | spk EER AMI n200 within 115M fresh | `runs/spk_frame.json`:`variants/c115_fresh/ami_n200/eer_within_meeting` | 0.1741 (x100) | 17.4 | yes |
| 126 | 0.6B core | spk EER AMI n200 within 0.6B | `runs/spk_frame.json`:`variants/c0p6b_fresh/ami_n200/eer_within_meeting` | 0.1617 (x100) | 16.2 | yes |
| 127 | 0.6B core | spk AMI paired delta | `runs/spk_frame.json`:`paired_within_n200/ami\|c0p6b_fresh-c115_fresh/delta` | -0.0124 (x100) | -1.2 | yes |
| 128 | 0.6B core | spk AMI paired CI lo | `runs/spk_frame.json`:`paired_within_n200/ami\|c0p6b_fresh-c115_fresh/ci95[0]` | -0.0409 (x100) | -4.1 | yes |
| 129 | 0.6B core | spk AMI paired CI hi | `runs/spk_frame.json`:`paired_within_n200/ami\|c0p6b_fresh-c115_fresh/ci95[1]` | 0.0193 (x100) | 1.9 | yes |
| 130 | 0.6B core | spk ICSI n200 within 115M fresh | `runs/spk_frame.json`:`variants/c115_fresh/icsi_n200/eer_within_meeting` | 0.0399 (x100) | 4.0 | yes |
| 131 | 0.6B core | spk ICSI n200 within 0.6B | `runs/spk_frame.json`:`variants/c0p6b_fresh/icsi_n200/eer_within_meeting` | 0.0286 (x100) | 2.9 | yes |
| 132 | 0.6B core | spk ICSI paired delta | `runs/spk_frame.json`:`paired_within_n200/icsi\|c0p6b_fresh-c115_fresh/delta` | -0.0112 (x100) | -1.1 | yes |
| 133 | 0.6B core | spk ICSI paired CI lo | `runs/spk_frame.json`:`paired_within_n200/icsi\|c0p6b_fresh-c115_fresh/ci95[0]` | -0.0265 (x100) | -2.7 | yes |
| 134 | 0.6B core | spk ICSI paired CI hi | `runs/spk_frame.json`:`paired_within_n200/icsi\|c0p6b_fresh-c115_fresh/ci95[1]` | 0.007 (x100) | 0.7 | yes |
| 135 | 0.6B core | ami head_b_eot miss 6 s % | `runs/tsvad_turn_cores.json`:`ami/systems/head_b_eot/6s/miss_rate` | 0.553 (x100) | 55.3 | yes |
| 136 | 0.6B core | ami head_b_0p6b_eot miss 6 s % | `runs/tsvad_turn_cores.json`:`ami/systems/head_b_0p6b_eot/6s/miss_rate` | 0.5645 (x100) | 56.5 | yes |
| 137 | 0.6B core | ami hybrid_dyn_b_eot miss 6 s % | `runs/tsvad_turn_cores.json`:`ami/systems/hybrid_dyn_b_eot/6s/miss_rate` | 0.4412 (x100) | 44.1 | yes |
| 138 | 0.6B core | ami hybrid_dyn_b_0p6b_eot miss 6 s % | `runs/tsvad_turn_cores.json`:`ami/systems/hybrid_dyn_b_0p6b_eot/6s/miss_rate` | 0.4687 (x100) | 46.9 | yes |
| 139 | 0.6B core | icsi head_b_eot miss 6 s % | `runs/tsvad_turn_cores.json`:`icsi/systems/head_b_eot/6s/miss_rate` | 0.3395 (x100) | 34.0 | yes |
| 140 | 0.6B core | icsi head_b_0p6b_eot miss 6 s % | `runs/tsvad_turn_cores.json`:`icsi/systems/head_b_0p6b_eot/6s/miss_rate` | 0.2902 (x100) | 29.0 | yes |
| 141 | 0.6B core | icsi hybrid_dyn_b_eot miss 6 s % | `runs/tsvad_turn_cores.json`:`icsi/systems/hybrid_dyn_b_eot/6s/miss_rate` | 0.3218 (x100) | 32.2 | yes |
| 142 | 0.6B core | icsi hybrid_dyn_b_0p6b_eot miss 6 s % | `runs/tsvad_turn_cores.json`:`icsi/systems/hybrid_dyn_b_0p6b_eot/6s/miss_rate` | 0.2688 (x100) | 26.9 | yes |
| 143 | 0.6B core | ami head_b_eot open miss 6 s % | `runs/tsvad_turn_cores.json`:`ami/systems/head_b_eot/6s/strata/open/miss_rate` | 0.6133 (x100) | 61.3 | yes |
| 144 | 0.6B core | ami head_b_0p6b_eot open miss 6 s % | `runs/tsvad_turn_cores.json`:`ami/systems/head_b_0p6b_eot/6s/strata/open/miss_rate` | 0.5758 (x100) | 57.6 | yes |
| 145 | 0.6B core | ami hybrid_dyn_b_eot open miss 6 s % | `runs/tsvad_turn_cores.json`:`ami/systems/hybrid_dyn_b_eot/6s/strata/open/miss_rate` | 0.2466 (x100) | 24.7 | yes |
| 146 | 0.6B core | ami hybrid_dyn_b_0p6b_eot open miss 6 s % | `runs/tsvad_turn_cores.json`:`ami/systems/hybrid_dyn_b_0p6b_eot/6s/strata/open/miss_rate` | 0.2345 (x100) | 23.4 | yes |
| 147 | 0.6B core | icsi head_b_eot open miss 6 s % | `runs/tsvad_turn_cores.json`:`icsi/systems/head_b_eot/6s/strata/open/miss_rate` | 0.4953 (x100) | 49.5 | yes |
| 148 | 0.6B core | icsi head_b_0p6b_eot open miss 6 s % | `runs/tsvad_turn_cores.json`:`icsi/systems/head_b_0p6b_eot/6s/strata/open/miss_rate` | 0.3628 (x100) | 36.3 | yes |
| 149 | 0.6B core | icsi hybrid_dyn_b_eot open miss 6 s % | `runs/tsvad_turn_cores.json`:`icsi/systems/hybrid_dyn_b_eot/6s/strata/open/miss_rate` | 0.3791 (x100) | 37.9 | yes |
| 150 | 0.6B core | icsi hybrid_dyn_b_0p6b_eot open miss 6 s % | `runs/tsvad_turn_cores.json`:`icsi/systems/hybrid_dyn_b_0p6b_eot/6s/strata/open/miss_rate` | 0.2582 (x100) | 25.8 | yes |
| 151 | 0.6B core | AMI hybrid_dyn P50 115M | `runs/tsvad_turn_cores.json`:`ami/systems/hybrid_dyn_b_eot/6s/p50_ms` | 2960 | 2960 | yes |
| 152 | 0.6B core | AMI hybrid_dyn P50 0.6B | `runs/tsvad_turn_cores.json`:`ami/systems/hybrid_dyn_b_0p6b_eot/6s/p50_ms` | 3920 | 3920 | yes |
| 153 | 0.6B core | ami head all paired | `runs/tsvad_turn_cores.json`:`ami/paired/head_b_0p6b_eot - head_b_eot \| 6s/all/miss_diff` | 0.0115 (x100) | 1.1 | yes |
| 154 | 0.6B core | ami head all CI lo | `runs/tsvad_turn_cores.json`:`ami/paired/head_b_0p6b_eot - head_b_eot \| 6s/all/miss_diff_ci[0]` | -0.015 (x100) | -1.5 | yes |
| 155 | 0.6B core | ami head all CI hi | `runs/tsvad_turn_cores.json`:`ami/paired/head_b_0p6b_eot - head_b_eot \| 6s/all/miss_diff_ci[1]` | 0.04 (x100) | 4.0 | yes |
| 156 | 0.6B core | ami head open paired | `runs/tsvad_turn_cores.json`:`ami/paired/head_b_0p6b_eot - head_b_eot \| 6s/open/miss_diff` | -0.0376 (x100) | -3.8 | yes |
| 157 | 0.6B core | ami head open CI lo | `runs/tsvad_turn_cores.json`:`ami/paired/head_b_0p6b_eot - head_b_eot \| 6s/open/miss_diff_ci[0]` | -0.1031 (x100) | -10.3 | yes |
| 158 | 0.6B core | ami head open CI hi | `runs/tsvad_turn_cores.json`:`ami/paired/head_b_0p6b_eot - head_b_eot \| 6s/open/miss_diff_ci[1]` | 0.0265 (x100) | 2.6 | yes |
| 159 | 0.6B core | ami hybrid_dyn all paired | `runs/tsvad_turn_cores.json`:`ami/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/all/miss_diff` | 0.0275 (x100) | 2.8 | yes |
| 160 | 0.6B core | ami hybrid_dyn all CI lo | `runs/tsvad_turn_cores.json`:`ami/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/all/miss_diff_ci[0]` | 0.0007 (x100) | 0.1 | yes |
| 161 | 0.6B core | ami hybrid_dyn all CI hi | `runs/tsvad_turn_cores.json`:`ami/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/all/miss_diff_ci[1]` | 0.0534 (x100) | 5.3 | yes |
| 162 | 0.6B core | ami hybrid_dyn open paired | `runs/tsvad_turn_cores.json`:`ami/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/open/miss_diff` | -0.0121 (x100) | -1.2 | yes |
| 163 | 0.6B core | ami hybrid_dyn open CI lo | `runs/tsvad_turn_cores.json`:`ami/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/open/miss_diff_ci[0]` | -0.0627 (x100) | -6.3 | yes |
| 164 | 0.6B core | ami hybrid_dyn open CI hi | `runs/tsvad_turn_cores.json`:`ami/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/open/miss_diff_ci[1]` | 0.0384 (x100) | 3.8 | yes |
| 165 | 0.6B core | icsi head all paired | `runs/tsvad_turn_cores.json`:`icsi/paired/head_b_0p6b_eot - head_b_eot \| 6s/all/miss_diff` | -0.0493 (x100) | -4.9 | yes |
| 166 | 0.6B core | icsi head all CI lo | `runs/tsvad_turn_cores.json`:`icsi/paired/head_b_0p6b_eot - head_b_eot \| 6s/all/miss_diff_ci[0]` | -0.0708 (x100) | -7.1 | yes |
| 167 | 0.6B core | icsi head all CI hi | `runs/tsvad_turn_cores.json`:`icsi/paired/head_b_0p6b_eot - head_b_eot \| 6s/all/miss_diff_ci[1]` | -0.0272 (x100) | -2.7 | yes |
| 168 | 0.6B core | icsi head open paired | `runs/tsvad_turn_cores.json`:`icsi/paired/head_b_0p6b_eot - head_b_eot \| 6s/open/miss_diff` | -0.1325 (x100) | -13.2 | yes |
| 169 | 0.6B core | icsi head open CI lo | `runs/tsvad_turn_cores.json`:`icsi/paired/head_b_0p6b_eot - head_b_eot \| 6s/open/miss_diff_ci[0]` | -0.1953 (x100) | -19.5 | yes |
| 170 | 0.6B core | icsi head open CI hi | `runs/tsvad_turn_cores.json`:`icsi/paired/head_b_0p6b_eot - head_b_eot \| 6s/open/miss_diff_ci[1]` | -0.0692 (x100) | -6.9 | yes |
| 171 | 0.6B core | icsi hybrid_dyn all paired | `runs/tsvad_turn_cores.json`:`icsi/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/all/miss_diff` | -0.053 (x100) | -5.3 | yes |
| 172 | 0.6B core | icsi hybrid_dyn all CI lo | `runs/tsvad_turn_cores.json`:`icsi/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/all/miss_diff_ci[0]` | -0.0739 (x100) | -7.4 | yes |
| 173 | 0.6B core | icsi hybrid_dyn all CI hi | `runs/tsvad_turn_cores.json`:`icsi/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/all/miss_diff_ci[1]` | -0.0313 (x100) | -3.1 | yes |
| 174 | 0.6B core | icsi hybrid_dyn open paired | `runs/tsvad_turn_cores.json`:`icsi/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/open/miss_diff` | -0.1209 (x100) | -12.1 | yes |
| 175 | 0.6B core | icsi hybrid_dyn open CI lo | `runs/tsvad_turn_cores.json`:`icsi/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/open/miss_diff_ci[0]` | -0.1782 (x100) | -17.8 | yes |
| 176 | 0.6B core | icsi hybrid_dyn open CI hi | `runs/tsvad_turn_cores.json`:`icsi/paired/hybrid_dyn_b_0p6b_eot - hybrid_dyn_b_eot \| 6s/open/miss_diff_ci[1]` | -0.069 (x100) | -6.9 | yes |
| 177 | 2 VAD | ours F1 AMI | `runs/vad_layers.json`:`q1_probes/served_head/ami_dev/f1` | 0.9485 | 0.949 | yes |
| 178 | 2 VAD | ours miss at FPR 0.075 % | `runs/vad_layers.json`:`q1_probes/served_head/ami_dev/miss_at_fpr0.075` | 0.1207 (x100) | 12.1 | yes |
| 179 | 2 VAD | ours F1 ICSI | `runs/vad_layers.json`:`q1_probes/served_head/icsi_dev/f1` | 0.8999 | 0.900 | yes |
| 180 | 2 VAD | MarbleNet F1 | `runs/baselines_sd.json`:`vad/marblenet_frame_vad_v2/sweep/0.5/f1` | 0.9367 | 0.937 | yes |
| 181 | 2 VAD | Silero v5 F1 | `runs/baselines_sd.json`:`vad/silero_v5.1.2/sweep/0.5/f1` | 0.9147 | 0.915 | yes |
| 182 | 3 speaker | served head ami_n64 within % | `runs/spk_head.json`:`variants/stage1_served/eval/ami_n64/eer_within_meeting` | 0.1438 (x100) | 14.4 | yes |
| 183 | 3 speaker | served head ami_n64 all % | `runs/spk_head.json`:`variants/stage1_served/eval/ami_n64/eer` | 0.0854 (x100) | 8.5 | yes |
| 184 | 3 speaker | served head ami_n200 within % | `runs/spk_head.json`:`variants/stage1_served/eval/ami_n200/eer_within_meeting` | 0.1978 (x100) | 19.8 | yes |
| 185 | 3 speaker | served head ami_n200 all % | `runs/spk_head.json`:`variants/stage1_served/eval/ami_n200/eer` | 0.113 (x100) | 11.3 | yes |
| 186 | 3 speaker | served head icsi_n64 within % | `runs/spk_head.json`:`variants/stage1_served/eval/icsi_n64/eer_within_meeting` | 0.0522 (x100) | 5.2 | yes |
| 187 | 3 speaker | served head icsi_n64 all % | `runs/spk_head.json`:`variants/stage1_served/eval/icsi_n64/eer` | 0.0466 (x100) | 4.7 | yes |
| 188 | 3 speaker | served head icsi_n200 within % | `runs/spk_head.json`:`variants/stage1_served/eval/icsi_n200/eer_within_meeting` | 0.0702 (x100) | 7.0 | yes |
| 189 | 3 speaker | served head icsi_n200 all % | `runs/spk_head.json`:`variants/stage1_served/eval/icsi_n200/eer` | 0.0643 (x100) | 6.4 | yes |
| 190 | 3 speaker | titanet_large n64 within % | `runs/baselines_sd.json`:`spk/titanet_large/n64/eer_within_meeting` | 0.0822 (x100) | 8.2 | yes |
| 191 | 3 speaker | titanet_large n200 within % | `runs/baselines_sd.json`:`spk/titanet_large/n200/eer_within_meeting` | 0.1197 (x100) | 12.0 | yes |
| 192 | 3 speaker | wespeaker_resnet34_lm n64 within % | `runs/baselines_sd.json`:`spk/wespeaker_resnet34_lm/n64/eer_within_meeting` | 0.0963 (x100) | 9.6 | yes |
| 193 | 3 speaker | wespeaker_resnet34_lm n200 within % | `runs/baselines_sd.json`:`spk/wespeaker_resnet34_lm/n200/eer_within_meeting` | 0.1188 (x100) | 11.9 | yes |
| 194 | 3 speaker | speechbrain_ecapa n64 within % | `runs/baselines_sd.json`:`spk/speechbrain_ecapa/n64/eer_within_meeting` | 0.0963 (x100) | 9.6 | yes |
| 195 | 3 speaker | speechbrain_ecapa n200 within % | `runs/baselines_sd.json`:`spk/speechbrain_ecapa/n200/eer_within_meeting` | 0.1304 (x100) | 13.0 | yes |
| 196 | 4 diar | ours DER | `runs/baselines_sd.json`:`ours/diar_frame_der/der_pooled` | 0.3937 | 0.394 | yes |
| 197 | 4 diar | Sortformer v2 DER | `runs/baselines_sd.json`:`sortformer_v2/diar_frame_der/der_pooled` | 0.2014 | 0.201 | yes |
| 198 | 4 diar | pyannote 3.1 DER | `runs/baselines_sd.json`:`pyannote:speaker-diarization-3.1/diar_frame_der/der_pooled` | 0.2204 | 0.220 | yes |
| 199 | 4 diar | Nemotron-3 max DER AMI-64 | `runs/diar_distill.json`:`eval/nemotron3_max/ami/frame_der/der_pooled` | 0.2321 | 0.232 | yes |
| 200 | 4 diar | served head DER ICSI-64 | `runs/diar_distill.json`:`eval/served_stage1/icsi/frame_der/der_pooled` | 0.3541 | 0.354 | yes |
| 201 | 5.1 EOT AMI | hybrid miss 6 s % | `runs/completeness.json`:`dyntimeout/blocks/C_extended_windows_stream/systems/hybrid/6s/fixed_5pct_turn_fc/miss_rate` | 0.6191 (x100) | 61.9 | yes |
| 202 | 5.1 EOT AMI | hybrid CI lo | `runs/completeness.json`:`dyntimeout/blocks/C_extended_windows_stream/systems/hybrid/6s/fixed_5pct_turn_fc/miss_rate_ci[0]` | 0.5873 (x100) | 58.7 | yes |
| 203 | 5.1 EOT AMI | hybrid CI hi | `runs/completeness.json`:`dyntimeout/blocks/C_extended_windows_stream/systems/hybrid/6s/fixed_5pct_turn_fc/miss_rate_ci[1]` | 0.6505 (x100) | 65.0 | yes |
| 204 | 5.1 EOT AMI | hybrid open % | `runs/completeness.json`:`dyntimeout/blocks/C_extended_windows_stream/systems/hybrid/6s/fixed_5pct_turn_fc/strata/open/miss_rate` | 0.4587 (x100) | 45.9 | yes |
| 205 | 5.1 EOT AMI | head OR dyn miss 6 s % | `runs/completeness.json`:`dyntimeout/blocks/C_extended_windows_stream/systems/head_or_dyn_silero\|head/6s/fixed_5pct_turn_fc/miss_rate` | 0.5825 (x100) | 58.2 | yes |
| 206 | 5.1 EOT AMI | head OR dyn open % | `runs/completeness.json`:`dyntimeout/blocks/C_extended_windows_stream/systems/head_or_dyn_silero\|head/6s/fixed_5pct_turn_fc/strata/open/miss_rate` | 0.3148 (x100) | 31.5 | yes |
| 207 | 5.1 EOT AMI | Silero timeout miss % | `runs/completeness.json`:`dyntimeout/blocks/C_extended_windows_stream/systems/silero_timeout/6s/fixed_5pct_turn_fc/miss_rate` | 0.7271 (x100) | 72.7 | yes |
| 208 | 5.1 EOT AMI | Silero timeout open % | `runs/completeness.json`:`dyntimeout/blocks/C_extended_windows_stream/systems/silero_timeout/6s/fixed_5pct_turn_fc/strata/open/miss_rate` | 0.2673 (x100) | 26.7 | yes |
| 209 | 5.1 EOT AMI | head OR dyn - Silero all | `runs/completeness.json`:`dyntimeout/blocks/C_extended_windows_stream/paired/head_or_dyn_silero\|head - silero_timeout \| 6s/all/miss_diff` | -0.1446 (x100) | -14.5 | yes |
| 210 | 5.1 EOT AMI | head OR dyn - Silero open | `runs/completeness.json`:`dyntimeout/blocks/C_extended_windows_stream/paired/head_or_dyn_silero\|head - silero_timeout \| 6s/open/miss_diff` | 0.0475 (x100) | 4.8 | yes |
| 211 | 5.1 EOT AMI | EOU posterior miss % | `runs/baselines_turn.json`:`C_extended_windows_stream/systems/eou_posterior/6s/fixed_5pct_turn_fc/miss_rate` | 0.7007 (x100) | 70.1 | yes |
| 212 | 5.1 EOT AMI | hybrid - Silero (sign flipped) | `runs/baselines_turn.json`:`C_extended_windows_stream/paired/silero_timeout - hybrid_stream_causal_dominant \| 6s/all/miss_diff` | 0.108 (x-100, sign of a − b flipped) | -10.8 | yes |
| 213 | 5.1 EOT AMI | hybrid - Silero CI (flipped hi) | `runs/baselines_turn.json`:`C_extended_windows_stream/paired/silero_timeout - hybrid_stream_causal_dominant \| 6s/all/miss_diff_ci[1]` | 0.1422 (x-100, sign of a − b flipped) | -14.2 | yes |
| 214 | 5.1 EOT AMI | hybrid - Silero CI (flipped lo) | `runs/baselines_turn.json`:`C_extended_windows_stream/paired/silero_timeout - hybrid_stream_causal_dominant \| 6s/all/miss_diff_ci[0]` | 0.07 (x-100, sign of a − b flipped) | -7.0 | yes |
| 215 | 5.1 EOT AMI | hybrid - Silero open (flipped) | `runs/baselines_turn.json`:`C_extended_windows_stream/paired/silero_timeout - hybrid_stream_causal_dominant \| 6s/open/miss_diff` | -0.1914 (x-100, sign of a − b flipped) | 19.1 | yes |
| 216 | 5.1 EOT AMI | hybrid - EOU (flipped) | `runs/baselines_turn.json`:`C_extended_windows_stream/paired/eou_posterior - hybrid_stream_causal_dominant \| 6s/all/miss_diff` | 0.0816 (x-100, sign of a − b flipped) | -8.2 | yes |
| 217 | 5.2 EOT ICSI | head OR dyn all % | `runs/baselines_turn_icsi_dyn.json`:`C_extended_windows_stream/frozen_ami/systems/head_or_dyn_silero\|head/6s/icsi_at_ami_all/miss_rate` | 0.795 (x100) | 79.5 | yes |
| 218 | 5.2 EOT ICSI | head OR dyn open % | `runs/baselines_turn_icsi_dyn.json`:`C_extended_windows_stream/frozen_ami/systems/head_or_dyn_silero\|head/6s/icsi_at_ami_all/strata/open/miss_rate` | 0.5048 (x100) | 50.5 | yes |
| 219 | 5.2 EOT ICSI | head OR dyn FC % | `runs/baselines_turn_icsi_dyn.json`:`C_extended_windows_stream/frozen_ami/systems/head_or_dyn_silero\|head/6s/icsi_at_ami_all/fc_rate` | 0.0183 (x100) | 1.8 | yes |
| 220 | 5.2 EOT ICSI | Silero all % | `runs/baselines_turn_icsi_dyn.json`:`C_extended_windows_stream/frozen_ami/systems/silero_timeout/6s/icsi_at_ami_all/miss_rate` | 0.8735 (x100) | 87.4 | yes |
| 221 | 5.2 EOT ICSI | Silero open % | `runs/baselines_turn_icsi_dyn.json`:`C_extended_windows_stream/frozen_ami/systems/silero_timeout/6s/icsi_at_ami_all/strata/open/miss_rate` | 0.455 (x100) | 45.5 | yes |
| 222 | 5.2 EOT ICSI | Silero FC % | `runs/baselines_turn_icsi_dyn.json`:`C_extended_windows_stream/frozen_ami/systems/silero_timeout/6s/icsi_at_ami_all/fc_rate` | 0.0358 (x100) | 3.6 | yes |
| 223 | 5.2 EOT ICSI | head OR dyn - Silero all | `runs/baselines_turn_icsi_dyn.json`:`C_extended_windows_stream/frozen_ami/paired/head_or_dyn_silero\|head - silero_timeout \| 6s/all/miss_diff` | -0.0785 (x100) | -7.9 | yes |
| 224 | 5.2 EOT ICSI | CI lo | `runs/baselines_turn_icsi_dyn.json`:`C_extended_windows_stream/frozen_ami/paired/head_or_dyn_silero\|head - silero_timeout \| 6s/all/miss_diff_ci[0]` | -0.0995 (x100) | -10.0 | yes |
| 225 | 5.2 EOT ICSI | CI hi | `runs/baselines_turn_icsi_dyn.json`:`C_extended_windows_stream/frozen_ami/paired/head_or_dyn_silero\|head - silero_timeout \| 6s/all/miss_diff_ci[1]` | -0.0576 (x100) | -5.8 | yes |
| 226 | 5.3 TurnBench | predictive OR Silero recall (held-out) | `runs/dyadic_train.json`:`turnbench_predictive/mh/families/predictive_or_silero/tb_halves_pooled/recall` | 0.8619 | 0.862 | yes |
| 227 | 5.3 TurnBench | predictive OR Silero P50 ms | `runs/dyadic_train.json`:`turnbench_predictive/mh/families/predictive_or_silero/tb_halves_pooled/p50` | 1137 | 1137 | yes |
| 228 | 5.3 TurnBench | predictive OR Silero FP | `runs/dyadic_train.json`:`turnbench_predictive/mh/families/predictive_or_silero/tb_halves_pooled/fp_rate` | 0.0922 | 0.092 | yes |
| 229 | 5.3 TurnBench | VAP recall | `runs/turnbench_latency.json`:`reference/published_dev_rescored/vap/recall` | 0.8409 | 0.841 | yes |
| 230 | 5.3 TurnBench | VAP FP | `runs/turnbench_latency.json`:`reference/published_dev_rescored/vap/fp_rate` | 0.0452 | 0.045 | yes |
| 231 | 5.4 oto | hybrid_energy miss 6 s % | `runs/dyadic_train.json`:`oto_dev/C_extended_windows/crossfit/systems/hybrid_energy/6s/fixed_5pct_turn_fc/miss_rate` | 0.0751 (x100) | 7.5 | yes |
| 232 | 5.4 oto | hybrid_energy CI lo | `runs/dyadic_train.json`:`oto_dev/C_extended_windows/crossfit/systems/hybrid_energy/6s/fixed_5pct_turn_fc/miss_rate_ci[0]` | 0.0648 (x100) | 6.5 | yes |
| 233 | 5.4 oto | hybrid_energy CI hi | `runs/dyadic_train.json`:`oto_dev/C_extended_windows/crossfit/systems/hybrid_energy/6s/fixed_5pct_turn_fc/miss_rate_ci[1]` | 0.0867 (x100) | 8.7 | yes |
| 234 | 5.4 oto | primary timeout miss 6 s % | `runs/dyadic_train.json`:`oto_dev/C_extended_windows/crossfit/systems/timeout_primary_oracle/6s/fixed_5pct_turn_fc/miss_rate` | 0.1 (x100) | 10.0 | yes |
| 235 | 6 LID | lid_aug 2 s acc % | `runs/lid.json`:`systems/lid_aug/2s/acc` | 0.7498 (x100) | 75.0 | yes |
| 236 | 6 LID | lid_aug full acc % | `runs/lid.json`:`systems/lid_aug/full/acc` | 0.8929 (x100) | 89.3 | yes |
| 237 | 6 LID | AmberNet 2 s % | `runs/lid.json`:`systems/ambernet/2s/acc` | 0.951 (x100) | 95.1 | yes |
| 238 | 6 LID | AmberNet full % | `runs/lid.json`:`systems/ambernet/full/acc` | 0.9953 (x100) | 99.5 | yes |
| 239 | 7 E2E | Sortformer server total RTF (pipecat C) | `runs/e2e_final.json`:`components_live/pipecat:C/server_total/rtf_wall` | 0.7955 | 0.80 | yes |
| 240 | 7 E2E | Nemotron-3 server total RTF (pipecat CN) | `runs/e2e_final.json`:`components_live/pipecat:CN/server_total/rtf_wall` | 0.6347 | 0.63 | yes |
| 241 | 7 E2E | Nemotron-3 server total RTF (livekit DN) | `runs/e2e_final.json`:`components_live/livekit:DN/server_total/rtf_wall` | 0.6503 | 0.65 | yes |
| 242 | 7 E2E | ASR pass live RTF (pipecat C) | `runs/e2e_final.json`:`components_live/pipecat:C/server_asr/rtf_wall` | 0.1491 | 0.15 | yes |
| 243 | 7 E2E | turn pass live RTF (pipecat C) | `runs/e2e_final.json`:`components_live/pipecat:C/server_turn/rtf_wall` | 0.1497 | 0.15 | yes |
| 244 | 7 E2E | Sortformer diar live RTF | `runs/e2e_final.json`:`components_live/pipecat:C/server_diar/rtf_wall` | 0.496 | 0.50 | yes |
| 245 | 7 E2E | Nemotron-3 diar live RTF | `runs/e2e_final.json`:`components_live/pipecat:CN/server_diar/rtf_wall` | 0.3005 | 0.30 | yes |
| 246 | D diar fix clips | before pooled DER (3 clips) | `scratch/diar/offline/n3_R1_legacy.json`:`aggregate/pooled_der` | 0.395 | 0.395 | yes |
| 247 | D diar fix clips | after (uncut, registry spk, hold) DER | `scratch/diar/offline/n3_R2_reg_hold.json`:`aggregate/pooled_der` | 0.2468 | 0.247 | yes |
| 248 | D diar fix clips | before mean per-final acc | `scratch/diar/offline/n3_R1_legacy.json`:`aggregate/final_speaker_acc_mean` | 0.739 | 0.74 | yes |
| 249 | D diar fix clips | after registry spk + timeout acc | `scratch/diar/offline/n3_R2_reg_hold.json`:`aggregate/final_speaker_acc_mean` | 0.833 | 0.83 | yes |
| 250 | D diar fix clips | after registry spk + timeout_any acc | `scratch/diar/offline/n3_R3_reg_hold_any.json`:`aggregate/final_speaker_acc_mean` | 0.791 | 0.79 | yes |
| 251 | D diar fix clips | after registry TitaNet + timeout_any acc | `scratch/diar/offline/n3_R4_tita_any.json`:`aggregate/final_speaker_acc_mean` | 0.88 | 0.88 | yes |
| 252 | D diar fix clips | mix6 diarizer count after | `scratch/diar/offline/n3_R2_reg_hold.json`:`items/mix6_s0/diar_count` | 6 | 6 | yes |
| 253 | D diar fix clips | ICSI diarizer count after | `scratch/diar/offline/n3_R2_reg_hold.json`:`items/icsi_Bmr021_940s/diar_count` | 5 | 5 | yes |
| 254 | D diar fix clips | mix6 diarizer count before | `scratch/diar/offline/n3_R1_legacy.json`:`items/mix6_s0/diar_count` | 4 | 4 | yes |
| 255 | D diar fix clips | mix6 purity timeout | `scratch/diar/offline/n3_R2_reg_hold.json`:`items/mix6_s0/purity_mean` | 0.819 | 0.82 | yes |
| 256 | D diar fix clips | mix6 purity timeout_any | `scratch/diar/offline/n3_R3_reg_hold_any.json`:`items/mix6_s0/purity_mean` | 0.858 | 0.86 | yes |
| 257 | D diar fix clips | mix6 TitaNet acc | `scratch/diar/offline/n3_R4_tita_any.json`:`items/mix6_s0/final_speaker_acc` | 1 | 1.00 | yes |
| 258 | D diar fix clips | shed legacy DER | `scratch/diar/offline/n3_R5_legacy_shed.json`:`aggregate/pooled_der` | 0.8085 | 0.809 | yes |
| 259 | D diar fix clips | shed hold DER | `scratch/diar/offline/n3_R6_reg_any_shed.json`:`aggregate/pooled_der` | 0.3285 | 0.329 | yes |
| 260 | D diar fix clips | shed legacy RTF | `scratch/diar/offline/n3_R5_legacy_shed.json`:`aggregate/wall_rtf_mean` | 0.27 | 0.27 | yes |
| 261 | D diar fix clips | shed hold RTF | `scratch/diar/offline/n3_R6_reg_any_shed.json`:`aggregate/wall_rtf_mean` | 0.436 | 0.44 | yes |
| 262 | D diar fix clips | Sortformer v2 before DER | `scratch/diar/offline/sf_R1_legacy.json`:`aggregate/pooled_der` | 0.6725 | 0.673 | yes |
| 263 | D diar fix AMI-64 | before DER | `scratch/diar/offline/ami_A_legacy.json`:`aggregate/pooled_der` | 0.2448 | 0.245 | yes |
| 264 | D diar fix AMI-64 | after DER (doc: 0.229) | `scratch/diar/offline/ami_B_uncut_hold.json`:`aggregate/pooled_der` | 0.2448 | 0.229 | **NO** |
| 265 | D diar fix AMI-64 | after per-final acc (doc: 0.772) | `scratch/diar/offline/ami_B_uncut_hold.json`:`aggregate/final_speaker_acc_mean` | 0.765 | 0.772 | **NO** |
| 266 | D diar fix AMI-64 | after finals (doc: 206) | `scratch/diar/offline/ami_B_uncut_hold.json`:`aggregate/n_scored_finals` | 241 | 206 | **NO** |
| 267 | D diar fix AMI-64 | after final ids err (doc: -0.93) | `scratch/diar/offline/ami_B_uncut_hold.json`:`aggregate/final_ids_count_error_mean` | -1.031 | -0.93 | **NO** |
| 268 | D diar fix AMI-64 | after count abs err (doc: 0.19) | `scratch/diar/offline/ami_B_uncut_hold.json`:`aggregate/count_abs_error_mean` | 0.219 | 0.19 | **NO** |
| 269 | D diar fix AMI-64 | before per-final acc | `scratch/diar/offline/ami_A_legacy.json`:`aggregate/final_speaker_acc_mean` | 0.926 | 0.926 | yes |
| 270 | D diar fix AMI-64 | shed legacy DER | `scratch/diar/offline/ami_A_legacy_shed.json`:`aggregate/pooled_der` | 0.4895 | 0.490 | yes |
| 271 | D diar fix AMI-64 | shed hold DER | `scratch/diar/offline/ami_B_uncut_shed.json`:`aggregate/pooled_der` | 0.3353 | 0.335 | yes |
| 272 | D diar fix AMI-64 | shed legacy count err | `scratch/diar/offline/ami_A_legacy_shed.json`:`aggregate/count_error_mean` | -1.281 | -1.28 | yes |
| 273 | D diar fix AMI-64 | shed hold count err | `scratch/diar/offline/ami_B_uncut_shed.json`:`aggregate/count_error_mean` | -0.25 | -0.25 | yes |
| 274 | D diar fix live | mix6 live timeout acc | `scratch/diar/logs/new_timeout_mix6_s0.json`:`sessions_scored[0]/final_speaker_acc` | 0.867 | 0.87 | yes |
| 275 | D diar fix live | mix6 live ids | `scratch/diar/logs/new_timeout_mix6_s0.json`:`sessions_scored[0]/n_ids` | 7 | 7 | yes |
| 276 | D diar fix live | ICSI live timeout acc | `scratch/diar/logs/new_timeout_icsi_Bmr021_940s.json`:`sessions_scored[0]/final_speaker_acc` | 0.8 | 0.80 | yes |
| 277 | D diar fix live | ICSI before live acc | `scratch/diar/logs/n3_icsi_Bmr021_940s.json`:`sessions_scored[0]/final_speaker_acc` | 0.583 | 0.58 | yes |
| 278 | D diar fix live | two sessions RTF (ICSI) | `scratch/diar/logs/new_concurrent.json`:`sessions_scored[0]/rtf` | 0.6102 | 0.61 | yes |
| 279 | D diar fix live | two sessions RTF (mix) | `scratch/diar/logs/new_concurrent.json`:`sessions_scored[1]/rtf` | 0.6657 | 0.67 | yes |
| 280 | D diar fix live | before concurrent mix count err (doc -2) | `scratch/diar/logs/n3_concurrent.json`:`sessions_scored[1]/count_error` | -1 | -2 | **NO** |
| 281 | D diar fix live | before concurrent mix acc (doc 0.68) | `scratch/diar/logs/n3_concurrent.json`:`sessions_scored[1]/final_speaker_acc` | 0.667 | 0.68 | **NO** |
| 282 | D diar fix live | before concurrent mix purity (doc 0.76) | `scratch/diar/logs/n3_concurrent.json`:`sessions_scored[1]/purity_mean` | 0.669 | 0.76 | **NO** |
| 283 | D distill | run1 block6 AMI DER | `runs/diar_distill.json`:`eval/b6_1800/ami/frame_der/der_pooled` | 0.3848 | 0.385 | yes |
| 284 | D distill | run2 blocks4+6 AMI DER | `runs/diar_distill.json`:`eval/mix46_1800/ami/frame_der/der_pooled` | 0.3673 | 0.367 | yes |
| 285 | D distill | run1 ICSI DER | `runs/diar_distill.json`:`eval/b6_1800/icsi/frame_der/der_pooled` | 0.1932 | 0.193 | yes |
| 286 | D distill | run2 ICSI DER | `runs/diar_distill.json`:`eval/mix46_1800/icsi/frame_der/der_pooled` | 0.196 | 0.196 | yes |
| 287 | D distill | teacher ICSI DER | `runs/diar_distill.json`:`eval/nemotron3_max/icsi/frame_der/der_pooled` | 0.2364 | 0.236 | yes |
| 288 | D distill | run2 AMI count err | `runs/diar_distill.json`:`eval/mix46_1800/ami/count/mean` | -0.656 | -0.66 | yes |
| 289 | D distill | teacher AMI count err | `runs/diar_distill.json`:`eval/nemotron3_max/ami/count/mean` | -0.141 | -0.14 | yes |
| 290 | D distill | frozen tensors identical | `runs/diar_distill.json`:`train/b6_1800/frozen_tensors_identical` | 704 | 704 | yes |
| 291 | P perf | A/B RTF ratio default/none | `runs/perf.json`:`ab/none_default/ratio_B_over_A/rtf` | 0.616 | 0.62 | yes |
| 292 | P perf | A/B CPU ratio | `runs/perf.json`:`ab/none_default/ratio_B_over_A/cpu_rtf` | 0.665 | 0.67 | yes |
| 293 | P perf | A/B ASR pass ratio | `runs/perf.json`:`ab/none_default/ratio_B_over_A/asr_rtf` | 0.695 | 0.70 | yes |
| 294 | P perf | A/B turn pass ratio | `runs/perf.json`:`ab/none_default/ratio_B_over_A/turn_rtf` | 0.415 | 0.42 | yes |
| 295 | P perf | A/B diarizer ratio | `runs/perf.json`:`ab/none_default/ratio_B_over_A/diar_rtf` | 0.708 | 0.71 | yes |
| 296 | P perf | A/B chunk p95 ratio | `runs/perf.json`:`ab/none_default/ratio_B_over_A/chunk_ms_p95` | 0.376 | 0.38 | yes |
| 297 | P perf | A/B RTF none | `runs/perf.json`:`ab/none_default/A/rtf` | 1.389 | 1.39 | yes |
| 298 | P perf | A/B RTF default | `runs/perf.json`:`ab/none_default/B/rtf` | 0.8556 | 0.86 | yes |
| 299 | P perf | A/B messages | `runs/perf.json`:`ab/none_default/compare/n_msgs` | 1313 | 1313 | yes |
| 300 | P perf | A/B turn_end decision diffs | `runs/perf.json`:`ab/none_default/compare/decision_diffs/turn_end` | 0 | 0 | yes |
| 301 | P perf | A/B final decision diffs | `runs/perf.json`:`ab/none_default/compare/decision_diffs/final` | 0 | 0 | yes |
| 302 | P perf | A/B max eot diff | `runs/perf.json`:`ab/none_default/compare/maxdiff/eot` | 1e-05 | 0.00001 | yes |
| 303 | P perf | A/B max vad diff | `runs/perf.json`:`ab/none_default/compare/maxdiff/vad` | 0 | 0.0 | yes |
| 304 | P perf | before RTF e2e5 (none) | `runs/perf.json`:`baseline_none/e2e5/rtf` | 0.8449 | 0.84 | yes |
| 305 | P perf | before RTF long (none) | `runs/perf.json`:`baseline_none/long/rtf` | 1.3796 | 1.38 | yes |
| 306 | P perf | after RTF e2e5 components | `runs/perf.json`:`components_default/e2e5/rtf` | 0.4525 | 0.45 | yes |
| 307 | P perf | after RTF e2e5 streams1 | `runs/perf.json`:`streams_default/streams1_default/rtf` | 0.5137 | 0.51 | yes |
| 308 | P perf | after RTF long | `runs/perf.json`:`components_default/long/rtf` | 0.6185 | 0.62 | yes |
| 309 | P perf | after chunk p50 e2e5 | `runs/perf.json`:`components_default/e2e5/chunk_ms_p50` | 78.86 | 79 | yes |
| 310 | P perf | after chunk p95 e2e5 | `runs/perf.json`:`components_default/e2e5/chunk_ms_p95` | 125.37 | 125 | yes |
| 311 | P perf | Sortformer after RTF | `runs/perf.json`:`sortformer_default/rtf` | 0.6142 | 0.61 | yes |
| 312 | P perf | K=2 aggregate RTF | `runs/perf.json`:`streams_default/streams2_default/agg_rtf` | 1.3077 | 1.31 | yes |
| 313 | P perf | K=2 realtime | `runs/perf.json`:`streams_default/streams2_default/lat_growth_ms` | 17492.4 | 17492.4 | yes |
| 314 | T TS-VAD offline | AMI frame F1 5 s print | `runs/improve_115m.json`:`frame/ami/primary/tsvad_spk_vp5p0/all/f1` | 0.7428 | 0.743 | yes |
| 315 | T TS-VAD offline | AMI Sortformer column bound (spk head) F1 | `runs/improve_115m.json`:`frame/ami/primary/sortformer_vp_spk_vp5p0/all/f1` | 0.6292 | 0.629 | yes |
| 316 | T TS-VAD offline | ICSI frame F1 5 s print | `runs/improve_115m.json`:`frame/icsi/primary/tsvad_spk_vp5p0/all/f1` | 0.882 | 0.882 | yes |
| 317 | T TS-VAD offline | ICSI Sortformer oracle column F1 | `runs/improve_115m.json`:`frame/icsi/primary/sortformer_oracle_column/all/f1` | 0.7944 | 0.794 | yes |
| 318 | T TS-VAD offline | AMI TS-VAD hybrid miss % | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_tsvad_spk/6s/miss_rate` | 0.3933 (x100) | 39.3 | yes |
| 319 | T TS-VAD offline | AMI TS-VAD hybrid_dyn miss % | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_dyn_tsvad_spk/6s/miss_rate` | 0.3424 (x100) | 34.2 | yes |
| 320 | T TS-VAD offline | AMI TS-VAD hybrid_dyn open % | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_dyn_tsvad_spk/6s/strata/open/miss_rate` | 0.1789 (x100) | 17.9 | yes |
| 321 | T TS-VAD offline | AMI Sortformer causal hybrid miss % | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_causal_dominant/6s/miss_rate` | 0.6191 (x100) | 61.9 | yes |
| 322 | T TS-VAD offline | ICSI TS-VAD hybrid miss % | `runs/improve_115m.json`:`turn_bench/icsi/systems/hybrid_tsvad_spk/6s/miss_rate` | 0.2036 (x100) | 20.4 | yes |
| 323 | T TS-VAD offline | ICSI TS-VAD hybrid_dyn miss % | `runs/improve_115m.json`:`turn_bench/icsi/systems/hybrid_dyn_tsvad_spk/6s/miss_rate` | 0.1868 (x100) | 18.7 | yes |
| 324 | T TS-VAD offline | ICSI TS-VAD hybrid_dyn open % | `runs/improve_115m.json`:`turn_bench/icsi/systems/hybrid_dyn_tsvad_spk/6s/strata/open/miss_rate` | 0.1316 (x100) | 13.2 | yes |
| 325 | T TS-VAD offline | ICSI Sortformer causal hybrid miss % (baselines_turn_icsi) | `runs/baselines_turn_icsi.json`:`C_extended_windows_stream/crossfit_icsi/systems/hybrid_stream_causal_dominant/6s/fixed_5pct_turn_fc/miss_rate` | 0.685 (x100) | 68.5 | yes |
| 326 | T TS-VAD live | T-C missed_3s | `runs/e2e_tsvad.json`:`paired/all\|T-C/missed_3s/delta` | 0.1076 (x100) | 10.8 | yes |
| 327 | T TS-VAD live | T-C missed_3s CI lo | `runs/e2e_tsvad.json`:`paired/all\|T-C/missed_3s/ci[0]` | 0.0558 (x100) | 5.6 | yes |
| 328 | T TS-VAD live | T-C missed_3s CI hi | `runs/e2e_tsvad.json`:`paired/all\|T-C/missed_3s/ci[1]` | 0.1577 (x100) | 15.8 | yes |
| 329 | T TS-VAD live | T-C missed_6s | `runs/e2e_tsvad.json`:`paired/all\|T-C/missed_6s/delta` | 0.0942 (x100) | 9.4 | yes |
| 330 | T TS-VAD live | T-C missed_6s CI lo | `runs/e2e_tsvad.json`:`paired/all\|T-C/missed_6s/ci[0]` | 0.0411 (x100) | 4.1 | yes |
| 331 | T TS-VAD live | T-C missed_6s CI hi | `runs/e2e_tsvad.json`:`paired/all\|T-C/missed_6s/ci[1]` | 0.1487 (x100) | 14.9 | yes |
| 332 | T TS-VAD live | T-C cut_ins | `runs/e2e_tsvad.json`:`paired/all\|T-C/cut_ins/delta` | -0.4058 | -0.41 | yes |
| 333 | T TS-VAD live | T-C cut_ins CI lo | `runs/e2e_tsvad.json`:`paired/all\|T-C/cut_ins/ci[0]` | -0.6667 | -0.67 | yes |
| 334 | T TS-VAD live | T-C cut_ins CI hi | `runs/e2e_tsvad.json`:`paired/all\|T-C/cut_ins/ci[1]` | -0.1159 | -0.12 | yes |
| 335 | T TS-VAD live | T-D missed_3s | `runs/e2e_tsvad.json`:`paired/all\|T-D/missed_3s/delta` | 0.0134 (x100) | 1.3 | yes |
| 336 | T TS-VAD live | T-D missed_3s CI lo | `runs/e2e_tsvad.json`:`paired/all\|T-D/missed_3s/ci[0]` | -0.0355 (x100) | -3.6 | yes |
| 337 | T TS-VAD live | T-D missed_3s CI hi | `runs/e2e_tsvad.json`:`paired/all\|T-D/missed_3s/ci[1]` | 0.0623 (x100) | 6.2 | yes |
| 338 | T TS-VAD live | T-D missed_6s | `runs/e2e_tsvad.json`:`paired/all\|T-D/missed_6s/delta` | 0.009 (x100) | 0.9 | yes |
| 339 | T TS-VAD live | T-D missed_6s CI lo | `runs/e2e_tsvad.json`:`paired/all\|T-D/missed_6s/ci[0]` | -0.0311 (x100) | -3.1 | yes |
| 340 | T TS-VAD live | T-D missed_6s CI hi | `runs/e2e_tsvad.json`:`paired/all\|T-D/missed_6s/ci[1]` | 0.0512 (x100) | 5.1 | yes |
| 341 | T TS-VAD live | T-D cut_ins | `runs/e2e_tsvad.json`:`paired/all\|T-D/cut_ins/delta` | 0 | 0.00 | yes |
| 342 | T TS-VAD live | T-D cut_ins CI lo | `runs/e2e_tsvad.json`:`paired/all\|T-D/cut_ins/ci[0]` | -0.1739 | -0.17 | yes |
| 343 | T TS-VAD live | T-D cut_ins CI hi | `runs/e2e_tsvad.json`:`paired/all\|T-D/cut_ins/ci[1]` | 0.1884 | 0.19 | yes |
| 344 | T TS-VAD live | Th-C missed_3s | `runs/e2e_tsvad.json`:`paired/all\|Th-C/missed_3s/delta` | 0.1166 (x100) | 11.7 | yes |
| 345 | T TS-VAD live | Th-C missed_3s CI lo | `runs/e2e_tsvad.json`:`paired/all\|Th-C/missed_3s/ci[0]` | 0.0546 (x100) | 5.5 | yes |
| 346 | T TS-VAD live | Th-C missed_3s CI hi | `runs/e2e_tsvad.json`:`paired/all\|Th-C/missed_3s/ci[1]` | 0.1898 (x100) | 19.0 | yes |
| 347 | T TS-VAD live | Th-C missed_6s | `runs/e2e_tsvad.json`:`paired/all\|Th-C/missed_6s/delta` | 0.0583 (x100) | 5.8 | yes |
| 348 | T TS-VAD live | Th-C missed_6s CI lo | `runs/e2e_tsvad.json`:`paired/all\|Th-C/missed_6s/ci[0]` | 0.0089 (x100) | 0.9 | yes |
| 349 | T TS-VAD live | Th-C missed_6s CI hi | `runs/e2e_tsvad.json`:`paired/all\|Th-C/missed_6s/ci[1]` | 0.1188 (x100) | 11.9 | yes |
| 350 | T TS-VAD live | Th-D missed_6s | `runs/e2e_tsvad.json`:`paired/all\|Th-D/missed_6s/delta` | -0.0269 (x100) | -2.7 | yes |
| 351 | T TS-VAD live | Th-D missed_6s CI lo | `runs/e2e_tsvad.json`:`paired/all\|Th-D/missed_6s/ci[0]` | -0.078 (x100) | -7.8 | yes |
| 352 | T TS-VAD live | Th-D missed_6s CI hi | `runs/e2e_tsvad.json`:`paired/all\|Th-D/missed_6s/ci[1]` | 0.0228 (x100) | 2.3 | yes |
| 353 | T TS-VAD live | T missed 3 s pooled % | `runs/e2e_tsvad.json`:`paired/all\|T-C/missed_3s/a` | 0.4529 (x100) | 45.3 | yes |
| 354 | T TS-VAD live | C missed 3 s pooled % | `runs/e2e_tsvad.json`:`paired/all\|T-C/missed_3s/b` | 0.3453 (x100) | 34.5 | yes |
| 355 | T TS-VAD live | D missed 3 s pooled % | `runs/e2e_tsvad.json`:`paired/all\|T-D/missed_3s/b` | 0.4395 (x100) | 44.0 | yes |
| 356 | T TS-VAD live | sessions per system | `runs/e2e_tsvad.json`:`n_records/T` | 69 | 69 | yes |
| 357 | T TS-VAD live | AMI T missed 3 s % | `runs/e2e_tsvad.json`:`table/ami\|mono\|T/missed_3s` | 0.2 (x100) | 20 | yes |
| 358 | T TS-VAD live | AMI Th missed 3 s % | `runs/e2e_tsvad.json`:`table/ami\|mono\|Th/missed_3s` | 0 (x100) | 0 | yes |
| 359 | T TS-VAD live | AMI Th dead air ms | `runs/e2e_tsvad.json`:`table/ami\|mono\|Th/dead_air_ms_median` | 1763 | 1763 | yes |
| 360 | T TS-VAD live | AMI C dead air ms | `runs/e2e_tsvad.json`:`table/ami\|mono\|C/dead_air_ms_median` | 1683 | 1683 | yes |
| 361 | T TS-VAD live | oto user C missed 3 s % | `runs/e2e_tsvad.json`:`table/oto\|user\|C/missed_3s` | 0.0566 (x100) | 6 | yes |
| 362 | T TS-VAD live | TurnBench user C missed 3 s % | `runs/e2e_tsvad.json`:`table/turnbench\|user\|C/missed_3s` | 0.2321 (x100) | 23 | yes |
| 363 | N exp2 | AMI (b) hybrid_dyn - served hybrid_dyn | `runs/tsvad_turn.json`:`ami/paired/hybrid_dyn_b_eot - hybrid_dyn_served \| 6s/all/miss_diff` | 0.0988 (x100) | 9.9 | yes |
| 364 | N exp2 | AMI CI lo | `runs/tsvad_turn.json`:`ami/paired/hybrid_dyn_b_eot - hybrid_dyn_served \| 6s/all/miss_diff_ci[0]` | 0.0626 (x100) | 6.3 | yes |
| 365 | N exp2 | AMI CI hi | `runs/tsvad_turn.json`:`ami/paired/hybrid_dyn_b_eot - hybrid_dyn_served \| 6s/all/miss_diff_ci[1]` | 0.1322 (x100) | 13.2 | yes |
| 366 | N exp2 | ICSI (b) hybrid_dyn - served | `runs/tsvad_turn.json`:`icsi/paired/hybrid_dyn_b_eot - hybrid_dyn_served \| 6s/all/miss_diff` | 0.135 (x100) | 13.5 | yes |
| 367 | N exp2 | ICSI CI lo | `runs/tsvad_turn.json`:`icsi/paired/hybrid_dyn_b_eot - hybrid_dyn_served \| 6s/all/miss_diff_ci[0]` | 0.1074 (x100) | 10.7 | yes |
| 368 | N exp2 | ICSI CI hi | `runs/tsvad_turn.json`:`icsi/paired/hybrid_dyn_b_eot - hybrid_dyn_served \| 6s/all/miss_diff_ci[1]` | 0.1632 (x100) | 16.3 | yes |
| 369 | N exp2 | ICSI P50 diff CI lo | `runs/tsvad_turn.json`:`icsi/paired/hybrid_dyn_b_eot - hybrid_dyn_served \| 6s/all/p50_diff_ci[0]` | -480 | -480 | yes |
| 370 | N exp2 | ICSI P50 diff CI hi | `runs/tsvad_turn.json`:`icsi/paired/hybrid_dyn_b_eot - hybrid_dyn_served \| 6s/all/p50_diff_ci[1]` | -240 | -240 | yes |
| 371 | N exp2b | AMI hybrid_dyn P50 | `runs/hybrid_fast.json`:`tsvad/ami/hybrid_dyn/p50_ms` | 3280 | 3280 | yes |
| 372 | N exp2b | AMI hybrid_fast 160 P50 | `runs/hybrid_fast.json`:`tsvad/ami/hybrid_fast_160ms/p50_ms` | 3200 | 3200 | yes |
| 373 | N exp2b | AMI hybrid_fast 160 FC % | `runs/hybrid_fast.json`:`tsvad/ami/hybrid_fast_160ms/fc_rate` | 0.0637 (x100) | 6.4 | yes |
| 374 | N exp2b | AMI hybrid_dyn FC % | `runs/hybrid_fast.json`:`tsvad/ami/hybrid_dyn/fc_rate` | 0.0534 (x100) | 5.3 | yes |
| 375 | N exp2b | ICSI hybrid_dyn P50 | `runs/hybrid_fast.json`:`tsvad/icsi/hybrid_dyn/p50_ms` | 2080 | 2080 | yes |
| 376 | N exp2b | ICSI hybrid_fast 240 P50 | `runs/hybrid_fast.json`:`tsvad/icsi/hybrid_fast_240ms/p50_ms` | 2080 | 2080 | yes |
| 377 | F frontier | ami hybrid_dyn 0.05 miss % | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.05/miss` | 0.3491 (x100) | 34.9 | yes |
| 378 | F frontier | ami hybrid_dyn 0.05 open % | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.05/open_miss` | 0.1822 (x100) | 18.2 | yes |
| 379 | F frontier | ami hybrid_dyn 0.05 P50 | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.05/p50_ms` | 3280 | 3280 | yes |
| 380 | F frontier | ami hybrid_dyn 0.1 miss % | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.1/miss` | 0.2238 (x100) | 22.4 | yes |
| 381 | F frontier | ami hybrid_dyn 0.1 open % | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.1/open_miss` | 0.1271 (x100) | 12.7 | yes |
| 382 | F frontier | ami hybrid_dyn 0.1 P50 | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.1/p50_ms` | 2320 | 2320 | yes |
| 383 | F frontier | icsi hybrid_dyn 0.05 miss % | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.05/miss` | 0.2043 (x100) | 20.4 | yes |
| 384 | F frontier | icsi hybrid_dyn 0.05 open % | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.05/open_miss` | 0.1019 (x100) | 10.2 | yes |
| 385 | F frontier | icsi hybrid_dyn 0.05 P50 | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.05/p50_ms` | 2080 | 2080 | yes |
| 386 | F frontier | icsi hybrid_dyn 0.1 miss % | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.1/miss` | 0.1334 (x100) | 13.3 | yes |
| 387 | F frontier | icsi hybrid_dyn 0.1 open % | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.1/open_miss` | 0.0741 (x100) | 7.4 | yes |
| 388 | F frontier | icsi hybrid_dyn 0.1 P50 | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.1/p50_ms` | 1600 | 1600 | yes |
| 389 | F frontier | AMI Silero timeout 10 % miss | `runs/eot_frontier.json`:`ami/timeout_silero/0.1/miss` | 0.6674 (x100) | 66.7 | yes |
| 390 | F frontier | AMI timeout on P(target) 10 % miss | `runs/eot_frontier.json`:`ami/timeout_tsvad/0.1/miss` | 0.5626 (x100) | 56.3 | yes |
| 391 | F frontier | ICSI timeout on P(target) 10 % miss | `runs/eot_frontier.json`:`icsi/timeout_tsvad/0.1/miss` | 0.3285 (x100) | 32.9 | yes |
| 392 | S spk head exp3 | c115_warm AMI n200 within | `runs/spk_frame.json`:`variants/c115_warm/ami_n200/eer_within_meeting` | 0.1713 (x100) | 17.1 | yes |
| 393 | S spk head exp3 | c115_warm ICSI n200 within | `runs/spk_frame.json`:`variants/c115_warm/icsi_n200/eer_within_meeting` | 0.0444 (x100) | 4.4 | yes |
| 394 | S spk head exp3 | c115_warm AMI n64 within | `runs/spk_frame.json`:`variants/c115_warm/ami_n64/eer_within_meeting` | 0.1232 (x100) | 12.3 | yes |
| 395 | S spk head exp3 | c115_fresh AMI n200 within | `runs/spk_frame.json`:`variants/c115_fresh/ami_n200/eer_within_meeting` | 0.1741 (x100) | 17.4 | yes |
| 396 | S spk head exp3 | c115_fresh ICSI n200 within | `runs/spk_frame.json`:`variants/c115_fresh/icsi_n200/eer_within_meeting` | 0.0399 (x100) | 4.0 | yes |
| 397 | S spk head exp3 | c115_fresh AMI n64 within | `runs/spk_frame.json`:`variants/c115_fresh/ami_n64/eer_within_meeting` | 0.1154 (x100) | 11.5 | yes |
| 398 | S spk head exp3 | warm-fresh AMI | `runs/spk_frame.json`:`paired_within_n200/ami\|c115_warm-c115_fresh/delta` | -0.0028 (x100) | -0.3 | yes |
| 399 | S spk head exp3 | warm-fresh ICSI | `runs/spk_frame.json`:`paired_within_n200/icsi\|c115_warm-c115_fresh/delta` | 0.0045 (x100) | 0.5 | yes |
| 400 | R frontier pilots | Nemotron-3 session cpWER % | `runs/frontier_sa_captions.json`:`session/nemotron3/cpwer` | 0.32444 (x100) | 32.4 | yes |
| 401 | R frontier pilots | oracle session cpWER % | `runs/frontier_sa_captions.json`:`session/oracle/cpwer` | 0.323011 (x100) | 32.3 | yes |
| 402 | R frontier pilots | speaker-agnostic WER % | `runs/frontier_sa_captions.json`:`session/agnostic/wer` | 0.306813 (x100) | 30.7 | yes |
| 403 | R frontier pilots | codec cb0 top-1 block 9 % | `runs/frontier_codec_probe.json`:`linear/L9/cb0[0]` | 0.447052 (x100) | 44.7 | yes |
| 404 | R frontier pilots | codec cb0 top-1 log-mel % | `runs/frontier_codec_probe.json`:`linear/mel_stack8x5/cb0[0]` | 0.0974637 (x100) | 9.7 | yes |
| 405 | 6 FLEURS TDT | es WER % | `runs/hybrid_asr.json`:`fleurs/es/wer/wer` | 0.0391 (x100) | 3.9 | yes |
| 406 | 6 FLEURS TDT | pl WER % (n 30) | `runs/hybrid_asr.json`:`fleurs/pl/wer/wer` | 0.1083 (x100) | 10.8 | yes |
| 407 | 7 E2E | AMI A cut-ins/min | `runs/e2e_final.json`:`table/ami\|mono\|pipecat\|A/cut_ins_per_min` | 7.463 | 7.46 | yes |
| 408 | 7 E2E | AMI C pipecat cut-ins/min | `runs/e2e_final.json`:`table/ami\|mono\|pipecat\|C/cut_ins_per_min` | 1.357 | 1.36 | yes |
| 409 | 7 E2E | TB mono B missed 3 s % | `runs/e2e_final.json`:`table/turnbench\|mono\|livekit\|B/missed_3s` | 0.6964 (x100) | 70 | yes |
| 410 | 7 E2E | AMI C pipecat dead air | `runs/e2e_final.json`:`table/ami\|mono\|pipecat\|C/dead_air_ms_median` | 1683 | 1683 | yes |
| 411 | 7 E2E | AMI A first text ms | `runs/e2e_final.json`:`table/ami\|mono\|pipecat\|A/first_text_ms_median` | 3689 | 3689 | yes |
| 412 | 7 E2E | AMI C - A cut-ins per clip | `runs/e2e_final.json`:`paired/ami\|mono\|pipecat\|C-A/cut_ins_per_clip/delta` | -1.8 | -1.8 | yes |
| 413 | 3 speaker | TitaNet n64 all % | `runs/baselines_sd.json`:`spk/titanet_large/n64/eer` | 0.0659 (x100) | 6.6 | yes |
| 414 | 3 speaker | TitaNet n200 all % | `runs/baselines_sd.json`:`spk/titanet_large/n200/eer` | 0.109 (x100) | 10.9 | yes |
| 415 | 3 speaker | TitaNet ICSI n64 within % (teacher) | `runs/spk_head.json`:`variants/spk_distill_titanet/eval/icsi_n64/teacher/eer_within_meeting` | 0.0102 (x100) | 1.0 | yes |
| 416 | 3 speaker | TitaNet ICSI n200 all % (teacher) | `runs/spk_head.json`:`variants/spk_distill_titanet/eval/icsi_n200/teacher/eer` | 0.0218 (x100) | 2.2 | yes |
| 417 | 3 speaker | TitaNet ICSI n200 within % (teacher) | `runs/spk_head.json`:`variants/spk_distill_titanet/eval/icsi_n200/teacher/eer_within_meeting` | 0.0206 (x100) | 2.1 | yes |
| 418 | RTF table | Sortformer server CPU-s per audio-s | `runs/e2e_final.json`:`components_live/pipecat:C/_process/server_cpu_s_per_audio_s` | 1.0207 | 1.02 | yes |
| 419 | RTF table | Nemotron-3 server CPU-s per audio-s | `runs/e2e_final.json`:`components_live/pipecat:CN/_process/server_cpu_s_per_audio_s` | 0.7075 | 0.71 | yes |
| 420 | RTF table | Pipecat default CPU-s per audio-s | `runs/e2e_final.json`:`components_live/pipecat:A/_process/driver_cpu_s_per_audio_s` | 0.4314 | 0.43 | yes |
| 421 | RTF table | LiveKit default CPU-s per audio-s | `runs/e2e_final.json`:`components_live/livekit:B/_process/driver_cpu_s_per_audio_s` | 0.2797 | 0.28 | yes |
| 422 | RTF table | 0.6B encoder streaming CPU chunk mean / 160 ms (RTF) | `runs/enc_0p6b.json`:`rtf/chunk_ms_mean` | 739.4 (/160 ms) | 4.6 | yes |
| 423 | RTF table | first partial server lag after (ms, clip 1) | `runs/perf.json`:`components_default/e2e5/first_partial_server_lag_ms[0]` | 20.9 | 20.9 | yes |
| 424 | RTF table | first partial after onset (ms, clip 2) | `runs/perf.json`:`components_default/e2e5/first_partial_after_onset_ms[1]` | 1440 | 1440 | yes |
| 425 | T TS-VAD offline | AMI TS-VAD hybrid CI lo | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_tsvad_spk/6s/miss_rate_ci[0]` | 0.3622 (x100) | 36.2 | yes |
| 426 | T TS-VAD offline | AMI TS-VAD hybrid CI hi | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_tsvad_spk/6s/miss_rate_ci[1]` | 0.4275 (x100) | 42.8 | yes |
| 427 | T TS-VAD offline | ICSI TS-VAD hybrid CI lo | `runs/improve_115m.json`:`turn_bench/icsi/systems/hybrid_tsvad_spk/6s/miss_rate_ci[0]` | 0.1817 (x100) | 18.2 | yes |
| 428 | T TS-VAD offline | ICSI TS-VAD hybrid CI hi | `runs/improve_115m.json`:`turn_bench/icsi/systems/hybrid_tsvad_spk/6s/miss_rate_ci[1]` | 0.227 (x100) | 22.7 | yes |
| 429 | T TS-VAD offline | AMI Sortformer hybrid CI lo (this file) | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_causal_dominant/6s/miss_rate_ci[0]` | 0.5861 (x100) | 58.6 | yes |
| 430 | T TS-VAD offline | AMI Sortformer hybrid CI hi (this file) | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_causal_dominant/6s/miss_rate_ci[1]` | 0.6535 (x100) | 65.3 | yes |
| 431 | T TS-VAD offline | ICSI Sortformer hybrid CI lo | `runs/baselines_turn_icsi.json`:`C_extended_windows_stream/crossfit_icsi/systems/hybrid_stream_causal_dominant/6s/fixed_5pct_turn_fc/miss_rate_ci[0]` | 0.6581 (x100) | 65.8 | yes |
| 432 | T TS-VAD offline | ICSI Sortformer hybrid CI hi | `runs/baselines_turn_icsi.json`:`C_extended_windows_stream/crossfit_icsi/systems/hybrid_stream_causal_dominant/6s/fixed_5pct_turn_fc/miss_rate_ci[1]` | 0.7122 (x100) | 71.2 | yes |
| 433 | T TS-VAD offline | AMI TS-VAD head P50 ms | `runs/improve_115m.json`:`turn_bench/ami/systems/head_tsvad_spk/6s/p50_ms` | 4240 | 4240 | yes |
| 434 | T TS-VAD offline | AMI TS-VAD hybrid_dyn P50 ms | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_dyn_tsvad_spk/6s/p50_ms` | 3200 | 3200 | yes |
| 435 | T TS-VAD live | T cut-ins per session | `runs/e2e_tsvad.json`:`paired/all\|T-C/cut_ins/a` | 0.5217 | 0.52 | yes |
| 436 | T TS-VAD live | C cut-ins per session | `runs/e2e_tsvad.json`:`paired/all\|T-C/cut_ins/b` | 0.9275 | 0.93 | yes |
| 437 | P perf | A/B load at end (min) | `runs/perf.json`:`ab/none_default/load_end[0]` | 7 | 7.0 | yes |
| 438 | P perf | before e2e5 load at start | `runs/perf.json`:`baseline_none/e2e5/load_start[0]` | 4.7 | 4.7 | yes |
| 439 | 1.3 dual lookahead | [70,0] ami WER % | `runs/hybrid_asr.json`:`english/results/served_la0/ami/wer_normalize_text/wer` | 0.2568 (x100) | 25.68 | yes |
| 440 | 1.3 dual lookahead | [70,0] icsi WER % | `runs/hybrid_asr.json`:`english/results/served_la0/icsi/wer_normalize_text/wer` | 0.2987 (x100) | 29.87 | yes |
| 441 | 1.3 dual lookahead | [70,0] libri WER % | `runs/hybrid_asr.json`:`english/results/served_la0/libri/wer_normalize_text/wer` | 0.0248 (x100) | 2.48 | yes |
| 442 | 1.3 dual lookahead | [70,13] ami CI lo | `runs/hybrid_asr.json`:`english/results/served_la13/ami/wer_normalize_text/ci95[0]` | 0.2054 (x100) | 20.54 | yes |
| 443 | 1.3 dual lookahead | [70,13] ami CI hi | `runs/hybrid_asr.json`:`english/results/served_la13/ami/wer_normalize_text/ci95[1]` | 0.2579 (x100) | 25.79 | yes |
| 444 | 1.3 dual lookahead | [70,13] icsi CI lo | `runs/hybrid_asr.json`:`english/results/served_la13/icsi/wer_normalize_text/ci95[0]` | 0.2192 (x100) | 21.92 | yes |
| 445 | 1.3 dual lookahead | [70,13] icsi CI hi | `runs/hybrid_asr.json`:`english/results/served_la13/icsi/wer_normalize_text/ci95[1]` | 0.2773 (x100) | 27.73 | yes |
| 446 | 1.3 dual lookahead | [70,13] libri CI lo | `runs/hybrid_asr.json`:`english/results/served_la13/libri/wer_normalize_text/ci95[0]` | 0.0142 (x100) | 1.42 | yes |
| 447 | 1.3 dual lookahead | [70,13] libri CI hi | `runs/hybrid_asr.json`:`english/results/served_la13/libri/wer_normalize_text/ci95[1]` | 0.0249 (x100) | 2.49 | yes |
| 448 | 1.3 dual lookahead | [70,1] ICSI CI lo | `runs/hybrid_asr.json`:`english/results/served_la1/icsi/wer_normalize_text/ci95[0]` | 0.2475 (x100) | 24.75 | yes |
| 449 | 1.3 dual lookahead | [70,1] ICSI CI hi | `runs/hybrid_asr.json`:`english/results/served_la1/icsi/wer_normalize_text/ci95[1]` | 0.3024 (x100) | 30.24 | yes |
| 450 | 1.4 decoder adapt | ICSI adapted WER % | `runs/hybrid_asr.json`:`adapt/_adapt/sets/icsi/normalize_text/adapted/wer` | 0.2717 (x100) | 27.17 | yes |
| 451 | 1.4 decoder adapt | Libri adapted WER % | `runs/hybrid_asr.json`:`adapt/_adapt/sets/libri/normalize_text/adapted/wer` | 0.0233 (x100) | 2.33 | yes |
| 452 | S spk head exp3 | c115_warm AMI n64 all | `runs/spk_frame.json`:`variants/c115_warm/ami_n64/eer` | 0.0789 (x100) | 7.9 | yes |
| 453 | S spk head exp3 | c115_warm AMI n200 all | `runs/spk_frame.json`:`variants/c115_warm/ami_n200/eer` | 0.105 (x100) | 10.5 | yes |
| 454 | S spk head exp3 | c115_warm ICSI n200 all | `runs/spk_frame.json`:`variants/c115_warm/icsi_n200/eer` | 0.0446 (x100) | 4.5 | yes |
| 455 | S spk head exp3 | c115_fresh AMI n64 all | `runs/spk_frame.json`:`variants/c115_fresh/ami_n64/eer` | 0.0789 (x100) | 7.9 | yes |
| 456 | S spk head exp3 | c115_fresh AMI n200 all | `runs/spk_frame.json`:`variants/c115_fresh/ami_n200/eer` | 0.1073 (x100) | 10.7 | yes |
| 457 | S spk head exp3 | c115_fresh ICSI n200 all | `runs/spk_frame.json`:`variants/c115_fresh/icsi_n200/eer` | 0.0424 (x100) | 4.2 | yes |
| 458 | S spk head exp3 | warm ICSI n64 within | `runs/spk_frame.json`:`variants/c115_warm/icsi_n64/eer_within_meeting` | 0.0264 (x100) | 2.6 | yes |
| 459 | S spk head exp3 | fresh ICSI n64 within | `runs/spk_frame.json`:`variants/c115_fresh/icsi_n64/eer_within_meeting` | 0.025 (x100) | 2.5 | yes |
| 460 | S spk head exp3 | warm-fresh AMI CI lo | `runs/spk_frame.json`:`paired_within_n200/ami\|c115_warm-c115_fresh/ci95[0]` | -0.015 (x100) | -1.5 | yes |
| 461 | S spk head exp3 | warm-fresh AMI CI hi | `runs/spk_frame.json`:`paired_within_n200/ami\|c115_warm-c115_fresh/ci95[1]` | 0.0141 (x100) | 1.4 | yes |
| 462 | D distill | teacher ICSI FA | `runs/diar_distill.json`:`eval/nemotron3_max/icsi/frame_der/fa` | 0.2283 | 0.228 | yes |
| 463 | D distill | served head AMI count err | `runs/diar_distill.json`:`eval/served_stage1/ami/count/mean` | -0.719 | -0.72 | yes |
| 464 | D distill | teacher ICSI count err | `runs/diar_distill.json`:`eval/nemotron3_max/icsi/count/mean` | 0.062 | 0.06 | yes |
| 465 | D diar fix clips | shed hold mix acc | `scratch/diar/offline/n3_R6_reg_any_shed.json`:`items/mix6_s0/final_speaker_acc` | 0.906 | 0.91 | yes |
| 466 | D diar fix clips | Sortformer before mix acc | `scratch/diar/offline/sf_R1_legacy.json`:`items/mix6_s0/final_speaker_acc` | 0.25 | 0.25 | yes |
| 467 | D diar fix clips | TitaNet run RTF | `scratch/diar/offline/n3_R4_tita_any.json`:`aggregate/wall_rtf_mean` | 0.57 | 0.57 | yes |
| 468 | F frontier | ami hybrid_dyn 0.03 miss % | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.03/miss` | 0.4497 (x100) | 45.0 | yes |
| 469 | F frontier | ami hybrid_dyn 0.03 open % | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.03/open_miss` | 0.2585 (x100) | 25.9 | yes |
| 470 | F frontier | ami hybrid_dyn 0.03 P50 | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.03/p50_ms` | 4960 | 4960 | yes |
| 471 | F frontier | ami timeout P(target) 0.03 miss % | `runs/eot_frontier.json`:`ami/timeout_tsvad/0.03/miss` | 0.6848 (x100) | 68.5 | yes |
| 472 | F frontier | ami Silero timeout 0.03 miss % | `runs/eot_frontier.json`:`ami/timeout_silero/0.03/miss` | 0.7485 (x100) | 74.9 | yes |
| 473 | F frontier | ami hybrid_dyn 0.15 miss % | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.15/miss` | 0.1674 (x100) | 16.7 | yes |
| 474 | F frontier | ami hybrid_dyn 0.15 open % | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.15/open_miss` | 0.0805 (x100) | 8.1 | yes |
| 475 | F frontier | ami hybrid_dyn 0.15 P50 | `runs/eot_frontier.json`:`ami/hybrid_dyn/0.15/p50_ms` | 1840 | 1840 | yes |
| 476 | F frontier | ami timeout P(target) 0.15 miss % | `runs/eot_frontier.json`:`ami/timeout_tsvad/0.15/miss` | 0.4938 (x100) | 49.4 | yes |
| 477 | F frontier | ami Silero timeout 0.15 miss % | `runs/eot_frontier.json`:`ami/timeout_silero/0.15/miss` | 0.6191 (x100) | 61.9 | yes |
| 478 | F frontier | icsi hybrid_dyn 0.03 miss % | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.03/miss` | 0.2348 (x100) | 23.5 | yes |
| 479 | F frontier | icsi hybrid_dyn 0.03 open % | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.03/open_miss` | 0.1481 (x100) | 14.8 | yes |
| 480 | F frontier | icsi hybrid_dyn 0.03 P50 | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.03/p50_ms` | 2320 | 2320 | yes |
| 481 | F frontier | icsi timeout P(target) 0.03 miss % | `runs/eot_frontier.json`:`icsi/timeout_tsvad/0.03/miss` | 0.5793 (x100) | 57.9 | yes |
| 482 | F frontier | icsi Silero timeout 0.03 miss % | `runs/eot_frontier.json`:`icsi/timeout_silero/0.03/miss` | 0.8857 (x100) | 88.6 | yes |
| 483 | F frontier | icsi hybrid_dyn 0.15 miss % | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.15/miss` | 0.1059 (x100) | 10.6 | yes |
| 484 | F frontier | icsi hybrid_dyn 0.15 open % | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.15/open_miss` | 0.0648 (x100) | 6.5 | yes |
| 485 | F frontier | icsi hybrid_dyn 0.15 P50 | `runs/eot_frontier.json`:`icsi/hybrid_dyn/0.15/p50_ms` | 1360 | 1360 | yes |
| 486 | F frontier | icsi timeout P(target) 0.15 miss % | `runs/eot_frontier.json`:`icsi/timeout_tsvad/0.15/miss` | 0.2378 (x100) | 23.8 | yes |
| 487 | F frontier | icsi Silero timeout 0.15 miss % | `runs/eot_frontier.json`:`icsi/timeout_silero/0.15/miss` | 0.7233 (x100) | 72.3 | yes |
| 488 | F frontier | ami timeout P(target) 0.05 miss % | `runs/eot_frontier.json`:`ami/timeout_tsvad/0.05/miss` | 0.6314 (x100) | 63.1 | yes |
| 489 | F frontier | ami Silero timeout 0.05 miss % | `runs/eot_frontier.json`:`ami/timeout_silero/0.05/miss` | 0.7197 (x100) | 72.0 | yes |
| 490 | F frontier | icsi timeout P(target) 0.05 miss % | `runs/eot_frontier.json`:`icsi/timeout_tsvad/0.05/miss` | 0.5312 (x100) | 53.1 | yes |
| 491 | F frontier | icsi Silero timeout 0.05 miss % | `runs/eot_frontier.json`:`icsi/timeout_silero/0.05/miss` | 0.8422 (x100) | 84.2 | yes |
| 492 | F frontier | icsi Silero timeout 0.1 miss % | `runs/eot_frontier.json`:`icsi/timeout_silero/0.1/miss` | 0.7851 (x100) | 78.5 | yes |
| 493 | T TS-VAD offline | ICSI Sortformer bound spk F1 | `runs/improve_115m.json`:`frame/icsi/primary/sortformer_vp_spk_vp5p0/all/f1` | 0.6898 | 0.690 | yes |
| 494 | T TS-VAD offline | AMI oracle hybrid % | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_oracle/6s/miss_rate` | 0.287 (x100) | 28.7 | yes |
| 495 | T TS-VAD offline | AMI oracle hybrid CI lo | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_oracle/6s/miss_rate_ci[0]` | 0.2616 (x100) | 26.2 | yes |
| 496 | T TS-VAD offline | AMI oracle hybrid open % | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_oracle/6s/strata/open/miss_rate` | 0.1713 (x100) | 17.1 | yes |
| 497 | T TS-VAD offline | AMI Sortformer causal hybrid open % | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_causal_dominant/6s/strata/open/miss_rate` | 0.4587 (x100) | 45.9 | yes |
| 498 | T TS-VAD offline | ICSI Sortformer hybrid open % | `runs/baselines_turn_icsi.json`:`C_extended_windows_stream/crossfit_icsi/systems/hybrid_stream_causal_dominant/6s/fixed_5pct_turn_fc/strata/open/miss_rate` | 0.5522 (x100) | 55.2 | yes |
| 499 | T TS-VAD offline | AMI TS-VAD hybrid P50 | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_tsvad_spk/6s/p50_ms` | 4160 | 4160 | yes |
| 500 | T TS-VAD offline | ICSI TS-VAD hybrid P50 | `runs/improve_115m.json`:`turn_bench/icsi/systems/hybrid_tsvad_spk/6s/p50_ms` | 1920 | 1920 | yes |
| 501 | T TS-VAD offline | AMI TS-VAD hybrid open % | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_tsvad_spk/6s/strata/open/miss_rate` | 0.379 (x100) | 37.9 | yes |
| 502 | T TS-VAD offline | ICSI TS-VAD hybrid open % | `runs/improve_115m.json`:`turn_bench/icsi/systems/hybrid_tsvad_spk/6s/strata/open/miss_rate` | 0.2147 (x100) | 21.5 | yes |
| 503 | T TS-VAD offline | AMI hybrid_dyn CI lo | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_dyn_tsvad_spk/6s/miss_rate_ci[0]` | 0.3107 (x100) | 31.1 | yes |
| 504 | T TS-VAD offline | AMI hybrid_dyn CI hi | `runs/improve_115m.json`:`turn_bench/ami/systems/hybrid_dyn_tsvad_spk/6s/miss_rate_ci[1]` | 0.3749 (x100) | 37.5 | yes |
| 505 | T TS-VAD offline | ICSI hybrid_dyn CI lo | `runs/improve_115m.json`:`turn_bench/icsi/systems/hybrid_dyn_tsvad_spk/6s/miss_rate_ci[0]` | 0.1658 (x100) | 16.6 | yes |
| 506 | T TS-VAD offline | ICSI hybrid_dyn CI hi | `runs/improve_115m.json`:`turn_bench/icsi/systems/hybrid_dyn_tsvad_spk/6s/miss_rate_ci[1]` | 0.2101 (x100) | 21.0 | yes |
| 507 | T TS-VAD offline | ICSI hybrid_dyn P50 | `runs/improve_115m.json`:`turn_bench/icsi/systems/hybrid_dyn_tsvad_spk/6s/p50_ms` | 1920 | 1920 | yes |
| 508 | 6 FLEURS TDT | tr WER % (lowest unsupported) | `runs/hybrid_asr.json`:`fleurs/tr/wer/wer` | 0.9947 (x100) | 99.5 | yes |
| 509 | N exp2 | ICSI P50 (b) | `runs/tsvad_turn.json`:`icsi/paired/hybrid_dyn_b_eot - hybrid_dyn_served \| 6s/all/p50_a` | 1520 | 1520 | yes |
| 510 | T TS-VAD live | Th-C cut-ins | `runs/e2e_tsvad.json`:`paired/all\|Th-C/cut_ins/delta` | -0.2174 | -0.22 | yes |
| 511 | T TS-VAD live | Th-C cut-ins CI lo | `runs/e2e_tsvad.json`:`paired/all\|Th-C/cut_ins/ci[0]` | -0.4493 | -0.45 | yes |
| 512 | T TS-VAD live | Th-C cut-ins CI hi | `runs/e2e_tsvad.json`:`paired/all\|Th-C/cut_ins/ci[1]` | 0.0145 | 0.01 | yes |
| 513 | T TS-VAD live | ami/mono/T dead air median ms | `runs/e2e_tsvad.json`:`table/ami\|mono\|T/dead_air_ms_median` | 2221 | 2221 | yes |
| 514 | T TS-VAD live | ami/mono/D dead air median ms | `runs/e2e_tsvad.json`:`table/ami\|mono\|D/dead_air_ms_median` | 2561 | 2561 | yes |
| 515 | T TS-VAD live | turnbench/user/T dead air median ms | `runs/e2e_tsvad.json`:`table/turnbench\|user\|T/dead_air_ms_median` | 1986 | 1986 | yes |
| 516 | T TS-VAD live | turnbench/user/D dead air median ms | `runs/e2e_tsvad.json`:`table/turnbench\|user\|D/dead_air_ms_median` | 2311 | 2311 | yes |
| 517 | T TS-VAD live | oto/mono/T dead air median ms | `runs/e2e_tsvad.json`:`table/oto\|mono\|T/dead_air_ms_median` | 2001 | 2001 | yes |
| 518 | T TS-VAD live | oto/mono/D dead air median ms | `runs/e2e_tsvad.json`:`table/oto\|mono\|D/dead_air_ms_median` | 2252 | 2252 | yes |
| 519 | T TS-VAD live | oto/user/T dead air median ms | `runs/e2e_tsvad.json`:`table/oto\|user\|T/dead_air_ms_median` | 2006 | 2006 | yes |
| 520 | T TS-VAD live | oto/user/D dead air median ms | `runs/e2e_tsvad.json`:`table/oto\|user\|D/dead_air_ms_median` | 2343 | 2343 | yes |
| 521 | T TS-VAD live | turnbench/mono/T dead air median ms | `runs/e2e_tsvad.json`:`table/turnbench\|mono\|T/dead_air_ms_median` | 1948 | 1948 | yes |
| 522 | T TS-VAD live | turnbench/mono/D dead air median ms | `runs/e2e_tsvad.json`:`table/turnbench\|mono\|D/dead_air_ms_median` | 1696 | 1696 | yes |
| 523 | P perf | A/B chunk p95 none ms | `runs/perf.json`:`ab/none_default/A/chunk_ms_p95` | 655.74 | 656 | yes |
| 524 | D diar fix clips | before mix6_s0 acc | `scratch/diar/offline/n3_R1_legacy.json`:`items/mix6_s0/final_speaker_acc` | 0.8 | 0.80 | yes |
| 525 | D diar fix clips | before mix6_s0 ids | `scratch/diar/offline/n3_R1_legacy.json`:`items/mix6_s0/n_ids` | 4 | 4 | yes |
| 526 | D diar fix clips | before icsi_Bmr021_940s acc | `scratch/diar/offline/n3_R1_legacy.json`:`items/icsi_Bmr021_940s/final_speaker_acc` | 0.583 | 0.58 | yes |
| 527 | D diar fix clips | before icsi_Bmr021_940s ids | `scratch/diar/offline/n3_R1_legacy.json`:`items/icsi_Bmr021_940s/n_ids` | 4 | 4 | yes |
| 528 | D diar fix clips | before ami_IS1008b_1670s acc | `scratch/diar/offline/n3_R1_legacy.json`:`items/ami_IS1008b_1670s/final_speaker_acc` | 0.833 | 0.83 | yes |
| 529 | D diar fix clips | before ami_IS1008b_1670s ids | `scratch/diar/offline/n3_R1_legacy.json`:`items/ami_IS1008b_1670s/n_ids` | 3 | 3 | yes |
| 530 | D diar fix clips | n3_R2_reg_hold icsi_Bmr021_940s acc | `scratch/diar/offline/n3_R2_reg_hold.json`:`items/icsi_Bmr021_940s/final_speaker_acc` | 0.8 | 0.80 | yes |
| 531 | D diar fix clips | n3_R2_reg_hold icsi_Bmr021_940s ids | `scratch/diar/offline/n3_R2_reg_hold.json`:`items/icsi_Bmr021_940s/n_ids` | 4 | 4 | yes |
| 532 | D diar fix clips | n3_R3_reg_hold_any mix6_s0 acc | `scratch/diar/offline/n3_R3_reg_hold_any.json`:`items/mix6_s0/final_speaker_acc` | 0.8 | 0.80 | yes |
| 533 | D diar fix clips | n3_R3_reg_hold_any mix6_s0 ids | `scratch/diar/offline/n3_R3_reg_hold_any.json`:`items/mix6_s0/n_ids` | 7 | 7 | yes |
| 534 | D diar fix clips | n3_R3_reg_hold_any icsi_Bmr021_940s acc | `scratch/diar/offline/n3_R3_reg_hold_any.json`:`items/icsi_Bmr021_940s/final_speaker_acc` | 0.824 | 0.82 | yes |
| 535 | D diar fix clips | n3_R3_reg_hold_any icsi_Bmr021_940s ids | `scratch/diar/offline/n3_R3_reg_hold_any.json`:`items/icsi_Bmr021_940s/n_ids` | 4 | 4 | yes |
| 536 | D diar fix clips | n3_R3_reg_hold_any ami_IS1008b_1670s acc | `scratch/diar/offline/n3_R3_reg_hold_any.json`:`items/ami_IS1008b_1670s/final_speaker_acc` | 0.75 | 0.75 | yes |
| 537 | D diar fix clips | n3_R3_reg_hold_any ami_IS1008b_1670s ids | `scratch/diar/offline/n3_R3_reg_hold_any.json`:`items/ami_IS1008b_1670s/n_ids` | 3 | 3 | yes |
| 538 | D diar fix clips | n3_R4_tita_any icsi_Bmr021_940s acc | `scratch/diar/offline/n3_R4_tita_any.json`:`items/icsi_Bmr021_940s/final_speaker_acc` | 0.765 | 0.77 | yes |
| 539 | D diar fix clips | n3_R4_tita_any icsi_Bmr021_940s ids | `scratch/diar/offline/n3_R4_tita_any.json`:`items/icsi_Bmr021_940s/n_ids` | 5 | 5 | yes |
| 540 | D diar fix clips | n3_R4_tita_any ami_IS1008b_1670s acc | `scratch/diar/offline/n3_R4_tita_any.json`:`items/ami_IS1008b_1670s/final_speaker_acc` | 0.875 | 0.88 | yes |
| 541 | D diar fix clips | n3_R4_tita_any ami_IS1008b_1670s ids | `scratch/diar/offline/n3_R4_tita_any.json`:`items/ami_IS1008b_1670s/n_ids` | 5 | 5 | yes |
| 542 | D diar fix clips | n3_R6_reg_any_shed icsi_Bmr021_940s acc | `scratch/diar/offline/n3_R6_reg_any_shed.json`:`items/icsi_Bmr021_940s/final_speaker_acc` | 0.727 | 0.73 | yes |
| 543 | D diar fix clips | n3_R6_reg_any_shed icsi_Bmr021_940s ids | `scratch/diar/offline/n3_R6_reg_any_shed.json`:`items/icsi_Bmr021_940s/n_ids` | 4 | 4 | yes |
| 544 | D diar fix clips | n3_R6_reg_any_shed ami_IS1008b_1670s acc | `scratch/diar/offline/n3_R6_reg_any_shed.json`:`items/ami_IS1008b_1670s/final_speaker_acc` | 0.571 | 0.57 | yes |
| 545 | D diar fix clips | n3_R6_reg_any_shed ami_IS1008b_1670s ids | `scratch/diar/offline/n3_R6_reg_any_shed.json`:`items/ami_IS1008b_1670s/n_ids` | 5 | 5 | yes |
| 546 | D diar fix clips | sf_R1_legacy icsi_Bmr021_940s acc | `scratch/diar/offline/sf_R1_legacy.json`:`items/icsi_Bmr021_940s/final_speaker_acc` | 0.429 | 0.43 | yes |
| 547 | D diar fix clips | sf_R1_legacy icsi_Bmr021_940s ids | `scratch/diar/offline/sf_R1_legacy.json`:`items/icsi_Bmr021_940s/n_ids` | 4 | 4 | yes |
| 548 | D diar fix clips | sf_R1_legacy ami_IS1008b_1670s acc | `scratch/diar/offline/sf_R1_legacy.json`:`items/ami_IS1008b_1670s/final_speaker_acc` | 0.571 | 0.57 | yes |
| 549 | D diar fix clips | sf_R1_legacy ami_IS1008b_1670s ids | `scratch/diar/offline/sf_R1_legacy.json`:`items/ami_IS1008b_1670s/n_ids` | 4 | 4 | yes |
| 550 | D diar fix clips | n3_R5_legacy_shed mix6_s0 acc | `scratch/diar/offline/n3_R5_legacy_shed.json`:`items/mix6_s0/final_speaker_acc` | 0.333 | 0.33 | yes |
| 551 | D diar fix clips | n3_R5_legacy_shed mix6_s0 ids | `scratch/diar/offline/n3_R5_legacy_shed.json`:`items/mix6_s0/n_ids` | 1 | 1 | yes |

## 2. Mismatches found, and what was changed

1. **`research/archive/DIARIZATION_FIX.md` §3.4, AMI dev 64 windows, "after" row** (and the §3.4 reading and item 1 of the
   proposed default; also commit message c67693a): the doc gave DER **0.229** (.183 / .021 / .025), count error
   -0.09 / 0.19, final-id error -0.93, 206 finals (3 null), per-final accuracy 0.772. Those are the aggregate of the
   resumable run's first pass, 57 of 64 windows (`scratch/diar/logs/run_ami.txt`). The complete file
   `scratch/diar/offline/ami_B_uncut_hold.json` `aggregate` (n = 64) has DER **0.2448** (.195 / .024 / .025), count
   -0.094 / 0.219, final-id error -1.031, 241 finals (3 null), accuracy 0.765, RTF 0.439: the pooled DER and count
   error equal the "before" row (`ami_A_legacy.json`, 0.2448) to four decimals. **Fixed in the doc** (row, reading,
   proposed default); the report says "AMI-64 DER unchanged at 0.245" instead of "0.245 -> 0.229". The commit message
   is history and was not rewritten.
2. **`research/archive/DIARIZATION_FIX.md` §1.3, Nemotron-3 two-session "mix" row** (count -2, acc 0.68, purity 0.76):
   `scratch/diar/logs/n3_concurrent.json` has no `clip2` and scored the second session against the ICSI reference
   (n_ref 5, 9 of 23 finals scored: count -1, acc 0.667, purity 0.669). No stored file holds the doc's values.
   **Annotated in the doc**; the report does not use the row.
3. **`research/archive/BULLETPROOF.md`** said `tests/test_bulletproof.py` has 45 tests; `pytest --collect-only` collects
   **48** (commit 548b629 also says 48). **Fixed in the doc.** The report's "48 tests" is the collected count; the
   note has 36 failure-mode rows (sections 1-5), not 48.
4. **`research/archive/IMPROVEMENTS.md` §3, TitaNet-L row**: "8.2 / –" sat in the "AMI n=64 all / within" column, i.e. the
   within-meeting value in the "all" slot. JSON: 6.6 / 8.2 (n = 64), 10.9 / 12.0 (n = 200) (`runs/baselines_sd.json`
   `spk/titanet_large`), ICSI 1.0 within (n = 64), 2.2 / 2.1 (n = 200) (`runs/spk_head.json`
   `variants/spk_distill_titanet/eval/icsi_*/teacher`). **Fixed in the doc.**
5. **"Identical outputs" for the performance pass** (`research/archive/PERFORMANCE.md` §3, README, CHANGELOG):
   `runs/perf.json` `ab/none_default/compare` stores `identical: false`, `within_tol: true`, 0 decision differences
   (turn_end, final, partial, primary), `maxdiff` eot 1e-05, vad 0. PERFORMANCE.md's "1311 of 1313 messages
   bit-identical" is consistent with that but is not itself stored (`n_num` 0, `examples` empty). The report, README
   and CHANGELOG now say "decisions identical, probabilities within 1e-5"; PERFORMANCE.md is left as written (its
   own sentence already names the 2 differing eot values).
6. **TS-VAD frame result in the 2026-09-27 report §11**: F1 0.745 (precision 0.721, miss 0.229, FA 0.189) from a
   scratch file of an earlier protocol version; `runs/improve_115m.json` `frame/ami/primary/tsvad_spk_vp5p0/all`
   has F1 0.7428, precision 0.7503, miss 0.2645, FA 0.1545. The report now uses the JSON (and lists it in its
   Appendix B).
7. **Speaker head after experiment 3**: the hand-off summary quoted "19.8 -> 17.2 AMI / 7.0 -> 4.2 ICSI". The JSON
   has two heads: warm start 17.1 / 4.4, fresh init 17.4 / 4.0 (`runs/spk_frame.json`, within-meeting, n = 200). The
   report states the range 17.1-17.4 / 4.0-4.4; 17.2 / 4.2 is neither head.
8. **Robustness "48 failure modes, 3 real bugs"** (hand-off summary): 48 is the test count (item 3). The three
   defects in shipped code that BULLETPROOF.md documents as found and fixed are: a late `turn_policy` accepted
   silently when audio had arrived before the session object existed; non-finite input samples poisoning the encoder
   caches for the rest of the session; and a hung or dead `--final-asr` worker making `end` wait forever. The report
   uses that wording. None of these has a JSON (they are pinned by tests).

## 3. Numbers in the report without a runs/ JSON (stated as such there)

| number | source |
|---|---|
| 0.6B vs 115M VAD F1 (0.9329 / 0.9398 AMI, 0.9039 / 0.9079 ICSI, paired CIs) | `scratch/hybrid_asr/core_0p6b_rtf.json` `vad` (checked above, rows under "0.6B core") |
| diarization-fix clip, AMI-64 and live tables | `scratch/diar/offline/*.json`, `scratch/diar/logs/*.json` (checked above) |
| TS-VAD live server RTF (T median 0.34, max 0.50; Th 0.29, max 0.34), peak RSS 1.6 GB | `scratch/e2e_tsvad/runs/pipecat_{T,Th}_*.jsonl` `raw.server_stats` (recomputed: 0.3403 / 0.4982, 0.2858 / 0.3382, 1615.8-1630.7 MB) |
| served TS-VAD path == offline (max track diff 5.3e-4, turn posterior 4.0e-3, 179 / 179 frames >= 0.997, 1830 frames) | `scratch/improve/serve_check.json` (recomputed: 5.27e-4, 3.96e-3, 179 / 179, 1830) |
| 48 bulletproof tests | `pytest --collect-only tests/test_bulletproof.py` |
| on-device runtime study RTF 0.16 (1 thread, random weights) | `research/archive/ONDEVICE.md` (its JSON was lost in a reboot) |
| distillation package GPU estimate (~6 A100 hours) | `scripts/research/distill_0p6b_to_115m/README.md` (an estimate, not a measurement) |
| TurnBench / otoSpeech / LID / E2E rows carried from 2026-09-27 | checked above against their JSON |
