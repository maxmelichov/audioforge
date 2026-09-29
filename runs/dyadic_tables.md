### behavior_sd: 652 human-party turn ends, 137 conversations (post-end frames p50 57, end reasons {'resume': 410, 'trail': 242, 'meeting_end': 0})

Block C (6 s windows), cross-fitted <= 5 % per-turn FC (natives at their fixed point):

| system | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % (n) | taken miss % (n) |
|---|---|---|---|---|---|---|
| timeout_primary_oracle | 4.5 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 320 | 100.0 (0) | 0.0 (652) |
| timeout_1040ms_primary_oracle | 0.1 | 0.6 [0.1, 1.2] | 0.6 [0.1, 1.2] | 1040 | 100.0 (0) | 0.6 (652) |
| timeout_any_speaker_oracle | 3.7 | 18.1 [15.1, 21.0] | 25.0 [21.5, 28.2] | 80 | 100.0 (0) | 18.1 (652) |
| silero_timeout | 6.4 | 97.7 [96.5, 98.8] | 99.7 [99.2, 100.0] | inf | 100.0 (0) | 97.7 (652) |
| silero_timeout_1000ms | 8.0 | 96.5 [95.0, 97.8] | 99.3 [98.5, 100.0] | inf | 100.0 (0) | 96.5 (652) |
| head_trail6_oracle_track | 4.3 | 3.0 [1.8, 4.5] | 6.9 [5.1, 9.0] | 880 | 100.0 (0) | 3.0 (652) |
| hybrid_oracle | 5.7 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 240 | 100.0 (0) | 0.0 (652) |
| head_trail6+silero_timeout | 4.6 | 3.0 [1.8, 4.5] | 6.9 [5.1, 9.0] | 880 | 100.0 (0) | 3.0 (652) |
| eou_posterior | 4.8 | 57.5 [53.4, 61.3] | 62.6 [58.9, 66.3] | inf | 100.0 (0) | 57.5 (652) |
| eou_native (<EOU> emitted) | 2.9 | 63.8 [60.0, 67.3] | 68.2 [64.5, 71.6] | inf | 100.0 (0) | 63.8 (652) |

Paired differences at 6 s (miss points, 95 % CI; open / taken point estimates):

| a - b | all | open | taken |
|---|---|---|---|
| head_trail6_oracle_track - timeout_primary_oracle | +3.0 [+1.8, +4.5] | +nan | +3.0 |
| hybrid_oracle - timeout_primary_oracle | +0.0 [+0.0, +0.0] | +nan | +0.0 |
| hybrid_oracle - head_trail6_oracle_track | -3.0 [-4.6, -1.8] | +nan | -3.0 |
| silero_timeout - timeout_primary_oracle | +97.7 [+96.5, +98.8] | +nan | +97.7 |
| head_trail6_oracle_track - silero_timeout | -94.7 [-96.3, -92.8] | +nan | -94.7 |
| head_trail6+silero_timeout - silero_timeout | -94.7 [-96.3, -92.8] | +nan | -94.7 |
| eou_posterior - silero_timeout | -40.2 [-44.4, -36.1] | +nan | -40.2 |
| eou_posterior - timeout_primary_oracle | +57.5 [+53.4, +61.3] | +nan | +57.5 |
| timeout_any_speaker_oracle - timeout_primary_oracle | +18.1 [+15.1, +21.0] | +nan | +18.1 |

Frozen AMI operating points (fitted on all 974 AMI dev turns, applied unchanged):

| system | AMI point | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % | taken miss % |
|---|---|---|---|---|---|---|---|
| timeout_primary_oracle | {"timeout_threshold": 18.0, "k_frames": 19, "k_ms": 1520.0} | 0.0 | 8.3 [6.1, 10.3] | 8.3 [6.1, 10.3] | 1520 | 100.0 | 8.3 |
| timeout_any_speaker_oracle | {"timeout_threshold": 17.0, "k_frames": 18, "k_ms": 1440.0} | 0.0 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | inf | 100.0 | 100.0 |
| silero_timeout | {"timeout_threshold": 23.200000762939453, "k_frames": 24, "k_ms": 1920.0} | 0.1 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | inf | 100.0 | 100.0 |
| head_trail6_oracle_track | {"theta": 0.9791455821976657} | 0.9 | 6.8 [5.0, 8.8] | 14.5 [11.9, 17.5] | 1280 | 100.0 | 6.8 |
| hybrid_oracle | {"theta": 0.9798172116279602, "timeout_threshold": 23.0} | 0.9 | 5.0 [3.2, 6.7] | 5.0 [3.2, 6.7] | 1280 | 100.0 | 5.0 |
| head_trail6+silero_timeout | {"theta": 0.9798172116279602, "timeout_threshold": 46.79999923706055, "k_frames": 47} | 0.9 | 7.0 [5.1, 9.1] | 15.3 [12.5, 18.5] | 1280 | 100.0 | 7.0 |

### dailytalk: 650 human-party turn ends, 157 conversations (post-end frames p50 45, end reasons {'resume': 554, 'trail': 96, 'meeting_end': 0})

Block C (6 s windows), cross-fitted <= 5 % per-turn FC (natives at their fixed point):

| system | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % (n) | taken miss % (n) |
|---|---|---|---|---|---|---|
| timeout_primary_oracle | 4.6 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 720 | 0.0 (40) | 0.0 (610) |
| timeout_1040ms_primary_oracle | 3.2 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1040 | 0.0 (40) | 0.0 (610) |
| timeout_any_speaker_oracle | 5.1 | 50.1 [46.0, 54.1] | 64.5 [60.9, 68.1] | inf | 0.0 (40) | 53.2 (610) |
| silero_timeout | 4.8 | 72.4 [68.6, 76.0] | 80.1 [76.9, 83.2] | inf | 5.3 (40) | 76.8 (610) |
| silero_timeout_1000ms | 2.0 | 89.6 [87.2, 92.1] | 92.8 [90.7, 94.8] | inf | 10.0 (40) | 95.0 (610) |
| head_trail6_oracle_track | 4.9 | 1.3 [0.5, 2.2] | 3.9 [2.6, 5.5] | 880 | 2.7 (40) | 1.2 (610) |
| hybrid_oracle | 5.2 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 720 | 0.0 (40) | 0.0 (610) |
| head_trail6+silero_timeout | 5.4 | 1.3 [0.5, 2.3] | 3.7 [2.4, 5.3] | 960 | 2.7 (40) | 1.2 (610) |
| eou_posterior | 4.9 | 87.9 [85.2, 90.5] | 92.1 [89.9, 94.1] | inf | 76.3 (40) | 88.6 (610) |
| eou_native (<EOU> emitted) | 0.5 | 93.3 [91.5, 95.2] | 96.8 [95.2, 98.0] | inf | 84.6 (40) | 93.9 (610) |

Paired differences at 6 s (miss points, 95 % CI; open / taken point estimates):

| a - b | all | open | taken |
|---|---|---|---|
| head_trail6_oracle_track - timeout_primary_oracle | +1.3 [+0.5, +2.2] | +2.7 | +1.2 |
| hybrid_oracle - timeout_primary_oracle | +0.0 [+0.0, +0.0] | +0.0 | +0.0 |
| hybrid_oracle - head_trail6_oracle_track | -1.3 [-2.2, -0.5] | -2.7 | -1.2 |
| silero_timeout - timeout_primary_oracle | +72.4 [+68.6, +76.0] | +5.3 | +76.8 |
| head_trail6_oracle_track - silero_timeout | -71.1 [-74.9, -67.2] | -2.6 | -75.6 |
| head_trail6+silero_timeout - silero_timeout | -71.1 [-74.9, -67.2] | -2.6 | -75.5 |
| eou_posterior - silero_timeout | +15.5 [+11.4, +19.5] | +71.0 | +11.9 |
| eou_posterior - timeout_primary_oracle | +87.9 [+85.2, +90.5] | +76.3 | +88.6 |
| timeout_any_speaker_oracle - timeout_primary_oracle | +50.1 [+46.0, +54.1] | +0.0 | +53.2 |

Frozen AMI operating points (fitted on all 974 AMI dev turns, applied unchanged):

| system | AMI point | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % | taken miss % |
|---|---|---|---|---|---|---|---|
| timeout_primary_oracle | {"timeout_threshold": 18.0, "k_frames": 19, "k_ms": 1520.0} | 1.2 | 3.1 [1.9, 4.7] | 3.1 [1.9, 4.7] | 1520 | 10.3 | 2.6 |
| timeout_any_speaker_oracle | {"timeout_threshold": 17.0, "k_frames": 18, "k_ms": 1440.0} | 0.1 | 97.7 [96.5, 98.8] | 98.6 [97.7, 99.4] | inf | 77.5 | 99.0 |
| silero_timeout | {"timeout_threshold": 23.200000762939453, "k_frames": 24, "k_ms": 1920.0} | 0.0 | 99.7 [99.2, 100.0] | 99.9 [99.5, 100.0] | inf | 97.5 | 99.8 |
| head_trail6_oracle_track | {"theta": 0.9791455821976657} | 1.5 | 8.4 [6.4, 10.7] | 23.0 [20.1, 26.2] | 1600 | 15.0 | 8.0 |
| hybrid_oracle | {"theta": 0.9798172116279602, "timeout_threshold": 23.0} | 1.5 | 3.1 [1.7, 4.5] | 3.1 [1.7, 4.5] | 1600 | 10.0 | 2.7 |
| head_trail6+silero_timeout | {"theta": 0.9798172116279602, "timeout_threshold": 46.79999923706055, "k_frames": 47} | 1.5 | 8.4 [6.4, 10.7] | 23.9 [20.9, 27.1] | 1600 | 15.0 | 8.0 |

### oto: 581 human-party turn ends, 16 conversations (post-end frames p50 52, end reasons {'resume': 380, 'trail': 201, 'meeting_end': 0})

Block C (6 s windows), cross-fitted <= 5 % per-turn FC (natives at their fixed point):

| system | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % (n) | taken miss % (n) |
|---|---|---|---|---|---|---|
| timeout_primary_oracle | 5.2 | 11.1 [8.4, 13.8] | 11.1 [8.4, 13.8] | 1600 | 1.8 (177) | 14.9 (404) |
| timeout_1040ms_primary_oracle | 19.3 | 1.3 [0.4, 2.4] | 1.3 [0.4, 2.4] | 1040 | 0.0 (177) | 1.7 (404) |
| timeout_any_speaker_oracle | 5.5 | 75.8 [72.2, 79.2] | 81.4 [78.3, 84.4] | inf | 38.8 (177) | 91.7 (404) |
| silero_timeout | 7.4 | 77.7 [74.1, 81.2] | 83.8 [80.7, 86.7] | inf | 45.1 (177) | 91.8 (404) |
| silero_timeout_1000ms | 27.2 | 54.6 [49.5, 59.4] | 61.9 [56.9, 66.4] | inf | 0.0 (177) | 72.9 (404) |
| head_trail6_oracle_track | 5.5 | 12.9 [10.1, 15.8] | 32.8 [28.9, 36.8] | 1680 | 12.0 (177) | 13.4 (404) |
| hybrid_oracle | 6.7 | 10.2 [7.6, 12.7] | 10.2 [7.6, 12.7] | 1680 | 1.9 (177) | 13.6 (404) |
| head_trail6+silero_timeout | 6.9 | 11.5 [8.7, 14.2] | 29.0 [25.2, 33.1] | 1760 | 4.9 (177) | 14.4 (404) |
| eou_posterior | 2.4 | 94.0 [91.9, 95.9] | 94.2 [92.1, 96.0] | inf | 88.4 (177) | 96.5 (404) |
| eou_native (<EOU> emitted) | 10.3 | 72.7 [68.7, 76.5] | 77.3 [73.8, 80.9] | inf | 46.2 (177) | 84.1 (404) |

Paired differences at 6 s (miss points, 95 % CI; open / taken point estimates):

| a - b | all | open | taken |
|---|---|---|---|
| head_trail6_oracle_track - timeout_primary_oracle | +1.9 [-0.5, +4.2] | +10.1 | -1.6 |
| hybrid_oracle - timeout_primary_oracle | -0.9 [-2.2, +0.2] | +0.0 | -1.3 |
| hybrid_oracle - head_trail6_oracle_track | -2.8 [-4.8, -0.8] | -10.1 | +0.3 |
| silero_timeout - timeout_primary_oracle | +66.6 [+62.8, +70.6] | +43.2 | +76.8 |
| head_trail6_oracle_track - silero_timeout | -64.8 [-69.1, -60.6] | -33.1 | -78.4 |
| head_trail6+silero_timeout - silero_timeout | -66.2 [-70.1, -62.3] | -40.2 | -77.4 |
| eou_posterior - silero_timeout | +16.3 [+12.6, +20.2] | +43.4 | +4.7 |
| eou_posterior - timeout_primary_oracle | +82.9 [+79.8, +86.1] | +86.6 | +81.5 |
| timeout_any_speaker_oracle - timeout_primary_oracle | +64.7 [+60.8, +68.5] | +36.9 | +76.8 |

Frozen AMI operating points (fitted on all 974 AMI dev turns, applied unchanged):

| system | AMI point | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % | taken miss % |
|---|---|---|---|---|---|---|---|
| timeout_primary_oracle | {"timeout_threshold": 18.0, "k_frames": 19, "k_ms": 1520.0} | 7.1 | 9.8 [7.3, 12.6] | 9.8 [7.3, 12.6] | 1520 | 1.3 | 13.4 |
| timeout_any_speaker_oracle | {"timeout_threshold": 17.0, "k_frames": 18, "k_ms": 1440.0} | 6.4 | 74.6 [71.2, 78.1] | 80.3 [77.2, 83.4] | inf | 35.2 | 91.4 |
| silero_timeout | {"timeout_threshold": 23.200000762939453, "k_frames": 24, "k_ms": 1920.0} | 2.8 | 80.2 [76.9, 83.4] | 85.7 [82.8, 88.6] | inf | 49.1 | 93.9 |
| head_trail6_oracle_track | {"theta": 0.9791455821976657} | 16.7 | 4.5 [2.9, 6.5] | 12.0 [9.2, 15.1] | 1200 | 2.8 | 5.3 |
| hybrid_oracle | {"theta": 0.9798172116279602, "timeout_threshold": 23.0} | 15.8 | 4.1 [2.5, 5.9] | 4.1 [2.5, 5.9] | 1200 | 1.4 | 5.2 |
| head_trail6+silero_timeout | {"theta": 0.9798172116279602, "timeout_threshold": 46.79999923706055, "k_frames": 47} | 15.8 | 4.5 [2.9, 6.4] | 11.9 [9.1, 14.8] | 1200 | 2.7 | 5.2 |

### turnbench: 419 both-party turn ends, 10 conversations (post-end frames p50 58, end reasons {'resume': 255, 'trail': 164, 'meeting_end': 0})

Block C (6 s windows), cross-fitted <= 5 % per-turn FC (natives at their fixed point):

| system | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % (n) | taken miss % (n) |
|---|---|---|---|---|---|---|
| timeout_primary_oracle | 6.0 | 5.3 [3.1, 7.5] | 5.3 [3.1, 7.5] | 1440 | 2.4 (51) | 5.7 (368) |
| timeout_1040ms_primary_oracle | 6.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1040 | 0.0 (51) | 0.0 (368) |
| timeout_any_speaker_oracle | 5.2 | 80.3 [76.3, 84.2] | 84.4 [80.9, 88.1] | inf | 7.5 (51) | 88.5 (368) |
| silero_timeout | 7.9 | 90.7 [87.6, 93.6] | 93.5 [91.0, 95.9] | inf | 42.9 (51) | 96.5 (368) |
| silero_timeout_1000ms | 15.5 | 82.5 [78.2, 86.4] | 85.9 [82.1, 89.5] | inf | 3.3 (51) | 89.8 (368) |
| head_trail6_oracle_track | 4.3 | 4.2 [2.5, 6.3] | 9.7 [6.8, 12.5] | 1120 | 8.5 (51) | 3.7 (368) |
| hybrid_oracle | 7.6 | 2.3 [0.8, 3.9] | 2.3 [0.8, 3.9] | 800 | 2.4 (51) | 2.3 (368) |
| head_trail6+silero_timeout | 7.4 | 3.6 [1.8, 5.5] | 7.8 [5.0, 10.4] | 1120 | 2.3 (51) | 3.8 (368) |
| eou_posterior | 7.9 | 92.8 [90.2, 95.1] | 95.9 [93.8, 97.7] | inf | 62.8 (51) | 96.5 (368) |
| eou_native (<EOU> emitted) | 1.4 | 96.6 [94.9, 98.1] | 98.6 [97.3, 99.5] | inf | 88.2 (51) | 97.8 (368) |

Paired differences at 6 s (miss points, 95 % CI; open / taken point estimates):

| a - b | all | open | taken |
|---|---|---|---|
| head_trail6_oracle_track - timeout_primary_oracle | -1.1 [-3.7, +1.5] | +6.1 | -2.0 |
| hybrid_oracle - timeout_primary_oracle | -3.0 [-5.1, -1.0] | +0.0 | -3.4 |
| hybrid_oracle - head_trail6_oracle_track | -1.9 [-3.4, -0.7] | -6.1 | -1.4 |
| silero_timeout - timeout_primary_oracle | +85.3 [+81.9, +88.9] | +40.4 | +90.8 |
| head_trail6_oracle_track - silero_timeout | -86.4 [-90.1, -82.9] | -34.4 | -92.8 |
| head_trail6+silero_timeout - silero_timeout | -87.1 [-90.4, -83.7] | -40.5 | -92.7 |
| eou_posterior - silero_timeout | +2.1 [-0.8, +4.9] | +19.9 | -0.0 |
| eou_posterior - timeout_primary_oracle | +87.4 [+84.2, +90.6] | +60.4 | +90.8 |
| timeout_any_speaker_oracle - timeout_primary_oracle | +75.0 [+70.7, +79.3] | +5.1 | +82.8 |

Frozen AMI operating points (fitted on all 974 AMI dev turns, applied unchanged):

| system | AMI point | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % | taken miss % |
|---|---|---|---|---|---|---|---|
| timeout_primary_oracle | {"timeout_threshold": 18.0, "k_frames": 19, "k_ms": 1520.0} | 1.9 | 8.3 [5.6, 10.9] | 8.3 [5.6, 10.9] | 1520 | 2.2 | 9.0 |
| timeout_any_speaker_oracle | {"timeout_threshold": 17.0, "k_frames": 18, "k_ms": 1440.0} | 2.1 | 88.5 [85.3, 91.4] | 92.0 [89.2, 94.4] | inf | 23.9 | 96.7 |
| silero_timeout | {"timeout_threshold": 23.200000762939453, "k_frames": 24, "k_ms": 1920.0} | 3.8 | 92.3 [89.8, 94.8] | 94.5 [92.3, 96.8] | inf | 47.8 | 98.0 |
| head_trail6_oracle_track | {"theta": 0.9791455821976657} | 6.4 | 3.8 [2.0, 5.8] | 8.2 [5.4, 10.8] | 1040 | 8.9 | 3.2 |
| hybrid_oracle | {"theta": 0.9798172116279602, "timeout_threshold": 23.0} | 6.2 | 2.8 [1.2, 4.4] | 2.8 [1.2, 4.4] | 1040 | 2.2 | 2.9 |
| head_trail6+silero_timeout | {"theta": 0.9798172116279602, "timeout_threshold": 46.79999923706055, "k_frames": 47} | 6.4 | 3.6 [1.8, 5.6] | 8.4 [5.8, 10.8] | 1040 | 6.7 | 3.2 |

### TurnBench dev (official scorer, EOT task, dev operating point = highest recall at fp_rate <= 0.10)

| system | operating point | recall | fp_rate | latency p10 / p50 / p90 ms |
|---|---|---|---|---|
| rms_timeout | 2.0 | 0.648 | 0.071 | 604 / 1653 / 1905 |
| silero_timeout | 1.5 | 0.815 | 0.075 | 844 / 1399 / 1525 |
| silero_mix_timeout | 1.2 | 0.117 | 0.091 | 901 / 1210 / 2418 |
| head_trail6 | 0.9973489046096802 | 0.752 | 0.099 | 237 / 1058 / 2258 |
| hybrid | [0.9988547563552856, 1.5] | 0.835 | 0.080 | 760 / 1390 / 1537 |
| head+silero_mix | [0.9988547563552856, 1.5] | 0.353 | 0.051 | 914 / 1884 / 2807 |
| eou_posterior | -749.5036926269531 | 0.763 | 0.084 | 401 / 1097 / 2151 |
| eou_native | None | 0.123 | 0.003 | 648 / 1788 / 2733 |

Published dev predictions re-scored with the same vendored scorer (their committed operating points):

| baseline | recall | fp_rate | p50 ms | published test recall / fp / p50 |
|---|---|---|---|---|
| vap | 0.841 | 0.045 | 463 | 0.845 / 0.055 / 368 |
| smart_turn_v3 | 0.754 | 0.100 | 1010 | 0.752 / 0.047 / None |
| kyutai_semantic_vad | 0.803 | 0.100 | 1024 | 0.773 / 0.059 / 1007 |
| rms_vad | 0.595 | 0.547 | -98 | - |
| espnet_turntaking | 0.836 | 0.074 | 895 | - |
| espnet_turntaking_perchannel | 0.640 | 0.100 | 846 | - |
| wavlm_large_causal | 0.472 | 0.100 | 683 | - |
| mimi_endpointer | 0.759 | 0.047 | 742 | - |
| openai_server_vad | 0.933 | 0.564 | 281 | - |
| openai_semantic_vad | 0.310 | 0.037 | 763 | - |
| oracle_annotator | 1.000 | 0.000 | 0 | - |