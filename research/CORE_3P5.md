# Nemotron 3.5 ASR streaming 0.6B as the 0.6B core: measured, not adopted

2026-10-01. Script: `scripts/research/core_0p6b_heads.py --core 3.5` (stages `import`, `verify`, `wer`, `wer_report`,
`live_wer`, `core`). Numbers: `runs/core_3p5.json`. Mac only: MPS, one job at a time through `scripts/dev/gate.sh`.

**Decision (user, 2026-10-01): the English `nemotron-speech-streaming-en-0.6b` stays the `--core 0.6b`.** Head work
on the 3.5 was stopped before any head was trained. Only the bare-ASR row and its cost were finished, so the
comparison is on record.

## What the model is

- `nvidia/nemotron-3.5-asr-streaming-0.6b`:
  - pinned at revision `ea30d66debe3740a08b573244286791d423d6b3e`;
  - `nemotron-3.5-asr-streaming-0.6b.nemo` is 2 368 284 501 bytes, sha256 `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
    (equal to the HF LFS etag).
- **Licence:** OpenMDW-1.1. It permits commercial use and modification, with no copyleft. A redistribution must keep
  the licence text and the notices. Rights end if you sue claiming the model infringes your IP.
- **Architecture:** the English 0.6B's cache-aware FastConformer (24 × 1024, RNNT), plus a language prompt.
  - The prompt is a one-hot over 128 slots, concatenated with each encoder frame, then Linear(1152, 2048), ReLU,
    Linear(2048, 1024), then the joint. It touches only the RNNT path: the encoder blocks the heads read do not depend
    on it.
  - 638.4 M parameters (English 0.6B: 618.5 M; most of the difference is the 13 088-piece multilingual vocabulary).
- **Trained contexts:** left 56 with right context 0 / 3 / 6 / 13, i.e. 80 / 320 / 560 / 1120 ms chunks. **160 ms
  ([56, 1]) is not among them**; the English 0.6B was trained at [70, 1].

## Conversion and streaming check

- `audioforge.nemo_import` imports the checkpoint. The prompt kernel becomes `heads.rnnt.joint.enc`
  (`heads/asr.py PromptedLinear`, `RNNTHead.set_prompt("en-US" | "auto" | locale)`).
  - The `<xx-XX>` language-tag pieces are tokenizer specials: kept in the token ids, dropped from the decoded text.
  - Read back tensor-identical. Filterbank / window match NeMo's to 4e-9 / 6e-8.
  - ReLU as the kernel's activation was confirmed empirically: SiLU and GELU give empty transcripts.
- **Streaming equals offline at [56, 1]** (3 LibriSpeech utterances):
  - `stream_step` vs the masked offline forward: max |d| ≤ 1.9e-6;
  - `StreamingSession` text equals the offline text for 3 of 3.
- The AFM is kept at `/Volumes/ExternalSSD/nvidia-audio-models/runs/nemo_nemotron_3p5_asr_streaming_0.6b.afm`.
  **There is no hub component, pin or served asset.**

## Word error rate (bare ASR, greedy RNNT, 200 utterances per set, 95 % utterance-bootstrap CI)

At 160 ms chunks. The 115M and English 0.6B rows are their [70, 1] hypotheses of the same sets (`runs/hybrid_asr.json`).
`normalize_text` is the published protocol. `nofill` also drops fillers (uh / um), because the 3.5 tends to leave
fillers out.

| set | 115M | English 0.6B | 3.5, en-US prompt | 3.5, auto | 3.5 en-US − English 0.6B [CI] |
|---|---:|---:|---:|---:|---|
| LibriSpeech-200 | 2.29 % | 2.20 % | 2.59 % [1.95, 3.23] | 2.50 % | +0.39 [−0.02, +0.78] |
| AMI-200 | 24.43 % | **11.16 %** | 18.43 % [16.43, 20.87] | 20.42 % | **+7.28 [+5.68, +9.02]** |
| AMI-200, nofill | 21.40 % | **10.58 %** | 15.22 % | 17.15 % | +4.64 [+2.99, +6.36] |
| ICSI-200 | 27.29 % | **14.35 %** | 22.09 % [19.54, 24.74] | 22.20 % | **+7.74 [+5.99, +9.57]** |
| ICSI-200, nofill | 26.48 % | **13.83 %** | 20.44 % | 20.44 % | +6.61 [+5.04, +8.40] |
| 32 live calls, every word (16 TurnBench clips × mono / user channel; 95 % clip CI) | 22.47 % | **13.70 %** [10.77, 16.96] | 16.99 % [13.59, 21.14] | 16.99 % | |
| the same, user channel only | 17.09 % | **7.69 %** [5.74, 9.39] | 11.17 % [9.24, 13.44] | 11.24 % | |

The 3.5 at its trained chunk sizes (en-US prompt) does not close the gap:

| set | 80 ms [56, 0] | 160 ms [56, 1] | 320 ms [56, 3] | English 0.6B at 160 ms |
|---|---:|---:|---:|---:|
| LibriSpeech-200 | 2.70 % | 2.59 % | 2.50 % | 2.20 % |
| AMI-200 | 19.02 % | 18.43 % | 17.24 % (+6.08 [+4.70, +7.69] vs English) | 11.16 % |
| ICSI-200 | 23.18 % | 22.09 % | 19.90 % (+5.55 [+4.05, +7.02]) | 14.35 % |

Language prompt:
- **auto** (detect the language) is no better than en-US on English. AMI is +2.0 points (CI [+0.2, +4.8]).
- It also mislabels some utterances: AMI 8 tags other than en-US (hi-IN, fr-FR, de-DE …), LibriSpeech 8 × vi-VN.
- The tag comes only after the terminal punctuation: 81 of 200 AMI utterances got no tag at all.

## Cost (bare core, `StreamingSession`, 160 ms chunks, 20 LibriSpeech utterances, best of 3)

| | English 0.6B | 3.5 |
|---|---:|---:|
| MPS ms per chunk p50 / p95 | 30.8 / 35.2 | 31.7 / 37.2 (RTF 0.20) |
| peak RSS / MPS driver memory | 4.9 GB / 3.3 GB | 4.8 GB / 3.2 GB |

CPU cost and stream counts were not measured: the work stopped first. The encoder is the same size, so they are
expected to match the English 0.6B's 99.5 ms per chunk on 2 threads (one real-time stream).

## Why it is not the 0.6B core

- English is the product's language. On English meetings and calls the 3.5 has **7–8 points more word errors** than
  the English 0.6B: AMI 18.4 vs 11.2 %, ICSI 22.1 vs 14.4 %, live calls 17.0 vs 13.7 %. The CIs are far from zero.
- Even at its best trained chunk (320 ms, double the latency) it stays 5.5–6 points behind.
- It costs the same to run, so nothing makes up for that. NVIDIA's own model card recommends the English model for
  English-only use.
- What it adds is 40 locales and a language tag. For language ID we already have a head: FLEURS-17 87.4 / 95.3 % at
  2 s / full on the English 0.6B. Whether the 3.5's prompt makes that head redundant was not measured.

## Not done (stopped by the decision)

- No heads were trained or measured on the 3.5: VAD, speaker, TS-VAD, v5, per-frame turn, LID.
- No presets, no served model, no `served_check`.
- The 3.5 feature caches (5.2 GB of a started VAD cache) were deleted. The .afm, the WER hypotheses
  (`scratch/core_3p5/wer`) and the logs are kept.

DONE
