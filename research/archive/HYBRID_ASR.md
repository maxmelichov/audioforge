# Hybrid front end: the served 115M streaming model plus a better transcript

2026-09-28 (the hybrid agent left no write-up; this one is assembled from the committed code, `runs/hybrid_asr.json`
and the measurements finished in the issues pass, `research/archive/ISSUES_FIXED.md` B.2-B.3).

**Question.** The served model (`runs/stage1_served.afm`: NVIDIA's 115M cache-aware streaming FastConformer hybrid
encoder, RNNT head, att_context [70, 1] = 160 ms chunks) keeps every live decision cheap, but its meeting WER is
24.4 % on AMI dev segments (`runs/final_asr.json`) against 9.7 % for NVIDIA's offline Parakeet-TDT 0.6B v3. Two ways
to give the agent a better transcript **without touching a single live decision** were built into `audioforge.serve`:

1. **`--final-asr tdt_v3`** (hybrid): a second, offline model transcribes each finished turn once, in its own process
   (`audioforge.final_asr.FinalASRWorker`); the streaming final is sent as before with `"source": "stream"` and a
   second final follows with `"source": "tdt_v3"`, the same `t`, `speaker`, the span and `latency_ms`.
2. **`--asr-lookahead R`** (dual lookahead): a second, text-only cache-aware pass of the *same* 115M model at
   att_context [70, R] (R = 13: 1.12 s chunks, 1.04 s lookahead; `serve.LookaheadStream`), sent as
   `"source": "lookahead"`. No second model, ~+10 % encoder compute.

Both are off by default. The Pipecat and LiveKit adapters take `final_source="stream"|"offline"`: with `offline` the
agent answers the second source's transcript and the turn is committed when that final has arrived.

## 1. Import notes and licence (Parakeet-TDT 0.6B v3)

- `nvidia/parakeet-tdt-0.6b-v3`, **CC-BY-4.0** (card: "GOVERNING TERMS: Use of this model is governed by the
  CC-BY-4.0 license"). Attribution is required when the transcripts or a model trained on them are redistributed
  (`research/archive/TEACHERS.md` "Licences").
- Imported with `audioforge.nemo_import` from the memory-mapped `.nemo` (no `.afm` copy, `data/nemo/`): the only
  change against the RNNT import is the head. `tdt_durations(nc)` reads the duration set from
  `model_defaults.tdt_durations`, `loss.tdt_kwargs.durations` and `decoding.durations` (all must agree) and checks it
  against `joint.num_extra_outputs`; the joint output is `[vocab..., blank, durations...]` = 8192 + 1 + 5 rows,
  NeMo's `RNNTJoint` layout. Greedy decoding takes the argmax token and the argmax duration; blank with duration 0
  advances one frame; `max_symbols` 10 per frame; `GreedyTransducerStream` decodes frames arriving in pieces and
  equals the offline decode (`tests/test_hybrid_asr.py`). Real import: 725 tensors, 723 loaded (strict count
  `n_loaded == archive - 2`, the two are the preprocessor's featurizer buffers), 264 zero biases (`use_bias: false`),
  627.5M parameters, 6 s to load (`research/archive/NEMO_IMPORT.md` "Parakeet-TDT").
- v3 has **no language token**: one 8192-piece SentencePiece shared by 25 languages, no prompt and no language-like
  piece in the vocabulary (`hybrid_asr.py lid_vocab`, `runs/hybrid_asr.json["lid_vocab"]`). Its language decision
  is only visible in the text it writes (section 5).
- Machine rule kept: it is never loaded in a process that holds parakeet-ctc-0.6b or the nemotron 0.6B; the
  `nemotron` stage of `scripts/hybrid_asr.py` was therefore **not run** (no `nemotron_0.6b_160ms` rows).

## 2. Protocol

`scripts/final_asr.py`'s sets and scoring, reused as they are: **AMI-200** (AMI dev single-speaker segments 1-15 s,
`random.Random(0).sample`, sorted), **LibriSpeech-200** (test-clean, first 200), and the new **ICSI-200** (ICSI dev
meetings Bmr021 / Bns001, the same sampling; `hybrid_asr.py prep_icsi`). Scoring `audioforge.teachers.normalize_text`
on both sides (primary), plus `nofill` (fillers dropped) and Whisper's `EnglishTextNormalizer`. 1000-resample
utterance bootstrap CIs; paired deltas on the same resamples. Served rows: `m.transcribe` at the given att_context,
batch 4 (the masked offline forward equals chunk-by-chunk cache-aware streaming, `tests/test_streaming.py`). TDT v3:
batch 1, CPU, 2 threads, fp32, greedy. RTF here = decode wall / audio on a shared, loaded machine ("under load").


## 3. Dual lookahead: the same 115M model at [70,0] / [70,1] / [70,13] (ISSUES_FIXED B.2)

`scripts/research/hybrid_asr.py lookahead` (the served `.afm`, masked offline forward at each att_context; ICSI-200 by
`prep_icsi`). WER % with `normalize_text` (primary), 1000-resample bootstrap CIs; RTF = batch-4 decode wall / audio
under load. `runs/hybrid_asr.json["english"]`.

| set (n = 200) | [70,0] (no lookahead) | **[70,1] (served, 160 ms)** | [70,13] (`--asr-lookahead 13`) | RTF [70,0] / [70,1] / [70,13] |
|---|---:|---:|---:|---|
| AMI dev segments | 25.68 [22.97, 28.63] | **24.43 [22.04, 27.22]** | 23.02 [20.54, 25.79] | 0.022 / 0.021 / 0.025 |
| ICSI dev segments | 29.87 [27.19, 33.04] | **27.29 [24.75, 30.24]** | 24.78 [21.92, 27.73] | 0.086 / 0.041 / 0.054 |
| LibriSpeech test-clean | 2.48 [1.93, 3.09] | **2.29 [1.78, 2.87]** | 1.92 [1.42, 2.49] | 0.029 / 0.026 / 0.027 |

Paired deltas (WER points, same resamples; Whisper `EnglishTextNormalizer` in brackets):

| delta | AMI-200 | ICSI-200 | LibriSpeech-200 |
|---|---:|---:|---:|
| [70,13] - [70,1] | **-1.41 [-2.48, -0.32]** (-1.34 [-2.50, -0.14]) | **-2.50 [-3.80, -1.32]** (-3.09 [-4.34, -1.93]) | **-0.37 [-0.64, -0.11]** (-0.41 [-0.69, -0.13]) |
| [70,0] - [70,1] | +1.25 [+0.35, +2.25] | +2.58 [+1.60, +3.72] | +0.19 [+0.02, +0.39] |

**Reading.** The 1.04 s lookahead pass is a small, certain gain (1.4 / 2.5 / 0.4 WER points; every CI excludes
zero) for ~10-20 % more encoder time and no second model; it stays far from an offline model on meetings (section 4).
The served [70,1] row equals the `final_asr.json` served row exactly (paired delta 0.000). The live decisions are
untouched: `tests/test_hybrid_asr.py::test_lookahead_heads_bit_identical_and_text_partition` checks every frame,
turn_end, partial and streaming final with and without `--asr-lookahead`.

## 4. Parakeet-TDT 0.6B v3 as the per-turn final (`--final-asr tdt_v3`), English

Same sets and scoring; TDT v3 batch 1, CPU, 2 threads, greedy. The other systems' rows are the `final_asr.json`
hypotheses re-scored on the same resamples (`parakeet` = parakeet-ctc-0.6b; the Whisper rows exist for AMI and
LibriSpeech only; **ICSI-200 was transcribed with the served model only**: the TDT v3 / Whisper ICSI rows were not run
before the machine reboot and the prepared ICSI audio lived in the wiped scratch).

| system | AMI-200 WER | LibriSpeech-200 WER | RTF (AMI, under load) |
|---|---:|---:|---:|
| served [70,1] (streaming, 115M) | 24.43 [22.04, 27.22] | 2.29 [1.78, 2.87] | 0.021 |
| served [70,13] (lookahead pass) | 23.02 [20.54, 25.79] | 1.92 [1.42, 2.49] | 0.025 |
| parakeet-ctc-0.6b | 19.32 [17.37, 21.57] | 1.68 [1.21, 2.16] | 0.049 |
| Whisper small | 21.16 [18.93, 23.71] | 2.42 [1.88, 2.96] | 0.280 |
| Whisper large-v3-turbo | 19.63 [17.45, 21.89] | 1.47 [1.03, 1.96] | 1.469 |
| **Parakeet-TDT 0.6B v3** | **9.72 [8.22, 11.54]** (nofill 9.77, Whisper-norm 9.50) | **2.03 [1.51, 2.55]** | 0.065 |

Card values for reference: LibriSpeech test-clean 1.93, AMI *test* 11.31 (full sets, PnC removed; our AMI numbers are
200 dev segments with our normaliser, not the card's protocol).

Paired deltas, TDT v3 minus the row (WER points): AMI: vs served [70,1] **-14.70 [-17.02, -12.71]**, vs served
[70,13] -13.30 [-15.85, -11.11], vs parakeet-ctc -9.60 [-11.25, -8.02], vs Whisper small -11.43 [-13.61, -9.20],
vs Whisper turbo -9.91 [-11.85, -7.74]. LibriSpeech: vs served [70,1] -0.26 [-0.71, +0.20] (n.s.), vs parakeet-ctc
+0.35 [-0.04, +0.78] (n.s.), vs Whisper turbo +0.56 [+0.11, +1.04].

**Reading.** On meetings the offline TDT pass removes 60 % of the served model's word errors (24.4 -> 9.7 %), at
RTF 0.065 in its own process (the live path's RTF is unchanged); on read speech the served model is already within
noise of it. The lookahead pass recovers only 1 of the 15 points. For a product whose agent reads the transcript, the
hybrid (`--final-asr tdt_v3`, `final_source="offline"` in the adapters) is the change that matters; the lookahead is
the cheap fallback when a second model cannot be run.

## 5. FLEURS-17: TDT v3 per language, and the language it writes in

`hybrid_asr.py run --set fleurs [--n-lang 30]` on the FLEURS-17 test subset of `research/archive/LID.md` (17 languages, 10 of
them among v3's 25). **Sample sizes differ and are stated per row**: the 10 supported languages were transcribed in
full (150 utterances each) before the reboot; the 7 others (pl, uk and the 5 unsupported) with the first **30**
utterances each afterwards (the earlier transcripts lived in the wiped scratch, and the `--n-lang 30` budget is what the
machine rules allowed). WER % (Whisper `BasicTextNormalizer`; `EnglishTextNormalizer` for en), CER % for zh / ja (no
word boundaries), 1000-resample CIs; the card's full-test-set WER for reference. `runs/hybrid_asr.json["fleurs"]`.

| lang | in v3's 25 | n | WER [CI] | CER | card WER | empty hyps |
|---|---|---:|---|---:|---:|---:|
| en | yes | 150 | 7.0 [5.5, 8.8] | 4.4 | 4.85 | 0 |
| es | yes | 150 | 3.9 [2.9, 5.1] | 2.4 | 3.45 | 0 |
| it | yes | 150 | 5.0 [3.5, 6.7] | 2.7 | 3.00 | 0 |
| fr | yes | 150 | 6.3 [5.2, 7.4] | 3.0 | 5.15 | 0 |
| de | yes | 150 | 6.3 [4.9, 7.8] | 3.1 | 5.04 | 0 |
| pt | yes | 150 | 6.5 [5.1, 8.3] | 3.9 | 4.76 | 0 |
| nl | yes | 150 | 8.8 [7.3, 10.4] | 4.0 | 7.48 | 0 |
| ru | yes | 150 | 8.9 [7.2, 10.9] | 3.8 | 5.51 | 0 |
| uk | yes | **30** | 7.4 [4.8, 10.3] | 2.4 | 6.79 | 0 |
| pl | yes | **30** | 10.8 [6.5, 16.4] | 5.7 | 7.31 | 0 |
| tr | no | 30 | 99.5 [97.6, 101.3] | 60.7 | - | 2 |
| hi | no | 30 | 99.9 [99.6, 100.0] | 157.6 | - | 0 |
| fa | no | 30 | 102.7 [100.6, 106.4] | 116.0 | - | 2 |
| he | no | 150 | 111.0 [108.8, 113.3] | 115.7 | - | 9 |
| ar | no | 150 | 131.8 [128.2, 135.2] | 125.5 | - | 0 |
| zh | no | 30 | (239) | **139.3** | - | 13 |
| ja | no | 30 | (447) | **168.4** | - | 0 |

Supported languages land 1-3 WER points above the card (our normaliser keeps more, and n is small); unsupported ones
are transcribed as Latin-script (or Cyrillic) nonsense in some supported language: WER >= 100 %, and for zh 13 of 30
outputs are empty. RTF (batch 1, 2 threads, under load) 0.033-0.048 for every language.

**Language ID from the text (`lid_text`).** v3 has no language token (section 1), so its language decision is read off
the transcript with langid.py (candidates = v3's 25 + our 17; empty = wrong). On the **full utterances** the 10
supported languages are identified **100 % (1260 / 1260**: 150 x 8 + 30 x 2); every unsupported language is 0 %
(tr 1 / 30), i.e. v3 never writes the right language when it does not know it, it writes Maltese / English / Croatian
(the `top_outputs` per language are in the JSON). On the **2 s clips** from the speech onset (`fleurs2s`, n = 30 per
language) supported-language accuracy is **91.7 % (275 / 300)**: en 93, ru 87, es 93, fr 97, de 83, pt 93, it 93, nl
93, pl 100, uk 83; errors are near neighbours (uk -> ru / bg, ru -> uk / bg, de -> en) and 4 empties. AmberNet on the
same utterances, restricted to the same 10 languages (`research/archive/LID.md`): 99.9 % full, 95.1 % at 2 s. So the
transcript's language is a usable LID for the supported set (free with the final), a dedicated LID model is still 3.5
points better at 2 s, and neither the text nor v3 itself says anything reliable for a language outside the 25: an
out-of-set utterance comes back as fluent-looking text in the wrong language, with no flag.
