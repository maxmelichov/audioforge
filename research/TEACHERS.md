# NVIDIA teachers on this Mac

*Measured 2026-09-25 on an Apple M5 (24 GB), torch 2.14, transformers 5.17. No NeMo.*

We use NVIDIA's released ASR checkpoints for two jobs: as **teachers** that pseudo-label unlabeled
audio (the Granary pattern: label a lot of audio with a strong model, then distill it into a
student), and as a **real baseline** for our students. Code: `audioforge/teachers.py`
(`Teacher`, `pseudo_label`) and `scripts/benchmark_teachers.py`.

## What runs, and how

All four target models load through `transformers` and run **fully on MPS in float32**. The
benchmark was run with `PYTORCH_ENABLE_MPS_FALLBACK` unset, so there was no CPU fallback.

| Model | transformers class | Decoder | Runs? |
|---|---|---|---|
| `nvidia/parakeet-ctc-0.6b` | `ParakeetForCTC` (`AutoModelForCTC`) | CTC | yes, MPS |
| `nvidia/parakeet-ctc-1.1b` | `ParakeetForCTC` | CTC | yes, MPS |
| `nvidia/parakeet-tdt-0.6b-v3` | `ParakeetForTDT` (`AutoModelForTDT`) | TDT | yes, MPS |
| `nvidia/nemotron-speech-streaming-en-0.6b` | `NemotronAsrStreamingForRNNT` (`AutoModelForRNNT`) | RNNT | yes, MPS (offline mode) |

Two things needed working around:

1. **librosa.** In transformers 5.17 the Parakeet and Nemotron feature extractors are gated on
   `librosa`. Without it, `AutoProcessor.from_pretrained` fails with
   `ImportError: ParakeetFeatureExtractor requires the librosa library`. librosa is used for one
   thing only, `librosa.filters.mel` (the Slaney mel filterbank). Installing it would also pull in
   numba, llvmlite, scipy and scikit-learn. Instead, `teachers._patch_librosa()` puts in a numpy
   copy of that one function (it matches transformers' own float64 filterbank to within 3e-9) and
   puts the real feature-extractor classes back in place of transformers' dummy classes. After
   that, both `AutoProcessor` and `pipeline("automatic-speech-recognition", ...)` resolve normally.
   If librosa is installed, the patch does nothing.
2. **dtype.** With `dtype="auto"` the checkpoints load in bf16 while the processor produces fp32
   features, which fails with `RuntimeError: Input type (float) and bias type (c10::BFloat16)
   should be the same`. `Teacher` loads with an explicit dtype (default float32) and casts the
   inputs to match.

The Nemotron model runs in the offline `generate` mode, which uses the default lookahead of 13
frames (1.04 s; with the current frame, a 14-frame chunk of 1.12 s, att_context_size [70,13]). That is
the configuration the card's LibriSpeech number (2.32%, 1.12 s chunk) refers to. The
chunked streaming API (`processor.set_num_lookahead_tokens`, `TextIteratorStreamer`) is not wrapped
here.

## Benchmark

LibriSpeech test-clean, first 200 utterances by sorted id (`1089-134686-0000` to `1221-135767-0012`,
30.3 min, 4634 words, only 4 of test-clean's 40 speakers). References are lowercased. Both sides go through `normalize_text`: lowercase, hyphens
become spaces, punctuation is dropped, in-word apostrophes are kept. Decoding is greedy with batch 8,
and utterances are length-sorted into batches.
RTFx = audio seconds / transcription wall seconds, measured after one warm-up batch with the device
synchronized. Peak memory is the highest MPS driver allocation during transcription, weights
included. Each model runs in its own process.

```
.venv/bin/python scripts/benchmark_teachers.py --student runs/parakeet_tdt_ctc.afm --save-hyps <dir> --json <file>
```

| model | kind | decoder | params | device | dtype | WER % | RTFx | peak mem GB | license |
|---|---|---|---:|---|---|---:|---:|---:|---|
| parakeet-ctc-0.6b | teacher | ctc | 609M | mps | float32 | 1.73 | 135.0 | 4.10 | cc-by-4.0 |
| parakeet-ctc-1.1b | teacher | ctc | 1063M | mps | float32 | 1.64 | 62.8 | 6.13 | cc-by-4.0 |
| parakeet-tdt-0.6b-v3 | teacher | tdt | 627M | mps | float32 | 2.01 | 57.9 | 4.99 | cc-by-4.0 |
| nemotron-speech-streaming-en-0.6b | teacher | rnnt | 618M | mps | float32 | 2.03 | 34.1 | 8.23 | nvidia-open-model-license |
| parakeet_tdt_ctc.afm | student | tdt | 2M | mps | float32 | 100.09 | 121.9 | 1.14 | ours |

- **Sanity checks.** The WERs agree with `jiwer` exactly. They are close to each card's full
  test-clean number (2620 utterances): 1.87 / 1.83 / 1.93 / 2.32. Nemotron's 2.32 is its 1.12 s
  chunk setting ([70,13]); smaller chunks score 2.46 / 2.56 / 2.80. The cards use the Whisper English
  normalizer (Open ASR Leaderboard), which maps British to American spellings, and our normalize_text
  does not, so the comparison is approximate. At batch 1, parakeet-ctc-0.6b gives the same WER
  (1.73%) and identical transcripts on all 200 utterances, at RTFx 76 on MPS.
- **Significance.** The 0.6b-vs-1.1b CTC gap is 80 vs 76 errors and is not significant on this
  subset: in a paired bootstrap 1.1b wins only 68% of utterance resamples (95% CI of the difference
  -0.20 to +0.40 pp) and 63% of speaker resamples; 0.6b is better on speaker 1089. The 1.73/1.64
  values were reproduced exactly on CPU; the RTFx and memory columns are MPS-only measurements from
  a single run.
- **Why TDT-v3 scores worse here.** It writes British spellings where LibriSpeech uses American ones
  (colour/color, grey/gray, honour/honor, counseled/counselled, favourite, jeweller's), and it spells
  the name as daedalus instead of dedalus. These are 16 of its 93 errors (about 17%). Mapping them
  back gives 1.66% WER, level with parakeet-ctc-1.1b. Most of the remaining errors are compound
  words split or merged (woodbegirt, horseplay, watermill) and misheard rare words or names. The
  model is multilingual and writes punctuation and casing. TDT-v3 wrote no digits on this subset.
- **The student row only exercises `--student`.** The existing `runs/*.afm` models were trained on
  the synthetic tone language, so ~100% WER is expected. Students trained on LibriSpeech can be
  compared directly with `--teachers none --student runs/<model>.afm`, which applies the same
  normalization to the same 200 utterances.
- Model load (from the HF cache) takes 7–11 s. RTFx does not include it.

## Timestamps

`Teacher.transcribe(audios, timestamps=True)` returns
`{"text", "words": [{"word","start","end"}], "tokens": [...]}` in seconds, at 80 ms resolution.

| Model | Source | Quality |
|---|---|---|
| parakeet-tdt-0.6b-v3 | TDT durations → `processor.decode(..., durations=)`, the same post-processing NeMo uses | good: token start plus predicted duration |
| parakeet-ctc-0.6b / 1.1b | computed in `teachers.py` from runs of the frame-level CTC argmax (transformers exposes no CTC timestamps) | good for word starts. CTC spikes are short, so word ends are tight or early |
| nemotron-speech-streaming-en-0.6b | RNNT emission frames (1-frame spans) | emission times, not alignments. On `1089-134686-0001` its word starts are 0.2–0.9 s later than TDT/CTC. Do not use them for segmentation |

Char-level timestamps are not produced. Tokens are BPE pieces, so a character-level alignment would
need forced alignment.

## Licenses, and what they mean for pseudo-labels

| Teacher | License (catalog + card) |
|---|---|
| parakeet-ctc-0.6b, parakeet-ctc-1.1b, parakeet-tdt-0.6b-v3 | **CC-BY-4.0** |
| nemotron-speech-streaming-en-0.6b | **NVIDIA Open Model License** (`license: other`) |

- **CC-BY-4.0** allows any use, including commercial use and derivatives, as long as NVIDIA is
  credited, the license is linked, and changes are indicated. The card places no restriction on
  outputs. Transcripts of your own audio are best treated as model outputs, and a student
  distilled from them is at most an adaptation. The safe practice is to keep attribution, e.g.
  "pseudo-labels generated with NVIDIA parakeet-tdt-0.6b-v3 (CC-BY-4.0)", in the dataset and model
  card. This is the cleanest choice for labels you might redistribute.
- **NVIDIA Open Model License** (the version dated Oct 24, 2025 on nvidia.com): the model is
  commercially usable; "NVIDIA claims no ownership rights in outputs. You are responsible for
  outputs"; derivative models are yours, but when you distribute them you must include "Licensed by
  NVIDIA Corporation under the NVIDIA Open Model License"; your rights terminate if you circumvent
  its guardrails. It does not forbid training on outputs. Using it as a teacher is fine, but it
  adds a notice obligation and more restrictive terms than CC-BY. Prefer the CC-BY teachers unless
  you need its streaming behavior.
- The license of the **audio** you label still applies. For example, LibriSpeech is CC-BY-4.0. None
  of this is legal advice.

## Recipe: pseudo-label a folder of audio

```bash
# 1. (optional) check the teacher on a few files
.venv/bin/python -m audioforge.teachers transcribe a.flac b.wav --teacher parakeet-ctc-1.1b --timestamps

# 2. label everything under a folder (wav/flac/ogg/mp3/opus, any sample rate, mono-mixed and resampled to 16 kHz)
.venv/bin/python -m audioforge.teachers label /path/to/audio data/pseudo/train.jsonl \
    --teacher parakeet-ctc-1.1b --wav-dir data/pseudo/wav --normalize --batch-size 8
#    -> {"audio_filepath": "data/pseudo/wav/x.wav", "duration": 7.1, "text": "...", "teacher": "parakeet-ctc-1.1b"}
#    A JSONL manifest also works as input. Its existing "text" is kept as "ref_text", so you can measure label WER.

# 3. train a student on it (the manifest is what audioforge.data.read_manifest reads)
#    recipe: manifest: {train: data/pseudo/train.jsonl, val: ...}, tokenizer: {kind: spm, vocab_size: 1024}
```

In Python:
`pseudo_label("in.jsonl" | "folder/", "out.jsonl", Teacher.from_name("parakeet-ctc-1.1b"), wav_dir=..., normalize=True, timestamps=False)`.

Choosing a teacher and settings:
- **parakeet-ctc-1.1b** has the lowest WER here (not significantly below 0.6b, see above) and writes plain lowercase text, which is what our spm
  students want. It labels about 1 hour of audio per minute on MPS.
- **parakeet-ctc-0.6b** labels about 2 hours per minute, at +0.09 WER (within noise on 200 utterances).
- **parakeet-tdt-0.6b-v3** is the choice for punctuation, casing or timestamps, or for the other
  24 European languages.
- `--wav-dir` is needed for non-WAV sources, because `audioforge.data.load_wav` reads WAV only.
- Cut long recordings into segments of 40 s or less first (a VAD, or the TDT word timestamps).
  These models use full attention, so memory grows quadratically with length. (In transformers the
  parakeet relative positions are computed on the fly and are not capped at
  max_position_embeddings=5000 frames = 400 s; only the Nemotron streaming model raises past 5000
  frames.)
- Granary also filters labels, for example by agreement between two teachers or by
  confidence/length checks. With two teachers, running both and keeping the utterances where their
  normalized outputs agree is a cheap filter.
