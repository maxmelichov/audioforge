# The 0.6B cache-aware streaming encoder as a voice-agent front end (nemotron-speech-streaming-en-0.6b)

2026-09-26. Question: does NVIDIA's bigger streaming encoder carry more speaker information (the measured
bottleneck of our turn-taking, BRIEF.md) or a better VAD than the frozen 115M hybrid encoder we serve, and what
does it cost on a laptop CPU? Everything below is measured with `scripts/enc0p6b.py`; results in `runs/enc_0p6b.json`.

## Model, licence, import

| | 0.6B (`nvidia/nemotron-speech-streaming-en-0.6b`) | 115M (`stt_en_fastconformer_hybrid_large_streaming_multi`) |
|---|---|---|
| Licence | **NVIDIA Open Model License** (card; not CC-BY-4.0). Commercial use allowed under its terms; not a permissive CC licence, so it stays a separate row in the catalog | CC-BY-4.0 |
| Encoder | FastConformer 24 x d1024, 8 heads (d_k 128), 128 mels, `use_bias: false`, `xscaling: false`, layer-norm conv, causal dw_striding subsampling, chunked_limited att contexts [70,13] [70,6] [70,1] [70,0] (1.12 / 0.56 / 0.16 / 0.08 s chunks) | 17 x d512, 8 heads, 80 mels, biases, xscaling, same contexts |
| Decoder | RNNT only (target `EncDecRNNTBPEModel`), 2-layer LSTM prediction net (640), joint 640, SentencePiece 1024 + blank; outputs punctuation and capitalization. The config keeps an `aux_ctc` stub with no CTC weights | hybrid RNNT + CTC |
| Params | 618.1M in the checkpoint (618.5M with the 264 zero biases) | 114.8M |
| Checkpoint | 2.47 GB `.nemo` (`data/nemo/`), 653 tensors | 0.46 GB |

Chosen over `nemotron-3.5-asr-streaming-0.6b` (same encoder family plus language-ID prompt conditioning, 40
locales; nothing extra for an English front end) and `multitalker-parakeet-streaming-0.6b-v1` (same base, but a
different interface: learnable **speaker kernels injected at the pre-encode layer**, driven only by per-frame
target-speaker activity from a streaming Sortformer, one encoder instance per speaker, no enrollment. That is the
mechanism our `FastConformerEncoder.speaker_kernels` mirrors; importing it later would map its kernels onto
`speaker_kernel_layers`, with the activity input taken from our Sortformer track).

Importer (`audioforge/nemo_import.py`, the hybrid path, all changes opt-in and covered by
`tests/test_nemo_import.py`): checkpoints above 1 GB are extracted once to `data/nemo/.extracted/` and loaded
memory-mapped (one fp32 copy in RAM instead of two); `pred_rnn_layers` maps onto our `PredictionNet(layers=2)`
(one `nn.LSTM(num_layers=2)`, as NeMo); the `aux_ctc` section becomes a CTC head only when `ctc_decoder.*` tensors
exist; `use_bias: false` fills exactly the missing conformer-layer biases with zeros (exact) and counts them
(`import_info.zero_biases`); anything else unmapped, missing or mis-shaped still raises. Real import: 653 tensors,
651 loaded, 264 zero biases (24 layers x 11), mel filterbank recomputed to 3.7e-9, window to 6e-8, 8.7 s to load.
Saved as `runs/nemo_nemotron_speech_streaming_en_0.6b.afm` (2.19 GB). Tests: a synthetic 0.6B-style archive with
a strict tensor count (`n_loaded == archive - 2`, `zero_biases == 11 x layers`, no CTC head, bit-identical
encoder output, stream_step == offline at [70,13] and [70,1]) and a real-checkpoint count test gated on
`AUDIOFORGE_BIG_TESTS=1` (it loads 2.5 GB).

## ASR check: LibriSpeech test-clean, first 200 utterances (30.3 min), greedy RNNT, CPU fp32

The offline forward with the chunked-limited mask is what cache-aware streaming computes: `stream_step` fed
(R+1)*8 mel frames per call reproduces it to max |diff| 1.2e-6 ([70,13]) / 3.1e-6 ([70,1]) over 3 utterances,
identical frame counts, and `StreamingSession` (raw audio in 160 ms pieces) returns exactly the offline transcript
for all 3 x 2 cases (`runs/enc_0p6b.json: equality`). WER uses `teachers.normalize_text` (lowercase, punctuation
stripped, no number expansion; 0 hypotheses contained digits); the card uses the whisper normalizer on the full
test-clean, so the card column is a different subset and normalization.

| att context | chunk / lookahead | 0.6B WER % | card (full test-clean) | 115M hybrid RNNT (NEMO_IMPORT.md) |
|---|---|---:|---:|---:|
| [70, 1] (our 160 ms setting) | 0.16 s / 0.08 s | **2.20** | 2.56 | 2.29 |
| [70, 13] (card default) | 1.12 s / 1.04 s | **2.03** | 2.32 | 1.92 |

52-55 of 200 utterances differ from the reference; the top differences are spelling normalization
(`daedalus`/`dedalus`, `st`/`saint`, `bede's`/`beads`), not recognition errors. On this clean subset the 0.6B is
within noise of the 115M (its card advantage is on hard sets: AMI 14.7 % at 160 ms).

## Cost on CPU (2 threads)

| | 0.6B | 115M (BASELINES.md / ONDEVICE.md) |
|---|---|---|
| Offline batched encode+decode, RTF | 0.071-0.072 (batch 4, 200 utts) ; 0.083 for one 20 s window | - |
| `StreamingSession`, 160 ms chunks, Python path | **739 ms per 160 ms chunk** (p50 749, p95 1050, max 2123), RTF **4.6** over 60 s | 176 ms per chunk (RTF 1.1) same path; ~30 ms per chunk (RTF ~0.5 whole system) in `audioforge.runtime` |
| Resident memory | 4.85 GB peak (fp32 weights 2.5 GB + mmap pages + activations); 4.65 GB right after load | 3.4 GB for the whole served system |
| Load from .nemo | 4-9 s (memory-mapped) | 1.5 s |

At 160 ms chunks the per-step overhead dominates (2 encoder frames per step through 24 layers with a 70-frame
rel-pos cache, plus greedy RNNT): 4.2x the 115M's per-chunk time in the same Python path, so it is not real time
there. It would need the compiled `audioforge.runtime` path (not measured for this checkpoint; scaling the 115M
number by parameters gives roughly 150-200 ms per chunk, i.e. RTF ~1 on 2 threads) or 560 ms chunks. Batched
offline encoding is cheap (RTF 0.07), which is why feature caching works: the 178 min of AMI/ICSI audio below
were encoded in about 15 min of single-process time.

## Cached features

`data/cache/enc0p6b/<set>/<segment_id>.npz`: `h` = fp16 `(24, T, 1024)`, every block's full frame sequence at
[70,1] (no pooling), plus `vad` labels for the VAD sets; `meta.json` per set. Sets: `spk_ami_200`, `spk_ami_64`,
`spk_icsi_200`, `spk_icsi_64` (the exact `scripts/spk_head.dev_segments` ids), `vad_ami_dev`, `vad_icsi_dev`
(the BASELINES 64 x 20 s diar windows), `vad_ami_train` (the 300-window seeded AMI train cap the 115M probes in
`runs/vad_layers.json` were fitted on; 100 min), and `asr_ami_train` (930 segments, 108 min; see the run log).
Written chunked and resumable (`features --set <name>`, per-item files, time budget per process).

## Untrained per-layer speaker probe (mean-pooled block output, cosine, `spk_head.eer_block` trials)

Within-meeting EER % per block (the number that matters for enrollment inside one conversation):

| set | 0.6B, blocks 1..24 | 115M, blocks 1..17 |
|---|---|---|
| AMI dev n=200 | 35 34 34 33 33 34 34 33 34 **32 32** 33 34 34 36 39 39 42 44 47 47 48 48 52 | 35 **33 33** 34 35 34 35 36 37 43 46 48 49 48 48 50 49 |
| ICSI dev n=200 | 30 28 28 27 28 30 30 25 30 25 **24 24** 26 28 31 34 33 41 46 49 50 50 49 51 | 33 32 30 **29** 30 31 35 34 36 43 46 47 49 48 48 49 49 |
| AMI dev n=64 | 34 **30** 32 30 31 31 32 33 32 33 31 33 34 35 34 39 38 41 44 47 47 48 47 49 | 31 **30 30 30** 31 31 35 34 36 41 43 43 45 44 44 47 46 |
| ICSI dev n=64 | 34 30 28 26 27 29 29 25 30 23 22 **20** 22 25 30 32 27 38 46 51 54 54 52 54 | 36 34 30 **28** 29 31 34 33 35 44 48 48 51 51 50 50 51 |

All-pairs EER % (easier: cross-meeting trials), best block: AMI n=200 18 % (blocks 2-3) vs 115M 19 % (block 2);
ICSI n=200 23 % (blocks 2-4, 8) vs 24 % (block 4). Summary:

| | 0.6B | 115M | TitaNet-L (BASELINES) |
|---|---|---|---|
| AMI n=200 within / all | 32 % (block 10-11) / 18 % | 33 % (block 2) / 19 % | 8.2 % / 6.6 % |
| ICSI n=200 within / all | 24 % (block 11-12) / 23 % | 29 % (block 4) / 24 % | - |
| top block, within | 52 / 51 % (chance) | 49 / 49 % (chance) | |

## Per-layer VAD probe (vad_layers protocol: fit on 300 AMI train windows, BASELINES 64-window scoring)

`mlp64` = Linear(1024,64)-SiLU-Linear(64,1) as the served head; logreg = sklearn C=1; AMI dev F1 / AUC at 0.5,
miss at FPR 0.075 (the Silero operating point); ICSI dev = held-out corpus.

| block | 0.6B logreg AMI F1 / AUC | 0.6B mlp64 AMI F1 / AUC / miss@0.075 | 0.6B mlp64 ICSI F1 / AUC | 115M mlp64 AMI F1 / AUC | 115M mlp64 ICSI F1 / AUC |
|---|---|---|---|---|---|
| 1 | 0.900 / 0.933 | 0.924 / 0.954 / 0.171 | 0.905 / 0.899 | 0.932 / 0.958 | 0.904 / 0.893 |
| 2 | 0.906 / 0.942 | 0.906 / 0.947 / 0.199 | 0.905 / 0.907 | 0.938 / 0.962 | 0.910 / 0.909 |
| 3 | 0.908 / 0.943 | 0.911 / 0.948 / 0.200 | 0.907 / 0.902 | 0.943 / 0.967 | 0.911 / 0.921 |
| 4 | 0.927 / 0.954 | 0.922 / 0.950 / 0.195 | 0.907 / 0.905 | 0.948 / 0.968 | 0.909 / 0.915 |
| 5 | 0.931 / 0.957 | 0.926 / 0.956 / 0.178 | 0.906 / 0.900 | 0.946 / 0.968 | 0.909 / 0.911 |
| 6 | 0.930 / 0.958 | 0.928 / 0.956 / 0.169 | 0.905 / 0.902 | 0.946 / 0.969 | 0.908 / 0.923 |
| 7 | 0.928 / 0.957 | 0.932 / 0.958 / 0.152 | 0.905 / 0.902 | 0.948 / 0.969 | 0.904 / 0.926 |
| 8 | 0.928 / 0.958 | 0.935 / 0.960 / 0.141 | 0.909 / 0.903 | 0.948 / 0.968 | 0.901 / 0.920 |
| 9 | 0.929 / 0.958 | 0.940 / 0.964 / 0.130 | 0.905 / 0.893 | **0.950 / 0.969** | 0.903 / 0.911 |
| 10 | 0.926 / 0.954 | 0.935 / 0.960 / 0.148 | 0.917 / 0.901 | 0.949 / 0.968 | 0.903 / 0.907 |
| 11 | 0.926 / 0.951 | 0.939 / 0.961 / 0.149 | **0.918 / 0.907** | 0.947 / 0.966 | 0.900 / 0.905 |
| 12 | 0.923 / 0.952 | 0.939 / 0.960 / 0.156 | 0.912 / 0.904 | 0.946 / 0.963 | 0.900 / 0.893 |
| 13 | 0.919 / 0.952 | 0.938 / 0.960 / 0.156 | 0.901 / 0.902 | 0.948 / 0.959 | 0.898 / 0.876 |
| 14 | 0.929 / 0.955 | 0.941 / 0.962 / 0.154 | 0.905 / 0.903 | 0.946 / 0.954 | 0.896 / 0.858 |
| 15 | 0.934 / 0.959 | 0.945 / 0.966 / 0.129 | 0.904 / 0.895 | 0.943 / 0.947 | 0.895 / 0.827 |
| 16 | 0.935 / 0.959 | 0.943 / 0.964 / 0.132 | 0.901 / 0.879 | 0.942 / 0.939 | 0.894 / 0.805 |
| 17 | 0.940 / 0.963 | **0.946 / 0.966 / 0.122** | 0.903 / 0.866 | 0.939 / 0.935 | 0.892 / 0.785 |
| 18 | 0.941 / 0.962 | 0.946 / 0.964 / 0.133 | 0.902 / 0.886 | | |
| 19 | 0.941 / 0.961 | 0.946 / 0.960 / 0.145 | 0.900 / 0.882 | | |
| 20 | 0.941 / 0.955 | 0.944 / 0.958 / 0.150 | 0.900 / 0.862 | | |
| 21 | 0.939 / 0.952 | 0.945 / 0.956 / 0.170 | 0.899 / 0.854 | | |
| 22 | 0.935 / 0.943 | 0.941 / 0.949 / 0.197 | 0.895 / 0.828 | | |
| 23 | 0.936 / 0.935 | 0.939 / 0.940 / 0.237 | 0.895 / 0.802 | | |
| 24 | 0.934 / 0.932 | 0.933 / 0.935 / 0.246 | 0.893 / 0.782 | | |

Reference points (BASELINES.md / `runs/vad_layers.json`): served 115M head (trained, layer mix) AMI F1 0.949 /
AUC 0.970 / miss 0.121, ICSI 0.900 / 0.919; Silero v5 0.915; MarbleNet 0.937; pyannote offline 0.966.

## Verdict

- **Speaker information: a little more, in the middle, not enough to matter.** The 0.6B's best untrained block
  is within-meeting 32 % on AMI n=200 (115M: 33 %) and 24 % on ICSI n=200 (115M: 29 %; n=64: 20 % vs 28 %). The
  gain is on ICSI only and sits in blocks 8-13 (a wider plateau than the 115M's blocks 2-4), and both encoders'
  top blocks are at chance (49-52 %). Against TitaNet-L's 8 % within-meeting EER the bigger ASR encoder is still
  a poor speaker representation: ASR training erases identity at the top of both models, and 5x the parameters do
  not put more of it back in the lower blocks. It does not move the identity bottleneck (BRIEF.md); the TitaNet
  enrollment track remains the lever.
- **VAD: equal.** Best probe block 17: AMI F1 0.946 / AUC 0.966 / miss 0.122 vs the 115M's block 9 0.940 / 0.960
  / 0.138, and on held-out ICSI 0.918 vs 0.911 (block 11 vs 3). Within a point of our served head (0.949 / 0.970)
  and of what a 64-hidden probe on the 115M already gives. The 0.6B's VAD-best block is deep (17 of 24), so a
  "gated early exit" design (VAD_LAYERS question) is worse here, not better.
- **ASR: equal on clean speech** (2.20 / 2.03 % vs 2.29 / 1.92 % at 160 ms / 1.1 s), card gains are on hard sets.
- **Cost: 5x.** 4.85 GB resident fp32 vs 3.4 GB for our whole served system, 739 ms per 160 ms chunk in the
  Python streaming path (115M: 176 ms), i.e. not real time on 2 threads without the compiled runtime, and even
  there roughly the whole CPU budget for the encoder alone.
- **How to use it: as a cached offline feature extractor, not a served encoder.** Batched encoding runs at RTF
  0.07, so caching full per-layer sequences (fp16, 24 x T x 1024) is cheap, and any head experiment (speaker,
  VAD, turn) can read them without ever loading the 2.5 GB model in the loop. That is the right way to keep
  probing it (e.g. a trained speaker head on blocks 8-13, or distillation targets); serving it on a laptop CPU at
  160 ms chunks is not, unless the compiled runtime is measured at RTF < 0.5 with headroom for the diarizer.
- Not done: no head was trained on the cached features (untrained probes only); the compiled-runtime RTF was not
  measured for this checkpoint; card-style whisper normalization was not applied.

Reproduce: `scripts/enc0p6b.py {import,wer --att 70,1|70,13,equality,rtf,features --set <set>,spk_probe,vad_probe,report}`
(2 threads, each stage its own process, resumable). Files: `runs/enc_0p6b.json`, `runs/enc0p6b/hyps_ls200_*.jsonl`,
`data/cache/enc0p6b/`.
