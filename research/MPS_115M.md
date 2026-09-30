# The served 115M model on the Mac GPU (MPS)

Date: 2026-09-30. Machine: Apple M5 laptop, torch 2.14.0, fp32 only. Model: `runs/stage1_served_v2.afm` (the served
115M checkpoint, `--mode single`). Script: `scripts/research/mps_115m.py`. Results: `runs/mps_115m.json`. Every run
went through `scripts/dev/gate.sh` (2 threads) with one torch job at a time.

The first brief also asked for nemotron-speech-streaming-en-0.6b. The user dropped that part. This page covers the
115M model only.

## Result

| 115M served | ASR core ms / 160 ms chunk p50 / p95 | ASR core RTF | full engine ms / chunk p50 / p95 | full engine RTF | real-time streams | WER LS-200 | peak memory |
|---|---|---|---|---|---|---|---|
| CPU, 2 threads | 17.6 / 20.0 | 0.113 | 29.8 / 32.1 | 0.191 | **4** (5: p95 180 ms) | 2.29 % | RSS 1124 MB |
| MPS | **10.2 / 11.9** | **0.065** | 28.7 / 31.5 | 0.183 | **5** (p95 152 ms; 6: 184 ms) | 2.29 % | RSS 1124 MB + 524 MB MPS driver |

- **The MPS engine gives the same events as the CPU engine.** On `examples/audio/two_party_call_16s.wav` in single
  mode with the stored voice print, the turn_end, final and voiceprint events are identical (same times, speakers,
  texts and policy). The streamed encoder output differs from CPU by at most 6.5e-6 absolute (1.8e-6 relative; the
  largest value is 1.87). The RNNT token ids are identical. `tests/test_gpu_device.py` checks the same thing with the
  stored print and with the print taken live after agent_end.
- **The MPS decodes are the same as CPU.** All 200 LibriSpeech-200 hypotheses are identical on CPU and MPS: 106
  errors in 4634 words, 2.29 %. That equals the masked-offline number in `runs/hybrid_asr.json` (`served_la1/libri`),
  and all 200 streamed hypotheses match the masked-offline ones.
- **The ASR core alone is 1.7x faster on MPS** (encoder + greedy RNNT: 10.2 vs 17.6 ms per chunk).
- **The full single-mode engine is only 4 % faster on MPS** (28.7 vs 29.8 ms per chunk). On MPS, the turn pass and the
  heads cost more than on CPU (18.5 vs 12.2 ms per chunk). The turn pass is the second, speaker-conditioned encoder
  pass. A cProfile run of the MPS engine puts about 55 % of the time in the turn pass. The heads run tiny batch-1
  steps with a host read after almost every one: the TS-VAD frame `.cpu()`, the turn head's `duration_counters`, the
  LID `.tolist()` and the RNNT `int(argmax)`. Each read waits for the GPU queue to drain. So on MPS, time is spent
  launching kernels and waiting on syncs, not on arithmetic. This explanation comes from the profile; it was not
  isolated op by op.
- **MPS holds one more stream than CPU: 5 vs 4.** The per-stream compute is the same (29 ms). The CPU's p95 grows
  faster with the number of streams: 4 streams give p95 137 ms on CPU and 126 ms on MPS; 5 streams give 180 vs 152 ms.
  At 5 streams MPS is close to the 160 ms limit.

## Part 1: what changed for MPS

This ports the device handling of PR #1 (`gpu-run/5090-2026-09-29`, `--device cuda`) to accept `mps` too. The PR
itself was not merged.

- `audioforge/serve.py`: `Engine.load` keeps `mps` when `torch.backends.mps.is_available()`, and keeps `cuda[:N]`
  when a CUDA GPU is visible (`gpu_available`). On a GPU it turns fast-conv off (fast-conv is a CPU path). On CUDA only
  it also turns TF32 off. Any other device, or a GPU the process cannot see, still falls back to CPU and counts
  `device_fallback`. The TS-VAD head is moved to the model's device.
- `audioforge/server/streams.py`: tensors made on the host are created on the model's device (`_turn_step`: spk_act,
  cols, prim; `run_turn_on_diar`: act).
- `audioforge/tsvad_stream.py`: `.cpu()` before `.numpy()` in `TSVADTrack.feed`. The print and guard logic is
  unchanged.
- `audioforge/enrollment.py`: `ColumnEmbedder` embeds on the speaker head's device and returns host float32.
- `audioforge/lid.py`: the VAD keep mask is created on the head's device.
- `audioforge.load(device=...)`, `audioforge-bench --device`, `bench_serve.py run --device`, and the `--device` help
  text (docs/CONFIGURATION.md was regenerated).

**Every op on the streaming path runs natively on MPS.** Nothing needed a per-op CPU fallback. The runs were made with
`PYTORCH_ENABLE_MPS_FALLBACK` unset, so a missing MPS kernel would have raised an error. None did. That covers the
STFT and mel, the cache-aware relative-position attention, the causal depthwise convolutions, the subsampling, the
RNNT predictor LSTM and joint, the GRU heads (TS-VAD, LID, turn) and the speaker head. Two parts already ran on the
host by design, and they do so on CPU too:
- `heads/turn.duration_counters` computes its int64 cummax on CPU and returns the result on the device.
- The greedy RNNT loop reads one argmax to the host per step.

## Protocol

- **ASR core:** `model.StreamingSession` (mel, cache-aware encoder at att_context [70,1], greedy RNNT). Batch 1,
  160 ms chunks, 2 threads. The CPU run uses fast-conv and `perf.DEFAULT`, as the server does. The MPS run uses
  `perf.DEFAULT` only. Timing: the first 20 LibriSpeech-200 utterances (156.5 s, 988 chunks), one warm-up utterance,
  then the best of 3 passes by RTF. Each chunk is timed around `torch.mps.synchronize()`. WER: all 200 utterances,
  streamed, `normalize_text`.
- **Full engine:** `audioforge.load(device=...)` (single mode) on the bundled 16 s clip with the stored print, fed in
  160 ms blocks. The figure is `Session.chunk_ms`, the server's own per-chunk compute. Warm-up, then the best of 3.
- **Streams:** the `scripts/research/streams_cpu.py` pattern. K single-mode sessions are fed interleaved, one 160 ms
  block at a time, on one thread. Each session gets a different 60 s AMI Mix-Headset window. K streams count as real
  time while the p95 of the summed per-block compute stays under 160 ms. At K = 1 the CPU p95 is 32.7 ms here, against
  44.9 ms in `runs/streams_cpu.json`. The engine got faster since that run (the perf fast paths and the vad_head
  rule), so the CPU now holds 4 streams, not 3.
- **Memory:** peak RSS of the process. On MPS, `torch.mps.driver_allocated_memory()` is reported as well. The
  streams runs show about 2.3 GB RSS because they load whole AMI meetings into memory.
- AMI-200 was not run: it was optional, and the LibriSpeech check already shows identical decodes.

## Recommendation

`--device mps` is correct: it gives the same events and the same decodes as CPU. It is worth using on a Mac when more
streams per process matter (5 vs 4), or to take load off the CPU. It barely lowers the latency of a single stream.
To get more out of the GPU, remove the per-frame host syncs from the heads. Batching streams would help too.
