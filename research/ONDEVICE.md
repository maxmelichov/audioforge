# On-device streaming: runtime and CPU benchmark (2026-09-25)

Target (GAME_CHANGER.md §3): run the **full pipeline** (mel → cache-aware encoder → all heads → decoding) at
**RTF ≤ 0.3 on one M-series CPU core** for a ~100–120M-parameter model.

**Answer: met for the 111M model (M = 512×17) at 160 ms chunks, fp32, one thread.** The median RTF was 0.16
in a quiet window and 0.19 at load average 15. ONNX Runtime int8 came in at about 0.12 RTF measured as CPU
time. These are lower bounds on a contended machine, not clean numbers: see "Measurement conditions".

## What was built

* `audioforge/runtime.py`: `StreamingRuntime` plus `ChunkEncoder`, a **fixed-shape** rewrite of
  `FastConformerEncoder.stream_step` that reuses the model's weights.
  * The first chunk (no mel history), cache slots not yet filled, and a partial last chunk are all handled
    with masks, so input shapes never change.
  * Keys and values sit in a **ring buffer** of `left + chunk` slots. Keys are stored already RoPE-rotated,
    so the order of slots doesn't matter. The torch backends write the new chunk in place; the export graph
    uses gather/where and returns the caches as outputs (the Riva / sherpa-onnx pattern).
  * RoPE sin/cos are computed once per step. The joint's prediction projection is cached per emitted token.
  * Pointwise convs run as `nn.Linear`, so int8 covers them. The depthwise conv runs as `unfold * w`
    (see Profile).
  * Every head runs each chunk: TDT greedy, CTC greedy, VAD and EOU frame heads, and the speaker layer-mix.
    The speaker embedding is pooled on demand.
  * Backends: `eager | trace | compile | onnx | coreml`, plus `quantize="int8"` (dynamic, qnnpack) and
    `dtype=bf16|fp16`.
* `scripts/bench_ondevice.py`: `sweep | one | report | profile`. Each measurement runs in its own subprocess.
  Weights are random. The TDT joint is biased so every backend decodes a fixed token rate with duration 1,
  which is a conservative decode cost.
* `tests/test_runtime.py` (13 tests):
  * The runtime matches `StreamingSession` frame by frame, for `att` in {[16,1],[8,3],[16,0],[6,1]}, with
    whole and partial last chunks.
  * The streamed speaker embedding matches the offline one.
  * Trace matches eager.
  * On the trained `voice_agent_frontend.afm`, the runtime's text equals `StreamingSession`'s.
  * The int8 path runs and leaves the caller's model untouched.
  * `warmup()` is idempotent.
  * The ONNX step matches eager.
* `export_chunk_onnx()` lives in runtime.py. `train.export_onnx` was not modified.

## Model sizes (random init, exact parameter counts)

All models: causal, 80 mels, subsampling 256 channels, conv kernel 9. Heads: TDT (pred/joint 640, V=1024,
durations 0–4) + CTC + VAD + EOU (hidden 128) + speaker (emb 192, `from_layers: all`).

| name | d_model × layers × heads | total params | encoder |
|---|---|---|---|
| S | 256 × 18 × 4 | 33.84M | 28.1M |
| **M** | 512 × 17 × 8 (= NVIDIA FastConformer-L streaming, EOU-120M shape) | **110.74M** | 104.3M |
| L | 1024 × 24 × 8 (≈ Parakeet / Nemotron 0.6B) | 590.43M | 582.5M |
| S512x12 (literal shape from the brief) | 512 × 12 × 8 | 80.50M | 74.0M |
| M768x16 (literal shape from the brief) | 768 × 16 × 8 | 226.82M | 219.6M |

The brief's "512/12 ≈ 30–40M" and "768/16 ≈ 100–120M" do not match their shapes: those shapes give 80M and
227M. For the parameter targets I used 256×18 and 512×17.

## Results: CPU, 1 thread, [70,1] = 160 ms chunks, 60 s streamed audio, median of 3 repeats

Each step is the full pipeline: mel + encoder + TDT/CTC/VAD/EOU + TDT greedy, plus the speaker embedding
amortized every 1 s over the last 5 s.

| size | backend | ms/chunk (reps) | ms per 80 ms audio | **RTF** | p90 ms | CPU ms/chunk | load+prep s | cold 1st step ms | 1st result ms | peak RSS MB | load avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S | eager fp32 | 12.5 (12.4/12.5/13.6) | 6.3 | **0.078** | 13.3 | 12.4 | 0.12 | 13 | 189 | 544 | 29 |
| S | int8 (torch dynamic) | 15.3 (15.1/15.3/16.0) | 7.7 | **0.096** | 16.4 | 15.1 | 0.17 | 33 | 191 | 480 | 23 |
| M | eager fp32 | 25.7 (24.9/25.7/27.4) | 12.8 | **0.161** | 28.2 | 25.5 | 0.31 | 32 | 202 | 963 | 15 |
| M | int8 (torch dynamic) | 36.5 (30.2/36.5/59.9) | 18.2 | **0.228** | 41.3 | 34.7 | 0.68 | 113 | 213 | 1027 | 11 |
| S | eager fp32, dw fix | 11.9 (11.4/11.9/12.8) | 6.0 | **0.074** | 19.8 | 11.2 | 0.29 | 30 | 188 | 464 | 12 |
| S | int8, dw fix | 17.0 (14.7/17.0/17.1) | 8.5 | **0.106** | 24.9 | 15.8 | 0.19 | 36 | 193 | 491 | 12 |
| M | eager fp32, dw fix | 59.5 (42.7/59.5/60.7) | 29.7 | 0.372 (CPU: 0.28) | 97.1 | 45.2 | 0.77 | 63 | 236 | 956 | 14 |
| M | int8, dw fix | 69.6 (67.7/69.6/100.6) | 34.8 | 0.435 (CPU: 0.27) | 112.4 | 43.6 | 0.73 | 169 | 246 | 1089 | 15 |

How to read the columns:
* RTF = ms per chunk / 160.
* "1st result" = chunk buffering (160 ms) + mel window look-ahead (16 ms) + compute for that chunk.
* Peak RSS is for the whole process, and for int8 it includes the transient fp32 copy.

The last two M rows ran while other agents' jobs came back. Their wall-clock time is about 1.4× their CPU time,
so the machine was contended. To remove the load difference, I ran an **interleaved A/B in one process** (20 s
audio, 3 alternations, CPU ms/chunk). Replacing the depthwise `Conv1d` with `unfold`:
* S: 20.9 / 18.9 / 20.2 → 13.5 / 13.3 / 11.6
* M: 47.1 / 48.7 / 56.1 → 42.4 / 29.9 / 30.7

That is a 35–40% cut. At this load level M is at about **30 ms per 160 ms chunk, RTF ≈ 0.19**. Before the fix,
the quietest window gave RTF 0.16. With the fix, a quiet window should come in lower still.

**Other backends: indicative only.** These runs were contended (load 20–30), measured as wall clock, and done
before the ring-buffer and dw changes. The full sweep's saved files were lost when the machine rebooted.
* S [70,1], 1 thread, 60 s, ms/chunk:
  * torch: eager 41.5, trace 33.2, `torch.compile` 28.2, int8 27.7, trace+int8 31.0, bf16 27.0, fp16 18.6
  * **ONNX Runtime: fp32 8.8, int8 5.7**
  * Core ML: CPU_ONLY 10.3, ALL 12.6, CPU_AND_NE fp16 9.1
* S at 4 threads: torch eager 193.5, trace 157.8, int8 152.4 (5–10× slower than 1 thread on the loaded box);
  ORT fp32 12.8, ORT int8 10.1.
* S [70,6] (560 ms chunks), 1 thread: eager 27.6 ms/chunk = RTF 0.049; ORT fp32 16.1; ORT int8 18.1.
* M [70,1], ORT int8, 10 s: wall 86 ms/chunk at load 26, but **CPU 18.4 ms/chunk = RTF 0.115**.

Not measured, per the coordinator's constraints after the overload: L, M768x16, S512x12, and a clean full sweep.

## Does int8 change transcripts? (trained tiny models, `synthetic_dataset('asr', 64, seed=2)`)

| model | fp32 session | runtime fp32 | torch int8 (enc / enc+dec) | bf16 | fp16 | ORT fp32 | ORT int8 |
|---|---|---|---|---|---|---|---|
| voice_agent_frontend.afm | WER 0.00% | 64/64 identical | 64/64, WER 0.00% | 64/64 | 64/64 | 64/64 | 64/64 |
| nemotron_streaming_rnnt.afm | WER 0.50% | 64/64 identical | 63/64, WER 1.49% | 64/64 | 64/64 | 64/64 | 63/64, WER 1.49% |

int8 changes one utterance in 128 ("yes no set" → "yes note"). That is close to neutral, but these are tiny
models on synthetic audio. The check has to be repeated on a real LibriSpeech-trained model before relying on it.

## Where the time goes (torch.profiler + module timers, M [70,1], eager fp32, 1 thread, before the dw fix)

| stage | share |
|---|---|
| encoder step + head projections | **98%** |
| TDT greedy (LSTM 640 + joint) | 1% |
| mel (STFT + filterbank) | 0.2% |
| frame heads, CTC greedy, buffers, Python | 0.2% |

| inside the encoder step | share |
|---|---|
| feed-forward ×2 (Linear, 4× expansion) | 42% |
| **conv module: pw1/pw2 + depthwise Conv1d** | **38%** (the depthwise part is almost all of it) |
| attention qkv + out projections | 9% |
| inline ops (RoPE, SDPA, cache writes, GLU/SiLU, residuals) | 8% |
| pre-encode (subsampling convs) | 3% |
| layer norms, head projections | 2% |

Finding: on CPU, PyTorch's depthwise `Conv1d(groups=D)` over a 10–15-frame input costs **1.1–2.0 ms per call**,
against **52–77 µs** for `unfold`-multiply-sum. That is 20–35× in an interleaved microbenchmark, and 17 calls
per chunk at M. `ChunkEncoder` now uses `unfold` on the torch backends; the export graph keeps the Conv op,
which ORT and Core ML run efficiently. The reference `ConvModule` in `fastconformer.py` has the same cost. The
same substitution there would speed up `StreamingSession`, but it is optional.

After the fix, the step is dominated by Linear layers, as expected: about 2N FLOPs per frame. The remaining
levers are ORT/Core ML kernels (3–4× over torch eager for S) and int8 through ORT. Torch's qnnpack dynamic int8
is **slower** than fp32 on the M5 at 1 thread: it quantizes activations every call, while fp32 GEMM gets
Accelerate/SME.

## Export blockers (exact errors) and fixes

1. **`stream_step` as-is.** It does export with the TorchScript exporter, and steady-state steps match (9.5e-7),
   but the state has variable shape: no mel cache on the first chunk, and a KV cache that grows for the first
   `left/cs` steps. The graph is only valid for the traced cache lengths. Feeding the first step fails with
   `[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: onnx::Concat_1`. `StreamState`
   is also not an exportable input. **Fix:** `ChunkEncoder`, with masks and a fixed ring.
2. **dynamo exporter.** `torch.onnx.export(..., dynamo=True)` fails with `ModuleNotFoundError: No module named
   'onnxscript'` (not installed). The legacy exporter (`dynamo=False`) works for `ChunkEncoder`.
3. **Core ML** (coremltools 9.0: "Torch version 2.14.0 has not been tested… 2.7.0 is the most recent").
   * `ValueError: Rank-0 (input None) is unsupported`. Fix: scalar inputs passed as shape (1,).
   * `TypeError: only 0-dimensional arrays can be converted to Python scalars` while converting an `int` op.
     Fix: shape-dependent `torch.arange(x.shape[2])` replaced by static buffers.
   * After both fixes it converts and matches eager (max VAD diff 1.2e-7, identical tokens) on CPU_ONLY and ALL.
   * The fp16 CPU_AND_NE run worked in a smoke test and failed once in the sweep with no traceback (likely
     memory pressure). Unresolved.
4. **ORT, 4 threads.** One process aborted at exit with `libc++abi: terminating due to uncaught exception of type
   std::__1::system_error: recursive_mutex lock failed: Invalid argument`. This happens during teardown; the
   worker now calls `os._exit` after printing its result.
5. **torch int8.** `quantize_dynamic` failed with `Didn't find engine for operation quantized::linear_prepack
   NoQEngine` until the engine was set to `qnnpack`. It is also deprecated (the warning points to torchao).

## Recommendations

* **Target M (512×17, 111M) at [70,1] / 160 ms.** It fits the one-core budget at fp32. Deploy through the
  exported chunk graph on **ONNX Runtime** (int8 weights: about 0.12 RTF measured as CPU time) or **Core ML**.
  Keep torch eager as the reference path.
* Don't use torch dynamic int8 for speed on Apple CPUs. Use ORT int8 (or Core ML fp16/int8 palettization) if
  you need headroom.
* 227M (768×16) is about 2× M. It would need ORT int8 to stay under 0.3 on one core. The 0.6B model is not a
  one-core model.
* A larger chunk ([70,6], 560 ms) cuts per-audio cost 3–4×. Use it for latency-tolerant modes; turn-taking
  needs [70,1].
* Algorithmic latency to the first result is about 176 ms plus compute (≈190–235 ms total at 160 ms chunks).
* Next step: rerun `python scripts/bench_ondevice.py sweep` on an **idle** machine. The script records CPU time
  and load average per run so contention is visible.

## Measurement conditions

Apple M5 (10 cores, 24 GB), torch 2.14, onnxruntime 1.30, coremltools 9.0, CPU only,
`torch.set_num_threads(1)`, one benchmark process at a time.

The machine was **shared and at times badly overloaded**: other agents were training on MPS, swap was full,
and load averages reached 20–190. It eventually rebooted, which wiped the first full sweep. The table above
comes from runs at load 11–29. Under contention, macOS can also move a single thread onto an efficiency core.
Across time windows the same configuration varied by up to about 2× (M eager fp32: 25.7 vs 45 CPU ms/chunk),
so treat every number as a lower bound on what a dedicated core can do.

Commands:
* `python scripts/bench_ondevice.py one --size M --att 70,1 --backend eager --threads 1 --seconds 60 --repeats 3`
* `python scripts/bench_ondevice.py profile --size M --seconds 15`
