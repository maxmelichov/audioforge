# Performance: what every RTF means, what the served model costs, and what was made faster (2026-09-28)

Everything here is CPU, Apple M5 laptop (10 cores, 24 GB), 2 torch threads unless a row says otherwise, the served
models `runs/stage1_served.afm` (115M FastConformer, [70, 1] attention context = 160 ms chunks, RNNT + VAD + turn +
speaker heads) and `runs/nemo_nemotron3_diar.afm` / `runs/nemo_sortformer_v2.afm`. The measurement tool is
`scripts/research/bench_serve.py` (in-process `serve.Engine` / `serve.Session`, the exact code the WebSocket server runs), the
numbers are in `runs/perf.json`, and the load average and swap at measurement time are stored next to every number
because this machine is shared: read a wall-clock figure together with its load column. Production is GPU; the design
target is "lightweight real-time first", so the 2-thread CPU figures are the ones the product is judged by.

**RTF** everywhere in this repository is *compute time / audio time*: 0.5 means one second of audio costs half a
second of compute. It is not one number for a model. The same 115M encoder measures 0.02 when it decodes whole
utterances in batches and 0.15 when it runs 160 ms chunk by 160 ms chunk inside the live server, and both are
correct for what they measure. Section 1 labels every figure the repository states; section 2 is the per-component
measurement of the served path; section 3 is the optimisation pass, with outputs pinned identical.

## 1. Audit: the figures the repository states, and what each one measures

| figure | what it measures | where it comes from | conditions | quote it for |
|---|---|---|---|---|
| **0.015 / 0.022** | ASR *only*, **offline**: `SpeechModel.transcribe` over whole utterances, **batch 4**, one masked chunked-limited forward per batch (the same [70, 1] mask the streaming path uses, so the WER is the streaming WER), greedy RNNT over the whole sequence, `torch.inference_mode`; decode wall time / audio time. LibriSpeech-200 / AMI-200. No diarizer, no heads, no per-chunk work | `runs/final_asr.json` `results/served/{libri,ami}/rtf_cpu2` (`scripts/research/final_asr.py`) | 2 threads, load < 6 | the *model's* raw cost against the other ASR models in the same table (Whisper, Parakeet), never the server |
| **0.065** | Parakeet-TDT 0.6B v3, offline, **batch 1**, one finished turn at a time (`serve --final-asr tdt_v3` runs it exactly so, in a worker process) | `runs/hybrid_asr.json` `english/results/tdt_v3/ami/rtf_logged` | 2 threads, machine under load | the extra cost of the per-turn transcript; it is off the streaming path (worker process), so it adds CPU, not chunk latency |
| **0.16** (the "earlier" figure) | the *on-device runtime* study: `audioforge/runtime.py`'s fixed-shape `ChunkEncoder` over a **111M model with random weights** (M = 512 x 17, the served encoder's size), mel + encoder + TDT / CTC greedy + VAD / EOU heads per 160 ms chunk, **1 thread**, 60 s of audio, no diarizer, no RNNT-per-frame decode, no turn pass, no server | `research/archive/ONDEVICE.md` (median of 3 repeats, quiet window; 0.19 at load 15; the saved JSON was lost in a reboot) | 1 thread, fp32 | "can the encoder + heads run in real time on one core" (yes); it is neither the server nor the served checkpoint |
| **0.15 + 0.15** | inside the live server: the **ASR pass** (mel, encoder chunk step, RNNT decode frame by frame, VAD head) and the **turn pass** (a second, speaker-conditioned encoder pass over the same mel chunks plus the turn head, `--turn-input diar`), each as the server's own per-component time sums / audio time | `runs/e2e_final.json` `components_live/*/server_asr`, `server_turn` (`rtf_wall`); `research/E2E_FINAL.md` section 7 | 2 threads, 1x paced sessions, 3489 s of audio per framework, load < 6 | the streaming core's live cost; the two passes together are the "0.30 for the 115M model live" |
| **0.50 / 0.30** | the diarizer alone, live: Sortformer v2 at the 0.32 s card setting / Nemotron-3-Diarization 100M (max pooling, 4 columns, left context 1) | `runs/e2e_final.json` `server_diar` | same | which diarizer to ship (Nemotron-3) |
| **0.64 / 0.80** | the **whole server**, live, at 1x, per session (session wall compute / audio): ASR pass + turn pass + diarizer + policy; Nemotron-3 / Sortformer v2. Process CPU is 0.71 / 1.02 CPU-s per audio-s (both threads counted) | `runs/e2e_final.json` `server_total` (0.635-0.650 / 0.795-0.802 across Pipecat / LiveKit and C / D) | same | **the product number**: what one 2-thread CPU session costs; README, FINAL_REPORT rows 14 and section 8 |
| **0.85** | `audioforge-bench --diarizer nemotron3 --repeat 2` on the bundled 10 s clip, as fast as possible, best of 2 | README quick start, 2026-09-28 | 2 threads, **machine shared with other jobs** (load 4-7) | nothing beyond "it ran"; same code path as 0.64, slower only through load |
| **0.9957** | `stats.rtf` of the quickstart client's session (10 s clip at 1x) on the same loaded machine | README quick start | same loaded machine | same: a loaded-machine observation, labelled as such in the README |
| 0.37-0.52 / 0.34-0.42 | the whole server, older stack (`stage1_heads_pretrained`, turn head on the session's own frames = no second encoder pass, Sortformer v2 at the 1.04 s setting), 2 / 4 threads, 1x, load 3-6 | README "Live streaming server", `research/archive/INTEGRATION.md` section 4, `docs/CONFIGURATION.md` section 3 | 2 / 4 threads | history; the served stack since 2026-09-26 runs the second pass and the 0.32 s diarizer setting, which is where the 0.80 comes from (0.15 turn pass + 0.1 for the 3x more frequent diarizer steps + Sortformer's larger per-step cost) |
| 0.49 / 0.40 | the older stack at the 0.32 s / 1.04 s diarizer settings, 2 AMI windows | README diarizer-setting table | 2 threads, load 1.5-2.4 | the cost of the 0.32 s setting (+0.1) |
| 0.001 / 0.02-0.03 | the LID head / AmberNet | `docs/CONFIGURATION.md` section 9 | | optional extras |
| 4.6 | the NVIDIA 0.6B streaming encoder, offline batch decode | FINAL_REPORT section 10 | | why the 0.6B encoder is not in the stack |

Two more numbers that look like the same thing and are not:

* **"first partial"**: `stats.first_partial_ms` (45-200 ms in the README) is the *server lag*: the time from the
  arrival of the audio block that produced the first non-empty partial to the partial being sent. The 0.8-1.2 s in
  FINAL_REPORT is the time from *speech onset* to the first partial, which is dominated by the RNNT emitting late
  (a word is emitted a few frames after it ends) and by the 160 ms chunk plus 80 ms lookahead. Section 2 reports both.
* **peak RSS**: `stats.peak_rss_mb` is the *process lifetime* peak (`ru_maxrss`), so it includes the transient of
  loading and converting the checkpoints; the steady-state RSS after the load is lower (section 2 gives both).

### Why the same model shows 0.02 offline and 0.15 live

The offline decode and the ASR pass of the live server run the same weights and the same [70, 1] attention mask, and
produce the same transcript. The 7x cost difference is all shape and bookkeeping:

1. **Chunk size.** Offline, each conformer layer sees the whole utterance at once (hundreds of frames) and each
   `nn.Linear` is one large GEMM that runs at full BLAS efficiency on 2 threads. Live, a 160 ms chunk is **2 encoder
   frames**: every linear layer is a 2-row GEMM (17 layers x 6 linears x 2 rows), which the BLAS executes at a small
   fraction of peak; the fixed cost per kernel launch, not the arithmetic, sets the time. Section 2 measures the
   encoder layers at about 45 % of the ASR pass while they hold about 95 % of the arithmetic.
2. **Two encoder passes.** The served turn head is speaker-kernel conditioned, so the server runs the encoder a
   second time over the same mel chunks with the diarizer's primary-speaker activity (`--turn-input diar`): the
   0.15 "turn pass" is a full second encoder at chunk shape. Offline there is one pass and no turn head.
3. **Cache-aware attention at chunk shape.** Each streaming step re-runs the convolutional subsampling on a
   16 + 16-mel window (its 15-frame receptive field) to produce 2 frames, concatenates 70 cached keys / values per
   layer, and recomputed the relative-position projection `linear_pos(pe(72))`, a 143-row GEMM per layer per step,
   larger than the rest of the step's work (fixed in section 3, bit-identical).
4. **Per-frame Python.** The RNNT is decoded frame by frame in Python (predictor + joint per frame, so the turn
   head's text state at frame v uses the tokens of frames <= v), the VAD and turn heads step once per frame, and
   the session builds one JSON frame message per 80 ms. Offline the greedy decode is one batched loop over the
   sequence with no per-frame protocol work.
5. **Threads.** With 2-row GEMMs the second thread is nearly idle: the live process CPU time (0.71 CPU-s per audio-s
   for the Nemotron server) is only 1.1x its wall time, so the 2 threads give about 1.1x, not 2x, and the "2 threads"
   of the offline figure and of the live figure buy very different speedups.
6. **Machine load.** Every live figure was taken on a laptop shared with other jobs; `runs/e2e_final.json` at load
   < 6, the README quick start at load 4-7 (0.85 / 1.0 for the same configuration that measures 0.64 at lower load).
   CPU time per audio second (`cpu_rtf` in `runs/perf.json`) is the load-robust figure; wall RTF is what a user sees.

### Which number to quote

* "How fast is the model?" against other ASR models: **0.015 / 0.022** (offline, batch 4, 2 threads), the same
  protocol as the Whisper / Parakeet rows next to it.
* "What does one live session cost?": **0.64 with Nemotron-3, 0.80 with Sortformer v2** at 2 threads before this
  pass, and the section 3 figure after it; always with the thread count and the diarizer named.
* "Does it keep up?": chunk-time p95 against the 160 ms chunk and the backlog, not RTF alone (section 2).
* "How many sessions per core?": the concurrency table of section 2, not 1 / RTF (the second thread is mostly idle,
  so sessions can share it).
* Never quote the 0.16 on-device figure for the server (random weights, no diarizer, 1 thread) nor the quick start's
  0.85 / 1.0 for anything but "it ran on a loaded laptop".

### Doc lines changed by this audit

* README scorecard rows 1-2 (and FINAL_REPORT rows 1-2, table 1.1) said "2.29 % WER streaming, RTF 0.015": the WER
  is the streaming-mask WER but the RTF is the offline batch-4 decode; the README row now says so.
* README "On-device" row said "RTF 0.16 on one CPU core" without saying it was a random-weight runtime study with no
  diarizer; it now does.
* README's top summary said "RTF 0.63 vs 0.80" and the options list "0.64 vs 0.80" for the same `runs/e2e_final.json`
  rows (0.635-0.650); both now say 0.64.
* README quick start now links here for what its 0.85 / 1.0 mean.

## 2. Measured: the served path per component

`scripts/research/bench_serve.py components` streams the 5 AMI windows of E2E_FINAL (`e2e5`, 88 s) and one 120 s window
(`long`: after ~30 s the diarizer's speaker cache and FIFO are full, which is the steady state of a real session)
through `serve.Session` in 20 ms blocks with wall-clock timers around each leaf; `runs/perf.json`
`components_none` / `components_default`. Nemotron-3-Diarization (max pooling, 4 columns, left context 1), turn
policy `hybrid_dyn` so that Silero is on the path. The two columns per set are **before** (`--perf none`, the
2026-09-27 code) and **after** (`--perf default`, section 3), in ms per 160 ms of audio; RTF = ms / 160. Load
average at the run: 6.6-9.8 for the "before" columns (another agent's test suite was running), 3.5-4.1 for the
"after" columns, swap 14.3 / 7.3 GB. The A/B of section 3 is the fair before/after comparison; this table is for
the *shape* of the cost.

| component (per 160 ms of audio) | calls per 160 ms | e2e5 before | e2e5 after | long before | long after | what it is |
|---|---|---|---|---|---|---|
| ASR pass: mel | 8 | 0.7 | 0.3 | 0.6 | 0.3 | incremental log-mel, one `torch.stft` per 20 ms block |
| ASR pass: subsampling | 2 -> 1 | 23.8 | 4.6 | 23.3 | 4.6 | the 3 depthwise-separable stride-2 convs over the 16 + 16 mel window; run once for both passes after `share_subsample` |
| ASR pass: 17 conformer layers | 1 | 59.0 | 16.9 | 59.0 | 14.7 | 2-frame chunk, 70-frame KV cache, relative-position attention, fast-conv convolutions |
| ASR pass: RNNT frame decode | 2 | 0.8 | 0.3 | 1.3 | 0.3 | predictor + joint per frame in Python |
| ASR pass: VAD head | 1 | 0.06 | 0.02 | 0.06 | 0.02 | layer-mix of the 17 hidden outputs + MLP |
| **ASR pass total** (`feed_frames`) | | **72.8** | **22.4** | **74.0** | **20.2** | = E2E_FINAL's "ASR pass" (0.15 there at lower load; 0.14 after) |
| turn pass: conditioned encoder (17 layers + kernels) | 1 | 77.6 | 16.5 | 76.2 | 15.1 | second encoder stream over the same mel with the primary's activity; after `share_subsample` it has no subsampling of its own |
| turn pass: turn head step | 2 | 1.3 | 0.6 | 1.2 | 0.5 | GRU step + text features per frame |
| **turn pass total** | | **79.0** | **17.2** | **77.5** | **15.7** | = E2E_FINAL's "turn pass" (0.15 there; 0.11 after) |
| diarizer: window encoder (0.5M) + mel | 0.67 | 0.6 | 0.2 | 0.4 | 0.1 | Nemotron-3's small FastConformer over [1 left + 3 chunk + 1 right] frames every 240 ms |
| diarizer: Sortformer head (98.7M transformer) | 0.67 | 85.1 | 31.5 | 141.4 | 61.5 | the transformer over [speaker cache 188, FIFO 188, chunk 3, right 1] rows, every 240 ms; the rows fill up over the first 30 s, hence e2e5 vs long |
| **diarizer total** | | **86.0** | **31.8** | **142.1** | **61.8** | = E2E_FINAL's 0.30 for Nemotron-3 on the short clips |
| Silero VAD (hybrid_dyn) | 8 | 1.3 | 0.6 | 1.3 | 0.6 | ONNX, per 32 ms window |
| Python and everything else (policies, JSON messages, rings) | | 1.9 | 0.8 | 2.2 | 1.0 | session total minus the leaves |
| **session total** | | **240.1** (RTF 1.50) | **72.4** (RTF 0.45) | **296.2** (RTF 1.85) | **99.0** (RTF 0.62) | |
| chunk time p50 / p95 / max ms | | 238 / 456 / 1340 | 79 / 125 / 336 | 311 / 606 / 1606 | 119 / 180 / 339 | `stats.chunk_ms_*` |
| first partial: server lag ms (5 clips) | | 54-74 | 20-24 | 103 | 19 | `stats.first_partial_ms`: block arrival -> partial sent |
| first partial after speech onset ms | | 640-1440 | same | – | – | first `partial` with text minus the first frame with VAD > 0.5 (identical decisions) |
| speaker head, one 5 s embedding (63 frames, block-4 tap) ms | | 0.59 | 0.20 | | | not on the per-chunk path: it embeds on demand (enrollment); 0.2-0.6 ms per call |
| RSS MB: after load / end of run / process peak | | 765 / 963 / 1026 | 1451 / 1113 / 1554 | | | see the RSS note below |

Reading:

* **Before**, the cost was evenly split three ways (ASR pass 73, turn pass 79, diarizer 86-142 ms per 160 ms), and
  inside the two encoder passes almost all of it was the 17 conformer layers (59 + 65 ms) and the subsampling
  (24 ms, run twice). The layers' arithmetic at 2 frames is tiny; the time is the fixed cost of ~100 small kernels
  per layer per step and of the 143-row `linear_pos` GEMM, which section 3 removes.
* **After**, the two encoder passes together cost 38 ms per 160 ms (RTF 0.24) and the diarizer's transformer head
  is the largest item: 32 ms on short clips, 62 ms in the steady state (RTF 0.20-0.38). That head is 98.7M
  parameters over up to 380 rows every 240 ms, i.e. ~75 GFLOP per step: it is arithmetic-bound (about 0.8 TFLOP/s
  on 2 threads here), so no exact CPU trick shortens it; only a smaller speaker cache / FIFO (a model setting, which
  changes its outputs) or lower precision would.
* Python is not the problem: policies, messages and bookkeeping are 0.8-2.2 ms per 160 ms (1-3 %).
* The **p95 chunk time** is set by the diarizer's 240 ms cadence: every third 160 ms chunk carries a diarizer step
  (32-62 ms after), which is why p95 (125-180 ms) is 1.6x p50 (79-119 ms) and why "p95 < 60 ms" cannot be met with
  this head on 2 threads even at RTF 0.45; the server stays real-time because the mean, not the p95, is what the
  backlog integrates.
* **First partial**: the server lag (audio block in -> partial out) fell from 54-74 to 19-24 ms; the time from
  speech onset to the first partial is unchanged (640-1440 ms), because it is the RNNT's emission delay plus the
  160 ms chunk, not compute.
* **RSS**: the process needs about 1.0-1.1 GB in steady state with Nemotron-3 (`rss_end` 0.93-1.13 GB across the
  runs); the lifetime peak (`stats.peak_rss_mb`, `ru_maxrss`) is 1.2-1.55 GB because loading builds the model and
  then loads the state dict (a second copy of the 443 MB ASR weights, freed after the copy but never returned to the
  OS by macOS). `linear_t` adds nothing measurable (a probe that re-lays out the served encoder in a loaded process:
  1082 -> 1083 MB; `runs/perf.json` `rss_probe`). The 765 MB "after load" of the before-run is an artefact of the
  14 GB of swap in use at that moment (pages of the process were swapped out), so RSS rows taken under heavy swap
  are not comparable. The "< 1.2 GB" target holds for the working set, not for the lifetime peak; a meta-device load
  with `load_state_dict(assign=True)` would remove the transient (not done: the model's non-persistent buffers
  would need re-creation).

### Concurrency: sessions per 2 threads

`bench_serve.py run --streams K`: K sessions over 60 s AMI windows at different offsets, interleaved block by block
on one worker (as the server's single executor thread), queueing latency simulated in virtual time from the measured
compute (`runs/perf.json` `streams_default`; `--perf default`, load 4.6-5.6):

| K | compute per stream (RTF) | aggregate RTF | p95 chunk ms | real time? |
|---|---|---|---|---|
| 1 (e2e5 clips) | 0.51 | 0.51 | 162 | yes |
| 2 (60 s, steady state) | 0.65 | 1.31 | 180 | no: queueing latency grows 17 s over the minute |
| 3 | 0.59 | 1.77 | 173 | no |

With one worker and a diarizer head at RTF 0.2-0.4 per session, **one session per 2 threads** is what this machine
sustains at load ~5 (aggregate 1.31 at K = 2). On a quiet machine the steady-state single-session RTF is ~0.47
(section 3's A/B ratio applied to the E2E baseline), so 2 sessions would be marginal and 3 are out of reach without
batching the diarizer head across sessions (section 3, not done). The target "≥ 3 sessions per 2 threads" is not met;
GPU production, where the 380-row transformer is one small kernel launch, is where that number is recovered.

## 3. Optimisation pass

Rule: an option is accepted only if the protocol messages of the 5 AMI windows (1313 messages: every frame's VAD,
turn probability, four speaker columns and primary, every partial, turn_end and final) are identical to the
unoptimised run, or differ only in the last printed decimal of a probability with no decision changed, **and** it
lowers RTF or p95 in an interleaved A/B (`bench_serve.py ab`: both engines in one process, A, B alternated per clip,
so both see the same load). Unit tests on tiny models pin bit-identical outputs for each exact option
(`tests/test_perf.py`); `RUN_REAL=1` re-runs the identity check on the served checkpoints.

### Accepted (on by default: `audioforge-serve --perf default`, `audioforge.perf.DEFAULT`)

| option | what it changes | why it is exact | measured (A/B `none` vs `default`, Nemotron-3, e2e5, load 7-8) |
|---|---|---|---|
| `linear_t` | every encoder / diarizer `nn.Linear` weight stored column-major (`W.t().contiguous().t()`: same values, same shape) so the BLAS runs `x @ Wt` (NN) instead of `x @ W^T` (NT) | a GEMM computes each output element by the same dot product in either layout; Accelerate rounds identically for ≥ 2 rows. A 1-row input takes a GEMV whose rounding can differ in the last bit; in the served path only a 1-frame chunk is 1-row, which is where the 2 `eot` values below come from | the 2-row 512 -> 2048 GEMM: 134 -> 30 us; overall the largest item of the set |
| `pos_cache` | `RelPositionMultiHeadAttention` caches `linear_pos(pe(L))` per L (the cache + chunk length, one value in steady state) instead of recomputing the (2L-1) x d GEMM (143 rows at [70, 1]) on every call of every layer | same tensor, computed once | 17 layers x 2 passes x 143-row GEMM per 160 ms removed |
| `share_subsample` | the speaker-conditioned turn pass reuses the ASR pass's subsampled frames for the same mel chunk at the same stream position (speaker kernels act after the subsampling) | the same tensor is fed to the conditioned layers | the subsampling runs once instead of twice: 23.8 -> 4.6 ms per 160 ms |
| `joint_cache` | the per-frame RNNT decode keeps `joint.pred(g)` until a token is emitted (g only changes then) | same op on the same tensor | 0.8 -> 0.3 ms per 160 ms |
| **set total** | | | **RTF 1.39 -> 0.86 (0.62x), CPU 0.67x; ASR pass 0.70x, turn pass 0.42x, diarizer 0.71x; chunk p95 656 -> 247 ms (0.38x)**; messages: 1311 of 1313 bit-identical, 2 `eot` values differ by 1e-5 (0.34875 vs 0.34874 at t = 3.44 / 3.52 s of TS3004b; the rounded 5th decimal), 0 decision differences |

Before / after at comparable load (single runs, `runs/perf.json` `baseline_none` vs `streams1_default` / `components_default`):

| | before (`--perf none`) | after (`--perf default`) | E2E_FINAL live baseline (2026-09-27, load < 6) |
|---|---|---|---|
| server RTF, Nemotron-3, e2e5 clips | 0.84 (load 4.7) | 0.45-0.51 (load 3.5-5) | 0.64 |
| ASR pass / turn pass / diarizer | 0.28 / 0.28 / 0.28 | 0.14-0.15 / 0.11-0.12 / 0.21-0.24 | 0.17 / 0.17 / 0.30 |
| steady state (120 s window) | 1.38 (load 4.8-6.8) | 0.62 (load 3.8-4.1) | – |
| chunk p50 / p95 ms | 132 / 259 | 79-85 / 125-162 | – |
| server RTF, Sortformer v2 0.32 s, e2e5 | – | 0.61 (load 3.7; ASR 0.14 / turn 0.10 / diarizer 0.38; peak RSS 3.58 GB) | 0.80 |

Applying the A/B ratio (0.62x) to the E2E baseline gives **~0.40 with Nemotron-3 and ~0.50 with Sortformer v2 for
the whole server at 2 threads on the quiet machine of E2E_FINAL**; the direct measurements above at load 3.5-5 are
0.45-0.51. The target "≤ 0.35" is not reached: the remaining cost is 0.20-0.38 of diarizer head plus 0.24 of the two
encoder passes.

### Rejected

| candidate | result | why rejected |
|---|---|---|
| `fast_subsample` (the two depthwise 3x3 stride-2 `Conv2d` as nine strided multiply-adds) | A/B `default` vs `all`: RTF 0.974x (ASR pass 0.88x, 3.5 -> ~1 ms per 160 ms), tokens identical on 30 s, messages: 0 decision differences, `vad` differs by 1e-4 (4th decimal) on some frames, `eot` by 1e-5 | not exact (float rounding, max 4e-7 on the subsampled frames) for a 2.6 % gain; kept as an opt-in (`--perf all`) |
| `torch.inference_mode` instead of `no_grad` around the session | 20.5 vs 20.8 ms per 160 ms (encoder + RNNT microbenchmark), within noise | no measurable gain: the graph is already built under `no_grad` and the tensors are tiny |
| int8 dynamic quantisation of the encoder linears (`torch.ao.quantization.quantize_dynamic`) | `RuntimeError: Didn't find engine for operation quantized::linear_prepack NoQEngine` on this torch 2.14 macOS arm64 wheel | cannot run here; ONDEVICE.md measured torch int8 slower than fp32 (15.3 vs 12.5 ms per chunk) on this class of machine and ORT int8 faster, so the CPU route for int8 is ONNX Runtime, not torch; WER / turn deltas were therefore not measured |
| `torch.compile` of the 17 conformer layers (`dynamic=True`) | 26.4 vs 20.8 ms per 160 ms after a 22 s compile, and recompile-limit warnings from the streaming cache shapes | slower and fragile on CPU at 2-frame chunks |
| channels-last / fused ops | the convolutions are already the `fast_conv` unfold / linear path; the attention is SDPA | no separate kernel to fuse on CPU at these shapes |
| batching sessions through one encoder / diarizer call | not implemented: the cache-aware encoder state and the diarizer's per-session speaker cache are per-session tensors of different lengths; batching them needs padded batched streaming states in `FastConformerEncoder` and `AOSC` | the only route to ≥ 2-3 sessions per 2 threads on CPU (the diarizer head at 380 rows x K would run at the same GEMM efficiency); estimated a week of work with the identity tests; left as the next step |
| one shared encoder pass for the ASR and turn heads | not possible for the served head: it is speaker-kernel conditioned from layer 0, so its encoder input differs from the ASR pass's from the first layer; only the subsampling is shared (`share_subsample`) | a head-only conditioned turn head (`stage1_turn_v5_headonly`, research/archive/TURN_ABLATION.md) would remove the whole 0.11-0.15 turn pass, at the accuracy cost measured there |

### Tests

`tests/test_perf.py` (11 tests, tiny models): `linear_t` values unchanged and bit-identical encoder output for
≥ 2-row inputs; `pos_cache` bit-identical streaming tokens and frame events with the cache in use; `joint_cache`
identical tokens; `share_subsample` identical protocol messages with `--turn-input diar` and the memo hit;
`fast_subsample` within 1e-5; the whole exact set gives identical messages for the `timeout` and `hybrid` policies;
`Engine` applies the default set and `perf="none"` turns it off; `RUN_REAL=1` repeats the message identity on the
served checkpoints over the bundled 10 s clip.

### Reproduce

```bash
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/bench_serve.py ab --diar nemotron --clips e2e5 --opt-a none --opt-b default --out runs/perf/ab.json
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/bench_serve.py components --diar nemotron --clips long --opt default --out runs/perf/comp.json
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/bench_serve.py run --diar nemotron --streams 2 --opt default --out runs/perf/s2.json
RUN_REAL=1 .venv/bin/python -m pytest -q tests/test_perf.py
```
