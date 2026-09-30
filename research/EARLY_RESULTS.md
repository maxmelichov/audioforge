# Early results and server notes (moved from the README)

These sections were the bulk of the top-level README until 2026-09-29, when the README was cut down to what a user
needs. They are kept here verbatim (headings one level down): the laptop-scale synthetic results of the original
recipes, the real-speech status of 2026-09-26, the first measurements of the live server, and the corrections log of
the 2026-09-25 verification pass. The final numbers are in [FINAL_REPORT.md](FINAL_REPORT.md); where they differ, the
final report wins. Paths in the text were updated for the 2026-09-28 / 09-29 moves (`research/recipes/`,
`scripts/research/`).

### Verified behavior

- **The losses are exact.** RNNT and TDT losses match brute-force enumeration over all alignments
  (`tests/test_losses.py`).
- **Streaming matches offline.** Chunk-by-chunk cache-aware encoding equals the offline masked
  forward to 1e-4, for 4 context configurations. The incremental mel matches offline mel, and
  StreamingSession text equals offline decoding (`tests/test_streaming.py`).
- **Every recipe runs end to end.** Each one builds, trains, evaluates, saves to `.afm` and
  reloads bit-exact (`tests/test_recipes.py`).

### Results (laptop scale, synthetic data)

| Recipe | Type | Params | Result on synthetic validation set (seed 1, n=64; same 8 speakers and 24-word lexicon as training, about 1/3 of transcripts also appear in train) |
|---|---|---:|---|
| `parakeet_tdt_ctc` | reproduction | 2.3M | TDT WER **0.0%**, CTC WER 1.0% |
| `nemotron_streaming_rnnt` | reproduction | 2.3M | streaming RNNT WER **0.0%** at 160 ms chunks (CTC aux 30%) |
| `canary_aed` | reproduction | 2.8M | AED WER **2.1%** on mixed ASR + "translation" prompts (4 errors / 189 words; fresh seeds 1.4–1.8%; swapping the prompt flips the output task) |
| `sortformer_diar` | reproduction | 2.3M | DER **4.0%** on 64 held-out 2-speaker mixtures; 4.7–5.7% on fresh 256-mixture sets. Frame-level, 80 ms, no collar, mean per utterance. Each mixture is one speaker turn A→B, and only ~3% of speech overlaps. Baselines: true speech regions split at the midpoint 15%, true change point 2.4% |
| SALM (`train-salm --steps 800 --encoder-from runs/parakeet_tdt_ctc.afm`) | reproduction | 2.6M | WER **9.1%** on 16 utterances (5/55 words; a CPU re-run gave 5.7% on 200 fresh utterances) (encoder initialized from Parakeet run, LM fine-tuned) |
| **`voice_agent_frontend`** | new | 2.4M | one pass: WER **0.0%**, VAD frame acc 91% (base rate 51%; errors are ±1-frame boundary shifts), speaker ID **100%** (closed set: the same 8 synthetic speakers seen in training). EOU is **not yet solved**: frame acc 88% is below the 92% always-"no EOU" baseline (precision 40%, recall 91%, F1 0.55), and the head fires before speech ends in 66% of utterances |
| **`speaker_attributed_asr`** | new | 2.7M | DER 8.8%, per-speaker WER **3.0%** with oracle speaker activity; ~18% when conditioned on its own diarization output (smaller than a cascade, not more accurate) |
| **`codec_token_enhancer`** | new | 2.8M | clean-token accuracy **25%** from noisy input (4 x 800-way; uniform chance 0.125%, majority-code baseline 6.7–6.9%, re-encoding the noisy audio 1.6%; 27% on speech frames only); decoded log-mel L1 0.41 vs 0.70 for codes of the noisy input (clean-code ceiling 0.28). The decoded mel is **no better than a per-frame MLP's tokens** and is beaten by 1-parameter spectral subtraction |

**Fresh-set re-check.** The validation set above shares speakers, lexicon and generator with training.
Re-checked on text-disjoint audio and unseen speakers, the numbers stay within ~1.5 points, with two
exceptions: speaker ID is 100% on training speakers but 38% on unseen ones, and codec token accuracy
falls from 25.1% to 22.2%. SALM was not re-checked this way. See
[`research/archive/VERIFICATION.md`](archive/VERIFICATION.md) (C08).

### Ablations (verified, verdicts mixed)

Matched-budget ablations from [`research/archive/VERIFICATION.md`](archive/VERIFICATION.md) §3 (CPU, one seed,
synthetic data). All three verdicts are **mixed**.

- **One encoder, five heads vs separate models (`voice_agent_frontend`): mixed.** The inference gain is real:
  4× fewer encoder FLOPs (3.75 vs 15.0 GFLOPs per 60 s), analyze() 3.3× faster, 3.5× fewer parameters
  (2.42M vs 8.51M). At equal training steps accuracy is **much worse**: TDT WER 51.7% vs 0.26% at 1500
  steps, VAD 86% vs 94%. "No accuracy cost" is not supported at matched training budgets.
- **Codec-token enhancer: mixed.** 26.5% token accuracy beats majority (6.9%), noisy re-encoding (1.6%)
  and the best per-frame MLP (23.2%). As enhancement it has no advantage: the decoded mel is tied with a
  per-frame MLP's tokens and beaten by 1-parameter spectral subtraction (and by ~10 dB mel-SNR). The
  limit is the codec: even perfect tokens decode to 1.05 dB mel-SNR.
- **Speaker-attributed ASR in one model vs a cascade: mixed.** The only real gain is ~43% fewer
  parameters (2.71M vs 4.75M). Compute and latency are the same (one re-encode per speaker either way).
  DER is worse (7.2% vs 5.1%). With 3 speakers every system fails (0% speaker-count accuracy). Smaller,
  not better.

Example of cache-aware streaming (`cli stream runs/nemotron_streaming_rnnt.afm runs/samples/s0.wav`):

```
  0.80s  r
  0.96s  rig
  1.12s  right
  1.60s  right open
  2.08s  right open call
  2.40s  right open call left
final: right open call left
```

### What it took to make them learn (each fix is one of NVIDIA's principles)

| Symptom | Cause | Fix, and the NVIDIA practice it mirrors |
|---|---|---|
| AED WER 105%: the decoder ignored the audio and acted as an LM | cross-attention never aligned | auxiliary CTC on the transcript → 2.1% (*Canary encoders start from CTC models*) |
| SALM WER 100% | a frozen LM gives too weak a signal to a random encoder | init the encoder from a trained ASR run + adapt the LM → 9.1% (*Canary-Qwen = canary-1b-flash encoder + LoRA*) |
| streaming WER ~96% | utterance-level (per_feature) normalization can't stream, plus a short schedule (the two were changed together; no run isolates which one mattered) | frame-local normalization (here fixed global stats; NVIDIA's streaming FastConformer uses raw log-mel, `normalize: NA`) + 2k steps → 0.0% |
| multi-head ASR starved | the AAM speaker loss dominated the clipped shared gradient | per-head `grad_scale` |
| speaker ID at chance in multi-head | top layer is speaker-invariant after ASR training | `from_layers: all` learned layer mix → 100% (*NEST: weighted sum of layers for speaker tasks*) |
| codec: one code for every frame | unnormalized latents saturate FSQ's tanh | per-frame norm before FSQ; mel-codec for fast training (*NVIDIA mel-codec*) |

These runs use about 2.3–2.8M-parameter models trained for about 1–2.5 minutes each on an Apple M5
(the later speaker_aware_turn and streaming_sortformer runs take longer), on a synthetic tone language. They show that each recipe learns its task. They are **not**
comparable to NVIDIA's benchmark numbers. To scale up, override `encoder.d_model`/`n_layers`
and point `data` at real manifests.


### Real-speech status (2026-09-26)

Synthetic results above prove the code works; these are the numbers on real audio.

| What | Result | Where |
|---|---|---|
| NVIDIA's models as local teachers/benchmarks | parakeet-ctc-1.1b 1.64% WER, 63x realtime on M5 | `research/archive/TEACHERS.md` |
| NVIDIA weights loaded into audioforge (no NeMo) | streaming ASR 1.92% WER at 1 s lookahead, 2.48% at 0; Parakeet-CTC 1.68% | `research/archive/NEMO_IMPORT.md` |
| NVIDIA Streaming Sortformer v2 (CC-BY-4.0) in audioforge | DER 0.20 on AMI dev windows (our head 0.38, trivial 0.31) | `research/archive/SORTFORMER_IMPORT.md` |
| Our heads on the frozen NVIDIA encoder (AMI dev, unseen speakers, 160 ms chunks) | VAD 0.92 acc; speaker EER 15%; ASR gate held (1.23%) | `research/archive/STAGE1.md` |
| Encoder fine-tuning | WER gate trips (2.1 -> 4.3% in 500 steps) even at lr x0.05 with a KL anchor; encoder stays frozen | `research/archive/PLAN.md` §6 |
| **Speaker-aware end-of-turn on real meetings** | **leak-free end-to-end win** (eot-bench v2: n=974 AMI dev turns, label-free enrollment, 6 s horizon): hybrid rule (head trained on 6 s trails OR timeout) misses 61.9% of turn ends vs 74.8% for the deployable silence timeout on the same diarizer track (CI excludes 0), ties it on floor-open ends (45.9 vs 46.1); with oracle enrollment 28.7%. Remaining gap = label-free speaker enrollment (~37 pts); **arming the primary at the agent's own TTS end** (eot-bench v2 §9, `serve --enroll after_agent`; the agent end from the labels as a stand-in) cuts it: hybrid 51.6% with TitaNet-L voice following (−10.3 [−13.5, −6.9] vs 61.9; floor-open 34.0 vs 45.9) and 56.0% with no embedding (−5.9 [−8.6, −3.5]) at 6 s; at 2 s only the embedding-free variant is better (−3.9), the voice-followed one fires late (+8.5). 0.32 s diarizer setting: same misses, 0.5–0.8 s less latency. Clean speaker activity: 560 ms / 1.6% misses vs 1440 ms / 6.3%. | `research/archive/EOT_BENCH_V2.md`, `research/archive/STAGE1.md` |
| On-device | 111M model at 160 ms chunks: RTF 0.16 on one CPU core (a fixed-shape runtime study with random weights: mel + encoder + heads + greedy decode, no diarizer, no second encoder pass; not the served server, whose live figure is 0.64-0.80 at 2 threads, `research/archive/PERFORMANCE.md` §1) | `research/archive/ONDEVICE.md` |

In real meetings a mid-turn hesitation (median ~1.2 s) is longer than the silence at a true speaker
change (~1 s, and 45% of changes overlap), so silence timeouts cannot work in principle
(`research/archive/AMI.md`); whether a learned speaker-aware head can beat them at low latency is still
open — the remaining lever is training it on real diarizer tracks.

### Live streaming server (2026-09-26)

`audioforge/serve.py` is a WebSocket server (one session per connection) that runs the frozen NVIDIA streaming ASR
with our VAD/turn heads (`runs/stage1_heads_pretrained.afm`, 160 ms chunks) next to NVIDIA's Streaming Sortformer
(`runs/nemo_sortformer_v2.afm`, card 0.32 s "ultra low latency" setting by default) on one 80 ms frame clock. `scripts/stream_client.py`
streams a file in real time and logs every event. The protocol is in the module docstring: int16 PCM in;
`ready` / `frame` / `partial` / `turn_end` / `final` / `stats` JSON out.

```bash
audioforge-serve --port 8765 --threads 2      # = python -m audioforge.serve --asr models/stage1_served.afm --diar models/nemo_sortformer_v2.afm
                                              # [--diar-config low_latency] for the 1.04 s setting
python scripts/stream_client.py audio.wav --url ws://127.0.0.1:8765 --log ev.jsonl --summary s.json
```

Everything below was measured on 2026-09-26 with the paths of that day (`runs/stage1_heads_pretrained.afm`,
`runs/nemo_sortformer_v2.afm`); the served model built by `audioforge-download` is the later `stage1_served.afm`
(same encoder, retrained speaker head; `scripts/research/make_served_model.py`). The flag reference with every measurement is
[`docs/CONFIGURATION.md`](../docs/CONFIGURATION.md).

- **Default end-of-turn policy (`turn_policy: "timeout"`): plain silence timeout (1000 ms) on the diarizer's
  label-free primary column.** The primary is the column with the most activity over the last 5 s (no oracle, no
  enrollment). The turn ends once the primary has been inactive for `timeout_ms`, whether or not another column is
  active. The older rule, which also waits until no other column is active, is available as
  `turn_policy: "timeout_quiet"` (its `turn_end` is still tagged `timeout`). Why: at n=200 AMI dev turns
  (STAGE1.md) the plain timeout misses 38.4 % at <= 5 % false cutoffs, while "primary silent AND nobody
  else" misses 66-68 %, no better than the served head (69 %). The server used to implement the second rule while
  citing the first rule's number (INTEGRATION_VERIFY.md D1). STAGE1's 38.4 % picks the column using the
  oracle primary. The head-vs-timeout comparison for the shipped rule has to be made on the label-free primary,
  which is what eot-bench v2 (EOT_BENCH_V2.md) does. The head's probability is exposed only as the `eot`
  field and as the opt-in `head` / `both` policies.
- **Diarizer setting (`--diar-config`, default `low_latency_032`, since 2026-09-26).** Two NVIDIA card settings via
  `AOSCConfig.preset` (`streaming_diar.SORTFORMER_PRESETS`): `low_latency_032` = chunk 3 + right context 1 (0.32 s)
  and `low_latency` = chunk 6 + right context 7 (1.04 s, the previous default and the setting STAGE1 scored). Why the
  switch: research/archive/EOT_BENCH_V2.md section 7 (974 AMI dev turns, leak-free cross-fit, <= 5 % FC) gives the same miss
  rates at both settings (causal binding, 6 s: hybrid 63.7 vs 61.9 %, head 65.8 vs 66.5 %, timeout 70.1 vs 74.8 %,
  CIs overlapping) but 0.5-0.8 s lower P50 wherever systems fire (oracle timeout 2720 vs 3200 ms, head 2320 vs
  2960 ms; nominal emission delay 240 vs 840 ms). The cost is DER 28.3 vs 26.3 on the dev windows. The `ready`
  message now also carries `diar_config` and `column_lag_ms` (mean structural column lag); existing keys unchanged.
- Speaker columns lag the audio: a Sortformer column is final 80-240 ms (mean 160) after its frame ends with
  `low_latency_032` and 560-960 ms (mean 760) with `low_latency`, plus compute. The debug stat
  `diar_lag_ms_mean_measured` is derived from sample counts, so it always reads the structural mean at 1x; it is not
  a wall-clock measurement (D8). Wall-clock measurement, both settings, one process, CPU 2 threads, 1x, 2 AMI dev
  windows (IS1008b_003 15.2 s, ES2011b_027 20 s; load average ~1.5-2.4):

  | `--diar-config` | RTF | chunk p50 / p95 ms | frame arrival lag p50 / p95 / max ms | speaker-column age on arrival p50 (p95) ms | max backlog ms | timeout `turn_end` t |
  |---|---|---|---|---|---|---|
  | `low_latency_032` (default) | 0.49 / 0.53 | 80-89 / 148-163 | 126 / 210-229 / 242-380 | 210-227 (291) | 120-140 | 4.64 s, 16.64 s |
  | `low_latency` | 0.40 / 0.43 | 37 / 174-195 | 124-125 / 154-160 / 233-383 | 857 (1035) | 120 | 4.88 s, 16.88 s |

  The 0.32 s setting runs the diarizer about 3x as often (RTF +0.1) and still keeps up at 1x on 2 threads (lag slope
  2.4-2.6 ms per s of audio, backlog never above 140 ms). Speaker columns arrive ~630 ms younger, and the same 1 s
  timeout fires 240 ms earlier on these windows (the eot-bench gains above come from re-tuned operating points).
  Transcripts are identical (WER 28.3 / 25.8 % in both). ASR frames arrive p50 ~125 ms after their audio; the
  6-86 ms figure is the structural availability, without compute.
- Measured on this Mac (CPU, shared with a GPU job, load average 3-6), 3 AMI dev turn windows plus 1 LibriSpeech
  utterance, streamed at 1x. At 2 threads: RTF 0.37-0.52, p50 / p95 processing per 160 ms chunk 27-45 / 170-260 ms,
  first partial 45-120 ms after its audio was sent, peak RSS 3.1-3.3 GB. The server keeps up (backlog <= 200 ms).
  At 4 threads: RTF 0.34-0.42. The server patches the conformer convolutions with an equivalent unfold/linear CPU
  path (identical tokens), which gives about 3x on the ASR and 2x on the diarizer. Without it, RTF was 1.2-1.3 and
  the server fell behind.
- **Many speakers in the room (2026-09-28, [`research/archive/DIARIZATION_FIX.md`](archive/DIARIZATION_FIX.md)).** As
  shipped, `final.speaker` was the 5 s dominant column (not the turn's speaker), `--diarizer nemotron3` cut the
  diarizer to 4 of its 8 columns (speakers 5-8 invisible), and under load shedding every final became speaker 0.
  Fix: the launcher no longer adds `--diar-spks 4` (all 8 Nemotron-3 columns; `frame.speakers` is 8 long) and adds
  `--shed-diar hold` for either diarizer (last stable column at half diarizer cadence when shed, instead of everyone
  becoming speaker 0); a bare `python -m audioforge.serve` keeps the old `vad` rule. Opt-in: `--diar-labels registry`
  (stable voice-keyed ids per session, `speakers_seen` in `stats`, `speaker_conf` / `diar_shed` on finals;
  `--diar-embed titanet` for TitaNet-L embeddings) and session config `turn_policy: "timeout_any"` (ends every
  speaker's turn, `turn_end.policy: "change"` on a speaker change). On three 4-6-speaker clips DER 0.395 → 0.247 with
  the speaker count exact, per-final accuracy 0.74 → 0.83 (0.88 with TitaNet); forced shedding 0.809 → 0.329; AMI
  20 s windows unchanged (0.245). Flag reference:
  [`docs/CONFIGURATION.md` §7.4](../docs/CONFIGURATION.md#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any).
- **Known user: the TS-VAD turn path (2026-09-28, behind flags; [`research/archive/IMPROVEMENTS.md`](archive/IMPROVEMENTS.md)
  §1).** `--turn-input tsvad` feeds the turn head the output of a 0.26 M target-speaker VAD head on the ASR pass's
  block 4, conditioned on the user's voice print: sent by the client (`{"type": "enroll", "embedding": [192
  floats]}`, `--enroll explicit`), or taken from the next `--tsvad-print-s` seconds of speech after `agent_end`
  (`--enroll after_agent_arm`), optionally refreshed (`--tsvad-refresh-s`); each print is announced as a
  `voiceprint` message. `--diar-off` replaces the diarizer columns by [P(user), P(other)] so no diarizer runs. Offline
  on AMI / ICSI meetings it cuts missed turn ends from 61.9 to 39.3 % / 68.5 to 20.4 %; live on 69 mostly two-party
  sessions it misses more than the default `timeout` (+10.8 [+5.6, +15.8] points at 3 s) and equals `hybrid_dyn` +
  arming at 0.43x the server compute (RTF 0.34 vs 0.80). Use it for a product that holds its user's voice print in
  rooms, not as the two-party default (`runs/improve_115m.json`, `runs/e2e_tsvad.json`).
- The stage-1 turn head is speaker-kernel conditioned. The default `--turn-input session` feeds it the session's own
  encoder frames, with no speaker input, so it is off its training distribution. `--turn-input diar` runs a second
  encoder pass conditioned on the diarizer's primary column (RTF 0.58-0.61). On the 3 AMI windows both modes cut
  turns early.

### Corrections log (2026-09-25)

An adversarial verification pass ([`research/archive/VERIFICATION.md`](archive/VERIFICATION.md)) found these
claims wrong or overstated. The corrected text is above; the originals are kept here so the history stays visible.

- "four new solutions" → three: the docs' own table marks only 3 designs as new (C09).
- "56 tests", "4 reproductions, 3 new" recipes → 127 tests collected, 10 recipes: counts were stale (C09).
- "0.3–2M-parameter models, about a minute each" → 2.3–2.8M, 1–2.5 min: measured sizes and run times (C09).
- "held-out synthetic set" → same speakers/lexicon as training, ~1/3 transcripts shared; speaker ID 38% on unseen speakers (C08).
- EOU "acc 88% / recall 91%" → below the 92% always-"no EOU" baseline, fires early in 66% of utterances: EOU is not solved (C05).
- Speaker-attributed WER "3.0%" → 3.0% only with oracle activity, ~18% end to end (C06).
- Codec "chance ~0.5%" → uniform 0.125%, majority 6.7–6.9%; decoded mel no better than a per-frame MLP (C07, §3.2).
- SALM "~3M, frozen LM" → 2.6M, LM fine-tuned (freeze_llm=False); 9.1% is 5/55 words (C04).
- Sortformer "overlapping mixtures" → one A→B turn with ~3% overlap; fresh sets 4.7–5.7% (C03).
- Streaming fix "raw log-mel was the cause" → NVIDIA's streaming model uses raw log-mel; the cause was utterance-level normalization or the schedule, not isolated (C10).
- Multi-head "no accuracy cost" → not supported at matched training steps (TDT WER 51.7% vs 0.26%) (§3.1).
- Research docs: catalog counts (4 gated, not 5; 51 of 54, not 51 of 53; family row C17 refuted), Canary-1B 24 layers, Nemotron lookahead 1.04 s, "≤600 ms beats every commercial API" refuted (LiveKit 543 ms, JoinIn AI Baton 577 ms), VibeVoice-ASR-Streaming exists, the 18 ms streaming step not reproduced (~26 ms under load) (C14–C28).
