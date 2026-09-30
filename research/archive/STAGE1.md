# Stage 1: first real-speech numbers for the fused front-end (AMI dev)

Checkpoint: `runs/stage1_heads_pretrained.step500.afm`. This is the only checkpoint of
`research/recipes/stage1_heads_pretrained.yaml`: frozen NVIDIA 115M cache-aware FastConformer + RNNT/CTC, with new
VAD, EOU, speaker, Sortformer and speaker-kernel turn heads (4.9M trainable params). Training was planned for
**2000 steps**. It crashed with an MPS out-of-memory error in the RNNT joint just after step 800, so the last saved
checkpoint is **step 500**. The trainer never ran its end-of-run validation, so these are the first dev
numbers for this model.

All numbers are on **AMI dev** (IS1008b, ES2011b, TS3004b, IB4002). These meetings and their speakers are
not in train. Each set is the recipe's seeded cap (`n_val`, seed 0) through `ami.recipe_data`. Everything ran
on CPU with 2 threads and loaded the model once per process. A full run takes about 1 minute.

## Commands

```bash
S=<scratch dir>   # load_model extracts the archive via tempfile -> keep it out of /tmp
TMPDIR=$S .venv/bin/python scripts/eval_stage1.py --ckpt runs/stage1_heads_pretrained.step500.afm \
    --n 64 --tasks asr,spk --out runs/stage1_step500_eval.json
TMPDIR=$S .venv/bin/python scripts/eval_stage1.py --ckpt runs/stage1_heads_pretrained.step500.afm \
    --n 64 --tasks diar,turn --out runs/stage1_step500_eval.json
```
Results are merged into `--out`, so tasks can be split across processes. `--n-asr` defaults to `n // 2` = 32.

## Training curve (runs/stage1_heads_pretrained.log)

| step | vad | eou | spk | diar | turn |
|---|---|---|---|---|---|
| 1 | 0.746 | 0.749 | – | 0.698 | 1.029 |
| 50 | 0.212 | 0.641 | 10.97 | 0.492 | 0.865 |
| 200 | 0.164 | 0.380 | 6.52 | 0.415 | 0.247 |
| 400 | 0.133 | 0.296 | 4.74 | 0.404 | 0.134 |
| 500 (ckpt) | 0.180 | 0.391 | – | 0.204 | 0.237 |
| 800 | 0.199 | 0.364 | – | 0.234 | 0.151 |

Batches are single-source, so each logged value comes from whichever source was drawn: `spk` only on
AMI-turn batches, and diar is lower on synthetic batches. The curve is noisy and was still falling for turn
and spk (spk was 4.7 to 9.8 at steps 400 to 800) when training crashed. The WER gate held
(`wer_gate step=500 wer=0.0113 baseline=0.0113`), which confirms the frozen encoder, RNNT and CTC.

## 1. Turn-taking: AMI turn windows (64 examples, mean 9.1 s)

This uses `conversation.eot_bench` with the scores from `heads.turn.turn_scores` and the baselines from
`conversation.baselines`, following the same code path as `TurnHead.evaluate_model`. Latency is the dead
air after the true end of turn, and a false cutoff (FC) is a firing inside the turn.

| system | P50 / P90 @ ≤5 % FC | miss @ ≤5 % FC | FC @ P50 ≤ 400 ms |
|---|---|---|---|
| (a) turn head, oracle primary activity | 1440 ms / inf | 47.5 % | unreachable |
| (b) turn head, own diarization (`diar` source) | inf / inf | 95.1 % | unreachable |
| (c) silence timeout, oracle primary activity (1.12 s timeout) | 1200 / 1200 ms | 3.3 % | 9.4 % (80 ms timeout) |
| (d) silence timeout, oracle any-speaker activity | inf / inf | 60.7 % | unreachable |

Diagnostics:
- The primary activity fed to the turn head in (b) misses 85.1 % of the primary's speech frames. That is
  the causal prefix re-run. A single offline pass still misses 83.3 %.
- AMI examples have every field that `eot_bench` and `baselines` need (`spk_act`, `primary_act`,
  `onset_frame`, `turn_end_frame`, `spk_targets`). Nothing is missing.

How to read this table. Several points make it easier than the synthetic benchmark in some ways and harder
in others:
- **The latency ceiling is about 2 s.** A turn window ends at most 2 s after the turn end (`trail_sec`),
  and earlier if the same speaker resumes. The post-end region is 1.04 / 2.00 / 2.08 s (min / median /
  max). Any latency beyond that counts as a miss (inf), so a 47 % miss means "no firing within 1 to 2 s".
- **The model is quantised to 1.12 s and the baselines are not.** The encoder is `att_context_size`
  [70, 13], so eot_bench uses chunk = 14 frames: a decision is only emitted at the end of the 1.12 s chunk
  containing it. The model's minimum dead air is 80 to 1120 ms, with a median of about 600 ms, so **P50 ≤
  400 ms is structurally out of reach for it**. The timeouts run with zero look-ahead.
- **The oracle-primary timeout (c) is almost the label itself.** EOT is defined as every frame after the
  last primary-active frame. The only thing that can fool it is a within-turn pause, and AMI word timings
  absorb pauses shorter than about 0.5 s (research/archive/AMI.md, caveat 1). Only 9.4 % of these turns contain
  a hesitation, which is exactly (c)'s FC at an 80 ms timeout. This is an oracle upper bound, not a
  deployable baseline.
- **(d) is the deployable speaker-blind baseline, and it fails.** In 61 % of turns someone else talks
  within the post-end window, which matches AMI's 43 % overlapping switches plus quick replies. Silence of
  "anyone" never comes.

## 2. Diarization: AMI diar windows (64 × 20 s, offline head pass, `metrics.frame_der`)

This is a frame-level DER (80 ms frames) with no collar, overlap included, and a words-only reference with
4 columns. The primary number is the **pooled** DER (errors / speaker-frames). The per-window mean, which is
Trainer.evaluate's `der_diar`, is dominated by near-silent windows: it is 1.75 for the head and 9.8 for
"one speaker always", so it is not reported further.

| system | DER (pooled) | miss | FA | confusion |
|---|---|---|---|---|
| diar head, step 500 | **0.432** | 0.219 | 0.074 | 0.139 |
| all silence | 1.000 | 1.000 | 0 | 0 |
| one speaker always on | 0.569 | 0.144 | 0.257 | 0.168 |
| one speaker = oracle any-speaker VAD | 0.312 | 0.144 | 0 | 0.168 |

The diar head beats the trivial baselines but is **worse than a speaker-blind oracle VAD** (0.43 vs 0.31).
Its mean output probability per column on turn windows is 0.665 / 0.135 / 0.061 / 0.016. **The head has
collapsed onto column 0 and acts as a VAD.** This also explains the turn result in (b):
- The primary is the first speaker to arrive in only 9 of the 64 turn windows. The windows start up to 4 s
  before the turn, so someone else usually speaks first.
- `primary_column` therefore selects columns 1 or 2, which are almost never on, and that gives the 83 %
  miss.
- The best single column would miss only 13.5 %, and that column is column 0, which fires for anyone.

## 3. VAD (the same 64 diar windows; label = any speaker active, `train.derive_labels`)

| system | frame acc | recall | precision |
|---|---|---|---|
| VAD head | **0.928** | 0.962 | 0.945 |
| always speech | 0.769 | 1.000 | 0.769 |

## 4. Speaker verification (64 AMI asr-mode single-speaker segments, 15 speakers, mean 6.2 s)

EER is `metrics.eer` over the cosine scores of all 2016 pairs (152 target pairs). The within-meeting trials
(487 pairs) are the honest ones: in a cross-meeting pair, room and channel alone give the answer away.

| embedding | EER all pairs | EER within-meeting |
|---|---|---|
| speaker head (AAM, 190 train speakers) | 20.4 % | 37.0 % |
| baseline: mean-pooled layer mix, untrained | 20.6 % | 34.9 % |

**At step 500 the speaker head has learned nothing that transfers to unseen speakers.** It does no better
than mean-pooling its own input features, and within a meeting both are close to chance.

## 5. ASR sanity (frozen RNNT, 32 AMI asr-mode segments, 544 words, `teachers.normalize_text`)

| data | WER |
|---|---|
| LibriSpeech test-clean first 50 (wer_gate) | 1.13 % |
| AMI dev single-speaker segments (Mix-Headset) | **23.2 %** |

The errors are typical of conversational speech: dropped or substituted fillers ("uh" → "the"), short
utterances ("i'll do the notes yeah thanks" → "alcia thanks"), and domain words ("sharing folder" →
"scheduling follow"). This is the first honest measure of how the pretrained ASR does on meetings. It
matters for the turn head's text branch, which reads this RNNT's greedy decode.

## 6. Against the synthetic ablation (runs/archive/turn_ablation_v3.json, 1000 synthetic conversations)

The metric is the same function (`eot_bench`). The data, look-ahead and latency ceiling differ, so only
the pattern can be compared, not the absolute numbers.

| system | synthetic v3: P50 / P90 @5 % FC, miss | synthetic: FC @P50 ≤ 400 | AMI dev step 500: P50 / P90, miss | AMI: FC @P50 ≤ 400 |
|---|---|---|---|---|
| timeout, oracle primary | 2560 / 2560, 0 % | 46.3 % | 1200 / 1200, 3.3 % | 9.4 % |
| timeout, any speaker | 1840 / inf, 10.9 % | 59.3 % | inf, 60.7 % | unreachable |
| turn head, speaker + text, oracle | 1680 / 2480, 5.8 % | 31.8 % | 1440 / inf, 47.5 % | unreachable |
| turn head, speaker + text, own diar | 1680 / 2480, 5.2 % | 41.0 % | inf, 95.1 % | unreachable |

- On synthetic data, hesitations are frequent and long, so the oracle timeout is weak and the turn head beats it.
- On AMI, hesitations are mostly invisible in the labels (9 % of turns), so the oracle timeout nearly
  equals the label, while the any-speaker timeout collapses because of overlap.
- The v3 synthetic models are small standalone encoders, not this pretrained backbone with its 1.12 s
  chunks.

## Interpretation

**What is learned at step 500:**
- **VAD works on real meetings:** 92.8 % accuracy against 76.9 % for the always-speech baseline, with 96 %
  recall.
- The frozen backbone is intact (WER gate unchanged).
- The turn head, given oracle primary activity, fires within the 2 s window for about half of the turns
  at 5 % false cutoffs. It is not yet the speaker-aware win it should be, and (a) and (d) do not see the
  same information: (a) has oracle primary activity, (d) only speaker-blind activity. (d), which misses
  61 %, shows that the speaker-aware signal is what AMI needs.

**What is not learned:**
- **Speaker identity for unseen speakers:** EER equals the untrained mean-pooling baseline.
- **Diarization beyond activity:** the Sortformer head collapsed to one column. Its DER of 0.43 is worse
  than a speaker-blind oracle VAD (0.31), with 14 % confusion and 22 % miss.
- The turn head has not yet matched the oracle-primary timeout. On AMI that timeout is close to an
  oracle of the label: 1200 ms / 3.3 % miss against the head's 1440 ms / 47.5 %. Part of the gap is
  structural: 1.12 s chunk quantisation against zero look-ahead, and a ceiling of about 2 s.

**What is limited by the diarizer:** the entire "own diarization" turn path.
- The primary track misses 85 % of the primary's speech (83 % even offline). That is the column collapse
  combined with arrival-rank enrollment, when the primary is the first speaker to arrive in only 14 % of
  windows.
- On top of that, the recipe has no `prefix_prob`, so the prefix re-run weakness from TURN_ABLATION.md A1
  is still present.
- Until the diar head separates speakers, a turn head conditioned on the model's own activity cannot beat
  the silence baselines on real speech.

**Recommended next steps:**
- Resume from step 500 with the OOM fixed (for example, a smaller RNNT `fused_batch_size`, or no RNNT
  loss on frozen heads).
- Add `from_layers: all` and `prefix_prob` to the diar head, plus a higher weight or more steps.
- Re-run this script at every checkpoint.

Caveats:
- n = 64 per set, which gives FC-rate resolution of 1.6 points and wide confidence intervals.
- The mixed-down headset audio is not the far-field condition.
- The AMI dev set here has 4 meetings only.

## Stage 1 v2 — final model (2000 steps, heads at [70,1] = 160 ms, diar from_layers all + prefix loss)

`scripts/eval_stage1.py --ckpt runs/stage1_heads_pretrained.afm --n 64 --out runs/archive/stage1_final_eval.json` (AMI dev, unseen meetings/speakers)

| metric | step 500 (v1, 1.12 s chunks) | final v2 (160 ms chunks) | baseline |
|---|---|---|---|
| turn head, oracle activity, P50 @≤5% FC / miss | 1440 ms / 47.5% | **1440 ms / 6.6%** | oracle-primary timeout 1200 ms / 3.3% (≈ label; AMI hides pauses < 0.5 s) |
| turn head, own diarization, miss | 95.1% | 93.4% | any-speaker timeout: 60.7% miss |
| diar pooled DER (miss / FA / conf) | 0.432 | **0.384** (0.24 / 0.06 / 0.09) | one speaker + oracle VAD: 0.312 |
| primary-activity miss from own diar | 0.85 | 0.64 | — |
| VAD acc / recall | 0.928 / 0.962 | 0.932 / 0.960 | always-speech 0.769 |
| speaker EER all / within-meeting | 20.4% / 37.0% | **14.5% / 30.8%** | untrained features 20.4% / 34.9% |
| frozen RNNT WER on AMI segments | 23.2% | 23.2% | LibriSpeech gate 1.13% (held at every check) |

Reading: with the correct 160 ms context the turn head now fires reliably given clean speaker activity,
and speaker identity starts to transfer; **diarization does not** — the frozen, ASR-trained encoder
plus a small Sortformer head is still worse than a trivial baseline, so the end-to-end turn path
stays blocked. Next levers: stage 2 (gentle unfreeze with the KL anchor) and NVIDIA's own streaming
Sortformer as a ported diarizer / teacher.

## Turn head with an external diarizer (NVIDIA streaming Sortformer v2)

Question: the end-to-end turn path is blocked by the own diar head (above). Does a real diarizer unblock it,
and does the speaker-aware turn head then beat what a developer would build today (NVIDIA's diarizer + a
silence timeout on the user's track)?

```bash
S=<scratch dir>
TMPDIR=$S .venv/bin/python scripts/eval_stage1.py --ckpt runs/stage1_heads_pretrained.afm --n 64 --tasks turn \
    --diar-ckpt runs/nemo_sortformer_v2.afm --diar-cache $S/sortformer_turn64.pkl \
    --out runs/archive/stage1_turn_external_diar.json      # ~220 s on 2 CPU threads (streaming diarizer 150 s)
```

Setup (same 64 AMI dev turn windows, mean 9.1 s, same `eot_bench` / `silence_scores` code as section 1):
- The Sortformer (`runs/nemo_sortformer_v2.afm`, see SORTFORMER_IMPORT.md) runs on its own 128-mel
  preprocessor and encoder (`model.encode` of the diarizer). **Offline** = one head pass over the whole
  window (non-causal within the window: an upper bound). **Streaming** = `StreamingDiarizer`, window mode,
  the card's low-latency setting (chunk 6, right context 7, FIFO 188, update 144, cache 188, encoder left
  context 188): **input-buffer latency (6 + 7) × 80 = 1040 ms**. Frame t's activity is available at frame
  (t // 6 + 1) · 6 + 7, i.e. 560–960 ms (mean 760 ms) later than a zero-look-ahead timeout would see it. Combined
  with the turn head's own chunk, it adds 0–880 ms (mean 293 ms) to the head's emission time. Latencies of the
  streaming rows use this exact emission time (`eot_bench_emit`: post-end scores re-indexed by emission time,
  pre-end frames unchanged, so a firing on a pre-end frame is still a cutoff; it reproduces eot_bench's own chunk rule
  exactly, and a check in the script asserts this).
- **Primary column = "enrollment by who is talking":** the diarizer column with the largest overlap
  (p > 0.5 on oracle primary frames, ties broken by the soft overlap) over frames [onset, turn_end) only, chosen per
  example and per source. It never looks at frames after the turn end. It is optimistic: it assumes the system knows which slot
  is the user, the stand-in for voice enrollment. The column is fixed for the window. Offline and streaming pick the same
  column in 92 % of windows. This replaces the own-diar source's arrival-rank rule, which fails on AMI because the
  primary arrives first in few windows.
- The turn head is conditioned on the chosen column's probability as `spk_act`, with the same speaker-kernel encode,
  greedy-decoded text state and decode as `turn_scores` (`turn_scores_given_act`).
- Frame alignment: the diarizer and the turn model both give `n_frames(len(audio))` frames. All 64 windows
  had equal counts (offline and streaming). The code still crops to the common length, or edge-pads by one frame.
- Checkpoint note: `runs/stage1_heads_pretrained.afm` loads with `att_context_size` [70, 13], so the turn
  head emits in 14-frame (1.12 s) chunks, as in `runs/archive/stage1_final_eval.json` (`chunk_frames` 14).

| system (64 AMI dev turns) | P50 / P90 @ ≤5 % FC | miss @ ≤5 % FC | FC @ P50 ≤ 400 ms | primary-act miss / FA vs oracle | post-end frames "active" |
|---|---|---|---|---|---|
| (a) turn head + oracle activity | 1440 / 2160 ms | 6.6 % | unreachable | 0 / 0 | 0 |
| (b) turn head + own diar (causal prefixes, arrival rank) | inf / inf | 93.4 % | unreachable | 63.7 % / 8.9 % | 15.8 % |
| (c) turn head + Sortformer **offline** (upper bound) | 1920 / inf | 32.8 % | unreachable | 12.9 % / 26.6 % | 18.7 % |
| (d) turn head + Sortformer **streaming**, low latency (+1040 ms buffer) | 2080 / inf | 36.1 % | unreachable | 16.0 % / 28.8 % | 22.7 % |
| (e1) cascade: timeout on Sortformer offline track | 1600 / inf | 26.2 % | 45.3 % | as (c) | as (c) |
| (e2) cascade: timeout on Sortformer streaming track | 2400 / inf | 32.8 % | unreachable | as (d) | as (d) |
| (e3) timeout on own-diar track | inf / inf | 91.8 % | unreachable | as (b) | as (b) |
| (f1) timeout, oracle primary activity | 1200 / 1200 ms | 3.3 % | 9.4 % | 0 / 0 | 0 |
| (f2) timeout, oracle any-speaker activity | inf / inf | 60.7 % | unreachable | — | — |

FA is per primary-speech frame (DER convention). Per non-primary frame it is 6.1 % for own diar, 18.3 % for
offline and 19.9 % for streaming. "Post-end active" is the share of frames after the turn end on which the track still says the
primary is talking. Those frames are what keep a timeout, or a speaker-conditioned head, from firing. Miss @ ≤5 % FC
means no firing inside the window, plus, for streaming rows, the emission delay after it. n = 64, so one turn is 1.6 points.

Interpretation:
- **A real diarizer closes most of the oracle to own-diar gap, but not all of it.** Miss at 5 % FC falls from 93 % (own diar)
  to 33 % offline and 36 % streaming. Oracle is 7 %. The Sortformer track finds the primary's speech: it misses
  13–16 % of primary frames, against 64 % for own diar. The remaining gap is **false alarm**. The chosen column is "on" for
  19–23 % of post-end frames. The likely cause is the next speaker's speech landing in the user's slot around the switch,
  which AMI's overlaps make common. This was not checked per example. The turn head was trained on clean binary oracle activity, so it has never seen this kind of error.
- **The turn head does not beat the diarizer + timeout cascade.** In real time, (d) against (e2), the head is 320 ms
  faster at P50 (2080 vs 2400 ms) but misses 2 more turns (36.1 vs 32.8 %). That is a tie at this n. Offline, the plain
  timeout is better: 1600 ms / 26.2 % against the head's 1920 ms / 32.8 %. The head's 1.12 s emission chunks cost
  it latency that the timeout does not pay. P50 ≤ 400 ms is unreachable for every real-time system. Only the offline-track
  timeout gets there, at 45 % false cutoffs.
- Every real system has P90 = inf. More than 10 % of the turns that are not cut off never fire within the
  roughly 2 s post-end window (plus the streaming emission delay).
- Consequence: the lever is the activity track's post-end false alarms, not the diarizer's recall. Two fixes follow.
  (1) Train the turn head's `spk_act` on real diarizer tracks: Sortformer outputs, streaming, with enrollment-picked columns.
  Then it learns to discount a slot that stays on through a speaker change, which the speaker kernels and the text state
  should make possible. (2) Rerun with the 160 ms head chunk that the v2 recipe intends (`[70, 1]`).

Caveats:
- The column choice uses oracle activity inside the turn. It is an optimistic enrollment stand-in, not speaker ID.
- The streaming procedure is ours, not NeMo's (SORTFORMER_IMPORT.md, streaming caveat).
- Windows start at most about 4 s before the turn, so the speaker cache is barely exercised.
- The P50 values for streaming rows can exceed the 2.08 s window ceiling because of the emission delay.
- There is one checkpoint, one seed and n = 64.

## Stage 1 retrained at the true 160 ms context (2026-09-26)

The v2 run above had a duplicated `encoder:` key in its recipe and actually ran at [70,13] (1.12 s
chunks). Retrained with `att_context_size: [70, 1]` (saved config verified). Same 64 AMI dev turns.

| system | P50 @≤5% FC | miss @≤5% FC | FC @P50≤400 ms |
|---|---|---|---|
| turn head + oracle activity | 1520 ms | **23.0%** (was 6.6% with 1.12 s lookahead) | 9.4% |
| turn head + own diar | inf | 72.1% | 76.6% |
| turn head + Sortformer offline / streaming | inf / inf | 60.7% / 82.0% | 37.5% / — |
| **timeout on Sortformer track offline / streaming** | 1600 / 2400 ms | **26.2% / 32.8%** | 45.3% / — |
| timeout on oracle primary | 1200 ms | 3.3% | 9.4% |
| timeout on any-speaker activity | inf | 60.7% | — |

Other heads at 160 ms: VAD acc 0.921, DER 0.394, speaker EER 15.0% / 32.2% within meeting, RNNT WER
25.0% on AMI segments (gate 1.23% at [70,1], held).

**Reading (honest):** without a second of future audio the head loses most of its edge. On real
meetings at low latency it does **not** beat a silence timeout on the same diarizer track — the
cascade a developer can build today with NVIDIA's Sortformer. The head has never been trained on
real diarizer tracks (only oracle / synthetic noise), so its failure mode — the next speaker landing
in the primary's slot right after the turn end — was never in its training data. That is the one
untried lever; if it does not close the gap, the speaker-aware-turn claim is not supported on AMI.

## Turn head trained on real Sortformer tracks (2026-09-26, `research/recipes/stage1_turn_on_sortformer.yaml`)

Only the turn head trained (1500 steps, 160 ms context, p_ext 0.7 on offline Sortformer tracks of the
3274 AMI training turns). Same 64 AMI dev turns; streaming tracks at inference.

| system | P50 @≤5% FC | miss @≤5% FC | FC @P50≤400 ms |
|---|---|---|---|
| **turn head + oracle activity** | **880 ms** | 4.9% | **6.3%** |
| timeout on oracle primary | 1200 ms | 3.3% | 9.4% |
| turn head + Sortformer offline / streaming | 2000 / 2800 ms | 48.4% / 49.2% | 31% / — |
| **timeout on Sortformer track offline / streaming** | 1600 / 2400 ms | **26.2% / 32.8%** | 45% / — |
| turn head + own diar | inf | 95% | — |

**Verdict on AMI:** given clean speaker activity the learned head now beats a silence timeout at low
latency on real meetings (880 vs 1200 ms at ≤5% false cutoffs; 6.3% vs 9.4% false cutoffs at 400 ms).
Driven by a real streaming diarizer it does not: 49% vs 33% misses for the Sortformer + timeout
cascade. Diarizer errors around speaker changes (post-end activity in the primary slot, 15–23% of
frames) hurt the learned head more than a timeout. Untested remedies: train on *streaming* tracks,
feed all four diarizer columns instead of one primary track, more conversational data. The
speaker-aware-turn claim is therefore **supported only with oracle speaker information** and
**not supported end to end** on AMI at this scale.

### n = 200 (same model, first 200 AMI dev turns) — supersedes the n=64 table above

| system | P50 @≤5% FC | miss @≤5% FC | FC @P50≤400 ms |
|---|---|---|---|
| turn head + oracle activity | 1440 ms | 29.5% | 11.0% |
| timeout on oracle primary | 1440 ms | 6.3% | 14.0% |
| turn head + Sortformer offline / streaming | inf / inf | 64.2% / 69.0% | 33.5% / — |
| timeout on Sortformer offline / streaming | 1760 / 2640 ms | 32.1% / 38.4% | 40.5% / — |
| timeout on any-speaker activity | inf | 77.0% | — |

**Corrected verdict:** the n=64 subset was favorable. At n=200 the learned head is only marginally
better than a silence timeout at the aggressive 400 ms operating point (11% vs 14% false cutoffs)
and far worse at ≤5% false cutoffs (29.5% vs 6.3% misses), even with oracle speaker activity.
**The speaker-aware end-of-turn claim is not supported on AMI at this scale**, with or without a
real diarizer. Always report n ≥ 200.

## Turn head v3 (2026-09-26): all four diarizer columns + silence counters + future-activity aux, trained on streaming tracks

`research/recipes/stage1_turn_v3.yaml`, 2000 steps, only the turn head trains; AMI dev n=200.

| system | P50 @≤5% FC | miss @≤5% FC | FC @P50≤400 ms |
|---|---|---|---|
| **turn head v3 + oracle activity** | **560 ms** | **5.8%** | **8.0%** |
| timeout on oracle primary | 1440 ms | 6.3% | 14.0% |
| turn head v3 + Sortformer offline / streaming | inf / inf | 57.4% / 62.6% | 42.5% / — |
| timeout on Sortformer offline / streaming | 1760 / 2640 ms | 32.1% / 38.4% | 40.5% / — |
| "primary silent & nobody else" rule on Sortformer | inf | 66–68% | — |

**Reading:** given a clean speaker track the learned head now dominates the silence timeout on real
meetings at 160 ms — 2.6x less dead air at equal false-cutoff and miss rates. Driven by the
streaming diarizer it improved (69% → 63% misses) but still loses to the timeout (38%). The wall is
diarizer noise: either make the head robust to it or get a better diarizer.

## Consolidated turn-taking results (2026-09-26, AMI dev n=200, misses at ≤5% per-turn false cutoffs)

| training / decision variant | oracle activity (P50 / miss) | Sortformer streaming track (miss) |
|---|---|---|
| timeout on the track (baseline) | 1440 ms / 6.3% | **38.4%** |
| v3: 4 columns + counters + future-aux, streaming tracks | **560 ms / 5.8%** | 62.6% |
| v4: + speaker kernels retrained on noisy tracks | 880 / 10.5% | 62.1% |
| v5: track reaches only the head (no kernels) | 1920 / 47.4% | 82.6% |
| v3 + ICSI (7117 turns) | 720 / 9.5% | 61.1% |
| v3 on flush-free tracks | 720 / 7.9% | 60.0% |
| hybrid (head OR timeout), v3 | 560 / **1.6%** | 37.4% |

Leak-free scoring (eot-bench v2, n=974, label-free enrollment, 2 s horizon): timeout 92.5%, v3 86.0%;
v3 worse than the timeout on floor-open ends. Every streaming system is bounded by the diarizer's
~0.9 s lag plus the 2 s window.

**Verdict for this session:** the speaker-aware head beats the silence timeout on real meetings only
when it is given clean speaker activity (2.6x less dead air at equal false cutoffs; hybrid 1.6% misses).
Driven by NVIDIA's streaming diarizer it does not, and none of the levers tried — retraining the
kernels, head-only input, 2x more conversational data, flush-free tracks, a hybrid decision rule —
closes the gap; the best end-to-end number is 60% misses vs 38% for the timeout on the same track.
The bottleneck is the *interaction* between diarizer errors and the speaker-conditioned encoder,
which was trained on clean activity and cannot be retrained without hurting the clean path.
Untried: a lower-latency diarizer setting (0.32 s), a diarizer fine-tuned on word-level labels, and
joint training of diarizer + turn head (in-pass self-conditioning, IDEAS.md).

## Turn head v3 trained on 6 s trails (2026-09-26, `research/recipes/stage1_turn_v3_trail6.yaml`)

Why: under eot-bench v2 at the 6 s horizon (research/archive/EOT_BENCH_V2.md §3) head v3 misses more than the plain deployable
timeout (79.6 vs 74.8 % all; 80.3 vs 46.1 % floor-open). v3 was trained on 2 s trails, so it never saw more than 2 s
of post-end silence. The fix under test: the same recipe (only the turn head trains, 2000 steps, same conditioning and
val), but the AMI train windows get `trail_sec: 6.0`, `window_sec: 20` (lead 4 s).

**Examples.** 3274 train turns, the same turns as the 16 s / 2 s set. The trail still stops at the primary's next
turn. 1344 keep the full 6.0 s, and 1353 have ≥ 75 post-end frames (9 stop less than one frame short). 1929 stop at
the same speaker's next turn, 1 at the meeting end. The 20 s cap never clips a trail (only the lead: 343 onsets
clipped). Post-end frames p5 / p50 / p95 = 16 / 58 / 76 (the 2 s set: ≤ 26).

**Tracks.** Flush-fixed streaming Sortformer tracks (window + 14 frames of the following meeting audio). There are
3274 of them in `data/ami/cache/sortformer/train_trail6/`: `ext_tracks.split_dirname`, and a recipe `trail_sec` ≠ 2
selects `<split>_trail<x>`, so the default caches are untouched. MPS took 2099 s (4 resumable runs). `--verify-cpu 3`:
max |MPS − CPU| = 4.5e-6. Training (MPS) took 1096 s and hit 3274 / 3274 tracks (`require: true`, no fallback). WER
gate flat at 0.0123.

**eot-bench v2** (n = 974 AMI dev turns, cross-fitted threshold at ≤ 5 % per-turn false cutoffs, miss % with 95 %
bootstrap CI; `runs/archive/turn_v3_trail6_eval_leakfree.json` vs `runs/archive/turn_v3_eval_leakfree.json`, same tracks and folds).
Causal = label-free `causal_dominant` enrollment. Timeout (deployable) = silence timeout on the causally bound
streaming track, armed at its first active frame.

6 s horizon (block C, extended windows):

| system | all | floor-open (236) | taken (738) |
|---|---|---|---|
| timeout, causal (deployable) | 74.8 [72.0, 77.5] | **46.1 [39.6, 52.6]** | 83.6 [80.8, 86.4] |
| head v3, causal | 79.6 [77.0, 82.1] | 80.3 [74.3, 85.5] | 79.4 [76.5, 82.2] |
| **head trail6, causal** | **66.5 [63.4, 69.7]** | 57.3 [50.9, 63.9] | **69.4 [66.4, 72.7]** |
| timeout, oracle enrollment, label arming (leaky) | 29.2 [26.7, 32.0] | 10.7 [6.9, 14.8] | 34.9 [31.3, 38.1] |
| head v3, oracle | 49.7 [46.7, 53.2] | 52.2 [45.9, 58.8] | 48.9 [44.9, 52.4] |
| head trail6, oracle | 30.3 [27.3, 33.1] | 21.7 [16.7, 27.2] | 33.0 [29.2, 36.4] |

2 s horizon (block A, default windows, the v1 inputs):

| system | all | floor-open | taken |
|---|---|---|---|
| timeout, causal (deployable) | 92.5 [90.8, 94.1] | 81.1 [75.8, 86.0] | 96.0 [94.4, 97.3] |
| head v3, causal | 85.9 [83.7, 88.2] | 88.1 [83.7, 92.1] | 85.2 [82.7, 87.8] |
| **head trail6, causal** | **79.8 [77.4, 82.5]** | **75.0 [69.4, 80.7]** | **81.3 [78.5, 84.0]** |
| head v3, oracle | 72.9 [70.4, 75.9] | 77.5 [72.3, 82.8] | 71.5 [67.9, 74.8] |
| head trail6, oracle | 53.8 [50.7, 56.9] | 59.9 [53.7, 66.7] | 51.8 [47.9, 55.5] |

(Block C at the stricter 2 s emission horizon: timeout 96.7, v3 93.2, trail6 92.4 %.)

Paired differences, miss points [95 % CI]:
- **trail6 − deployable timeout, causal, 6 s:** all **−8.3 [−11.7, −4.8]**, floor-open **+11.2 [+3.4, +18.6]**,
  taken −14.2 [−17.8, −10.6]. At 2 s (block A): all −12.7 [−15.6, −9.8], open −6.1 [−13.3, +0.3].
- **trail6 − v3, causal** (`runs/stage1_turn_v3_trail6.paired_vs_v3.json`): 6 s all −13.1 [−16.0, −10.3], open −23.0
  [−29.9, −16.5]. 2 s (A) all −6.1 [−8.5, −3.6], open −13.1 [−19.4, −7.0].
- trail6 − v3, oracle enrollment: 6 s −19.4 [−22.1, −16.9] (open −30.6). 2 s (A) −19.2 [−22.3, −16.3].
- Enrollment leak inside the head (causal − oracle, trail6, 6 s): +36.2 [+32.8, +39.5]. For v3: +29.9.

**v1 protocol, n = 200** (fixed dev tracks, oracle enrollment, label arming, in-sample threshold;
`runs/archive/stage1_turn_v3_trail6_eval_n200.json`). The v3 column is the fixed-cache rerun of v3
(`runs/archive/stage1_turn_v3_eval_n200_fixedtracks.json`; hybrid from `runs/turn_v3_hybrid_n200.json`, artifacted cache).

| system (P50 / miss at ≤ 5 % FC) | v3 | trail6 |
|---|---|---|
| head + oracle activity | 560 ms / 5.8 % | 1040 ms / 12.1 % |
| head + Sortformer offline | inf / 57.4 % | 1920 ms / 45.8 % |
| head + Sortformer streaming | inf / 62.1 % | 2800 ms / 47.9 % |
| timeout on streaming track | 2640 ms / 38.4 % | 2640 ms / 38.4 % |
| timeout on oracle primary | 1440 ms / 6.3 % | 1440 ms / 6.3 % |
| hybrid (head OR timeout), oracle / streaming | 560 / 1.6 %, 2640 / 37.4 % | 1040 / 4.2 %, 2640 / 38.4 % |

**Reading.** Training on 6 s trails fixes most of what v3 lacked. On the causal streaming track at 6 s the head now
beats the deployable timeout overall (66.5 vs 74.8 %, −8.3 points, CI excludes 0). It is also better than v3 at both
horizons (−13 points at 6 s, −6 at 2 s), and on open ends it gains 23 points over v3. But on floor-open ends, the case
where an agent should reply, it still loses to the timeout: 57.3 vs 46.1 %, +11.2 [+3.4, +18.6]. So "long silence ⇒
end" is only partly learned. The gain comes mostly from taken ends (−14 points), where the other columns carry the
signal. With oracle enrollment the head now matches the leaky label-armed timeout (30.3 vs 29.2 %). The causal − oracle
gap grows to 36 points, so enrollment noise, not the horizon, is now the dominant loss. There is a price on the v1 /
clean path: with oracle activity at a 2 s window, P50 goes 560 → 1040 ms and miss 5.8 → 12.1 %. The head waits longer
before firing because in training long silences are no longer the end of the window. On the stream at n = 200 it is
still behind the timeout (47.9 vs 38.4 %). Next steps: bias the open-floor case (a silence-duration prior or mixing 2 s
and 6 s trails), or a hybrid with the timeout at the 6 s horizon.

Commands:
```
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 TMPDIR=$S .venv/bin/python scripts/make_sortformer_tracks.py \
    --split train --track-source stream --device mps --trail-sec 6 --window-sec 20 --budget 540   # x4; then --verify-cpu 3
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 .venv/bin/python -m audioforge.cli train \
    research/recipes/stage1_turn_v3_trail6.yaml -o runs/stage1_turn_v3_trail6.afm
# W = a work dir whose tracks/ holds the 974 extended-window dev tracks (reused, not regenerated)
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage scores --v2-work $W --v2-tag trail6 --v2-budget 500 --threads 2 \
    --v2-bindings oracle,causal_dominant,oracle@2s,causal_dominant@2s --ckpt runs/stage1_turn_v3_trail6.afm   # x2
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage report --v2-work $W --v2-tag trail6 --n-boot 1000 \
    --ckpt runs/stage1_turn_v3_trail6.afm --out runs/archive/turn_v3_trail6_eval_leakfree.json
.venv/bin/python scripts/eval_stage1.py --tasks turn --n 200 --ckpt runs/stage1_turn_v3_trail6.afm \
    --diar-ckpt runs/nemo_sortformer_v2.afm --diar-cache data/ami/cache/sortformer/dev --hybrid \
    --out runs/archive/stage1_turn_v3_trail6_eval_n200.json
```
