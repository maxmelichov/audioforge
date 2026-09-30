# Layer routing for the diarization and turn heads

2026-09-27. research/archive/SPK_HEAD.md found that the speaker head read a near-uniform mix of all 17 encoder blocks, and
that reading block 4 alone halved its within-meeting EER. Speaker identity lives in blocks 1-9 and is at chance from
block 10 up. The served diarization head (`runs/stage1_served.afm`, `heads.diar`, `from_layers: all`) also reads a
flat mix: 0.048-0.079 per block, uniform = 0.059. This note asks two questions:

1. **Diarization.** Does the diar head get better if it reads single blocks 2, 4, 6, 8 or 12, or a learned mix
   restricted to blocks 1-9?
2. **Turn head.** trail6's turn head reads the top layer of the speaker-conditioned encoder pass. Does a learned mix
   of blocks 4-12 of that same pass make it confident earlier?

Code: `scripts/layer_routing.py` (stages train / diar_eval / turn_pair / turnbench / report). Recipes:
`research/recipes/diar_layer_route.yaml` and `research/recipes/turn_layer_route_mid.yaml`. Tests: `tests/test_layer_routing.py`.
Numbers: `runs/layer_routing.json`.

## Code changes

- `audioforge/train.py` `init_from_afm`: a `layer_mix.<head>` whose size changed (`all` -> `[0..8]`) is dropped and
  re-initialised to a uniform mix. Before, only a mix that disappeared was dropped, and a re-sized one failed to load.
- `audioforge/model.py`:
  - `SpeechModel.forward` now skips a weight-0 head before it runs the speaker-conditioned encoder pass. Before, a
    frozen turn head at weight 0 still paid for a second encoder pass on every batch that carried `spk_act`.
  - `from_layers` on a `condition_on_speaker` head now reads its tap or mix of the **conditioned** pass, in training
    and through the new `SpeechModel.cond_head_input`. `heads/turn.py turn_scores` and
    `eval_stage1.turn_scores_given_act` use it (the latter covers eot-bench v2 and TurnBench).
  - Without `from_layers`, every path is exactly the old `encode(..., spk_act=...)[0]`. A test confirms that a
    one-block tap of the top block reproduces the untapped scores exactly.
- `audioforge/serve.py`: raises `NotImplementedError` for a speaker-conditioned turn head with `from_layers`. The
  streaming kernel path reads the top layer only, so such a head would otherwise be served wrongly and silently. No
  such head is recommended (below).

## Part 1: diarization head

**Setup.**
- **Init and training.** Head-only retrains from `runs/stage1_served.afm`: `init.train_only: [heads.diar,
  layer_mix.diar]`, every other head at weight 0, no conditioning. The driver asserts that the other 704 tensors are
  bit-identical after training.
- **Data and objective.** The same as stage 1's diar head (`research/recipes/stage1_heads_pretrained.yaml`): Sortformer
  PIL+BCE with the prefix loss (p 0.5), on synthetic conversations + AMI turn windows + AMI diar windows (16 s),
  single-source batches. 1000 steps, batch 6, lr 1e-3, cosine schedule, seed 0, about 4-5 min per run on MPS.
- **Warm start.** The head's weights start from the served diar head in every variant.
  - The control (`diar_all17`, `from_layers: all`) also keeps its learned mix.
  - A single tap drops the mix.
  - The blocks 1-9 mix starts uniform.
  - This favours the control, whose head was already trained on its own input.
- **Evaluation.** `diar_eval` runs the BASELINES.md diarization code on the same 64 AMI dev windows of 20 s
  (`bench_sd_baselines.diar_windows` / `model_pass` / `pooled_der` / `pyannote_der`). It gives pooled frame DER with
  `eval_stage1.der_parts` (80 ms, threshold 0.5, no collar, overlap scored) and pyannote.metrics DER at collar 0 /
  0.25 / 0.5, where pyannote's collar is the total width, so 0.5 = ±0.25 s.
- **Held-out corpus.** ICSI dev: 64 windows of 20 s from Bmr021 / Bns001 (seeded cap, windows with more than 4
  speakers dropped). Never trained on.
- **Paired differences.** Window bootstraps (2000) of the pooled DER.
- **Reproduction check.** The served row reproduces BASELINES exactly: 0.3937 (.2456 / .0590 / .0891), pyannote 0.394
  / 0.346 / 0.312. Sortformer v2 gives 0.2014.
- **Kill line (pre-registered).** A tap must beat the all-17 control by ≥ 0.03 DER on **both** AMI and ICSI.

| variant | input | AMI frame DER (miss / FA / conf) | AMI pyannote c0 / c0.25 / c0.5 | ICSI frame DER (miss / FA / conf) | ICSI pyannote c0 / c0.25 / c0.5 | Δ vs control, AMI [95 % CI] | Δ vs control, ICSI [95 % CI] |
|---|---|---|---|---|---|---|---|
| served stage-1 head (no retrain) | all 17, learned | 0.394 (.246 / .059 / .089) | 0.394 / 0.346 / 0.312 | 0.354 (.079 / .209 / .066) | 0.354 / 0.241 / 0.170 | +0.012 [-0.004, +0.030] | +0.001 [-0.019, +0.024] |
| **control: all 17, learned** | all 17 | 0.382 (.235 / .067 / .079) | 0.382 / 0.331 / 0.292 | 0.353 (.071 / .221 / .061) | 0.353 / 0.240 / 0.171 | - | - |
| block 2 | block 2 | 0.398 (.225 / .059 / .114) | 0.398 / 0.349 / 0.310 | 0.338 (.053 / .198 / .086) | 0.338 / 0.222 / 0.152 | +0.017 [-0.010, +0.045] | -0.015 [-0.043, +0.013] |
| block 4 | block 4 | 0.374 (.214 / .060 / .099) | 0.374 / 0.325 / 0.288 | 0.348 (.070 / .203 / .075) | 0.348 / 0.236 / 0.169 | -0.008 [-0.022, +0.009] | -0.005 [-0.025, +0.016] |
| **block 6** | block 6 | **0.364** (.236 / .063 / .065) | 0.364 / 0.311 / 0.270 | **0.336** (.077 / .205 / .053) | 0.336 / 0.222 / 0.154 | **-0.017** [-0.031, -0.002] | **-0.018** [-0.036, +0.000] |
| block 8 | block 8 | 0.380 (.243 / .062 / .075) | 0.380 / 0.328 / 0.287 | 0.352 (.080 / .210 / .062) | 0.352 / 0.240 / 0.168 | -0.001 [-0.014, +0.012] | -0.001 [-0.017, +0.017] |
| block 12 | block 12 | 0.425 (.277 / .061 / .088) | 0.425 / 0.377 / 0.339 | 0.388 (.118 / .206 / .064) | 0.388 / 0.275 / 0.201 | +0.044 [+0.024, +0.066] | +0.035 [+0.009, +0.062] |
| mix of blocks 1-9 | 1-9, learned | 0.381 (.235 / .068 / .077) | 0.381 / 0.328 / 0.289 | 0.353 (.076 / .215 / .061) | 0.353 / 0.240 / 0.171 | -0.001 [-0.010, +0.009] | -0.001 [-0.013, +0.013] |
| *seed 1:* control | all 17 | 0.381 (.238 / .065 / .078) | 0.381 / 0.331 / 0.293 | 0.348 (.072 / .214 / .062) | 0.348 / 0.236 / 0.167 | -0.001 [-0.009, +0.008] | -0.005 [-0.020, +0.010] |
| *seed 1:* block 6 | block 6 | 0.361 (.232 / .060 / .069) | 0.361 / 0.308 / 0.268 | 0.318 (.067 / .202 / .049) | 0.318 / 0.204 / 0.134 | -0.019 [-0.036, -0.003] vs seed-1 control | -0.030 [-0.054, -0.010] vs seed-1 control |
| NVIDIA Sortformer v2 (reference) | own encoder | 0.201 (.140 / .045 / .016) | 0.201 / 0.167 / 0.159 | 0.209 (.011 / .191 / .007) | 0.209 / 0.096 / 0.042 | -0.180 [-0.233, -0.126] | -0.144 [-0.175, -0.115] |

- The seed-1 rows repeat the control and the best tap with a different data order, dropout and prefix draws. The
  control moved by 0.001 / 0.005, so training noise is about 0.005.
- Hypothesised speakers per window on AMI: 1.8-2.0 for every variant of ours, against 2.62 in the reference and 2.52
  for Sortformer. Our head under-counts speakers whichever block it reads.
- ICSI's FA is about 0.2 for every system, Sortformer included. The ICSI word-level reference leaves speech
  unlabelled, so compare ICSI rows with each other, not with AMI.

**Learned layer weights** (softmax over blocks 1-17):
- served stage-1 head: 0.054 0.054 0.063 0.067 0.072 0.079 0.074 0.069 0.063 0.053 0.050 0.049 0.049 0.048 0.049
  0.050 0.059
- control after 1000 more steps: 0.049 0.050 0.064 0.071 0.080 **0.092** 0.086 0.075 0.062 0.048 0.044 0.043 0.043
  0.043 0.045 0.045 0.059 (seed 1: the same to ±0.001)
- mix of blocks 1-9: 0.095 0.094 0.100 0.103 0.115 **0.125 0.128** 0.121 0.120

Both learned mixes lean toward blocks 5-7 and stay far from one-hot. The single-tap rows agree: block 6 is the best
input. The blocks 1-9 mix, although it starts uniform, ends where the all-17 control does (0.381 vs 0.382). Dropping
blocks 10-17 alone changes nothing.

**Verdict (Part 1): kill line not met; no tap is adopted.**
- **Block 6 is the best tap, and its gain is real but small.** It is −0.017 / −0.018 (seed 0) and −0.019 / −0.030
  (seed 1) on AMI / ICSI. The seed average is −0.019 on AMI and −0.023 on ICSI, with the AMI CIs excluding 0 in both
  seeds.
- **It misses the pre-registered bar on AMI in both seeds** (≥ 0.03 on both corpora).
- **Block 4 and block 8 tie the control. Block 2 trades AMI for ICSI.** On AMI its confusion goes up (.114 vs .079).
- **Block 12 is clearly worse** (+0.044 / +0.035), mostly from miss: the speaker-blind top half hurts.
- So the result does not repeat the speaker head's story. The diar head's error is not "reading the wrong layers":
  - Its learned mix already down-weights blocks 10-17.
  - It fails by missing speech (miss 0.21-0.24 on AMI vs Sortformer's 0.14) and by under-counting speakers.
  - Neither of those is fixed by routing.
- **The gap to Sortformer v2 (0.16-0.18 DER) is about 10x the best routing gain.**

**Turn head on our diar track: not worth it.** The best diar head is 0.361-0.364 on AMI, against 0.201 for
Sortformer v2: 0.16 behind, far outside the 0.05 gate. We did not run eot-bench v2 with it as the track for trail6
(`causal_dominant` binding). Sortformer v2 stays the product diarizer and the turn stack's track (as in BASELINES.md).

**Transplant recommendation: no.** Keep `heads.diar` in `runs/stage1_served.afm` as it is.
- Block 6 would buy about 0.02 DER. That is below the pre-registered bar, on a head the product does not use.
- A single-tap diar head is also untested in `StreamingDiarizer`. Its assert only covers mixed heads, and the
  streaming path feeds the top layer.
- No `runs/stage1_diar_<tap>.afm` is saved, because no tap passed. The block-6 checkpoints (seed 0 and 1) are in the
  scratch dir if anyone wants the +0.02.

## Part 2: turn head input tap

**Setup.**
- **The tapped head.** `research/recipes/turn_layer_route_mid.yaml` is the trail6 recipe with one change: `heads.turn.from_layers:
  [3..11]`, a learned softmax mix of blocks 4-12 of the **speaker-conditioned** pass, starting uniform. The kernels at
  encoder layers [0, 2], the text path (RNNT PredictionNet on the conditioned top layer) and the conditioning are
  unchanged: p_ext 0.9 on streaming Sortformer tracks, ext_noise.
- **Init and training.** Head-only (`heads.turn` + `layer_mix.turn`), warm-started from `runs/stage1_served.afm`.
  Its turn head is trail6 (`runs/served_model_build.json`). The schedule is trail6's own: 2000 steps, batch 6, lr
  1e-3, the same data (AMI 20 s turn windows with 6 s trails + synthetic conversations at 0.3). 20 min on MPS, so the
  full schedule fit.
- **Top-layer control (`turn_topctl`).** The same recipe with `from_layers: null`, the same warm start and steps: it is
  trail6 trained 2000 more steps. It separates "which input" from "more training". The driver asserted 746 tensors
  bit-identical in both runs.
- **Learned weights (blocks 4-12):** 0.107 0.103 0.097 0.098 0.100 0.110 0.118 0.126 **0.141**. The mix drifts
  toward block 12, the top of the allowed range.

### eot-bench v2, AMI dev (EOT_BENCH_V2.md §7 protocol)

n = 974, causal_dominant binding, 1.04 s streaming Sortformer tracks. The threshold is cross-fitted at ≤ 5 % per-turn
FC (2-fold leave-meetings-out); the paired bootstraps use 1000 resamples. The trail6 scores are the committed §7 ones
(same work dir), and they reproduce §7: 79.8 / 66.5 / 57.3.

"P50 fired" is the median latency over turns that fire in time. P50 over all turns (with misses as ∞) is ∞ for every
row, because miss > 50 %.

| system | block A 2 s: all / open / taken miss % | A: FC held-out | A: P50 fired | block C 6 s: all / open / taken miss % | C 6 s: P50 fired | block C 2 s: all / open |
|---|---|---|---|---|---|---|
| trail6 (served) | 79.8 / 75.0 / 81.3 | 5.3 % | 2240 ms | 66.5 / 57.3 / 69.4 | 2640 ms | 92.4 / 92.3 |
| top-layer control (+2000 steps) | 83.4 / 78.1 / 85.1 | 4.6 % | 2160 ms | 70.5 / 61.9 / 73.1 | 2640 ms | 92.9 / 91.7 |
| **mid mix, blocks 4-12** | 82.1 / 84.6 / 81.3 | 4.8 % | 2240 ms | 66.9 / 62.7 / 68.1 | 2720 ms | 94.1 / 95.0 |

Paired miss differences in points [95 % CI]:

| comparison | A 2 s all | A 2 s open | A 2 s taken | C 6 s all | C 6 s open | C 6 s taken | C 2 s all |
|---|---|---|---|---|---|---|---|
| mid − trail6 | +2.3 [+0.1, +4.6] | **+9.6 [+4.2, +14.6]** | −0.0 [−2.7, +2.4] | +0.4 [−2.2, +2.7] | **+5.5 [+0.6, +10.0]** | −1.2 [−4.0, +1.6] | +1.7 [+0.1, +3.2] |
| control − trail6 | +3.6 [+1.5, +5.6] | +3.1 [−1.4, +6.9] | +3.7 [+1.5, +5.8] | +4.0 [+1.9, +6.1] | +4.7 [+0.3, +9.2] | +3.8 [+1.3, +6.2] | +0.5 [−1.0, +2.0] |
| mid − control | −1.3 [−3.7, +1.1] | +6.5 [+1.1, +11.6] | −3.8 [−6.4, −1.3] | **−3.6 [−6.0, −1.2]** | +0.8 [−4.1, +5.8] | −5.0 [−7.7, −2.1] | +1.2 [−0.3, +2.8] |

### TurnBench dev (scripts/bench_turnbench.py, official scorer)

38 two-channel conversations, the mixed mono with channel Silero tracks as `spk_act` / cols. The dev operating point
is the maximum recall at fp_rate ≤ 0.10 (their rule). The hybrid is head OR Silero per channel. trail6 =
`runs/turnbench_dev_trail6.json`, whose head scores are CPU. The new heads were scored on MPS, in the same work dir
and with the same sweep code (`--stage sweep --tag`). There are no CIs: the official scorer gives none.

| system | head alone at the dev op-point: recall / fp / P50 | hybrid at the dev op-point: recall / fp / P50 | head, matched fp 0.03: recall / P50 | fp 0.05: recall / P50 | fp 0.07: recall / P50 |
|---|---|---|---|---|---|
| trail6 | 0.752 / 0.099 / 1058 ms | 0.835 / 0.080 / 1390 ms | 0.581 / 1689 ms | 0.672 / 1447 ms | 0.727 / 1238 ms |
| top-layer control | 0.704 / 0.093 / 1194 ms | 0.838 / 0.085 / 1378 ms | 0.413 / 1690 ms | 0.580 / 1544 ms | 0.672 / 1288 ms |
| mid mix, blocks 4-12 | 0.746 / 0.070 / 1154 ms | 0.825 / 0.074 / 1396 ms | 0.604 / 1594 ms | 0.689 / 1377 ms | 0.746 / 1152 ms |

The matched-fp columns interpolate each head's threshold curve linearly (`interp_matched_fp` in the json). The
mid-mix curve's highest-fp point is 0.070: no threshold in its quantile grid lands between 0.07 and 0.10. That is why
its dev op-point sits at fp 0.070.

### Verdict (Part 2): no

**Does the mid-block tap make the head confident earlier?** Not in a way that survives both benchmarks.
- **TurnBench.** At matched fp the mid-mix head fires about 70-95 ms earlier than trail6 (P50 1594 / 1377 / 1152 vs
  1689 / 1447 / 1238 ms at fp 0.03 / 0.05 / 0.07), with 0.02 more recall. Against the equally trained top-layer
  control it is 95-165 ms earlier, with 0.07-0.19 more recall.
  - That is less than one 160 ms encoder chunk.
  - The official scorer gives no CI.
  - At the dev op-point the hybrid, the deployable form, is unchanged: 0.825 / 1396 ms vs 0.835 / 1390 ms.
- **eot-bench v2 (AMI).** There is no latency gain: P50 among fired turns is 2240 vs 2240 ms (A) and 2720 vs 2640 ms
  (C 6 s). The tap costs floor-open recall, the case where an agent should reply: +9.6 [+4.2, +14.6] points of open
  miss at 2 s and +5.5 [+0.6, +10.0] at 6 s vs trail6. Overall it ties trail6 at 6 s (+0.4) and is worse at 2 s
  (+2.3, +1.7).
- **More training at the top layer hurts** (control − trail6: +3.6 to +4.0 points, CIs exclude 0). trail6's 2000
  steps were already enough, and 2000 more overfit the AMI train turns.
- **Against that control, the mid mix is better on taken ends** (−5.0 at 6 s) **and worse on open ends at 2 s**
  (+6.5). Mid blocks carry more "someone else is talking" (speaker / acoustic) information. That helps a taken
  floor, not an open one.
- **The learned mix itself drifts toward block 12:** the objective wants the top of the allowed range.

**Transplant recommendation: no.** Keep trail6's top-layer input in `runs/stage1_served.afm`.
- The floor-open regression on AMI outweighs a sub-chunk TurnBench latency shift that the hybrid does not keep.
- Serving a tapped, speaker-conditioned turn head would also need the streaming kernel path in `audioforge/serve.py`
  wired to per-layer outputs (it now refuses such a head).
- No `runs/stage1_turn_<tap>.afm` is saved. The mid-mix and control checkpoints are in the scratch dir.

## Commands

```bash
S=<scratch>/lr; export PYTHONPATH=. TMPDIR=$S
# Part 1 (MPS, one job at a time; ~4-5 min each). Taps: heads.diar.from_layers=[1] [3] [5] [7] [11] [0,1,2,3,4,5,6,7,8]
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 .venv/bin/python scripts/layer_routing.py train \
    --recipe research/recipes/diar_layer_route.yaml --out $S/diar_b6.afm --tag diar_b6 name=audioforge_train_diar_b6 "heads.diar.from_layers=[5]"
.venv/bin/python scripts/layer_routing.py diar_eval --ckpt $S/diar_b6.afm --tag diar_b6        # CPU, 2 threads, ~90 s
.venv/bin/python scripts/layer_routing.py diar_eval --ckpt runs/stage1_served.afm --tag served_stage1
.venv/bin/python scripts/layer_routing.py diar_eval --ckpt runs/nemo_sortformer_v2.afm --tag sortformer_v2
.venv/bin/python scripts/layer_routing.py report [--control diar_all17_s1]
# Part 2 (MPS ~20 min each; the control adds heads.turn.from_layers=null)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 .venv/bin/python scripts/layer_routing.py train \
    --recipe research/recipes/turn_layer_route_mid.yaml --out $S/turn_mid4to12.afm --tag turn_mid4to12 --section turn --head turn \
    --trained heads.turn.,layer_mix.turn name=audioforge_train_turn_mid4to12
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage scores --v2-work <scratch>/trail6/work --v2-tag turn_mid4to12 \
    --v2-budget 530 --v2-bindings causal_dominant,causal_dominant@2s --ckpt $S/turn_mid4to12.afm   # repeat until 0 left
.venv/bin/python scripts/layer_routing.py turn_pair --work <scratch>/trail6/work --tags turn_mid4to12,turn_topctl
.venv/bin/python scripts/layer_routing.py turn_pair --work <scratch>/trail6/work --base turn_topctl --tags turn_mid4to12
W=<scratch>/dyadic/turnbench/work_official
.venv/bin/python scripts/bench_turnbench.py --stage head --work $W --tag turn_mid4to12 --ckpt $S/turn_mid4to12.afm --device mps
.venv/bin/python scripts/bench_turnbench.py --stage sweep --work $W --tag turn_mid4to12 --ckpt $S/turn_mid4to12.afm \
    --out $S/turnbench_turn_mid4to12.json
.venv/bin/python scripts/layer_routing.py turnbench --tags trail6=runs/turnbench_dev_trail6.json,turn_mid4to12=$S/turnbench_turn_mid4to12.json,turn_topctl=$S/turnbench_turn_topctl.json
.venv/bin/python -m pytest -q tests/test_layer_routing.py
```

The `name=audioforge_train_<tag>` override only names the process, so the machine's GPU check
(`pgrep -f "audioforge.*train"`) sees these runs.
