# Turn head v3: ablation commands (recipes/stage1_turn_v3.yaml)

All arms share the recipe's data (AMI turn windows with the cached **streaming** Sortformer tracks, `ext_tracks:
{source: stream, fallback: offline}`; synthetic conversations at weight 0.3), `trainer.conditioning` (`p_ext 0.9`,
`flip 0.01`, `ext_noise p 0.3`), `init.from runs/stage1_heads_pretrained.afm` with everything but `heads.turn`
frozen, 2000 steps, batch 6. Only the TurnHead v3 flags differ. Each run is one GPU job (`-o` names the output).
With torch 2.14 a high watermark of 0.5 alone makes `model.to("mps")` fail ("invalid low watermark ratio 1.4"),
so the low ratio is set too.

| arm | act_columns | duration_feats | future_act_aux |
|---|---|---|---|
| A. duration_feats only | 1 | true | off |
| B. act_columns only | 4 | false | off |
| C. both | 4 | true | off |
| D. both + future_aux (recipe default) | 4 | true | horizons [6, 12, 25], weight 0.3 |

```bash
R=recipes/stage1_turn_v3.yaml
PY="PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 .venv/bin/python -m audioforge.cli train"
# A. duration_feats only (single primary track, counters of the kernel track)
eval $PY $R heads.turn.act_columns=1 heads.turn.duration_feats=true heads.turn.future_act_aux=null \
    -o runs/stage1_turn_v3_dur.afm 2>&1 | tee runs/stage1_turn_v3_dur.log
# B. act_columns only (all four columns + primary one-hot, no counters)
eval $PY $R heads.turn.act_columns=4 heads.turn.duration_feats=false heads.turn.future_act_aux=null \
    -o runs/stage1_turn_v3_cols.afm 2>&1 | tee runs/stage1_turn_v3_cols.log
# C. both
eval $PY $R heads.turn.act_columns=4 heads.turn.duration_feats=true heads.turn.future_act_aux=null \
    -o runs/stage1_turn_v3_both.afm 2>&1 | tee runs/stage1_turn_v3_both.log
# D. both + future_aux = the recipe as written
eval $PY $R -o runs/stage1_turn_v3.afm 2>&1 | tee runs/stage1_turn_v3.log
```

Optional controls (same override syntax):
- no track corruption: `trainer.conditioning.ext_noise=0`
- offline train tracks (the old mismatch): `'data.mix[0]'` cannot be addressed by a dotted override; copy the recipe
  and set `ext_tracks: offline`.

Evaluation of every arm (AMI dev, n = 200; always report n >= 200) runs through a research eval driver (not
published) on CPU, 2 threads, with the diarizer runs/nemo_sortformer_v2.afm and the cached dev streaming tracks in
data/ami/cache/sortformer/dev (974/974 cached; identical to recomputing them), so only the cheap offline pass is
recomputed and no resumable ~9 min streaming-diarizer pass is needed (checked: the stored
runs/stage1_turn_on_sortformer_eval_n64.json reproduces with zero differences; the stored n = 200 turn task took
362 s). Outputs go to runs/<arm>_eval_n200.json. The output adds, next to the unchanged rows, `turn_head_input`
(act_columns heads get all four columns + the enrollment column) and the non-learned
`duration_rule_sortformer_{offline,stream}` baseline.

Bars to beat on the same 200 dev turns (stage-1 results; streaming rows include the 1040 ms diarizer buffer):

| system | P50 @ <= 5 % FC | miss @ <= 5 % FC |
|---|---|---|
| timeout on the Sortformer streaming primary column (strongest non-learned rule found) | 2640 ms | 38.4 % |
| duration rule: primary silent >= k AND no other column active (streaming) | inf | 68.4 % |
| previous head (stage1_turn_on_sortformer) + Sortformer streaming | inf | 69.0 % |
| timeout on oracle primary activity (not deployable) | 1440 ms | 6.3 % |
