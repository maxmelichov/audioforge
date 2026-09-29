# Training recipes

YAML recipes for `audioforge train <recipe> -o out.afm [key=value ...]` (`audioforge/train.py`). Nothing here is
needed to run the server: `audioforge-download` rebuilds the served model from NVIDIA's weights plus
`assets/served_heads_v0.1.pt`. Each recipe's header comment says what it is for and which note reports it.

Moved from `recipes/` to `research/recipes/` on 2026-09-29; the provenance fields inside `runs/*.json` still cite the
old `recipes/<name>.yaml` paths.

| group | recipes |
|---|---|
| Reproductions of NVIDIA designs (synthetic data, minutes on a laptop) | `parakeet_tdt_ctc`, `nemotron_streaming_rnnt`, `canary_aed`, `sortformer_diar`, `streaming_sortformer` |
| New designs from the same parts (synthetic data) | `voice_agent_frontend`, `speaker_attributed_asr`, `codec_token_enhancer`, `speaker_aware_turn`; `voice_agent_frontend_librispeech` on real speech |
| The served heads on NVIDIA's frozen 115M encoder | `stage1_heads_pretrained` -> `stage1_turn_v3` -> `stage1_turn_v3_trail6` (the served turn model) + `spk_relational_titanet` (the served speaker head); combined by `scripts/research/make_served_model.py` |
| Experiments (reported in the named note; not served) | `stage1_turn_on_sortformer`, `stage1_turn_v3_ami_icsi`, `stage1_turn_v4_kernels`, `stage1_turn_v5_headonly`, `stage1_turn_dyadic*`, `stage1_completeness`, `stage1_rnnt_yieldhold*`, `stage2_unfreeze_pretrained`, `spk_layer_sweep_*`, `spk_distill_titanet*`, `diar_layer_route`, `turn_layer_route_mid`, `diar_distill_nemotron3`, `lid_head`, `lid_aug`, `asr_meeting_decoder_adapt` |

`scripts/research/run_all.sh` trains every synthetic recipe at laptop scale into `runs/`.
