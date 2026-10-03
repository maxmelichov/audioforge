# repo_rules_001: the repository rules, checked mechanically

Date: 2026-10-03, main at 3fa7779. Script: `uv run plans/repo_rules/repo_rules_001.py`
(or `chore rules`). Rules from [AGENTS.md](../../AGENTS.md) and [docs/PROJECT.md](../../docs/PROJECT.md).

| check | rule | outcome on failure |
|---|---|---|
| root | the root holds only README, LICENSE, NOTICE, CHANGELOG, CITATION, CONTRIBUTING, AGENTS.md, CLAUDE.md, pyproject.toml, chorefile, .gitignore, .pre-commit-config.yaml and the top directories (tracked or untracked-not-ignored) | FAIL |
| size | no file over 700 lines; the existing ones are listed with "known" when `plans/file_size/*.md` names them | report only |
| pep723 | every `scripts/*.py`, `scripts/dev/*.py`, `plans/*/*.py` carries a `# /// script` block (`scripts/research/` is the later pass in `plans/uv_scripts/remaining.md`) | FAIL |
| chore | `chorefile` at the root; `chore list` exits 0 when chore is installed | FAIL |
| commits | no `Co-Authored-By` / `Claude-Session` trailer in commits after 4895faf | FAIL |

What was done to reach PASS (commit 3fa7779): chore 1.10.0 installed (getchore/chore, MIT, release tarball,
sha256-checked, into `~/.local/bin`; the project `.venv` untouched), `chorefile` written, PEP 723 headers on
`scripts/stream_client.py` and `scripts/dev/gen_config_doc.py`. The root needed no moves: nothing plan- or
scratch-like was tracked or untracked there (`logs/`, `*.egg-info/` and the caches are git-ignored).

## Output

```
[ok] root: 0 entries outside the allowed set
[report] size: 46 tracked text file(s) over 700 lines (46 known in plans/file_size/, 0 not in that plan)
      3419  known  scripts/research/core_0p6b_heads.py
      3094  known  research/FINAL_REPORT.html
      2358  known  research/archive/raw/cards/canary-1b-v2.md
      2144  known  audioforge/serve.py
      2101  known  scripts/research/core_0p6b_turn.py
      1935  known  scripts/research/turn_v5.py
      1656  known  scripts/research/eval_stage1.py
      1525  known  scripts/research/fixall.py
      1503  known  scripts/research/final_compare.py
      1392  known  scripts/research/e2e_final.py
      1341  known  research/archive/raw/cards/parakeet-tdt-0.6b-v3.md
      1297  known  scripts/research/tsvad.py
      1238  known  scripts/research/turn_data.py
      1206  known  demo/archive/render.py
      1199  known  demo/archive/render_v2.py
      1193  known  tests/test_bulletproof.py
      1190  known  audioforge/train.py
      1130  known  scripts/research/contamination_probe.py
      1126  known  scripts/research/lid_fix.py
      1108  known  scripts/research/eot_latency.py
      1011  known  research/FINAL_REPORT.md
       980  known  audioforge/integrations/livekit.py
       966  known  audioforge/heads/turn.py
       946  known  scripts/research/turn_v4.py
       945  known  scripts/research/hybrid_asr.py
       928  known  scripts/research/bench_turnbench_latency.py
       907  known  audioforge/conversation.py
       899  known  audioforge/streaming_diar.py
       898  known  audioforge/nemo_import.py
       821  known  docs/CONFIGURATION.md
       821  known  demo/archive/render_v4.py
       811  known  research/archive/DYADIC.md
       810  known  research/archive/EOT_BENCH_V2.md
       810  known  audioforge/datasets/dyadic.py
       797  known  tests/test_pipecat_integration.py
       795  known  scripts/research/tswer.py
       790  known  tests/test_serve.py
       770  known  tests/test_livekit_integration.py
       763  known  research/archive/raw/cards/nemotron-3.5-asr-streaming-0.6b.md
       762  known  audioforge/datasets/ami.py
       753  known  audioforge/integrations/pipecat.py
       747  known  scripts/research/single_model_distill/single_distill.py
       742  known  scripts/research/lid.py
       728  known  scripts/research/bargein.py
       721  known  audioforge/model.py
       707  known  docs/UPSTREAM_PRS.md
[ok] pep723: 7/7 scripts carry a `# /// script` block
[ok] chore: `chore list` OK, 13 tasks
[ok] commits: 0/3 commit(s) after 4895faf carry a trailer

RESULT: PASS
```
