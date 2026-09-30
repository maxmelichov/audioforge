# demo/

Launch images for audioforge, built from committed run files (`runs/*.json`); no number is typed by hand.

| path | what it is |
|---|---|
| `images/*_v8*.png`, `images/*_v9*.png` | the current architecture and results images (plus `_square` and `_notext` variants) |
| `images/EXPLAINER.md` | what each current image shows and where every number comes from |
| `images/redesign/` | the image pipeline: HTML/CSS pages, `export_single.py` (reads `runs/*.json`), `render.py` (Playwright) |
| `images/DESIGN_NOTES.md` | design notes the redesign pages cite |
| `v5/shots/numbers_b.{json,js}`, `v5/shots/export_b.py`, `v5/shots_b.json` | the earlier number set that `images/redesign/` still reads (`export_single.py`, `render.py`, `results.html`) |
| `archive/` | earlier showcase-video pipelines (PIL renderers v1-v4, the v5 HTML film, narration TTS, scripts), older image versions and stills, kept for reference; the old `README.md` there documents them |

Renders go to `$DEMO_OUT` (default `/Volumes/ExternalSSD/nvidia-audio-models/demo_out`), not to the repository.
