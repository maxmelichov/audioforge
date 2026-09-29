# Release checklist (before the repository goes public)

Status legend: [x] done, [ ] open, [?] needs a decision from the maintainer.

## 1. Licence

- [?] **Choose the code licence.** `LICENSE` currently holds Apache-2.0 and `pyproject.toml` says
  `license = "Apache-2.0"`; both were added by an assistant and have not been confirmed. Alternatives that fit the
  dependencies equally well: MIT, BSD-3-Clause. Whatever is chosen must be compatible with the vendored TurnBench
  scorer (`integrations/turnbench_scorer/`, MIT) and must not claim the model weights (see below). If you change
  it, update `LICENSE`, `NOTICE` (first paragraph), `pyproject.toml` (`license`), `packages/audioforge-client/LICENSE`
  and its `pyproject.toml`.
- [x] `NOTICE` lists every third-party licence: NVIDIA weights (CC-BY-4.0 for the served encoder, Sortformer v2,
  Parakeet, TitaNet; OpenMDW-1.1 for Nemotron-3-Diarization; NVIDIA Open Model License for the 0.6B streaming
  encoder, Parakeet-EOU, Sortformer v2.1; NGC terms for AmberNet), Silero (MIT), TurnBench scorer (MIT), the
  LibriSpeech example clip (CC-BY-4.0). Source: `research/FINAL_REPORT.md` section 9 and `docs/MODELS.md`.
- [ ] `assets/served_heads_v0.1.pt` (19.6 MB) is this project's trained tensors only; it is redistributed with the
  code. Confirm it should carry the code licence (the heads were trained on AMI / ICSI / LibriSpeech features on top
  of the frozen NVIDIA encoder; ICSI is LDC-licensed, the weights themselves are not ICSI data).

## 2. Secrets and personal data

- [ ] **Rotate the Hugging Face token** that was used on this machine during the project
  (`~/.cache/huggingface/token`), regardless of the scan below: it was in the environment of many scripted runs.
- [ ] Secrets scan on the tracked tree, must print nothing (run from the repo root):
  ```bash
  git grep -nE 'hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}' -- . ':!research/raw'
  git grep -nIoE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}' -- . ':!research/raw' ':!research/papers' | grep -v 'noreply@\|example\.\|@gmail.com$'
  git log --format='%ae %ce' | sort -u          # author / committer e-mails that will become public
  ```
  Last run 2026-09-28: no tokens in tracked files; no personal e-mail addresses in tracked file contents. The commit
  metadata carries the author e-mail; if it should not be public, rewrite history before pushing (or accept it).
- [?] **Author e-mail in the commit metadata** (2026-09-28): `git log --format='%ae' | sort | uniq -c` shows two
  author addresses (171 commits under one, 1 under another). Decide which may be public; if either may not, rewrite
  the history (`git filter-repo --mailmap`) before the first push.
- [ ] Untracked files are not scanned by `git grep`. Before the first push run the same greps with `grep -rn` over
  the working tree minus `data/ .venv/ models/`, or push from a clean clone.
- [ ] `research/raw/` holds Hugging Face API dumps (model cards and metadata, public data). Check nothing personal was
  captured in `research/raw/all.json` (author handles are public HF usernames).

## 3. Large files

- [ ] Nothing over 20 MB tracked. Check:
  ```bash
  git ls-files -z | xargs -0 du -k | sort -rn | head -20
  git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '$1=="blob" && $3 > 10000000' | sort -k3 -rn
  ```
  Last run 2026-09-28: largest tracked file `assets/served_heads_v0.1.pt` 19.2 MB (2026-09-29: + `assets/lid_distill.pt` 3.7 MB, `assets/tsvad_spk.pt` 1.0 MB), then `runs/*.json` under 3.2 MB.
  The second command also lists blobs that were deleted but remain in history.
- [x] `data/`, `models/`, `runs/*.afm|pt|onnx`, `*.log`, `demo/clips`, `demo/events`, renders and scratch venvs are
  in `.gitignore` (`runs/*.log` files committed before 2026-09-28 stay tracked).
- [ ] GitHub rejects pushes with files over 100 MB and warns at 50 MB; if history contains any (it should not: the
  `.afm` checkpoints were never committed), use `git filter-repo` before the first push.
- [ ] `assets/served_heads_v0.1.pt` should also be attached to the `v0.1.0` GitHub release: `audioforge/hub.py`
  falls back to `HEADS_URL` (…/releases/download/v0.1.0/served_heads_v0.1.pt) when the file is not in the checkout
  (i.e. for wheel installs).
- [ ] **Single-model mode is the default (2026-09-29): attach `assets/tsvad_spk.pt` (1.0 MB, sha256 `dbc6230d…`) and
  `assets/lid_distill.pt` (3.7 MB, sha256 `07de4e4d…`) to the same `v0.1.0` release.** `audioforge-download` fetches
  them from `RELEASE_URL` when they are not in the checkout, and fails with one line if the download fails. Check the
  hashes against `audioforge/hub.py` `COMPONENTS` after uploading. `lid_distill.pt` was trained on AmberNet's
  outputs (NGC Terms of Use): confirm redistribution is allowed (research/LID.md) before attaching it.
- [ ] Docs say "stored 5 s of clean speech, 10 s for meetings; live grabs are not enough" wherever the default mode
  is introduced: README (voice sample, quickstart), docs/CONFIGURATION.md §1 / §13, docs/PROTOCOL.md §4.5,
  docs/ARCHITECTURE.md. The quickstart output in the README was captured with the default mode on 2026-09-29.

## 4. Code and docs

- [x] (2026-09-28) The workstreams listed below were committed; the tree was clean at 40c7cc4 before the final-report
  refresh. The item is kept for the record:
- [ ] ~~Other agents' uncommitted work at the time of this checklist~~ (`audioforge/serve.py`, `final_asr.py`,
  `modules/relpos.py`, `streaming_diar.py`, the integrations, `tests/test_serve_shipped.py`, and the untracked
  `uc_stream.py`, `perf.py`, `heads/uc_turn.py`, `scripts/research/uc_turn.py`, `scripts/research/bench_serve.py`,
  `scripts/frontier_*.py`, `demo/`, `docs/UPSTREAM_PRS.md`, `research/HYBRID_ASR.md`, `research/IMPROVE_115M.md`,
  `runs/improve_115m.json`, `tests/test_bulletproof.py`, `tests/test_uc_*.py`) must be committed or dropped.
  `ruff check` currently reports 17 F401/F841 findings, all in those untracked files; CI will fail until they are
  fixed (`ruff check --fix` handles 13 of them).
- [x] (2026-09-29) `bench_serve.py`, `frontier_*.py`, `uc_turn.py` (and the other research drivers and drafts) are in
  `scripts/research/`; the layout changes are listed in CHANGELOG.md.
- [ ] The `Homepage` / `Source` URLs in `pyproject.toml` and `HEADS_URL` in `audioforge/hub.py` assume the repository
  name `maxmelichov/audioforge`; change them if the public name differs.
- [ ] **Upstream PRs**: `docs/UPSTREAM_PRS.md` holds ready-to-paste PRs for Pipecat and LiveKit Agents on the user's
  forks; **no PR has been opened**. Order: publish `audioforge-client` to PyPI, `uv lock` in both forks, then open
  the PRs. Decide before the repository goes public whether they are opened now or later.
- [x] Full test suite run once through the gate on 2026-09-28: 718 passed, 5 skipped, 7 failed, the 7 all in the
  untracked `tests/test_bulletproof.py` (recorded in CHANGELOG.md). Re-run after the other workstreams commit.
- [ ] Tag `v0.1.0`, create the GitHub release with `assets/served_heads_v0.2.pt` (ships), `assets/served_heads_v0.1.pt`
  (the measured build), `assets/tsvad_spk.pt` and `assets/lid_distill.pt` attached, then (optionally)
  `python -m build` and upload `audioforge` and `packages/audioforge-client` to PyPI.

## 5. Not for the release

- Never commit `data/`, `runs/*.afm`, `runs/*.pt` or anything under `models/`; they are reproducible via
  `audioforge-download` and `scripts/research/prepare_*.py`.
- The Hugging Face cache (`~/.cache/huggingface`) is not part of the repository and is never written by the
  release tooling.

## 6. Open items after the 2026-09-28 final-report refresh

- [?] **Licence confirmation** (section 1): Apache-2.0 in `LICENSE` / `pyproject.toml` is a placeholder until the
  maintainer confirms it.
- [?] **Author e-mail** in the commit metadata (section 2).
- [ ] **Rotate the Hugging Face token** (section 2).
- [ ] **Upstream PRs** (section 4): prepared, not opened.
- [ ] **GPU distillation run**: `scripts/research/distill_0p6b_to_115m/` (README: steps, `setup.sh`, `preflight.sh`),
  ~6 A100 hours; bring back `runs/distill_0p6b.json`. Not required for the release; it is the next research step.
- [ ] **Watch and listen to the demo v4 render** (`showcase_v4.mp4`, `showcase_v4_square.mp4` on the external SSD)
  end to end before posting `demo/linkedin_post.md`; the narration was only checked by Whisper. The post's point 2
  still says the TS-VAD path is "not served yet": it has been served since 2026-09-28 (behind flags; live result in
  `research/FINAL_REPORT.md` §5.6), so that sentence needs an update before posting.
- [ ] Optional before release: the ~5 h CPU Sortformer-track rebuild (`scripts/research/tsvad_chain.sh`) for the
  paired TS-VAD CIs (FINAL_REPORT §11).
- [x] `research/FINAL_REPORT.md` / `.html`, README scorecard and CHANGELOG refreshed on 2026-09-28; the report's numbers read
  back from their JSON (`research/VERIFICATION_2026-09-28.md`); ruff and the fast CI subset run once through
  `scripts/dev/gate.sh` (result in CHANGELOG.md).
