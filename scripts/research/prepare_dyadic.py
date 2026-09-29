"""Download the dyadic slices (research/DYADIC.md): Behavior-SD (CC BY 4.0, synthetic two-channel dialogues, one
HF tar shard) and kyutai/DailyTalkContiguous (CC BY-SA 4.0, stereo acted dialogues with word timestamps).

  .venv/bin/python scripts/research/prepare_dyadic.py --corpus behavior_sd   # test/0001.tar (~730 MB) -> data/behavior_sd/test/0001/
  .venv/bin/python scripts/research/prepare_dyadic.py --corpus dailytalk --hours 3   # first dialogues of dailytalk.jsonl (~1.7 GB)
  .venv/bin/python scripts/research/prepare_dyadic.py --corpus turnbench             # gated: needs an HF token with access (4.2 GB)
  .venv/bin/python scripts/research/prepare_dyadic.py --corpus oto --shards 23       # gated: first 23 shards (~7.4 GB of 19.4)
  .venv/bin/python scripts/research/prepare_dyadic.py --corpus <c> --labels [--n N | --hours H] [--budget 540]   # caches

Resumable (huggingface_hub, local_dir layout); records the licence text / card in <root>/LICENSE.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audioforge.datasets import dyadic as D  # noqa: E402


def behavior_sd(root: Path, shard: str):
    from huggingface_hub import hf_hub_download
    root.mkdir(parents=True, exist_ok=True)
    for f in ("README.md", "dataset_infos.json"):
        hf_hub_download(D.BSD_REPO, f, repo_type="dataset", local_dir=root)
    (root / "LICENSE.md").write_text(D.BSD_LICENSE_NOTE)
    tar = Path(hf_hub_download(D.BSD_REPO, shard, repo_type="dataset", local_dir=root))
    out = root / shard.replace(".tar", "")
    done = out / ".extracted"
    if not done.exists():
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as tf:
            tf.extractall(out, filter="data")
        done.write_text(time.strftime("%Y-%m-%d"))
    n = len(list(out.glob("*.json")))
    print(f"behavior_sd: {shard} -> {out} ({n} dialogues)")


def dailytalk(root: Path, hours: float, workers: int):
    from huggingface_hub import hf_hub_download
    root.mkdir(parents=True, exist_ok=True)
    for f in ("README.md", "dailytalk.jsonl"):
        hf_hub_download(D.DT_REPO, f, repo_type="dataset", local_dir=root)
    (root / "LICENSE.md").write_text(D.DT_LICENSE_NOTE)
    rows = [json.loads(l) for l in (root / "dailytalk.jsonl").read_text().splitlines() if l.strip()]
    sel, cum = [], 0.0
    for r in rows:  # the corpus order (dialogue 0, 1, 2 ...): a fixed prefix, no selection by content
        if cum + r["duration"] > hours * 3600:
            break
        sel.append(r)
        cum += r["duration"]
    files = [p for r in sel for p in (r["path"], r["path"].replace(".wav", ".json"))]
    t0 = time.time()

    def one(f):
        for k in range(5):
            try:
                return hf_hub_download(D.DT_REPO, f, repo_type="dataset", local_dir=root)
            except Exception as e:  # noqa: BLE001
                print(f"  {f}: {type(e).__name__}: {e} (retry {k})", flush=True)
                time.sleep(3)
        raise RuntimeError(f)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, files))
    (root / "slice.json").write_text(json.dumps({"hours": round(cum / 3600, 3), "n": len(sel),
                                                 "paths": [r["path"] for r in sel]}, indent=1))
    print(f"dailytalk: {len(sel)} dialogues, {cum / 3600:.2f} h in {time.time() - t0:.0f}s -> {root}")


def turnbench(root: Path):
    """TurnBench dev (mundo-ai/turn-benchmark-dev, gated, Dataset Public License v1.0: non-commercial, attribution,
    no voice cloning): the 3 parquet shards (4.2 GB) + LICENSE + README, as the MIT scorer's local-directory source."""
    from huggingface_hub import snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(D.TB_REPO, repo_type="dataset", local_dir=root, max_workers=3)
    (root / "LICENSE.md").write_text(D.TB_LICENSE_NOTE)
    print(f"turnbench: -> {root}", sorted(p.name for p in (root / "data").glob("*.parquet")))


def oto(root: Path, shards: int, workers: int):
    """otoSpeech-full-duplex-processed-141h (gated, CC BY 4.0; the card forbids attempts to identify speakers): the
    first ``shards`` WebDataset tar shards (~320 MB each, whole conversations per sample) -> <root>/data/train/."""
    from huggingface_hub import hf_hub_download
    root.mkdir(parents=True, exist_ok=True)
    hf_hub_download(D.OTO_REPO, "README.md", repo_type="dataset", local_dir=root)
    (root / "LICENSE.md").write_text(D.OTO_LICENSE_NOTE)
    names = [f"data/train/shard-{k:06d}.tar" for k in range(shards)]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda f: hf_hub_download(D.OTO_REPO, f, repo_type="dataset", local_dir=root), names))
    print(f"oto: {shards} shards -> {root}/data/train")


def labels(corpus: str, root, n: int | None, hours: float | None, budget: float, activity_source: str,
           cache_dtype: str = "float32", part: tuple[int, int] = (0, 1)):
    """Build the labels cache (words per channel; Silero per channel for dailytalk / oto) and the 16 kHz mix cache
    of the first ``n`` conversations (or ``hours``) of the slice, within ``budget`` seconds (resumable). ``part`` =
    (k, K): only positions i % K == k (parallel processes on disjoint conversations)."""
    import time
    t0 = time.time()
    ids = D.slice_ids(corpus, root, hours) if hours else D.list_ids(corpus, root)
    ids = ids[:n] if n else ids
    ids = [c for i, c in enumerate(ids) if i % part[1] == part[0]]
    done = 0
    for cid in ids:
        if time.time() - t0 > budget:
            break
        ds = D.Dyadic([cid], corpus, root, activity_source=activity_source, verbose=False, cache_dtype=cache_dtype)
        ds._clip(cid, 0.0, 0.1)  # builds the mix cache
        done += 1
    left = len(ids) - done
    print(f"labels {corpus}: {done} of {len(ids)} conversations ready in {time.time() - t0:.0f}s, {left} left"
          + (" (re-run to resume)" if left else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labels", action="store_true", help="build the labels + mix caches instead of downloading")
    p.add_argument("--n", type=int, default=None, help="--labels: first n conversations")
    p.add_argument("--budget", type=float, default=540.0)
    p.add_argument("--activity-source", default="silero", choices=list(D.ACTIVITY_SOURCES))
    p.add_argument("--corpus", required=True, choices=["behavior_sd", "dailytalk", "turnbench", "oto"])
    p.add_argument("--shards", type=int, default=1, help="oto: number of ~320 MB tar shards")
    p.add_argument("--root", default=None)
    p.add_argument("--shard", default=D.BSD_DEFAULT_SHARD)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--cache-dtype", default="float32", choices=["float32", "float16"], help="--labels: mix cache dtype")
    p.add_argument("--part", default="0/1", help="--labels: k/K = only conversations at positions i %% K == k")
    a = p.parse_args()
    if a.labels:
        import torch
        torch.set_num_threads(2)
        labels(a.corpus, a.root, a.n, a.hours if a.corpus != "dailytalk" or a.hours != 3.0 else None, a.budget,
               a.activity_source, a.cache_dtype, tuple(int(x) for x in a.part.split("/")))
        return
    if a.corpus == "behavior_sd":
        behavior_sd(Path(a.root) if a.root else D.BSD_ROOT, a.shard)
    elif a.corpus == "dailytalk":
        dailytalk(Path(a.root) if a.root else D.DT_ROOT, a.hours, a.workers)
    elif a.corpus == "turnbench":
        turnbench(Path(a.root) if a.root else D.TB_ROOT)
    else:
        oto(Path(a.root) if a.root else D.OTO_ROOT, a.shards, min(a.workers, 3))


if __name__ == "__main__":
    main()
