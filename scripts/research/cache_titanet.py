"""Cache frozen NVIDIA TitaNet-Large embeddings (the speaker-distillation teacher) per single-speaker segment.

research/SPK_HEAD.md. One L2-normalised 192-d embedding per example, keyed by ``data.segment_id`` (LibriSpeech
``id``; ``meeting:start:speaker:n_samples`` for AMI / ICSI asr-mode segments), written to the ``.npz`` named by
the source's ``spk_teacher`` key (arrays ``ids`` and ``emb``). Resumable: existing ids are skipped and the file is
rewritten every ``--save-every`` seconds and at ``--budget`` (run again to continue). CPU, 2 threads.

  .venv/bin/python scripts/research/cache_titanet.py --recipe research/recipes/spk_distill_titanet.yaml --source ami_asr
  .venv/bin/python scripts/research/cache_titanet.py --recipe research/recipes/spk_distill_titanet.yaml --source librispeech --budget 500
  .venv/bin/python scripts/research/cache_titanet.py --set icsi_dev            # held-out sets (all dev segments)

``--source`` names a ``data.mix`` entry (or ``val``) of the recipe; the data is loaded exactly as the trainer
loads it (minus the teacher attach). ``--set`` presets: ami_dev, icsi_dev -> data/cache/titanet/<set>.npz.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audioforge.data import segment_id  # noqa: E402

PRESETS = {
    "ami_dev": ({"ami": {"mode": "asr", "val_split": "dev"}}, "val", "data/cache/titanet/ami_dev.npz"),
    "icsi_dev": ({"icsi": {"mode": "asr", "val_split": "dev"}}, "val", "data/cache/titanet/icsi_dev.npz"),
}
TITANET_LOCAL = ROOT / "data/nemo/speakerverification_en_titanet_large.nemo"


def source_block(recipe: str, source: str) -> tuple[dict, str, str]:
    """(data block without spk_teacher, split, teacher file) of a mix entry / the val entry of a recipe."""
    import yaml
    from audioforge.train import MIX_META
    cfg = yaml.safe_load(Path(recipe).read_text())
    d = cfg["data"]
    if source == "val":
        entry, split = d.get("val") or d["mix"][0], "val"
    else:
        entry = next((e for e in d.get("mix", []) if e.get("name") == source), None)
        if entry is None and source in d:  # single-source recipe: the data block itself
            entry = d
        if entry is None:
            raise SystemExit(f"{recipe}: no mix entry named {source!r} (have {[e.get('name') for e in d.get('mix', [])]})")
        split = "train"
    spec = entry.get("spk_teacher")
    if not spec:
        raise SystemExit(f"{recipe}: source {source!r} has no spk_teacher key")
    out = spec if isinstance(spec, str) else spec["file"]
    block = {k: v for k, v in entry.items() if k not in MIX_META and k != "spk_teacher"}
    return block, split, out.format(split=split)


def load_existing(path: Path) -> tuple[list[str], list[np.ndarray]]:
    if not path.exists():
        return [], []
    z = np.load(path, allow_pickle=False)
    return [str(i) for i in z["ids"].tolist()], list(z["emb"])


def save(path: Path, ids: list[str], embs: list[np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.array(ids), emb=np.stack(embs).astype(np.float32) if embs else np.zeros((0, 192), np.float32))
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", default="research/recipes/spk_distill_titanet.yaml")
    ap.add_argument("--source", default=None, help="mix entry name or 'val' of --recipe")
    ap.add_argument("--set", default=None, choices=sorted(PRESETS), help="held-out preset instead of --source")
    ap.add_argument("--out", default=None, help="override the output .npz")
    ap.add_argument("--budget", type=float, default=480.0, help="seconds of embedding per process")
    ap.add_argument("--save-every", type=float, default=60.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    from audioforge.train import load_data
    if a.set:
        block, split, out = PRESETS[a.set]
    elif a.source:
        block, split, out = source_block(a.recipe, a.source)
    else:
        raise SystemExit("give --source or --set")
    out = Path(a.out or out)
    t0 = time.time()
    data = load_data({"data": block, "trainer": {}}, split)
    ids_all = [segment_id(e) for e in data]
    ids, embs = load_existing(out)
    have = set(ids)
    todo = {}
    for i, sid in enumerate(ids_all):
        if sid not in have and sid not in todo:
            todo[sid] = i
    print(f"[{out.name}] {len(data)} examples, {len(set(ids_all))} unique ids, {len(have)} cached, {len(todo)} to embed "
          f"(data load {time.time() - t0:.0f}s)", flush=True)
    if not todo:
        return
    from audioforge.nemo_import import import_titanet
    teacher = import_titanet(str(TITANET_LOCAL) if TITANET_LOCAL.exists() else "nvidia/speakerverification_en_titanet_large")
    order = sorted(todo.items(), key=lambda kv: len(data[kv[1]]["audio"]))  # length-sorted batches: less padding
    t_start = t_save = time.time()
    sec_audio = 0.0
    for b in range(0, len(order), a.batch_size):
        chunk = order[b: b + a.batch_size]
        audios = [np.asarray(data[i]["audio"], np.float32) for _, i in chunk]
        E = teacher.embed(audios, batch_size=len(audios)).numpy()
        for (sid, _), e in zip(chunk, E):
            ids.append(sid)
            embs.append(e.astype(np.float32))
        sec_audio += sum(len(x) for x in audios) / 16000
        now = time.time()
        if now - t_save > a.save_every:
            save(out, ids, embs)
            t_save = now
            print(f"  {len(ids)} cached ({sec_audio / 60:.1f} min audio, rtf {(now - t_start) / max(1e-6, sec_audio):.3f})",
                  flush=True)
        if now - t_start > a.budget:
            save(out, ids, embs)
            print(f"budget reached: {len(order) - b - len(chunk)} left, re-run to continue -> {out}", flush=True)
            return
    save(out, ids, embs)
    print(f"done: {len(ids)} embeddings -> {out} ({time.time() - t_start:.0f}s, "
          f"rtf {(time.time() - t_start) / max(1e-6, sec_audio):.3f})", flush=True)


if __name__ == "__main__":
    main()
