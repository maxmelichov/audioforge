"""research/DIARIZATION_FIX.md section 5: distil NVIDIA's diarizer into our own head on the frozen served encoder.

  teacher --corpus ami|icsi [--split train --budget 540]   cache Nemotron-3 (max pool, 8 columns) posteriors per
                                                          16 s diar window -> data/cache/diar_teacher/<corpus>_<split>.npz
  train   --variant b6|mix46 --out CKPT [--steps 1000]     head-only distillation on MPS (waits while another training
                                                          runs), checkpoint on the SSD, read back and frozen-checked
  eval    --ckpt CKPT [--n 64] [--tag T]                  pooled frame DER + speaker-count error on AMI dev and ICSI dev
                                                          (the teacher too: --ckpt runs/nemo_nemotron3_diar.afm --pool max)

Every stage goes through scripts/dev/gate.sh; results are merged into runs/diar_distill.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
CACHE = ROOT / "data" / "cache" / "diar_teacher"
JSON = ROOT / "runs" / "diar_distill.json"
TEACHER = ROOT / "runs" / "nemo_nemotron3_diar.afm"


def _merge(section: str, tag: str, rec: dict):
    d = json.loads(JSON.read_text()) if JSON.exists() else {}
    d.setdefault(section, {})[tag] = rec
    JSON.write_text(json.dumps(d, indent=1, default=float))


def _windows(corpus: str, split: str, n: int | None = None, window_sec: float = 16.0):
    if corpus == "ami":
        from audioforge.datasets.ami import recipe_data
    else:
        from audioforge.datasets.icsi import recipe_data
    src = {"mode": "diar", "window_sec": window_sec}
    if split == "train":
        src["train_split"] = "train"
        return recipe_data({"data": {corpus: src}}, "train")
    src.update(val_split=split, n_val=n, seed=0)
    return recipe_data({"data": {corpus: src}}, "val")


def _teacher(pool: str = "max"):
    from audioforge.nemo_import import load_any
    m = load_any(str(TEACHER), "cpu")
    for h in m.heads.values():
        if hasattr(h, "pool"):
            h.pool = pool
    return m


# --------------------------------------------------------------------------- teacher cache
def cmd_teacher(a):
    from audioforge.data import segment_id
    from bench_sd_baselines import model_pass
    torch.set_num_threads(a.threads)
    out = CACHE / f"{a.corpus}_{a.split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, np.ndarray] = {}
    if out.exists():
        z = np.load(out, allow_pickle=False)
        done = {str(i): e[: int(n)] for i, e, n in zip(z["ids"].tolist(), z["emb"], z["lens"])}
    data = _windows(a.corpus, a.split)
    todo = [d for d in data if segment_id(d) not in done]
    print(f"[teacher] {a.corpus}/{a.split}: {len(data)} windows, {len(done)} cached, {len(todo)} to do", flush=True)
    model = _teacher(a.pool)
    t0, n_new = time.time(), 0

    def save():
        ids = list(done)
        T = max(len(done[i]) for i in ids)
        emb = np.zeros((len(ids), T, 8), np.float16)
        lens = np.zeros(len(ids), np.int32)
        for k, i in enumerate(ids):
            emb[k, : len(done[i])] = done[i]
            lens[k] = len(done[i])
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(ids), emb=emb, lens=lens)
        os.replace(tmp, out)

    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > a.budget:
            print(f"[teacher] budget: {n_new} new windows cached; rerun to continue", flush=True)
            break
        b = todo[i: i + a.batch]
        with torch.inference_mode():
            pd, _ = model_pass(model, b, batch_size=a.batch)
        for d, p in zip(b, pd):
            done[segment_id(d)] = np.asarray(p, np.float16)
        n_new += len(b)
        if n_new % (a.batch * 20) == 0:
            save()
    save()
    z = np.load(out, allow_pickle=False)  # read-back check
    assert len(z["ids"]) == len(done) and z["emb"].shape[2] == 8, out
    print(f"[teacher] {out}: {len(done)} windows ({time.time() - t0:.0f}s, {os.path.getsize(out) / 1e6:.1f} MB)", flush=True)


# --------------------------------------------------------------------------- training
VARIANTS = {"b6": ["heads.diar.from_layers=[5]"], "mix46": ["heads.diar.from_layers=[3,5]"],
            "b6_nodistill": ["heads.diar.from_layers=[5]", "heads.diar.distill=null"]}


def mps_busy() -> list[str]:
    """Other project trainings that may hold the GPU (audioforge.train / research train drivers above 25 % CPU)."""
    ps = subprocess.run(["ps", "-axo", "%cpu,pid,args"], capture_output=True, text=True).stdout.splitlines()
    busy = []
    for line in ps[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3 or float(parts[0]) < 25:
            continue
        cmd = parts[2]
        if ".venv/bin/python" in cmd and str(os.getpid()) != parts[1] and any(
                k in cmd for k in ("audioforge.train", "train.py", "layer_routing.py train", "tsvad_turn.py",
                                   "spk_head.py train", "diar_distill.py train", "--device mps")):
            busy.append(cmd[:100])
    return busy


def cmd_train(a):
    from audioforge.train import load_recipe, run_recipe, load_model
    from layer_routing import frozen_check, layer_weights
    waited = 0
    cpu = "trainer.device=cpu" in a.overrides  # smoke runs on the CPU do not need the GPU
    while not cpu and (busy := mps_busy()) and waited < a.wait_max:
        if waited % 300 == 0:
            print(f"[train] waiting for the GPU: {busy}", flush=True)
        time.sleep(30)
        waited += 30
    if not cpu and mps_busy():
        sys.exit("diar_distill train: another training still holds the GPU; try later")
    overrides = VARIANTS[a.variant] + [f"trainer.max_steps={a.steps}"] + a.overrides
    cfg = load_recipe(a.recipe, overrides)
    t0 = time.time()
    _, metrics = run_recipe(a.recipe, overrides, out=a.out)
    sec = time.time() - t0
    chk = frozen_check(cfg["init"]["from"], a.out, ("heads.diar", "layer_mix.diar"))
    m = load_model(a.out, "cpu")  # read-back check
    n_head = int(sum(p.numel() for p in m.heads["diar"].parameters()))
    rec = {"variant": a.variant, "recipe": a.recipe, "overrides": overrides, "ckpt": a.out, "sec": round(sec, 1),
           "steps": a.steps, "final_eval": metrics, **chk, "head_params": n_head, "head_cfg": m.head_cfg["diar"],
           "layer_weights": layer_weights(m, "diar"), "size_mb": round(os.path.getsize(a.out) / 1e6, 1)}
    _merge("train", a.tag or a.variant, rec)
    print(f"[train] {a.variant}: {sec:.0f}s, head {n_head / 1e6:.2f}M params, frozen identical "
          f"{chk['frozen_tensors_identical']}, changed {chk['changed_tensors']}; val {metrics}", flush=True)


# --------------------------------------------------------------------------- evaluation
def count_error(hard: list[np.ndarray], refs: list[np.ndarray], min_s: float = 1.0):
    k = int(min_s / 0.08)
    e = [int((h.sum(0) >= k).sum()) - int((r.sum(0) >= k).sum()) for h, r in zip(hard, refs)]
    return {"mean": round(float(np.mean(e)), 3), "mean_abs": round(float(np.mean(np.abs(e))), 3),
            "hyp_mean": round(float(np.mean([(h.sum(0) >= k).sum() for h in hard])), 2),
            "ref_mean": round(float(np.mean([(r.sum(0) >= k).sum() for r in refs])), 2)}


def cmd_eval(a):
    from bench_sd_baselines import model_pass, pooled_der, _fit
    from layer_routing import icsi_diar_windows, layer_weights
    from audioforge.heads.turn import _diar_name
    torch.set_num_threads(a.threads)
    if a.pool:
        model = _teacher(a.pool) if Path(a.ckpt).resolve() == TEACHER.resolve() else None
    else:
        model = None
    if model is None:
        from audioforge.nemo_import import load_any
        model = load_any(a.ckpt, "cpu")
        if a.pool:
            for h in model.heads.values():
                if hasattr(h, "pool"):
                    h.pool = a.pool
    diar = _diar_name(model)
    rec = {"ckpt": a.ckpt, "head": diar, "head_cfg": model.head_cfg.get(diar), "pool": a.pool,
           "layer_weights": layer_weights(model, diar),
           "head_params": int(sum(p.numel() for p in model.heads[diar].parameters()))}
    from bench_sd_baselines import diar_windows
    for corpus, fn in (("ami", diar_windows), ("icsi", icsi_diar_windows)):
        if corpus not in a.corpora.split(","):
            continue
        val = fn(a.n)
        t0 = time.time()
        with torch.inference_mode():
            pd, _ = model_pass(model, val)
        refs = [np.asarray(v["spk_targets"], np.float32) for v in val]
        hard = [(_fit(np.asarray(p), len(r)) > 0.5).astype(np.float32) for p, r in zip(pd, refs)]
        if hard[0].shape[1] > 8:
            hard = [h[:, :8] for h in hard]
        r = {"n": len(val), "frame_der": pooled_der(hard, refs), "count": count_error(hard, refs),
             "sec": round(time.time() - t0, 1)}
        rec[corpus] = r
        print(f"[eval] {a.tag} {corpus}: DER {r['frame_der']} count {r['count']} ({r['sec']}s)", flush=True)
    old = (json.loads(JSON.read_text()) if JSON.exists() else {}).get("eval", {}).get(a.tag, {})
    _merge("eval", a.tag, {**old, **rec})  # --corpora ami then icsi: both kept under one tag


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("teacher")
    p.add_argument("--corpus", choices=["ami", "icsi"], required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--pool", default="max")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--budget", type=float, default=540)
    p.set_defaults(fn=cmd_teacher)
    p = sub.add_parser("train")
    p.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    p.add_argument("--recipe", default=str(ROOT / "research/recipes/diar_distill_nemotron3.yaml"))
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--tag", default=None)
    p.add_argument("--wait-max", type=float, default=2400)
    p.add_argument("overrides", nargs="*")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("eval")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--pool", default=None)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--corpora", default="ami,icsi")
    p.set_defaults(fn=cmd_eval)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
