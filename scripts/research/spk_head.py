"""Speaker head: architecture vs supervision (research/SPK_HEAD.md). Train / evaluate / report stages.

  train   run a recipe (GPU, one job on the machine), then assert that every tensor outside heads.spk / layer_mix.spk
          is bit-identical to the init checkpoint (the ASR is untouched, so no WER gate is needed).
  eval    speaker EER on AMI dev (n = 64 and 200: the exact trial lists and code of scripts/eval_stage1.eval_spk /
          BASELINES.md, all pairs and within-meeting) and on ICSI dev as held-out (same trial construction),
          the teacher-student cosine against the cached TitaNet-L embeddings, the effective layer weights and
          the head's parameter count. CPU, 2 threads.
  layers  the untrained baseline per layer: mean-pooled features of every encoder layer, same trials.
  report  the markdown table (runs/spk_head.json -> stdout).

Everything merges into --json (default runs/spk_head.json) under ``variants[<tag>]``.

  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 .venv/bin/python scripts/research/spk_head.py train \\
      --recipe research/recipes/spk_layer_sweep_L8.yaml --out runs/spk_sweep_L8.afm
  TMPDIR=<scratch> .venv/bin/python scripts/research/spk_head.py eval --ckpt runs/spk_sweep_L8.afm --tag L8
  .venv/bin/python scripts/research/spk_head.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from audioforge.data import Collate, load_teacher_cache, segment_id  # noqa: E402
from audioforge.metrics import eer  # noqa: E402

TITANET_CACHE = {"ami": ROOT / "data/cache/titanet/ami_dev.npz", "icsi": ROOT / "data/cache/titanet/icsi_dev.npz"}
TRAINED = ("heads.spk.", "layer_mix.spk")


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _merge(path: Path, tag: str, part: str, rec: dict):
    res = _json(path)
    res.setdefault("variants", {}).setdefault(tag, {})[part] = rec
    path.write_text(json.dumps(res, indent=1))


def _trials(val):
    spk = torch.tensor([v["speaker"] for v in val])
    meet = [v["meeting"] for v in val]
    iu = torch.triu_indices(len(val), len(val), 1)
    lab = (spk[iu[0]] == spk[iu[1]]).long()
    same = torch.tensor([meet[a] == meet[b] for a, b in iu.T.tolist()])
    return iu, lab, same


def eer_block(E: torch.Tensor, val) -> dict:
    """bench_sd_baselines.eer_block: all-pairs and within-meeting EER of unit embeddings E over val's trials."""
    iu, lab, same = _trials(val)
    E = torch.nn.functional.normalize(E.float(), dim=-1)
    s = (E[iu[0]] * E[iu[1]]).sum(-1)
    return {"eer": round(eer(s, lab), 4), "eer_within_meeting": round(eer(s[same], lab[same]), 4),
            "mean_cos_same_within": round(float(s[(lab == 1) & same].mean()), 3),
            "mean_cos_diff_within": round(float(s[(lab == 0) & same].mean()), 3),
            "n_segments": len(val), "n_speakers": int(len({v["speaker"] for v in val})), "n_trials": int(len(lab)),
            "n_target": int(lab.sum()), "n_within": int(same.sum())}


def dev_segments(corpus: str, n: int):
    """The recipe path with a seeded cap: AMI = eval_stage1.ami_dev('asr', n); ICSI = the same through icsi.recipe_data."""
    if corpus == "ami":
        from eval_stage1 import ami_dev
        return ami_dev("asr", n)
    from audioforge.datasets.icsi import recipe_data
    return recipe_data({"data": {"icsi": {"mode": "asr", "val_split": "dev", "n_val": n, "seed": 0}}}, "val")


@torch.no_grad()
def head_embeddings(model, val, batch_size=8, per_layer=False):
    """Speaker-head embeddings (N, E) of val; with per_layer also the mean-pooled features of every layer (L, N, D)."""
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "speaker")
    head, col = model.heads[name], Collate(None)
    embs, layers = [], []
    for i in range(0, len(val), batch_size):
        b = col([{"audio": v["audio"]} for v in val[i: i + batch_size]])
        enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
        embs.append(head.embed(model.head_input(name, enc, hidden), elen))
        if per_layer:
            m = (torch.arange(enc.shape[1])[None] < elen[:, None]).float()[..., None]
            layers.append(torch.stack([torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1) for h in hidden]))
    E = torch.cat(embs)
    return (E, torch.cat(layers, 1)) if per_layer else E


def _load(ckpt):
    from audioforge.train import load_model
    return load_model(ckpt, "cpu")


# --------------------------------------------------------------------------- stages
def stage_train(a):
    from audioforge.train import load_model, load_recipe, run_recipe
    cfg = load_recipe(a.recipe)
    init = cfg["init"]["from"]
    t0 = time.time()
    model, metrics = run_recipe(a.recipe, a.overrides, out=a.out)
    sec = time.time() - t0
    src = load_model(init, "cpu").state_dict()
    out = load_model(a.out, "cpu")
    sd = out.state_dict()
    changed = [k for k in src if k in sd and not torch.equal(src[k], sd[k])]
    bad = [k for k in changed if not k.startswith(TRAINED)]
    assert not bad, f"tensors outside heads.spk changed: {bad[:5]}"
    frozen_same = sum(1 for k in src if k in sd and not k.startswith(TRAINED))
    tag = a.tag or Path(a.out).stem
    rec = {"recipe": a.recipe, "ckpt": a.out, "init": init, "sec": round(sec, 1), "final_eval": metrics,
           "frozen_tensors_identical": frozen_same, "changed_tensors": changed,
           "rnnt_ctc_encoder_unchanged": True, "spk_cfg": cfg["heads"]["spk"],
           "layer_weights": _lw(out)}
    _merge(Path(a.json), tag, "train", rec)
    print(f"[train] {tag}: {sec:.0f}s, {frozen_same} frozen tensors identical, changed {len(changed)} "
          f"(all under heads.spk / layer_mix.spk)", flush=True)


def _lw(model):
    w = model.layer_weights("spk")
    return None if w is None else [round(float(x), 4) for x in w]


def stage_eval(a):
    torch.set_num_threads(a.threads)
    model = _load(a.ckpt)
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "speaker")
    rec = {"ckpt": a.ckpt, "head_params": int(sum(p.numel() for p in model.heads[name].parameters())),
           "spk_cfg": model.head_cfg[name], "layer_weights": _lw(model), "att_context_size": list(model.encoder.att_context_size)}
    for corpus in ("ami", "icsi"):
        cache = load_teacher_cache(TITANET_CACHE[corpus]) if TITANET_CACHE[corpus].exists() else {}
        for n in (64, 200):
            val = dev_segments(corpus, n)
            t0 = time.time()
            E = head_embeddings(model, val, a.batch_size)
            sec = time.time() - t0
            r = eer_block(E, val)
            r["sec"] = round(sec, 1)
            r["rtf_shared_pass"] = round(sec / sum(v["duration"] for v in val), 4)
            if corpus == "ami" and n == 64:
                from eval_stage1 import eval_spk  # the BASELINES / STAGE1 code path, as a cross-check
                r["eval_stage1"] = eval_spk(model, n, a.batch_size)
                assert abs(r["eval_stage1"]["eer_spk_head_within_meeting"] - r["eer_within_meeting"]) < 1e-6
            ids = [segment_id(v) for v in val]
            if cache and all(i in cache for i in ids):
                T = torch.tensor(np.stack([cache[i] for i in ids]))
                r["teacher_student_cos"] = round(float((torch.nn.functional.normalize(E, dim=-1) * T).sum(-1).mean()), 4)
                r["teacher"] = eer_block(T, val)
            rec[f"{corpus}_n{n}"] = r
            print(corpus, n, {k: v for k, v in r.items() if k in ("eer", "eer_within_meeting", "teacher_student_cos")},
                  flush=True)
    _merge(Path(a.json), a.tag, "eval", rec)


def stage_layers(a):
    """Untrained baseline: mean-pooled features of each encoder layer (and of the head's mix) on the same trials."""
    torch.set_num_threads(a.threads)
    model = _load(a.ckpt)
    rec = {"ckpt": a.ckpt}
    for corpus in ("ami", "icsi"):
        for n in (64, 200):
            val = dev_segments(corpus, n)
            _, L = head_embeddings(model, val, a.batch_size, per_layer=True)
            rows = [eer_block(L[i], val) for i in range(L.shape[0])]
            rec[f"{corpus}_n{n}"] = {"per_layer": [{"layer": i, "eer": r["eer"], "eer_within_meeting": r["eer_within_meeting"]}
                                                    for i, r in enumerate(rows)]}
            best = min(rows, key=lambda r: r["eer_within_meeting"])
            print(corpus, n, "best layer (within)", rows.index(best), best["eer_within_meeting"],
                  "top layer", rows[-1]["eer_within_meeting"], flush=True)
    _merge(Path(a.json), "untrained_meanpool", "layers", rec)


def stage_report(a):
    res = _json(Path(a.json))
    v = res.get("variants", {})
    print("| variant | loss / layers | AMI dev n=64 all / within | AMI dev n=200 all / within | ICSI dev n=64 within | "
          "ICSI dev n=200 all / within | teacher cos (AMI200 / ICSI200) |")
    print("|---|---|---|---|---|---|---|")
    for tag, r in v.items():
        e = r.get("eval")
        if not e:
            continue
        def pc(b, k="eer"):
            return f"{100 * b[k]:.1f}"
        a64, a200, i64, i200 = e["ami_n64"], e["ami_n200"], e["icsi_n64"], e["icsi_n200"]
        cfg = e["spk_cfg"]
        loss = "distill" if cfg.get("distill") else "AAM"
        print(f"| {tag} | {loss}, from_layers {cfg.get('from_layers')} | {pc(a64)} / {pc(a64, 'eer_within_meeting')} | "
              f"{pc(a200)} / {pc(a200, 'eer_within_meeting')} | {pc(i64, 'eer_within_meeting')} | "
              f"{pc(i200)} / {pc(i200, 'eer_within_meeting')} | "
              f"{a200.get('teacher_student_cos', '-')} / {i200.get('teacher_student_cos', '-')} |")
    u = v.get("untrained_meanpool", {}).get("layers")
    if u:
        print("\nuntrained mean-pooled layer (within-meeting EER %, AMI n=200 / ICSI n=200):")
        print(" ".join(f"L{i + 1}:{100 * r['eer_within_meeting']:.0f}/{100 * s['eer_within_meeting']:.0f}"
                       for i, (r, s) in enumerate(zip(u["ami_n200"]["per_layer"], u["icsi_n200"]["per_layer"]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("train", "eval", "layers", "report"))
    ap.add_argument("--recipe")
    ap.add_argument("--out")
    ap.add_argument("--ckpt", default="runs/stage1_heads_pretrained.afm")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--json", default="runs/spk_head.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    if a.stage == "train":
        stage_train(a)
    elif a.stage == "eval":
        a.tag = a.tag or Path(a.ckpt).stem
        stage_eval(a)
    elif a.stage == "layers":
        stage_layers(a)
    else:
        stage_report(a)


if __name__ == "__main__":
    main()
