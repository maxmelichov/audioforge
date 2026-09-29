"""Evaluate a LibriSpeech-trained voice-agent front end (one encoder pass, every head).

    python -m audioforge.datasets.eval_librispeech runs/voice_agent_frontend_librispeech.afm \
        --split dev-clean --n 200 --out runs/voice_agent_frontend_librispeech.eval.json

* WER of every text head (greedy), on ``n`` seeded-random utterances of ``split`` (no duration cap
  unless --max-sec), also broken down by duration (<=12 s / >12 s).
* VAD frame accuracy/recall, EOU frame accuracy/recall/precision against the energy-VAD labels.
* Speaker: with ``--recipe-val`` the model's own held-out validation utterances are used (training
  speakers -> closed-set accuracy). Otherwise speakers are unseen, so we report verification EER
  over all same/different-speaker pairs of the ``n`` embeddings (cosine scores).
* Streaming: ``StreamingSession`` on ``--stream`` utterances; its text must equal offline decoding.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time

import numpy as np
import torch

from ..metrics import eer, wer
from ..model import TEXT_HEADS, StreamingSession
from ..train import load_model
from .librispeech import LibriSpeech, recipe_data


def evaluate(model, data: list[dict], batch_size: int = 16, closed_set: bool = False) -> dict:
    m, res = model.eval(), {"n": len(data), "hours": round(sum(len(e["audio"]) for e in data) / 16000 / 3600, 3)}
    text_heads = [k for k, v in m.head_cfg.items() if v["type"] in TEXT_HEADS]
    frame_heads = [k for k, v in m.head_cfg.items() if v["type"] == "frame"]
    spk_heads = [k for k, v in m.head_cfg.items() if v["type"] == "speaker"]
    hyps = {k: [] for k in text_heads}
    frame_stats = {k: np.zeros(4) for k in frame_heads}  # tp, fp, fn, tn
    embs = []
    for i in range(0, len(data), batch_size):
        chunk = data[i: i + batch_size]
        out = m.analyze([e["audio"] for e in chunk])  # one shared encoder pass for all heads
        for k in text_heads:
            hyps[k] += out[k]
        for k in frame_heads:
            key = m.heads[k].key
            for b, e in enumerate(chunk):
                T = min(out["frames"][b], len(e[key]))
                p = (out[k][b, :T] > 0.5).cpu().numpy()
                t = np.asarray(e[key][:T]) > 0.5
                frame_stats[k] += [(p & t).sum(), (p & ~t).sum(), (~p & t).sum(), (~p & ~t).sum()]
        for k in spk_heads:
            embs.append(out[k].cpu())
    refs = [e["text"] for e in data]
    long = np.array([len(e["audio"]) > 12 * 16000 for e in data])
    for k in text_heads:
        res[f"wer_{k}"] = round(wer(refs, hyps[k]), 4)
        for name, sel in (("le12s", ~long), ("gt12s", long)):
            if sel.any():
                res[f"wer_{k}_{name}"] = round(wer([r for r, s in zip(refs, sel) if s],
                                                   [h for h, s in zip(hyps[k], sel) if s]), 4)
    for k, (tp, fp, fn, tn) in frame_stats.items():
        res[f"acc_{k}"] = round((tp + tn) / max(1, tp + fp + fn + tn), 4)
        res[f"recall_{k}"] = round(tp / max(1, tp + fn), 4)
        res[f"precision_{k}"] = round(tp / max(1, tp + fp), 4)
        res[f"pos_rate_{k}"] = round((tp + fn) / max(1, tp + fp + fn + tn), 4)
    if spk_heads:
        e = torch.cat(embs)
        spk = torch.tensor([ex["speaker"] for ex in data])
        res["n_speakers"] = int(spk.unique().numel())
        if closed_set:
            W = torch.nn.functional.normalize(m.heads[spk_heads[0]].W.detach().cpu(), dim=-1)
            res["spk_acc"] = round(float(((e @ W.T).argmax(-1) == spk).float().mean()), 4)
        pairs = torch.tensor(list(itertools.combinations(range(len(data)), 2)))
        scores = (e[pairs[:, 0]] * e[pairs[:, 1]]).sum(-1)
        labels = (spk[pairs[:, 0]] == spk[pairs[:, 1]]).long()
        res["spk_eer"] = round(eer(scores, labels), 4)
        res["spk_pairs"] = [int(labels.sum()), int((1 - labels).sum())]  # same, different
    res["examples"] = [dict(id=ex.get("id"), ref=r, **{k: hyps[k][j] for k in text_heads})
                       for j, (ex, r) in enumerate(zip(data, refs))]
    return res


def streaming_check(model, data: list[dict], chunk_ms: int = 160) -> list[dict]:
    out = []
    for ex in data:
        offline = model.transcribe([ex["audio"]])[0]
        s = StreamingSession(model)
        step = int(16000 * chunk_ms / 1000)
        partials = []
        for i in range(0, len(ex["audio"]), step):
            partials.append(s.feed(ex["audio"][i: i + step]))
        final = s.feed(np.zeros(0, np.float32), final=True)
        out.append(dict(id=ex.get("id"), match=final == offline, streaming=final, offline=offline,
                        n_chunks=len(partials)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sec", type=float, default=None, help="duration cap for the eval pool (default none)")
    ap.add_argument("--recipe-val", action="store_true",
                    help="evaluate on the recipe's own held-out val utterances (closed-set speakers)")
    ap.add_argument("--stream", type=int, default=5, help="utterances for the streaming == offline check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    model = load_model(a.model, a.device)
    t0 = time.time()
    if a.recipe_val:
        data, where = recipe_data(model.cfg, "val"), "recipe val (held-out, training speakers)"
    else:
        ds = LibriSpeech(a.split, a.root, max_sec=a.max_sec, min_sec=0.0)
        idx = sorted(random.Random(a.seed).sample(range(len(ds)), min(a.n, len(ds))))
        data, where = ds.utterances(idx), f"{a.split}: {len(idx)} seeded-random utterances"
    res = {"model": a.model, "eval_set": where, **evaluate(model, data, closed_set=a.recipe_val)}
    if a.stream:
        res["streaming"] = streaming_check(model, data[: a.stream])
        res["streaming_match"] = f"{sum(s['match'] for s in res['streaming'])}/{len(res['streaming'])}"
    res["eval_sec"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("examples", "streaming")}, indent=1))
    for ex in res["examples"][:3]:
        print(json.dumps(ex))
    for s in res.get("streaming", []):
        print("stream", json.dumps(s))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
