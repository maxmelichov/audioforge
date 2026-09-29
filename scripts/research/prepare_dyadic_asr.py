"""Per-channel ASR transcripts of oto conversations (research/DYADIC.md section 8): oto has no transcripts, but the turn
head's text branch was trained on each window's PRIMARY words (AMI reference transcript, aligned through the ASR
joint). For the dyadic training windows the primary's words are the frozen ASR head's greedy decode of the party's OWN
channel (one party, clean), each word stamped with the emission time of its last token. Written once per
conversation to data/oto141/cache/asr/<id>.json = {"0": [[t_sec, word], ...], "1": [...], "model": ...};
``datasets/dyadic.py`` (asr_text: true) puts the party's words emitted inside a window into its ``text``.

The encoder runs on --device (MPS when free) over 60 s segments with 8 s of left context (the kept part starts after
the overlap), the greedy transducer loop on the CPU. Resumable within --budget seconds:
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 .venv/bin/python scripts/research/prepare_dyadic_asr.py \
      --n-conv 160 --split train --split-mod 8 --dev-res 0,1,2 --device mps --budget 540
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audioforge.datasets import dyadic as D  # noqa: E402
from audioforge.datasets.ami import FRAME_SEC, SR  # noqa: E402
from audioforge.heads.turn import greedy_decode_frames  # noqa: E402

CKPT = ROOT / "runs" / "stage1_turn_v3_trail6.afm"  # its rnnt head = the frozen ASR every stage-1 run shares
SEG_SEC, OVL_SEC = 60.0, 8.0


def asr_path(cid: str) -> Path:
    return D.OTO_ROOT / "cache" / "asr" / f"{cid}.json"


def words_of(tok, ids: list[int], times: list[float]) -> list[list]:
    """Token ids + emission times -> [[t_last_token, word], ...] (SentencePiece '▁' starts a word)."""
    out, cur, t_last = [], "", None
    for i, t in zip(ids, times):
        piece = tok.sp.id_to_piece(int(i))
        if piece.startswith("▁") and cur:
            out.append([round(t_last, 3), cur])
            cur = ""
        cur += piece.replace("▁", "")
        t_last = t
    if cur:
        out.append([round(t_last, 3), cur])
    return out


@torch.no_grad()
def channel_words(model, asr_cpu, x: np.ndarray, dev) -> list[list]:
    """Greedy decode of one 16 kHz channel in SEG_SEC pieces with OVL_SEC of left context."""
    seg, ovl = int(SEG_SEC * SR), int(OVL_SEC * SR)
    pieces, s = [], 0
    while s < len(x):
        a = max(0, s - ovl)
        pieces.append((a, min(len(x), s + seg), s))
        s += seg
    ids, times = [], []
    B = 6
    for j in range(0, len(pieces), B):
        grp = pieces[j: j + B]
        L = max(b - a for a, b, _ in grp)
        audio = torch.zeros(len(grp), L)
        for k, (a, b, _) in enumerate(grp):
            audio[k, : b - a] = torch.from_numpy(np.asarray(x[a:b], np.float32))
        lens = torch.tensor([b - a for a, b, _ in grp])
        enc, elen = model.encode(audio.to(dev), lens.to(dev))[:2]
        enc = enc.float().cpu()
        for k, (a, b, keep) in enumerate(grp):
            hyp, fr = greedy_decode_frames(asr_cpu, enc[k, : int(elen[k])])
            for i, f in zip(hyp, fr):
                t = a / SR + (f + 1) * FRAME_SEC  # emission = the end of the frame that emitted it
                if t > keep / SR:  # the overlap belongs to the previous piece
                    ids.append(i)
                    times.append(t)
    return words_of(model.tokenizer, ids, times)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-conv", type=int, default=160)
    p.add_argument("--split", default="train", choices=["train", "dev", "all"])
    p.add_argument("--split-mod", type=int, default=8)
    p.add_argument("--dev-res", default="0,1,2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--budget", type=float, default=540.0)
    a = p.parse_args()
    torch.set_num_threads(2)
    from audioforge.train import load_model
    ids = D.split_ids(D.list_ids("oto")[: a.n_conv], a.split, a.split_mod, [int(r) for r in a.dev_res.split(",")])
    todo = [c for c in ids if not asr_path(c).exists()]
    print(f"asr: {len(ids)} conversations, {len(todo)} to do", flush=True)
    if not todo:
        return
    dev = torch.device(a.device)
    model = load_model(str(CKPT), "cpu")
    asr_name = next(k for k, v in model.head_cfg.items() if v["type"] in ("rnnt", "tdt"))
    asr_cpu = copy.deepcopy(model.heads[asr_name]).eval()
    model.to(dev).eval()
    t0, done = time.time(), 0
    for cid in todo:
        if time.time() - t0 > a.budget:
            break
        x, sr, _ = D.read_stereo("oto", D.OTO_ROOT, cid)
        ch = D.resample16k(x, sr)
        out = {str(c): channel_words(model, asr_cpu, ch[c], dev) for c in (0, 1)}
        out["model"] = str(CKPT.relative_to(ROOT)) + f":{asr_name} greedy, per channel, {SEG_SEC:.0f} s pieces"
        path = asr_path(cid)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(out))
        os.replace(tmp, path)
        done += 1
        print(f"  {cid}: {len(out['0'])} / {len(out['1'])} words ({time.time() - t0:.0f}s)", flush=True)
    print(f"asr: {done} done in {time.time() - t0:.0f}s, {len(todo) - done} left", flush=True)


if __name__ == "__main__":
    main()
