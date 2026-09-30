"""Overlap robustness of speaker representations on AMI dev turn windows (research/archive/SPK_HEAD.md, "Overlap robustness").

Question: does a frame-level lower-layer embedding average to garbage in overlap, follow the louder speaker, or keep
the primary (turn owner)? Frames of the eval_stage1 AMI dev turn windows are labelled per speaker NAME with the
word-level activity (ami.AMI.acts), tiled into 0.96 s spans (12 frames); a span is *single* (>= 9 active frames,
no frame with >= 2 speakers) or *overlap* (>= 6 frames with >= 2 speakers). Each span is embedded with one
representation and compared (cosine) with every meeting speaker's clean enrollment, the mean embedding of that
speaker's single-speaker asr-mode segments elsewhere in the meeting (segments intersecting the window are excluded).

Representations (``--reps``): ``block<k>`` = mean of encoder block k's frame vectors (1-based block, 0-based tap
k-1); ``head:<afm>`` = that checkpoint's speaker head on its own input frames (pool + emb over the span's frames,
enrollment = head.embed of the segments); ``titanet`` = TitaNet-L on the span's audio (enrollment from the cached
segment embeddings, data/cache/titanet/ami_dev.npz). All our representations share the frozen stage-1 encoder,
which is run once per window.

Reported per representation: overlap spans -> fraction whose nearest enrollment is (a) the louder speaker
(no per-speaker headset audio here, so louder = more labelled frames in the span, ties skipped), (b) the primary
(over spans where the primary is active), (c) neither present speaker; mean cosine to the primary / the other
present speakers / absent speakers; single spans -> nearest = active speaker rate, mean cosine to the active /
absent speakers. CPU, 2 threads, ~3-5 min for 64 windows.

  TMPDIR=<scratch> .venv/bin/python scripts/research/spk_overlap.py --reps block4,block8,head:runs/spk_sweep_L4.afm,titanet
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
from audioforge.datasets.ami import AMI, FRAME_SEC, SR, frames  # noqa: E402
from audioforge.enrollment import ColumnEmbedder, speaker_head_name  # noqa: E402

SPAN = 12  # frames = 0.96 s


def unit(x):
    x = np.asarray(x, np.float32)
    return x / max(1e-8, float(np.linalg.norm(x)))


def span_classes(lab: np.ndarray):
    """lab (T, n_spk) 0/1 -> list of (a, b, kind, present, louder) with kind in {single, overlap}."""
    out = []
    T = lab.shape[0]
    for a in range(0, T - SPAN + 1, SPAN):
        seg = lab[a: a + SPAN]
        n_act = seg.sum(1)
        n_over = int((n_act >= 2).sum())
        n_on = int((n_act >= 1).sum())
        cnt = seg.sum(0)
        present = [i for i in range(lab.shape[1]) if cnt[i] >= 3]
        if n_over >= SPAN // 2:
            kind = "overlap"
        elif n_on >= 9 and n_over == 0:
            kind = "single"
        else:
            continue
        top = np.argsort(-cnt)
        louder = int(top[0]) if len(top) > 1 and cnt[top[0]] > cnt[top[1]] else None
        out.append((a, a + SPAN, kind, present, louder))
    return out


class Rep:
    """One representation: span embedding from a window's (enc, hidden) or audio; enrollment from segments."""

    def __init__(self, name, model=None, head_model=None, teacher=None, cache=None):
        self.name, self.model, self.head_model, self.teacher, self.cache = name, model, head_model, teacher, cache
        if name.startswith("block"):
            self.k = int(name[5:]) - 1
        if head_model is not None:
            self.hname = speaker_head_name(head_model)
            self.embedder = ColumnEmbedder(head_model.heads[self.hname])

    def frames(self, enc, hidden):
        """(T, D) frame vectors of the representation for one window."""
        if self.head_model is not None:
            return self.head_model.head_input(self.hname, enc, hidden)[0]
        return hidden[self.k][0]

    def spans(self, fr, audio, spans):
        """(n_spans, E) unit embeddings."""
        if self.teacher is not None:
            clips = [np.asarray(audio[int(a * FRAME_SEC * SR): int(b * FRAME_SEC * SR)], np.float32) for a, b, *_ in spans]
            return self.teacher.embed(clips, batch_size=16).numpy()
        if self.head_model is not None:
            x = torch.stack([fr[a:b] for a, b, *_ in spans])
            return self.embedder(x, torch.ones(x.shape[:2], dtype=torch.bool))
        return np.stack([unit(fr[a:b].mean(0).numpy()) for a, b, *_ in spans])

    def segment_embedding(self, seg_ex, enc, elen, hidden):
        if self.teacher is not None:
            return unit(self.cache[segment_id(seg_ex)])
        if self.head_model is not None:
            return self.head_model.heads[self.hname].embed(self.head_model.head_input(self.hname, enc, hidden), elen)[0].numpy()
        return unit(hidden[self.k][0, : int(elen[0])].mean(0).numpy())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", default="block4,block8,titanet")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ckpt", default="runs/stage1_heads_pretrained.afm", help="the shared frozen encoder")
    ap.add_argument("--json", default="runs/spk_head.json")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from audioforge.train import load_model
    from eval_stage1 import ami_dev
    t0 = time.time()
    base = load_model(a.ckpt, "cpu")
    reps = []
    for r in a.reps.split(","):
        if r == "titanet":
            from audioforge.nemo_import import import_titanet
            tn = import_titanet(str(ROOT / "data/nemo/speakerverification_en_titanet_large.nemo"))
            reps.append(Rep(r, teacher=tn, cache=load_teacher_cache(ROOT / "data/cache/titanet/ami_dev.npz")))
        elif r.startswith("head:"):
            hm = load_model(r[5:], "cpu")
            esd, bsd = hm.encoder.state_dict(), base.encoder.state_dict()
            assert all(torch.equal(esd[k], bsd[k]) for k in esd), "head checkpoint has a different encoder"
            reps.append(Rep(r, head_model=hm))
        else:
            reps.append(Rep(r, model=base))
    val = ami_dev("turn", a.n)
    meetings = sorted({v["meeting"] for v in val})
    ds = AMI(meetings, verbose=False)
    names = {m: sorted(ds.acts[m]) for m in meetings}
    gid2name = {ds.gid(n): n for m in meetings for n in names[m]}
    # enrollment segments: single-speaker asr-mode segments per meeting, embedded once per representation
    segs = ds.asr()
    col = Collate(None)
    seg_emb = {r.name: [] for r in reps}
    for s in segs:
        b = col([{"audio": s["audio"]}])
        enc, elen, hidden = base.encode(b["audio"], b["audio_len"], return_hidden=True)
        for r in reps:
            seg_emb[r.name].append(r.segment_embedding(s, enc, elen, hidden))
    print(f"{len(segs)} enrollment segments embedded ({time.time() - t0:.0f}s)", flush=True)
    stats = {r.name: {"overlap": {"n": 0, "louder": 0, "n_louder": 0, "primary": 0, "n_primary": 0, "neither": 0,
                                  "cos_primary": [], "cos_other_present": [], "cos_absent": []},
                      "single": {"n": 0, "nearest_active": 0, "cos_active": [], "cos_absent": []}} for r in reps}
    n_spans = {"single": 0, "overlap": 0}
    for v in val:
        m, T = v["meeting"], len(v["spk_act"])
        a0, b0 = v["start"], v["start"] + len(v["audio"]) / SR
        lab = np.stack([frames(ds.acts[m][n], T, a0) for n in names[m]], 1)
        prim_name = gid2name[v["speaker"]]
        pi = names[m].index(prim_name)
        spans = span_classes(lab)
        if not spans:
            continue
        b = col([{"audio": v["audio"]}])
        enc, elen, hidden = base.encode(b["audio"], b["audio_len"], return_hidden=True)
        Tm = min(T, int(elen[0]))
        spans = [s for s in spans if s[1] <= Tm]
        for kind in n_spans:
            n_spans[kind] += sum(1 for s in spans if s[2] == kind)
        for r in reps:
            # enrollment per speaker from segments of this meeting outside the window
            E = {}
            for s, e in zip(segs, seg_emb[r.name]):
                if s["meeting"] != m or (s["start"] < b0 and s["start"] + s["duration"] > a0):
                    continue
                E.setdefault(gid2name[s["speaker"]], []).append(e)
            enrolled = [n for n in names[m] if n in E]
            if len(enrolled) < 2 or prim_name not in enrolled:
                continue
            M = np.stack([unit(np.mean(E[n], 0)) for n in enrolled])  # (S, E)
            idx = [names[m].index(n) for n in enrolled]
            fr = r.frames(enc, hidden) if r.teacher is None else None
            X = r.spans(fr, v["audio"], spans)
            cos = X @ M.T  # (n_spans, S)
            for (sa, sb, kind, present, louder), c in zip(spans, cos):
                present_e = [j for j, i in enumerate(idx) if i in present]
                if not present_e:
                    continue
                absent_e = [j for j in range(len(idx)) if j not in present_e]
                near = int(np.argmax(c))
                st = stats[r.name][kind]
                st["n"] += 1
                if kind == "single":
                    st["nearest_active"] += int(near in present_e)
                    st["cos_active"].append(float(c[present_e].mean()))
                    if absent_e:
                        st["cos_absent"].append(float(c[absent_e].mean()))
                    continue
                pj = idx.index(pi) if pi in idx else None
                if louder is not None and louder in idx:
                    st["n_louder"] += 1
                    st["louder"] += int(near == idx.index(louder))
                if pj is not None and pj in present_e:
                    st["n_primary"] += 1
                    st["primary"] += int(near == pj)
                    st["cos_primary"].append(float(c[pj]))
                    others = [j for j in present_e if j != pj]
                    if others:
                        st["cos_other_present"].append(float(c[others].mean()))
                st["neither"] += int(near not in present_e)
                if absent_e:
                    st["cos_absent"].append(float(c[absent_e].mean()))
    out = {"n_windows": len(val), "span_frames": SPAN, "n_spans": n_spans, "louder_by": "label frames in span (mixed audio only)",
           "reps": {}}
    for r in reps:
        o, s = stats[r.name]["overlap"], stats[r.name]["single"]
        f = lambda x, n: round(x / n, 3) if n else None  # noqa: E731
        mean = lambda xs: round(float(np.mean(xs)), 3) if xs else None  # noqa: E731
        out["reps"][r.name] = {
            "overlap": {"n": o["n"], "nearest_is_louder": f(o["louder"], o["n_louder"]), "n_louder_defined": o["n_louder"],
                        "nearest_is_primary": f(o["primary"], o["n_primary"]), "n_primary_present": o["n_primary"],
                        "nearest_is_neither_present": f(o["neither"], o["n"]),
                        "cos_primary": mean(o["cos_primary"]), "cos_other_present": mean(o["cos_other_present"]),
                        "cos_absent": mean(o["cos_absent"])},
            "single": {"n": s["n"], "nearest_is_active": f(s["nearest_active"], s["n"]),
                       "cos_active": mean(s["cos_active"]), "cos_absent": mean(s["cos_absent"])}}
        print(r.name, json.dumps(out["reps"][r.name]), flush=True)
    out["sec"] = round(time.time() - t0, 1)
    p = Path(a.json)
    res = json.loads(p.read_text()) if p.exists() else {}
    res.setdefault("overlap", {}).update(out["reps"])
    res["overlap"]["_meta"] = {k: v for k, v in out.items() if k != "reps"}
    p.write_text(json.dumps(res, indent=1))
    print(f"-> {p} ({out['sec']}s)")


if __name__ == "__main__":
    main()
