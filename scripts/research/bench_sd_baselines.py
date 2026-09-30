"""Dedicated VAD / diarization / speaker-verification models vs our heads, on the eval_stage1 AMI dev data.

Same data and metric code as ``scripts/research/eval_stage1.py --tasks diar,spk`` (research/archive/STAGE1.md):
  VAD   64 x 20 s AMI dev diar windows, label = any speaker active (train.derive_labels), 80 ms frames.
        Finer outputs are max-pooled onto the 80 ms grid (sd.pool_probs), the rule the labels are built with.
        Pooled acc / recall / precision / F1 / FPR at several thresholds (a DET sweep), + CPU time.
  diar  the same windows, pooled frame DER with eval_stage1.der_parts (best permutation, no collar, overlap
        scored, threshold 0.5), hypotheses mapped to (T, 4) arrival-order columns (top-4 by activity);
        + pyannote.metrics DiarizationErrorRate against the same frame reference (collar 0, 0.25, 0.5 = +-0.25 s).
  spk   AMI asr-mode single-speaker dev segments (64: 15 speakers, 2016 trials; 200: 19900 trials), cosine
        scores, metrics.eer, all pairs and within-meeting pairs (eval_stage1.eval_spk's trial lists).

Stages (one model-loading process at a time; each merges into --out):
  vad         Silero VAD v5.1.2 + v6 (pip silero-vad), WebRTC VAD modes 0-3, NVIDIA Frame-VAD MarbleNet v2.0 (ported)
  ours        our model (runs/stage1_heads_pretrained.afm): VAD-head scores (sweep), diar head through pyannote.metrics,
              speaker-head EER at n=64 and n=200 (eval_stage1.eval_spk)
  sortformer  NVIDIA streaming Sortformer v2 offline pass (runs/nemo_sortformer_v2.afm): frame DER + pyannote.metrics
              + as a VAD (any column)
  pyannote    --pipeline pyannote/speaker-diarization-community-1 | pyannote/speaker-diarization-3.1
  spk         TitaNet-Large (ported, audioforge.nemo_import.import_titanet), SpeechBrain ECAPA-TDNN,
              WeSpeaker ResNet34-LM (pyannote/wespeaker-voxceleb-resnet34-LM): EER n=64 and n=200

  TMPDIR=<scratch> .venv/bin/python scripts/research/bench_sd_baselines.py --stage vad --out runs/baselines_sd.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from audioforge.baselines import sd  # noqa: E402
from audioforge.metrics import eer  # noqa: E402
from audioforge.train import derive_labels  # noqa: E402
from eval_stage1 import ami_dev, der_parts  # noqa: E402

THRESH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SCRATCH = Path(os.environ.get("BASELINES_SCRATCH", "/private/tmp/claude-501/-Users-maxm/"
                              "54361310-ccc6-4257-a73f-3341f209b7ca/scratchpad/baselines_sd"))
SILERO_V5_URL = "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.jit"


# --------------------------------------------------------------------------- data
def diar_windows(n: int = 64) -> list[dict]:
    return [derive_labels(dict(v), ["vad"]) for v in ami_dev("diar", n)]


def audio_sec(val) -> float:
    return float(sum(len(v["audio"]) for v in val) / sd.SR)


# --------------------------------------------------------------------------- scoring
def pooled_der(preds: list[np.ndarray], refs: list[np.ndarray]) -> dict:
    """eval_stage1.der_parts summed over windows (the STAGE1 'pooled' DER)."""
    m = f = c = s = 0.0
    for p, r in zip(preds, refs):
        T = min(len(p), len(r))
        mi, fa, co, sp = der_parts(torch.as_tensor(p[:T]).float(), torch.as_tensor(r[:T]).float())
        m, f, c, s = m + mi, f + fa, c + co, s + sp
    s = max(1.0, s)
    return {"der_pooled": round((m + f + c) / s, 4), "miss": round(m / s, 4), "fa": round(f / s, 4),
            "confusion": round(c / s, 4), "speaker_frames": int(s)}


def pyannote_der(hyps: list, refs: list[np.ndarray], collars=(0.0, 0.25, 0.5)) -> dict:
    """pyannote.metrics DER (pooled over windows) vs the frame reference rasterised to segments.
    hyps: Annotations (continuous) or (T, S) frame matrices. pyannote's collar is the TOTAL width (0.5 = +-0.25 s)."""
    from pyannote.core import Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    out = {}
    for col in collars:
        met = DiarizationErrorRate(collar=col, skip_overlap=False)
        for h, r in zip(hyps, refs):
            ha = sd.matrix_to_annotation(h, prefix="h") if isinstance(h, np.ndarray) else h
            met(sd.matrix_to_annotation(r, prefix="r"), ha, uem=Segment(0, len(r) * sd.FRAME_SEC))
        comp = met[:]
        tot = max(1e-9, comp["total"])
        out[f"collar_{col}"] = {"der": round(abs(met), 4), "miss": round(comp["missed detection"] / tot, 4),
                                "fa": round(comp["false alarm"] / tot, 4), "confusion": round(comp["confusion"] / tot, 4),
                                "total_sec": round(comp["total"], 1)}
    return out


def vad_from_matrix(mats, labels) -> dict:
    """Any-speaker activity of a diarization output as a VAD (single operating point)."""
    return sd.score_vad([m.max(1) for m in mats], labels, [0.5])["0.5"]


def trials(val):
    spk = torch.tensor([v["speaker"] for v in val])
    meet = [v["meeting"] for v in val]
    iu = torch.triu_indices(len(val), len(val), 1)
    lab = (spk[iu[0]] == spk[iu[1]]).long()
    same = torch.tensor([meet[a] == meet[b] for a, b in iu.T.tolist()])
    return iu, lab, same


def eer_block(E: torch.Tensor, val) -> dict:
    iu, lab, same = trials(val)
    E = torch.nn.functional.normalize(E.float(), dim=-1)
    s = (E[iu[0]] * E[iu[1]]).sum(-1)
    return {"eer": round(eer(s, lab), 4), "eer_within_meeting": round(eer(s[same], lab[same]), 4),
            "mean_cos_same": round(float(s[lab == 1].mean()), 3), "mean_cos_diff": round(float(s[lab == 0].mean()), 3),
            "mean_cos_same_within": round(float(s[(lab == 1) & same].mean()), 3),
            "mean_cos_diff_within": round(float(s[(lab == 0) & same].mean()), 3),
            "n_segments": len(val), "n_speakers": int(len({v["speaker"] for v in val})), "n_trials": int(len(lab)),
            "n_target": int(lab.sum()), "n_within": int(same.sum())}


# --------------------------------------------------------------------------- stages
def stage_vad(a) -> dict:
    val = diar_windows(a.n)
    labels = [v["vad"] > 0.5 for v in val]
    dur = audio_sec(val)
    res = {"n": len(val), "audio_sec": dur, "speech_frac": round(float(np.mean(np.concatenate(labels))), 4)}

    def run(name, fn, thresholds, **meta):
        t0 = time.time()
        scores = [fn(v["audio"], len(l)) for v, l in zip(val, labels)]
        sec = time.time() - t0
        res[name] = dict(meta, sec=round(sec, 1), rtf=round(sec / dur, 5), sweep=sd.score_vad(scores, labels, thresholds))
        print(name, res[name]["sec"], res[name]["sweep"].get("0.5") or res[name]["sweep"], flush=True)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    v5 = SCRATCH / "silero_vad_v5.1.2.jit"
    if not v5.exists():
        torch.hub.download_url_to_file(SILERO_V5_URL, str(v5))
    from importlib import metadata, resources
    v6 = Path(str(resources.files("silero_vad.data").joinpath("silero_vad.jit")))
    for name, path in (("silero_v5.1.2", v5), (f"silero_pip_{metadata.version('silero-vad')}", v6)):
        m = torch.jit.load(str(path)).eval()
        n16 = sum(p.numel() for n_, p in m.named_parameters() if "8k" not in n_ and "_8k" not in n_)

        def fn(x, T, m=m):
            p, hop = sd.silero_probs(m, x)
            return sd.pool_probs(p, hop, T)
        run(name, fn, THRESH, params_jit_total=sum(p.numel() for p in m.parameters()), params_16k=n16,
            frame_ms=32, lookahead_ms=0, file=str(path.name))
    for mode in range(4):
        def fn(x, T, mode=mode):
            d, hop = sd.webrtc_decisions(x, mode)
            return sd.pool_probs(d, hop, T)
        run(f"webrtc_mode{mode}", fn, [0.5], frame_ms=30, lookahead_ms=0, params=0)
    mb = sd.load_nemo_conv("nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0")

    def fn(x, T):
        p = mb.probs(x)
        return sd.pool_probs(p, mb.hop, T, offset=-mb.hop / 2)  # 20 ms frames centred on k*20 ms
    run("marblenet_frame_vad_v2", fn, THRESH, params=mb.import_info["params"], frame_ms=20,
        lookahead_ms="~1460 (offline symmetric conv receptive field)", import_info=mb.import_info)
    return res


def _load_ours(path):
    from audioforge.nemo_import import load_any
    return load_any(path, "cpu")


@torch.no_grad()
def model_pass(model, val, batch_size=8):
    """-> per window: diar probs (T, S) or None, vad probs (T,) or None (the eval_stage1 offline pass)."""
    from audioforge.data import Collate
    from audioforge.heads.turn import _diar_name
    diar = _diar_name(model)
    vad = next((k for k, v in model.head_cfg.items() if v["type"] == "frame" and v.get("key") == "vad"), None)
    col = Collate(getattr(model, "tokenizer", None))
    out_d, out_v = [], []
    for i in range(0, len(val), batch_size):
        b = col(val[i: i + batch_size])
        enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
        pd = model.heads[diar](model.head_input(diar, enc, hidden), elen).sigmoid() if diar else None
        pv = model.heads[vad].decode(model.head_input(vad, enc, hidden), elen) if vad else None
        for j in range(len(elen)):
            T = int(elen[j])
            out_d.append(pd[j, :T].numpy() if pd is not None else None)
            out_v.append(pv[j].reshape(pv.shape[1], -1)[:T, 0].numpy() if pv is not None else None)
    return out_d, out_v


def _fit(p, T):
    if len(p) >= T:
        return p[:T]
    return np.concatenate([p, np.zeros((T - len(p),) + p.shape[1:], p.dtype)])


def stage_model(a, ckpt, tag) -> dict:
    from eval_stage1 import eval_spk
    val = diar_windows(a.n)
    labels = [v["vad"] > 0.5 for v in val]
    refs = [np.asarray(v["spk_targets"], np.float32) for v in val]
    t0 = time.time()
    model = _load_ours(ckpt)
    res = {"ckpt": ckpt, "load_sec": round(time.time() - t0, 1),
           "att_context_size": list(getattr(getattr(model, "encoder", None), "att_context_size", []) or []),
           "params": int(sum(p.numel() for p in model.parameters()))}
    t0 = time.time()
    pd, pv = model_pass(model, val)
    res["pass_sec"] = round(time.time() - t0, 1)
    res["rtf"] = round(res["pass_sec"] / audio_sec(val), 5)
    if pd[0] is not None:
        hard = [(_fit(p, len(r)) > 0.5).astype(np.float32) for p, r in zip(pd, refs)]
        # eval_stage1 scores T = min(model frames, label frames); _fit pads with silence if the model is shorter
        res["diar_frame_der"] = pooled_der(hard, refs)
        res["diar_pyannote_metrics"] = pyannote_der(hard, refs)
        res["diar_as_vad"] = vad_from_matrix(hard, labels)
        print(tag, "DER", res["diar_frame_der"], res["diar_pyannote_metrics"], flush=True)
    if pv[0] is not None:
        res["vad_sweep"] = sd.score_vad([_fit(p, len(l)) for p, l in zip(pv, labels)], labels, THRESH)
        print(tag, "VAD", res["vad_sweep"]["0.5"], flush=True)
    if tag == "ours":
        for n in (64, 200):
            t0 = time.time()
            r = eval_spk(model, n, 8)
            res[f"spk_n{n}"] = dict(r, sec=round(time.time() - t0, 1))
            print(tag, "spk", n, res[f"spk_n{n}"], flush=True)
    return res


def stage_pyannote(a) -> dict | None:
    """Resumable: per-window segments cached in SCRATCH/pyannote/<pipeline>/ (json), --budget seconds per process
    (~10 s per 20 s window on 2 threads); scores are computed once every window is cached."""
    from pyannote.core import Annotation, Segment
    val = diar_windows(a.n)
    labels = [v["vad"] > 0.5 for v in val]
    refs = [np.asarray(v["spk_targets"], np.float32) for v in val]
    cache = SCRATCH / "pyannote" / a.pipeline.split("/")[-1]
    cache.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(len(val)) if not (cache / f"{i:03d}.json").exists()]
    load = 0.0
    if todo:
        from pyannote.audio import Pipeline
        t0 = time.time()
        pipe = Pipeline.from_pretrained(a.pipeline)
        load = time.time() - t0
        t_start = time.time()
        for i in todo:
            if time.time() - t_start > a.budget:
                print(f"budget reached: {len(todo) - todo.index(i)} windows left, re-run", flush=True)
                return None
            t0 = time.time()
            out = pipe({"waveform": torch.as_tensor(val[i]["audio"], dtype=torch.float32)[None], "sample_rate": sd.SR})
            ann = getattr(out, "speaker_diarization", out)  # 4.x DiarizeOutput or 3.x Annotation
            segs = [(float(s.start), float(s.end), str(lab)) for s, _, lab in ann.itertracks(yield_label=True)]
            (cache / f"{i:03d}.json").write_text(json.dumps({"sec": time.time() - t0, "segments": segs}))
    recs = [json.loads((cache / f"{i:03d}.json").read_text()) for i in range(len(val))]
    anns, mats = [], []
    for rec, r in zip(recs, refs):
        ann = Annotation()
        for k, (s, e, lab) in enumerate(rec["segments"]):
            ann[Segment(s, e), k] = lab
        anns.append(ann)
        mats.append(sd.segments_to_matrix(rec["segments"], len(r)))
    sec = float(sum(r_["sec"] for r_ in recs))
    n_spk = [len(a_.labels()) for a_ in anns]
    res = {"pipeline": a.pipeline, "load_sec": round(load, 1), "sec": round(sec, 1), "rtf": round(sec / audio_sec(val), 5),
           "diar_frame_der": pooled_der(mats, refs), "diar_pyannote_metrics_continuous": pyannote_der(anns, refs),
           "diar_pyannote_metrics_rasterised": pyannote_der(mats, refs), "diar_as_vad": vad_from_matrix(mats, labels),
           "mean_speakers_found": round(float(np.mean(n_spk)), 2), "windows_over_4_speakers": int(sum(k > 4 for k in n_spk)),
           "mean_ref_speakers": round(float(np.mean([(r.sum(0) > 0).sum() for r in refs])), 2), "cache": str(cache)}
    from importlib import metadata
    res["pyannote_audio"] = metadata.version("pyannote.audio")
    print(json.dumps(res)[:1500], flush=True)
    return res


def stage_spk(a) -> dict:
    res = {}
    sets = {n: ami_dev("asr", n) for n in (64, 200)}
    for n, val in sets.items():
        res[f"data_n{n}"] = {"n": len(val), "speakers": len({v["speaker"] for v in val}),
                             "mean_sec": round(float(np.mean([v["duration"] for v in val])), 2),
                             "audio_sec": round(audio_sec(val), 1)}

    def run(name, embed_fn, **meta):
        res[name] = dict(meta)
        for n, val in sets.items():
            t0 = time.time()
            E = embed_fn([v["audio"] for v in val])
            sec = time.time() - t0
            res[name][f"n{n}"] = dict(eer_block(E, val), sec=round(sec, 1), rtf=round(sec / audio_sec(val), 5))
            print(name, n, res[name][f"n{n}"], flush=True)

    if "titanet" in a.spk_models:
        from audioforge.nemo_import import import_titanet
        m = import_titanet()
        run("titanet_large", lambda xs: m.embed(xs), params=m.import_info["params"],
            params_without_classifier=m.import_info["params"] - m.decoder.final.weight.numel(), import_info=m.import_info)
        del m
    if "ecapa" in a.spk_models:
        from speechbrain.inference.speaker import EncoderClassifier
        m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(SCRATCH / "sb_ecapa"),
                                           run_opts={"device": "cpu"})

        @torch.no_grad()
        def emb(xs):
            return torch.cat([m.encode_batch(torch.as_tensor(np.asarray(x, np.float32))[None])[:, 0] for x in xs])
        run("speechbrain_ecapa", emb, params=int(sum(p.numel() for p in m.mods.embedding_model.parameters())))
        del m
    if "wespeaker" in a.spk_models:
        from pyannote.audio import Inference, Model
        m = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        inf = Inference(m, window="whole")

        @torch.no_grad()
        def emb(xs):
            return torch.stack([torch.as_tensor(np.asarray(inf({"waveform": torch.as_tensor(np.asarray(x, np.float32))[None],
                                                                "sample_rate": sd.SR})).reshape(-1)) for x in xs])
        run("wespeaker_resnet34_lm", emb, params=int(sum(p.numel() for p in m.parameters())))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("vad", "ours", "sortformer", "pyannote", "spk"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ckpt", default="runs/stage1_heads_pretrained.afm")
    ap.add_argument("--sortformer", default="runs/nemo_sortformer_v2.afm")
    ap.add_argument("--pipeline", default="pyannote/speaker-diarization-community-1")
    ap.add_argument("--spk-models", default="titanet,ecapa,wespeaker")
    ap.add_argument("--budget", type=float, default=480.0, help="pyannote: seconds of inference per process")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="runs/baselines_sd.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    if a.stage == "vad":
        r = stage_vad(a)
        key = "vad"
    elif a.stage == "ours":
        r, key = stage_model(a, a.ckpt, "ours"), "ours"
    elif a.stage == "sortformer":
        r, key = stage_model(a, a.sortformer, "sortformer"), "sortformer_v2"
    elif a.stage == "pyannote":
        r, key = stage_pyannote(a), "pyannote:" + a.pipeline.split("/")[-1]
        if r is None:
            return
    else:
        r, key = stage_spk(a), "spk"
    r["stage_sec"] = round(time.time() - t0, 1)
    res = json.loads(out.read_text()) if out.exists() else {}  # re-read: merge with stages run meanwhile
    if key == "spk" and "spk" in res:
        res["spk"].update(r)
    else:
        res[key] = r
    out.write_text(json.dumps(res, indent=1))
    print(f"[{key}] {r['stage_sec']}s -> {out}")


if __name__ == "__main__":
    main()
