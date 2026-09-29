"""Build the served checkpoint: the trail6 turn model with the relational speaker head transplanted.

    PYTHONPATH=. .venv/bin/python scripts/research/make_served_model.py [--check-wer 100] [--out runs/stage1_served.afm]

runs/stage1_served.afm = runs/stage1_turn_v3_trail6.afm (encoder, RNNT, CTC, VAD, EOU, diar and turn heads) with
``heads.spk`` copied from runs/stage1_spk_relational.afm (research/SPK_HEAD.md recommendation: block-4 relational
head, ``from_layers: [3]``, 14.4 % AMI / 5.2 % ICSI within-meeting EER at n = 64 vs 32 % for the all-layer stage-1
head). The all-layer mix parameter ``layer_mix.spk`` is dropped (a single tap has none). Every other tensor is
asserted bit-identical to trail6, the speaker tensors bit-identical to the relational checkpoint, and the config
differs in ``heads.spk`` only. ``--check-wer N`` transcribes the first N LibriSpeech test-clean utterances
(data/librispeech/test-clean-first200.jsonl) with both models and asserts identical hypotheses (and reports the WER).
Speaker EER of the result: ``scripts/research/spk_head.py eval --ckpt runs/stage1_served.afm --tag stage1_served``.

``--vad-head H`` builds the v2 checkpoint instead (research/VAD_SINGLE.md): ``--base`` (default runs/stage1_served.afm)
with ``heads.vad`` replaced by the single-tap head in H (scripts/research/vad_single.py train -> <SSD>/runs/vad_single/
L<k>/head.pt; ``from_layers: [k]``) and ``layer_mix.vad`` dropped; every other tensor bit-identical to the base and the
config differs in ``heads.vad`` only:

    PYTHONPATH=. .venv/bin/python scripts/research/make_served_model.py \
        --vad-head /Volumes/ExternalSSD/nvidia-audio-models/runs/vad_single/L3/head.pt --out runs/stage1_served_v2.afm"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audioforge.train import load_model, save_model  # noqa: E402

TURN_CKPT = "runs/stage1_turn_v3_trail6.afm"
SPK_CKPT = "runs/stage1_spk_relational.afm"
OUT = "runs/stage1_served.afm"
LIBRI = ROOT / "data/librispeech/test-clean-first200.jsonl"
SPK_PREFIX = "heads.spk."
MIX_KEY = "layer_mix.spk"
BASE = "runs/stage1_served.afm"
VAD_PREFIX = "heads.vad."
VAD_MIX = "layer_mix.vad"
TRAIN_ONLY_KEYS = ("distill",)  # loss-only keys of the speaker recipe; the head's structure keys are all kept


def transplant(turn_path: str, spk_path: str) -> tuple[dict, dict, dict]:
    """(config, state_dict, report) of the served model: turn checkpoint + the speaker checkpoint's heads.spk."""
    turn, spk = load_model(turn_path, "cpu"), load_model(spk_path, "cpu")
    sd_t, sd_s = turn.state_dict(), spk.state_dict()
    scfg = dict(spk.cfg["heads"]["spk"])
    assert scfg.get("from_layers") == [3], f"expected a block-4 tap (from_layers [3]) in {spk_path}: {scfg}"
    new_spk = {k: v for k, v in scfg.items() if k not in TRAIN_ONLY_KEYS}
    new_spk["weight"] = turn.cfg["heads"]["spk"].get("weight", 1.0)
    cfg = json.loads(json.dumps(turn.cfg))  # deep copy
    cfg["heads"]["spk"] = new_spk
    sd = {}
    n_turn = n_spk = 0
    for k, v in sd_t.items():
        if k == MIX_KEY:
            continue
        if k.startswith(SPK_PREFIX):
            continue
        sd[k] = v.clone()
        n_turn += 1
    for k, v in sd_s.items():
        if k.startswith(SPK_PREFIX):
            sd[k] = v.clone()
            n_spk += 1
    rep = {"turn_ckpt": turn_path, "spk_ckpt": spk_path, "tensors_from_turn": n_turn, "tensors_from_spk": n_spk,
           "dropped": [MIX_KEY] if MIX_KEY in sd_t else [], "spk_cfg": new_spk,
           "spk_cfg_before": turn.cfg["heads"]["spk"], "heads": list(cfg["heads"])}
    return cfg, sd, rep


def verify(out_path: str, turn_path: str, spk_path: str) -> dict:
    """Reload the archive and assert: non-speaker tensors == turn ckpt, speaker tensors == spk ckpt, no layer_mix.spk,
    config differs in heads.spk only."""
    served, turn, spk = load_model(out_path, "cpu"), load_model(turn_path, "cpu"), load_model(spk_path, "cpu")
    sd, sd_t, sd_s = served.state_dict(), turn.state_dict(), spk.state_dict()
    assert MIX_KEY not in sd, "layer_mix.spk must be gone"
    assert set(sd) == (set(sd_t) - {MIX_KEY}), (set(sd) ^ (set(sd_t) - {MIX_KEY}))
    same_turn = [k for k in sd if not k.startswith(SPK_PREFIX) and torch.equal(sd[k], sd_t[k])]
    diff_turn = [k for k in sd if not k.startswith(SPK_PREFIX) and not torch.equal(sd[k], sd_t[k])]
    assert not diff_turn, f"non-speaker tensors differ from {turn_path}: {diff_turn[:5]}"
    spk_keys = [k for k in sd if k.startswith(SPK_PREFIX)]
    bad = [k for k in spk_keys if not torch.equal(sd[k], sd_s[k])]
    assert spk_keys and not bad, f"speaker tensors differ from {spk_path}: {bad[:5]}"
    ct, cs = json.loads(json.dumps(turn.cfg)), json.loads(json.dumps(served.cfg))
    ct["heads"].pop("spk"), cs["heads"].pop("spk")
    assert ct == cs, "config differs outside heads.spk"
    assert served.layer_tap["spk"] == [3] and "spk" not in served.layer_mix
    return {"identical_non_spk_tensors": len(same_turn), "spk_tensors_from_relational": len(spk_keys),
            "layer_weights_spk": [round(float(x), 3) for x in served.layer_weights("spk")],
            "spk_head_params": int(sum(p.numel() for p in served.heads["spk"].parameters()))}


def swap_vad(base_path: str, head_path: str) -> tuple[dict, dict, dict]:
    """(config, state_dict, report): the base checkpoint with heads.vad replaced by the head file's single-tap head."""
    base = load_model(base_path, "cpu")
    blob = torch.load(head_path, map_location="cpu", weights_only=False)
    hcfg = dict(blob["cfg"])
    fl = hcfg.get("from_layers")
    assert isinstance(fl, list) and len(fl) == 1, f"expected a single-layer tap in {head_path}: {hcfg}"
    assert hcfg["type"] == "frame" and hcfg.get("key") == "vad", hcfg
    cfg = json.loads(json.dumps(base.cfg))
    before = cfg["heads"]["vad"]
    cfg["heads"]["vad"] = hcfg
    sd = {k: v.clone() for k, v in base.state_dict().items() if not k.startswith(VAD_PREFIX) and k != VAD_MIX}
    n_base = len(sd)
    for k, v in blob["state_dict"].items():
        sd[VAD_PREFIX + k] = v.clone()
    rep = {"base_ckpt": base_path, "vad_head": head_path, "tensors_from_base": n_base,
           "tensors_from_head": len(blob["state_dict"]), "dropped": [VAD_MIX] if VAD_MIX in base.state_dict() else [],
           "vad_cfg": hcfg, "vad_cfg_before": before, "heads": list(cfg["heads"])}
    return cfg, sd, rep


def verify_vad(out_path: str, base_path: str, head_path: str) -> dict:
    """Reload: non-VAD tensors == base, heads.vad == the head file, no layer_mix.vad, config differs in heads.vad only."""
    served, base = load_model(out_path, "cpu"), load_model(base_path, "cpu")
    blob = torch.load(head_path, map_location="cpu", weights_only=False)
    sd, sd_b = served.state_dict(), base.state_dict()
    assert VAD_MIX not in sd, "layer_mix.vad must be gone"
    assert set(sd) == set(sd_b) - {VAD_MIX}, set(sd) ^ (set(sd_b) - {VAD_MIX})
    diff = [k for k in sd if not k.startswith(VAD_PREFIX) and not torch.equal(sd[k], sd_b[k])]
    assert not diff, f"non-VAD tensors differ from {base_path}: {diff[:5]}"
    bad = [k for k, v in blob["state_dict"].items() if not torch.equal(sd[VAD_PREFIX + k], v)]
    assert not bad, bad
    cb, cs = json.loads(json.dumps(base.cfg)), json.loads(json.dumps(served.cfg))
    cb["heads"].pop("vad"), cs["heads"].pop("vad")
    assert cb == cs, "config differs outside heads.vad"
    k = blob["cfg"]["from_layers"][0]
    assert served.layer_tap["vad"] == [k] and "vad" not in served.layer_mix
    return {"identical_non_vad_tensors": sum(1 for k_ in sd if not k_.startswith(VAD_PREFIX)),
            "vad_layer": int(k), "vad_head_params": int(sum(p.numel() for p in served.heads["vad"].parameters()))}


@torch.no_grad()
def check_wer(out_path: str, turn_path: str, n: int, batch_size: int = 8) -> dict:
    """First n LibriSpeech test-clean utterances: identical hypotheses for both models, WER reported."""
    import numpy as np
    from audioforge.data import load_wav
    from audioforge.metrics import wer
    from audioforge.teachers import normalize_text
    rows = [json.loads(l) for l in LIBRI.read_text().splitlines() if l.strip()][:n]
    audios = [load_wav(str(ROOT / r["audio_filepath"]), 16000).astype(np.float32) for r in rows]
    refs = [normalize_text(r["text"]) for r in rows]
    res = {}
    for tag, path in (("served", out_path), ("trail6", turn_path)):
        m = load_model(path, "cpu")
        t0 = time.time()
        hyps = []
        for i in range(0, len(audios), batch_size):
            hyps += m.transcribe(audios[i: i + batch_size])
        hyps = [normalize_text(h) for h in hyps]
        res[tag] = {"wer": round(wer(refs, hyps), 4), "n": len(rows), "sec": round(time.time() - t0, 1), "hyps": hyps}
        del m
    assert res["served"]["hyps"] == res["trail6"]["hyps"], "transcripts differ between served and trail6"
    return {"n": len(rows), "wer_served": res["served"]["wer"], "wer_trail6": res["trail6"]["wer"],
            "identical_hyps": True, "audio_sec": round(sum(r["duration"] for r in rows), 1),
            "sec": {k: v["sec"] for k, v in res.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--turn", default=TURN_CKPT)
    ap.add_argument("--spk", default=SPK_CKPT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--vad-head", default=None, help="build the v2 checkpoint: --base with this single-tap VAD head")
    ap.add_argument("--base", default=BASE, help="--vad-head: the checkpoint whose heads.vad is replaced")
    ap.add_argument("--check-wer", type=int, default=0, help="LibriSpeech test-clean utterances to transcribe")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json", default=None, help="write the report here")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    from audioforge.model import SpeechModel
    if a.vad_head:
        assert Path(a.out).resolve() != Path(a.base).resolve(), "--vad-head: write a new checkpoint, not over --base"
        cfg, sd, rep = swap_vad(a.base, a.vad_head)
        src = load_model(a.base, "cpu")
        model = SpeechModel(cfg, src.tokenizer)
        model.load_state_dict(sd, strict=True)
        save_model(model.eval(), a.out)
        rep["out"] = a.out
        rep["verify"] = verify_vad(a.out, a.base, a.vad_head)
        print(json.dumps(rep, indent=1), flush=True)
        if a.check_wer:
            rep["wer"] = check_wer(a.out, a.base, a.check_wer)
            print(json.dumps(rep["wer"]), flush=True)
        rep["sec"] = round(time.time() - t0, 1)
        if a.json:
            Path(a.json).write_text(json.dumps(rep, indent=1))
        print(f"[make_served_model] wrote {a.out} in {rep['sec']}s", flush=True)
        return
    cfg, sd, rep = transplant(a.turn, a.spk)
    src = load_model(a.turn, "cpu")
    model = SpeechModel(cfg, src.tokenizer)
    model.load_state_dict(sd, strict=True)
    save_model(model.eval(), a.out)
    rep["out"] = a.out
    rep["verify"] = verify(a.out, a.turn, a.spk)
    print(json.dumps({k: v for k, v in rep.items()}, indent=1), flush=True)
    if a.check_wer:
        rep["wer"] = check_wer(a.out, a.turn, a.check_wer)
        print(json.dumps(rep["wer"]), flush=True)
    rep["sec"] = round(time.time() - t0, 1)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    print(f"[make_served_model] wrote {a.out} in {rep['sec']}s", flush=True)


if __name__ == "__main__":
    main()
