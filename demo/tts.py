"""Narration with Qwen3-TTS-12Hz-1.7B-CustomVoice (Apache-2.0), one wav per script bullet, then joined with timings.

    scripts/dev/gate.sh $TTS_PY demo/tts.py candidates            # male voices x first ~20 s of the full script -> <out>/voice_candidates/
    scripts/dev/gate.sh $TTS_PY demo/tts.py synth --script demo/script_full.md --out narration_full \
        [--speaker Ryan] [--device mps] [--budget 540]

TTS_PY is a separate venv (APFS volume: uv venv --python 3.12 /Volumes/afdev/venvs/tts && uv pip install qwen-tts
soundfile); the repo venv keeps its own transformers. The model is read from ~/.cache/huggingface (never written).
Machine rules: MPS is the only GPU job and one at a time; every call goes through scripts/dev/gate.sh and stops
after --budget seconds (exit 3 = not finished, run again: every unit is cached, so the run resumes).

Outputs (DEMO_OUT, default /Volumes/ExternalSSD/nvidia-audio-models/demo_out; nothing large on the internal disk):
  <DEMO_OUT>/<out>.wav (24 kHz mono), <DEMO_OUT>/<out>.json ({"units": [{"scene", "text", "start", "end", "wav"}], ...})
  <DEMO_OUT>/<out>_units/NN_<speaker>.wav   per-unit cache
The candidate step writes voice_candidates/metrics.json: F0 median (autocorrelation pitch on voiced 40 ms frames),
voiced fraction, clipping, seconds per character, so the choice of voice is a measurement, not a feeling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
INSTRUCT = ("Deep, calm, intriguing male narrator. Slow, unhurried pace, warm low register, documentary voice-over, "
            "clear articulation, quiet confidence, no excitement.")
GAP_S = 0.35       # between bullets
SCENE_GAP_S = 0.7  # between scenes
MALE_SPEAKERS = "Ryan,Aiden,Uncle_Fu,Eric,Dylan"  # the CustomVoice male ids (config.json talker_config.spk_id)


def parse_script(path: Path) -> list[dict]:
    units, scene = [], "intro"
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("## scene:"):
            scene = line.split(":", 1)[1].strip()
        elif line.startswith("- "):
            raw = line[2:].strip()
            # "{Pipecat|Pipe cat}": the caption shows the first form, the voice speaks the second (pronunciation)
            import re
            spoken = re.sub(r"\{([^|}]*)\|([^}]*)\}", r"\2", raw)
            caption = re.sub(r"\{([^|}]*)\|([^}]*)\}", r"\1", raw)
            units.append({"scene": scene, "text": spoken, "caption": caption})
    return units


INSTRUCT_LIVELY = ("Warm, energetic tech presenter on stage, confident and friendly, clear articulation, natural emphasis, "
                   "a smile in the voice.")


def set_instruct(name: str):
    global INSTRUCT
    INSTRUCT = {"deep": INSTRUCT, "lively": INSTRUCT_LIVELY}.get(name, name)


def load_model(device: str):
    import torch
    from qwen_tts import Qwen3TTSModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # read the cache, never download
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    try:
        m = Qwen3TTSModel.from_pretrained(MODEL, device_map=device, dtype=dtype)
    except TypeError:
        m = Qwen3TTSModel.from_pretrained(MODEL, device_map=device, torch_dtype=dtype)
    return m


def synth_one(model, text: str, speaker: str, tempo: float = 1.0) -> tuple[np.ndarray, int]:
    wavs, sr = model.generate_custom_voice(text=text, language="English", speaker=speaker, instruct=INSTRUCT)
    w = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    w = trim(w, sr)
    if abs(tempo - 1.0) > 1e-3:
        w = stretch(w, sr, tempo)
    return w, sr


def stretch(w: np.ndarray, sr: int, tempo: float) -> np.ndarray:
    """Pitch-preserving tempo change with ffmpeg's atempo (2 threads); the narrator stays deep, only the pace moves."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.wav", Path(td) / "b.wav"
        sf.write(a, w, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-i", str(a), "-filter:a", f"atempo={tempo:.3f}", str(b)], check=True)
        out, _ = sf.read(b, dtype="float32")
    return out


def trim(w: np.ndarray, sr: int, thr_db: float = -45.0, pad_s: float = 0.08) -> np.ndarray:
    n = int(sr * 0.01)
    k = len(w) // n
    if k == 0:
        return w
    rms = np.sqrt(np.mean(w[: k * n].reshape(k, n) ** 2, 1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    idx = np.nonzero(db > thr_db)[0]
    if not len(idx):
        return w
    a = max(0, idx[0] * n - int(pad_s * sr))
    b = min(len(w), (idx[-1] + 1) * n + int(pad_s * sr))
    return w[a:b]


def f0_stats(w: np.ndarray, sr: int, fmin=60.0, fmax=300.0) -> dict:
    """Median F0 (Hz) over voiced frames: normalized autocorrelation peak in [fmin, fmax] on 40 ms frames, hop 10 ms."""
    n, hop = int(sr * 0.04), int(sr * 0.01)
    lo, hi = int(sr / fmax), int(sr / fmin)
    rms_all = np.sqrt(np.mean(w ** 2) + 1e-12)
    f0s = []
    for s in range(0, len(w) - n, hop):
        fr = w[s: s + n]
        if np.sqrt(np.mean(fr ** 2)) < 0.3 * rms_all:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[n - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[lo:hi]
        p = int(np.argmax(seg)) + lo
        if seg.max() > 0.5:
            f0s.append(sr / p)
    f0s = np.array(f0s)
    return {"f0_median_hz": round(float(np.median(f0s)), 1) if len(f0s) else None,
            "f0_std_hz": round(float(np.std(f0s)), 1) if len(f0s) else None,
            "f0_p10_hz": round(float(np.percentile(f0s, 10)), 1) if len(f0s) else None,
            "f0_p90_hz": round(float(np.percentile(f0s, 90)), 1) if len(f0s) else None,
            "voiced_frames": int(len(f0s))}


def metrics(w: np.ndarray, sr: int, text: str) -> dict:
    m = f0_stats(w, sr)
    m.update({"dur_s": round(len(w) / sr, 2), "sec_per_char": round(len(w) / sr / max(1, len(text)), 4),
              "peak": round(float(np.abs(w).max()), 3), "clipped_frac": round(float(np.mean(np.abs(w) > 0.99)), 5),
              "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-9)), 1)})
    return m


def cmd_candidates(a):
    units = parse_script(ROOT / a.script)
    text_units, total = [], 0
    for u in units:  # first ~20 s worth (about 15 chars/s at a slow pace -> ~300 chars)
        text_units.append(u["text"]); total += len(u["text"])
        if total > 300:
            break
    out = OUT_DIR / ("voice_candidates" if a.instruct == "deep" else f"voice_candidates_{a.instruct}")
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "metrics.json"
    res = json.loads(mpath.read_text())["speakers"] if mpath.exists() else {}
    model = load_model(a.device)
    t_start = time.time()
    for spk in a.speakers.split(","):
        if spk in res and (out / f"{spk}.wav").exists() and not a.force:
            continue
        if time.time() - t_start > a.budget:
            print("budget reached; run again to continue", flush=True); sys.exit(3)
        t0 = time.time()
        parts = []
        for t in text_units:
            w, sr = synth_one(model, t, spk)
            parts.append(w); parts.append(np.zeros(int(GAP_S * sr), np.float32))
        w = np.concatenate(parts)
        sf.write(out / f"{spk}.wav", w, sr)
        res[spk] = metrics(w, sr, " ".join(text_units)) | {"synth_sec": round(time.time() - t0, 1)}
        print(spk, res[spk], flush=True)
        mpath.write_text(json.dumps({"instruct": INSTRUCT, "text": text_units, "speakers": res}, indent=1))
    print("candidates ->", out)


def cmd_synth(a):
    units = parse_script(ROOT / a.script)
    out = OUT_DIR / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    udir = OUT_DIR / (out.name + "_units")
    udir.mkdir(parents=True, exist_ok=True)
    model = None
    sr = None
    pieces, t = [], 0.0
    prev_scene = None
    t_start = time.time()
    for i, u in enumerate(units):
        key = f"{u['text']}|{a.tempo:.3f}|{INSTRUCT}"
        h = hashlib.sha1(key.encode()).hexdigest()[:8]
        cache = udir / f"{i:02d}_{a.speaker}_{h}.wav"
        if not cache.exists() and a.retake_set.isdisjoint({i}):  # same line elsewhere in the script: reuse its take
            same = sorted(udir.glob(f"*_{a.speaker}_{h}.wav"))
            if same:
                cache = same[0]  # text-keyed: an edited line re-synthesises
        w = sr = None
        if cache.exists() and not a.force and i not in a.retake_set:
            w, sr = sf.read(cache, dtype="float32")
        tries = 0
        while w is None or (len(w) / sr / len(u["text"]) > a.max_spc and tries < a.retries):
            # missing, or delivered too slowly (sampling sometimes drawls a line): synthesise again, keep the shortest
            if time.time() - t_start > a.budget:
                print(f"budget reached after unit {i - 1}; run again to continue", flush=True); sys.exit(3)
            if model is None:
                model = load_model(a.device)
            w2, sr = synth_one(model, u["text"], a.speaker, a.tempo)
            if w is None or len(w2) < len(w):
                w = w2
                sf.write(cache, w, sr)
            tries += 1
        if prev_scene is not None:
            gap = SCENE_GAP_S if u["scene"] != prev_scene else GAP_S
            pieces.append(np.zeros(int(gap * sr), np.float32)); t += gap
        u["start"] = round(t, 3)
        pieces.append(w); t += len(w) / sr
        u["end"] = round(t, 3)
        u["wav"] = str(cache)
        u["metrics"] = metrics(w, sr, u["text"])
        prev_scene = u["scene"]
        print(f"{i:02d} {u['start']:7.2f}-{u['end']:7.2f} f0 {u['metrics']['f0_median_hz']} {u['text'][:70]}", flush=True)
    pieces.append(np.zeros(int(0.8 * sr), np.float32)); t += 0.8
    w = np.concatenate(pieces)
    peak = np.abs(w).max()
    if peak > 0.95:
        w = w * (0.95 / peak)
    sf.write(str(out) + ".wav", w, sr)
    Path(str(out) + ".json").write_text(json.dumps({"model": MODEL, "license": "Apache-2.0", "speaker": a.speaker, "tempo": a.tempo,
                                                    "instruct": INSTRUCT, "sr": sr, "duration_s": round(t, 3),
                                                    "script": a.script, "units": units}, indent=1))
    print("wrote", str(out) + ".wav", round(t, 1), "s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates"); c.add_argument("--script", default="demo/script_full.md")
    c.add_argument("--speakers", default=MALE_SPEAKERS); c.add_argument("--device", default="mps")
    c.add_argument("--budget", type=float, default=480); c.add_argument("--force", action="store_true")
    c.add_argument("--instruct", default="deep", help="deep | lively | free text")
    s = sub.add_parser("synth"); s.add_argument("--script", required=True); s.add_argument("--out", required=True)
    s.add_argument("--speaker", default="Dylan"); s.add_argument("--device", default="mps"); s.add_argument("--force", action="store_true")
    s.add_argument("--budget", type=float, default=480, help="seconds of synthesis per call (machine rule: < 10 min); exit 3 = run again")
    s.add_argument("--tempo", type=float, default=1.12, help="pitch-preserving pace factor (ffmpeg atempo); Dylan's natural pace is ~0.10 s/char")
    s.add_argument("--max-spc", type=float, default=0.115, help="re-synthesise a unit slower than this many seconds per character (after --tempo)")
    s.add_argument("--retries", type=int, default=2)
    s.add_argument("--retake", default="", help="comma-separated unit indices to synthesise again (a Whisper check mis-heard them)")
    s.add_argument("--instruct", default="deep", help="deep | lively | free text")
    a = ap.parse_args()
    set_instruct(a.instruct)
    a.retake_set = {int(x) for x in getattr(a, "retake", "").split(",") if x.strip()}
    {"candidates": cmd_candidates, "synth": cmd_synth}[a.cmd](a)


if __name__ == "__main__":
    main()
