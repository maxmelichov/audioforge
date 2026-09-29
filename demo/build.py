"""Build the showcase videos end to end (one command; every step idempotent and resumable).

    PYTHONPATH=. .venv/bin/python demo/build.py [--skip-tts] [--skip-events] [--only full|short|square] [--speaker Ryan]

Steps (see demo/README.md):
  1. demo/extract_numbers.py                  -> demo/numbers.json (every on-screen number, read from runs/*.json)
  2. demo/tts.py synth  (TTS venv, MPS, gated) -> $DEMO_OUT/narration_{full,short}.{wav,json}   (Qwen3-TTS 1.7B CustomVoice)
  3. audioforge.serve + demo/record_events.py  -> demo/events/*.json   (real server, real clips, real event timings)
  4. demo/render.py (segments, then mux)      -> $DEMO_OUT/showcase_full.mp4 (16:9), showcase.mp4 (16:9, 60-90 s),
                                                showcase_square.mp4 (1:1, 60-90 s), *_720p.mp4 copies
Machine rules (scripts/dev/gate.sh): every model launch (TTS, server) and every render goes through the gate; the TTS on
MPS is the only GPU job and runs alone; each call is bounded (< 10 min) and re-run until it reports done; ffmpeg at
2 threads; every large file lives on the external SSD (DEMO_OUT), not in the repo.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
OUT_DIR = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
GATE = str(ROOT / "scripts" / "dev" / "gate.sh")
PY = str(ROOT / ".venv" / "bin" / "python")
TTS_PY = os.environ.get("TTS_PY", "/Volumes/afdev/venvs/tts/bin/python")  # APFS volume: exFAT cannot hold a venv
PORT = int(os.environ.get("DEMO_PORT", "8791"))
SERVER = [PY, "-m", "audioforge.serve", "--asr", "runs/stage1_served.afm", "--diar", "runs/nemo_nemotron3_diar.afm",
          "--diar-pool", "max", "--diar-spks", "4", "--threads", "2", "--enroll", "after_agent_arm",
          "--silero", "data/silero/silero_vad_v5.onnx", "--final-asr", "tdt_v3", "--lid", "ambernet",
          "--debug-fields", "--port", str(PORT)]
CLIPS = [  # (clip wav, policy, agent_end times from the clip labels = the other party's floor-turn ends, reference text)
    ("demo/clips/oto_f619bed3.wav", "hybrid_dyn", "9.7,27.1,37.9", None),
    ("demo/clips/ami_IS1008b_003.wav", "hybrid_dyn", "3.12",
     "to inflate something ou made out yeah of titanium the the though i'm sorry because uh the last meeting we supposed to "
     "discuss about the financial thing uh let me go quickly maybe if i can go back i know the project plan and the budget"),
    ("demo/clips/libri_1089-134686-0000.wav", "hybrid_dyn", "",
     "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled "
     "out in thick peppered flour fattened sauce"),
]
GUARD = "audioforge.*train|tsvad.py train|train.py"
NSEG = {"full": 6, "short": 2}  # render segments per cut (each well under 10 min)


def sh(cmd, check=True, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=check, cwd=ROOT, env={**os.environ, "PYTHONPATH": "."}, **kw)


def gated(cmd, **kw):
    return sh([GATE] + cmd, **kw)


def busy(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def step_numbers():
    sh([PY, "demo/extract_numbers.py"], stdout=subprocess.DEVNULL)


def step_tts(speaker: str):
    env = {**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.6", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.5", "PYTHONPATH": "."}
    for cut in ("full", "short"):
        for _ in range(12):  # each call synthesises for <= --budget seconds and exits 3 when more remains
            r = gated([TTS_PY, "demo/tts.py", "synth", "--script", f"demo/script_{cut}.md", "--out", f"narration_{cut}",
                       "--speaker", speaker, "--device", "mps", "--budget", "480"], check=False, env=env)
            if r.returncode == 0:
                break
            if r.returncode != 3:
                sys.exit(f"tts failed ({r.returncode})")
        else:
            sys.exit("tts did not finish in 12 calls")


def step_events():
    if busy(f"audioforge.serve.*--port {PORT}"):
        sys.exit(f"an audioforge.serve is already on port {PORT}")
    if busy(GUARD):
        print("note: a training job is running; the server (CPU, 2 threads) waits at the gate like every other launch", flush=True)
    (DEMO / "events").mkdir(exist_ok=True)
    log = open(DEMO / "events" / "server.log", "w")
    srv = subprocess.Popen([GATE] + SERVER, cwd=ROOT, env={**os.environ, "PYTHONPATH": "."}, stdout=log, stderr=subprocess.STDOUT)
    try:
        import socket
        for _ in range(900):
            time.sleep(1)
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", PORT)) == 0:
                    break
            if srv.poll() is not None:
                sys.exit("server exited; see demo/events/server.log")
        time.sleep(5)
        for wav, policy, ae, ref in CLIPS:
            out = DEMO / "events" / (Path(wav).stem + ".json")
            cmd = [PY, "demo/record_events.py", wav, "--out", str(out), "--policy", policy, "--agent-end-s", ae,
                   "--url", f"ws://127.0.0.1:{PORT}"]
            if ref:
                cmd += ["--ref-text", ref]
            sh(cmd)
            time.sleep(2)
    finally:
        srv.terminate()
        try:
            srv.wait(20)
        except subprocess.TimeoutExpired:
            srv.kill()


def step_render(only=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [("full", "1920x1080", "showcase_full.mp4"), ("short", "1920x1080", "showcase.mp4"), ("short", "1080x1080", "showcase_square.mp4")]
    for cut, size, name in jobs:
        tag = "square" if size == "1080x1080" else cut
        if only and tag != only:
            continue
        out = OUT_DIR / name
        n = NSEG[cut]
        for i in range(n):
            gated([PY, "demo/render.py", "--cut", cut, "--size", size, "--out", str(out), "--seg", str(i), "--nseg", str(n)])
        gated([PY, "demo/render.py", "--cut", cut, "--size", size, "--out", str(out), "--mux", "--nseg", str(n)])
        mb = out.stat().st_size / 1e6
        print(f"{out}: {mb:.1f} MB")
        if size == "1920x1080":  # a lighter upload copy
            small = out.with_name(out.stem + "_720p.mp4")
            sh(["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-i", str(out), "-vf", "scale=1280:720", "-c:v", "libx264", "-threads", "2",
                "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(small)])
            print(f"{small}: {small.stat().st_size / 1e6:.1f} MB")
    # poster frames (small PNGs, the only render artefacts kept in the repo)
    (DEMO / "stills").mkdir(exist_ok=True)
    for cut, size, name, t in (("short", "1920x1080", "poster_16x9", 0.0), ("short", "1080x1080", "poster_1x1", 0.0), ("full", "1920x1080", "architecture", None)):
        if only and (("square" if size == "1080x1080" else cut) != only):
            continue
        if t is None:  # middle of the architecture scene
            import json
            tl = json.loads((OUT_DIR / "showcase_full.timeline.json").read_text())
            sc = next(s for s in tl["scenes"] if s["name"] == "architecture")
            t = sc["end"] - 1.0
        tmp = OUT_DIR / "scratch" / name
        gated([PY, "demo/render.py", "--cut", cut, "--size", size, "--out", str(tmp), "--preview", str(t)])
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp) + ".png", "-vf", "scale=iw/2:-1", str(DEMO / "stills" / (name + ".png"))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tts", action="store_true")
    ap.add_argument("--skip-events", action="store_true")
    ap.add_argument("--only", choices=["full", "short", "square"], default=None)
    ap.add_argument("--speaker", default=os.environ.get("DEMO_SPEAKER", "Dylan"))
    a = ap.parse_args()
    step_numbers()
    if not a.skip_tts:
        step_tts(a.speaker)
    if not a.skip_events:
        step_events()
    step_render(a.only)


if __name__ == "__main__":
    main()
