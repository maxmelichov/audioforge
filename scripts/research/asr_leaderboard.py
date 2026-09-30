"""WER and RTFx the way the Open ASR Leaderboard reports them (research/METRICS.md, runs/asr_leaderboard.json).

Sets: LibriSpeech test-clean (2620 utterances) and test-other (2939), full, from data/librispeech/LibriSpeech.
Scoring: Whisper's EnglishTextNormalizer on reference and hypothesis (the leaderboard's normalizer), corpus WER.
RTFx = total audio seconds / total decode seconds (model load excluded), CPU, 2 threads.

Systems:
  served         the served 115M streaming model's transcript: runs/stage1_served.afm (the ASR tensors of
                 stage1_served_v2 are bit-identical), RNNT head, att_context [70, 1] = 160 ms chunks; m.transcribe runs
                 the masked offline forward, which equals chunk-by-chunk cache-aware streaming (tests/test_streaming.py),
                 batch 4
  whisper_small  faster-whisper small, int8, greedy, en, no timestamps, no VAD (the default stacks' STT model), batch 1

Stages (each <= --budget s, resumable, through scripts/dev/gate.sh):
  run --system S --set test-clean|test-other      -> <scratch>/metrics/asr/<system>_<set>.jsonl
  report                                          -> runs/asr_leaderboard.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
WORK = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/metrics/asr")
LS = ROOT / "data" / "librispeech" / "LibriSpeech"
SETS = ("test-clean", "test-other")
SYSTEMS = ("served", "whisper_small")


def manifest(set_name: str) -> list[dict]:
    out = []
    for tr in sorted((LS / set_name).glob("*/*/*.trans.txt")):
        for line in tr.read_text().splitlines():
            uid, text = line.split(" ", 1)
            out.append({"id": uid, "path": str(tr.parent / f"{uid}.flac"), "text": text})
    return out


def cmd_run(a):
    import final_asr as F

    from audioforge.data import load_wav
    WORK.mkdir(parents=True, exist_ok=True)
    man = manifest(a.set)
    p = WORK / f"{a.system}_{a.set}.jsonl"
    done = {json.loads(x)["id"] for x in p.read_text().splitlines()} if p.exists() else set()
    todo = [m for m in man if m["id"] not in done]
    if not todo:
        print(f"{a.system} {a.set}: done ({len(man)})"); return
    fn, bs = F.make_system(a.system)
    if not done:
        fn([load_wav(todo[0]["path"], 16000)[:48000]])  # warm-up, not timed
    t_start, n = time.time(), 0
    with p.open("a") as f:
        while todo and time.time() - t_start < a.budget:
            batch, todo = todo[:bs], todo[bs:]
            xs = [load_wav(m["path"], 16000).astype(np.float32) for m in batch]
            t0 = time.perf_counter()
            hyps = fn(xs)
            dt = time.perf_counter() - t0
            for m, x, h in zip(batch, xs, hyps):
                f.write(json.dumps({"id": m["id"], "hyp": h, "sec": dt / len(batch), "audio_sec": len(x) / 16000}) + "\n")
            f.flush()
            n += len(batch)
    print(f"{a.system} {a.set}: +{n}, {len(man) - len(todo)}/{len(man)}" + ("; done" if not todo else " (re-run)"), flush=True)


def cmd_report(a):
    import final_asr as F
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    from audioforge.metrics import edit_distance
    en = EnglishTextNormalizer(json.loads(Path(F._snap("openai/whisper-tiny"), "normalizer.json").read_text()))
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "protocol": __doc__, "results": {}}
    for s in SYSTEMS:
        for st in SETS:
            p = WORK / f"{s}_{st}.jsonl"
            if not p.exists():
                continue
            man = {m["id"]: m for m in manifest(st)}
            recs = [json.loads(x) for x in p.read_text().splitlines()]
            err = n = 0
            for r in recs:
                ref, hyp = en(man[r["id"]]["text"]).split(), en(r["hyp"]).split()
                err += edit_distance(ref, hyp); n += len(ref)
            aud, sec = sum(r["audio_sec"] for r in recs), sum(r["sec"] for r in recs)
            out["results"][f"{s}|{st}"] = {"n_utts": len(recs), "n_total": len(man), "complete": len(recs) == len(man),
                                           "wer_pct": round(100 * err / n, 2), "ref_words": n,
                                           "rtfx": round(aud / sec, 1), "audio_h": round(aud / 3600, 2)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["results"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", choices=SYSTEMS, required=True)
    r.add_argument("--set", choices=SETS, required=True)
    r.add_argument("--budget", type=float, default=500)
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(ROOT / "runs" / "asr_leaderboard.json"))
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
