"""Whisper-small check of every narration unit: transcribe each unit wav, compare with the script text after
normalisation (lowercase, punctuation off, numbers spelled out, "audio forge" joined). Lines with any word error are
listed for a re-take.

    scripts/dev/gate.sh .venv/bin/python demo/v5/check_voice.py $DEMO_OUT/narration_v5.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def spell(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + spell(n % 100))
    return spell(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + spell(n % 1000))


def norm(s: str) -> list[str]:
    s = s.replace("%", " percent").lower().replace("audio forge", "audioforge").replace("turnends", "turn ends").replace("turnover", "turn over").replace("audio-forge", "audioforge").replace("-", " ").replace("pipecat", "pipe cat").replace("livekit", "live kit").replace("dye arizer", "diarizer").replace("parra keet", "parakeet").replace("pair a keet", "parakeet")
    s = re.sub(r"(\d+)\.(\d+)", lambda m: spell(int(m.group(1))) + " point " + " ".join(ONES[int(c)] for c in m.group(2)), s)
    s = re.sub(r"\d+", lambda m: spell(int(m.group(0))), s)
    s = re.sub(r"[^a-z' ]", " ", s)
    return [w.replace("'", "") for w in s.split() if w.strip("'")]


def edits(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def main():
    from faster_whisper import WhisperModel
    snaps = sorted((Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots").iterdir())
    path = str(snaps[-1])  # read the cache directly (never written)
    wm = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
    units = json.loads(Path(sys.argv[1]).read_text())["units"]
    bad, tot_e, tot_n = [], 0, 0
    for i, u in enumerate(units):
        segs, _ = wm.transcribe(u["wav"], language="en", beam_size=5)
        hyp = " ".join(s.text for s in segs).strip()
        r, h = norm(u["text"]), norm(hyp)
        e = edits(r, h)
        tot_e += e; tot_n += len(r)
        print(f"{i:02d} [{u['scene']}] errors {e}/{len(r)}\n   script: {u['text']}\n   heard : {hyp}", flush=True)
        if e:
            bad.append(i)
    print(f"total {tot_e}/{tot_n} word errors; re-take units: {bad}")


if __name__ == "__main__":
    main()
