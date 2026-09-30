"""Audio-only preview of the four example shots straight from the AUDIO manifest (no picture, no data.json):
intro line -> clip (L, then R for the split interrupt; L+R once otherwise) with the agent lines of the manifest ->
outro line, a narration-only shot before and after (music steadiness), and the end fade. Runs audio.mix + qa_audio.

    .venv/bin/python demo/archive/v5/audio/preview.py   ->  $DEMO_OUT/v5_data/clips/preview_mix.wav, preview_qa.json
"""
import json
import sys
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import audio as A  # noqa: E402

OUT = A.DEMO_OUT
SR = 48000
PLAN = [("pre", ["chain"], None), ("ex_call_today", ["call_1", "P", "call_2"], "P"),
        ("ex_interrupt", ["ex_int_1", "LR", "ex_int_3"], "LR"), ("ex_words", ["words_1", "LR", "words_2"], "LR"),
        ("ex_room", ["room_1", "C", "room_2"], "C"), ("post", ["live_interrupts", "live_first", "chart_icsi"], None), ("end", ["end"], None)]


def main():
    man = json.loads((OUT / "v5_data" / "clips_manifest.json").read_text())
    units = {u["scene"]: u for u in json.loads((OUT / "narration_v5.json").read_text())["units"]}
    shots, t = [], 0.0
    for name, segs, _ in PLAN:
        sh = {"name": name, "start": t, "voice": [], "clips": [], "agents": []}
        lt = 0.3
        m = man["clips"].get(name)
        for sg in segs:
            if sg in ("L", "R", "LR", "C", "P"):
                t0, t1 = m["span"]
                lt += 0.25
                sh["clips"].append({"start": t + lt, "clip": name, "t0": t0, "t1": t1})
                over = t1 - t0
                sides = ["L", "R"] if sg == "LR" else ([sg] if sg in ("L", "R", "P") else [])
                for side in sides:
                    for a in m["agent_voice"].get(side, []):
                        if a.get("line"):
                            sh["agents"].append({"start": t + lt + a["t"] - t0, "wav": a["wav"], "dur": a["dur"]})
                            over = max(over, a["t"] - t0 + a["dur"])
                lt += over + 0.35
            else:
                u = units.get(sg) or list(units.values())[len(sh["voice"]) % len(units)]
                d = u["end"] - u["start"]
                sh["voice"].append({"start": t + lt, "wav": u["wav"], "dur": d})
                lt += d + 0.35
        dur = lt + (1.8 if name == "end" else 0.5)
        sh.update(end=t + dur, dur=dur)
        shots.append(sh)
        t += dur
    mix, stems = A.mix(shots, t, SR, {})
    sf.write(OUT / "v5_data" / "clips" / "preview_mix.wav", mix, SR)
    rep = A.qa_audio(mix, stems, shots, SR, t)
    (OUT / "v5_data" / "clips" / "preview_qa.json").write_text(json.dumps({"timeline": shots, "qa": rep}, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("ok", "fail", "mix_integrated_lufs", "mix_true_peak_dbtp", "clip_lufs", "music_level_db_p5_p95",
                                          "agent_vs_clip_db", "last_frame_peak")}, default=float))


if __name__ == "__main__":
    main()
