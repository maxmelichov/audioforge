"""The opening example (today_call): a real call through Pipecat's default stack with >= 2 cut-ins, neutral content,
not used elsewhere in the film. Same chain as demo/archive/v5/audio/clips.py (clarity EQ, -16 LUFS, -1 dBTP, 150 ms fades),
Whisper-small WER against the Whisper large-v3-turbo pseudo-reference (otoSpeech has no transcript), agent lines placed
with clips.place_agent (never cut mid-word). Writes $DEMO_OUT/v5_data/clips/ex_call_today.wav + opening.json (a record
in the clips_manifest.json format, merged by export.py).

    HF_HUB_OFFLINE=1 PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python demo/archive/v5/opening.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent / "audio"))
import clips as C  # noqa: E402

OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out")) / "v5_data"
NAME, T0, T1 = "oto_5f9148c5dfff61031a7c9ef711020cad", 39.9, 46.2   # "…season was great, although it was hectic, but it was good. All right, so let's dive into…"


def main():
    pc, clips = C.load(); cache = C.load_cache()
    src = C.src_of(NAME)
    dst = OUT / "clips" / "ex_call_today.wav"
    info = C.process(src, T0, T1, dst)
    ref, rsrc = C.words_ref(NAME, clips, cache)
    hyp = C.transcribe("small", dst, offset=T0)
    w, n = C.wer(ref, hyp, T0, T1)
    a = C.rec(pc, clips, NAME, "pipecat", "A")
    cuts = [c for c in a["cut_in_t"] if T0 <= c < T1]
    resp = [e["response"] for e in a["ends"] if e.get("response") is not None and T0 <= e["response"] < T1]
    events = sorted([(c, "cut_in") for c in cuts] + [(r, "response") for r in resp])
    placed = C.place_agent(events, T1, C.agent_lines())
    meta = clips[NAME]
    onset = meta.get("first_onset_user")
    ft = onset + a["first_text_ms_after_onset"] / 1000 if a.get("first_text_ms_after_onset") is not None and onset is not None else None
    rec = {"clip_id": "ex_call_today", "kind": "call_today", "source_clip": NAME, "source": "otoSpeech full-duplex (CC-BY-4.0), the caller's own channel",
           "source_wav": str(src), "wav": str(dst), "offset": T0, "span": [T0, T1], "dur_s": round(T1 - T0, 3),
           "wer_after": round(w, 4), "wer_ref_words": n, "ref_source": rsrc, "whisper_small_after": " ".join(x["w"] for x in hyp),
           "ref_text": " ".join(x["w"] for x in ref if T0 <= (x["s"] + x["e"]) / 2 < T1), "lufs": info["lufs"], "true_peak_dbtp": info["true_peak_dbtp"],
           "loudness_method": info["method"], "agent_voice": {"P": placed},
           "systems": {"P": {"system": "pipecat/A", "label": "Pipecat default", "record": f"runs/e2e_final.json per_clip['oto|user|pipecat|A|{NAME}']",
                             "cut_ins": cuts, "responses": resp, "first_text": ft, "onset": onset}}}
    # subtitles: Whisper small on the processed clip (the turbo pseudo-reference drops a word here: "although it was
    # ___ but it was good"); both are automatic transcripts, the fine print says so
    subs = [[x["s"], x["e"], x["w"], "user"] for x in hyp if T0 <= (x["s"] + x["e"]) / 2 < T1]
    (OUT / "opening.json").write_text(json.dumps({"clip": rec, "subs": subs}, indent=1))
    print(json.dumps({k: rec[k] for k in ("span", "wer_after", "wer_ref_words", "ref_text", "whisper_small_after", "lufs", "true_peak_dbtp")}, indent=1))
    print("cut-ins", cuts, "responses", resp, "agent", [(p["t"], p["text"]) for p in placed])


if __name__ == "__main__":
    main()
