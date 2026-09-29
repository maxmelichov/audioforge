"""audioforge in-process: stream the bundled clip through the served model, no server (README "Python API").

Single-model mode (the default): the session follows one known user, given by their stored voice print."""
import json

import soundfile as sf

import audioforge

fe = audioforge.load()                                   # single-model mode: the models audioforge-download installed
user = json.load(open("examples/audio/two_party_call_16s.voiceprint.json"))  # or fe.voiceprint(>= 5 s of clean speech)
pcm, sr = sf.read("examples/audio/two_party_call_16s.wav", dtype="int16")
s = fe.session(turn_policy="hybrid_dyn", sample_rate=sr)
s.enroll(user)
blocks = [pcm[i:i + sr // 50] for i in range(0, len(pcm), sr // 50)]   # 20 ms, like a microphone
for events in [s.feed(b) for b in blocks] + [s.end()]:
    for ev in events:
        if ev["type"] in ("turn_end", "final"):
            print(f'{ev["t"]:5.2f}s {ev["type"]:8s} speaker={ev.get("speaker")} {ev.get("text", ev.get("policy"))}')
        elif ev["type"] == "stats":
            print(f'RTF {ev["rtf"]}, peak RSS {ev["peak_rss_mb"]} MB')
