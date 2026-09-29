"""research/IMPROVEMENTS.md section 1: does the served --turn-input tsvad path reproduce the offline benchmark?

For N eot-bench v2 AMI dev windows: stream the window's audio through serve.Session (turn_input tsvad, explicit
5 s voice print = the benchmark's stored print, 20 ms blocks, policy head) and compare per frame (i) the live TS-VAD
[P(target), P(other)] with the offline bind_tsvad_spk track and (ii) the live turn-head eot with the offline
scores_tsvad_spk. -> <scratch>/improve/serve_check.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import tsvad as T  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
torch.set_num_threads(2)
from audioforge.serve import Engine, Session, SessionConfig  # noqa: E402
from audioforge.train import load_model  # noqa: E402

asr = load_model(str(ROOT / "runs" / "stage1_served.afm"), "cpu")
diar = load_model(str(ROOT / "runs" / "nemo_sortformer_v2.afm"), "cpu")
eng = Engine(asr, diar, threads=2, turn_input="tsvad", enroll="explicit")
ext, meta, ds = T.bench_windows("ami")
kidx, VP = T.load_vprints("ami")
rng = np.random.default_rng(0)
res = []
for i in rng.choice(len(ext), N, replace=False):
    v = ext[int(i)]
    k = kidx[T.wkey(v)]
    if not bool(VP["has_5p0"][k, 0]):
        continue
    t0 = time.time()
    s = Session(eng, SessionConfig(turn_policy="head"))
    s.arm_enrollment("enroll", 0, VP["spk_5p0"][k, 0].tolist())
    x = np.asarray(v["audio"], np.float32)
    cap = {}
    orig = s.asr.run_turn_on_diar

    def wrapped(*a, **kw):
        r = orig(*a, **kw)
        cap.update({vv: pp for vv, pp, _ in r})
        return r
    s.asr.run_turn_on_diar = wrapped
    msgs = []
    for j in range(0, len(x), 320):
        msgs += s.process(x[j:j + 320])
    msgs += s.finish()
    live = np.stack([s.asr.tsvad_p[f] for f in range(len(s.asr.tsvad_p))])
    off = np.load(T.bind_path("ami", "tsvad_spk", v))
    sc = np.load(T.score_path("ami", "tsvad_spk", v))
    eot = np.array([cap.get(f, np.nan) for f in range(max(cap) + 1)])
    n = min(len(live), len(off))
    m_ = min(len(eot), len(sc))
    ok = ~np.isnan(eot[:m_])
    r = {"key": T.wkey(v), "frames": int(n), "tsvad_maxabs": float(np.abs(live[:n] - off[:n]).max()),
         "tsvad_meanabs": float(np.abs(live[:n] - off[:n]).mean()),
         "eot_maxabs": float(np.abs(eot[:m_][ok] - sc[:m_][ok]).max()) if ok.any() else None,
         "eot_meanabs": float(np.abs(eot[:m_][ok] - sc[:m_][ok]).mean()) if ok.any() else None,
         "eot_crossings_0997_live_offline": [int(((eot[:m_][ok] >= 0.997)).sum()), int((sc[:m_][ok] >= 0.997).sum())],
         "sec": round(time.time() - t0, 1), "audio_s": round(len(x) / 16000, 1)}
    print(r, flush=True)
    res.append(r)
out = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/improve/serve_check.json")
out.write_text(json.dumps(res, indent=1))
