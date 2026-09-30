"""VAD ROC-AUC and F1 (the metrics Silero reports) for Silero v5 and NVIDIA MarbleNet v2 on the AMI dev VAD set
(64 x 20 s windows, 80 ms frames, label = any speaker active; the exact frames, labels and pooling of
scripts/research/bench_sd_baselines.py --stage vad). audioforge's shipped block-4 VAD head has its AUC in
runs/vad_single.json (eval > ami_dev > L3 > auc, same windows). -> runs/vad_auc.json. Through scripts/dev/gate.sh.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
WORK = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/metrics/vad")
SILERO_V5_URL = "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.jit"


def main():
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(2)
    import bench_sd_baselines as B

    from audioforge.baselines import sd
    val = B.diar_windows(64)
    labels = [v["vad"] > 0.5 for v in val]
    y = np.concatenate(labels).astype(int)
    WORK.mkdir(parents=True, exist_ok=True)
    v5 = WORK / "silero_vad_v5.1.2.jit"
    if not v5.exists():
        torch.hub.download_url_to_file(SILERO_V5_URL, str(v5))
    m = torch.jit.load(str(v5)).eval()
    sil = [sd.pool_probs(*sd.silero_probs(m, v["audio"]), len(l)) for v, l in zip(val, labels)]
    mb = sd.load_nemo_conv("nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0")
    mar = [sd.pool_probs(mb.probs(v["audio"]), mb.hop, len(l), offset=-mb.hop / 2) for v, l in zip(val, labels)]
    out = {"data": "AMI dev, 64 x 20 s windows, 80 ms frames, any-speaker label (bench_sd_baselines --stage vad)",
           "n_frames": int(len(y)), "speech_frac": round(float(y.mean()), 4)}
    for name, sc in (("silero_v5.1.2", sil), ("marblenet_frame_vad_v2", mar)):
        s = np.concatenate(sc)
        f1 = sd.score_vad(sc, labels, [0.5])["0.5"]["f1"]
        out[name] = {"auc": round(float(roc_auc_score(y, s)), 4), "f1_at_0.5": f1}
    vs = json.loads((ROOT / "runs" / "vad_single.json").read_text())["eval"]["ami_dev"]["L3"]
    out["audioforge_block4_head"] = {"auc": vs["auc"], "f1_at_0.5": vs["f1"], "n_frames": vs["n_frames"],
                                     "source": "runs/vad_single.json eval > ami_dev > L3"}
    (ROOT / "runs" / "vad_auc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
