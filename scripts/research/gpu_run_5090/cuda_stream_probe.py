"""research/GPU_RUN_2026-09-29.md: the streaming Engine / Session on the bundled two-party clip (single mode, hybrid_dyn,
no stored print), 3 runs after warm-up, on cpu (fast-conv) or cuda (TF32 off as Engine.load sets it; --tf32 keeps
PyTorch's default). Prints audioforge.bench.run_once's report of the best run.

    PYTHONPATH=. python scripts/research/gpu_run_5090/cuda_stream_probe.py cuda|cpu [--tf32]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from audioforge import hub
from audioforge.bench import run_once
from audioforge.data import load_wav
from audioforge.launch import TSVAD_FILE, find_head
from audioforge.serve import Engine
from audioforge.server.cli import MODES
from audioforge.train import load_model

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
fast = dev == "cpu"
if dev.startswith("cuda") and "--tf32" not in sys.argv:  # as Engine.load(device="cuda")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(2)
asr_p = str(hub.find_model("asr"))
kw = {**MODES["single"], "tsvad": str(find_head(TSVAD_FILE, None)), "silero": str(hub.find_model("silero")), "lid": None}
asr = load_model(asr_p, dev)
eng = Engine(asr, None, name="probe", threads=2, fast=fast, **kw)
audio = load_wav(str(hub.ROOT / "examples/audio/two_party_call_16s.wav"), 16000).astype(np.float32)
eng.warmup()
runs = [run_once(eng, audio, "hybrid_dyn") for _ in range(3)]
best = min(runs, key=lambda r: r["rtf"])
print(json.dumps({"dev": dev, "rtf_all": [r["rtf"] for r in runs], **best}, indent=1))
if dev.startswith("cuda"):
    print("peak cuda MB", torch.cuda.max_memory_allocated() / 2**20)
