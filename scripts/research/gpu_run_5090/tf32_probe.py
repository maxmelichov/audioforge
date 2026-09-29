"""research/GPU_RUN_2026-09-29.md: encoder output difference cuda vs cpu on the bundled clip (whole-utterance forward,
[70,1]), with cuDNN TF32 on (PyTorch's default) and off (what serve --device cuda sets).

    PYTHONPATH=. python scripts/research/gpu_run_5090/tf32_probe.py
"""
import soundfile as sf
import torch

from audioforge import hub
from audioforge.train import load_model

x, sr = sf.read("examples/audio/two_party_call_16s.wav", dtype="float32")
p = str(hub.find_model("asr"))
cpu = load_model(p, "cpu").eval()
gpu = load_model(p, "cuda").eval()
def enc(m, dev):
    a = torch.as_tensor(x, device=dev)[None]
    with torch.no_grad():
        f, fl = m.preprocessor(a, torch.tensor([a.shape[1]], device=dev))
        e, el, *_ = m.encoder(f, fl) if not hasattr(m, "encode") else m.encode(a, torch.tensor([a.shape[1]], device=dev), [70, 1])
    return e.float().cpu()
ref = enc(cpu, "cpu")
print("cudnn.allow_tf32 default", torch.backends.cudnn.allow_tf32, "matmul tf32", torch.backends.cuda.matmul.allow_tf32)
for tf in (True, False):
    torch.backends.cudnn.allow_tf32 = tf
    g = enc(gpu, "cuda")
    d = (g - ref).abs()
    print(f"cudnn tf32={tf}: max abs diff {d.max():.3e}, mean {d.mean():.3e}, rel {(d.norm() / ref.norm()):.3e}")
