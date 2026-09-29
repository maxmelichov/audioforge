#!/usr/bin/env bash
# Preflight for the two-RTX-5090 box (research/SINGLE_MODEL.md part B): GPUs, VRAM, driver / torch / CUDA versions,
# the sm_120 check, free disk, RAM, the files that must come from the laptop, and whether the download hosts answer.
# Read-only; run from the repo root, before or after setup.sh:
#     bash scripts/research/single_model_distill/preflight.sh
set -u
echo "== host"; uname -a; grep -qi microsoft /proc/version 2>/dev/null && echo "WSL2 detected (the CUDA driver must be the Windows host's)"
echo "== GPUs (nvidia-smi)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap --format=csv
else
  echo "nvidia-smi not found: no NVIDIA driver visible"
fi
echo "== PyTorch (need CUDA >= 12.8 wheels for sm_120, bf16, a kernel on every GPU)"
PY=${PY:-.venv/bin/python}; [ -x "$PY" ] || PY=python3
"$PY" - <<'PYEOF' 2>&1
try:
    import torch
    print("torch", torch.__version__, "built for CUDA", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
    print("cuda available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
    ok = torch.cuda.is_available()
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cap = torch.cuda.get_device_capability(i)
        torch.ones(8, device=f"cuda:{i}").sum().item()  # an actual kernel (catches wheel / arch mismatches)
        print(f"  cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GiB capability {cap} bf16 {torch.cuda.is_bf16_supported()}")
        if cap == (12, 0) and tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2]) < (12, 8):
            print("  !! sm_120 needs a PyTorch build for CUDA >= 12.8 (setup.sh installs the cu128 wheels)"); ok = False
    print("PREFLIGHT TORCH", "OK" if ok else "NOT OK")
except Exception as e:
    print("torch not usable yet:", e)
PYEOF
echo "== disk (need >= 120 GB free: datasets ~30 GB, 0.6B teacher cache ~16 GB, checkpoints ~10 GB)"; df -h . | tail -1
echo "== RAM (>= 32 GB recommended: the manifest stage holds turn-window labels of 206 meetings)"; free -g 2>/dev/null | head -2 || true
echo "== files from the laptop (not downloadable: they carry this project's trained heads)"
for f in runs/stage1_served.afm runs/tsvad_spk.pt; do [ -f "$f" ] && echo "  ok      $f" || echo "  MISSING $f (scp from the laptop, see README)"; done
[ -f runs/nemo_nemotron_speech_streaming_en_0.6b.afm ] && echo "  ok      0.6B teacher" || echo "  (0.6B teacher not imported yet: setup.sh, only needed for --asr-kd)"
echo "== download hosts"
for u in https://github.com https://pypi.org/simple/ https://download.pytorch.org/whl/cu128/ https://huggingface.co \
         https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ https://www.openslr.org/resources/12/; do
  code=$(curl -s -o /dev/null -m 15 -w "%{http_code}" -I "$u" || echo "fail")
  printf "  %-50s %s\n" "$u" "$code"
done
echo "== ffmpeg"; command -v ffmpeg >/dev/null && ffmpeg -version | head -1 || echo "ffmpeg missing (setup.sh installs it)"
