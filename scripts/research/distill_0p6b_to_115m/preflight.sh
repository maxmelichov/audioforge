#!/usr/bin/env bash
# Preflight for the GPU box (research/IMPROVEMENTS.md section 7): GPUs, VRAM, driver / torch / CUDA versions, free disk,
# and whether the download hosts answer. Read-only; run from the repo root (before or after setup.sh):
#     bash scripts/research/distill_0p6b_to_115m/preflight.sh
set -u
echo "== host"; uname -a; grep -qi microsoft /proc/version 2>/dev/null && echo "WSL2 detected (CUDA driver must be the Windows host's)"
echo "== GPUs (nvidia-smi)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap --format=csv
else
  echo "nvidia-smi not found: no NVIDIA driver visible"
fi
echo "== PyTorch"
PY=${PY:-.venv/bin/python}; [ -x "$PY" ] || PY=python3
"$PY" - <<'PYEOF' 2>&1
try:
    import torch
    print("torch", torch.__version__, "built for CUDA", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
    print("cuda available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GiB capability {torch.cuda.get_device_capability(i)}"
              f" bf16 {torch.cuda.is_bf16_supported()}")
except Exception as e:
    print("torch not usable yet:", e)
PYEOF
echo "== disk (need >= 60 GB free; 200 GB comfortable)"; df -h . | tail -1
echo "== RAM"; free -g 2>/dev/null | head -2 || true
echo "== download hosts"
for u in https://github.com https://pypi.org/simple/ https://download.pytorch.org/whl/cu128/ https://huggingface.co \
         https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ https://raw.githubusercontent.com https://www.openslr.org/resources/12/; do
  code=$(curl -s -o /dev/null -m 15 -w "%{http_code}" -I "$u" || echo "fail")
  printf "  %-50s %s\n" "$u" "$code"
done
echo "== ffmpeg"; command -v ffmpeg >/dev/null && ffmpeg -version | head -1 || echo "ffmpeg missing (setup.sh installs it)"
