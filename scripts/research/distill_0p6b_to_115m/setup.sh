#!/usr/bin/env bash
# One-time setup of a rented Linux GPU box (1 x A100 / H100, CUDA 12 driver, >= 200 GB disk) for
# scripts/research/distill_0p6b_to_115m (research/archive/IMPROVEMENTS.md section 7). Run from the repo root:
#     bash scripts/research/distill_0p6b_to_115m/setup.sh [--skip-data]
# Every step is resumable (re-run after an interruption). Nothing here needs a GPU.
set -euo pipefail
SKIP_DATA=0
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-data) SKIP_DATA=1; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
ROOT=$(pwd)
test -f pyproject.toml || { echo "run from the repo root"; exit 2; }

# 1. environment: Python >= 3.10 venv with the repo's research extras + CUDA torch
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip wheel
# RTX 5090 (Blackwell, sm_120) needs a PyTorch build with CUDA >= 12.8 (cu128 wheels or newer)
.venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -q -e ".[research,nemo-import]" pyarrow soundfile huggingface_hub
command -v ffmpeg >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg; }
.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA device visible to PyTorch"
cv = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
assert cv >= (12, 8), f"PyTorch built for CUDA {torch.version.cuda}; Blackwell (sm_120) needs >= 12.8"
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(f"cuda:{i} {torch.cuda.get_device_name(i)} capability {cap}")
    assert cap >= (12, 0) or cap < (10, 0), f"unexpected capability {cap}"
    torch.ones(8, device=f"cuda:{i}").sum().item()  # a kernel actually runs (catches wheel / arch mismatches)
EOF

# 2. models: the served 115M student must be copied from the laptop (it carries our trained heads):
#      scp laptop:nvidia-audio-models/runs/stage1_served.afm runs/
#    the 0.6B teacher is imported from NVIDIA's .nemo (Hugging Face nvidia/nemotron-speech-streaming-en-0.6b,
#    NVIDIA Open Model License; accept it on the model page, then `huggingface-cli login` with a read token)
test -f runs/stage1_served.afm || { echo "copy runs/stage1_served.afm from the laptop first (see README)"; exit 3; }
if [ ! -f runs/nemo_nemotron_speech_streaming_en_0.6b.afm ]; then
  mkdir -p data/nemo
  .venv/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("nvidia/nemotron-speech-streaming-en-0.6b", "nemotron-speech-streaming-en-0.6b.nemo",
                    local_dir="data/nemo")
print(p)
EOF
  AUDIOFORGE_0P6B_AFM=runs/nemo_nemotron_speech_streaming_en_0.6b.afm .venv/bin/python scripts/research/core_0p6b.py import
fi
# LibriSpeech test-clean-first200 for the WER gate / eval (the package's manifest uses repo-relative paths)
mkdir -p data/librispeech
cp -n scripts/research/distill_0p6b_to_115m/test-clean-first200.jsonl data/librispeech/ || true

[ "$SKIP_DATA" = 1 ] && { echo "setup done (data skipped)"; exit 0; }

# 3. datasets under the repo's data root (resumable downloads, md5-checked by the prepare scripts)
.venv/bin/python scripts/research/prepare_ami.py --n-train 136 --n-dev 4 --n-eval 0          # ~80 h train meetings
.venv/bin/python scripts/research/prepare_icsi.py --n-train 70                             # all 70 train meetings
.venv/bin/python scripts/research/prepare_librispeech.py --splits train-clean-100 test-clean
df -h .
echo "setup done"
