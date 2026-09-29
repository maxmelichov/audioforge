#!/usr/bin/env bash
# One-time setup of the two-RTX-5090 Linux / WSL2 box for scripts/research/single_model_distill (research/SINGLE_MODEL.md
# part B). Run from the repo root:  bash scripts/research/single_model_distill/setup.sh [--skip-data] [--no-teacher]
# Every step is resumable (re-run after an interruption). Nothing here needs a GPU except the final sm_120 check.
set -euo pipefail
SKIP_DATA=0; TEACHER=1
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-data) SKIP_DATA=1; shift ;;
    --no-teacher) TEACHER=0; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
test -f pyproject.toml || { echo "run from the repo root"; exit 2; }

# 1. environment: Python >= 3.10 venv, the research extras, CUDA 12.8 PyTorch (RTX 5090 = Blackwell sm_120)
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip wheel
.venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -q -e ".[research,nemo-import]" pyarrow soundfile huggingface_hub onnxruntime
command -v ffmpeg >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg; }
.venv/bin/python - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "no CUDA device visible to PyTorch"
cv = tuple(int(x) for x in (torch.version.cuda or "0.0").split(".")[:2])
assert cv >= (12, 8), f"PyTorch built for CUDA {torch.version.cuda}; Blackwell (sm_120) needs >= 12.8"
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    print(f"cuda:{i} {torch.cuda.get_device_name(i)} capability {cap} bf16 {torch.cuda.is_bf16_supported()}")
    torch.ones(8, device=f"cuda:{i}").sum().item()  # a kernel actually runs on this GPU
    x = torch.randn(64, 64, device=f"cuda:{i}", dtype=torch.bfloat16); (x @ x).float().sum().item()
PYEOF

# 2. models from the laptop (they carry this project's trained heads):
#      scp laptop:nvidia-audio-models/runs/{stage1_served.afm,tsvad_spk.pt} runs/
for f in runs/stage1_served.afm runs/tsvad_spk.pt; do
  test -f "$f" || { echo "copy $f from the laptop first (see README)"; exit 3; }
done
# optional: the 0.6B teacher for --asr-kd (Hugging Face nvidia/nemotron-speech-streaming-en-0.6b, NVIDIA Open Model
# License: accept it on the model page, then `huggingface-cli login` with a read token)
if [ "$TEACHER" = 1 ] && [ ! -f runs/nemo_nemotron_speech_streaming_en_0.6b.afm ]; then
  mkdir -p data/nemo
  .venv/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('nvidia/nemotron-speech-streaming-en-0.6b', 'nemotron-speech-streaming-en-0.6b.nemo', local_dir='data/nemo'))"
  AUDIOFORGE_0P6B_AFM=runs/nemo_nemotron_speech_streaming_en_0.6b.afm .venv/bin/python scripts/research/core_0p6b.py import
fi
mkdir -p data/librispeech
cp -n scripts/research/distill_0p6b_to_115m/test-clean-first200.jsonl data/librispeech/ || true

[ "$SKIP_DATA" = 1 ] && { echo "setup done (data skipped)"; exit 0; }

# 3. datasets under the repo's data root (resumable, md5-checked by the prepare scripts)
.venv/bin/python scripts/research/prepare_ami.py --n-train 136 --n-dev 4 --n-eval 0     # train + the 4 dev meetings (eot-bench, AMI-200)
.venv/bin/python scripts/research/prepare_icsi.py --n-train 70 --n-dev 2 --n-eval 3 --max-minutes 120  # rerun resumes
.venv/bin/python scripts/research/prepare_librispeech.py --splits train-clean-100 test-clean
df -h .
echo "setup done"
