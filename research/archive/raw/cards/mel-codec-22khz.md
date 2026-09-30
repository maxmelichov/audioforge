---
license: other
license_name: nsclv1
license_link: https://developer.nvidia.com/downloads/license/nsclv1
pipeline_tag: feature-extraction
---


# NVIDIA NeMo Mel Codec 22khz
<style>
img {
display: inline-table;
vertical-align: small;
margin: 0;
padding: 0;
}
</style>
[![Model architecture](https://img.shields.io/badge/Model_Arch-HiFi--GAN-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-64.4M-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-multilingual-lightgrey#model-badge)](#datasets)

The NeMo Mel Codec is a neural audio codec which compresses mel-spectrograms into a quantized representation and reconstructs audio. The model can be used as a vocoder for speech synthesis.

The model works with full-bandwidth 22.05kHz speech. It might have lower performance with low-bandwidth speech (e.g. 16kHz speech upsampled to 22.05kHz) or with non-speech audio.

| Sample Rate | Frame Rate | Bit Rate   | # Codebooks | Codebook Size | Embed Dim   | FSQ Levels   |
|:-----------:|:----------:|:----------:|:-----------:|:-------------:|:-----------:|:------------:|
| 22050       | 86.1       | 6.9kpbs    | 8           | 1000          | 32          | [8, 5, 5, 5] |

## Model Architecture
The NeMo Mel Codec model uses a residual network encoder and [HiFi-GAN](https://arxiv.org/abs/2010.05646) decoder. We use [Finite Scalar Quantization (FSQ)](https://arxiv.org/abs/2309.15505), with 8 codebooks and 1000 entries per codebook.

For more details please refer to [our paper](https://arxiv.org/abs/2406.05298).

### Input
  - **Input Type:** Audio 
  - **Input Format(s):** .wav files
  - **Input Parameters:** One-Dimensional (1D)
  - **Other Properties Related to Input:** 22050 Hz Mono-channel Audio

### Output
  - **Output Type**: Audio 
  - **Output Format:** .wav files
  - **Output Parameters:** One Dimensional (1D)
  - **Other Properties Related to Output:** 22050 Hz Mono-channel Audio


## How to Use this Model

The model is available for use in the [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset. 

### Inference
For inference, you can follow our [Audio Codec Inference Tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Audio_Codec_Inference.ipynb) which automatically downloads the model checkpoint. Note that you will need to set the ```model_name``` parameter to "nvidia/mel-codec-22khz".

Alternatively, you can use the code below, which also handles the automatic checkpoint download:

```python
import librosa
import torch
import soundfile as sf
from nemo.collections.tts.models import AudioCodecModel

model_name = "nvidia/mel-codec-22khz"
path_to_input_audio = ??? # path of the input audio
path_to_output_audio = ??? # path of the reconstructed output audio

nemo_codec_model = AudioCodecModel.from_pretrained(model_name).eval()

# get discrete tokens from audio
audio, _ = librosa.load(path_to_input_audio, sr=nemo_codec_model.sample_rate)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
audio_tensor = torch.from_numpy(audio).unsqueeze(dim=0).to(device)
audio_len = torch.tensor([audio_tensor[0].shape[0]]).to(device)

with torch.no_grad():
  encoded_tokens, encoded_len = nemo_codec_model.encode(audio=audio_tensor, audio_len=audio_len)
  
  # Reconstruct audio from tokens
  reconstructed_audio, _ = nemo_codec_model.decode(tokens=encoded_tokens, tokens_len=encoded_len)

# save reconstructed audio
output_audio = reconstructed_audio.cpu().numpy().squeeze()
sf.write(path_to_output_audio, output_audio, nemo_codec_model.sample_rate)

```

### Training

For fine-tuning on another dataset please follow the steps available at our [Audio Codec Training Tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Audio_Codec_Training.ipynb). Note that you will need to set the ```CONFIG_FILENAME``` parameter to the "mel_codec_22050.yaml" config. You also will need to set ```pretrained_model_name``` to "nvidia/mel-codec-22khz".

## Training, Testing, and Evaluation Datasets:


### Training Datasets

The NeMo Audio Codec is trained on a total of 28.7k hrs of speech data from 105 languages.

  - [MLS English](https://www.openslr.org/94/) - 25.5k hours, 4.3k speakers, English
  - [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) - 3.2k hours, 100k speakers, 105 languages.

### Test Datasets

  - [MLS English](https://www.openslr.org/94/) - 15 hours, 42 speakers, English
  - [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) - 2 hours, 1356 speakers, 59 languages

## Performance

We evaluate our codec using several objective audio quality metrics. We evaluate [ViSQOL](https://github.com/google/visqol) and [PESQ](https://lightning.ai/docs/torchmetrics/stable/audio/perceptual_evaluation_speech_quality.html) for perception quality, [ESTOI](https://ieeexplore.ieee.org/document/7539284) for intelligbility, and mel spectrogram and STFT distances for spectral reconstruction accuracy. Metrics are reported on the test set for both the MLS English and CommonVoice data. The model has not been trained or evaluated on non-speech audio.

| Dataset     | ViSQOL     |PESQ        |ESTOI       |Mel Distance |STFT Distance|
|:-----------:|:----------:|:----------:|:----------:|:-----------:|:-----------:|
| MLS English | 4.48       | 3.43       | 0.92       | 0.069       | 0.034       |
| CommonVoice | 4.51       | 3.21       | 0.91       | 0.100       | 0.057       |

## Software Integration

### Supported Hardware Microarchitecture Compatibility:
- NVIDIA Ampere
- NVIDIA Blackwell
- NVIDIA Jetson
- NVIDIA Hopper
- NVIDIA Lovelace
- NVIDIA Pascal
- NVIDIA Turing
- NVIDIA Volta

### Runtime Engine

- Nemo 2.0.0

### Preferred Operating System

- Linux

## License/Terms of Use
This model is for research and development only (non-commercial use) and the license to use this model is covered by the [NSCLv1](https://developer.nvidia.com/downloads/license/nsclv1). 

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.
Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).