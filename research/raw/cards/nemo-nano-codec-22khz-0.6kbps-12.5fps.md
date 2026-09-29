---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf
pipeline_tag: feature-extraction
---


# NVIDIA NeMo NanoCodec
<style>
img{
display: inline-table;
vertical-align: small;
margin: 0;
padding: 0;
}
</style>
[![Model architecture](https://img.shields.io/badge/Model_Arch-NemoNanoCodec-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-62M-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-multilingual-lightgrey#model-badge)](#datasets)


The [NeMo NanoCodec](https://arxiv.org/abs/2508.05835v1) is a neural audio codec that leverages finite scalar quantization and adversarial training with large speech language models to  achieve state-of-the-art audio compression across different bitrate and frame rate ranges.  
Model variant details:

| Sample Rate | Frame Rate | Bit Rate   | # Codebooks | Codebook Size | Embed Dim   | FSQ Levels   |
|:-----------:|:----------:|:----------:|:-----------:|:-------------:|:-----------:|:------------:|
| 22050       | 12.5      | 0.6kpbs   | 4          | 4032          | 16          | [9, 8, 8, 7] |

This model is ready for commercial/non-commercial use.



## NeMo NanoCodec variants

Model | Sample Rate | Frame Rate | Bit Rate   | # Codebooks | Codebook Size | Embed Dim   | FSQ Levels   |
:-----------:|:-----------:|:----------:|:----------:|:-----------:|:-------------:|:-----------:|:------------:|
[1.78kbps-12.5fps](https://huggingface.co/nvidia/nanocodec-22khz-1.78kbps-12.5fps)| 22050       | 12.5      | 1.78kpbs   | 13          | 2016          | 52          | [8, 7, 6, 6] |
[0.6kbps-12.5fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps) | 22050       | 12.5      | 0.6kpbs   | 4          | 4032          | 16          | [9, 8, 8, 7] |
[1.89kbps-21.5fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps) | 22050       | 21.5       | 1.89kpbs   | 8          | 2016          | 32          | [8, 7, 6, 6] |

⚠️ **Note on 0.6kbps-12.5fps**  
This variant is designed for **fine-tuning with a limited set of speakers**, as shown in our [S2S Duplex paper](https://www.isca-archive.org/interspeech_2025/hu25f_interspeech.html).  
It is **not recommended** for general-purpose audio encoding or decoding.

ℹ️ **Recommended Variants**  
Both **1.78kbps-12.5fps** and **1.89kbps-21.5fps** achieve similar audio reconstruction quality.  
However, our [Magpie TTS](https://build.nvidia.com/nvidia/magpie-tts-multilingual) model performs best with **1.89kbps-21.5fps**.

## License/Terms of Use
[NVIDIA Open Model License Agreement](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf)

### Deployment Geography:
<br>Global<br>

### Use Case: 
<br> This model can be used for audio compression and can also serve as a component in the training of speech generation models.<br>

### Release Date:  
<br>Huggingface [08/11/2025] via https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps<br> 

## Model Architecture
NeMo NanoCodec is composed of a fully convolutional generator neural network and three discriminators. The generator comprises an encoder, followed by vector quantization, and a [HiFi-GAN-based](https://arxiv.org/abs/2010.05646) decoder. 

The non-causal encoder consists of five residual blocks, each block containing three residual layers similar to the [multi-receptive field fusion (MRF) module](https://arxiv.org/abs/2010.05646). The causal decoder, based on the HiFi-GAN vocoder, uses upsampling rates that are the reverse of the encoder's One-Dimensional (1D) convolutional strides. 

For the vector quantization, we have used [Finite Scalar Quantization (FSQ)](https://arxiv.org/abs/2309.15505) with thirteen codebooks and four dimensions per code and 2016 codes per codebook.  For the discriminators, we utilize three neural networks, all employing a squared-GAN and feature-matching loss. We adopt the [multi-period discriminator](https://arxiv.org/abs/2010.05646), [multi-band multi-scale STFT discriminator](https://arxiv.org/abs/2306.06546), and  [WavLM-based discriminator](https://arxiv.org/abs/2409.12117). 

For more details please check [our paper](https://arxiv.org/abs/2508.05835v1).

**This model was developed based on [NVIDIA Low Frame-rate Speech Codec](https://huggingface.co/nvidia/low-frame-rate-speech-codec-22khz)**

** This model has 62M of model parameters.**


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

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

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

## Model Version(s):
<br> v12.5.1.78 <br>

## How to Use this Model

The model is available for use in the [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset. 

### Inference

For inference, you can refer to our [Audio Codec Inference Tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Audio_Codec_Inference.ipynb), which automatically downloads the model checkpoint. Ensure that you set the model_name parameter to "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps".

Alternatively, you can use the code below, which also handles the automatic checkpoint download:

```
import librosa
import torch
import soundfile as sf
from nemo.collections.tts.models import AudioCodecModel

path_to_input_audio = ??? # path of the input audio
path_to_output_audio = ??? # path of the reconstructed output audio

# load audio codec model
nemo_codec_model = AudioCodecModel.from_pretrained("nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps").eval()

# get discrete tokens from audio
audio, _ = librosa.load(path_to_input_audio, sr=nemo_codec_model.sample_rate)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
audio_tensor = torch.from_numpy(audio).unsqueeze(dim=0).to(device)
audio_len = torch.tensor([audio_tensor[0].shape[0]]).to(device)

encoded_tokens, encoded_len = nemo_codec_model.encode(audio=audio_tensor, audio_len=audio_len)

# Reconstruct audio from tokens
reconstructed_audio, _ = nemo_codec_model.decode(tokens=encoded_tokens, tokens_len=encoded_len)

# save reconstructed audio
output_audio = reconstructed_audio.cpu().numpy().squeeze()
sf.write(path_to_output_audio, output_audio, nemo_codec_model.sample_rate)

```

If preferred, you can manually download the [checkpoint](https://huggingface.co/nvidia/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps/resolve/main/nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo) and use the provided code to run inference on the model:

```
import librosa
import torch
import soundfile as sf
from nemo.collections.tts.models import AudioCodecModel

codec_path = ??? # set here the model .nemo checkpoint path
path_to_input_audio = ??? # path of the input audio
path_to_output_audio = ??? # path of the reconstructed output audio

# load audio codec model
nemo_codec_model = AudioCodecModel.restore_from(restore_path=codec_path, map_location="cpu").eval()

# get discrete tokens from audio
audio, _ = librosa.load(path_to_input_audio, sr=nemo_codec_model.sample_rate)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
audio_tensor = torch.from_numpy(audio).unsqueeze(dim=0).to(device)
audio_len = torch.tensor([audio_tensor[0].shape[0]]).to(device)

encoded_tokens, encoded_len = nemo_codec_model.encode(audio=audio_tensor, audio_len=audio_len)

# Reconstruct audio from tokens
reconstructed_audio, _ = nemo_codec_model.decode(tokens=encoded_tokens, tokens_len=encoded_len)

# save reconstructed audio
output_audio = reconstructed_audio.cpu().numpy().squeeze()
sf.write(path_to_output_audio, output_audio, nemo_codec_model.sample_rate)

```

### Training
For fine-tuning on another dataset, please follow the steps available at our [Audio Codec Training Tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Audio_Codec_Training.ipynb). Note that you will need to set the ```CONFIG_FILENAME``` parameter to the "audio_codec_low_frame_rate_22050.yaml" config. You also will need to set ```pretrained_model_name``` to "audio_codec_low_frame_rate_22khz".


## Training, Testing, and Evaluation Datasets:

The NeMo NanoCodec was trained on 28.7k hours of speech data spanning 105 languages. The model was evaluated using multilingual audiobook-style data and high-quality English recordings. For further details, refer to  [our paper](https://arxiv.org/abs/2508.05835v1). 


### Training Datasets
The NeMo NanoCodec is trained on a total of 28.7k hrs of speech data from 105 languages.

Link: [MLS English](https://www.openslr.org/94/) [25.5k]
  
      - Data Collection Method by Dataset: Human

      - Labeling Method by Dataset: Automated
  
Link: [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)[3.2k]
  
      - Data Collection Method by Dataset: Human
      
      - Labeling Method by Dataset: Human
  
### Test Datasets
  
Link: [MLS](https://www.openslr.org/94/)

    - Data Collection Method by Dataset: Human
    
    - Labeling Method by Dataset: Automated
   
    - Properties: We randomly selected 200 samples from each of the eight languages in the 44kHz MLS dataset. 

Link: [DAPS](https://zenodo.org/records/4660670)
  
      - Data Collection Method by Dataset: Human
      
      - Labeling Method by Dataset: Automated
   
      - Properties: To assess our models' performance on studio-quality audio, we utilized the F10 and M10 speakers from the DAPS Clear dataset. These speakers were also employed in the evaluation of the [DAC model](https://arxiv.org/abs/2306.06546).


### Evaluation Datasets

Link: [MLS English](https://www.openslr.org/94/)
  
      - Data Collection Method By Dataset: Human
      
      - Labeling Method by Dataset: Automated
  
      - Properties: We randomly selected 3,807 samples, including examples from multiple speakers.
  
Link: [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
  
      - Data Collection Method by Dataset: Human
      
      - Labeling Method by Dataset: Human
  
      - Properties: We randomly selected 1587 samples, including examples from multiple languages.



## Performance

We evaluated our codec using multiple objective audio quality metrics across two distinct test sets. Additionally, we compared our model's performance with state-of-the-art codecs. For further details, please refer to [our paper](https://arxiv.org/abs/2508.05835v1).


Variant results:
| Dataset     | Squim MOS (↑)     |PESQ (↑)      |Mel Dist. (↓)      | SECS (↓) | CER (↓)|
|:-----------:|:----------:|:----------:|:----------:|:-----------:|:-----------:|
| MLS |      4.407  |   2.012    |     0.205   |   0.701     |  7.792 | 
| DAPS |      4.662    |    2.205   |     0.204    |   0.656    | 1.469| 




## Inference:
**Engine:** Transformers <br>
**Test Hardware:** <br>
- FP32:   
  - 1x NVIDIA A100-80GB
  - 2x NVIDIA RTX 6000 Ada

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 

For more detailed information on ethical considerations for this model, please see the [Model Card++ Explainability](https://gitlab-master.nvidia.com/ajukic/nemo-model-overview/-/blob/main/models/nanocodec_22khz/explainalability-subcard.md), [Bias](https://gitlab-master.nvidia.com/ajukic/nemo-model-overview/-/blob/main/models/nanocodec_22khz/bias-subcard.md), [Safety & Security](https://gitlab-master.nvidia.com/ajukic/nemo-model-overview/-/blob/main/models/nanocodec_22khz/safety-subcard.md), and [Privacy Subcards](https://gitlab-master.nvidia.com/ajukic/nemo-model-overview/-/blob/main/models/nanocodec_22khz/privacy-subcard.md). 


Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

