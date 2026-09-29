---
license: other
license_name: nsclv1
license_link: https://developer.nvidia.com/downloads/license/nsclv1
pipeline_tag: feature-extraction
---


# NVIDIA Low Frame-rate Speech Codec
<style>
img {
display: inline-table;
vertical-align: small;
margin: 0;
padding: 0;
}
</style>
[![Model architecture](https://img.shields.io/badge/Model_Arch-Low_Frame--rate_Speech_Codec-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-112.7M-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-multilingual-lightgrey#model-badge)](#datasets)

The [Low Frame-rate Speech Codec](https://arxiv.org/abs/2409.12117) is a neural audio codec that leverages finite scalar quantization and adversarial training with large speech language models to achieve high-quality audio compression with a 1.89 kbps bitrate and 21.5 frames per second. 

| Sample Rate | Frame Rate | Bit Rate   | # Codebooks | Codebook Size | Embed Dim   | FSQ Levels   |
|:-----------:|:----------:|:----------:|:-----------:|:-------------:|:-----------:|:------------:|
| 22050       | 21.5       | 1.89kpbs    | 8           | 2016          | 32          | [8, 7, 6, 6] |

## Model Architecture
Low Frame-rate Speech Codec model is composed of a fully convolutional generator neural network and three discriminators. 
The generator comprises an encoder, followed by vector quantization, and a [HiFi-GAN-based](https://arxiv.org/abs/2010.05646) decoder. 
The encoder consists of five residual blocks, each block containing three residual layers similar to the [multi-receptive field fusion (MRF) module](https://arxiv.org/abs/2010.05646). 
For the vector quantization, we have used [Finite Scalar Quantization (FSQ)](https://arxiv.org/abs/2309.15505) with eight codebooks and four dimensions per code and 2016 codes per codebook. 
For the discriminators, we utilize three neural networks, all employing a squared-GAN and feature-matching loss. We adopt the [multi-period discriminator](https://arxiv.org/abs/2010.05646) and the [multi-scale complex STFT discriminator](https://arxiv.org/abs/2210.13438). 
Additionally, we proposed the use of Speech Language Models (SLMs) as a discriminator. SLMs encode information ranging from acoustic to semantic aspects, which could benefit our model's training, especially in low frame rate settings where accurate pronunciation is difficult to achieve due to the high compression rate. We adopted the [12-layer WavLM](https://arxiv.org/abs/2110.13900) as the SLM. During training, we resample the input audio to 16 kHz before feeding it into the WavLM model, extracting the intermediary layer features. These features are then fed to a discriminative head composed of four 1D convolutional layers. 

For more details please check [our paper](https://arxiv.org/abs/2409.12117).

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

For inference, you can refer to our [Audio Codec Inference Tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Audio_Codec_Inference.ipynb), which automatically downloads the model checkpoint. Ensure that you set the model_name parameter to "nvidia/low-frame-rate-speech-codec-22khz".

Alternatively, you can use the code below, which also handles the automatic checkpoint download:

```
import librosa
import torch
import soundfile as sf
from nemo.collections.tts.models import AudioCodecModel

path_to_input_audio = ??? # path of the input audio
path_to_output_audio = ??? # path of the reconstructed output audio

# load audio codec model
nemo_codec_model = AudioCodecModel.from_pretrained("nvidia/low-frame-rate-speech-codec-22khz").eval()

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

If preferred, you can manually download the [checkpoint](https://huggingface.co/nvidia/low-frame-rate-speech-codec-22khz/resolve/main/low-frame-rate-speech-codec-22khz.nemo) and use the provided code to run inference on the model:

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
For fine-tuning on another dataset please follow the steps available at our [Audio Codec Training Tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/tts/Audio_Codec_Training.ipynb). Note that you will need to set the ```CONFIG_FILENAME``` parameter to the "audio_codec_low_frame_rate_22050.yaml" config. You also will need to set ```pretrained_model_name``` to "audio_codec_low_frame_rate_22khz".


## Training, Testing, and Evaluation Datasets:

The Low Frame-rate Speech Codec was trained on 28.7k hours of speech data spanning 105 languages. The model was evaluated using multilingual audiobook-style data and high-quality English recordings. For further details, refer to  [our paper](https://arxiv.org/abs/2409.12117). 


### Training Datasets
The Low Frame-rate Speech Codec is trained on a total of 28.7k hrs of speech data from 105 languages.

  - [MLS English](https://www.openslr.org/94/) [25.5k]
  
      - Data Collection Method: by Human
      
      - Labeling Method: Automated
  
  -  [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)[3.2k]
  
      - Data Collection Method: by Human
      
      - Labeling Method: by Human
  
  

### Evaluation Datasets

  - [MLS English](https://www.openslr.org/94/)
  
      - Data Collection Method: by Human
      
      - Labeling Method: Automated
  
  -  [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0)
  
      - Data Collection Method: by Human
      
      - Labeling Method: by Human

### Test Datasets
  
  - [MLS](https://www.openslr.org/94/)

    - Data Collection Method: by Human
    
    - Labeling Method: Automated
   
    - Properties: We randomly selected 200 samples from each of the eight languages in the 44kHz MLS dataset. 

  - [DAPS](https://zenodo.org/records/4660670)
  
      - Data Collection Method: by Human
      
      - Labeling Method: Automated
   
      - Properties: To assess our models' performance on studio-quality audio, we utilized the F10 and M10 speakers from the DAPS Clear dataset. These speakers were also employed in the evaluation of the [DAC model](https://arxiv.org/abs/2306.06546).


## Performance

We evaluated our codec using multiple objective audio quality metrics across two distinct test sets. Additionally, we compared our model's performance with state-of-the-art codecs. For further details, please refer to [our paper](https://arxiv.org/abs/2409.12117).

Please note that the released checkpoint yields slightly different results compared to those reported in the paper. Due to legal data constraints, we retrained the model after removing one speaker from the training set. This retraining was performed for 170k steps, compared to the original 124k steps, leading to slight improvements across almost all metrics.

Paper results:
| Dataset     | Squim MOS (↑)     |SI-SDR(↑)      |Mel Dist. (↓)      |STFT Dist.(↓) | CER (↓)|
|:-----------:|:----------:|:----------:|:----------:|:-----------:|:-----------:|
| MLS | 4.43       | 4.46      | 0.147      | 0.061       | 2.09       | 
| DAPS | 4.68       | 6.93       | 0.142       | 0.058       | 0.86      |

Released checkpoint results:
| Dataset     | Squim MOS (↑)     |SI-SDR(↑)      |Mel Dist. (↓)      |STFT Dist.(↓) | CER (↓)|
|:-----------:|:----------:|:----------:|:----------:|:-----------:|:-----------:|
| MLS |  4.43     |  4.77   |    0.143  |  0.060     |     2.16  | 
| DAPS | 4.69       | 8.07       | 0.136       | 0.056      |    0.77  |



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