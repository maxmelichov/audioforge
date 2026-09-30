---
license: cc-by-nc-sa-4.0
library_name: NeMo
tags:
- NeMo
- speech
- audio
---
# SE Dereverberation SB 16kHz Small
<style>
img {
display: inline-table;
vertical-align: small;
margin: 0;
padding: 0;
}
</style>
[![Model architecture](https://img.shields.io/badge/Model_Arch-Schrödinger_Bridge-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-25M-lightgrey#model-badge)](#model-architecture)

## Model Overview

### Description

The model extracts speech for human or machine listeners. This is a generative speech dereverberation model based on the Schrödinger bridge. The model is trained on a publicly available research dataset. 

This model is for research and development only.

### License/Terms of Use
License to use this model is covered by the [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0) license.

## References

[1] [Schrödinger Bridge for Generative Speech Enhancement](https://arxiv.org/abs/2407.16074), Interspeech, 2024.

## Model Architecture
**Architecture Type:** Schrödinger Bridge<br>
**Network Architecture:** U-Net with convolutional layers<br>

## Input
**Input Type(s):** Audio <br>
**Input Format(s):** .wav files <br>
**Input Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Input:** 16000 Hz Mono-channel Audio <br>

## Output
**Output Type(s):** Audio <br>
**Output Format:** .wav files <br>
**Output Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Output:** 16000 Hz Mono-channel Audio <br>

## Software Integration
**Runtime Engine(s):**<br>
* NeMo-2.0.0 <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Ampere<br>
* NVIDIA Blackwell<br>
* NVIDIA Jetson<br>
* NVIDIA Hopper<br>
* NVIDIA Lovelace<br>
* NVIDIA Turing<br>
* NVIDIA Volta<br>

**Preferred Operating System(s)** <br>
* Linux<br>
* Windows<br>

## Model Version(s)
`se_der_sb_16k_small_v1.0`<br>

# Training, Testing, and Evaluation Datasets

## Training Dataset
**Link:**
[WSJ0](https://catalog.ldc.upenn.edu/LDC93S6A)

**Data Collection Method by dataset:** Human <br>

**Labeling Method by dataset:** Human<br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
WSJ0 was used for clean speech signals. The observed signals are simulated with room impulse responses with reverberation times between 0.4 seconds and 1.0 seconds, and without any background noise. The total size of the training dataset was approximately 25 hours.<br>

## Testing Dataset
**Link:**
[WSJ0](https://catalog.ldc.upenn.edu/LDC93S6A)

**Data Collection Method by dataset:** Human <br>

**Labeling Method by dataset:** Human<br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
WSJ0 was used for clean speech signals. The observed signals are simulated with room impulse responses with reverberation times between 0.4 seconds and 1.0 seconds, and without any background noise. The total size of the training dataset was approximately 2 hours.<br>

## Evaluation Dataset
**Link:**
[WSJ0](https://catalog.ldc.upenn.edu/LDC93S6A)

**Data Collection Method by dataset:** Human <br>

**Labeling Method by dataset:** Human<br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
WSJ0 was used for clean speech signals. The observed signals are simulated with room impulse responses with reverberation times between 0.4 seconds and 1.0 seconds, and without any background noise. The total size of the training dataset was approximately 2 hours.<br>

## Inference
**Engine:** NeMo 2.0 <br>

**Test Hardware:** NVIDIA v100<br>

# Performance

The model is trained on the training subset of the WSJ0-Reverb dataset using the auxiliary L1-norm loss [1].

The model is evaluated using several instrumental metrics: perceptual evaluation of speech quality (PESQ), extended short-term objective intelligibility (ESTOI) and scale-invariant signal-to-distortion ratio (SI-SDR). Word error rate (WER) is evaluated using the [FastConformer-Transducer-Large English ASR model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_transducer_large).

Metrics are reported on the test set of WSJ0-Reverb dataset using either SDE or ODE sampler.

| Signal        |PESQ  | ESTOI | SI-SDR/dB | WER / % |
|:-------------:|:----:|:-----:|:---------:|:-------:|
| Input         | 1.29 | 0.44  | -9.5      | 8.29    |
| Processed SDE | 2.79 | 0.89  |  7.4      | 4.27    |
| Processed ODE | 2.59 | 0.86  |  6.2      | 5.79    |

# How to use this model

The model is available for use in the NVIDIA NeMo toolkit, and can be used to process audio or for fine-tuning.

## Load the model
```
from nemo.collections.audio.models import AudioToAudioModel
model = AudioToAudioModel.from_pretrained('nvidia/se_der_sb_16k_small')
```

## Process audio
A single audio file can be processed as follows

```
import librosa
audio_in, _ = librosa.load(path_to_input_audio, sr=model.sample_rate)
audio_in_signal = torch.from_numpy(audio_in).view(1, 1, -1).to(device)
audio_in_length = torch.tensor([audio_in_signal.size(-1)]).to(device)

audio_out_signal, _ = model(input_signal=audio_in_signal, input_length=audio_in_length)
```

For processing several audio files at once, check the [process_audio script](https://github.com/NVIDIA/NeMo/blob/main/examples/audio/process_audio.py) in NeMo.

## Listen to audio
```
import soundfile as sf
audio_out = audio_out_signal.cpu().numpy().squeeze()
sf.write(path_to_output_audio, audio_out, samplerate=model.sample_rate)
```

## Change sampler configuration
```
model.sampler.process = 'ode' # default sampler is 'sde'
model.sampler.num_steps = 10 # default is 50 steps

audio_out_signal, _ = model(input_signal=audio_in_signal, input_length=audio_in_length)
```

# Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
