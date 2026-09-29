---
license: cc-by-nc-sa-4.0
library_name: NeMo
tags:
- NeMo
- speech
- audio
---
# SR SSL FlowMatching 16kHz 430M
<style>
img {
display: inline-table;
vertical-align: small;
margin: 0;
padding: 0;
}
</style>
[![Model architecture](https://img.shields.io/badge/Model_Arch-FlowMatching-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-430M-lightgrey#model-badge)](#model-architecture)

## Model Overview

### Description

This is a generative speech restoration model based on flow matching. The model is pre-trained on a publicly available Libri-Light dataset by using self-supervised learning technique. The model can be finetuned on various speech restoration tasks, such as speech denoising, bandwidth extraction, and codec artifact removal for human or machine listeners.

This model is for research and development only.

### License/Terms of Use
License to use this model is covered by the [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0) license.

## References

[1] [Generative Speech Foundation Model Pretraining for High-Quality Speech Extraction and Restoration](https://arxiv.org/abs/2409.16117), 2024.

## Model Architecture
**Architecture Type:** Conditional Flow Matching  <br>
**Network Architecture:** Transformer <br>

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
`sr_ssl_flowmatching_16k_430m_v1.0`<br>

# Training, Testing, and Evaluation Datasets

## Training Dataset
**Link:**
[Libri-Light](https://github.com/facebookresearch/libri-light)

**Data Collection Method by dataset:** Human <br>

**Labeling Method by dataset:** Not Applicable<br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
Approximately 60k hours of English speech data <br>

## Testing Dataset
**Link:** Not Applicable<br>

## Evaluation Dataset
**Link:** Not applicable<br>

## Inference
**Engine:** NeMo 2.0 <br>

**Test Hardware:** NVIDIA H100<br>

# How to use this model

The model is available for use in the NVIDIA NeMo toolkit, and can be used fine-tuning on various speech tasks.

## Load the model
```
from nemo.collections.audio.models import AudioToAudioModel
model = AudioToAudioModel.from_pretrained('nvidia/sr_ssl_flowmatching_16k_430m')
```

## Change sampler configuration
```
model.sampler.num_steps = 20 # default is 50 steps
```

## Finetuning
For finetuning, use `init_from_nemo_model` to provide a path to a local NeMo model or `init_from_pretrained_model` to download a pretrained NeMo model.
For example, use the following in finetuning configuration
```
init_from_pretrained_model: sr_ssl_flowmatching_16k_430m
```
An example of a finetuning configuration can be found in [NeMo](https://github.com/NVIDIA/NeMo/blob/main/examples/audio/conf/flow_matching_generative_finetuning.yaml).

# Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
