---
language:
- da
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
library_name: nemo
tags:
- automatic-speech-recognition
- parakeet 
- nemo
metrics:
- wer
---

# NVIDIA Parakeet-RNNT 110M (Danish)
`parakeet-rnnt-110m-da-dk`

## Description:

`parakeet-rnnt-110m-da-dk` is a 110-million-parameter automatic speech recognition (ASR) model trained to transcribe Danish speech.

This model uses the FastConformer architecture trained with RNN-T (RNN-Transducer) loss.


For more information, refer to the [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).

This model is ready for commercial/non-commercial use.

### License/Terms of Use: 

GOVERNING TERMS: Use of this model is governed by the NVIDIA Open Model License Agreement (found at [https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)).

### Deployment Geography:

Global

### Use Case:

Performing speech-to-text transcription in Danish language

### Release Date: 
Huggingface 10/15/25 via https://huggingface.co/nvidia/parakeet-rnnt-110m-da-dk/tree/main

## References:

[1] [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)

[2] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)


## Model Architecture: 
**Architecture Type:**  Transducer model<br>
**Network Architecture:** FastConformer encoder and LSTM prediction network

### Input

**Input Type(s):** Audio<br>
**Input Format(s):** 16000 Hz mono-channel<br>
**Input Parameters:** 1D<br>

### Output

**Output Type(s):** Text
**Output Format:** string
**Output Parameters:** 1D
**Other Properties Related to Output:** Not applicable

This model provides transcribed speech as a string for a given audio with Danish speech.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration: <br>

* NeMo-2.5 <br>

**Supported Hardware Microarchitecture Compatibility:** <br>

* NVIDIA Ampere <br>
* NVIDIA Blackwell <br>
* NVIDIA Hopper <br>

**Preferred Operating System(s):**

Linux


## Model Version: 

1.0

# Training and Evaluation Datasets: <br> 

The total size: ~10000 hours <br>
Total number of datasets: 3 <br> 

## Training datasets
Link: [CoRal (read aloud part)](https://huggingface.co/datasets/CoRal-project/coral)
  
      - Data Collection Method by Dataset: Human
      - Labeling Method by Dataset: Human
      - Properties: read aloud Danish speech
Link: [Mozilla Common Voice v17](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
  
      - Data Collection Method by Dataset: Human
      - Labeling Method by Dataset: Automated
      - Properties: sentences recorded by different people

Link: [Granary Danish](https://huggingface.co/datasets/nvidia/Granary) a cleaned version of Danish part of [MOSEL](https://huggingface.co/datasets/FBK-MT/mosel)

      - Data Collection Method by Dataset: Human
      - Labeling Method by Dataset: Automated
      - Properties: EU Parlament sessions

## Evaluation Datasets

Link: [CoRal (read aloud part)](https://huggingface.co/datasets/CoRal-project/coral)
  
      - Data Collection Method by Dataset: Human
      - Labeling Method by Dataset: Human
      - Properties: read aloud Danish speech
Link: [Mozilla Common Voice v17](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
  
      - Data Collection Method by Dataset: Human
      - Labeling Method by Dataset: Automated
      - Properties: sentences recorded by different people

Link: [FLEURS](https://huggingface.co/datasets/google/fleurs)
  
      - Data Collection Method by Dataset: Human
      - Labeling Method by Dataset: Human
      - Properties: sentences recorded by different people


## Performance

The table below summarizes the WER (%) of this model using RNN-T decoder with greedy decoding:

| **Dataset** | **WER (%)** |
|-------------|-------------|
| CoRal Test  | 10.7        |
| MCV v17     | 8.8         |
| Google FLEURS | 10.6      |

These WER scores were obtained using greedy decoding without an external language model.

## NVIDIA Riva: Deployment

[NVIDIA Riva](https://developer.nvidia.com/riva), is an accelerated speech AI SDK deployable on-prem, in all clouds, multi-cloud, hybrid, on edge, and embedded. 
Additionally, Riva provides: 

* World-class out-of-the-box accuracy for the most common languages with model checkpoints trained on proprietary data with hundreds of thousands of GPU-compute hours 
* Best in class accuracy with run-time word boosting (e.g., brand and product names) and customization of acoustic model, language model, and inverse text normalization 
* Streaming speech recognition, Kubernetes compatible scaling, and enterprise-grade support 

Although this model isn’t supported yet by Riva, the [list of supported models is here](https://huggingface.co/models?other=Riva).  
Check out [Riva live demo](https://developer.nvidia.com/riva#demos). 

## Inference

Acceleration Engine: Not Applicable 
Test Hardware: NVIDIA RTX 6000

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## How to Use this Model

To train, fine-tune or play with the model you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend you install it after you've installed latest PyTorch version.
```bash
pip install nemo_toolkit['all']
``` 

Alternatively, you can use a Docker container. The code below was verified to work with [NeMo Framework 25.09](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo?version=25.09).

### Automatically instantiate the model

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-rnnt-110m-da-dk")
```

### Transcribing using Python
Assuming you have an audio file with Danish speech simply do:
```python
output = asr_model.transcribe(['danish_audio.wav'])
print(output[0])
```



