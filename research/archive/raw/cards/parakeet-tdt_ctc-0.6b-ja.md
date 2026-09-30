---
language:
- ja
library_name: nemo
datasets:
- reazon-research/reazonspeech

thumbnail: null
tags:
- automatic-speech-recognition
- speech
- audio
- Transducer
- TDT
- CTC
- FastConformer
- Conformer
- pytorch
- NeMo
license: cc-by-4.0
model-index:
- name: parakeet-tdt_ctc-0.6b-ja
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: JSUT basic5000
      type: japanese-asr/ja_asr.jsut_basic5000
      split: test
      args:
        language: ja
    metrics:
    - name: Test CER
      type: cer
      value: 6.4
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Mozilla Common Voice 8.0
      type:  mozilla-foundation/common_voice_8_0
      config: ja
      split: test
      args:
        language: ja
    metrics:
    - name: Test CER
      type: cer
      value: 7.1
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Mozilla Common Voice 16.1
      type:  mozilla-foundation/common_voice_16_1
      config: ja
      split: dev
      args:
        language: ja
    metrics:
    - name: Dev CER
      type: cer
      value: 10.1
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Mozilla Common Voice 16.1
      type:  mozilla-foundation/common_voice_16_1
      config: ja
      split: test
      args:
        language: ja
    metrics:
    - name: Test CER
      type: cer
      value: 13.2
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: TEDxJP-10k
      type: laboroai/TEDxJP-10K
      split: test
      args:
        language: ja
    metrics:
    - name: Test CER
      type: cer
      value: 9.0
  
metrics:
- cer
pipeline_tag: automatic-speech-recognition
---

# Parakeet TDT-CTC 0.6B (ja)

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--TDT--CTC-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-0.6B-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-ja-lightgrey#model-badge)](#datasets)


`parakeet-tdt_ctc-0.6b-ja` is an ASR model that transcribes Japanese speech with Punctuations. This model is developed by [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) team.
It is an XL version of Hybrid FastConformer [1] TDT-CTC [2] (around 0.6B parameters) model.  
See the [model architecture](#model-architecture) section and [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer) for complete architecture details.

## NVIDIA NeMo: Training

To train, fine-tune or play with the model you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend you install it after you've installed latest PyTorch version.
```
pip install nemo_toolkit['asr']
``` 

## How to Use this Model

The model is available for use in the NeMo Framework [3], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

### Automatically instantiate the model

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-0.6b-ja")
```

### Transcribing using Python
Simply do:
```
output = asr_model.transcribe(['speech.wav'])
print(output[0].text)
```

### Transcribing many audio files

By default model uses TDT to transcribe the audio files, to switch decoder to use CTC, use decoding_type='ctc'

```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/parakeet-tdt_ctc-0.6b-ja" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

### Input

This model accepts 16000 Hz mono-channel audio (wav files) as input.

### Output

This model provides transcribed speech as a string for a given audio sample.

## Model Architecture

This model uses a Hybrid FastConformer-TDT-CTC architecture. 

FastConformer [1] is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).

TDT (Token-and-Duration Transducer) [2] is a generalization of conventional Transducers by decoupling token and duration predictions. Unlike conventional Transducers which produces a lot of blanks during inference, a TDT model can skip majority of blank predictions by using the duration output (up to 4 frames for this `parakeet-tdt_ctc-0.6b-ja` model), thus brings significant inference speed-up. The detail of TDT can be found here: [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://arxiv.org/abs/2304.06795).

## Training

The NeMo Framework [3] was used for training this model with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe.yaml).

The model was trained for 300k steps with dynamic bucketing and a batch duration of 600s per GPU on 32 NVIDIA A100 80GB GPUs, and then finetuned for 100k additional steps on the modified training data (predicted texts for training samples with CER>10%).

SentencePiece [4] tokenizer with 3072 tokens for this model was built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

### Datasets

The model was trained on ReazonSpeech v2.0 [5] speech corpus containing more than 35k hours of natural Japanese speech. 

## Performance

The following table summarizes the performance of this model in terms of Character Error Rate (CER%).

In CER calculation, punctuation marks and non-alphabet characters are removed, and numbers are transformed to words using `num2words` library [6].

|**Version**|**Decoder**|**JSUT basic5000**|**MCV 8.0 test**|**MCV 16.1 dev**|**MCV16.1 test**|**TEDxJP-10k**|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.23.0 | TDT | 6.4 | 7.1 | 10.1 | 13.2 | 9.0 |
| 1.23.0 | CTC | 6.5 | 7.2 | 10.2 | 13.3 | 9.1 |

These are greedy CER numbers without external LM.

## NVIDIA Riva: Deployment

[NVIDIA Riva](https://developer.nvidia.com/riva), is an accelerated speech AI SDK deployable on-prem, in all clouds, multi-cloud, hybrid, on edge, and embedded. 
Additionally, Riva provides: 

* World-class out-of-the-box accuracy for the most common languages with model checkpoints trained on proprietary data with hundreds of thousands of GPU-compute hours 
* Best in class accuracy with run-time word boosting (e.g., brand and product names) and customization of acoustic model, language model, and inverse text normalization 
* Streaming speech recognition, Kubernetes compatible scaling, and enterprise-grade support 

Although this model isn’t supported yet by Riva, the [list of supported models is here](https://huggingface.co/models?other=Riva).  
Check out [Riva live demo](https://developer.nvidia.com/riva#demos). 

## References
[1] [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)

[2] [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://arxiv.org/abs/2304.06795)

[3] [NVIDIA NeMo Framework](https://github.com/NVIDIA/NeMo)

[4] [Google SentencePiece Tokenizer](https://github.com/google/sentencepiece)

[5] [ReazonSpeech v2.0](https://huggingface.co/datasets/reazon-research/reazonspeech)

[6] [num2words library - Convert numbers to words in multiple languages](https://github.com/savoirfairelinux/num2words)

## Licence

License to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.