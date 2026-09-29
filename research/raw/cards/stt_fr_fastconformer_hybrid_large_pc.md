---
language:
- fr
library_name: nemo
datasets:
- multilingual_librispeech
- mozilla-foundation/common_voice_12_0
- VoxPopuli
thumbnail: null
tags:
- automatic-speech-recognition
- speech
- audio
- Transducer
- FastConformer
- CTC
- Transformer
- pytorch
- NeMo
- hf-asr-leaderboard
license: cc-by-4.0
model-index:
- name: stt_fr_fastconformer_hybrid_large_pc
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: common-voice-12-0
      type: mozilla-foundation/common_voice_12_0
      config: fr
      split: test
      args:
        language: fr
    metrics:
    - name: Test WER
      type: wer
      value: 7.92
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: Multilingual LibriSpeech
      type: facebook/multilingual_librispeech
      config: french
      split: test
      args:
        language: fr
    metrics:
    - name: Test WER
      type: wer
      value: 5.21
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: Vox Populi
      type: facebook/voxpopuli
      config: french
      split: test
      args:
        language: fr
    metrics:
    - name: Test WER
      type: wer
      value: 6.49
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: common-voice-12-0
      type: mozilla-foundation/common_voice_12_0
      config: French P&C
      split: test
      args:
        language: fr
    metrics:
    - name: Test WER P&C
      type: wer
      value: 9.11
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: Multilingual LibriSpeech
      type: facebook/multilingual_librispeech
      config: French P&C
      split: test
      args:
        language: fr
    metrics:
    - name: Test WER P&C
      type: wer
      value: 10.6
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: Vox Populi
      type: facebook/voxpopuli
      config: French P&C
      split: test
      args:
        language: fr
    metrics:
    - name: Test WER P&C
      type: wer
      value: 8.97
---

# NVIDIA FastConformer-Hybrid Large (fr)

<style>
img {
 display: inline;
}
</style>

| [![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--Transducer_CTC-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-115M-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-fr-lightgrey#model-badge)](#datasets)


This model transcribes speech in upper and lower case French alphabet along with spaces, periods, commas, and question marks.
It is a "large" version of FastConformer Transducer-CTC (around 115M parameters) model. This is a hybrid model trained on two losses: Transducer (default) and CTC.
See the [model architecture](#model-architecture) section and [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer) for complete architecture details.

## NVIDIA NeMo: Training

To train, fine-tune or play with the model you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend you install it after you've installed latest Pytorch version.
```
pip install nemo_toolkit['all']
``` 

## How to Use this Model

The model is available for use in the NeMo toolkit [3], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

### Automatically instantiate the model

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_fr_fastconformer_hybrid_large_pc")
```

### Transcribing using Python
First, let's get a sample
```
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
```
Then simply do:
```
output = asr_model.transcribe(['2086-149220-0033.wav'])
print(output[0].text)
```

### Transcribing many audio files

Using Transducer mode inference:
```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/stt_fr_fastconformer_hybrid_large_pc" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

Using CTC mode inference:
```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/stt_fr_fastconformer_hybrid_large_pc" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
 decoder_type="ctc"
```

### Input

This model accepts 16000 Hz Mono-channel Audio (wav files) as input.

### Output

This model provides transcribed speech as a string for a given audio sample.

## Model Architecture

FastConformer [1] is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. The model is trained in a multitask setup with joint Transducer and CTC decoder loss. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer) and about Hybrid Transducer-CTC training here: [Hybrid Transducer-CTC](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#hybrid-transducer-ctc).

## Training

The NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe.yaml).

The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

### Datasets

All the models in this collection are trained on a composite dataset (NeMo PnC ASRSET) comprising of 1800 hours of French speech:

- MCV12 (710 hrs)
- MLS (925 hrs)
- Voxpopuli (165 hrs)

## Performance

The performance of Automatic Speech Recognition models is measuring using Word Error Rate. Since this dataset is trained on multiple domains and a much larger corpus, it will generally perform better at transcribing audio in general.

The following tables summarizes the performance of the available models in this collection with the Transducer decoder. Performances of the ASR models are reported in terms of Word Error Rate (WER%) with greedy decoding. 


a) On data without Punctuation and Capitalization with Transducer decoder
| **Version** |     **Tokenizer**     | **Vocabulary Size** | **MCV12 DEV** | **MCV12 TEST** | **MLS DEV** | **MLS TEST** | **VOXPOPULI DEV** | **VOXPOPULI TEST** |
|:-----------:|:---------------------:|:-------------------:|:-------------:|:--------------:|:-----------:|:------------:|:-----------------:|:------------------:|
|    1.18.0    | SentencePiece Unigram |         1024        |      6.84     |      7.92      |     5.0     |      5.21     |        5.86       |         6.49        |


b) On data with Punctuation and Capitalization with Transducer decoder
| **Version** |     **Tokenizer**     | **Vocabulary Size** | **MCV12 DEV** | **MCV12 TEST** | **MLS DEV** | **MLS TEST** | **VOXPOPULI DEV** | **VOXPOPULI TEST** |
|:-----------:|:---------------------:|:-------------------:|:-------------:|:--------------:|:-----------:|:------------:|:-----------------:|:------------------:|
|    1.18.0    | SentencePiece Unigram |         1024        |      8.04     |      9.11      |    10.95    |     10.6     |       8.5       |        8.97       |


## Limitations
Since this model was trained on publically available speech datasets, the performance of this model might degrade for speech which includes technical terms, or vernacular that the model has not been trained on. The model might also perform worse for accented speech. The model only outputs the punctuations: ```'.', ',', '?' ``` and hence might not do well in scenarios where other punctuations are also expected.

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

[2] [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece)

[3] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

## Licence

License to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.