---
language:
- kk
- ru
library_name: nemo
datasets:
  - mozilla-foundation/common-voice-17-0
  - Kazakh-Speech-Dataset
  - Kazakh-Speech-Corpus-2
  - mozilla-foundation/common_voice_12_0
  - SberDevices/Golos
  - SOVA-Dataset
  - Dusha-Dataset
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
- name: stt_kk_ru_fastconformer_hybrid_large
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: common-voice-17-0
      type: mozilla-foundation/common_voice_17_0
      config: kk
      split: test
      args:
        language: kk
    metrics:
    - name: Test WER
      type: wer
      value: 15.48
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Kazakh Speech Dataset
      type: Kazakh-Speech-Dataset
      config: kk
      split: test
      args:
        language: kk
    metrics:
    - name: Test WER
      type: wer
      value: 7.08
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Kazakh Speech Corpus 2 (read)
      type: Kazakh-Speech-Corpus-2
      config: kk
      split: test
      args:
        language: kk
    metrics:
    - name: Test WER
      type: wer
      value: 4.43
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Kazakh Speech Corpus 2 (spontaneous)
      type: Kazakh-Speech-Corpus-2
      config: kk
      split: test
      args:
        language: kk
    metrics:
    - name: Test WER
      type: wer
      value: 15.25
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: common-voice-12-0
      type: mozilla-foundation/common_voice_12_0
      config: ru
      split: test
      args:
        language: ru
    metrics:
    - name: Test WER
      type: wer
      value: 6.29
  - task:
        type: Automatic Speech Recognition
        name: automatic-speech-recognition
    dataset:
        name: Sberdevices Golos (crowd)
        type: SberDevices/Golos
        config: crowd
        split: test
        args:
          language: ru
    metrics:
    - name: Test WER
      type: wer
      value: 2.46
  - task:
        type: Automatic Speech Recognition
        name: automatic-speech-recognition
    dataset:
        name: Sberdevices Golos (farfield)
        type: SberDevices/Golos
        config: farfield
        split: test
        args:
          language: ru
    metrics:
    - name: Test WER
      type: wer
      value: 5.98
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Sova (RuAudiobooksDevices)
      type: SOVA-Dataset
      config: ru
      split: test
      args:
        language: ru
    metrics:
    - name: Test WER
      type: wer
      value: 4.41
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Sova (RuDevices)
      type: SOVA-Dataset
      config: ru
      split: test
      args:
        language: ru
    metrics:
    - name: Test WER
      type: wer
      value: 19.83

---

# NVIDIA FastConformer-Hybrid Large (kk-ru)

<style>
img {
 display: inline;
}
</style>

| [![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--Transducer_CTC-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-115M-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-kk_ru-lightgrey#model-badge)](#datasets)


This model transcribes speech in lower case Kazakh and Russian alphabet.
It is a "large" version of FastConformer Transducer-CTC (around 115M parameters) model. This is a hybrid model trained on two losses: Token-and-Duration Transducer (default) and CTC.
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
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_kk_ru_fastconformer_hybrid_large")
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
 pretrained_name="nvidia/stt_kk_ru_fastconformer_hybrid_large" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

Using CTC mode inference:
```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/stt_kk_ru_fastconformer_hybrid_large" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
 decoder_type="ctc"
```

### Input

This model accepts 16000 Hz Mono-channel Audio (wav files) as input.

### Output

This model provides transcribed speech as a string for a given audio sample.

## Model Architecture

FastConformer [1] is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. The model is trained in a multitask setup with joint Token-and-Duration Transducer and CTC decoder loss. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer) and about Hybrid Transducer-CTC training here: [Hybrid Transducer-CTC](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#hybrid-transducer-ctc).

## Training

The NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe.yaml).

The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

### Datasets

The model is trained on two composite datasets comprising of 1550 hours of Kazakh speech:

- MCV 17.0 Kazakh (1 hrs)
- Kazakh Speech Dataset (KSD) (416 hrs)
- Kazakh Speech Corpus 2 (KSC2) (1133 hrs)

and approximately 850 hrs of Russian speech:

- Golos (604 hrs)
- Sova (122 hrs)
- Dusha (102 hrs)
- MCV12 (19 hrs)

## Performance

The performance of Automatic Speech Recognition models is measuring using Word Error Rate. Since this dataset is trained on multiple domains and a much larger corpus, it will generally perform better at transcribing audio in general.

The following tables summarizes the performance of the model with the Transducer decoder. Performances of the ASR models are reported in terms of Word Error Rate (WER%) with greedy decoding. 


a) On Kazakh data
| **Version** |     **Tokenizer**     | **Vocabulary Size** | **MCV 17.0 TEST** | **KSD TEST** | **KSC2 TEST Read** | **KSC2 TEST Spontaneous** |
|:-----------:|:---------------------:|:-------------------:|:-----------------:|:------------:|:------------------:|:-------------------------:|
|    2.0.0    | SentencePiece Unigram |         1024        |       15.48       |      7.08    |        4.43        |            15.25          |


b) On Russian data
| **Version** |     **Tokenizer**     | **Vocabulary Size** | **MCV12 TEST** | **Sova TEST RuDevices** | **Sova TEST RuAudiobooksDevices** | **GOLOS TEST FARFIELD** | **GOLOS TEST CROWD** | **DUSHA TEST** |
|:-----------:|:---------------------:|:-------------------:|:--------------:|:-----------------------:|:---------------------------------:|:-----------------------:|:--------------------:|:--------------:|
|    2.0.0    | SentencePiece Unigram |         1024        |      6.29      |          19.83          |               4.41                |           5.98          |          2.46        |      5.93      |


## Limitations
The model is non-streaming and outputs the speech as a string without capitalization and punctuation. Since this model was trained on publicly available speech datasets, the performance of this model might degrade for speech which includes technical terms, or vernacular that the model has not been trained on. 

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