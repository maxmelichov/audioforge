---
language:
- en
library_name: nemo
datasets:
- librispeech_asr
- fisher_corpus
- mozilla-foundation/common_voice_8_0
- National-Singapore-Corpus-Part-1
- vctk
- voxpopuli
- europarl
- multilingual_librispeech

thumbnail: null
tags:
- automatic-speech-recognition
- speech
- audio
- Transducer
- TDT
- FastConformer
- Conformer
- pytorch
- NeMo
- hf-asr-leaderboard
license: cc-by-4.0
widget:
- example_title: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: Librispeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
model-index:
- name: parakeet-tdt_ctc-1.1b
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: AMI (Meetings test)
      type: edinburghcstr/ami
      config: ihm
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 15.94
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Earnings-22
      type: revdotcom/earnings22
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 11.86
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: GigaSpeech
      type: speechcolab/gigaspeech
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 10.19
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: LibriSpeech (clean)
      type: librispeech_asr
      config: other
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 1.82
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: LibriSpeech (other)
      type: librispeech_asr
      config: other
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 3.67
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: SPGI Speech
      type: kensho/spgispeech
      config: test
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 2.24
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: tedlium-v3
      type: LIUM/tedlium
      config: release1
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 3.87
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Vox Populi
      type: facebook/voxpopuli
      config: en
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 6.19
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: Mozilla Common Voice 9.0
      type: mozilla-foundation/common_voice_9_0
      config: en
      split: test
      args:
        language: en
    metrics:
    - name: Test WER
      type: wer
      value: 8.69
  
metrics:
- wer
pipeline_tag: automatic-speech-recognition
---

# Parakeet TDT-CTC 1.1B PnC(en)

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--TDT-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-1.1B-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-en-lightgrey#model-badge)](#datasets)


`parakeet-tdt_ctc-1.1b` is an ASR model that transcribes speech with Punctuations and Capitalizations of English alphabet. This model is jointly developed by [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) and [Suno.ai](https://www.suno.ai/) teams.
It is an XXL version of Hybrid FastConformer [1] TDT-CTC [2] (around 1.1B parameters) model. This model has been trained with Local Attention and Global token hence this model can transcribe **11 hrs** of audio in one single pass. And for reference this model can transcibe 90mins of audio in <16 sec on A100.  
See the [model architecture](#model-architecture) section and [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer) for complete architecture details.

## NVIDIA NeMo: Training

To train, fine-tune or play with the model you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend you install it after you've installed latest PyTorch version.
```
pip install nemo_toolkit['all']
``` 

## How to Use this Model

The model is available for use in the NeMo toolkit [3], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

### Automatically instantiate the model

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-1.1b")
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

By default model uses TDT to transcribe the audio files, to switch decoder to use CTC, use decoding_type='ctc'

```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/parakeet-tdt_ctc-1.1b" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

### Input

This model accepts 16000 Hz mono-channel audio (wav files) as input.

### Output

This model provides transcribed speech as a string for a given audio sample.

## Model Architecture

This model uses a Hybrid FastConformer-TDT-CTC architecture. FastConformer [1] is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).


## Training

The NeMo toolkit [3] was used for finetuning this model for 20,000 steps over `parakeet-tdt-1.1` model. This model is trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_transducer_ctc_bpe.yaml).

The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

### Datasets

The model was trained on 36K hours of English speech collected and prepared by NVIDIA NeMo and Suno teams.

The training dataset consists of private subset with 27K hours of English speech plus 9k hours from the following public PnC datasets:

- Librispeech 960 hours of English speech
- Fisher Corpus
- National Speech Corpus Part 1
- VCTK
- VoxPopuli (EN)
- Europarl-ASR (EN)
- Multilingual Librispeech (MLS EN) - 2,000 hour subset
- Mozilla Common Voice (v7.0)

## Performance

The performance of Automatic Speech Recognition models is measuring using Word Error Rate. Since this dataset is trained on multiple domains and a much larger corpus, it will generally perform better at transcribing audio in general.

The following tables summarizes the performance of the available models in this collection with the Transducer decoder. Performances of the ASR models are reported in terms of Word Error Rate (WER%) with greedy decoding. 

|**Version**|**Tokenizer**|**Vocabulary Size**|**AMI**|**Earnings-22**|**Giga Speech**|**LS test-clean**| **LS test-other** | **SPGI Speech**|**TEDLIUM-v3**|**Vox Populi**|**Common Voice**|
|---------|-----------------------|-----------------|---------|---------------|------------|-----------|---------------|-------------|------|------|--------|
| 1.23.0  | SentencePiece Unigram | 1024            | 15.94         | 11.86         | 10.19      | 1.82      | 3.67 | 2.24 | 3.87 | 6.19 | 8.69 |

These are greedy WER numbers without external LM. More details on evaluation can be found at [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)


## Model Fairness Evaluation

As outlined in the paper "Towards Measuring Fairness in AI: the Casual Conversations Dataset", we assessed the parakeet-tdt_ctc-1.1b model for fairness. The model was evaluated on the CausalConversations-v1 dataset, and the results are reported as follows:

### Gender Bias:

| Gender | Male | Female | N/A | Other |
| :--- | :--- | :--- | :--- | :--- |
| Num utterances | 19325 | 24532 | 926 | 33 |
| % WER | 12.81 | 10.49 | 13.88 | 23.12 |

### Age Bias:

| Age Group | (18-30) | (31-45) | (46-85) | (1-100) |
| :--- | :--- | :--- | :--- | :--- |
| Num utterances | 15956 | 14585 | 13349 | 43890 |
| % WER | 11.50 | 11.63 | 11.38 | 11.51 |

(Error rates for fairness evaluation are determined by normalizing both the reference and predicted text, similar to the methods used in the evaluations found at https://github.com/huggingface/open_asr_leaderboard.)

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

[3] [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece)

[4] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

[5] [Suno.ai](https://suno.ai/)

[6] [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

[7] [Towards Measuring Fairness in AI: the Casual Conversations Dataset](https://arxiv.org/abs/2104.02821)


## Licence

License to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.