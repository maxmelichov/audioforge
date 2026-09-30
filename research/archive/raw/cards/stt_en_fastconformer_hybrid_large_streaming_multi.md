---
language:
- en
library_name: nemo
datasets:
- librispeech_asr
- fisher_corpus
- Switchboard-1
- WSJ-0
- WSJ-1
- National-Singapore-Corpus-Part-1
- National-Singapore-Corpus-Part-6
- vctk
- VoxPopuli-EN
- Europarl-ASR-EN
- Multilingual-LibriSpeech-2000hours
- mozilla-foundation/common_voice_8_0
- MLCommons/peoples_speech
thumbnail: null
tags:
- automatic-speech-recognition
- speech
- streaming
- audio
- Transducer
- Conformer
- CTC
- pytorch
- NeMo
license: cc-by-4.0
widget:
- example_title: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: Librispeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
model-index:
- name: stt_en_fastconformer_hybrid_large_streaming_multi
  results:
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
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
      value: 5.4
---

# NVIDIA Streaming Conformer-Hybrid Large (en-US)

<style>
img {
 display: inline;
}
</style>

| [![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--Hybrid-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-114M-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-en--US-lightgrey#model-badge)](#datasets)

This collection contains large-size versions of cache-aware FastConformer-Hybrid (around 114M parameters) with multiple look-ahead support, trained on a large scale english speech. 
These models are trained for streaming ASR, which be used for streaming applications with a variety of latencies (0ms, 80ms, 480s, 1040ms). 
These are the worst latency and average latency of the model for each case would be half of these numbers. You may find more detail and evalution results [here](https://arxiv.org/abs/2312.17279) [5].


## Model Architecture

These models are cache-aware versions of Hybrid FastConfomer which are trained for streaming ASR. You may find more info on cache-aware models here: [Cache-aware Streaming Conformer](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#cache-aware-streaming-conformer) [5].
The models are trained with multiple look-aheads which makes the model to be able to support different latencies. 
To learn on how to switch between different look-ahead, you may read the documentation on the cache-aware models.

FastConformer [4] is an optimized version of the Conformer model [1], and 
you may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer). 

The model is trained in a multitask setup with joint Transducer and CTC decoder loss [5]. You can find more about Hybrid Transducer-CTC training here: [Hybrid Transducer-CTC](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#hybrid-transducer-ctc). 
You may also find more on how to switch between the Transducer and CTC decoders in the documentation.

 


## Training

The NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_cache_aware_streaming/fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml). The SentencePiece tokenizers [2] for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).


### Datasets

All the models in this collection are trained on a composite dataset (NeMo ASRSET) comprising of several thousand hours of English speech:

- Librispeech 960 hours of English speech
- Fisher Corpus
- Switchboard-1 Dataset
- WSJ-0 and WSJ-1
- National Speech Corpus (Part 1, Part 6)
- VCTK
- VoxPopuli (EN)
- Europarl-ASR (EN)
- Multilingual Librispeech (MLS EN) - 2,000 hours subset
- Mozilla Common Voice (v7.0)
- People's Speech - 12,000 hrs subset

Note: older versions of the model may have trained on smaller set of datasets.

## Performance

The list of the available models in this collection is shown in the following tables for both CTC and Transducer decoders. 
Performances of the ASR models are reported in terms of Word Error Rate (WER%) with greedy decoding.

### Transducer Decoder

|  att_context_sizes | LS test-other ([70,13]-1040ms) | LS test-other ([70,6]-480ms) | LS test-other ([70,1]-80ms) | LS test-other ([70,0]-0s) | Train Dataset |
|-----------------------|-----------------------|---------------|------------|-----------|----------------|
| [[70,13],[70,6],[70,1],[70,0]] | 5.4           | 5.7                        | 6.4                   | 7.0       | NeMo ASRSET 3.0 |

### CTC Decoder

|att_context_sizes | LS test-other ([70,13]-1040ms) | LS test-other ([70,6]-480ms) | LS test-other ([70,1]-80ms) | LS test-other ([70,0]-0s) | Train Dataset |
|--------|---------------|---------------|------------|-----------|----------------|
|[[70,13],[70,6],[70,1],[70,0]] | 6.2           | 6.7                        | 7.8                   | 8.4       | NeMo ASRSET 3.0 |


## How to Use this Model

The model is available for use in the NeMo toolkit [3], and can be used as a pre-trained checkpoint for streaming or for fine-tuning on another dataset. 
You will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend you install it after you've installed latest Pytorch version.

```
pip install nemo_toolkit['all']
```

### Simulate Streaming ASR

You may use this script to simulate streaming ASR with these models: [cache-aware streaming simulation](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py). 
You may use --att_context_size to set the context size otherwise, the default, which is the first context size in the list (1040ms), is going to be used.

### Transcribing using Python
Cache-aware models are designed in a way that the model's predictions are the same in both offline and streaming mode. 

So you may use the regular transcribe function to get the transcriptions. First, let's get a sample:

```
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
```

Then simply do:

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")

# Optional: change the default latency. Default latency is 1040ms. Supported latencies: {0: 0ms, 1: 80ms, 16: 480ms, 33: 1040ms}.
# Note: These are the worst latency and average latency would be half of these numbers.
asr_model.encoder.set_default_att_context_size([70,13]) 

#Optional: change the default decoder. Default decoder is Transducer (RNNT). Supported decoders: {ctc, rnnt}.
asr_model.change_decoding_strategy(decoder_type='rnnt')

output = asr_model.transcribe(['2086-149220-0033.wav'])
print(output[0].text)
```

### Transcribing many audio files

Using Transducer mode inference:

```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_fastconformer_hybrid_large_streaming_multi" \
  audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

Using CTC mode inference:

```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_fastconformer_hybrid_large_streaming_multi" \
  audio_dir="<DIRECTORY CONTAINING AUDIO FILES>" \
  decoder_type="ctc"
```

To change between different look-aheads you may set att_context_size of the script transcribe_speech.py as the following:

```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_fastconformer_hybrid_large_streaming_multi" \
  audio_dir="<DIRECTORY CONTAINING AUDIO FILES>" \
  att_context_size=[70,0]
```
Supported values for att_context_size: {[70,0]: 0ms, [70,1]: 80ms, [70,16]: 480ms, [70,33]: 1040ms}.

### Input

This model accepts 16000 KHz Mono-channel Audio (wav files) as input.

### Output

This model provides transcribed speech as a string for a given audio sample.

## Limitations
Since this model was trained on publicly available speech datasets, the performance of this model might degrade for speech which includes technical terms, or vernacular that the model has not been trained on. The model might also perform worse for accented speech.

## NVIDIA Riva: Deployment

[NVIDIA Riva](https://developer.nvidia.com/riva), is an accelerated speech AI SDK deployable on-prem, in all clouds, multi-cloud, hybrid, on edge, and embedded. 
Additionally, Riva provides: 

* World-class out-of-the-box accuracy for the most common languages with model checkpoints trained on proprietary data with hundreds of thousands of GPU-compute hours 
* Best in class accuracy with run-time word boosting (e.g., brand and product names) and customization of acoustic model, language model, and inverse text normalization 
* Streaming speech recognition, Kubernetes compatible scaling, and enterprise-grade support 

Although this model isn’t supported yet by Riva, the [list of supported models is here](https://huggingface.co/models?other=Riva).  
Check out [Riva live demo](https://developer.nvidia.com/riva#demos). 

## References


[1] [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)

[2] [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece)

[3] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

[4] [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)

[5] [Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition
](https://arxiv.org/abs/2312.17279)
