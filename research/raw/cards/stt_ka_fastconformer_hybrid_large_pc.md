---
language:
- ka
library_name: nemo
datasets:
- mozilla-foundation/common_voice_17_0
- google/fleurs
metrics:
- wer
- cer
tags:
- automatic-speech-recognition
- speech
- audio
- CTC
- FastConformer
- Transformer
- NeMo
- Georgian language
- pytorch
- hybrid
license: cc-by-4.0
model-index:
- name: stt_ka_fastconformer_hybrid_large_pc
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: common-voice-17-0
      type: mozilla-foundation/common_voice_17_0
      config: ka
      split: test
      args:
        language: ka
    metrics:
    - name: Test WER
      type: wer
      value: 5.73
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: Fleurs
      type: google/fleurs
      config: ka
      split: test
      args:
        language: ka
    metrics:
    - name: Test WER
      type: wer
      value: 13.44
---

# NVIDIA FastConformer-Hybrid Large (ka)

This model transcribes speech in Georgian alphabet along with spaces, periods, commas, and question marks.
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
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_ka_fastconformer_hybrid_large_pc")
```

### Transcribing using Python
Simply do:
```
output = asr_model.transcribe(['sample.wav'])
print(output[0].text)
```
### Transcribing many audio files

Using Transducer mode inference:
```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py
 pretrained_name="nvidia/stt_ka_fastconformer_hybrid_large_pc"
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```
Using CTC mode inference:
```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py
 pretrained_name="nvidia/stt_ka_fastconformer_hybrid_large_pc"
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
 decoder_type="ctc"
```

### Input

This model accepts 16000 Hz Mono-channel Audio (wav files) as input.

### Output

This model provides transcribed speech as a string for a given audio sample.

## Model Architecture

FastConformer [1] is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. The model is trained in a multitask setup with a Transducer decoder loss. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).

## Training

The NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml).
The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

### Datasets

The model in this collection is trained on two datasets comprising approxinately 163 hours of Georgian speech:

- Mozilla Common Voice (v17.0) - 96 hours of validated data, 63 hours of unvalidated data
- Google Fleurs - 4 hours

## Performance

The performance of Automatic Speech Recognition models is measuring using Word Error Rate. 
The following tables summarizes the performance of the available models in this collection with the Transducer decoder. Performances of the ASR models are reported in terms of Word Error Rate (WER%) with greedy decoding.
|**Tokenizer**|**Vocabulary Size**|**MCV-test**|**Fleurs test**| Train Dataset |
|-----------------------|-----------------|---------------|---------------|------------|
| SentencePiece Unigram | 1024 | 5.73 | 13.44 | MCV(Train,Dev,Other),Fleurs(Train,Dev)|

## Limitations

Since this model was trained on publically available speech datasets, the performance of this model might degrade for speech which includes technical terms, or vernacular that the model has not been trained on. The model might also perform worse for accented speech.

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