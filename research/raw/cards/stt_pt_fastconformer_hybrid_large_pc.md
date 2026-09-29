---
license: cc-by-nc-4.0
language:
- pt
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
library_name: nemo
datasets:
- mozilla-foundation/common_voice_16_0
- facebook/multilingual_librispeech
tags:
- FastConformer
- NeMo
- Portuguese
---

# Model Overview

## Description:
STT PT FastConformer Hybrid Transducer-CTC Large transcribes text in upper and lower case Portuguese alphabet along with spaces, period, comma, question mark.  This collection contains the Brazilian Portuguese FastConformer Hybrid (Transducer and CTC) Large model (around 115M parameters) with punctuation and capitalization trained on around 2200h hours of Portuguese speech. 
See the [model architecture](#model-architecture) section and [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer) for complete architecture details.

It utilizes a Google SentencePiece [1] tokenizer with a vocabulary size of 128.

This model is ready for non-commercial use.


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
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_pt_fastconformer_hybrid_large_pc")
```

### Transcribing using Python
Having instantiated the model, simply do:
```
asr_model.transcribe([path_to_audio_file])
```

### Transcribing many audio files

Using Transducer mode inference:
```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/stt_pt_fastconformer_hybrid_large_pc" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

Using CTC mode inference:
```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/stt_pt_fastconformer_hybrid_large_pc" 
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

The NeMo toolkit [3] was used for training the models for over several hundred epochs. The model was trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/speech_to_text_finetune.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/asr_finetune/speech_to_text_finetune.yaml).
The tokenizers for this model was built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

The model was initialized with the weights of [Spanish FastConformer Hybrid (Transducer and CTC) Large P&C model](https://huggingface.co/nvidia/stt_es_fastconformer_hybrid_large_pc) and fine-tuned to Portuguese using the labeled and unlabeled data(with pseudo-labels).
The MLS dataset was used as unlabeled data as it does not contain punctuation and capitalization.

## Training Dataset:

The model was trained on around 2200 hours of Portuguese speech data.

- [Mozilla Common Voice 16.0  Portuguese](https://commonvoice.mozilla.org/en/datasets) [83h]

    - Data Collection Method: by Human
    
    - Labeling Method: by Human
 
- [Multilingual Librispeech](https://www.openslr.org/94/) [160h]

    - Data Collection Method: by Human
    
    - Labeling Method: Pseudo-labels
 
- Proprietary corpus [2000h]

  - Data Collection Method: by Human
  
  - Labeling Method: Pseudo-labels



## Testing Dataset:

**Link:**
1. [Mozilla Common Voice 16(MCV16)](https://commonvoice.mozilla.org/en/datasets) <br>
2.  [Multilingual Librispeech](https://www.openslr.org/94/) <br>


## Performance

**Test Hardware:** A5000 GPU

The performance of Automatic Speech Recognition models is measured using Character Error Rate (CER) and Word Error Rate (WER).
The following table summarize the performance of the available model in this collection with the Transducer and CTC decoders.

 

 | Model     | MCV %WER/CER test |MLS %WER/CER test |
 |-----------|--------------|---------------|
 | RNNT head | 12.03 / 3.20 | 24.78 / 5.92  |
 | CTC head  | 12.83 / 3.39 | 25.7 / 6.18  |


### License/Terms of Use:
The model weights are distributed under a research-friendly non-commercial CC BY-NC 4.0 license

## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## References:
[1] [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece)  <br>