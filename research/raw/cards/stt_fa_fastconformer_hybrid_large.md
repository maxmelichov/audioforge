---
language:
- fa
library_name: nemo
datasets:
- Mozilla-CommonVoice-15.0-Persian
thumbnail: null
tags:
- automatic-speech-recognition
- speech
- audio
- CTC
- Transducer
- FastConformer
- Transformer
- pytorch
- NeMo
license: cc-by-4.0
model-index:
- name: stt_fa_fastconformer_hybrid_large
  results:
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: Mozilla Common Voice 15.0 Persian
      type: mozilla-foundation/common_voice_15_0
      config: fa
      split: test (custom)
      args:
        language: fa
    metrics:
    - name: Test (custom) WER CTC
      type: wer
      value: 13.16
    - name: Test (custom) CER CTC
      type: cer
      value: 3.85
    - name: Test (custom) WER RNNT
      type: wer
      value: 15.48
    - name: Test (custom) CER RNNT
      type: cer
      value: 4.63
---

# NVIDIA FastConformer-Hybrid Large (fa)

<style>
img {
 display: inline;
}
</style>

| [![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--Transducer_CTC-lightgrey#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-115M-lightgrey#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-fa-lightgrey#model-badge)](#datasets)


This model transcribes speech in Persian alphabet.
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
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_fa_fastconformer_hybrid_large")
```

### Transcribing using Python
Having instantiated the model, simply do:
```
output = asr_model.transcribe(['sample.wav'])
print(output[0].text)
```

### Transcribing many audio files

Using Transducer mode inference:
```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/stt_fa_fastconformer_hybrid_large" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

Using CTC mode inference:
```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/stt_fa_fastconformer_hybrid_large" 
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

The NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/speech_to_text_finetune.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/asr_finetune/speech_to_text_finetune.yaml).

The tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

This model was initialized with the weights of [English FastConformer Hybrid (Transducer and CTC) Large P&C model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_pc) and fine-tuned to Persian data.

### Datasets

This model was trained on Mozilla CommonVoice Persian Corpus 15.0.

In order to leverage the entire validated data portion, the standard train/dev/test splits were discarded and replaced with custom splits. The custom splits may be reproduced by:
- grouping utterances with identical transcript and sorting utterances (ascendingly) by the (transcript occupancy, transcript) pairs;
- selecting the first 10540 utterances for the test set (to maintain the original size);
- selecting the second 10540 utterances for the dev set;
- selecting the remaining data for the training set.
- 
The transcripts were additionally normalized according to the following script (empty results were discarded):

```python
import unicodedata
import string

SKIP = set(
    list(string.ascii_letters)
    + [
        "=",  # occurs only 2x in utterance (transl.): "twenty = xx"
        "ā",  # occurs only 4x together with "š"
        "š",
        # Arabic letters
        "ة",  # TEH MARBUTA
    ]
)

DISCARD = [
    # "(laughter)" in Farsi
    "(خنده)",
    # ASCII
    "!",
    '"',
    "#",
    "&",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    ":",
    ";",
    # Unicode punctuation?
    "–",
    "“",
    "”",
    "…",
    "؟",
    "،",
    "؛",
    "ـ",
    # Unicode whitespace?
    "ً",
    "ٌ",
    "َ",
    "ُ",
    "ِ",
    "ّ",
    "ْ",
    "ٔ",
    # Other
    "«",
    "»",
]

REPLACEMENTS = {
    "أ": "ا",
    "ۀ": "ە",
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ﯽ": "ی",
    "ﻮ": "و",
    "ے": "ی",
    "ﺒ": "ب",
    "ﻢ": "ﻡ",
    "٬": " ",
    "ە": "ه",
}


def maybe_normalize(text: str) -> str | None:

    # Skip selected with banned characters
    if set(text) & SKIP:
        return None  # skip this

    # Remove hashtags - they are not being read in Farsi CV
    text = " ".join(w for w in text.split() if not w.startswith("#"))

    # Replace selected characters with others
    for lhs, rhs in REPLACEMENTS.items():
        text = text.replace(lhs, rhs)

    # Replace selected characters with empty strings
    for tok in DISCARD:
        text = text.replace(tok, "")

    # Unify the symbols that have the same meaning but different Unicode representation.
    text = unicodedata.normalize("NFKC", text)

    # Remove hamza's that were not merged with any letter by NFKC.
    text = text.replace("ء", "")

    # Remove double whitespace etc.
    return " ".join(t for t in text.split() if t)
```

## Performance

 The performance of Automatic Speech Recognition models is measuring using Character Error Rate (CER) and Word Error Rate (WER).

 The model obtains the following scores on our custom dev and test splits of Mozilla CommonVoice Persian 15.0:

 | Model     | %WER/CER dev | %WER/CER test |
 |-----------|--------------|---------------|
 | RNNT head | 15.44 / 3.89 | 15.48 / 4.63  |
 | CTC head  | 13.18 / 3.38 | 13.16 / 3.85  |
 
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
[1] [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)

[2] [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece)

[3] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

## Licence

License to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.