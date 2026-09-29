---
license: cc-by-4.0
datasets:
- mozilla-foundation/common_voice_17_0
- google/fleurs
- MASC
- tarteel-ai/everyayah
language:
- ar
pipeline_tag: automatic-speech-recognition
library_name: nemo
metrics:
- WER
tags:
- speech-recognition
- ASR
- Arabic
- Conformer
- Transducer
- CTC
- NeMo
- hf-asr-leaderboard
- speech
- audio
model-index:
- name: stt_ar_fastconformer_hybrid_large_pc_v2.0
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: MASC
      split: test
      type: masc
      args:
        language: ar
    metrics:
    - name: Test WER
      type: wer
      value: 16.67
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: MCV17
      type: mozilla-foundation/common_voice_17_0
      split: test
      args:
        language: ar
    metrics:
    - name: Test WER
      type: wer
      value: 25.60
  - task:
      type: Automatic Speech Recognition
      name: automatic-speech-recognition
    dataset:
      name: FLEURS
      type: google/fleurs
      split: test
      args:
        language: ar
    metrics:
    - name: Test WER
      type: wer
      value: 12.94
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Tarteel AI's EveryAyah Dataset
      type: tarteel-ai/everyayah
      split: test
      args:
        language: ar
    metrics:
    - name: Test WER
      type: wer
      value: 6.55
---

<!-- nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 Overview -->

# NVIDIA FastConformer-Hybrid PCD Large (ar)

## Description:

This model transcribes speech in Arabic language with punctuation and diacritical marks support.
It is a "large" version of FastConformer Transducer-CTC (around 115M parameters) model and is trained on two losses: Transducer (default) and CTC.
See the  section [Model Architecture](#Model-Architecture) and [NeMo documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#fast-conformer) for complete architecture details.
The model transcribes text in Arabic with diacritical marks and supports periods, Arabic commas and Arabic question marks.

This model is ready for commercial and non-commercial use.

## License
License to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## References

[1] [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)
[2] [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece)
[3] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)
[4] [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

## Model Architecture: <br> 
FastConformer [1] is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling.
The model is trained in a multitask setup with hybrid Transducer decoder (RNNT) and Connectionist Temporal Classification (CTC) loss.
You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).

Model utilizes a [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece) [2] tokenizer with a vocabulary size of 1024.

## Input:
**Input Type(s):** Audio <br>
**Input Format(s):** .wav files <br>
**Input Parameters:** 1D <br>
**Other Properties Related to Input:** 16000 Hz Mono-channel Audio, Pre-Processing Not Needed <br>

## Output: 
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** 1D <br>
**Other Properties Related to Output:** May Need Inverse Text Normalization; Does Not Handle Special Characters; Outputs text in Arabic with diacritical marks <br> 

## Limitations

The model is non-streaming and outputs the speech as a string with diacritical marks.
Not recommended for word-for-word transcription and punctuation as accuracy varies based on the characteristics of input audio (unrecognized word, accent, noise, speech type, and context of speech).
Since this model was trained on publicly available speech datasets, the performance of this model might degrade for speech which includes technical terms, or vernacular that the model has not been trained on.

## How to Use this Model

The model is available for use in the NeMo toolkit [3], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

### Automatically instantiate the model

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0")
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
 pretrained_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

Using CTC mode inference:
```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
 decoder_type="ctc"
``` 

## Training

The [NVIDIA NeMo Toolkit] [3] was used for training the model for two hundred epochs.
Model is trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py).

The tokenizer for these model was built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

## Training, Testing, and Evaluation Datasets
### Training Datasets
The model is trained on composite dataset comprising of around 1100 hours of Arabic speech:

- [Massive Arabic Speech Corpus (MASC)](https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus) [690h] 
    - Data Collection Method: Automated
    - Labeling Method: Automated
- [Mozilla Common Voice 17.0 Arabic](https://commonvoice.mozilla.org/en/datasets) [65h]
    - Data Collection Method: by Human
    - Labeling Method: by Human
- [Google Fleurs Arabic](https://huggingface.co/datasets/google/fleurs) [5h]
    - Data Collection Method: by Human
    - Labeling Method: by Human
- [TarteelAI Everyayah](https://huggingface.co/datasets/tarteel-ai/everyayah) [390h]
    - Data Collection Method: by Human
    - Labeling Method: by Human

### Evaluation Datasets
- [Massive Arabic Speech Corpus (MASC)](https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus)
    - Data Collection Method: Automated
    - Labeling Method: Automated
- [Mozilla Common Voice 17.0 Arabic](https://commonvoice.mozilla.org/en/datasets)
    - Data Collection Method: by Human
    - Labeling Method: by Human
- [Google Fleurs Arabic](https://huggingface.co/datasets/google/fleurs)
    - Data Collection Method: by Human
    - Labeling Method: by Human
- [TarteelAI Everyayah](https://huggingface.co/datasets/tarteel-ai/everyayah)
    - Data Collection Method: by Human
    - Labeling Method: by Human

### Test Datasets
- [Massive Arabic Speech Corpus (MASC)](https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus)
    - Data Collection Method: Automated
    - Labeling Method: Automated
- [Mozilla Common Voice 17.0 Arabic](https://commonvoice.mozilla.org/en/datasets)
    - Data Collection Method: by Human
    - Labeling Method: by Human
- [Google Fleurs Arabic](https://huggingface.co/datasets/google/fleurs)
    - Data Collection Method: by Human
    - Labeling Method: by Human
- [TarteelAI Everyayah](https://huggingface.co/datasets/tarteel-ai/everyayah)
    - Data Collection Method: by Human
    - Labeling Method: by Human

## Software Integration
### Runtime Engine
- Nemo 2.0.0

### Supported Hardware Microarchitecture Compatibility:
- NVIDIA Ampere
- NVIDIA Blackwell
- NVIDIA Jetson
- NVIDIA Hopper
- NVIDIA Lovelace
- NVIDIA Pascal
- NVIDIA Turing
- NVIDIA Volta

### Preferred Operating System
- Linux

## Model Version(s): 
nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 - supports diacritical marks


## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.
When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

<!-- For more detailed information on ethical considerations for this model, please see the [Model Card++](https://docs.google.com/document/d/1cFbfEnlbBG_I5hTRiYuZAI1PgdPYRfsmXpE5-zJDdXU/edit?tab=t.0#heading=h.7jylogfmrbiw) Explainability, Bias, Safety & Security, and Privacy Subcards. -->

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Explainability
- High-Level Application and Domain: Automatic Speech Recognition
- Describe how this model works: The model transcribes audio input into text for the Arabic language
- Verified to have met prescribed quality standards: Yes
- Performance Metrics: Word Error Rate (WER), Character Error Rate (CER), Real-Time Factor
- Potential Known Risks: Transcripts may not be 100% accurate. Accuracy varies based on the characteristics of input audio (Domain, Use Case, Accent, Noise, Speech Type, Context of speech, etcetera).

### Performance

**Test Hardware:**: A5000 GPU
**Batch Size**: 32
**Precision**: float32
**Use AMP**: False
**Matmul Precision**: High

The performance of Automatic Speech Recognition models is measured using Word Error Rate (WER) and Char Error Rate (CER).
Since this dataset is trained on multiple domains, it will generally perform well at transcribing audio in general.

The following tables summarize the performance of the available models in this collection with the Transducer decoder.
Performances of the ASR models are reported in terms of Word Error Rate (WER%) and Inverse Real-Time Factor (RTFx) with greedy decoding on test sets. 

- Transducer
|**Version**|**Tokenizer**|**Vocabulary Size**|**MASC Test WER**|**MASC Test RTFx**|**MCV test WER**|**MCV test RTFx**|**FLEURS test WER**|**FLEURS test RTFx**|**EveryAyah test WER**|**EveryAyah test RTFx**|
|----------|-------------|-------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| 2.0.0  | SentencePiece Unigram | 1024 | 16.67 | 1679 | 25.60 | 1535 | 12.94 | 1392 | 6.55 | 1004 |

- CTC
|**Version**|**Tokenizer**|**Vocabulary Size**|**MASC Test WER**|**MASC Test RTFx**|**MCV test WER**|**MCV test RTFx**|**FLEURS test WER**|**FLEURS test RTFx**|**EveryAyah test WER**|**EveryAyah test RTFx**|
|----------|-------------|-------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| 2.0.0  | SentencePiece Unigram | 1024 | 17.56 | 2116 | 27.42 | 2039 | 14.21 | 1837 | 7.61 | 1688 |

These are greedy WER numbers without external LM. More details on evaluation can be found at [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) [4].

## Bias
- Was the model trained with a specific accent? No
- Have any special measures been taken to mitigate unwanted bias? No
- Participation considerations from adversely impacted groups [protected classes]
(https://www.senate.ca.gov/content/protected-classes) in model design and testing: No

## Privacy 
- Generatable or reverse engineerable personal data? No
- If applicable, was a notice provided to the individuals prior to the collection of any personal data used? Not applicable
- If personal data was collected for the development of the model, was it collected directly by NVIDIA? Not applicable
- Is there dataset provenance? Yes
- If data is labeled, was it reviewed to comply with privacy laws? Yes
- Is data compliant with data subject requests for data correction or removal, if such a request was made? No, not possible with externally-sourced data
- Is a mechanism in place to honor data subject rights of access or deletion of personal data? No 
- How often is the training dataset reviewed?: Before Release

## Safety & Security
### Use Case Restrictions:

- Non-streaming ASR model
- Output text requires Inverse Text Normalization
- Model is noise-sensitive
- Model can have poor performance in dialectal Arabic speech

Model is not applicable for life-critical applications.

### Access Reactions:
The Principle of Least Privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training and dataset license constraints adhered to.

## NVIDIA Riva: Deployment

[NVIDIA Riva](https://developer.nvidia.com/riva) is an accelerated speech AI SDK deployable on-prem, in all clouds, multi-cloud, hybrid, on edge, and embedded.
Additionally, Riva provides: 

* World-class out-of-the-box accuracy for the most common languages with model checkpoints trained on proprietary data with hundreds of thousands of GPU-compute hours 
* Best in class accuracy with run-time word boosting (e.g., brand and product names) and customization of acoustic model, language model, and inverse text normalization 
* Streaming speech recognition, Kubernetes compatible scaling, and enterprise-grade support 

Although this model isn’t supported yet by Riva, the [list of supported models is here](https://huggingface.co/models?other=Riva).  
Check out [Riva live demo](https://developer.nvidia.com/riva#demos). 
