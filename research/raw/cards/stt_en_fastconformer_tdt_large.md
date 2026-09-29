---
license: cc-by-4.0
datasets:
- librispeech_asr
- fisher_corpus
- Switchboard-1
- WSJ-0
- WSJ-1
- National-Singapore-Corpus-Part-1
- National-Singapore-Corpus-Part-6
- vctk
- voxpopuli
- europarl
- multilingual_librispeech
- mozilla-foundation/common_voice_8_0
- MLCommons/peoples_speech
language:
- en
pipeline_tag: automatic-speech-recognition
library_name: NeMo
metrics:
- WER
- CER
tags:
- ASR
- English
- Conformer
- Transducer
- TDT
- NeMo
- hf-asr-leaderboard
- speech
- audio
- pytorch
model-index:
- name: stt_en_fastconformer_tdt_large
  results:
    - task:
        name: Automatic Speech Recognition
        type: automatic-speech-recognition
      dataset:
        name: LibriSpeech test clean
        type: librispeech_asr
        split: test-clean
      metrics:
      - name: WER
        type: wer
        value: 1.83
    - task:
        name: Automatic Speech Recognition
        type: automatic-speech-recognition
      dataset:
        name: LibriSpeech test other
        type: librispeech_asr
        split: test-other
      metrics:
      - name: WER
        type: wer
        value: 3.73
    - task:
        name: Automatic Speech Recognition
        type: automatic-speech-recognition
      dataset:
        name: MCV11 test
        type: mozilla-foundation/common_voice_8_0
        split: test
      metrics:
      - name: WER
        type: wer
        value: 6.38
    - task:
        name: Automatic Speech Recognition
        type: automatic-speech-recognition
      dataset:
        name: MLS test
        type: multilingual_librispeech
        split: test
      metrics:
      - name: WER
        type: wer
        value: 4.59
    - task:
        name: Automatic Speech Recognition
        type: automatic-speech-recognition
      dataset:
        name: NSC1 test
        type: National-Singapore-Corpus-Part-1
        split: test
      metrics:
      - name: WER
        type: wer
        value: 4.43
    - task:
        name: Automatic Speech Recognition
        type: automatic-speech-recognition
      dataset:
        name: PS
        type: MLCommons/peoples_speech
        split: test
      metrics:
      - name: WER
        type: wer
        value: 11.34
    - task:
        name: Automatic Speech Recognition
        type: automatic-speech-recognition
      dataset:
        name: Voxpopuli test
        type: voxpopuli
        split: test
      metrics:
      - name: WER
        type: wer
        value: 5.40
    - task:
        name: Automatic Speech Recognition
        type: automatic-speech-recognition
      dataset:
        name: Fisher test
        type: fisher_corpus
        split: test
      metrics:
      - name: WER
        type: wer
        value: 9.71
---

# Nvidia FastConformer TDT Large (en)

STT En FastConformer TDT Large model transcribes speech in lowercase English without punctuation marks. It is a "large" version of FastConformer TDT (around 115M parameters) model. See the section [Model Architecture](#Model-Architecture) and [NeMo documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#fast-conformer) for complete architecture details.

This model is ready for commercial and non-commercial use.

## License

License to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

## References

[1] [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)

[2] [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece)

[3] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)

[4] [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

<!-- ## NVIDIA NeMo: Training

To train, fine-tune or play with the model you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).
We recommend you install it after you've installed the latest Pytorch version.
```
pip install nemo_toolkit['all']
``` 
 -->
## Model Architecture

FastConformer [1] is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling.
The model is trained in a multitask setup with hybrid Transducer decoder (RNNT) and Connectionist Temporal Classification (CTC) loss.
You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).

The model utilizes a [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece) [2] tokenizer with a vocabulary size of 1024.

### Input
  - **Input Type:** Audio
  - **Input Format(s):** .wav files
  - **Input Parameter(s):** Two Dimensional (1D)
  - **Other Properties Related to Input:** 16000 Hz Mono-channel Audio, Pre-Processing Not Needed

### Output

This model provides transcribed speech as a string for a given audio sample.
  - **Output Type**: Text 
  - **Output Format:** String
  - **Output Parameters:** One Dimensional (1D)
  - **Other Properties Related to Output:** May Need Inverse Text Normalization; Does Not Handle Special Characters; Outputs text in lowercase English without punctuation marks

## Limitations

The model is non-streaming and outputs the speech as a string without punctuation and capitalization.
Not recommended for word-for-word transcription and punctuation as accuracy varies based on the characteristics of input audio (unrecognized word, accent, noise, speech type, and context of speech).
Since this model was trained on publicly available speech datasets, the performance of this model might degrade for speech which includes technical terms, or vernacular that the model has not been trained on.

## How to Use this Model

The model is available for use in the NeMo toolkit [3], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.

### Automatically instantiate the model

```python
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecRNNTModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_tdt")
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

```shell
python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py 
 pretrained_name="nvidia/stt_en_fastconformer_tdt" 
 audio_dir="<DIRECTORY CONTAINING AUDIO FILES>"
```

## Training

The [NVIDIA NeMo Toolkit] [3] was used for training the model.
The model is trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py).

The tokenizer for this model was built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

## Training, Testing, and Evaluation Datasets
### Training Datasets
The model is trained on composite dataset comprising of around 24k hours of English speech:
- [Librispeech](https://openslr.elda.org/145/) [960h]
    - Data Collection Method: Human
    - Labeling Method: Human

- [Fisher Corpus](https://catalog.ldc.upenn.edu/LDC2004S13) [1900h]
    - Data Collection Method: Human
    - Labeling Method: Human

- [Switchboard-1 Dataset](https://catalog.ldc.upenn.edu/LDC97S62) [310h]
    - Data Collection Method: Human
    - Labeling Method: Human

- [WSJ-0 and WSJ-1](https://catalog.ldc.upenn.edu/LDC93S6A) [80h]
    - Data Collection Method: Human
    - Labeling Method: Human

- [National Speech Corpus Part 1](https://www.imda.gov.sg/about-imda/emerging-technologies-and-research/artificial-intelligence/national-speech-corpus) [1850h]
    - Data Collection Method: Human
    - Labeling Method: Human

- [National Speech Corpus Part 6](https://www.imda.gov.sg/about-imda/emerging-technologies-and-research/artificial-intelligence/national-speech-corpus) [950h]
    - Data Collection Method: Human
    - Labeling Method: Human

- [VCTK](https://datashare.ed.ac.uk/handle/10283/2950) [80h]
    - Data Collection Method: Human
    - Labeling Method: Human

- [VoxPopuli (EN)](https://github.com/facebookresearch/voxpopuli) [350h]
    - Data Collection Method: Automated
    - Labeling Method: Automated

- [Europarl-ASR EN](https://www.mllp.upv.es/git-pub/ggarces/Europarl-ASR/) [1000h]
    - Data Collection Method: Automated
    - Labeling Method: Automated

- [Multilingual Librispeech (MLS EN)](https://www.openslr.org/94/) [2000h]
    - Data Collection Method: Human
    - Labeling Method: Human

- [Mozilla Common Voice 11.0](https://commonvoice.mozilla.org/en/datasets) [2000h]
    - Data Collection Method: Human
    - Labeling Method: Human

- [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) [12000h] 
    - Data Collection Method: Automated
    - Labeling Method: Automated

### Test Datasets
- [Librispeech](https://openslr.elda.org/145/)
    - Data Collection Method: Human
    - Labeling Method: Human

- [Mozilla Common Voice 11.0](https://commonvoice.mozilla.org/en/datasets)
    - Data Collection Method: Human
    - Labeling Method: Human

- [Multilingual Librispeech (MLS EN)](https://www.openslr.org/94/)
    - Data Collection Method: Human
    - Labeling Method: Human

- [National Speech Corpus Part 1](https://www.imda.gov.sg/about-imda/emerging-technologies-and-research/artificial-intelligence/national-speech-corpus)
    - Data Collection Method: Human
    - Labeling Method: Human

- [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech)
    - Data Collection Method: Automated
    - Labeling Method: Automated

- [VoxPopuli (EN)](https://github.com/facebookresearch/voxpopuli)
    - Data Collection Method: Automated
    - Labeling Method: Automated

- [Fisher Corpus](https://catalog.ldc.upenn.edu/LDC2004S13)
    - Data Collection Method: Human
    - Labeling Method: Human

### Evaluation Datasets
- [Librispeech](https://openslr.elda.org/145/)
    - Data Collection Method: Human
    - Labeling Method: Human

- [Mozilla Common Voice 11.0](https://commonvoice.mozilla.org/en/datasets)
    - Data Collection Method: Human
    - Labeling Method: Human

- [Multilingual Librispeech (MLS EN)](https://www.openslr.org/94/)
    - Data Collection Method: Human
    - Labeling Method: Human

- [National Speech Corpus Part 1](https://www.imda.gov.sg/about-imda/emerging-technologies-and-research/artificial-intelligence/national-speech-corpus)
    - Data Collection Method: Human
    - Labeling Method: Human

- [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech)
    - Data Collection Method: Automated
    - Labeling Method: Automated

- [VoxPopuli (EN)](https://github.com/facebookresearch/voxpopuli)
    - Data Collection Method: Automated
    - Labeling Method: Automated

- [Fisher Corpus](https://catalog.ldc.upenn.edu/LDC2004S13)
    - Data Collection Method: Human
    - Labeling Method: Human

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

## Model Version: 
1.0

## Inference:
**Engine:** N/A
**Test Hardware]:** A100-SXM4-80GB

## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.
When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards below.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Explainability

- High-Level Application and Domain: Automatic Speech Recognition
- Model Type: Speech to text
- Intended Users: developers, researchers, and users interested in transcribing audio
- Output: transcription text for the given audio
- Describe how this model works: The model transcribes audio input into text for the English language
- Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: N/A
-Technical Limitations: 
The model is non-streaming and outputs the speech as a string without punctuation and capitalization.
Not recommended for word-for-word transcription and punctuation as accuracy varies based on the characteristics of input audio (unrecognized word, accent, noise, speech type, and context of speech).
Since this model was trained on publicly available speech datasets, the performance of this model might degrade for speech which includes technical terms, or vernacular that the model has not been trained on.
- Verified to have met prescribed quality standards: Yes
- Performance Metrics: Word Error Rate (WER), Character Error Rate (CER), Real-Time Factor
- Potential Known Risks: Transcripts may not be 100% accurate. Accuracy varies based on the characteristics of input audio (Domain, Use Case, Accent, Noise, Speech Type, Context of speech, etcetera).
- Licensing: CC-BY-4.0
  
### Performance

- **Test Hardware:** A100-SXM4-80GB
- **Batch Size:** 32
- **Precision:** float32
- **Use AMP:** False
- **Matmul Precision:** High

The performance of Automatic Speech Recognition models is measured using Word Error Rate (WER) and Char Error Rate (CER).
Since this dataset is trained on multiple domains, it will generally perform well at transcribing audio in general.

The following tables summarize the performance of the available models in this collection with the Transducer decoder.
Performances of the ASR models are reported in terms of Word Error Rate (WER%) and Inverse Real-Time Factor (RTFx) with greedy decoding on test sets. 

- **Version:** 1.0
- **Tokenizer:** SentencePiece Unigram  
- **Vocabulary Size:** 1024  


| Metric          | LibriSpeech test clean | LibriSpeech test other | MCV11 test | MLS test | NSC1 test | PS    | Voxpopuli test | Fisher test |
|------------------|------------------------|-------------------------|------------|----------|-----------|-------|----------------|-------------|
| WER (%)         | 1.83                   | 3.73                    | 6.38       | 4.59     | 4.43      | 11.34 | 5.40           | 9.71        |
| RTFx            | 1526                   | 1496                    | 1573       | 1841     | 1613      | 820   | 1625           | 1564        |


These are greedy WER numbers without external LM. More details on evaluation can be found at [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) [4].

## Bias
- Was the model trained with a specific accent? No
- Measures taken to mitigate unwanted bias? No
- Participation considerations from adversely impacted groups [protected classes]
(https://www.senate.ca.gov/content/protected-classes) in model design and testing: No

## Privacy 
- Generatable or reverse engineerable personal data? No
- Personal data used to create this model: N/A
- Was consent obtained for any personal data used: N/A
- How often is the training dataset reviewed?: Before Release
- Is a mechanism in place to honor data subject right of access or deletion of personal data? N/A
- If personal data was collected for the development of the model, was it collected directly by NVIDIA? N/A
- If personal data was collected for the development of the model by NVIDIA, do you maintain or have access to disclosures made to data subjects? N/A
- If personal data was collected for the development of this AI model, was it minimized to only what was required? N/A
- Is there dataset provenance? Yes
- Does data labeling (annotation, metadata) comply with privacy laws? Yes
- Is data compliant with data subject requests for data correction or removal, if such a request was made? No, not possible with externally-sourced data

## Safety & Security
- Model Application: Automatic Speech Recognition
- Describe the life critical impact (if present):  N/A
- Use Case Restrictions: Abide by CC-BY-4.0


### Model and dataset restrictions: 
The Principle of Least Privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training and dataset license constraints adhered to.

## NVIDIA Riva: Deployment

[NVIDIA Riva](https://developer.nvidia.com/riva) is an accelerated speech AI SDK deployable on-prem, in all clouds, multi-cloud, hybrid, on edge, and embedded.
Additionally, Riva provides: 

* World-class out-of-the-box accuracy for the most common languages with model checkpoints trained on proprietary data with hundreds of thousands of GPU-compute hours 
* Best in class accuracy with run-time word boosting (e.g., brand and product names) and customization of acoustic model, language model, and inverse text normalization 
* Streaming speech recognition, Kubernetes compatible scaling, and enterprise-grade support 

Although this model isn’t supported yet by Riva, the [list of supported models is here](https://huggingface.co/models?other=Riva).  
Check out [Riva live demo](https://developer.nvidia.com/riva#demos).