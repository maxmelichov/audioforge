---
license: apache-2.0
---

# Model Overview

### Description:

NeMo Curator Speech Bandwidth Filter (NeMo Curator SBF) is a speech filtering model that filters out high fidelity speech data from low fidelity speech data thereby providing audio data which is only of high fidelity.

This model is ready for commercial use.

### License/Terms of Use:

Use of this model is governed by the NVIDIA Community Model License Agreement (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/). Additional Information: Apache 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

### Deployment Geography:

Global

### Use Case:

NeMo Curator Speech Bandwidth Filter (NeMo Curator SBF) is a speech filtering model that filters out high fidelity speech data from low fidelity speech data thereby providing audio data which is only of high fidelity. It is intended to be used by model developers and audio data curators who want to build datasets for audio model training purposes.

## Model Architecture

**Architecture Type:** Random Forest Classifier

**Network Architecture:** Random Forest Classifier (scikit-learn)

**Number of Model Parameters:** Not Applicable

## Input(s):

**Input Type(s):** Audio

**Input Format(s):** PCM F32

**Input Parameters:** One-Dimensional (1D)

**Other Properties Related to Input:** Pulse Code Modulation (PCM) audio samples with no encoding or pre-processing; 16 kHz or 48 kHz sampling rate required.

## Output(s):

**Output Type(s):** Integer

**Output Format:** Integer (1 or 0)

**Output Parameters:** One-Dimensional (1D)

**Other Properties Related to Output:** Integer label where 1 indicates full-band (high fidelity) and 0 indicates narrow-band (low fidelity).

This model uses a sklearn's Random Forest Classifier and runs entirely on CPU. It does not require GPU hardware or CUDA libraries for training or inference.

## Software Integration

**Runtime Engine(s):**
* NeMo Curator v26.04

**Supported Hardware Microarchitecture Compatibility:**
* x86 CPUs

**Preferred/Supported Operating System(s):**
* Linux
* Windows

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

This AI model can be embedded as an Application Programming Interface (API) call into the software environment described above.

## Model Version(s)

Curator v26.04   

# Training, Testing, and Evaluation Datasets

**Data Modality:**
* Audio

**Audio Training Data Size:**
* Less than 10,000 Hours

**Dataset partition:**

Training [80%], Testing [10%], Validation [10%]

NVIDIA models are trained on a diverse set of public and proprietary datasets. The NeMo Curator Speech Bandwidth Filter model is tested on a dataset that consists of diverse speech dataset.

**Data Collection Method by dataset:** [Hybrid: Human, Synthetic]

**Labeling Method by dataset:** [Hybrid: Human, Synthetic]

**Link:** [DAPS](https://ccrma.stanford.edu/~gautham/Site/daps.html)

**Properties:** The DAPS dataset has 15 versions of audio (3 professional versions and 12 consumer device/real-world environment combinations). Each version consists of about 4.5 hours of data (about 14 minutes from each of 20 speakers).  

**Link:** [LibriTTS](https://www.openslr.org/60/)

**Properties:** LibriTTS is a multi-speaker English corpus of approximately 585 hours of read English speech, which is resampled at 16 kHZ.

**Link:** [VCTK](https://datashare.ed.ac.uk/handle/10283/3443)

**Properties:** This CSTR VCTK Corpus includes speech data uttered by 110 English speakers with various accents. Each speaker reads out about 400 sentences, which were selected from a newspaper, the rainbow passage and an elicitation paragraph used for the speech accent archive.

**Link:** [HiFi-TTS](https://www.openslr.org/109/)

**Properties:** A multi-speaker English dataset for training text-to-speech models. The HiFi-TTS dataset contains about 291.6 hours of speech from 10 speakers with at least 17 hours per speaker sampled at 44.1 kHz.

**Link:** [DNS Challenge 5](https://github.com/microsoft/DNS-Challenge/tree/2db96d5f75257df764a6ef66513b4b97bc707f30)

**Properties:** Collated dataset of clean speech, noise and impulse response provided by Microsoft for the ICASSP 2023 Deep Noise Suppression Challenge.

**Link:** [OpenSLR 32 - High quality TTS data for four South African languages ](https://www.openslr.org/32)

**Properties:** Multi-speaker TTS data for four South African languages, Afrikaans, Sesotho, Setswana and isiXhosa.

# Testing Datasets

**Data Collection Method by dataset:** [Hybrid: Human, Synthetic]

**Labeling Method by dataset:** [Hybrid: Human, Synthetic]

**Properties:** The NeMo Curator Speech Bandwidth Filter model is tested on a dataset that consists of diverse speech dataset. Test data is taken by sampling 10% of training dataset mentioned above. The modality and data type is same as that of the training dataset.

# Evaluation Datasets

**Data Collection Method by dataset:** [Hybrid: Human, Synthetic]

**Labeling Method by dataset:** [Hybrid: Human, Synthetic]

**Properties:** The NeMo Curator Speech Bandwidth Filter model is tested on a dataset that consists of diverse speech dataset. Test data is taken by sampling 10% of training dataset mentioned above. The modality and data type is same as that of the training dataset.

# Inference

**Acceleration Engine:** None

**Test Hardware:**
* x86 CPUs

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

### Bias

|  Field  |  Response  |
|---------|------------|
| Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing: | Age (18+), Gender |
| Measures taken to mitigate against unwanted bias: | Evaluated using internal, proprietary data mix to achieve similar key performance indicators. |

### Explainability

| Field | Response |
|-------|----------|
| Intended Task/Domain:          | Audio Filter |
| Model Type:                    | Speech Bandwidth Filter |
| Intended Users:                | Audio Data Curators, Audio data evaluators  |
| Output:                        | Integer (1 or 0) |
| Describe how the model works:  | This model classifies whether the speech data contains high fidelity audio or low fidelity audio irrespective of the sample rate the speech file is present. |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | Age (18+), Gender |
| Verified to have met prescribed NVIDIA quality standards: | Yes |
| Performance Metrics:                 | Accuracy |
| Technical Limitations & Mitigation: | The model may not work well on a variety of demographic and regional representations of English or with very noisy or very low quality inputs. |
| Potential Known Risks: | The model may present lower accuracy for extremely emotive forms of speech and for noisy speech data. |
| Licensing: | [NVIDIA Community Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/) Additional Information: Apache 2.0 (https://www.apache.org/licenses/LICENSE-2.0). |

### Safety & Security

|  Field  |  Response  |
|---------|------------|
| Model Application(s):                             | Speech Filtering |
| Describe the life critical impact (if present):   | Not Applicable |
| Use Case Restrictions:                            | Abide by [NVIDIA Community Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/). Additional Information: Apache 2.0 (https://www.apache.org/licenses/LICENSE-2.0). |
| Model and dataset restrictions:                   | The Principle of Least Privilege (PoLP) is applied limiting access for dataset generation and model development. Restrictions enforce dataset access during training, and dataset license constraints adhered to. |

### Privacy

| Field | Response |
|-------|----------|
| Generatable or reverse engineerable personal data?  | No |
| Personal data used to create this model?            | Yes |
| Was consent obtained for any personal data used?    | Yes |
| How often is dataset reviewed?                      | Before Release |
| Is a mechanism in place to honor data subject right of access or deletion of personal data? | Yes |
| If personal data was collected for the development of the model, was it collected directly by NVIDIA?     | No |
| If personal data was collected for the development of the model by NVIDIA, do you maintain or have access to disclosures made to data subjects? | Not Applicable |
| If personal data was collected for the development of this AI model, was it minimized to only what was required? | Yes |
| Was data from user interactions with the AI model (e.g. user input and prompts) used to train the model? | No |
| Is there provenance for all datasets used in training?              | Yes |
| Does data labeling (annotation, metadata) comply with privacy laws? | Yes |
| Is data compliant with data subject requests for data correction or removal, if such a request was made? | No, not possible with externally-sourced data. |
| Applicable Privacy Policy | https://www.nvidia.com/en-us/about-nvidia/privacy-policy/ |

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).