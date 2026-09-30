---
license: other
language:
- en
tags:
- audio
- speech
- sound
- music
- reasoning
- audio understanding
- ASR
- audio captioning
- long-context
- audio-language-model
- long-audio
- timestamp-grounding
- instruction-tuned
datasets:
- nvidia/LongAudio
- nvidia/AF-Skills
- nvidia/AF-Chat
- nvidia/AF-Think
pipeline_tag: audio-text-to-text
library_name: transformers
---
# Model Overview

<div align="center" style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <a href="https://afnext-umd-nvidia.github.io/" style="text-decoration: none; display: flex; align-items: center;">
    <img src="static/afnext_logo.webp" alt="Audio Flamingo Next" width="120">
  </a>
</div>
<div align="center" style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <h2>
    Audio Flamingo Next: Next-Generation Open Audio-Language Models for Speech, Sound, and Music
  </h2>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://afnext-umd-nvidia.github.io/"><img src="https://img.shields.io/badge/Project%20Page-AF--Next-0F766E" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2604.10905"><img src="https://img.shields.io/badge/arXiv-2604.10905-B31B1B.svg" alt="arXiv"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo"><img src="https://img.shields.io/badge/GitHub-AF--Next-0E7490" alt="GitHub"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social" alt="GitHub stars"></a>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/nvidia/audio-flamingo-next-hf">
    <img src="https://img.shields.io/badge/🤗-Checkpoint-ED5A22.svg" alt="Checkpoint">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/LongAudio">
    <img src="https://img.shields.io/badge/🤗-Dataset:%20LongAudio-ED5A22.svg" alt="LongAudio dataset">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Skills">
    <img src="https://img.shields.io/badge/🤗-Dataset:%20AF--Skills-ED5A22.svg" alt="AF-Skills dataset">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Chat">
    <img src="https://img.shields.io/badge/🤗-Dataset:%20AF--Chat-ED5A22.svg" alt="AF-Chat dataset">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Think">
    <img src="https://img.shields.io/badge/🤗-Dataset:%20AF--Think-ED5A22.svg" alt="AF-Think dataset">
  </a>
  <a href="https://huggingface.co/spaces/nvidia/audio-flamingo-next">
    <img src="https://img.shields.io/badge/🤗-Gradio%20Demo-5F9EA0.svg" alt="Gradio Demo">
  </a>
</div>

Audio Flamingo Next (AF-Next) is the next-generation open large audio-language model in the Audio Flamingo series. `nvidia/audio-flamingo-next-hf` is the instruction-tuned checkpoint for general audio understanding, question answering, and conversation over speech, environmental sounds, and music.

## Description

Compared with Audio Flamingo 3, AF-Next adds:

- a stronger foundational audio-language model for speech, sound, and music
- training data scaled beyond academic benchmarks using public and internet-scale sources
- native long-audio support up to 30 minutes
- stronger multilingual ASR, multi-talker speech understanding, and long-form captioning
- timestamp-aware modeling through Rotary Time Embeddings (RoTE)

This checkpoint corresponds to **AF-Next-Instruct**, the post-trained assistant variant. It is the best default checkpoint if you want:

- general audio QA
- instruction following
- multi-turn audio chat
- long-form audio understanding
- timestamp-aware prompts

## Best For

- standard audio QA and instruction following across speech, sound, and music
- assistant-style responses for long-audio questions, follow-up questions, and multi-turn chat
- speech understanding tasks such as ASR, paralinguistic understanding, and multilingual AST / speech translation
- music captioning and broad audio description when you want a direct answer instead of a dense long-form caption

## AF-Next Variants

| Checkpoint | Use when you need |
| --- | --- |
| `nvidia/audio-flamingo-next-hf` | default QA, chat, ASR / AST, and direct assistant-style answers |
| `nvidia/audio-flamingo-next-think-hf` | explicit multi-step reasoning, timestamp-grounded evidence, and longer reasoning traces |
| `nvidia/audio-flamingo-next-captioner-hf` | dense long-form captions, timestamped scene breakdowns, and more descriptive outputs |

These Hub weights are released as an `audio-text-to-text` model. The broader AF-Next project also discusses streaming TTS and voice-to-voice interaction, but those components are not part of this checkpoint.

**This model is for non-commercial research purposes only.**

## Usage

### Install

AF-Next is supported in Transformers:

```bash
pip install --upgrade pip
pip install --upgrade transformers accelerate
```

### Notes

- The processor expects mono `16 kHz` audio.
- Audio is internally processed in `30`-second windows.
- The released processor is configured for up to `1800` seconds of audio, i.e. `30` minutes.
- Prompting matters: this checkpoint is strongest when the task and output format are explicit. Ask directly for QA, ASR, AST, timestamps, or speaker labels instead of using a generic prompt.

### Prompt Guide

| Task | Prompt | Recommended Checkpoint(s) |
| --- | --- | --- |
| ASR | `Transcribe the input speech.` | `Instruct`, `Think` |
| AST | `Translate any speech you hear from <src_lang> into <tgt_lang>.` | `Instruct`, `Think` |
| Short Audio Captioning | `Generate a caption for the input audio.` | `Captioner`, `Think` |
| Long Audio Captioning | `Generate a detailed caption for the input audio. In the caption, transcribe all spoken content by all speakers in the audio precisely.` | `Captioner`, `Think` |
| Music Captioning | `Summarize the track with precision: mention its musical style, BPM, key, arrangement, production choices, and the emotions or story it conveys.` | `Captioner`, `Instruct`, `Think` |
| Lyrics | `Generate a lyrics transcription from the input song.` | `Instruct`, `Captioner`, `Think` |
| QA | `What precise description did the commentator use for the punch that ended the fight?` | `Instruct`, `Think` |
| Timestamped Multi-Talker ASR | `Transcribe the input audio. If multiple speakers are present, provide diarized transcripts with speaker labels.`<br>`[Speaker 1] ...`<br>`[Speaker 2] ...` | `Instruct`, `Think` |

### Single-Turn Audio + Text

```python
import torch
from transformers import AutoModel, AutoProcessor

model_id = "nvidia/audio-flamingo-next-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

conversation = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Transcribe the speech, identify important background sounds, "
                        "and mention approximate timestamps for key events."
                    ),
                },
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/videoplayback_superman.wav",
                },
            ],
        }
    ]
]

batch = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

if "input_features" in batch:
    batch["input_features"] = batch["input_features"].to(model.dtype)

generated = model.generate(
    **batch,
    max_new_tokens=1024,
    repetition_penalty=1.2,
)

prompt_len = batch["input_ids"].shape[1]
completion = generated[:, prompt_len:]
text = processor.batch_decode(
    completion,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print(text)
```

### Multi-Turn Follow-Up

```python
conversation = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Transcribe the input speech, then keep the conversation open "
                        "for follow-up questions."
                    ),
                },
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Summer follows spring, the days grow longer, and the nights are warm.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Summarize the content in one sentence.",
                }
            ],
        },
    ]
]
```

## Training Summary

AF-Next is trained with a four-stage curriculum spanning pre-training, mid-training, post-training, and temporally grounded reasoning training. The paper describes:

- AF-Whisper-based audio modeling with broader multilingual and multi-talker speech coverage
- expanded training data from AudioSkills-XL, LongAudio-XL, AF-Think, AF-Chat, and MF-Skills
- `45K` additional multi-talker speech samples
- `200K+` long-form internet videos spanning roughly `5` to `30` minutes
- `2M+` real-world short audio skill samples mined from long-form audio
- `1M` multi-audio instruction examples
- `30K` multi-turn chat examples
- `386K` safety and instruction-following examples
- multilingual ASR and AST data from Emilia, CoVoST, MUST, Amazon-SIFT, ALI Meeting, aidatatang, AISHELL, and Granary
- training on `128` NVIDIA H100 GPUs

AF-Next-Instruct is obtained after GRPO-based post-training focused on multi-turn chat, safety, instruction following, and selected AudioSkills-XL skills.

## Architecture

The released checkpoint exposes `AudioFlamingoNextForConditionalGeneration` with `AudioFlamingoNextProcessor`. At a high level, AF-Next combines:

- an AF-Whisper audio encoder using `128`-bin log-mel features
- non-overlapping `30`-second audio chunking
- a `2`-layer MLP audio adaptor
- a Qwen2.5-family text backbone extended to long context
- RoTE for timestamp-aware temporal grounding

The released config uses:

- `audio_config.hidden_size = 1280`
- `audio_config.num_hidden_layers = 32`
- `text_config.hidden_size = 3584`
- `text_config.num_hidden_layers = 28`
- `text_config.max_position_embeddings = 131072`


## Limitations

The paper highlights several limitations:

- internet-scale audio is still noisy and uneven across domains, languages, and acoustic conditions
- low-resource languages, rare sound events, and specialized domains remain underrepresented
- long-context reasoning is still difficult when relevant evidence is sparse or far apart in time
- evaluation does not yet fully cover all supported capabilities, including multi-talker ASR, diarization, timestamped captioning, and voice-to-voice interaction

For most users, this is the best AF-Next checkpoint to start with. If you need explicit long-form reasoning traces, use `nvidia/audio-flamingo-next-think-hf`. If you want the most verbose descriptive captions, use `nvidia/audio-flamingo-next-captioner-hf`.

## License / Terms of Use

The model is released under the [NVIDIA OneWay Noncommercial License](static/NVIDIA-OneWay-Noncommercial-License_22Mar2022-academic.docx). Portions of the dataset generation are also subject to the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE) and OpenAI's [Terms of Use](https://openai.com/policies/terms-of-use).

## Citation

```bibtex
@misc{ghosh2026audioflamingonext,
  title={Audio Flamingo Next: Next-Generation Open Audio-Language Models for Speech, Sound, and Music},
  author={Sreyan Ghosh and Arushi Goel and Kaousheik Jayakumar and Lasha Koroshinadze and Nishit Anand and Zhifeng Kong and Siddharth Gururani and Sang-gil Lee and Jaehyeon Kim and Aya Aljafari and Chao-Han Huck Yang and Sungwon Kim and Ramani Duraiswami and Dinesh Manocha and Mohammad Shoeybi and Bryan Catanzaro and Ming-Yu Liu and Wei Ping},
  year={2026},
  howpublished={Technical report},
  url={https://afnext-umd-nvidia.github.io/}
}
```
