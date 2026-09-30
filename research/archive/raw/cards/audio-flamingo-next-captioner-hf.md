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
- audio captioning
- long-context
- long-form-captioning
- audio-language-model
- long-audio
- timestamp-grounding
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
    <img src="static/afnext_logo.webp" alt="Audio Flamingo Next Captioner" width="120">
  </a>
</div>
<div align="center" style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <h2>
    Audio Flamingo Next Captioner: Long-Form Audio Captioning for Speech, Sound, and Music
  </h2>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://afnext-umd-nvidia.github.io/"><img src="https://img.shields.io/badge/Project%20Page-AF--Next-0F766E" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2604.10905"><img src="https://img.shields.io/badge/arXiv-2604.10905-B31B1B.svg" alt="arXiv"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo"><img src="https://img.shields.io/badge/GitHub-AF--Next-0E7490" alt="GitHub"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social" alt="GitHub stars"></a>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/nvidia/audio-flamingo-next-captioner-hf">
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
  <a href="https://huggingface.co/spaces/nvidia/audio-flamingo-next-captioner">
    <img src="https://img.shields.io/badge/🤗-Gradio%20Demo-5F9EA0.svg" alt="Gradio Demo">
  </a>
</div>

`nvidia/audio-flamingo-next-captioner-hf` is the long-form captioning checkpoint in the Audio Flamingo Next family. It is designed for rich, descriptive outputs over long and complex audio, including speech-heavy recordings, environmental sound scenes, and music.

## Description

Audio Flamingo Next (AF-Next) is the next-generation open audio-language model in the Audio Flamingo series, built for speech, environmental sound, and music understanding with audio inputs up to `30` minutes.

This checkpoint corresponds to **AF-Next-Captioner**, the model obtained at the end of AF-Next mid-training after the training mixture is expanded with newly collected long-audio captioning and QA datasets. It is the best AF-Next variant when you want:

- long-form descriptive captions
- verbose summaries of long audio
- timestamp-aware scene breakdowns
- exhaustive descriptions of events, speakers, and background audio

## Best For

- detailed audio captions that integrate speech, sound effects, ambience, and music in one response
- prompts that ask for a broad caption while also transcribing spoken content by all speakers precisely
- long-form summaries, timestamp-aware event descriptions, and dense scene breakdowns
- showcase-style qualitative outputs where you want the model to describe how the audio evolves over time

## AF-Next Variants

| Checkpoint | Use when you need |
| --- | --- |
| `nvidia/audio-flamingo-next-hf` | default QA, chat, ASR / AST, and direct assistant-style answers |
| `nvidia/audio-flamingo-next-think-hf` | explicit multi-step reasoning, timestamp-grounded evidence, and longer reasoning traces |
| `nvidia/audio-flamingo-next-captioner-hf` | dense long-form captions, timestamped scene breakdowns, and more descriptive outputs |

Because this checkpoint comes before the RL-based assistant alignment stage, it is often more verbose and more caption-like than `nvidia/audio-flamingo-next-hf`. If you want chat-oriented or safer assistant-style behavior, use the instruct checkpoint instead.

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
- This checkpoint is best used with prompts that ask for descriptive, coherent, or timestamped captions.
- Prompting matters: this checkpoint is strongest when you ask explicitly for a dense caption, timestamped scene summary, lyrics, or a speaker-aware breakdown.

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

### Long-Form Captioning Example

```python
import torch
from transformers import AutoModel, AutoProcessor

model_id = "nvidia/audio-flamingo-next-captioner-hf"

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
                        "Write a detailed caption of this audio. Cover the speakers, "
                        "background sounds, major events, and how the scene changes over time."
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
    max_new_tokens=2048,
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

### Prompting Tip

AF-Next-Captioner responds best to prompts like:

- "Write a rich caption of the full audio."
- "Describe how the audio scene evolves over time."
- "Give a timestamped summary with speaker and sound-event details."

## Training Summary

AF-Next-Captioner is the checkpoint produced after AF-Next mid-training. In that stage, the model:

- retains the earlier AF-Next recognition and skill data mixture
- adds newly collected long-audio captioning and QA data
- increases the maximum audio length to `30` minutes
- increases the total context length to `128K`
- emphasizes long-form data by down-sampling the earlier stage mixture and assigning full blend weight to long-audio datasets

The broader AF-Next training recipe also includes:

- AudioSkills-XL, LongAudio-XL, AF-Think, AF-Chat, and MF-Skills
- `45K` additional multi-talker speech samples
- `200K+` long-form internet videos
- `2M+` real-world short audio skill samples
- `1M` multi-audio instruction examples
- `30K` multi-turn chat examples
- `386K` safety and instruction-following examples
- multilingual ASR and AST data from Emilia, CoVoST, MUST, Amazon-SIFT, ALI Meeting, aidatatang, AISHELL, and Granary
- training on `128` NVIDIA H100 GPUs

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

## Selected Results

From the AF-Next paper, the captioner-style variant is especially strong on broad open-ended benchmarks:

- `MMAU v05.15.25 average`: `75.76` for `+Captioner`
- `MMAR`: `63.0` for `+Captioner`
- `MMSU`: `63.3` for `+Captioner`

The paper positions this variant as the most caption-oriented AF-Next checkpoint, particularly for long-form descriptive prompting.

## Limitations

The paper highlights several limitations:

- internet-scale audio remains noisy and unevenly distributed across domains and languages
- low-resource languages, rare events, and specialized domains are still underrepresented
- very long-context reasoning remains challenging when evidence is distant or sparse
- evaluation does not yet fully cover all AF-Next capabilities, including diarization, timestamped captioning, and voice-to-voice interaction

If you want better assistant alignment, use `nvidia/audio-flamingo-next-hf`. If you want explicit reasoning traces, use `nvidia/audio-flamingo-next-think-hf`.

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
