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
- chain-of-thought
- audio understanding
- ASR
- audio captioning
- long-context
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
    <img src="static/afnext_logo.webp" alt="Audio Flamingo Next Think" width="120">
  </a>
</div>
<div align="center" style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <h2>
    Audio Flamingo Next Think: Temporally Grounded Audio Reasoning for Speech, Sound, and Music
  </h2>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://afnext-umd-nvidia.github.io/"><img src="https://img.shields.io/badge/Project%20Page-AF--Next-0F766E" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2604.10905"><img src="https://img.shields.io/badge/arXiv-2604.10905-B31B1B.svg" alt="arXiv"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo"><img src="https://img.shields.io/badge/GitHub-AF--Next-0E7490" alt="GitHub"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social" alt="GitHub stars"></a>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/nvidia/audio-flamingo-next-think-hf">
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
  <a href="https://huggingface.co/spaces/nvidia/audio-flamingo-next-think">
    <img src="https://img.shields.io/badge/🤗-Gradio%20Demo-5F9EA0.svg" alt="Gradio Demo">
  </a>
</div>

`nvidia/audio-flamingo-next-think-hf` is the temporally grounded reasoning checkpoint in the Audio Flamingo Next family. It is designed for harder long-audio question answering problems where the model needs to combine evidence across multiple events, speakers, or timestamps before answering.

## Description

Audio Flamingo Next (AF-Next) is the next-generation open audio-language model in the Audio Flamingo series, built for speech, environmental sound, and music understanding with audio inputs up to `30` minutes.

This checkpoint corresponds to **AF-Next-Think**, the reasoning-specialized variant. Starting from AF-Next-Instruct, the model is further trained on **AF-Think-Time**, a temporally grounded chain-of-thought dataset built from long and complex audio such as trailers, movie recaps, mystery stories, and long multi-party conversations.

AF-Next-Think is the right checkpoint if you want:

- deliberate long-form reasoning
- temporal evidence aggregation over long audio
- multi-step question answering
- prompts that ask for timestamp-grounded explanations

AF-Next-Think may emit reasoning traces enclosed in `<think> ... </think>` before giving the final answer. If you only need concise answers or assistant-style behavior, start with `nvidia/audio-flamingo-next-hf`.

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
- Use a larger `max_new_tokens` budget than the instruct model because reasoning traces can be long.
- Prompting matters: this checkpoint is strongest when you explicitly request step-by-step, timestamp-grounded reasoning and then a final answer.

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

### Temporal Reasoning Example

```python
import torch
from transformers import AutoModel, AutoProcessor

model_id = "nvidia/audio-flamingo-next-think-hf"

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
                        "Reason step by step with timestamps before answering. "
                        "How does the female speaker's tone change over the course "
                        "of the audio, and what evidence supports that?"
                    ),
                },
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/000000786159.31.wav",
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
    max_new_tokens=4096,
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

### Practical Prompting Tip

AF-Next-Think works best when the prompt makes the reasoning requirement explicit, for example:

- "Reason step by step with timestamps, then give the final answer."
- "Ground your explanation in moments from the audio."
- "Identify the relevant events first, then answer."

## Training Summary

AF-Next-Think inherits the full AF-Next training recipe and then adds a dedicated reasoning stage:

- AF-Next base training scales beyond academic benchmarks using public and internet-scale audio
- the model is trained on speech, sound, and music data, including AudioSkills-XL, LongAudio-XL, AF-Think, AF-Chat, and MF-Skills
- AF-Next introduces long-form internet audio, multi-talker speech understanding, multilingual ASR/AST, multi-audio data, safety data, and instruction-following data
- the reasoning stage uses **AF-Think-Time**, a temporally grounded reasoning dataset with `43K` question-answer-thinking-chain examples
- AF-Think-Time examples average `446.3` words of reasoning and are designed for long, complex audio
- the full AF-Next system is trained on `128` NVIDIA H100 GPUs

The paper describes AF-Next-Think as: AF-Next-Instruct followed by supervised fine-tuning on AF-Think-Time and then RL with the same post-training mixture.

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

From the AF-Next paper, the reasoning-specialized variant improves on harder reasoning benchmarks:

- `MMAU v05.15.25 average`: `75.01` for `+Think` vs `74.20` for AF-Next-Instruct
- `MMAU-Pro`: `58.7` for `+Think` vs `56.9` for AF-Next-Instruct
- `MMAR`: `61.0` for `+Think` vs `59.7` for AF-Next-Instruct
- `MMSU`: `61.2` for `+Think` vs `59.4` for AF-Next-Instruct

These gains are consistent with the intended use of AF-Next-Think: harder multi-step reasoning rather than the shortest or most concise answer style.

## Limitations

The paper highlights several limitations:

- internet-scale audio remains noisy and unevenly distributed across domains and languages
- long-context reasoning still becomes difficult when evidence is sparse or distributed far apart in time
- evaluation does not yet fully cover all AF-Next capabilities, including diarization, timestamped captioning, and voice-to-voice interaction
- reasoning traces can be verbose, so downstream systems may need to post-process or strip `<think>` blocks

If you do not need explicit reasoning traces, `nvidia/audio-flamingo-next-hf` is usually the better default. If you want dense descriptive outputs instead of explicit reasoning, use `nvidia/audio-flamingo-next-captioner-hf`.

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
