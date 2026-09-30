# Architecture explainers we studied, and what the v5 build copies

Ideas only: no footage, frames or assets from any of these are used. Every layer, size and tap point on screen
comes from our code (see the fine print of the build shot).

## 1. Brendan Bycroft, LLM Visualization — https://bbycroft.net/llm (code: https://github.com/bbycroft/llm-viz)
A GPT drawn as a real 3D object: each transformer block is a translucent slab, the blocks stack along one axis,
and the camera walks through them while the tensors being computed light up in place. The picture is the model
itself at its real shape, not a boxes-and-arrows cartoon. Pacing is slow and continuous: one camera move per
idea. **We copy:** the 17 FastConformer blocks as translucent slabs stacked in depth, a slow orbiting camera, and
the data (one 160 ms chunk) visibly moving through the actual stack.

## 2. NVIDIA GTC keynote product animations (e.g. the 2024 Blackwell reveal, https://blogs.nvidia.com/blog/2024-gtc-keynote/)
Hardware is shown as an exploded view: parts hang apart in space with labels, then fly in and lock together to
form the product, and regions glow as data passes. Each part appears on a beat and snaps with an ease-out.
**We copy:** the build: pieces enter from outside the frame, snap onto the stack at their real place, each with a
billboard label (name and size); our heads are the "parts" that lock onto the NVIDIA encoder.

## 3. Apple silicon keynote die shots (M-series launches, https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/)
One chip image; when the presenter names a function, only that region lights and a single big label appears.
Nothing else moves. **We copy:** one component lit at a time while the narrator names it, everything else
dimmed to 35 %, large flat labels, a single accent colour per component.

## 4. 3Blue1Brown, Transformers / Attention in transformers — https://www.3blue1brown.com/lessons/attention/
Vectors flow through blocks as coloured columns; attention is drawn as lines from a token to the tokens it looks
at, thicker for stronger weights; the camera re-frames to whatever is being explained. **We copy:** cache-aware
attention drawn literally: from the newest frame, lines reach back over the 70 cached frames and forward over
the lookahead only, so "streaming" is visible rather than written.

## 5. Jay Alammar, The Illustrated Transformer — https://jalammar.github.io/illustrated-transformer/
Progressive disclosure: first a black box, then the stack of encoders, then what is inside one block; clean
arrows and a small, consistent palette. **We copy:** the order of the build (audio -> features -> subsampling ->
stack -> decoder -> heads) and one colour per role (amber audio, white NVIDIA parts, cyan our heads).

## 6. NVIDIA, Scaling real-time voice agents with cache-aware streaming ASR — https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents
The speech-model explainer for exactly our encoder family: chunks enter, only the new "delta" is computed, and the
cache of past frames is reused. **We copy:** the chunk as the unit of motion (a slab every 160 ms) and the cache
drawn as the frames the attention reaches back to.

## 7. OpenAI, Introducing Whisper — https://openai.com/index/whisper/
The classic speech figure: waveform -> log-Mel spectrogram -> conv stem -> encoder blocks -> decoder emitting
tokens. **We copy:** the front of the pipeline drawn as real signals: the actual waveform of the clip turning into
its actual log-mel spectrogram, scrolling.

## Primary idea chosen
The exploded 3D stack (1 + 2): the encoder assembles from its real parts, then our heads fly in and lock onto
the same stack at their real tap points, lighting as they fire on live values, while one 160 ms chunk rises
through the translucent blocks and the camera orbits slowly; it ends by collapsing into one compact block
beside the four-model chain. Labels stay flat to camera and large. Built with CSS 3D transforms inside the render
page (no external assets). Fine print: "style inspired by exploded-view product animations and bbycroft.net/llm".
