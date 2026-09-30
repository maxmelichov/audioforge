# LinkedIn post (v5)

Video: `showcase_v5.mp4` (16:9) or `showcase_v5_square.mp4` (1:1) from `/Volumes/ExternalSSD/nvidia-audio-models/demo_out/`.
Every number below is the number on screen in the film, read from `runs/*.json` (`demo/archive/v5/export.py` writes each value
with its JSON path into `v5_data/data.json`), except the "words while you speak" line: the film's "first words
after 1.2 s" counted from the start of speech and was replaced here by the measured partial latency
(`runs/stt_latency.json`, research/METRICS.md). Claims policy: only results measured better than the default Pipecat /
LiveKit stacks, or capabilities they lack; one number per system; no jargon in the post body.

## Post text

Your voice agent talks over people. It shows no words until they stop. And in a room, it has no idea who is speaking.

I built audioforge to fix the listening side. Three small heads sit on NVIDIA's frozen streaming speech model and answer three questions as you talk: is someone speaking, who is it, and is the turn over. NVIDIA's speaker model tracks the room, and NVIDIA Parakeet rewrites each finished sentence. It drops into Pipecat and LiveKit.

Measured live, through the real Pipecat and LiveKit pipelines, on real calls (the caller's own microphone):

- Answers when you finish: on real phone calls (the caller's own channel), 6 % of turns get no answer within 3 seconds, against 13 % for LiveKit's default and 43 % for Pipecat's.
- Interrupts less: 1.3 false interruptions a minute against 2.2 for Pipecat's default.
- Words while you speak: each word appears in the transcript about 0.44 s after it is said (median); the default stacks' Whisper shows nothing until the sentence is over.

On benchmarks:

- Meetings it never saw: it catches 21 % of turn ends against 13 % for a silence timeout, and cuts people off half as often.
- Two-person calls (TurnBench): it catches 85 % of turn ends at the same false-alarm rate, against 75-81 % for the detectors in voice-agent frameworks.
- Final transcript: 9.5 % word errors with NVIDIA Parakeet in our stack, against 14.4 % for Whisper small, the default stacks' model.
- Knows who is talking: 4 of 4 people labelled in a real meeting; the default stacks have no speaker labels.

Every example in the video is a real recording through the real pipeline; the agent's voice was added for the demo at the recorded cut-in and response times, and the subtitles show what the caller says. Code and benchmarks are open: github.com/maxmelichov/audioforge

Voice-over synthesised with Qwen3-TTS 1.7B.

## Images (attach with the post or as the first comment)

- `demo/archive/images/architecture.png` (16:9) or `architecture_square.png` (1:1): the model figure, three small heads on NVIDIA's frozen streaming speech model.
- `demo/archive/images/results.png` (16:9) or `results_square.png` (1:1): the six results as ranked bars, audioforge in orange, the default stacks in grey.

Why they look the way they do, and the published figures they follow: `demo/images/DESIGN_NOTES.md`. Sources and the alternate layouts (A2 stacked figure, B2 stacked list): `demo/images/redesign/`.

Alt text (architecture): A white diagram. Audio goes into a dark block labelled "NVIDIA streaming speech model, 17 layers, frozen", drawn as 17 slabs with layer 4 in orange; words come out on the right. Three orange-outlined cards are the small heads: "Is someone speaking? yes" reads all 17 layers, "Who is speaking? Person 2" reads layer 4, "Is the turn over? not yet" reads a second pass told who is speaking. Nemotron-3 (who is in the room) and Parakeet (rewrites each finished sentence) sit alongside. A bottom row shows today's stack: Silero VAD, turn detector, Whisper, one model per question.

Alt text (results): Six small bar charts on white, audioforge in orange and the other systems in grey. Speech detection 94.9 vs 93.7 (NVIDIA MarbleNet) and 91.5 (Silero); missed turn ends with a 5 s voice print 62 to 39 % (AMI) and 69 to 20 % (ICSI); word errors 9.5 vs 14.4 % (Whisper small); turn ends caught 85 vs 81 (Silero timeout) and 75 % (smart-turn); talks over you 1.3 vs 2.2 times a minute (Pipecat default); no reply within 3 s 6 vs 13 (LiveKit) and 43 % (Pipecat).

## Hashtag sets (pick one)

A (broad): #VoiceAI #SpeechRecognition #ConversationalAI #OpenSource
B (technical): #ASR #TurnTaking #Diarization #FastConformer #NVIDIA
C (product): #VoiceAgents #Pipecat #LiveKit #RealTimeAI

## Alt text (video)

A navy-to-indigo film. It opens on the default voice-agent pipeline drawn as six tiles (mic, Silero VAD, turn detector, Whisper, LLM, TTS), then a phone-style call UI plays a real meeting through Pipecat's default stack: the agent's reply bubble slides in with a red "Interrupted" tag while the speaker is still talking. A flat diagram builds the system one idea at a time: NVIDIA's streaming speech model turning voice into words, three small heads answering "Is someone speaking?", "Who is speaking?" and "Is the turn over?" with live answers, then NVIDIA's speaker model and Parakeet. Split screens replay real recordings with their audio and subtitles: Pipecat's default interrupts a caller twice while audioforge waits and answers; LiveKit's default shows no words for nearly 8 seconds while audioforge shows them after about one; a real phone call with four interruptions against none; a four-person meeting where audioforge labels each voice. Charts show one number per system for each result, then a seven-row scorecard and the end card. A warm male narrator explains each shot; captions are burned in.
