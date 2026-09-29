# Narration v2, full cut (about 4 min). Presentation register: first person plural, present tense.
# One bullet = one synthesized unit = one caption. `## scene:` lines switch the visual. Every spoken number is on
# screen at that moment with its runs/*.json source in the corner (demo/numbers.json).

## scene: title
- Today we are presenting audioforge: one streaming model that does a voice agent's listening. It transcribes, detects speech, follows who is talking, and decides when your turn is over, on two laptop CPU threads.

## scene: stacks
- This is what exists today. Pipecat's and LiveKit's default stacks chain three separate models: a voice activity detector, a turn detector, and Whisper for the words. Each adds its own delay, and none of them knows who is speaking.
- Ours is one frozen NVIDIA streaming encoder with small heads, and NVIDIA's diarizer beside it. One pass, every 160 milliseconds.

## scene: ex_cutin
- Look at this example. The same meeting audio goes into both. On the left, Pipecat's default stack. On the right, ours, inside the same Pipecat pipeline.
- The user is still mid-sentence, and the default stack answers four times. Ours waits, and answers 1.7 seconds after she stops.

## scene: ex_deadair
- Second example: dead air. The user stops here. Both sides are now waiting. The default stack answers after 3.2 seconds. Ours after 1.6.

## scene: ex_partial
- Third: the words. Our first partial transcript arrives 0.3 seconds after she starts speaking. LiveKit's default stack shows its first text after 9 seconds, when Whisper gets the segment.

## scene: multispeaker
- Now a meeting with people talking over each other. The bands show who holds the floor. We bind the user to the first voice after the agent's own TTS ends, and the turn decision follows that speaker, not just silence.

## scene: voiceprint
- And the newest piece. With a five-second voice print, a target-speaker head tracks the user directly. Same clip, without the print on the left: it cuts in once. With the print on the right: no cut-in, and it answers earlier.
- Offline, on 974 AMI turn ends, missed ends fall from 61.9 to 39.3 percent, and on held-out ICSI from 68.5 to 20.4. We served this path last week; the live numbers are still being collected.

## scene: numbers
- Across all 974 turn ends, at the same false-cutoff budget, we miss 61.9 percent within six seconds. Smart-turn, LiveKit and NVIDIA's own end-of-utterance model miss 70 or more.
- Across the 37 recorded conversations, in the real frameworks, the default Pipecat stack cuts in 2.1 to 7.5 times per minute. Ours, 1.1 to 1.4. Our first partial arrives in about a second; theirs in two and a half to eleven.

## scene: architecture
- This is how it is built. Audio goes into one frozen NVIDIA FastConformer. Four small heads read the same pass: voice activity, speaker, turn, and the streaming transcript. Nemotron-3 diarizes beside it. Parakeet TDT rewrites each finished turn. The events go straight into Pipecat or LiveKit.

## scene: limits
- Here is what it is not good at yet. It cuts in less often than the defaults, not never. On two-party calls it is polite, not fast: it matches VAP's recall but fires 0.7 seconds later, because it waits for silence instead of predicting it. Speaker verification, diarization and language ID trail the dedicated NVIDIA models. And it costs two to four times the CPU of the default stacks.

## scene: end
- Code, benchmarks, and the negative results are open. Link below.
