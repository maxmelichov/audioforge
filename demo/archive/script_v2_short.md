# Narration v2, LinkedIn cut (60-90 s). Presentation register, first person plural, present tense.
# One bullet = one synthesized unit = one caption. Keep under ~800 characters (Dylan at 1.06x is ~0.085 s/char).

## scene: title
- Today we are presenting one streaming model that does a voice agent's listening: words, speech, who is talking, and when your turn is over.

## scene: ex_cutin
- Same audio, both sides. Left, Pipecat's default stack. Right, ours. She is mid-sentence, and the default answers four times. Ours waits.

## scene: ex_deadair
- Dead air. She stops here. The default answers after 3.2 seconds. Ours after 1.6.

## scene: ex_partial
- Words. Ours after 0.3 seconds. LiveKit's default after 9.

## scene: architecture
- One frozen NVIDIA encoder, four small heads, NVIDIA's diarizer beside it. Two CPU threads.

## scene: numbers
- On 974 meeting turn ends, at the same false-cutoff budget, we miss 61.9 percent. Every default detector misses 70 or more.

## scene: limits
- Not good at yet: fewer cut-ins, not zero. And on two-party calls it is polite, not fast: 0.7 seconds behind VAP.

## scene: end
- Code and benchmarks are open. Link below.
