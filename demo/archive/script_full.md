# Narration, full cut (3-5 min). One bullet = one synthesized unit = one caption. `## scene:` lines switch the visual.
# Every figure appears in demo/archive/numbers.json with its runs/*.json source; every claim is checked against
# research/FINAL_REPORT.md (scorecard + sections 1-8), research/IMPROVE_115M.md A.2 and research/E2E_FINAL.md.
# Pace with the Dylan voice is ~0.10 s per character: keep the narration under ~2700 characters (~4.5 min with gaps).

## scene: hook
- Every voice agent has the same two failure modes. It interrupts you mid-sentence, or it waits so long you wonder whether it is still there.
- Turn-taking is usually a stack of separate models, each with its own delay. Can one streaming model do most of it, on a laptop CPU?

## scene: architecture
- We took NVIDIA's streaming FastConformer, loaded it without NeMo, and kept it frozen. Fine-tuning broke the transcript every time, so everything learned lives in small heads.
- The encoder runs once per 160 millisecond chunk. The heads read that pass: transcription, voice activity, speaker, and end of turn.
- NVIDIA's streaming diarizer runs beside it, and Parakeet TDT rewrites each finished turn offline.

## scene: demo_oto
- Live, on a real two-party call. Words appear as they are recognized, the bands show who holds the floor, and the flash marks the decision that the turn is over.
- The counter is the dead air that decision cost.

## scene: demo_ami
- On a meeting, with people talking over each other, it keeps up on two CPU threads, and Parakeet TDT rewrites each final while the stream keeps running.

## scene: scorecard
- On 974 AMI meeting turn ends, at a five percent false-cutoff budget, the speaker-aware hybrid misses 61.9 percent within six seconds. Smart-turn, LiveKit and NVIDIA's own model miss 70 or more. On held-out ICSI the direction replicates.
- On TurnBench, Sesame's two-party benchmark, the held-out predictive trigger reaches recall 0.86 against 0.84 for VAP, but at twice the false positives and 670 milliseconds later. VAP wins there.
- Parakeet TDT finals reach 9.5 percent word error rate on meeting segments, where the streaming pass gives 20.6.
- The whole server runs at real-time factor 0.64 on two CPU threads with the Nemotron diarizer: 1.7 to 3.6 times the CPU of the default stacks.

## scene: tsvad
- The measured bottleneck is knowing who the user is. With the oracle speaker, misses fall from 62 to 29 percent.
- So we trained a target-speaker VAD head, a quarter million parameters on the same encoder. Given a five-second voice print, its track feeds the turn head instead of the diarizer's column.
- Missed turn ends drop from 61.9 to 39.3 percent on AMI, and from 68.5 to 20.4 on held-out ICSI, at the same budget. Offline for now; not served yet.

## scene: headtohead
- Inside the real Pipecat and LiveKit pipelines, on 37 recorded conversations, our default cuts users off 1.6 to 5.5 times less often than Pipecat's default stack, with similar dead air.
- Against LiveKit's default, dead air is comparable and we miss fewer ends, but cut in more on mono mixes. Our first partial arrives within about a second; theirs in two and a half to eleven.

## scene: tradeoff
- The honest tradeoff is politeness versus speed: the opt-in rules give near-zero cut-ins for about 0.9 seconds more dead air, and we fire after the silence begins, where VAP predicts it.
- What it is not good at: speaker verification, diarization and language ID trail the dedicated NVIDIA models, and on floor-open ends a plain silence timeout is as good or better.

## scene: next
- Next: serve the target-speaker head, bind the user from the agent's own TTS end, and a trigger that fires before the silence. Everything is on GitHub.

## scene: endcard
- github dot com, slash max melichov, slash nvidia audio models.
