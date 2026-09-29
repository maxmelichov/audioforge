# Narration, LinkedIn cut (60-90 s, sound-off first). One bullet = one synthesized unit = one caption.
# Every claim matches research/FINAL_REPORT.md (scorecard rows 6, 9, 14) and research/IMPROVE_115M.md A.2.
# Pace with the Dylan voice is ~0.10 s per character: keep the whole file under ~750 characters of narration.

## scene: hook
- Voice agents fail in two ways. They cut you off, or they leave you hanging.
- One streaming model that transcribes, hears speech, follows who is talking, and decides when your turn is over.

## scene: architecture
- One frozen NVIDIA encoder, small heads on top, NVIDIA's diarizer beside it.

## scene: demo_oto
- Live, on a real call: words as they are recognized, speaker bands, and a flash when it decides you are done.

## scene: scorecard
- On 974 AMI meeting turn ends, at matched false cutoffs, it misses 61.9 percent. Smart-turn, LiveKit and NVIDIA's detector miss 70 or more.
- Feed it a target-speaker track from a five-second voice print, and misses fall to 39 percent, and to 20 on held-out ICSI.
- It keeps up in real time on two CPU threads.

## scene: tradeoff
- The honest part: it is polite, not fast. It matches VAP's recall on two-party calls but fires about 0.7 seconds later.

## scene: endcard
- Code, benchmarks and negative results are open. Link below.
