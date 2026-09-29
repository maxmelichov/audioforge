# Narration v4: short sentences, concrete verbs, one or two per shot, in shot order (demo/render_v4.py SHOTS).
# Every spoken number is on screen at that moment. Voice: lively CustomVoice pick at 1.0x (see demo/README.md).
# "audioforge" instead of "ours" where a listener could hear "hours".

## scene: title
- This is audioforge. One model that listens for your voice agent.

## scene: two_systems
- Same call, two systems. Left, the default stack. Right, audioforge.

## scene: interrupt
- Watch the left. It jumps in while she is still talking. Now the right. It waits.

## scene: deadair
- She stops. The left needs three point two seconds. Audioforge, one point six.

## scene: words
- The catch with the default: no words until the turn is over. Audioforge shows them as she speaks.

## scene: room
- A meeting, four people. Audioforge tags the user and answers only her.

## scene: how
- How it works. Every 160 milliseconds, a chunk of audio enters one frozen NVIDIA encoder. Four small heads read the same pass: speech, speaker, turn, and words. NVIDIA's diarizer runs beside it, and at each turn end Parakeet TDT rewrites the sentence.

## scene: vpflow
- Give it five seconds of your voice. A tiny head tracks only you, and the turn head listens to that.

## scene: voiceprint
- Same clip, without and with the print. Without, it cuts in. With, it waits.

## scene: graph_wer
- Transcripts. Clean speech: we match Whisper small. Meetings: our live stream is worse, and the TDT rewrite fixes it.

## scene: graph_turn
- Missed turn ends: fewer than every detector we tested. With a voice print, far fewer.

## scene: graph_e2e
- In the real pipelines: fewer cut-ins than Pipecat, words ten times sooner than LiveKit.

## scene: graph_rtf
- All of it in real time, on two CPU threads.

## scene: honest
- Not perfect. It still interrupts sometimes, and on two-party calls it is slower than the best detector.

## scene: end
- It is open. Code, benchmarks, negative results.
