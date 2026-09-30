# Who watches this film, and what each shot owes them

## The viewer
A LinkedIn feed, sound off by default, thumb ready to scroll. Four kinds of people stop:

1. **Engineers building voice agents on Pipecat or LiveKit.** They run the default local stack every day:
   Silero VAD, a turn detector (smart-turn or LiveKit's model), Whisper, an LLM, a TTS. They know its failures by heart.
2. **Founders of voice-agent products.** They hear the failures as churn: "your bot keeps talking over me".
3. **People at Pipecat, LiveKit and NVIDIA.** They want to see their own components drawn correctly, and a clean
   way to drop this in.
4. **Investors.** They need the pain, the fix and the proof in one pass, without jargon.

## Their pains, in their words
- "The agent interrupts me mid-sentence." The top complaint. A pause to think is read as the end of the turn.
- "Nothing shows up until I stop talking." Whisper writes only after the turn (or a pause) is over, so the app
  and the LLM wait blind.
- "It has no idea who is talking." In a room, a car or a call with two people, the pipeline hears one blended voice.
- "I glue four models together." VAD, turn detector, ASR, diarizer: four runtimes, four failure modes.

## What they need from 90 seconds, in order
| need | shot that serves it |
|---|---|
| Recognise their own stack and its pain in the first 15 s | 1. the default chain drawn as they know it, then a real Pipecat default call cutting in |
| Understand what we built and where it sits | 2. the NVIDIA streaming ASR alone, then small heads growing on the same encoder pass, the diarizer, the TDT final |
| See before and after on the same audio | 3. split screens: cut-in vs waits, blank vs words, no speaker vs the user tagged |
| Believe it | real recordings, real benchmarks, the default stacks as baselines, the source file in fine print on every number |
| Know it drops in | end card: works inside Pipecat and LiveKit, open code, open benchmarks |

## What we may say (measured better, or a capability the defaults lack)
- Fewer cut-ins than Pipecat's default stack on every clip set (1.1-1.4 vs 2.1-7.5 per minute, 37 conversations).
- Words while the user speaks: first partial 0.8-1.2 s after onset vs 2.6-3.7 s (Pipecat) and 3.4-11.3 s (LiveKit).
- Fewer missed turn ends: 61.9 % vs 70.1-74.2 % for four detectors on 974 AMI turn ends at the same false-cutoff budget.
- Better final transcript than Whisper small: 9.5 vs 14.4 % WER on meetings, 1.9 vs 2.3 % on LibriSpeech.
- Knows who is talking (speaker-aware turns, per-person labels): a capability, not a race.
- One encoder pass for speech, speaker, turn and words, real time on two CPU threads: a capability, never "cheaper".

## What we cut, because it does not serve them or is not better
Voice print / TS-VAD, dead air, the "not perfect" shot, our streaming WER, our own diarization and language-ID
heads, speaker verification, VAP, compute comparisons, any claim against LiveKit's cut-ins. They live in the
research report, not in the film.

## Tone
A product film, not a lab talk: one idea per shot, the picture explains itself with the sound off, the narrator says
in plain words what is on screen, and every spoken number is on screen at that moment.
