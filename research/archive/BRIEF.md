# Project brief (paste this as context for any model or collaborator)

You are joining an ongoing research and engineering project. Read this fully before answering anything.

## Goal
Reverse-engineer how NVIDIA builds its speech models, rebuild that recipe as an open, lightweight, NeMo-free
PyTorch framework, and use it to answer one question with measurements rather than opinions: **can one single
streaming model, running on a laptop CPU, equal or beat the dedicated models for every task a voice agent's
audio front-end needs (ASR, VAD, speaker identity, diarization, end-of-turn), and can it do end-of-turn better
than anyone because it is speaker-aware?** Secondary goal: find genuinely new (tier-1) results on the way, and
kill ideas that do not survive measurement.

## What exists (repo `nvidia-audio-models`, private GitHub, ~85 commits, ~470 tests)
- **research/ANALYSIS.md**: all 149 NVIDIA audio models on Hugging Face analysed. The common denominator:
  16 kHz log-mel -> FastConformer encoder (8x depthwise-striding subsampling, 80 ms frames) -> swappable heads
  (CTC / RNNT / TDT / AED / Sortformer diarization / LLM projector); cache-aware streaming with left/right
  attention context; tokens as the task interface; CC-BY-4.0 models used as teachers; FSQ codecs at 12.5 fps.
- **`audioforge/`**: that recipe in plain PyTorch (CPU / MPS / CUDA). FastConformer, RNNT/TDT losses verified
  by brute force, streaming output == offline output as a unit test, multi-head SpeechModel (one encoder,
  many heads), streaming runtime, WebSocket server, Pipecat and LiveKit adapters.
- **NVIDIA weights load without NeMo** (`nemo_import.py`): hybrid streaming ASR (1.92% WER LibriSpeech at 1 s
  lookahead), Parakeet-CTC (1.68%), Streaming Sortformer v2 (DER 0.20 on AMI), Nemotron-3 diarization,
  Parakeet-Realtime-EOU, TitaNet-Large, MarbleNet. The framework is faithful to the originals.
- **Our single model**: the frozen NVIDIA 115M streaming encoder (160 ms chunks) plus our heads (VAD, EOU,
  speaker embedding, diarization, turn head). Runs at RTF ~0.5 on 2 CPU threads, 3.4 GB, end to end in
  Pipecat and LiveKit (verified, 25 clean runs each). Encoder fine-tuning always broke ASR (WER gate), so the
  encoder stays frozen; everything we learn is in heads.
- **Data**: AMI (20 meetings) and ICSI (CC-BY-4.0) with word-level turn and hesitation labels; LibriSpeech;
  synthetic conversations. Cached streaming diarizer tracks for training and evaluation.
- **eot-bench v2** (`research/archive/EOT_BENCH_V2.md`): our leak-free end-of-turn benchmark. 974 AMI dev turns,
  label-free causal speaker enrollment, 6 s and 2 s horizons, every system held to <=5% false cutoffs,
  strata for floor-open ends (nobody else speaks; the case a voice agent must answer) vs floor-taken ends,
  bootstrap CIs. Earlier versions of our eval had an enrollment leak that flattered us; v2 fixed it.

## Scorecard so far (AMI dev, same metric code for all rows; `research/archive/BASELINES.md`)
| Task | Ours | Best dedicated | Verdict |
|---|---|---|---|
| ASR streaming | 1.92% WER | parakeet-ctc-1.1b 1.64% offline | equal |
| VAD | F1 0.949 | Silero 0.915, MarbleNet 0.937, pyannote 0.966 offline | better than every streaming VAD |
| Diarization frame DER | 0.394 | Sortformer v2 0.201, pyannote 3.1 0.220 | worse; we ship Sortformer inside our runtime |
| Speaker EER within meeting | 32% | TitaNet-L 8%, WeSpeaker 10% | worse |
| Turn end, all ends, 6 s | 61.9% missed | Silero timeout 72.7%, Parakeet-EOU 70.1%, smart-turn+Silero 72.6%, LiveKit ~74% | better than all dedicated turn models |
| Turn end, floor-open ends | 45.9% | Silero timeout 26.7% | worse by 19 pts |
| Product dead air (Pipecat/LiveKit) | median 1.7-1.9 s | timeout-grade | equal; 0.3 s diarizer setting saved 0.2-0.7 s |

Key facts behind the table:
- Our turn-end margin comes entirely from ends where another person takes the floor within ~1 s; a
  speaker-unaware detector never sees silence there.
- On open ends we lose because **label-free enrollment picks the wrong primary speaker ~30% of the time** and
  the diarizer adds ~0.84 s. With the oracle primary speaker, misses fall from 62% to 29%. This is the
  measured bottleneck: turn-taking here is a speaker-identity problem, not a language problem.
- The dedicated turn models add nothing over their own VAD timeout on meeting audio (smart-turn AUC 0.58 on
  AMI; LiveKit's text model does not improve even with oracle transcripts). Out of the box they cut users
  off a lot (Pipecat default 50% of turns).
- Mid-turn hesitations (median ~1.2 s) are longer than the silence at real speaker changes (~1 s), so silence
  timeouts cannot work in principle on this audio.
- Negative results, all recorded: voice-embedding enrollment with our own speaker head made things worse
  (34% EER); retraining speaker kernels, head-only retrains, adding ICSI, newer NVIDIA diarizers, cache flush
  artifacts: none moved the number. The hybrid rule (head OR timeout) that won on the bench gave nothing in
  the product because the server served a different head; being re-measured.

## Experiments running right now (decision matrix)
1. **Held-out confirmation on ICSI** (1312 turns, 5 never-fitted meetings): does "our head OR an
   any-speaker Silero timeout" (59.6% / 34.3% open on AMI dev) hold with frozen AMI thresholds?
2. **TitaNet-L enrollment**: real speaker embedding as the follower; enrollment rules "first voice after the
   agent stops speaking" (label-free in a product) and explicit "say hello". If it closes a big part of the
   gap to oracle, identity is confirmed as the root bottleneck and it ships in the server (`--enroll`).
3. **YIELD/HOLD action tokens in the transducer** (the "semantic end-of-turn" idea): head-only retrain on
   the frozen encoder with all speakers' words plus tokens, LibriSpeech anchor, WER gate. First attempt with
   primary-only targets learned to delete speech (WER 2.3% -> 85%); retry running. Prediction: matches
   Parakeet-EOU, i.e. no gain on open ends.
4. **State-contamination probe**: corrupt the speaker conditioning for a window, restore it, measure how long
   the damage persists in turn decisions and words, where it lives (encoder cache vs head/decoder state),
   replay cost, correction-delay sweep. Gates a possible paper: "learning to repair streaming state under
   delayed speaker corrections".
5. **trail6 head served in Pipecat/LiveKit with diarizer input**: does the bench win survive into the product?
6. A code-cleanup pass on files no agent is editing.

## Plan documents
- `research/archive/PLAN.md` (§6 is the chronological findings log), `research/archive/DATA_PLAN.md` (why more turns alone
  will not fix turn-taking; top moves are a label-consistency pass and enrollment supervision; budget tiers
  Mac-only / one GPU-week / $1-3k), `research/archive/IDEAS.md`, `IDEAS2.md` (53 ideas screened, 2 survivors),
  `research/archive/VERIFICATION.md` (28 claims independently checked: 4 confirmed, 20 partial, 4 refuted).

## How we work
- Multiple parallel agents; one GPU training at a time; CPU agents throttled to 2 threads and <10 min per
  process; load guards; machine monitor. Everything is committed with tests; every claim gets a verifier;
  overstatements are corrected in a README corrections log.
- Standards for claims: leak-free, held-out where possible, matched false-cutoff rates, CIs, one dataset means
  one dataset ("on overlapping multi-party audio", not "universally").
- Ideas are ranked by the measured bottleneck (speaker identity), not by elegance. Proposals about syntax,
  prosody or feature kinematics are parked until they address that bottleneck or show a number.

## What we want from you
Reason from these measurements. If you propose an experiment, say which row of the scorecard it moves, what
the control is, and what result would kill it. Do not restate the NVIDIA recipe; assume it.
