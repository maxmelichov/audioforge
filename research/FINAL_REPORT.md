# Final report: one streaming model as a voice agent's audio front end, measured against the dedicated models and the shipping stacks

> **Status (2026-09-29): historical.** This report describes the room-mode product of 2026-09-28 (115M model +
> NVIDIA diarizer). The shipped product is now single-model mode, and the numbers to quote are the standard
> scorecard in [`research/METRICS.md`](METRICS.md). Terms below map as follows: "dead air" = end-of-turn latency,
> "cut-ins" = false interruptions, "missed within 3 s" = no response within 3 s (pooled over mono-mix sessions,
> which no system can answer in time). "First partial transcript ... after speech onset" counted from the start of
> speech, so it included the time it takes to say the first word; it is not STT latency (partial latency, word end
> to word shown, is 441 ms median). Language ID row 11 is an older 0.40 M head; the shipped distilled head scores
> 91.0 % / 97.8 % (`runs/lid.json`).

2026-09-28 (first version 2026-09-27; this refresh adds the work of 2026-09-28 and re-reads its numbers from their JSON, see
`research/VERIFICATION_2026-09-28.md`). Repository `nvidia-audio-models`. Written for a technical reader who did not
follow the project. Every number comes from a `runs/*.json` file (Appendix A lists them) or, for three tables of
2026-09-28, from a named scratch file on the external SSD (`scratch/` = `/Volumes/ExternalSSD/nvidia-audio-models/
scratch`); where a research document disagrees with its JSON, the JSON is used and the disagreement is listed in
Appendix B. "Measured" means run by us on this machine (Apple M5, CPU, 2 threads per model process, shared with
other jobs) with the same data and metric code for every row. "Published" (P) means taken from a paper, model card or
leaderboard, not re-run here; published numbers use other test sets and protocols and are context only. CIs are 95 %
bootstrap unless stated. The companion HTML file `research/FINAL_REPORT.html` is this document rendered for a phone.

## Executive summary

**What was built.** `runs/stage1_served.afm` is NVIDIA's 115M-parameter cache-aware streaming FastConformer
(`stt_en_fastconformer_hybrid_large_streaming_multi`, CC-BY-4.0), loaded without NeMo (`audioforge/nemo_import.py`)
and **kept frozen**, with small heads trained by us on top of its frames: a VAD head (33 K parameters), a speaker
embedding head (0.50 M, reading block 4, distilled from TitaNet-L), a diarization head (1.9 M) and a
speaker-conditioned end-of-turn head; a 0.26 M target-speaker VAD (TS-VAD) head was added on 2026-09-28. It runs at
160 ms chunks (80 ms lookahead) on a laptop CPU. Every attempt to fine-tune the encoder tripped the ASR gate
(LibriSpeech WER 2.05 → 4.31 % in 500 steps even at lr × 0.05 with a KL anchor), so everything learned lives in
heads. The served product (`audioforge/serve.py`, WebSocket; Pipecat and LiveKit adapters in `integrations/`) pairs
it with an NVIDIA diarizer (Nemotron-3-Diarization recommended, all 8 columns; Streaming Sortformer v2 is still the
download default because the benchmarks were run with it), because our own diarization head is not good enough, and
optionally with NVIDIA Parakeet-TDT 0.6B v3 as a per-turn offline transcript pass, TitaNet-L for voice enrollment and
AmberNet for language ID.

**Where it stands (details in the scorecard):**

- **Better than the dedicated models:** VAD on AMI meetings (F1 0.949 vs Silero v5 0.915 and MarbleNet 0.937, all
  streaming; the head was trained on the AMI word-level label it is scored with). End-of-turn on *all* turn ends of
  AMI and held-out ICSI: the speaker-aware rules miss fewer ends than every dedicated turn detector we ran at the same
  cross-fitted ≤ 5 % false-cutoff budget (AMI 58.2-61.9 % vs 70.1-74.5 %; ICSI 79.5 vs 87.4 % for Silero, Δ −7.9
  [−10.0, −5.8]). On two-party otoSpeech calls the hybrid beats a timeout on the same channel (7.5 vs 10.0 %).
- **Equal:** LibriSpeech ASR against Whisper small (2.29 vs 2.42 %, Δ −0.13 [−0.56, +0.30]) at 12x less CPU.
- **Worse:** meeting ASR (AMI 24.4 % vs 19.3-21.2 % offline; solved on the product side by **Parakeet-TDT 0.6B v3 per
  turn, 9.7 %**), speaker verification (19.8 vs 12.0 % EER for TitaNet-L at n = 200), diarization (our head 0.394 DER,
  worse than the one-speaker baseline), language ID (75.0 % at 2 s vs 95.1 % AmberNet), end-of-turn on **floor-open
  ends** (a Silero timeout is as good or better) and on TurnBench two-party calls (VAP reaches similar recall, 0.841 vs
  0.862, at half the false positives and 670 ms lower median latency).
- **In the product** (37 recorded conversations through the real Pipecat 1.12 and LiveKit Agents 1.8 pipelines): vs
  **Pipecat's default local stack** our default cuts users off 1.6-5.5x less often and misses fewer ends, with
  similar dead air; vs **LiveKit's default** dead air is comparable and we miss fewer ends but cut in more on mono
  mixes. (A "first partial transcript after speech onset" comparison stood here; it counted from the start of speech
  and was removed as misleading. See the status note above.)
- **Added on 2026-09-28** (scorecard rows 15-26):
  - *Rooms.* The served server cut Nemotron-3 to 4 of its 8 columns (speakers 5-8 were invisible) and labelled every
    turn speaker 0 under load shedding. Fixed: all 8 columns and hold-on-shed are the launcher defaults, a voice-keyed
    speaker registry and a `timeout_any` policy are opt-in. On three 4-6-speaker clips DER 0.395 → 0.247 with the
    speaker count exact; under forced shedding 0.809 → 0.329. On AMI's 20 s windows (≤ 4 speakers) DER is unchanged
    (0.245), and there the registry labels are less accurate than the column (0.93 vs 0.77), so it stays opt-in.
  - *Compute.* Exact CPU fast paths cut server compute to 0.62x in an interleaved A/B with no decision changed
    (probabilities within 1e-5); the Nemotron-3 server measures RTF 0.45-0.51 on the E2E clips and 0.62 in a long
    session's steady state (loaded machine). Robustness: 48 tests over 36 failure modes, 3 defects in shipped code fixed.
  - *TS-VAD turn path.* With the user's 5 s voice print, misses fall from 61.9 to 39.3 % on AMI and from 68.5 to
    20.4 % on held-out ICSI (offline, same budget). Served live on 69 mostly two-party sessions it misses **more** than
    the product default (+10.8 [+5.6, +15.8] points at 3 s) and equals the no-cut-in rule at 0.43x its compute: **a
    meeting-room gain, not a two-party gain.** Shipped behind flags.
  - *0.6B streaming core.* Meeting WER −13 points, VAD and every ICSI turn row better, but AMI turn misses with
    `hybrid_dyn` **+2.8 [+0.1, +5.3] worse** and five categories only equal: **not adopted** under the user's rule that
    every category must improve. On GPU it is the recommended *transcript* model next to the 115M core.
  - *Smaller gains:* decoder-only adaptation AMI 24.4 → 22.5 % (−1.9 [−2.8, −1.2]); the [70,13] lookahead pass −1.4 /
    −2.5 / −0.4 points (AMI / ICSI / LibriSpeech); speaker head 19.8 → 17.1-17.4 % AMI and 7.0 → 4.0-4.4 % ICSI EER;
    language ID read from the TDT v3 transcript 100 % (full utterances) / 91.7 % (2 s) on the 10 of its 25 languages
    in our FLEURS set.
  - *Negative:* a TS-VAD-trained predictive turn head (+9.9 / +13.5 points of misses), a fast path at silence onset
    (≤ 80 ms earlier), and distilling Nemotron-3 into a 1.9 M head (AMI DER 0.367-0.385 vs 0.232).

**The honest headline.** *A frozen NVIDIA streaming ASR encoder supports a multitask voice-agent front end (ASR,
VAD, speaker, turn) on a laptop CPU, and with an NVIDIA diarizer beside it the whole server runs at RTF 0.64 live on
two threads (0.45-0.51 on the same clips after the 2026-09-28 fast paths). Its VAD beats the streaming VADs on
meeting audio, and its speaker-aware turn rules miss fewer turn ends than every dedicated turn detector we measured on AMI and ICSI at matched false cutoffs. It does not match the
dedicated models for speaker verification, diarization, language ID or meeting ASR; its turn-end gain comes from ends
where someone else takes the floor, and on floor-open ends and TurnBench calls it does not beat a Silero timeout or
VAP.* The measured bottleneck is **who the user is**: with the oracle primary the AMI miss rate falls from 61.9 to
28.7 %, and a voice print fed through the TS-VAD head recovers two thirds of that offline (39.3 %) on meetings, but not
on two-party calls, where a 1 s timeout on the user's own channel is already strong.

**What to run** (§8): the 115M streaming core at 80 ms for VAD, turn, speaker and partial transcripts, with the
`--perf` fast paths on; Parakeet-TDT v3 per finished turn for the transcript on CPU, the 0.6B streaming model or TDT
v3 on GPU; Nemotron-3-Diarization with all 8 columns and hold-on-shed; `--diar-labels registry` + `timeout_any` for
rooms; the TS-VAD path behind flags when the product holds the user's voice print; AmberNet for language ID, off by
default. Next step: the prepared 0.6B → 115M distillation run (~6 A100 hours).

## Scorecard

Verdict = ours vs the strongest system **we measured** on the same data and metric. "P" marks published numbers
(different data or protocol). EOT = end of turn; FC = false cutoff (the system declares the end while the user is
still mid-turn); miss = the end is not detected within the horizon; "open" = floor-open ends.

| # | task | dataset / protocol | ours | strongest measured | Δ ours − strongest [95 % CI] | best published (P) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | ASR, read speech | LibriSpeech test-clean, first 200 utt., `normalize_text` WER, greedy | 2.29 % streaming 160 ms, RTF 0.015 (offline batch-4 decode, model only; live in the server the ASR pass is 0.15, `PERFORMANCE.md` §1) | Whisper large-v3-turbo 1.47 % (RTF 0.88); parakeet-ctc-0.6b 1.68 % (0.040); Whisper small 2.42 % (0.18) | +0.82 [+0.42, +1.25] vs turbo; +0.60 [+0.24, +1.03] vs parakeet; **−0.13 [−0.56, +0.30] vs Whisper small** | parakeet-tdt-0.6b-v3 card 1.93 %, full test-clean | **equal** to Whisper small at 12x less CPU; worse than the offline 0.6-0.8B models |
| 2 | ASR, meetings | AMI dev, 200 single-speaker segments (3271 words), same scoring | 24.4 % (20.6 % Whisper norm.), RTF 0.022 (offline batch-4 decode, model only) | Whisper turbo 19.6 % (12.9 %), RTF 1.47; parakeet-ctc 19.3 % (14.1 %), 0.049 | +4.80 [+2.53, +7.41] vs turbo; +5.11 [+3.38, +7.14] vs parakeet; +3.27 [+1.22, +5.85] vs Whisper small | tdt-0.6b-v3 card AMI-IHM 11.31 % | **worse**; fixed by the hybrid: **Parakeet-TDT v3 per turn 9.7 % (9.5 %)**, −14.7 [−17.0, −12.7] vs ours, RTF 0.065 |
| 3 | VAD | AMI dev, 64 × 20 s windows, 80 ms frames, "any speaker" word label, threshold 0.5 | F1 0.949, 12.1 % miss at FPR 0.075 | MarbleNet v2 0.937 (12.3 %); Silero v5 0.915 (13.8 %); pyannote seg-3.0 0.966 but offline at 57-66x the CPU | no CI (single pooled number) | – | **better** than every streaming VAD; caveat: label convention |
| 4 | speaker verification | AMI dev single-speaker segments, cosine EER, within-meeting pairs; n = 64 (487 trials) / n = 200 (5101) | 14.4 % / 19.8 % (block-4 relational head, 0.50 M) | TitaNet-L 8.2 % / 12.0 %; WeSpeaker 9.6 / 11.9 %; ECAPA 9.6 / 13.0 % | +6.2 / +7.8 points (no CI stored; ± ~3 at n = 64) | TitaNet-L 0.66 % VoxCeleb1-O (card) | **worse**; ICSI dev 5.2 / 7.0 % vs TitaNet 1.0 / 2.1 % (ICSI train speakers were in the head's training data) |
| 5 | diarization | AMI dev, 64 windows, pooled frame DER, no collar, overlap scored | 0.394 (ICSI 0.354) | Sortformer v2 offline 0.201 (ICSI 0.209); v2 streaming 0.32 s 0.28 on turn windows; Nemotron-3 max-pool 0.232 offline / 0.241 streaming; pyannote 3.1 0.220 | +0.19; layer routing recovers ≤ 0.03 (block 6: −0.017 [−0.031, −0.002]) | Nemotron-3 DIHARD3 13.55 % (card) | **worse** than the trivial one-speaker baseline (0.312); the product ships NVIDIA's diarizer |
| 6 | EOT, all ends | eot-bench v2, AMI dev, n = 974 turns, causal label-free primary, cross-fitted ≤ 5 % FC per turn, 6 s horizon | head OR dyn(Silero, head) 58.2 % [54.9, 61.5] (selected on this set); hybrid 61.9 % [58.7, 65.0] | Parakeet-Realtime-EOU 70.1 % [67.0, 73.0]; Silero timeout 72.7 % [69.8, 75.4]; smart-turn + Silero 72.6 %; LiveKit turn detectors + timeout 73.7-74.5 % | hybrid − Silero **−10.8 [−14.2, −7.0]**; hybrid − EOU **−8.2 [−11.8, −4.2]**; head OR dyn − Silero −14.5 [−17.4, −11.3] | eot-bench (LiveKit) dead air at 5 % FC: LiveKit v1 543 ms, smart-turn v3.2 1051 ms, VAD 1600 ms (P, different data) | **better** than every dedicated detector (realised FC 5.2-6.5 %, see §5) |
| 7 | EOT, floor-open ends | same, the 236 open ends | 31.5 % [25.6, 37.5] (head OR dyn); hybrid 45.9 % [39.3, 52.4] | **Silero timeout 26.7 % [21.1, 32.4]** | head OR dyn − Silero +4.8 [−1.6, +10.7]; hybrid − Silero +19.1 [+11.1, +27.0] | – | **equal / worse**: a silence timeout is as good on the ends an agent answers |
| 8 | EOT, held-out meetings | ICSI, 5 unseen meetings, n = 1312, AMI thresholds frozen, 6 s | head OR dyn 79.5 % [77.2, 81.7] all / 50.5 % [44.5, 56.7] open, FC 1.8 % | Silero timeout 87.4 % [85.5, 89.1] / **45.5 % [38.4, 52.5]**, FC 3.6 % | all **−7.9 [−10.0, −5.8]**; open +5.0 [−1.4, +11.5] | – | **better** on all ends, **equal** on open ends; only Silero and Sortformer timeouts were run on ICSI |
| 9 | EOT, two-party calls | TurnBench dev (Sesame), 38 conversations, 1904 ends, official scorer, FP ≤ 0.10 | predictive OR Silero 0.862 recall / FP 0.092 / P50 1137 ms (held-out halves); best held-out fixed hybrid 0.849 / 0.098 / 1316 ms | **VAP (rescored) 0.841 / 0.045 / 463 ms**; ESPnet dual-channel 0.836 / 0.074 / 895 ms; Kyutai 0.803 / 0.100 / 1024; smart-turn v3 0.754 / 0.100 / 1010 | recall +0.02 at 2x the FP and +674 ms P50 (scorer gives no CI) | leaderboard test: Vox Maru v1 0.960 / 0.072 / 548 ms; VAP test 0.845 / 0.055 / 368 ms | **worse** than VAP (latency); better than smart-turn, Kyutai, Parakeet-EOU on recall |
| 10 | EOT, two-party, real calls | otoSpeech dev, 60 conversations, 2175 human turn ends, eot-bench v2 protocol, oracle party channel | hybrid_energy 7.5 % [6.5, 8.7] missed at 6 s, FC 3.9 % | primary-channel timeout 10.0 % [8.8, 11.3] | −2.5 points | none known | **better** than a timeout on the same channel |
| 11 | language ID | FLEURS test, 17 languages, n = 2550, accuracy at 2 s / full utterance | 75.0 % / 89.3 % (0.40 M head) | **AmberNet 95.1 % / 99.5 %**; SpeechBrain ECAPA 94.6 / 99.3; Whisper base 83.6 / 98.5 | −20 / −10 points (CI ≈ ±1.5 at 75 %) | AmberNet 5.22 % error on VoxLingua107 (card) | **worse**; ships off by default, `serve --lid ambernet` wires in AmberNet |
| 12 | product, vs Pipecat default | E2E_FINAL, 5 AMI + 16 TurnBench + 16 oto clips × mono / user channel, Pipecat 1.12 at 1x | C: 1.1-1.4 cut-ins / min; missed at 3 s 6-68 %; dead air median 1.35-1.68 s | A (Silero + smart-turn v3.2 + Whisper small): 2.1-7.5 cut-ins / min; missed 43-75 %; dead air 1.05-2.55 s | cut-ins per clip −1.8 [−3.2, −0.4] AMI, −1.38 [−2.19, −0.63] TurnBench mix, −1.25 [−2.56, −0.13] oto mix; missed 3 s −7 to −40 points | – | **better**: far fewer cut-ins and misses, dead air similar |
| 13 | product, vs LiveKit default | same, LiveKit Agents 1.8 | C: dead air 1.31-1.66 s; missed 3 s 0-34 %; 1.5-4.5 cut-ins / min | B (Silero + EnglishModel + Whisper small): 1.31-2.66 s; 13-70 %; 1.0-3.4 / min | dead air +40 to −402 ms (CIs include 0); missed 3 s TurnBench mix −36 [−55, −17]; cut-ins per clip oto user −0.06 [−0.69, +0.50], TurnBench user +0.94 [+0.31, +1.63] | – | **equal** dead air, fewer misses, more cut-ins on mixes |
| 14 | compute | E2E_FINAL live runs, 2 threads, per audio second | server 0.80 wall (Sortformer) / 0.64 (Nemotron-3), 1.02 / 0.71 CPU s, 3.6 / 1.5 GB | Pipecat default 0.43 CPU s, 2.4 GB; LiveKit default 0.28 CPU s, 2.8 GB | – | – | **worse**: 1.7-3.6x the CPU of the default stacks; keeps up at 1x |
| 15 | diarization, many speakers in the served system (2026-09-28) | 3 clips (6-speaker LibriSpeech mix, ICSI Bmr021 5 speakers, AMI IS1008b 4), offline `Session`, Nemotron-3; per-final speaker accuracy after a Hungarian map | after the fix (8 columns, `registry` labels, hold-on-shed): pooled DER **0.247**, speaker count 6 / 5 / 4 (exact), accuracy 0.83 (served speaker head) / **0.88** (TitaNet); forced level-1 shedding DER **0.329** | before (4 columns, column labels, VAD-on-shed): DER 0.395, count 4 / 4 / 4, accuracy 0.74; shedding 0.809 with one id per clip | no CI (3 clips); AMI dev 64 windows: DER 0.245 → 0.245, accuracy 0.93 → 0.77 (registry worse on 20 s windows) | Nemotron-3 card: 8 speakers (P) | **better** for rooms of more than 4 speakers and under load; **equal** on AMI windows; registry opt-in |
| 16 | server compute after the performance pass | interleaved A/B `--perf none` vs `default`, 5 AMI windows, Nemotron-3, same process and load | RTF **0.86** (0.62x), CPU 0.67x, chunk p95 0.38x; 0 decision differences in 1313 messages, probabilities within 1e-5; direct 0.45-0.51 (load 3.5-5), 0.62 steady state (120 s window); Sortformer v2 0.61 | the same server before: RTF 1.39 in the A/B (load 7-8), 0.84 at load 4.7; E2E_FINAL live 0.64 / 0.80 | ratio from one A/B (no CI) | – | **better** (same outputs, less compute); still one session per 2 threads (K = 2 aggregate RTF 1.31) |
| 17 | robustness of the server | `tests/test_bulletproof.py`, `research/archive/BULLETPROOF.md` | 48 tests over 36 failure modes (audio, protocol, models, load shedding, clients); 3 defects in shipped code found and fixed; no default policy's measured behaviour changed | – | – | – | engineering; not a benchmark row |
| 18 | EOT with the TS-VAD track, offline | eot-bench v2, AMI dev 974 / ICSI held-out 1312 turns, 5 s voice print from elsewhere in the meeting, ≤ 5 % FC cross-fitted, 6 s | hybrid **39.3 % [36.2, 42.8]** AMI, **20.4 % [18.2, 22.7]** ICSI; `hybrid_dyn` 34.2 / 18.7 %, open 17.9 / 13.2 % | the shipped hybrid on the label-free Sortformer column: 61.9 % [58.6, 65.3] AMI, 68.5 % [65.8, 71.2] ICSI | CIs do not overlap (paired CIs pending the Sortformer-track rebuild, §11) | – | **better** on meetings, given a voice print; head P50 still 3.2-4.2 s on AMI |
| 19 | EOT with the TS-VAD track, live in Pipecat | 37 E2E clips × conditions = 69 sessions (64 two-party), 1x, system T = `--turn-input tsvad --diar-off`, `hybrid_dyn` | missed at 3 s 45.3 %; cut-ins 0.52 per session; server RTF 0.34 (median), 1.6 GB | C (product default, timeout): 34.5 %, 0.93; D (`hybrid_dyn` + arming, Sortformer): 44.0 %, 0.52; RTF 0.80 | T − C **+10.8 [+5.6, +15.8]** misses, −0.41 [−0.67, −0.12] cut-ins; T − D +1.3 [−3.6, +6.2], 0.00 [−0.17, +0.19] | – | **worse** than C on misses (pre-registered bar not met), **equal** to D at 0.43x the compute: a meeting-room gain, not a two-party gain; behind flags |
| 20 | ASR, meetings: decoder-only adaptation | AMI / ICSI / LibriSpeech 200, RNNT prediction net + joint trained 500 steps (encoder and all live heads frozen, 748 / 759 tensors bit-identical) | AMI **22.50 %** | served decoder 24.43 % | AMI **−1.93 [−2.77, −1.24]**; ICSI −0.12 [−0.78, +0.56]; LibriSpeech +0.04 [−0.09, +0.18] | – | **better** on AMI, **equal** on ICSI and read speech; not shipped by default (probably AMI-vocabulary adaptation) |
| 21 | ASR: dual-lookahead second pass | same model at [70,13] (1.04 s lookahead), `--asr-lookahead 13`, text only | AMI 23.02 %, ICSI 24.78 %, LibriSpeech 1.92 % | the served [70,1] pass: 24.43 / 27.29 / 2.29 % | **−1.41 [−2.48, −0.32]**, **−2.50 [−3.80, −1.32]**, **−0.37 [−0.64, −0.11]** | – | **better**, small; no live decision changes |
| 22 | 0.6B streaming core as the GPU core (user's rule: every category must improve) | nemotron-speech-streaming-en-0.6b vs the 115M under identical recipes; WER, VAD, speaker, eot-bench, first partial | AMI WER 11.16 %, ICSI 14.35 % ([70,1]); VAD F1 +0.007 / +0.004; ICSI turn rows −5 to −13 points; MPS RTF 0.24 | the 115M core: 24.43 / 27.29 %; CPU RTF 0.52 vs 1.68 (0.6B not real time on 2 threads) | AMI eot-bench `hybrid_dyn` **+2.8 [+0.1, +5.3]** (worse); LibriSpeech, first partial, speaker EER, AMI head rows: CIs include 0 | card AMI 14.71 % at 160 ms (P) | **not adopted**: one category **worse**, five **equal** (not shown to improve); recommended as the GPU transcript model |
| 23 | speaker verification, crop-level TitaNet distillation | AMI dev / ICSI dev within-meeting EER, n = 200, same 0.5 M head on block 4 | **17.1-17.4 %** AMI, **4.0-4.4 %** ICSI (two heads: warm start 17.1 / 4.4, fresh init 17.4 / 4.0) | TitaNet-L 12.0 / 2.1 %; shipped head 19.8 / 7.0 % | vs shipped −2.4 to −2.7 AMI (no paired CI vs the shipped head; warm vs fresh −0.3 [−1.5, +1.4]) | – | **better** than the shipped head, still **worse** than TitaNet; not transplanted (the TS-VAD head is tied to the shipped embedding space) |
| 24 | language ID from the TDT v3 transcript | FLEURS-17, langid.py on the per-turn transcript, v3's 10 supported languages of our 17 | **100 %** (1260 / 1260) full utterances, **91.7 %** (275 / 300) at 2 s; 0 % on every unsupported language | AmberNet on the same utterances: 99.9 % / 95.1 % | 2 s: −3.4 points (no CI) | – | **equal** on full utterances, **worse** at 2 s; free with the transcript; silent failure out of set |
| 25 | EOT latency / FC frontier (product knob) | eot-bench v2, TS-VAD track, lowest-P50 point within each FC budget (**in-sample**) | at 10 % FC `hybrid_dyn` misses **22.4 %** AMI / **13.3 %** ICSI, P50 2320 / 1600 ms | the same rule at 5 % FC: 34.9 % / 20.4 %, P50 3280 / 2080 ms; a timeout on P(target) at 10 %: 56.3 / 32.9 % | in-sample, no CI | – | a tunable knob, not a verdict |
| 26 | negative results of 2026-09-28 | eot-bench v2 / AMI-64 and ICSI-64 DER | TS-VAD-trained predictive turn head +9.9 [+6.3, +13.2] / +13.5 [+10.7, +16.3] points of misses; `hybrid_fast` P50 ≤ 80 ms earlier; distilled diar head AMI DER 0.367-0.385 | the served head on the same track; Nemotron-3 teacher 0.232 | see §10 | – | **worse**; killed by their pre-registered criteria |

## 1. ASR

### 1.1 LibriSpeech test-clean, first 200 utterances (4634 words; `runs/final_asr.json`, `runs/hybrid_asr.json`)

All systems on the same 200 utterances, greedy decoding, CPU with 2 threads; RTF = decode wall time / audio time,
model load excluded, machine shared with other jobs. Primary scoring is `audioforge.teachers.normalize_text` on
reference and hypothesis; "fillers dropped" also removes um / uh / mm-hmm from both sides; "Whisper normaliser" is
Whisper's `EnglishTextNormalizer`. Paired deltas are on the primary scoring (the JSON has them for that only, plus
Whisper-norm deltas for the TDT rows).

| system | params | WER % (normalize_text) [95 % CI] | WER % fillers dropped | WER % Whisper normaliser | Δ vs ours, normalize_text (paired) | Δ vs ours, Whisper norm. (paired) | CPU RTF, 2 threads |
|---|---|---|---|---|---|---|---|
| **ours, served 115M, streaming 160 ms (RNNT)** | 114.6M encoder + heads (shared) | 2.29 [1.78, 2.87] | 2.29 | 2.27 | ref | ref | 0.015 |
| NVIDIA parakeet-ctc-0.6b, offline | ~600M | 1.68 [1.21, 2.16] | 1.68 | 1.63 | **-0.60 [-1.03, -0.24]** | – | 0.040 |
| Whisper small (faster-whisper int8), offline | 244M | 2.42 [1.88, 2.96] | 2.42 | 2.29 | +0.13 [-0.30, +0.56] | – | 0.183 |
| Whisper large-v3-turbo (faster-whisper int8), offline | 809M | 1.47 [1.03, 1.96] | 1.47 | 1.37 | **-0.82 [-1.25, -0.42]** | – | 0.882 |
| **NVIDIA parakeet-tdt-0.6b-v3, offline per turn (the hybrid's final pass)** | 627M | 2.03 [1.51, 2.55] | 2.03 | 1.88 | +0.26 [-0.20, +0.71] | +0.39 [-0.09, +0.86] | 0.045 (batch 1, machine under load) |

- The served model streams at 160 ms chunks (att context [70, 1]); the others see the whole utterance. The 0.6-0.8
  point gap to parakeet-ctc-0.6b and Whisper turbo is significant and is the price of streaming with 80 ms
  lookahead: the same imported encoder at 1.04 s lookahead ([70, 13]) scores 1.92 % on these utterances
  (`research/archive/NEMO_IMPORT.md`, no JSON), and 2.26 % on the first 100 (`runs/served_model_build.json`).
- Parakeet-TDT v3 per turn is statistically equal to the streaming model here (+0.26 [−0.20, +0.71]); its value is on
  meetings.
- Published (P, full test-clean, model cards): parakeet-ctc-0.6b 1.87 %, parakeet-ctc-1.1b 1.83 %, parakeet-tdt-0.6b-v3
  1.93 %, nemotron-speech-streaming-0.6b 2.32 %.

### 1.2 AMI dev, 200 single-speaker segments (3271 words; measured)

Headset-mix audio of the 4 AMI dev meetings, segments of 1-15 s with no other speaker within 0.2 s, 200 drawn with
`random.Random(0)` from 354 (the set of `scripts/bench_yield_tokens.py --stage wer`).

| system | params | WER % (normalize_text) [95 % CI] | WER % fillers dropped | WER % Whisper normaliser | Δ vs ours, normalize_text (paired) | Δ vs ours, Whisper norm. (paired) | CPU RTF, 2 threads |
|---|---|---|---|---|---|---|---|
| **ours, served 115M, streaming 160 ms (RNNT)** | 114.6M encoder + heads (shared) | 24.43 [22.04, 27.22] | 21.40 | 20.63 | ref | ref | 0.022 |
| NVIDIA parakeet-ctc-0.6b, offline | ~600M | 19.32 [17.37, 21.57] | 14.96 | 14.14 | **-5.11 [-7.14, -3.38]** | – | 0.049 |
| Whisper small (faster-whisper int8), offline | 244M | 21.16 [18.93, 23.71] | 16.24 | 14.43 | **-3.27 [-5.85, -1.22]** | – | 0.280 |
| Whisper large-v3-turbo (faster-whisper int8), offline | 809M | 19.63 [17.45, 21.89] | 14.54 | 12.94 | **-4.80 [-7.41, -2.53]** | – | 1.469 |
| **NVIDIA parakeet-tdt-0.6b-v3, offline per turn (the hybrid's final pass)** | 627M | 9.72 [8.22, 11.54] | 9.77 | 9.50 | **+14.70 [+12.71, +17.02]** | **+11.13 [+9.15, +13.45]** | 0.065 (batch 1, machine under load) |

- **Verdict: worse, and the hybrid fixes it.** The served streaming model is 3.3-5.1 points behind every offline
  system (CIs exclude 0), and the gap widens with fillers removed (21.4 vs 14.5-16.2 %) or under the Whisper
  normaliser (20.6 vs 12.9-14.4 %): AMI references keep "um" and "uh", which Whisper drops. Parakeet-TDT 0.6B v3,
  run once on each finished turn, gets 9.7 % (9.5 % Whisper-normalised) at RTF 0.065 on this CPU (batch 1, machine
  under load), 14.7 [12.7, 17.0] points better than the streaming transcript and 4.6 [3.2, 6.3] points (Whisper norm.)
  better than parakeet-ctc-0.6b. `serve --final-asr tdt_v3` runs it in a worker process after `turn_end` and the
  adapters can answer that transcript (`final_source="offline"`).
- TDT v3 limits: 25 European languages by its card, no Hebrew, Arabic, Turkish, Persian, Hindi, Chinese or Japanese,
  and no built-in language ID (its vocabulary has no language tokens: `runs/hybrid_asr.json:lid_vocab`). The
  planned FLEURS-17 run of it is at 0/150 utterances per language (`fleurs` block), so its multilingual WER is
  unmeasured here.
- Cost is the streaming model's advantage: 2.2x cheaper than parakeet-ctc-0.6b, 3x than TDT v3, 13-66x than
  Whisper small / turbo, and it is the only system in the table that gives partials while the user speaks.
- Published AMI numbers (P, full AMI-IHM test, Open ASR Leaderboard protocol): parakeet-tdt-0.6b-v3 11.31 % (its
  card, as stored in `hybrid_asr.json`; `research/archive/DATA_PLAN.md` quotes 11.39), nemotron-speech-streaming-0.6b
  14.71 % at 160 ms, Parakeet-Realtime-EOU-120M 15.62 %, Whisper small 19.0 % (Whisper paper). Our dev subset and
  normaliser are not that protocol.

### 1.3 The same model at more lookahead, and ICSI (2026-09-28; `runs/hybrid_asr.json` `english`)

The served 115M model at three attention contexts on the three 200-utterance sets (ICSI-200: ICSI dev meetings Bmr021
/ Bns001, the AMI sampling rule), masked offline forward = cache-aware streaming, batch 4, `normalize_text`.

| set (n = 200) | [70,0] (no lookahead) | **[70,1] (served, 160 ms)** | [70,13] (`--asr-lookahead 13`, 1.04 s) | [70,13] − [70,1] (paired) |
|---|---:|---:|---:|---|
| AMI dev segments | 25.68 | **24.43 [22.04, 27.22]** | 23.02 [20.54, 25.79] | **−1.41 [−2.48, −0.32]** |
| ICSI dev segments | 29.87 | **27.29 [24.75, 30.24]** | 24.78 [21.92, 27.73] | **−2.50 [−3.80, −1.32]** |
| LibriSpeech test-clean | 2.48 | **2.29 [1.78, 2.87]** | 1.92 [1.42, 2.49] | **−0.37 [−0.64, −0.11]** |

- **Verdict: better, small.** `serve --asr-lookahead 13` runs this as a second, text-only pass (no second model,
  ~10-20 % more encoder time) and sends its final as `"source": "lookahead"`; every live decision is unchanged
  (`tests/test_hybrid_asr.py`). It recovers 1-2.5 of the 15 points that TDT v3 recovers on meetings. The live cost of
  waiting for either second final (dead air) is still unmeasured (§11).

### 1.4 Decoder-only meeting adaptation (2026-09-28; `runs/hybrid_asr.json` `adapt`)

Only the RNNT prediction net + joint (5.33 M of 119.65 M parameters) trained, 500 steps on AMI + ICSI train with a
LibriSpeech anchor and a WER gate (never fired); the encoder and every live head stay frozen (748 of 759 tensors
bit-identical to the served model), so no live decision can change.

| set (n = 200) | served | adapted decoder | adapted − served [95 % CI] |
|---|---:|---:|---|
| AMI dev | 24.43 | **22.50** | **−1.93 [−2.77, −1.24]** |
| ICSI dev | 27.29 | 27.17 | −0.12 [−0.78, +0.56] |
| LibriSpeech | 2.29 | 2.33 | +0.04 [−0.09, +0.18] |

- **Verdict: better on AMI, equal elsewhere.** ICSI does not move although ICSI train is half the meeting data,
  so the AMI gain is most likely vocabulary adaptation to the AMI scenario meetings. Not shipped as the default
  (`runs/asr_meeting_decoder_adapt.afm` stays a research checkpoint); TDT v3 per turn remains the transcript fix.
- The 0.6B streaming model's WER on the same sets (11.2 % AMI, 14.4 % ICSI at [70,1]) is in §8.1.

## 2. VAD (AMI dev, 64 × 20 s windows, 16 064 frames of 80 ms; `runs/baselines_sd.json`, `runs/vad_layers.json`)

Label: "any speaker active" from AMI word times. Pooled over frames, threshold 0.5.

| system | params | licence | F1 | acc | FPR | miss at FPR ≈ 0.075 | latency | CPU RTF (2 thr) |
|---|---|---|---|---|---|---|---|---|
| **ours (VAD head on the shared encoder)** | 33 K head | CC-BY-4.0 encoder + ours | **0.949** | 0.921 | 0.179 | **12.1 %** | 160 ms chunk | 0.0061 (whole model pass) |
| NVIDIA MarbleNet v2.0 | 91 K | NVIDIA OML | 0.937 | 0.903 | 0.204 | 12.3 % | 20 ms + 1.46 s lookahead | 0.0003 |
| Silero VAD v5.1.2 | 0.24 M | MIT | 0.915 (best threshold 0.938) | 0.876 | 0.075 | 13.8 % | 32 ms | 0.0023 |
| WebRTC VAD (modes 0-3) | – | BSD | 0.919 / 0.919 / 0.898 / 0.725 | | | | 10-30 ms | ≈ 0 |
| pyannote segmentation-3.0 (offline) | 1.5 M | MIT | **0.966** | 0.947 | 0.169 | – | offline, 10 s windows | 0.35-0.40 |
| Sortformer v2, any column (offline) | 118 M | CC-BY-4.0 | 0.933 | 0.901 | 0.086 | – | offline | 0.0087 |

- **Verdict: better than every streaming VAD, worse than offline pyannote.** Caveat: the head was trained on AMI train
  with exactly this word-level label; the dedicated VADs detect acoustic speech, so part of the margin is label
  convention (pauses under ~0.5 s are hidden in AMI word times). No CI. On ICSI dev the head's F1 is 0.900.
- `research/archive/VAD_LAYERS.md`: a probe on block 4 alone matches the served head (F1 0.9476 vs 0.9485; best block 9
  0.9496), so the head is at the ceiling of what one encoder block gives.

## 3. Speaker verification (AMI dev single-speaker segments, 15 speakers; `runs/spk_head.json`, `runs/baselines_sd.json`)

Cosine scoring, EER. Within-meeting pairs are the honest protocol (same room and channel). No bootstrap CI is stored;
the docs estimate about ±3 points at n = 64 and ±1 at n = 200.

| system | params / licence | AMI n=64 all / within | AMI n=200 all / within | ICSI dev n=64 all / within | ICSI dev n=200 all / within |
|---|---|---|---|---|---|
| **ours, served head (block 4, AAM + TitaNet-relational distillation)** | 0.50 M / ours | 8.5 / **14.4 %** | 11.3 / **19.8 %** | 4.7 / 5.2 % | 6.4 / 7.0 % |
| ours, block 4, AAM only | 0.50 M | 8.5 / 15.7 % | 11.0 / 19.1 % | 14.4 / 17.4 % | 14.8 / 17.6 % |
| ours, original stage-1 head (all 17 layers, AAM) | 0.50 M | 15.0 / 32.2 % | 16.3 / 29.2 % | 36.5 / 42.3 % | 35.3 / 41.0 % |
| NVIDIA TitaNet-L (the teacher) | 22.1 M / CC-BY-4.0 | 6.6 / **8.2 %** | 10.9 / **12.0 %** | 1.4 / 1.0 % | 2.2 / 2.1 % |
| WeSpeaker ResNet34-LM | 6.6 M / CC-BY-4.0 | 9.9 / 9.6 % | 10.6 / 11.9 % | – | – |
| SpeechBrain ECAPA-TDNN | 20.8 M / Apache-2.0 | 9.9 / 9.6 % | 12.6 / 13.0 % | – | – |

- **Verdict: worse** (6.2 points behind TitaNet-L at n = 64, 7.8 at n = 200 within meeting). The project's largest
  single gain came from reading block 4 instead of a near-uniform mix of all 17 layers (speaker information lives in
  blocks 1-9) and distilling TitaNet: 32.2 → 14.4 %. Training data: 45 AMI + 33 ICSI-train + 251 LibriSpeech
  speakers (329); ICSI *dev* is held out, the ICSI corpus is not. Adding the 251 LibriSpeech identities under AAM
  alone did not help (within-meeting 28.8 vs 19.1 %); the teacher objective did.
- Every embedding tested, TitaNet's included, averages the two voices during overlap (nearest = primary 0.34-0.40).
- The runtime ships TitaNet-L for `--enroll` voice following; the head is the cheap option on the shared pass.

### 3.1 Crop-level TitaNet distillation with more identities (2026-09-28; `runs/spk_frame.json`)

Same 0.5 M head shape on block 4, trained on 1.5-4 s crops of AMI + ICSI train and all 251 LibriSpeech
train-clean-100 speakers, cosine + relational (within-batch similarity matrix) loss against TitaNet-L plus AAM, batch
128, 4000 steps (130 s on MPS). Same trials as the table above.

| head (block 4) | AMI n=64 all / within | AMI n=200 all / within | ICSI n=64 within | ICSI n=200 all / within |
|---|---|---|---|---|
| shipped relational head | 8.5 / 14.4 | 11.3 / 19.8 | 5.2 | 6.4 / 7.0 |
| crop-level distillation, warm start | 7.9 / 12.3 | 10.5 / **17.1** | 2.6 | 4.5 / **4.4** |
| crop-level distillation, fresh init | 7.9 / 11.5 | 10.7 / **17.4** | 2.5 | 4.2 / **4.0** |
| TitaNet-L | 6.6 / 8.2 | 10.9 / 12.0 | 1.0 | 2.2 / 2.1 |

- **Verdict: better than the shipped head, still worse than TitaNet.** The pre-registered bar (AMI < 12 %, ICSI ≤ 5 %)
  is met on ICSI only; warm vs fresh is −0.3 [−1.5, +1.4] (AMI), the same head. No paired CI against the shipped head
  is stored. It is not in the served model: the TS-VAD head was trained on the shipped head's embedding space and
  would need retraining on the new prints first.

## 4. Diarization (AMI dev, 64 × 20 s windows; `runs/baselines_sd.json`, `runs/layer_routing.json`)

Pooled frame DER, 80 ms, threshold 0.5, no collar, overlap scored.

| system | params / licence | frame DER (miss / FA / conf) | DER collar ±0.25 s | mode |
|---|---|---|---|---|
| **ours, diar head** | 1.9 M / ours | 0.394 (.246 / .059 / .089) | 0.346 | 160 ms streaming head |
| **NVIDIA Streaming Sortformer v2 (in the product)** | 118 M / CC-BY-4.0 | **0.201** (.140 / .045 / .016) | 0.167 | offline; 0.252 streaming 1.04 s; ~0.28 at the served 0.32 s setting on the turn windows |
| NVIDIA Nemotron-3-Diarization 100M, max pooling | 99 M / OpenMDW-1.1 | 0.232 (.191 / .027 / .014) | – | offline; 0.241 streaming; ~2x faster than v2; AMI train + dev are in its card's training data |
| NVIDIA Sortformer v2.1 | 118 M | 0.242 | – | offline |
| pyannote speaker-diarization-3.1 | 8.1 M / MIT | 0.220 (.070 / .063 / .087) | **0.167** (0.132 at ±0.5 s) | offline |
| pyannote community-1 | 8.1 M / CC-BY-4.0 | 0.264 | 0.214 | offline |
| trivial: one speaker = oracle VAD | – | 0.312 | – | – |

- **Verdict: worse**, and worse than the trivial baseline. Routing the head to other encoder blocks (`research/
  LAYER_ROUTING.md`, blocks 2-12, mixes, 2 seeds) gained at most 0.017-0.030 DER against a pre-registered bar of
  0.03 on both corpora; the gap to Sortformer is 10x that. ICSI: head 0.354, Sortformer 0.209. The head under-counts
  speakers (1.8-2.0 per window vs 2.62).
- Nemotron-3 scores above Sortformer v2 here mainly through a label-convention miss (VAD-level miss 16-19 % vs 10 %),
  yet in the product it gives lower dead air (§7). Its card claims DIHARD3 13.55 % and CALLHOME 11.32 % (P).

### 4.1 Many speakers in the served system (2026-09-28; `research/archive/DIARIZATION_FIX.md`, scratch `diar/`)

User report: "diarization, and many speakers in the room, doesn't work". Reproduced live through the quickstart
server. Three causes at once: `final.speaker` was the 5 s dominant column, not the turn's speaker, and a turn only
ended when that column fell silent (other people's turns were glued on); the launcher cut Nemotron-3 to its first 4
arrival-order columns (`--diar-spks 4`), so speakers 5-8 had no column at all; and on this loaded laptop one 1x
session already ran at RTF 0.87-0.98, load shedding fired within a minute, and shedding replaced the diarizer by the
VAD in column 0, i.e. every turn became speaker 0. Fix: all 8 columns (launcher default), `--shed-diar hold` (the last
stable column at half diarizer cadence; launcher default), `--diar-labels registry` (stable voice-keyed ids per
session from the served speaker head, or TitaNet with `--diar-embed titanet`; opt-in) and `turn_policy: "timeout_any"`
(ends every speaker's turn; opt-in).

Offline `Session`, the three clips (6-speaker LibriSpeech mix 119 s, ICSI Bmr021 5 speakers 60 s, AMI IS1008b 4
speakers 60 s), 2 threads, loaded machine; accuracy = per-final speaker accuracy after a Hungarian map; DER pooled
over the clips (scratch `diar/offline/n3_R*.json`, `sf_R*.json`):

| configuration | mix acc / ids (ref 6) | ICSI acc / ids (ref 5) | AMI acc / ids (ref 4) | mean acc | pooled DER | diarizer speaker count | RTF |
|---|---|---|---|---|---|---|---|
| **before**: 4 columns, column labels, VAD-on-shed, `timeout` | 0.80 / 4 | 0.58 / 4 | 0.83 / 3 | 0.74 | 0.395 | 4 / 4 / 4 | 0.53 |
| 8 columns, `registry` (speaker head), hold, `timeout` | 0.87 / 7 | 0.80 / 4 | 0.83 / 3 | 0.83 | **0.247** | **6 / 5 / 4** | 0.53 |
| 8 columns, `registry` (speaker head), hold, `timeout_any` | 0.80 / 7 | 0.82 / 4 | 0.75 / 3 | 0.79 | 0.247 | 6 / 5 / 4 | 0.53 |
| 8 columns, `registry` (**TitaNet**), hold, `timeout_any` | **1.00 / 6** | 0.77 / 5 | 0.88 / 5 | **0.88** | 0.247 | 6 / 5 / 4 | 0.57 |
| before, shedding level 1 forced | 0.33 / 1 | 1.00 / 1 (one 60 s final) | 1.00 / 1 (one final) | – | 0.809 | 1 / 1 / 1 | 0.27 |
| 8 columns, `registry`, hold, `timeout_any`, shedding forced | 0.91 / 9 | 0.73 / 4 | 0.57 / 5 | 0.74 | **0.329** | 6 / 5 / 4 | 0.44 |
| Sortformer v2, before | 0.25 / 4 | 0.43 / 4 | 0.57 / 4 | 0.42 | 0.673 | 4 / 4 / 4 | 0.78 |

AMI dev, 64 × 20 s windows (the §4 set; scratch `diar/offline/ami_*.json`, complete 64-window files): DER **0.245 →
0.245** (identical to four decimals: at most 4 speakers per window), per-final accuracy 0.93 → 0.77 (a registry that
starts empty every 20 s sees one or two short turns per speaker), forced shedding 0.490 → **0.335** with the speaker
count error −1.28 → −0.25. Live through the server with the new flags (scratch `diar/logs/new_*.json`, quieter
machine): the mix gets 7 ids / accuracy 0.84-0.87 (before: 4 / 0.80), ICSI accuracy 0.80-0.81 (before 0.58), two
concurrent sessions RTF 0.61 / 0.67 with no shedding.

- **Verdict: better for rooms of more than 4 speakers and under load; equal on AMI's short windows.** The 8 columns
  fix the speaker count at the same RTF; the registry fixes the labels of long multi-party sessions (0.74 → 0.83 with
  the free speaker head, 0.88 with TitaNet) but is worse than the column label on short windows with few speakers,
  hence opt-in; hold-on-shed removes the speaker-0 collapse. No CIs (3 clips). Not fixed: overlapping speakers in one
  turn still get one label (purity 0.75-0.86), the speaker head over-splits a voice now and then (7 ids for 6), and
  Sortformer v2's column swaps in our streaming procedure make it the wrong diarizer for rooms.

### 4.2 Distilling Nemotron-3 into our own head (negative; `runs/diar_distill.json`)

A 1.92 M head on the frozen served encoder (block 6, or blocks 4 + 6), trained against Nemotron-3's cached posteriors
plus labels on AMI + ICSI train (1800 steps, 8-9 min MPS each). Bar: DER within 0.05 of the teacher on AMI-64 and
ICSI-64, count error no worse; kill after two runs if the gap stays above 0.10.

| model | AMI-64 DER | AMI count error | ICSI-64 DER | ICSI count error |
|---|---|---|---|---|
| Nemotron-3 teacher (8 columns, max pool) | **0.232** | −0.14 | 0.236 | +0.06 |
| served stage-1 head (before) | 0.394 | −0.72 | 0.354 | −0.28 |
| distilled, block 6 | 0.385 | −0.69 | **0.193** | −0.22 |
| distilled, blocks 4 + 6 | 0.367 | −0.66 | 0.196 | −0.23 |

- **Verdict: worse; killed.** The AMI gap stays 0.135-0.153 and the head still under-counts speakers. The ICSI
  "win" over the teacher is the teacher's false alarms against our word-level labels (its ICSI error is 0.228 FA),
  and there is no no-teacher control, so it cannot be credited to distillation. Nemotron-3's card lists AMI train +
  dev in its training data, so its AMI-dev DER is also not a clean target.

## 5. End of turn

Benchmark: **eot-bench v2** (`research/archive/EOT_BENCH_V2.md`): AMI dev, 974 turns (236 floor-open, 738 floor-taken),
label-free causal enrollment of the primary speaker on streaming Sortformer tracks, every system's threshold
cross-fitted over leave-meetings-out folds to ≤ 5 % false cutoffs per turn (realised held-out FC 4.8-6.5 %; up to
12 % within the open stratum), horizons 6 s and 2 s, 1000 paired bootstraps over turns. **Floor-open** ends (nobody
else speaks afterwards) are the ones a voice agent must answer; **floor-taken** ends are where another person starts.
An earlier "the head beats the timeout" result (STAGE1) turned out to be an enrollment leak worth +45 points, which is
why this protocol exists.

### 5.1 AMI dev (`runs/baselines_turn.json`, `runs/completeness.json`, `runs/baselines_turn_icsi*.json`)

| system | label use | 6 s miss, all [CI] | 6 s miss, open [CI] | realised FC | 2 s miss all / open |
|---|---|---|---|---|---|
| **head OR dyn(Silero, head)** (ours, the shipped `hybrid_dyn`; picked on this table) | label-free | **58.2 [54.9, 61.5]** | 31.5 [25.6, 37.5] | 6.1 | 73.2 / 29.7 |
| head OR Silero timeout (ours, `hybrid_silero`) | label-free | 59.6 [56.3, 62.6] | 34.3 [28.6, 40.4] | 6.6 | 78.4 / 58.1 |
| hybrid: head OR Sortformer-primary timeout (ours) | label-free | 61.9 [58.7, 65.0] | 45.9 [39.3, 52.4] | 6.5 | 79.1 / 72.9 |
| head alone (trail6) | label-free | 66.5 [63.4, 69.7] | 57.3 [50.9, 63.9] | 5.3 | 79.8 / 75.0 |
| hybrid, agent-end arming + TitaNet voice following | **label stand-in for agent_end** | 51.6 [48.5, 54.7] | 34.0 [28.0, 40.6] | 6.2 | 87.6 / – |
| hybrid, agent-end arming, no embedding (`after_agent_arm`) | stand-in | 56.0 [52.5, 59.1] | 40.3 [33.2, 47.1] | 6.1 | 75.3 / – |
| hybrid with the oracle primary | oracle (upper bound) | 28.7 [25.8, 31.6] | 17.1 [12.6, 22.3] | 5.5 | 53.1 / – |
| NVIDIA Parakeet-Realtime-EOU-120M, P(EOU) | – | 70.1 [67.0, 73.0] | 42.9 [36.4, 49.5] | 5.3 | 81.6 / 59.5 |
| Pipecat: smart-turn v3.2 + Silero timeout | – | 72.6 [69.7, 75.3] | 26.7 | 5.2 | 83.4 / 39.6 |
| Silero VAD timeout | – | 72.7 [69.8, 75.4] | **26.7 [21.1, 32.4]** | 4.8 | 83.5 / 39.6 |
| timeout on the Sortformer primary (our server's `timeout` policy) | – | 74.8 [72.0, 77.5] | 46.1 [39.6, 52.6] | 5.7 | 92.5 / 81.1 |
| LiveKit text turn detector (EN / multilingual, ASR or oracle text) + timeout | – | 73.7-74.2 | 28.2-30.9 | 5.1-6.2 | 88.8-90.5 / – |
| LiveKit audio turn-detector-v1-mini + timeout | – | 74.5 [71.5, 77.1] | 28.8 | 5.1 | 89.8 / 60.1 |
| smart-turn v3.2 alone / LiveKit audio alone / LiveKit text alone | – | 98.9 / 95.5 / 90.4-91.9 | 99.6 / 89.6 / 85.8-88.9 | 4.3-5.5 | – |

Paired differences (6 s): hybrid − Silero **−10.8 [−14.2, −7.0]** all, **+19.1 [+11.1, +27.0]** open; hybrid −
Parakeet-EOU −8.2 [−11.8, −4.2] all, +3.0 [−5.7, +11.9] open; head OR dyn − Silero −14.5 [−17.4, −11.3] all, +4.8
[−1.6, +10.7] open. At 2 s Parakeet-EOU ties the hybrid overall (+2.4 [−1.6, +5.9]).

- **Verdict (AMI dev): better on all ends, equal (best rule) to worse (hybrid) on floor-open ends.** The margin is on
  floor-taken ends: a speaker-unaware detector sees no silence when another person starts talking. The best rule was
  chosen after seeing this table, so its AMI numbers are in-sample; ICSI (§5.2) is the held-out test.
- Out of the box (no threshold fitting), Pipecat's default misses 48.3 % but cuts users off in 49.8 % of turns; the
  LiveKit audio detector 48.6 % at 35.5 % FC; Parakeet-EOU's native emission 79.1 % at 3.0 %. On meeting audio the
  dedicated turn models add nothing over their own VAD timeout (smart-turn's in-turn vs after-end AUC is 0.58 on AMI,
  0.70 on LibriSpeech).
- The measured bottleneck is identity: label-free enrollment agrees with the true primary at the turn end 70 % of the
  time (33 % for first-voice binding); with the oracle primary the hybrid's misses fall from 61.9 to 28.7 %. Binding
  the primary at the agent's own end of speech (product-realistic; the label stand-in here) recovers half of that
  (51.6 % with TitaNet following, CI excludes 0), but fires late at 2 s (+8.5 points).

### 5.2 ICSI held-out (`runs/baselines_turn_icsi.json`, `runs/baselines_turn_icsi_dyn.json`, `frozen_ami`)

5 meetings never used for fitting (Bmr021, Bns001, Bmr013, Bmr018, Bro021), 1312 turns (216 open), thresholds frozen
from AMI.

| system | 6 s miss all [CI] | 6 s miss open [CI] | FC per turn | 2 s all / open |
|---|---|---|---|---|
| **head OR dyn(Silero, head)** | **79.5 [77.2, 81.7]** | 50.5 [44.5, 56.7] | 1.8 | 90.2 / 61.7 |
| head OR Silero | 81.7 [79.5, 83.9] | 62.2 [56.1, 68.7] | 2.4 | 91.5 / 88.7 |
| hybrid | 82.1 [79.8, 84.4] | 71.5 [65.7, 77.3] | 1.4 | 91.8 / 89.2 |
| head alone | 82.9 | 75.2 | 1.1 | – |
| Silero VAD timeout | 87.4 [85.5, 89.1] | **45.5 [38.4, 52.5]** | 3.6 | 92.2 / 58.5 |
| Sortformer any-speaker timeout | 93.4 | 68.6 | 0.8 | 99.4 / 96.7 |
| oracle-primary timeout (upper bound) | 9.8 [8.2, 11.6] | 3.0 | 4.4 | – |

Paired, head OR dyn − Silero: 6 s all **−7.9 [−10.0, −5.8]**, open +5.0 [−1.4, +11.5]; 2 s all −2.0 [−3.2, −0.8].
Head OR dyn − head OR Silero, open: −11.7 [−17.0, −7.0]; head OR Silero − hybrid, open: −9.3 [−14.7, −4.1].

- **Verdict (ICSI): the direction replicates.** Better on all ends, equal on open ends; absolute misses are much
  higher on ICSI (hesitations are 2-3x as frequent as in AMI, `research/archive/ICSI.md`) and the frozen thresholds land at
  1-4 % FC, under budget. Cross-fitting the dynamic rule on ICSI itself gives 62.0 % but at 14.8 % FC, so it is not a
  shippable operating point. Parakeet-EOU, smart-turn and LiveKit were **not run on ICSI**; "better than every
  dedicated detector" is an AMI-dev statement.

### 5.3 TurnBench dev (two-party calls, official scorer; `runs/turnbench_dev.json`, `runs/dyadic_bench.json`, `runs/dyadic_train.json`, `runs/turnbench_latency.json`)

38 conversations, 7.3 h, 1904 turn ends and 1063 negatives (pauses, backchannels); a detection counts in
[−0.25 s, +3 s]; operating point = highest recall at FP ≤ 0.10. The scorer reports no CIs. Rows marked "rescored"
are the authors' published dev predictions scored by us with the vendored MIT scorer.

| system | thresholds | recall | FP rate | P10 / P50 latency |
|---|---|---|---|---|
| **ours, predictive trigger (dyadic head c) OR per-channel Silero**, held-out halves pooled | fitted on the other half | **0.862** | 0.092 (one half 0.121) | 400 / 1137 ms |
| ours, predictive OR Silero, thresholds fitted on otoSpeech (the E2E Dp point) | oto → TB | 0.866 | 0.175 (over budget) | 238 / 800 ms |
| ours, best fixed reactive policy, cross-fit held-out | other half | 0.849 | 0.098 | – / 1316 ms |
| ours, hybrid (trail6 head OR per-channel Silero) | picked on TB dev (in-sample) | 0.835 | 0.080 | 760 / 1390 ms |
| ours, hybrid, AMI thresholds frozen | AMI | 0.465 | 0.256 | – / 456 ms |
| ours, head alone (trail6) | picked on TB dev | 0.752 | 0.099 | 237 / 1058 ms |
| Silero timeout per channel (measured) | picked on TB dev | 0.815 | 0.075 | 844 / 1399 ms |
| Parakeet-EOU posterior (measured) | picked on TB dev | 0.763 | 0.084 | 401 / 1097 ms |
| **VAP (rescored)** | authors' | 0.841 | **0.045** | **−34 / 463 ms** |
| ESPnet turn-taking, dual channel (rescored) | authors' | 0.836 | 0.074 | 411 / 895 ms |
| Kyutai semantic VAD (rescored) | authors' | 0.803 | 0.100 | 224 / 1024 ms |
| smart-turn v3 (rescored) | authors' | 0.754 | 0.100 | 684 / 1010 ms |
| P: Vox Maru v1 (leaderboard, test) | | 0.960 | 0.072 | – / 548 ms |
| P: VAP (test) | | 0.845 | 0.055 | – / 368 ms |

- **Verdict: worse than VAP (and ESPnet) on latency; better than smart-turn, Kyutai and Parakeet-EOU on recall.** The
  AMI-trained head transfers to two-party audio without retraining once the party channel is known, but no policy we
  fitted reaches P50 ≤ 700 ms at FP ≤ 0.10 (that costs FP ≥ 0.185); VAP's P10 is negative, ours is not. The
  predictive trigger (fire on predicted future silence) is the only head-driven policy that beats the reactive rules
  at matched FP, and it is still 674 ms behind VAP's median. The 0.835 / 0.752 rows are TurnBench-dev-selected
  operating points; with AMI thresholds frozen the hybrid is 0.465 recall at FP 0.256.

### 5.4 otoSpeech (real two-party calls; `runs/dyadic_train.json`, `runs/dyadic_bench_oto.json`)

60 dev conversations, 2175 human turn ends, eot-bench v2 protocol (≤ 5 % FC, 6 s). Head input is the oracle party
track (Silero activity of the human channel); oto labels are Silero segments, which flatters silence rules.

| system | 6 s miss [CI] | FC | P50 |
|---|---|---|---|
| **hybrid_energy (ours, trained on oto)** | **7.5 [6.5, 8.7]** | 3.9 | 1680 ms |
| hybrid_trail6 (ours, AMI-trained) | 8.5 [7.4, 9.7] | 5.2 | 1600 ms |
| primary-channel timeout (oracle channel) | 10.0 [8.8, 11.3] | 4.3 | 1680 ms |
| 16-conversation slice: Silero timeout on the mono mix / Parakeet-EOU posterior | 77.7 / 94.0 | – | – |

- **Verdict: better than a timeout on the same channel (−2.5 points); much better than speaker-unaware detectors on
  the mono mix**, which is not a like-for-like comparison (they were not given the channel split).

### 5.5 Semantic signals (negative)

A 124 K-parameter text-free completeness head matches smart-turn v3.2 on smart-turn's own clean clips (frame AUC 0.995
vs 0.994) at 0.1 ms per chunk, but adds nothing on meetings (pause-vs-end AUC 0.47 on AMI; smart-turn 0.39) or on
TurnBench (0.280 recall alone). YIELD / HOLD action tokens added to the transducer were never emitted (0 of 974
windows; as a posterior score 94.4 % missed); NVIDIA's own Parakeet-EOU fires from silence, not from the words (P50
emission lag 2.16 s after the last word). LiveKit's text model does not help even with oracle transcripts.

### 5.6 The user's voice print: the TS-VAD turn path (2026-09-28; `runs/improve_115m.json`, `runs/e2e_tsvad.json`)

**Offline.** A 0.26 M target-speaker VAD head on block 4 (the ASR pass the server already runs), conditioned on a 5 s
voice print from elsewhere in the meeting, tracks the enrolled speaker better than a diarizer column bound by the same
print: frame F1 0.743 vs 0.629 (AMI) and 0.882 vs 0.690 on held-out ICSI, where it also beats the label-chosen oracle
Sortformer column (0.794). Fed to the served turn head instead of the label-free Sortformer column, at the same ≤ 5 %
cross-fitted false-cut budget (6 s horizon):

| input → rule | AMI all [CI] | AMI open | ICSI all [CI] | ICSI open |
|---|---|---|---|---|
| Sortformer causal column → hybrid (shipped) | 61.9 [58.6, 65.3] | 45.9 | 68.5 [65.8, 71.2] | 55.2 |
| Sortformer oracle column → hybrid (upper bound) | 28.7 [26.2, 31.1] | 17.1 | – | – |
| **TS-VAD track → hybrid** | **39.3 [36.2, 42.8]** (P50 4160 ms) | 37.9 | **20.4 [18.2, 22.7]** (1920 ms) | 21.5 |
| **TS-VAD track → `hybrid_dyn`** (refitted point) | **34.2 [31.1, 37.5]** (3200 ms) | **17.9** | **18.7 [16.6, 21.0]** (1920 ms) | **13.2** |

The served path reproduces this exactly: streaming the benchmark's audio through `serve.Session` with the stored print
gives a TS-VAD track within 5.3e-4 and turn posteriors within 4.0e-3 of the offline ones, with the same 179 of 1830
frames above the firing threshold (scratch `improve/serve_check.json`). The paired CIs against the Sortformer rows are
pending the ~5 h CPU rebuild of those tracks (§11).

**Live.** System T = `serve --turn-input tsvad --diar-off --enroll explicit`, `hybrid_dyn` at its refitted point, the
user's 5 s print sent at connect, no diarizer pass; the 37 E2E clips were rebuilt from `runs/e2e_final.json` (69
Pipecat sessions, 64 of them two-party calls, 223 scored ends) and compared with the stored C / D records of
E2E_FINAL (same clips, protocol and scorer, run the day before on the same machine, not re-run). Pooled, paired clip
bootstrap:

| comparison | missed at 3 s | missed at 6 s | cut-ins per session |
|---|---|---|---|
| T vs **C** (product default, 1 s timeout) | 45.3 vs 34.5 %: **+10.8 [+5.6, +15.8]** | +9.4 [+4.1, +14.9] | 0.52 vs 0.93: −0.41 [−0.67, −0.12] |
| T vs **D** (`hybrid_dyn` + arming, Sortformer) | 45.3 vs 44.0 %: +1.3 [−3.6, +6.2] | +0.9 [−3.1, +5.1] | 0.52 vs 0.52: 0.00 [−0.17, +0.19] |
| Th (head OR 1 s timeout on P(target), post hoc) vs C | +11.7 [+5.5, +19.0] | +5.8 [+0.9, +11.9] | −0.22 [−0.45, +0.01] |

Server RTF: T median 0.34 (max 0.50), Th 0.29, peak RSS 1.6 GB, vs 0.80 / 3.6 GB for C / D (scratch
`e2e_tsvad/runs/*.jsonl`). On the 5 AMI windows T matches C's misses (20 %) with no cut-ins and Th answers all five
within 3 s at C's dead air (1763 vs 1683 ms), but n = 5.

- **Verdict: a meeting-room gain, not a two-party gain.** The pre-registered bar (fewer misses than C, no more
  cut-ins than D) is **not met**: T misses 10.8 points more than C. It **equals D** on misses and cut-ins, with lower
  median dead air on four of five sets, at **0.43x the compute** and without a diarizer. The offline gain was
  measured on AMI / ICSI meetings, where the heads were trained; 64 of the 69 live sessions are two-party calls where
  C's 1 s timeout on the user channel is already strong (6-23 % missed) and every head-gated rule pays about a second
  of dead air. The path ships **behind flags** (`--turn-input tsvad [--diar-off]`), for a product that holds its
  user's voice print and runs in rooms.

### 5.7 Latency against false cutoffs: a product knob (2026-09-28; `runs/eot_frontier.json`)

Stored scores only; TS-VAD 5 s-print track feeding the served head; for each false-cutoff budget the point with the
lowest median latency whose realised per-turn FC over all turns is within the budget. **In-sample** (the point is
chosen on the evaluation set): it shows the trade-off, the cross-fitted rows above are the held-out numbers.

| FC budget | AMI `hybrid_dyn` miss / open / P50 | ICSI `hybrid_dyn` miss / open / P50 | timeout on P(target), miss AMI / ICSI | Silero timeout, miss AMI / ICSI |
|---|---|---|---|---|
| 3 % | 45.0 / 25.9 / 4960 ms | 23.5 / 14.8 / 2320 ms | 68.5 / 57.9 | 74.9 / 88.6 |
| 5 % | 34.9 / 18.2 / 3280 ms | 20.4 / 10.2 / 2080 ms | 63.1 / 53.1 | 72.0 / 84.2 |
| 10 % | **22.4 / 12.7 / 2320 ms** | **13.3 / 7.4 / 1600 ms** | 56.3 / 32.9 | 66.7 / 78.5 |
| 15 % | 16.7 / 8.1 / 1840 ms | 10.6 / 6.5 / 1360 ms | 49.4 / 23.8 | 61.9 / 72.3 |

- A deployment that tolerates one false cut in ten turns instead of one in twenty gets 12 fewer missed ends on AMI and
  7 fewer on ICSI, answering ~1 s / ~0.5 s earlier at the median. The head-gated rule dominates both timeouts at every
  budget. The Sortformer-input frontier is missing (its scores were lost with the scratch; §11).

### 5.8 Two latency attempts that failed (2026-09-28; `runs/tsvad_turn.json`, `runs/hybrid_fast.json`)

- **A TS-VAD-trained predictive turn head** (0.25 M, top layer of the ASR pass, no second encoder pass, VAP-style
  future-activity bins, then an end-of-turn output) on the TS-VAD track: `hybrid_dyn` misses **+9.9 [+6.3, +13.2]**
  points on AMI and **+13.5 [+10.7, +16.3]** on ICSI against the served head on the same track; it answers 400 ms
  earlier on ICSI (P50 −400 [−480, −240] ms) but on fewer turns. Killed (two variants, pre-registered). A future-silence
  predictor fires inside AMI's long within-turn pauses, which the benchmark counts as false cuts; the served head's
  speaker-conditioned second pass and text features carry the rest.
- **`hybrid_fast`** (read the head at silence onset + 160-480 ms, fire above a cross-fitted threshold): P50 moves by
  −80 ms on AMI (3280 → 3200) and 0 on ICSI, for +0.4 to +2.6 points of FC; no point beats `hybrid_dyn` by ≥ 300 ms
  at matched FC, so nothing was pinned. The head is not confident 200 ms into a pause.

## 6. Language ID (FLEURS test, 17 languages, n = 2550; `runs/lid.json`, `research/archive/LID.md`)

| system | params | 1 s | 2 s | 3 s | 5 s | full utterance |
|---|---|---|---|---|---|---|
| **ours, head `lid_aug` on blocks 8-12** | 0.40 M | 59.2 % | 75.0 % | 81.6 % | 85.7 % | 89.3 % |
| **NVIDIA langid_ambernet** | 28.9 M | **83.9** | **95.1** | **98.2** | **99.2** | **99.5** |
| SpeechBrain VoxLingua107 ECAPA | 21.2 M | 81.1 | 94.6 | 97.3 | 98.5 | 99.3 |
| Whisper base | 74 M | 61.5 | 83.6 | 90.9 | 96.3 | 98.5 |
| Whisper tiny | 37.8 M | 54.9 | 78.2 | 86.9 | 93.7 | 97.0 |

- **Verdict: worse.** Accuracy CIs are about ±1.5 points at 75 % and ±0.3 at 99 %. On accented English (EdAcc, 257
  segments) the head calls 18.7 % English (the older `lid_head` 31.1 %) against AmberNet 67.7 % and Whisper base
  95.3 %. The head costs 0.001 RTF; AmberNet 16-81 ms per call (1-8 s of audio). Language ID ships **off by
  default**; `serve --lid ambernet` runs NVIDIA's model (NGC terms) and announces a language after 1 s of speech at
  posterior 0.9 (96.8 % correct first call at 2.16 s).
- Parakeet-TDT v3, the hybrid's final ASR, has no language token either; its transcript's language is measured in §6.1.

### 6.1 Language ID read from the transcript (2026-09-28; `runs/hybrid_asr.json` `lid_text`, `fleurs`)

TDT v3 has no language token, so its language decision is read off its transcript with langid.py. On FLEURS-17 its
10 supported languages (of our 17) are transcribed at 3.9-10.8 % WER (the other 7 come back as fluent text in a wrong
language, WER ≥ 99 %).

| system | supported set, full utterance | supported set, 2 s from onset | unsupported languages |
|---|---|---|---|
| langid.py on the TDT v3 transcript | **100 %** (1260 / 1260) | 91.7 % (275 / 300) | 0 % (no flag) |
| AmberNet on the same utterances, restricted to the same 10 | 99.9 % | **95.1 %** | – |

- **Verdict: equal on full utterances, worse at 2 s (−3.4 points), free with the transcript.** Out of set it fails
  silently: a Turkish or Hebrew utterance comes back as confident-looking text in another language. AmberNet stays
  the recommendation when language ID matters (`--lid ambernet`, off by default).

## 7. Product: Pipecat and LiveKit end to end (`research/E2E_FINAL.md`, `runs/e2e_final.json`)

Seven systems answered the same recorded conversations at 1x through the real frameworks and were scored by when the
agent would have started to answer (Pipecat: the `LLMContextFrame` reaching the LLM; LiveKit: the committed user
turn). Clips: the 5 AMI windows of `INTEGRATION.md` §4 (mono), 16 TurnBench dev and 16 otoSpeech dev conversations
(35-57 s each) as a mono mix and as the user's own channel; 114 scored user turn ends. **Dead air** = response − true
end (median / P90 over ends answered within 6 s); **missed** = no response within 3 s / 6 s; **cut-in** = a response
inside a user turn. Every system × framework × condition × clip is one timed session (345 per framework, 0 errors).

| id | system |
|---|---|
| A | Pipecat default local stack: Silero VAD + smart-turn v3.2 (3 s fallback) + faster-whisper small |
| B | LiveKit default local stack: Silero VAD + `EnglishModel` turn detector + faster-whisper small behind `StreamAdapter` (evaluation only, LiveKit Model License) |
| C | ours, product default: `stage1_served.afm` + Sortformer v2 (0.32 s), `turn_policy: timeout` 1000 ms on the label-free primary |
| D | ours, opt-in rules: `hybrid_dyn` + `--enroll after_agent_arm` (`agent_end` = the other party's labelled turn end, a stand-in for the TTS-end event) |
| CN / DN | C / D with Nemotron-3-Diarization 100M (max pooling, 4 of 8 columns) instead of Sortformer v2 |
| Dp | ours, predictive trigger OR per-channel Silero, **offline** scorer on the two-channel input (not in `serve.py`) |

Values are Pipecat / LiveKit (A only in Pipecat, B only in LiveKit). Dead air is over answered ends, so a system that
misses more can show a lower median.

**AMI, 5 windows, mono** (Pipecat / LiveKit; Dp offline on the two-channel input)

| system | dead air median ms | dead air P90 ms | missed 3 s | missed 6 s | cut-ins per min | WER | first text ms |
|---|---|---|---|---|---|---|---|
| A Pipecat default | 2548 / – | 4605 / – | 60 % / – | 20 % / – | 7.46 / – | 44 % / – | 3689 / – |
| B LiveKit default | – / 2660 | – / 4280 | – / 40 % | – / 0 % | – / 3.39 | – / 43 % | – / 9190 |
| C ours timeout | 1683 / 1660 | 3042 / 2224 | 20 % / 0 % | 0 % / 0 % | 1.36 / 3.39 | 39 % / 39 % | 828 / 810 |
| CN ours+Nemotron-3 timeout | 1903 / 1380 | 3965 / 1776 | 40 % / 0 % | 0 % / 0 % | 1.36 / 2.04 | 39 % / 39 % | 819 / 810 |
| D ours hybrid_dyn+arm | 2561 / 2420 | 2990 / 2956 | 20 % / 20 % | 0 % / 0 % | 0.00 / 0.00 | 39 % / 39 % | 827 / 810 |
| DN ours+Nemotron-3 hybrid_dyn+arm | 2322 / 2340 | 2530 / 2500 | 20 % / 0 % | 20 % / 0 % | 0.00 / 0.00 | 39 % / 39 % | 821 / 810 |

**TurnBench dev, 16 clips, mono mix** (Pipecat / LiveKit; Dp offline on the two-channel input)

| system | dead air median ms | dead air P90 ms | missed 3 s | missed 6 s | cut-ins per min | WER | first text ms |
|---|---|---|---|---|---|---|---|
| A Pipecat default | 1054 / – | 4576 / – | 75 % / – | 62 % / – | 2.96 / – | 26 % / – | 2598 / – |
| B LiveKit default | – / 1714 | – / 2993 | – / 70 % | – / 66 % | – / 1.07 | – / 23 % | – / 11288 |
| C ours timeout | 1346 / 1312 | 3355 / 1933 | 68 % / 34 % | 62 % / 30 % | 1.15 / 4.52 | 27 % / 27 % | 1005 / 999 |
| CN ours+Nemotron-3 timeout | 1338 / 1271 | 5431 / 1488 | 71 % / 30 % | 64 % / 29 % | 1.07 / 3.62 | 27 % / 27 % | 1004 / 999 |
| D ours hybrid_dyn+arm | 1696 / 1628 | 3303 / 3278 | 70 % / 54 % | 64 % / 46 % | 0.49 / 1.97 | 27 % / 27 % | 1005 / 999 |
| DN ours+Nemotron-3 hybrid_dyn+arm | 1944 / 1755 | 2574 / 2617 | 66 % / 54 % | 62 % / 52 % | 0.41 / 0.58 | 27 % / 27 % | 1006 / 999 |
| Dp ours predictive (offline) | 786 | 1467 | 21 % | 20 % | 1.73 | – | – |

**TurnBench dev, 16 clips, user channel** (Pipecat / LiveKit; Dp offline on the two-channel input)

| system | dead air median ms | dead air P90 ms | missed 3 s | missed 6 s | cut-ins per min | WER | first text ms |
|---|---|---|---|---|---|---|---|
| A Pipecat default | 2099 / – | 3186 / – | 54 % / – | 32 % / – | 2.14 / – | 16 % / – | 2644 / – |
| B LiveKit default | – / 1350 | – / 3080 | – / 36 % | – / 25 % | – / 0.99 | – / 10 % | – / 5041 |
| C ours timeout | 1353 / 1330 | 1617 / 1500 | 23 % / 23 % | 20 % / 23 % | 1.32 / 2.22 | 17 % / 17 % | 942 / 936 |
| CN ours+Nemotron-3 timeout | 1285 / 1256 | 1406 / 1378 | 20 % / 18 % | 16 % / 18 % | 1.32 / 1.32 | 17 % / 17 % | 978 / 966 |
| D ours hybrid_dyn+arm | 2311 / 2308 | 2630 / 2562 | 38 % / 39 % | 32 % / 36 % | 0.74 / 0.82 | 17 % / 17 % | 942 / 936 |
| DN ours+Nemotron-3 hybrid_dyn+arm | 2256 / 2234 | 2573 / 2460 | 38 % / 39 % | 32 % / 36 % | 0.74 / 0.91 | 17 % / 17 % | 944 / 936 |
| Dp ours predictive (offline) | 786 | 1467 | 21 % | 20 % | 1.73 | – | – |

**otoSpeech, 16 clips, mono mix** (Pipecat / LiveKit; Dp offline on the two-channel input)

| system | dead air median ms | dead air P90 ms | missed 3 s | missed 6 s | cut-ins per min | WER | first text ms |
|---|---|---|---|---|---|---|---|
| A Pipecat default | 2072 / – | 4311 / – | 62 % / – | 34 % / – | 2.84 / – | – / – | 2828 / – |
| B LiveKit default | – / 1310 | – / 3286 | – / 45 % | – / 36 % | – / 2.13 | – / – | – / 4610 |
| C ours timeout | 1372 / 1350 | 3553 / 3414 | 42 % / 32 % | 32 % / 21 % | 1.26 / 3.86 | – / – | 1077 / 1075 |
| CN ours+Nemotron-3 timeout | 1302 / 1290 | 3052 / 1674 | 38 % / 17 % | 30 % / 15 % | 2.05 / 3.23 | – / – | 1072 / 1065 |
| D ours hybrid_dyn+arm | 2252 / 2215 | 4290 / 3458 | 57 % / 55 % | 49 % / 47 % | 0.79 / 2.37 | – / – | 1078 / 1075 |
| DN ours+Nemotron-3 hybrid_dyn+arm | 2162 / 1960 | 3379 / 2420 | 43 % / 34 % | 36 % / 30 % | 0.47 / 0.87 | – / – | 1072 / 1065 |
| Dp ours predictive (offline) | 1080 | 1367 | 19 % | 17 % | 1.50 | – | – |

**otoSpeech, 16 clips, user channel** (Pipecat / LiveKit; Dp offline on the two-channel input)

| system | dead air median ms | dead air P90 ms | missed 3 s | missed 6 s | cut-ins per min | WER | first text ms |
|---|---|---|---|---|---|---|---|
| A Pipecat default | 1114 / – | 3159 / – | 43 % / – | 13 % / – | 2.21 / – | – / – | 2864 / – |
| B LiveKit default | – / 1350 | – / 2368 | – / 13 % | – / 11 % | – / 1.58 | – / – | – / 3375 |
| C ours timeout | 1393 / 1385 | 1593 / 2147 | 6 % / 9 % | 4 % / 6 % | 1.26 / 1.50 | – / – | 1187 / 1180 |
| CN ours+Nemotron-3 timeout | 1271 / 1250 | 1422 / 1441 | 6 % / 6 % | 4 % / 6 % | 1.34 / 1.34 | – / – | 1184 / 1175 |
| D ours hybrid_dyn+arm | 2343 / 2320 | 2843 / 2820 | 13 % / 13 % | 8 % / 8 % | 0.87 / 0.87 | – / – | 1186 / 1180 |
| DN ours+Nemotron-3 hybrid_dyn+arm | 2302 / 2280 | 3122 / 2553 | 19 % / 13 % | 8 % / 9 % | 0.63 / 0.63 | – / – | 1182 / 1175 |
| Dp ours predictive (offline) | 1080 | 1367 | 19 % | 17 % | 1.50 | – | – |

Paired deltas (bootstrap over clips; n = 5 on AMI makes those CIs wide), compute and clean-run checks are in
`E2E_FINAL.md` §5-7. The essentials:

- **vs Pipecat default (A):** C has fewer cut-ins on every set (per clip −1.8 [−3.2, −0.4] AMI, −1.38 [−2.19, −0.63]
  TurnBench mix, −1.25 [−2.56, −0.13] oto mix; user-channel CIs include 0) and fewer misses at 3 s on every set (−7
  to −40 points; TurnBench user −30 [−51, −9], oto user −38 [−53, −20]); median dead air −0.7 to −0.9 s on three
  sets, +0.3 s on two, all CIs including 0. A's P90 dead air is 3.2-4.6 s everywhere.
- **vs LiveKit default (B):** dead air comparable (C − B +40, +35, −402, −20 ms on the two-party cells; CN 20-443
  ms below B, two of four CIs excluding 0); fewer misses on every set (TurnBench mix −36 [−55, −17] points); cut-ins
  equal on the oto user channel (−0.06 [−0.69, +0.50] per clip), higher on the TurnBench user channel (+0.94 [+0.31,
  +1.63]) and much higher on the mono mixes (+1.4 to +2.6 per clip), where B misses 45-70 % of ends. B's transcripts
  are better (10 vs 17 % WER on the user channel) and arrive 3-11 s late.
- **D / DN vs C:** +0.3 to +1.0 s median dead air and +2 to +23 points of misses on the mixes for near-zero cut-ins
  (0 on AMI in both frameworks; DN 0.4-0.9 per minute elsewhere). This is the INTEGRATION §8 finding on 37 clips.
- **Nemotron-3 vs Sortformer v2 (CN − C):** dead air 8-135 ms lower on the two-party sets (6 of 8 CIs exclude 0),
  misses equal or lower, cut-ins mixed; server RTF 0.63-0.65 vs 0.79-0.81, RSS 1.48 vs 3.58 GB, server CPU 53-73 %
  vs 77-102 % of a core. **Recommendation: default diarizer.**
- **Dp:** the lowest dead air (786 / 1080 ms) at 19-21 % missed, but offline, two-channel, and at an operating point
  that scores FP 0.175 on TurnBench dev.
- **Compute by component** (RTF wall at 2 threads): A = Silero 0.018 + smart-turn 0.017 + Whisper 0.223 (0.43 CPU s
  per audio s in total); B = Silero 0.034 + Whisper 0.132 + EOU 0.002 (0.28); ours = ASR pass 0.15 + turn pass 0.15 +
  diarizer 0.50 (Sortformer) or 0.30 (Nemotron-3) + Silero branch ≤ 0.004 = 0.80 / 0.64 server total (1.02 / 0.71
  CPU s per audio s). Ours is 1.7-3.6x the CPU of the default stacks and keeps up at 1x with backlog ≤ 0.7 s in all
  but two sessions.

### 7.1 Server performance pass (2026-09-28; `runs/perf.json`, `research/archive/PERFORMANCE.md`)

Four exact CPU fast paths, on by default (`--perf default`; `--perf none` restores the old path): column-major
`nn.Linear` weights (Accelerate's NN GEMM path), a cached relative-position projection, one shared subsampling for
the ASR and turn passes, and a cached RNNT joint projection. Acceptance rule: identical protocol decisions on the 5
AMI windows of E2E_FINAL, and lower RTF or p95 in an interleaved A/B (both engines in one process, alternated per
clip, so both see the same load).

| measurement (Nemotron-3, 2 threads) | before (`--perf none`) | after (`--perf default`) |
|---|---|---|
| interleaved A/B, 5 AMI windows (load 7-8) | RTF 1.39, chunk p95 656 ms | RTF 0.86 (**0.62x**), CPU 0.67x, chunk p95 0.38x; ASR pass 0.70x, turn pass 0.42x, diarizer 0.71x |
| messages (1313) | – | 0 decision differences (turn_end, final, partial, primary); probabilities within 1e-5 (eot), VAD identical |
| direct runs, 5 AMI windows | RTF 0.84 (load 4.7) | RTF 0.45-0.51 (load 3.5-5); chunk p50 / p95 79 / 125 ms |
| steady state (120 s window, full speaker cache) | RTF 1.38 | RTF 0.62 (the diarizer's 98.7 M transformer head is 0.38 of it) |
| Sortformer v2 0.32 s, 5 AMI windows | – | RTF 0.61, peak RSS 3.58 GB |
| two concurrent sessions, 60 s each | – | aggregate RTF 1.31: not real time (latency grows 17 s per minute) |

- **Verdict: better, same outputs.** Applying the A/B ratio to E2E_FINAL's quiet-machine baseline gives ~0.40
  (Nemotron-3) and ~0.50 (Sortformer v2) for the whole server; the target ≤ 0.35 was not reached. What remains is
  arithmetic-bound (the diarizer head over a 380-row cache every 240 ms, and two 17-layer encoder passes); one
  session per 2 threads is what one worker sustains; batching sessions is the route to more (not done). Rejected:
  `fast_subsample` (not exact, 2.6 % gain; opt-in `--perf all`), `torch.compile` (slower), int8 in torch (no engine on
  this wheel), `inference_mode` (no gain).

### 7.2 Which RTF to quote (`research/archive/PERFORMANCE.md` §1)

RTF = compute time / audio time, and the same model has several correct values depending on what is timed.

| figure | what it measures | source | quote it for |
|---|---|---|---|
| 0.015 / 0.022 | the 115M model only, **offline**, whole utterances in batches of 4 (LibriSpeech / AMI) | `runs/final_asr.json` `rtf_cpu2` | the model's raw cost against other ASR models in the same table |
| 0.065 | Parakeet-TDT v3, one finished turn at a time (batch 1), machine under load | `runs/hybrid_asr.json` `english/results/tdt_v3/ami/rtf_logged` | the extra CPU of the per-turn transcript (off the streaming path) |
| 0.15 + 0.15 | inside the live server: the ASR pass and the speaker-conditioned turn pass | `runs/e2e_final.json` `components_live/*/server_asr`, `server_turn` | the streaming core's live cost |
| 0.50 / 0.30 | the diarizer alone, live: Sortformer v2 (0.32 s) / Nemotron-3 | `runs/e2e_final.json` `server_diar` | which diarizer to ship |
| **0.80 / 0.64** | **the whole server, live, 1x, per session** (Sortformer / Nemotron-3), before the fast paths; process CPU 1.02 / 0.71 CPU-s per audio-s | `runs/e2e_final.json` `server_total`, `_process` | **the product number for the E2E results** |
| 0.45-0.51 / 0.62 | the whole server after the fast paths, in process on the 5 AMI windows / a 120 s window, load 3.5-5 | `runs/perf.json` | the current server cost |
| 0.34 | the whole server on the TS-VAD path with the diarizer off (T), live | scratch `e2e_tsvad/runs` | the known-user turn path |
| 0.16 | the on-device runtime study: 1 thread, random weights, no diarizer, no turn pass | `research/archive/ONDEVICE.md` (no JSON) | "can the encoder + heads run on one core" only |
| 4.6 / 1.68 / 0.24 | the 0.6B streaming encoder: streaming chunk time on CPU (2026-09-26 probe) / `StreamingSession` on CPU / on MPS | `runs/enc_0p6b.json`, `runs/hybrid_asr.json` `core_0p6b/rtf` | why the 0.6B is not the CPU core |

**Rule:** quote the whole-server live figure with its thread count, diarizer and machine load for "what does a
session cost", the offline batch figure only next to other models' offline figures, and chunk-time p95 plus backlog
(not RTF) for "does it keep up". Never quote the 0.16 or the quickstart's loaded-machine 0.85 / 1.0 for the server.

### 7.3 Robustness (`research/archive/BULLETPROOF.md`, `tests/test_bulletproof.py`)

36 failure modes a deployment can hit (bad audio: NaN / Inf, extreme amplitudes, odd block sizes, minutes of silence,
wrong sample rates; malformed or late protocol messages, floods, disconnects, 12 concurrent sessions; model failures:
NaN head outputs, missing Silero, a dead / failing / hanging final-ASR worker; load shedding; the three clients),
pinned by **48 tests** (tiny random models, ~1 min, a 30-minute soak included). Three were defects in shipped code: a
late `turn_policy` was accepted silently when audio had arrived before the session object existed; non-finite input
samples poisoned the encoder caches for the rest of the session; a hung or dead `--final-asr` worker made `end` wait
forever. Load shedding is now entered only while the session's own recent RTF is ≥ 0.9, so a burst that a fast server
drains by itself is never shed and paced and unpaced clients measure the same thing; no default policy's measured
behaviour changed. Still fragile: one compute thread for all sessions, and the soak is 30 min on tiny models, not an
hour on the served checkpoints.

## 8. Recommended strongest lightweight stack

| role | component | why | cost (CPU, 2 threads) |
|---|---|---|---|
| streaming core | the 115M encoder + our heads (`runs/stage1_served.afm`) at 160 ms chunks, 80 ms lookahead: VAD, turn head, speaker head, RNNT partials; `--perf default` (the exact fast paths, on by default) | one pass serves four tasks; best streaming VAD measured; partials 0.8-1.2 s after onset; the 0.6B core was not adopted (§8.1) | ASR pass 0.14 + turn pass 0.11 after the fast paths (0.15 + 0.15 before); ~1.0-1.1 GB steady state |
| transcript | **CPU: Parakeet-TDT 0.6B v3 on each finished turn** (`serve --final-asr tdt_v3`, worker process); `--asr-lookahead 13` as the zero-extra-memory option. **GPU: the 0.6B streaming model** (streaming partials at ~11 % AMI WER, RTF 0.24 on MPS; not yet wired into `serve.py`) **or TDT v3** (`--final-asr-device cuda`) | 9.7 % vs 24.4 % AMI WER for TDT v3; [70,13] −1.4 points; 0.6B 11.2 % at [70,1] | TDT: RTF 0.065 per turn, ~2.5 GB; lookahead: ~+10-20 % encoder time |
| diarizer | **Nemotron-3-Diarization 100M, max pooling, all 8 columns** (`audioforge-serve --diarizer nemotron3`; the launcher no longer cuts it to 4) with **`--shed-diar hold`** (launcher default) | lower dead air than Sortformer v2, RTF 0.30 vs 0.50, 1.5 vs 3.6 GB; 8 columns fix rooms of 5-8 people (§4.1); hold keeps speaker labels under load instead of collapsing to speaker 0 | RTF 0.20-0.39 after the fast paths |
| rooms (multi-party) | `--diar-labels registry` (+ `--diar-embed titanet` when TitaNet-L is downloaded) and session config `turn_policy: "timeout_any"` | per-final speaker accuracy 0.74 → 0.83 (0.88 with TitaNet) on 4-6-speaker clips, finals no longer glue several people's turns; worse than the column label on short few-speaker windows, hence opt-in | registry with the served head: free; TitaNet +0.04 RTF, +91 MB |
| turn policy | two-party: `timeout` 1000 ms (responsiveness) or `hybrid_dyn` (no cut-ins); multi-party rooms: `hybrid_dyn` + `after_agent_arm` with the TTS-end event | §7 | Silero branch < 0.005 |
| known user (voice print held by the product) | **behind flags:** `--turn-input tsvad --diar-off --enroll explicit` (print sent as `{"type": "enroll", "embedding": [...]}`), or `--enroll after_agent_arm` to take it live | equals D (no-cut-in rule) on misses and cut-ins at 0.43x the compute and no diarizer; 61.9 → 39.3 % misses on AMI meetings offline; but +10.8 points of misses at 3 s vs the default on two-party calls (§5.6), so not a default | server RTF 0.34, 1.6 GB |
| enrollment (default path) | `--enroll after_agent_arm` (bind the first voice after the agent stops, follow with the block-4 head); TitaNet-L (`--titanet`) when a real voice print is available | −5.9 to −10.3 points of misses at 6 s on AMI (label stand-in) | head: free; TitaNet-L 22 M, per-stride embedding |
| language ID | AmberNet (`--lid ambernet`), off by default; the TDT v3 transcript's language (langid.py) as a free check inside v3's 25 languages | 95 % at 2 s vs our head's 75 % and the transcript's 91.7 % | 16-81 ms per call |
| deployment | GPU in production, lightweight-first: the whole server runs on 2 CPU threads at RTF 0.45-0.64 (after / before the fast paths), one session per 2 threads | | |
| next step | the **0.6B → 115M distillation package** (`scripts/research/distill_0p6b_to_115m/`, smoke-tested): AMI + ICSI train + a LibriSpeech anchor (~170 h), teacher transcripts + encoder regression, WER gate as kill switch, no-teacher control; target AMI streaming WER ≤ 17 % with LibriSpeech ≤ 2.59 % | would bring most of the 0.6B's meeting gain to the 115M's cost, keeping every head | **~6 A100 hours** (estimate from the package README; ~5.5-6 h wall on two RTX 5090) |

Not in the stack: our diarization head (0.394 DER) and its distilled variant, our LID head, encoder fine-tuning, YIELD
tokens, the completeness head, the TS-VAD-trained predictive turn head, `hybrid_fast`, the 0.6B as the core, the
decoder-adapted RNNT head (AMI-specific gain).

### 8.1 Should the 0.6B streaming model replace the 115M core on GPU? Not adopted (2026-09-28)

The user's rule, pre-registered: adopt `nemotron-speech-streaming-en-0.6b` as the core only if **every** category
improves over the 115M, a category counting as improved when its paired delta favours the 0.6B with a CI excluding 0;
"equal within noise" is **not shown to improve**, and any category worse beyond noise means "not adopted". Heads were
trained with identical recipes on both encoders (VAD: the served recipe, 3000 steps; speaker: the crop-level recipe of
§3.1, block 4 vs block 11; turn: the variant-(b) head of §5.8 on each core's top layer with the TS-VAD track, weaker
than the served kernel head on both, so this compares cores, not products).

| category | 115M core | 0.6B core | 0.6B − 115M [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER AMI-200 [70,1] / [70,13] | 24.43 / 23.02 % | **11.16 / 9.81 %** | **−13.27 [−15.48, −11.23]** / **−13.21 [−15.48, −11.19]** | better |
| WER ICSI-200 [70,1] / [70,13] | 27.29 / 24.78 % | **14.35 / 12.39 %** | **−12.94 [−15.11, −10.90]** / **−12.39 [−14.95, −10.13]** | better |
| WER LibriSpeech-200 [70,1] / [70,13] | 2.29 / 1.92 % | 2.20 / 2.03 % | −0.09 [−0.46, +0.28] / +0.11 [−0.23, +0.47] | equal (not shown to improve) |
| VAD F1 AMI dev / ICSI dev (identical head recipe; scratch `hybrid_asr/core_0p6b_rtf.json`) | 0.9329 / 0.9039 | **0.9398 / 0.9079** | **+0.0069 [+0.0018, +0.0120]** / **+0.0040 [+0.0004, +0.0071]** | better |
| first partial, AMI-200 / ICSI-200 (median audio time) | 1.12 / 1.12 s | 1.12 / 1.12 s | 0.00 [−0.16, +0.16] / 0.00 [0.00, +0.16] | equal |
| speaker EER within meeting, AMI n = 200 / ICSI n = 200 | 17.4 / 4.0 % | 16.2 / 2.9 % | −1.2 [−4.1, +1.9] / −1.1 [−2.7, +0.7] | equal |
| eot-bench miss 6 s, AMI all / open, head alone | 55.3 / 61.3 % | 56.5 / 57.6 % | +1.1 [−1.5, +4.0] / −3.8 [−10.3, +2.6] | equal |
| **eot-bench miss 6 s, AMI all, `hybrid_dyn`** | 44.1 % (P50 2960 ms) | 46.9 % (P50 3920 ms) | **+2.8 [+0.1, +5.3]** | **worse** |
| eot-bench miss 6 s, AMI open, `hybrid_dyn` | 24.7 % | 23.4 % | −1.2 [−6.3, +3.8] | equal |
| eot-bench miss 6 s, ICSI all / open, head alone | 34.0 / 49.5 % | **29.0 / 36.3 %** | **−4.9 [−7.1, −2.7]** / **−13.2 [−19.5, −6.9]** | better |
| eot-bench miss 6 s, ICSI all / open, `hybrid_dyn` | 32.2 / 37.9 % | **26.9 / 25.8 %** | **−5.3 [−7.4, −3.1]** / **−12.1 [−17.8, −6.9]** | better |
| TurnBench recall at FP ≤ 0.10 | – | – | not measured (the AMI row already decides) | – |
| cost: CPU 2 threads, ms per 160 ms chunk (RTF) / MPS RTF | 83.6 (0.52) / 0.08 | 266.1 (**1.68**, not real time) / 0.24 | – | 3.2x the CPU |

- **Verdict: not adopted.** One category is **worse** beyond noise (AMI eot-bench with `hybrid_dyn`, +2.8 [+0.1, +5.3]
  points, and a later median answer), and five are only **equal** (LibriSpeech WER at both lookaheads, first partial,
  speaker EER on AMI and ICSI, the AMI head-alone and open-floor turn rows): equal is not better under the rule. The
  0.6B's gain is in *what* is said (12-13 WER points on meetings, at the per-turn TDT v3's level already at 160 ms), not
  in *when* a turn ends.
- **GPU recommendation:** keep the 115M core for the live heads and use the 0.6B as the **streaming transcript model**
  on GPU (MPS RTF 0.24, 3.2 GB of GPU memory), or TDT v3 per turn; or move its accuracy into the 115M by distillation
  (§8 next step).

## 9. Cost of what was measured

All runs on one Apple M5 laptop (CPU 2 threads for every measured number, MPS for training the heads): head training
1-25 min per run (TS-VAD 2.5-6 min, turn heads ~20 min per 2000 steps), eot-bench scoring ~1 h per 974-window set,
the E2E comparison 2 × 345 sessions at 1x (≈ 5 h of audio per framework plus setup and guard waits). No GPU
time was bought. 2026-09-28 added ~2 h of MPS head training (TS-VAD turn heads, speaker heads, two
diarization students, the decoder adaptation, the 0.6B VAD heads), ~5 h of live sessions (69 per TS-VAD system, the
diarization clips) and CPU scoring; still no GPU. Data: AMI, ICSI (LDC, licensed), LibriSpeech, FLEURS, EdAcc, TurnBench dev (evaluation only),
otoSpeech dev (never trained on for the reported runs), Behavior-SD, DailyTalk. Published-model licences: the
encoder, Sortformer v2, TitaNet-L, parakeet-ctc / -tdt are CC-BY-4.0; Nemotron-3-Diarization OpenMDW-1.1; MarbleNet,
Parakeet-EOU and the 0.6B streaming encoder NVIDIA Open Model License; AmberNet NGC terms; LiveKit's turn models the
LiveKit Model License (evaluation only). The repository's own code carries an Apache-2.0 `LICENSE` added before release,
not yet confirmed by the maintainer (`docs/RELEASE_CHECKLIST.md`).

## 10. Negative results and what they rule out

| experiment | result | what it rules out |
|---|---|---|
| Fine-tuning the encoder (top 6 blocks, lr × 0.05, KL anchor) | LibriSpeech WER 2.05 → 4.31 % in 500 steps | cheap multitask fine-tuning of this encoder on our data; everything stays in heads |
| One encoder + 5 heads vs 5 models at matched steps (synthetic) | 4x fewer FLOPs, but TDT WER 51.7 vs 0.26 % at 1500 steps | "no accuracy cost" for joint training at a matched budget |
| Diarization-head layer routing (blocks 2-12, mixes, 2 seeds) | best −0.017 to −0.030 DER vs a 0.19 gap to Sortformer | routing as the fix for our diarization head |
| Turn head on mid blocks 4-12 | open-end misses +9.6 [+4.2, +14.6] points at 2 s; < 1 chunk earlier on TurnBench | "earlier layers make the turn head fire earlier" |
| Gated encoder (skip compute in silence) | re-priming needs 420 frames (34 s); ~15x net FLOP loss; 8-17 % wall saving in the Python path only | saving encoder compute in silence |
| NVIDIA 0.6B streaming encoder | ASR 2.20 vs 2.29 %, VAD equal, speaker probe 30 vs 30 % EER on AMI (20 vs 28 % on ICSI), RTF 4.6 on CPU | "a bigger encoder recovers speaker identity" at this cost; a small ICSI gain remains untested with a trained head |
| More identities under AAM (251 LibriSpeech speakers) | within-meeting EER 28.8 vs 19.1 % | identity count alone as the speaker lever; the teacher objective is what helped |
| Label-free voice enrollment (first or dominant voice; our head or TitaNet) | +22 points of misses vs activity-based enrollment | picking the primary by voice alone, without a product event such as the agent's TTS end |
| YIELD / HOLD tokens in the transducer | never emitted; run 1 tripped the WER gate (2.3 → 85 %) | semantic end tokens as a head-only retrain on a frozen encoder |
| Completeness (semantic) head | matches smart-turn on clean clips; no gain on AMI or TurnBench | "semantic completeness is the missing signal" on meeting audio |
| Hybrid rules in the product | 0 cut-ins but +0.3 to +1.0 s median dead air; the head never fires inside the first second of silence | the bench's miss-rate gain translating into lower dead air |
| State contamination after a wrong speaker binding | abs Δp 0.12 → 0.04 over 2 s (half-life 0.8-1.9 s); it lives in the head's GRU; 12-35 % of turns change outcome | ignoring binding errors; the fix is a 15 s replay (~208 ms on CPU), not a learned repair |
| LID head on the English encoder | 75 % at 2 s vs 95 % AmberNet; accented English 19-31 % "en" | LID from the shared encoder with FLEURS-scale supervision |
| Multi-horizon (future activity) targets for the EOT head | no gain on eot-bench; the energy input helped only in-domain | "predict the future" as a free improvement of the reactive head |
| TS-VAD turn path served live as a default (2026-09-28) | +10.8 [+5.6, +15.8] points of misses at 3 s vs the product default on 69 mostly two-party sessions; equal to the no-cut-in rule at 0.43x compute | "the offline meeting gain carries to two-party calls"; the path stays behind flags |
| TS-VAD-trained predictive turn head, no second encoder pass (2026-09-28) | +9.9 [+6.3, +13.2] AMI / +13.5 [+10.7, +16.3] ICSI points of misses vs the served head on the same track | future-activity prediction as the eot-bench decision signal; dropping the speaker-conditioned second pass |
| `hybrid_fast` (head read at silence onset + 160-480 ms) (2026-09-28) | P50 −80 ms AMI, 0 ICSI, +0.4-2.6 points of FC | a ≥ 300 ms latency gain from the rule alone; the head is not confident early in a pause |
| Distilling Nemotron-3 into a 1.9 M head on the frozen encoder (2026-09-28) | AMI-64 DER 0.367-0.385 vs teacher 0.232; count error −0.66 vs −0.14 | dropping the second diarizer model with a head-only student at this data scale |
| 0.6B streaming encoder as the core (2026-09-28, full recipe) | meeting WER −13 points, but AMI `hybrid_dyn` misses +2.8 [+0.1, +5.3]; five categories equal | "a bigger core improves every task" (§8.1) |

## 11. Unfinished work (honest status)

- **Paired CIs of the TS-VAD rows against the Sortformer rows** (§5.6). The per-turn outcomes of the Sortformer-column
  systems (2026-09-27) were lost with the scratch in a reboot; `scripts/research/tsvad_chain.sh` rebuilds the tracks
  and scores (resumable; stopped at 62 / 974 AMI windows), **about 5 h of CPU**. The same rebuild gives the
  Sortformer-input rows of the latency frontier (§5.7) and of `hybrid_fast` (§5.8).
- **Improvement experiments 4 and 5** (pre-registered in `research/archive/IMPROVEMENTS.md`): enrolment quality (print length
  1.5-10 s, live after_agent_arm prints, refresh; driver `tsvad_enroll.py` prepared) and meeting-ASR partials from the
  decoder-adapted head combined with the lookahead pass. Not run (deprioritised for the 0.6B and distillation work).
- **TurnBench for the 0.6B core.** Not measured; the AMI turn row already fixed the verdict, but the table has no
  two-party turn row for either core under the identical recipe.
- **The distillation run itself** (§8, next step): package delivered and smoke-tested on ~1 h of audio on the laptop
  (all loss terms active, WER gate verified to fire and restore), the ~6 A100-hour run not done.
- **Live cost of the second transcripts.** The dead air of an agent that waits for the TDT v3 or [70,13] final
  (`scripts/hybrid_live.py` H / L / HM) was never measured.
- **The speaker head of §3.1 in the served model**: needs the TS-VAD head retrained on its embedding space first.
- **Human check of the demo video** (`demo/`, v4, 120 s in 16:9 and 1:1): the narration was checked only by Whisper
  (2.8 % WER on the script); nobody has listened to the final render end to end.
- Carried from 2026-09-27: Parakeet-EOU, smart-turn and LiveKit not run on ICSI; TurnBench test not submitted; not
  measured: a real LiveKit room or Pipecat transport, echo of the agent's own TTS, a real LLM / TTS.

## 12. Limitations and threats to validity

- **In-sample selection.** The shipped rule (head OR dyn) was selected on AMI dev; ICSI is the held-out confirmation
  and there only Silero and Sortformer timeouts were run as baselines. The TurnBench 0.835 / 0.752 rows are
  dev-selected operating points; the held-out rows (0.862, 0.849) are the fair ones, and one held-out half of the
  predictive rule is over the FP budget (0.121).
- **Label stand-ins.** Agent-end arming uses the other party's labelled turn end for the TTS-end event, in the
  benchmark and in the product runs. Two of the five AMI windows have no earlier other-speaker turn.
- **Label conventions.** The VAD head was trained on the AMI word-level label it is scored with; oto labels are
  Silero segments; TurnBench negatives are hand-labelled spans; Nemotron-3's AMI DER is inflated by its VAD
  convention and its card lists AMI train + dev as training data.
- **False-cutoff rates are not exactly matched.** Realised held-out FC is 4.8-6.5 % on AMI (up to 12 % in the open
  stratum) and 1-4 % on ICSI; rows are compared at the cross-fitted point, not at equal FC.
- **Speaker data overlap.** The speaker head's training set includes 33 ICSI-train speakers; ICSI dev speakers are
  disjoint from it, but the ICSI corpus is not "never trained on". ICSI's own splits are not speaker-disjoint.
- **Small samples.** 5 AMI windows in the product runs (one scored end each); 16 clips per two-party set; speaker
  EER at n = 64 has ~±3 points of noise; LID test prefixes hold 1-3 speakers per language for 8 of 17 languages.
- **ASR on AMI** uses 200 clean single-speaker segments and our normaliser, not the Open ASR Leaderboard protocol;
  overlapped speech is excluded, which flatters every system equally. The ASR paired deltas are on `normalize_text`;
  the Whisper-normalised gaps (6.2-7.7 points) have no CI.
- **Compute** was measured on one shared laptop at load < 6 with other jobs running; read RTF as an order of magnitude.
  Whisper ran through faster-whisper (CTranslate2, int8). TDT v3's RTF was measured at batch 1 under load.
- **Numbers without JSON.** `TEACHERS.md`, `NEMO_IMPORT.md`, `ONDEVICE.md` (sweep lost in a reboot) and
  `SORTFORMER_IMPORT.md`'s streaming DERs are quoted as context.
- **Licensing** is per component (§9); the LiveKit models were used for evaluation only.

- **Shared machine.** Every measurement of 2026-09-28 ran on one laptop shared with other agents' jobs (load 3-8,
  up to 14 GB of swap during some runs); the E2E_FINAL quiet check was relaxed from 150 % to 400 % of a core for the
  TS-VAD live runs; the RTF figures of §4.1 and §7.1 carry their load and the before / after comparisons that matter
  are interleaved A/Bs or forced-shedding runs, not separate runs at different times.
- **Rebuilt clips.** The E2E scratch was lost in a reboot; the 37 clips of the TS-VAD live runs were rebuilt from
  `runs/e2e_final.json` (every start and duration within 10 ms, every scored AMI turn end identical) and compared with
  E2E_FINAL's *stored* C / D records from the day before, not with a same-day re-run.
- **In-sample frontier points.** §5.7 picks each point on the evaluation set; its numbers show the trade-off and are
  optimistic as operating points.
- **AMI is in Nemotron-3's training data** (its card lists AMI train + dev): its AMI-dev DER (0.232) and the
  diarization-fix numbers on AMI windows are partly contaminated, and so is the distillation target of §4.2.
- **Scratch-only tables.** The diarization-fix tables, the 0.6B VAD rows and the TS-VAD live server RTFs are in files
  on the external SSD, not in `runs/`; they were read back for this report (`research/VERIFICATION_2026-09-28.md`).

## 13. Next steps, with rough cost

| step | scorecard row it moves | gate / kill criterion | rough cost |
|---|---|---|---|
| Run the 0.6B → 115M distillation package (`scripts/research/distill_0p6b_to_115m/`) with its no-teacher control | 2, 20-22 (streaming meeting WER at the 115M's cost) | AMI-200 streaming WER ≤ 17 % and LibriSpeech-200 ≤ 2.59 %; kill if the WER gate fires at every checkpoint or AMI is not 1.5 points below the control | ~6 A100 hours, ~60 GB disk |
| Rebuild the Sortformer tracks (`tsvad_chain.sh`) for the paired TS-VAD CIs and the Sortformer-input frontier | 18, 25 | – | ~5 h CPU |
| Wire the 0.6B streaming model as a GPU transcript source in `serve.py` (a `--final-asr`-style worker or a second streaming pass) and measure the live dead air of waiting for any second final | 2, product | keep if the wait adds < 300 ms median | 1-2 days engineering + GPU |
| Enrolment quality (experiment 4) and the TS-VAD path in rooms with a real voice print, on recorded multi-party calls with an agent channel | 18-19 | ≤ 2 points of misses lost vs the 5 s offline print | 1-2 days CPU + data |
| Retrain the TS-VAD head on the §3.1 speaker head's space, then transplant both | 4, 18, 23 | no TS-VAD F1 loss | ~1 h MPS |
| TurnBench for both cores; TurnBench test submission; VAP-style objective on the dyadic head | 9, 22 | kill if P50 stays > 900 ms at FP ≤ 0.10 | hours CPU; 2-4 GPU-days for the objective |
| Batch sessions through one encoder / diarizer call | 16 (sessions per worker) | ≥ 2 real-time sessions per 2 threads, outputs identical | ~1 week engineering |
| Run Parakeet-EOU, smart-turn and LiveKit on ICSI | 8 | – | 1-2 h CPU |
| A human listen of the demo v4 render before it is posted | – | – | 5 min |

## Appendix A. Sources and reproduction

Every command runs from the repository root with `PYTHONPATH=. .venv/bin/python`; the long ones are staged and
resumable (each call bounded by `--budget` seconds, repeat until "done").

| section | JSON | script / command |
|---|---|---|
| 1 ASR | `runs/final_asr.json`, `runs/hybrid_asr.json`, `runs/served_model_build.json` | `scripts/final_asr.py prep`, `run --system {served,parakeet,whisper_small,whisper_turbo} --set {ami,libri}`, `report`; `scripts/hybrid_asr.py run --set {ami,libri}`, `report` |
| 2 VAD | `runs/baselines_sd.json`, `runs/vad_layers.json` | `scripts/bench_sd_baselines.py --stage vad\|ours\|sortformer\|pyannote\|spk`; `scripts/vad_layers.py features\|probes\|compute\|silence\|report` |
| 3 speaker | `runs/spk_head.json`, `runs/baselines_sd.json` (`spk.*`) | `scripts/spk_head.py`, `scripts/cache_titanet.py`, `scripts/spk_overlap.py`, `scripts/research/make_served_model.py` |
| 4 diarization | `runs/baselines_sd.json`, `runs/layer_routing.json` | `scripts/layer_routing.py train\|diar_eval\|turn_pair\|turnbench\|report`; `python -m audioforge.nemo_import <.nemo> runs/<name>.afm`; `scripts/eval_stage1.py --tasks diar` |
| 5.1 EOT AMI | `runs/baselines_turn.json`, `runs/turn_trail6_hybrid_leakfree.json`, `runs/turn_trail6_primary_titanet.json`, `runs/turn_trail6_enroll_leakfree.json`, `runs/completeness.json` | `scripts/make_sortformer_tracks.py --track-source stream`; `scripts/eval_stage1.py --bench v2 --v2-stage tracks\|scores\|report\|embed\|bind`; `scripts/bench_turn_baselines.py --stage vad\|smartturn\|lkaudio\|eou\|asr\|lktext\|report`; `scripts/bench_completeness.py --stage dyntimeout` |
| 5.2 EOT ICSI | `runs/baselines_turn_icsi.json`, `runs/baselines_turn_icsi_dyn.json` | `scripts/bench_turn_icsi.py --stage tracks\|vad\|scores\|report`; `scripts/bench_turn_icsi_dyn.py --stage extract\|report` |
| 5.3 TurnBench | `runs/turnbench_dev.json`, `runs/turnbench_dev_trail6.json`, `runs/dyadic_bench.json`, `runs/dyadic_train.json`, `runs/turnbench_latency.json` | `scripts/prepare_dyadic.py`; `scripts/bench_turnbench.py`; `scripts/bench_turnbench_latency.py`; `scripts/bench_dyadic_heads.py --stage predictive`; scorer `integrations/turnbench_scorer/` |
| 5.4 otoSpeech | `runs/dyadic_train.json`, `runs/dyadic_bench_oto.json`, `runs/dyadic_tables.md` | `scripts/bench_turn_dyadic.py`; `scripts/summarize_dyadic.py` |
| 5.5 semantic | `runs/yield_tokens.json`, `runs/completeness.json`, `runs/turnbench_dev_completeness.json` | `scripts/bench_yield_tokens.py --stage decode\|wer\|segments\|report`; `scripts/bench_completeness.py`; `scripts/bench_turnbench_cmp.py` |
| 6 LID | `runs/lid.json` | `scripts/lid_data.py fetch\|edacc`; `scripts/lid.py onsets\|probe\|train\|eval_head\|baseline\|decide\|cost\|report` |
| 7 product | `runs/e2e_final.json`, `runs/integration_deadair_shipped.json`, `runs/integration_deadair_032*.json` | `scripts/e2e_final.py prepare\|queue --wt <clean worktree>\|predictive\|report`; INTEGRATION.md §3 for the adapters |
| 10 negatives | `runs/enc_0p6b.json`, `runs/contamination.json`, `runs/layer_routing.json`, `runs/vad_layers.json`, `runs/yield_tokens.json` | `scripts/enc0p6b.py`, `scripts/contamination_probe.py turn\|saasr\|report` |
| 11 TS-VAD | `runs/tsvad_sel_*.pt` (scratch `tsvad/frame_sel_ami.json`) | `scripts/tsvad.py trainfeats\|enroll\|train\|winfeats\|vprints\|frame\|bind\|scores\|report` |
| 1.3-1.4 lookahead, adaptation | `runs/hybrid_asr.json` (`english`, `adapt`) | `scripts/research/hybrid_asr.py prep_icsi`, `lookahead [--model ... --tag _adapt]`, `adapt_report`; `research/recipes/asr_meeting_decoder_adapt.yaml` |
| 3.1 speaker head | `runs/spk_frame.json` | `scripts/research/spk_frame.py` |
| 4.1 diarization fix | scratch `diar/offline/*.json`, `diar/logs/*.json` | `scripts/research/diar_fix.py run\|offline\|calib\|score`; `tests/test_diarization_fix.py` |
| 4.2 diar distillation | `runs/diar_distill.json` | `scripts/research/diar_distill.py`, `research/recipes/diar_distill_nemotron3.yaml` |
| 5.6 TS-VAD offline / live | `runs/improve_115m.json`, `runs/e2e_tsvad.json`; scratch `improve/serve_check.json`, `e2e_tsvad/runs/` | `scripts/research/tsvad.py winfeats\|vprints\|bind\|scores\|report`, `tsvad_serve_check.py`, `e2e_tsvad.py`; `serve --turn-input tsvad --diar-off` |
| 5.7 frontier | `runs/eot_frontier.json` | `scripts/research/eot_frontier.py` |
| 5.8 negatives | `runs/tsvad_turn.json`, `runs/hybrid_fast.json` | `scripts/research/tsvad_turn.py`, `hybrid_fast.py` |
| 6.1 transcript LID | `runs/hybrid_asr.json` (`lid_text`, `fleurs`) | `scripts/research/hybrid_asr.py run --set fleurs [--n-lang 30]`, `report` |
| 7.1-7.3 performance, robustness | `runs/perf.json` | `scripts/research/bench_serve.py ab\|components\|run --streams K`; `tests/test_perf.py`, `tests/test_bulletproof.py` |
| 8.1 0.6B core | `runs/hybrid_asr.json` (`core_0p6b`), `runs/spk_frame.json`, `runs/tsvad_turn_cores.json`; scratch `hybrid_asr/core_0p6b_rtf.json` (VAD) | `scripts/research/core_0p6b.py import\|vadfeats\|vad\|rtf\|report`, `tsvad_turn.py` (core-aware) |
| frontier pilots | `runs/frontier_sa_captions.json`, `runs/frontier_codec_probe.json` | `research/archive/FRONTIER.md` Part 2 |
| verification | `research/VERIFICATION_2026-09-28.md` | claim-by-claim read-back of this report |
| server | – | `audioforge-serve --diarizer nemotron3` (= `python -m audioforge.serve --asr models/stage1_served.afm --diar models/nemo_nemotron3_diar.afm --diar-pool max --diar-left 1 --shed-diar hold`, all 8 columns) `[--enroll after_agent_arm] [--final-asr tdt_v3] [--lid ambernet] [--diar-labels registry] [--turn-input tsvad --diar-off --enroll explicit]`; the E2E runs used `--diar-spks 4` |
| this report | `research/FINAL_REPORT.md` → `.html` | `scripts/final_report_html.py` |

Research documents by topic: `ANALYSIS.md` (the 149-model catalog), `NEMO_IMPORT.md`, `SORTFORMER_IMPORT.md`,
`TEACHERS.md`, `STAGE1.md`, `AMI.md`, `ICSI.md`, `EOT_BENCH_V2.md`, `BASELINES.md`, `DYADIC.md`, `SPK_HEAD.md`,
`LAYER_ROUTING.md`, `VAD_LAYERS.md`, `CONTAMINATION.md`, `YIELD_TOKENS.md`, `COMPLETENESS.md`, `ENC_0P6B.md`,
`LID.md`, `ONDEVICE.md`, `INTEGRATION.md`, `INTEGRATION_VERIFY.md`, `E2E_FINAL.md`, `INTERIM_SCORECARD.md`,
`VERIFICATION.md`, `DATA_PLAN.md`, `OUTSIDE.md`, `GAME_CHANGER.md`; added 2026-09-28: `IMPROVE_115M.md`,
`HYBRID_ASR.md`, `IMPROVEMENTS.md` (summary table of every experiment verdict first), `DIARIZATION_FIX.md`,
`PERFORMANCE.md`, `BULLETPROOF.md`, `ISSUES_FIXED.md`, `FRONTIER.md`, `VERIFICATION_2026-09-28.md`.

## Appendix B. Document vs JSON disagreements (the JSON value is used above)

1. `INTERIM_SCORECARD.md` §1 shows Whisper-normalised AMI WERs (20.63 / 14.14 / 14.43 / 12.94 %) next to paired
   deltas (+5.11 / +3.27 / +4.80) that `final_asr.json` computes on `normalize_text` (24.43 vs 19.32 / 21.16 /
   19.63). The Whisper-norm gaps are 6.49 / 6.20 / 7.69 points, without CIs. Its LibriSpeech column is likewise
   Whisper-norm; "equal on read speech" holds only against Whisper small, and "6-65x cheaper" is 2.2-66x.
2. `INTERIM_SCORECARD.md` §5 calls the TurnBench 0.835 / 0.752 rows "thresholds frozen from AMI"; `turnbench_dev.json`
   and `turnbench_latency.json:reference.dyadic_md_hybrid_in_sample` show they were picked on TurnBench dev. The
   AMI-frozen points are 0.465 / FP 0.256 (`turnbench_dev_trail6.json:frozen_ami`). It also omits ESPnet (0.836 /
   0.074 / 895 ms) and describes the predictive trigger's thresholds as oto-fitted (they are TurnBench-half-fitted;
   the oto-fitted point is 0.866 / FP 0.175 / 800 ms).
3. `INTERIM_SCORECARD.md` §3 says ICSI is "a corpus never trained on"; `spk_head.json` records 33 ICSI-train speakers
   in the shipped head's training set (ICSI dev is held out). §6's "0.6B carries no more speaker information" holds
   on AMI only; on ICSI the 0.6B probe is 20.3 vs 28.2 % EER (`enc_0p6b.json`). §5's "head-OR-nothing" comparator for
   the −9.3 ICSI delta is the hybrid (head OR primary timeout).
4. Realised false-cutoff rates exceed the 5 % target for several AMI rows (ours 6.5 %, EOU 5.3, smart-turn +
   timeout 5.2, LiveKit 5.1-5.7 %); the docs say "≤ 5 %" (`baselines_turn.json:*.fc_rate`).
5. `INTEGRATION.md` §8 / `INTERIM_SCORECARD.md`: hybrid_dyn "+0.66 s" pairs against the older §4 timeout rows; against
   the same run's control it is +0.72 to +1.25 s (`integration_deadair_shipped.json`). E2E_FINAL measures +0.76 /
   +0.88 s on the same windows.
6. `hybrid_asr.json:card.ami_test` stores 11.31 % for parakeet-tdt-0.6b-v3; `DATA_PLAN.md` and `GAME_CHANGER.md`
   quote 11.39 %.
7. `BASELINES.md` lists Nemotron-3 at 0.255 DER (mean pooling); `SORTFORMER_IMPORT.md` has 0.232 with max pooling,
   the setting the product uses. Neither has a JSON; the E2E JSON has the product effect.
8. `ENC_0P6B.md` gives the 115M VAD probe as 0.940 / 0.960 in the text and 0.950 / 0.969 in its own table
   (`vad_layers.json`: block 9 F1 0.9496, served head 0.9485).
9. AMI segment WER for the streaming model appears as 23.2 % (STAGE1, a run that silently used [70,13]), 25.0 %
   (STAGE1, the [70,1] retrain, other set) and 24.4 % (YIELD_TOKENS / final_asr, the 200-segment set used here).
10. `CONTAMINATION.md`'s "half-life 1.5-2 s" is not a JSON key; from the stored curves the value is 0.8-1.9 s.
    `VAD_LAYERS.md`'s "gating costs more than it saves" rests on its FLOP model; `vad_layers.json` records only the
    positive wall-time savings (8-17 %).
11. `README.md`'s "One model vs the dedicated models" table still shows the original speaker head (32.2 % EER) and
    "hybrid rule adds nothing yet" for the product; both are superseded (14.4 %; §7 here).
12. Code references to `research/archive/HYBRID_ASR.md` (serve.py, hybrid_asr.py, NEMO_IMPORT.md) and `research/
    IMPROVE_115M.md` (tsvad.py) pointed to documents that were not in the committed tree when this report was
    assembled (an untracked HYBRID_ASR.md draft and `runs/improve_115m.json` appeared during finalisation); §1.2 and
    §11 are the write-up used here.
13. `INTERIM_SCORECARD.md` §2 says pyannote costs "~50x" our CPU; `baselines_sd.json` gives 57-66x. §7's EdAcc
    "19-31 %" mixes two heads (`lid_aug` 18.7 %, `lid_head` 31.1 %).
14. `DIARIZATION_FIX.md` §3.4 (and its proposed default, and commit c67693a) gave AMI-64 DER **0.229** after the fix
    (acc 0.772, 206 finals): the first-pass aggregate of 57 of 64 windows. The complete 64-window file gives **0.2448**,
    identical to the "before" row (acc 0.765, 241 finals). The doc is corrected; this report says "unchanged at 0.245".
15. `DIARIZATION_FIX.md` §1.3's Nemotron-3 two-session "mix" row (count −2, acc 0.68, purity 0.76) has no matching
    stored values: `n3_concurrent.json` scored that session against the ICSI reference. Annotated in the doc; not used.
16. `BULLETPROOF.md` said 45 tests; pytest collects 48 (as commit 548b629 says). Corrected.
17. `IMPROVEMENTS.md` §3 put TitaNet's within-meeting 8.2 % in the "all" slot; JSON: 6.6 / 8.2 (n = 64), 10.9 / 12.0
    (n = 200). Corrected.
18. "Identical outputs" for the performance pass: `perf.json` stores `identical: false`, `within_tol: true`, 0 decision
    differences, max eot difference 1e-5. PERFORMANCE.md's "1311 of 1313 messages bit-identical" is consistent with that
    but not itself stored. This report, README and CHANGELOG say "decisions identical, probabilities within 1e-5".
19. The 2026-09-27 version of this report quoted the TS-VAD frame result from a scratch file (F1 0.745, precision
    0.721, miss 0.229, FA 0.189); `improve_115m.json` has F1 0.743, precision 0.750, miss 0.265, FA 0.155 (used here).
20. The experiment-3 speaker head is quoted elsewhere as "17.2 % / 4.2 %"; the JSON has two heads, 17.1 / 4.4 (warm
    start) and 17.4 / 4.0 (fresh init). This report gives the range.
21. Item 12 is resolved: `research/archive/HYBRID_ASR.md`, `research/IMPROVE_115M.md` and `runs/improve_115m.json` are now
    committed and agree with the JSON used here.
