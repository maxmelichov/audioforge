# Data plan: one model at or above the dedicated models, task by task

*2026-09-26. A writing and research note: no model was run for it. Every number points to a file in research/ (or
runs/) or to a URL. **(est.)** marks an estimate, with its arithmetic next to it. "Dedicated model" means the best
single-task system we have measured on the same data, usually an NVIDIA checkpoint running in our code.*

## 1. Where we stand, task by task

| task | ours (one frozen NVIDIA encoder + our heads) | dedicated model / deployable baseline, same data | gap | source |
|---|---|---|---|---|
| ASR | frozen NVIDIA 115M streaming RNNT: 1.92 % LibriSpeech (1 s lookahead), 2.48 % at 0; **25.0 % on AMI dev segments at 160 ms** | it *is* the NVIDIA model. Larger dedicated ASR: parakeet-ctc-1.1b 1.64 % on the same 200 clean utts; AMI-IHM card numbers 11.39 % (tdt-0.6b-v3), 15.62 % (EOU-120M @160 ms). The cards use a different normalizer, so not comparable | 0 by construction vs the source model; the ~10-point meeting gap belongs to the 115M model, not to fusion | NEMO_IMPORT.md, STAGE1.md "160 ms", TEACHERS.md, GAME_CHANGER.md §3 |
| VAD | frame acc **0.921** (160 ms); recall 0.96 | always-speech 0.769. MarbleNet (NVIDIA's dedicated VAD) **not measured here** | unknown: measure MarbleNet on the same 64 windows first | STAGE1.md §3 + "160 ms" |
| Diarization (pooled frame DER, no collar, word-level labels) | our Sortformer head on the frozen encoder **0.384–0.394**, worse than trivial one-speaker + oracle VAD (0.312) | Streaming Sortformer v2: **0.201** offline, 0.252 streaming at 1.04 s; Nemotron-3 (max pool) 0.232 / 0.241 | **+0.18–0.19** absolute; our head does not learn diarization. Product today = ported v2 | STAGE1.md v2, SORTFORMER_IMPORT.md |
| Speaker ID (EER, 64 AMI segments) | **15.0 % all pairs / 32.2 % within meeting** | untrained layer-mix features 20.4 / 34.9 %. TitaNet-L 0.66 % on VoxCeleb1-O (card); **TitaNet not measured on our AMI trials, not yet ported** | unknown on AMI; large on paper (0.66 vs 15 %, different test) | STAGE1.md §4, GAME_CHANGER.md §3 |
| Turn-taking (eot-bench v2, n = 974, causal enrollment, 6 s, ≤ 5 % per-turn FC) | **hybrid (trail6 head OR ~4 s timeout) 61.9 % [58.7, 65.0] misses** | deployable cascade (Sortformer stream + timeout) **74.8 % [72.0, 77.5]** | **−12.8 [−15.9, −9.8]**: we win overall; floor-open **tie** 45.9 vs 46.1 (−0.2 [−6.7, +5.9]) | EOT_BENCH_V2.md §7A |
| Turn-taking, enrollment | oracle binding: hybrid 28.7 %; causal: 61.9 % | same gap for the timeout (32.0 vs 74.8) | **~37 points** lost to label-free enrollment (stays at 0.32 s: 63.7 vs 26.8) | EOT_BENCH_V2.md §7A/B |

Reading: ASR is at parity by construction (frozen). VAD and speaker have not been compared with a dedicated model.
Diarization is the one clear loss (0.39 vs 0.20). Turn-taking already beats the deployable cascade overall; its
remaining loss is enrollment, and on floor-open ends it only ties.

## 2. Root causes we established

1. **Label conventions.**
   - AMI word timings hide pauses < 0.5 s (0.9 % of non-zero gaps are < 0.5 s; ICSI 48 %), so ICSI has ~9× more
     hesitation labels (2.02 vs 0.23 per turn). Energy separates labelled speech from silence with AUC **0.63 on AMI vs
     0.79–0.81 on ICSI**. → AMI.md caveat 1, ICSI.md.
   - At timeouts ≤ 800 ms, 50–60 % of the stream timeout's cutoffs fall in "speech" that is acoustically silent. An
     energy-corrected oracle timeout has 21–25 % FC at 400 ms, not 14 %. → IDEAS2.md §3 effect 5.
   - Forced-alignment vs word-level references: v2.1 and Nemotron-3 were trained and scored on NTT forced-alignment
     RTTMs. On our word-level labels they miss 15.6–19 % of speech at VAD level (v2: 10.3 %), so DER is 0.242 / 0.255 vs
     0.201. That is **4–5.4 DER points from the convention alone**. → SORTFORMER_IMPORT.md "Why the newer models score worse".
   - AMI+ICSI turn training did not help on AMI dev (61.1 vs 62.6 %), and the conventions differ (ICSI force-aligns
     laughter and breaths). → PLAN.md §6, STAGE1.md Consolidated.
2. **Trailing-window artifacts.**
   - The end-of-file flush changed the last 13 frames of every cached streaming track, up to |Δp| 0.95. The effect on
     the benchmark is small (≤ 1.4 points), but every turn head up to v3 was trained on it. → EOT_BENCH_V2.md §5.
   - 2 s trails hid most timeout firings. Heads trained on them never learned "long silence ⇒ end": v3 lost at 6 s by
     +4.9 points, and trail6 won by −8.3. → EOT_BENCH_V2.md §3–4, STAGE1.md trail-6.
3. **Enrollment.** The causal − oracle gap is +36.2 points inside the head and ~37 for the hybrid. Causal binding agrees
   with the oracle column 70 % of the time, the same at both diarizer latencies. → EOT_BENCH_V2.md §7B.
4. **Diarizer lag is not the cause.** The 0.32 s setting cuts emission delay 840 → 240 ms but leaves misses and binding
   unchanged. The median end lag is 1 frame. The real track errors are in-turn dropouts (P90 15 frames) and
   next-speaker leakage into the user's slot (22.5 % of post-end frames). → EOT_BENCH_V2.md §7B, TURN_ERRORS.md §1.
5. **Encoder conditioning trained on clean activity only.** Oracle track in the frozen speaker kernels + Sortformer
   everywhere else: 9.5 % misses; Sortformer in the kernels: 62.6 %. Retraining (v4) or removing (v5) the kernels made
   it worse. → TURN_ERRORS.md §3, STAGE1.md Consolidated.
6. **The frozen ASR encoder carries little diarization information, and unfreezing costs WER.** DER stays at 0.38
   after 2000 steps; unfreezing the top 6 blocks (lr×0.05, KL anchor) tripped the WER gate at step 500 (2.05 → 4.31 %).
   → PLAN.md §6.
7. **Small data.** All real training so far: 12 AMI meetings (7.53 h, 45 speakers, 3274 turns) + 12 ICSI (10.27 h);
   4 dev meetings / 15 speakers; an n = 64 result flipped at n = 200. → AMI.md, ICSI.md, PLAN.md §6.

## 3. Data plan (ordered by expected impact per unit cost)

Measured costs used below:
- **Heads on the frozen 115M encoder:** 0.46 s/step with all heads (4.9M new params: `runs/stage1_heads_pretrained.log`,
  913.8 s / 2000 steps) and 0.52–0.55 s/step for the turn head alone (`runs/stage1_turn_v3{,_trail6}.log`), at batch 6
  on MPS. The trail6 run: 1096 s / 2000 steps, 20 s windows.
- **Streaming Sortformer tracks on MPS:** 3274 × 20 s windows in 2099 s ≈ **31× realtime** (STAGE1.md trail-6).
- **Offline Sortformer on CPU:** ~43× (200 × ~10 s windows in 47 s, SORTFORMER_IMPORT.md).
- **ASR teacher:** parakeet-ctc-1.1b at **63×** on MPS (TEACHERS.md).
- **Disk:** WAV is 115 MB/h, and the float32 `.npy` cache adds ~2× (AMI.md: 1.4 GB wav + 2.6 GB cache). **This Mac has
  88 GB free**, so the cache must go (or become fp16) before (b).
- **Rented GPUs:** A100 $1.39–1.59/h, H100 $2.69–3.49/h (GAME_CHANGER.md §5, runpod.io/pricing).

| # | item | licence | hours | labels | fixes | expected effect | cost |
|---|---|---|---|---|---|---|---|
| a | label-consistency pass on AMI/ICSI | CC BY 4.0 | the existing 17.8 h, then all | self-derived + CTC teacher | causes 1, 2 | turn: FC budget no longer wasted on silent "speech"; diar: 4–5 DER points of convention error | CPU ~1 day; +4× disk if headset channels are used |
| b | full AMI + full ICSI + NOTSOFAR-1 + DiPCo | CC BY 4.0 / CDLA-Permissive | ~200 h real | human | cause 7 | not turn misses (2× data did nothing); kernel robustness and diarization distillation | ~35 GB disk; tracks ~10 h MPS; 1 epoch ≈ 1.4 h |
| e | enrollment supervision from repeated speakers | CC BY 4.0 (same corpora) | reuses (b) | derived from speaker ids | cause 3 (37 points) | largest lever on the headline turn number | CPU; TitaNet port ~1–2 days of engineering |
| c | teacher-labelled conversational audio | CC0 / CC BY | 300 h (Mac) → 10k h (rented) | NVIDIA teachers | causes 6, 7 | the data that makes one-pass distillation (stage 3) possible | Mac ≈ 340 h/day (est.); A100 ~10 GPU-h per 10k h |
| d | synthetic conversations | CC BY / Apache | unlimited | exact by construction | cause 2, identity, overlap | long post-end silences, known enrollment; not real hesitation or floor statistics | CPU generation; no labelling |

Item (e) is ranked above (c) because it attacks the largest measured gap (37 points) with data we already have.

### (a) Label-consistency fixes on AMI/ICSI

- **Source.** The existing downloads, plus the AMI individual headset channels (`Headset-0..3.wav`, about 4× the mix
  audio; AMI.md "To recover short within-segment pauses").
- **How labels are made.** Produce three activity views per speaker, all 80 ms:
  1. `words`: today's labels.
  2. `words∧energy`: word spans gated by that speaker's headset energy. This recovers the pauses under 0.5 s.
  3. `fa`: re-alignment with the local CTC teacher. parakeet-ctc word starts are good; word ends are early
     (TEACHERS.md "Timestamps").

  Where they exist, add the forced-alignment RTTMs that v2.1 / Nemotron-3 were trained on (the NTT AMI RTTMs named in
  SORTFORMER_IMPORT.md; source and licence **to verify**). ICSI gets view 2 from its per-speaker channels
  (`ICSIsignals/SPH`, ICSI.md).
- **Gates.** Energy AUC of `words∧energy` on AMI rises from 0.63 toward ICSI's 0.80; trivial-feature leak probe AUC
  ≈ 0.5 (IDEAS.md §6 item 4; the pilot gave 0.630).
- **What it fixes.**
  - One label convention across AMI and ICSI, so their mixture stops conflicting (cause 1).
  - A diarizer reference that matches what Sortformer v2.1 / Nemotron learned. It lets us fine-tune Nemotron-3 (best
    streaming DER 0.241, 2× faster, OpenMDW) on our convention, or score it fairly.
  - Flush-free, 6 s-trail windows everywhere (cause 2), which is already implemented.
- **Expected effect (est.).**
  - Diarization: up to the 4–5.4 points of convention error measured above. For example, Nemotron-3 streaming 0.241
    moves toward v2 offline 0.201 on a matched reference.
  - Turn: the energy-corrected oracle timeout shows that 7–11 FC points at 400 ms (21–25 % vs 14 %) come from label
    artefacts. Removing them from training targets should lower the head's pre-end scores inside silent "speech",
    which TURN_ERRORS.md §4 shows set the threshold.
  - No miss number is promised. The gate is a re-run of eot-bench v2.
- **Cost.** Alignment at ≥ 63× → ~0.3 h for 17.8 h (est.); AMI headset channels for 16 meetings ≈ 4 × 1.4 GB ≈ 6 GB;
  tracks need no re-cut (only activity labels change).

### (b) Real conversational hours

| corpus | licence (URL) | hours | use |
|---|---|---|---|
| AMI, all meetings (136 / 18 / 16 in the BUT split) | CC BY 4.0 (https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) | ~100 h total (https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml); train ≈ 80 h (est.) | train / dev / eval |
| ICSI, all 75 | CC BY 4.0 (https://groups.inf.ed.ac.uk/ami/icsi/license.shtml) | 71.7 h (65.5 h train) | train; not speaker-disjoint, so seen-speaker eval only |
| NOTSOFAR-1 real meetings (237 × ~6 min) + ~1000 h simulated | CC BY 4.0 (https://github.com/microsoft/NOTSOFAR1-Challenge) | ~24 h real | train (far-field) |
| DiPCo, 10 dinner-party sessions of 15–45 min | CDLA-Permissive-1.0 (https://zenodo.org/records/8122551) | ~3–7 h (est. from the session range) | train / eval, far-field |
| CHiME-6 | **CC BY-SA 4.0** (https://www.openslr.org/150/) | — | **eval only** (GAME_CHANGER.md share-alike policy) |
| AliMeeting (Mandarin) | **CC BY-SA 4.0** (https://www.openslr.org/119/) | 118.75 h | eval only; Mandarin |
| Fisher / CALLHOME | paid LDC agreement (GAME_CHANGER.md §4) | — | optional, if budget allows |

- **Turns (est., linear in hours).** AMI train ≈ 3274 × 80 / 7.53 ≈ 35k. ICSI train ≈ 3082 × 65.5 / 10.27 ≈ 20k.
  Total ≈ 54k, 17× today's 3274.
- **Disk / compute.** AMI ~11.5 GB + ICSI 8.3 GB + NOTSOFAR/DiPCo ≈ 25–35 GB WAV; the float32 cache (~2× more) does
  not fit in 88 GB, so drop it. Streaming tracks ≈ 54k × 0.64 s ≈ **9.7 MPS-hours**; one epoch ≈ 9.1k steps × 0.55 s ≈ **1.4 h**.
- **What it fixes and what it does not.**
  - Doubling turns (AMI+ICSI, 7117) did not move the causal miss (61.1 vs 62.6 %, PLAN.md §6). **More turns alone is
    not expected to lower turn misses.**
  - Its value: 190 + 60 speakers for identity and kernel robustness (cause 5), and real in-domain targets for the
    diarization distillation (stage 3).
  - It also enables a speaker-disjoint AMI eval with all 16 test meetings (dev + eval ≈ 1900 turns, AMI.md) instead of
    4 dev meetings.

### (e) Enrollment data: supervision for "which voice is the user"

- **Idea.** Enrollment is the measured 37-point loss, and the corpora already hold the supervision: an AMI scenario
  series (a/b/c/d) repeats the same 4 people, and ICSI speakers recur heavily (me013 is in 49 of 75 meetings, ICSI.md).
- **Construction.** For each turn window, draw an **enrollment clip** of 5–15 s of the primary speaker (est. range) from
  a *different* meeting (or, second choice, a distant part of the same meeting). Use only their single-speaker stretches
  (`asr` mode, AMI.md).
  Positives: the primary. Hard negatives: another participant of the same meeting (where EER is 32 %). Synthetic
  scenes (d): one LibriSpeech/VoxPopuli speaker is "the user", plus bystanders and an agent voice.
- **Labels.**
  - A binding target: which diarizer column is the user, at every frame.
  - A per-frame "user is speaking" target from the word labels.
  - TitaNet-L embeddings (CC BY 4.0, GAME_CHANGER.md §4) of the enrollment clip and of each column's recent segments.
    These give both a training-free binding rule and a distillation target for our speaker head.
- **Legal.** Every clip comes from CC BY / CC0 corpora whose speakers consented to recording and release. Do not use
  VoxCeleb (YouTube rights) or VoxBlink2 (NC) (GAME_CHANGER.md §4). An in-house "designated user" set would need
  written consent that covers voice-biometric use. It is not required for this plan.
- **Expected effect (est.).**
  - Assume the causal − oracle gap scales with binding disagreement: 33 points (61.9 → 28.7) over 30 points of
    disagreement (70 % agreement), ≈ 1.1 miss points per agreement point.
  - Then 90 % agreement gives ≈ 40 % hybrid misses, versus 74.8 % for the deployable timeout.
  - The assumption is linear and untested. It is the first thing to measure.
- **Gate.** On eot-bench v2, enrollment-bound (label-free) binding agreement must be ≥ 85 % at the turn end, and the
  hybrid's causal miss must fall by ≥ 10 points with a CI that excludes 0.
- **Cost.** Port TitaNet-L the way Sortformer was imported (~1–2 days, est.); a 23M model over ~54k windows is minutes
  on MPS (est.).

### (c) Teacher-labelled conversational audio at scale

- **Corpora we can use.**
  - VoxPopuli: CC0, 543 h of transcribed English from 1,313 speakers, plus untranscribed parliament audio
    (https://huggingface.co/datasets/facebook/voxpopuli). Mostly speeches with chair/speaker switches.
  - People's Speech: only the `cc-by-clean` / `cc-by-dirty` configs, skipping `*-sa`
    (https://huggingface.co/datasets/MLCommons/peoples_speech). The audio is archive.org public proceedings (government
    meetings, hearings), which are multi-speaker.
  - NOTSOFAR-1 simulated set: CC BY 4.0, ~1000 h.
  - Emilia-YODAS: CC BY 4.0, single-speaker segments. Use as synthetic source only.
  - YODAS2: CC BY 3.0, creators keep copyright (GAME_CHANGER.md §4). Use only after a legal read.
- **Excluded.** MSDWild ("ONLY for research purposes", https://github.com/X-LANCE/MSDWILD); Ego4D (signed licence
  agreement, terms not verified, https://ego4d-data.org/docs/start-here/); VoxConverse (annotations CC BY 4.0, but
  "copyright remains with the original owners of the video", https://github.com/joonson/voxconverse: eval only);
  AliMeeting and CHiME-6 (share-alike, see (b)); GigaSpeech / SPGISpeech (non-commercial).
- **Pseudo-label recipe.** It uses only local teachers already running in our code.
  1. Cut the audio into ≤ 40 s windows (TEACHERS.md: full-attention memory).
  2. **Words + timestamps:** parakeet-tdt-0.6b-v3 (durations → good spans) and parakeet-ctc-1.1b. Keep a window only if
     their normalized texts agree (TEACHERS.md, Granary-style filter). Never use Nemotron RNNT emission times: they
     are 0.2–0.9 s late.
  3. **Speaker tracks:** Streaming Sortformer v2, *offline* pass over window + 1.5 s of real following audio as the
     label (no flush artefact, EOT_BENCH_V2.md §5), and the *streaming* pass as the model input track. Also run
     Nemotron-3 max-pool, and keep windows where the v2 and Nemotron DER against each other is below a threshold
     (est. 0.15).
  4. **Identity:** TitaNet-L embeddings per diarized segment, clustered across a recording to get recording-level
     speaker ids. These give enrollment clips as in (e).
  5. **Turn labels:** derive them from the words + offline tracks with the same `floor` rules (AMI.md), 6 s trails.
- **Known biases, inherited by any student.**
  - The v2 streaming track leaks the next speaker into the user's slot (22.5 % of post-end frames) and drops out
    inside turns (P90 15 frames) (TURN_ERRORS.md §1). Offline labels reduce this but do not remove it: offline primary
    FA/speech is 0.205 (SORTFORMER_IMPORT.md).
  - Nemotron / v2.1 miss 16–19 % of speech on word-level references.
  - Parakeet drops fillers ("uh" → "the", STAGE1.md §5), under-labelling the disfluencies a turn head needs; floor
    rules on pseudo-words inherit its 25 % AMI WER.
- **What it fixes.** It gives the volume needed to distil Sortformer + TitaNet into our branch (stage 3). This is data
  for identity and diarization, not for turn semantics.
- **Expected effect (est.).** Student DER within ~2 points of the teacher in-domain; students usually trail their
  teacher, so "equal" is the upper bound. Anchor: teacher 0.201 offline / 0.252 streaming (64 windows), 26.3 % (6 s eot windows).
- **Cost.** Mac serial chain 1/(1/63 + 1/31 + 1/43) ≈ **14× realtime ≈ 340 h/day** (est.; excludes TitaNet and I/O);
  300 h ≈ 35 GB WAV is the Mac ceiling (88 GB free). Rented: parakeet-tdt-0.6b-v3 RTFx 3,333 → 10k h ≈ **3 A100-h**;
  Nemotron-3 RTFx 1,340 → ≈ **7.5 GPU-h** (GAME_CHANGER.md §4). 10k h ≈ 1.15 TB WAV (~0.6 TB FLAC).

### (d) Synthetic conversations (FastMSS four-move + our disfluency generator)

- **Source** (GAME_CHANGER.md §4). Speakers: LibriSpeech / MLS (CC BY 4.0), VoxPopuli (CC0), Common Voice (CC0;
  distributor terms unverified). Noise and rooms: MUSAN (CC BY 4.0), RIRS_NOISES (Apache-2.0); not WHAM! (NC).
- **Generator.** The four moves (hold / switch / interruption / backchannel, exponential pauses) and FDB-v3 disfluency
  types (PLAN.md §3). Use separate overlap settings per head, since FastMSS found overlap helps ASR and hurts
  diarization.
- **What it can teach.**
  - Arbitrarily long post-end silences. Trail6 vs v3 was worth −13.1 points at 6 s just from seeing > 2 s of post-end
    frames (STAGE1.md trail-6).
  - Exact multi-speaker activity, and a known user identity with free enrollment clips (e).
  - Controlled overlap and injected diarizer-like errors: dropouts up to ~20 frames, and leakage copied from the next
    speaker (TURN_ERRORS.md §7 rec. 2).
- **What it cannot teach.** Real hesitation-vs-switch statistics (synthetic hesitations were frequent and long, the
  opposite of AMI's labels, STAGE1.md §6); floor-taking semantics; real far-field channels. Its text must **never**
  reach the pretrained ASR heads: the WER gate tripped 2.13 → 8.18 % when it did (PLAN.md §6).
- **Gate.** Leak-probe AUC ≈ 0.5 (IDEAS.md §6.4). Every claim is made on AMI held-out data only (PLAN.md R3).
- **Cost.** CPU mixing only. Tracks are exact, so no teacher pass is needed. Streaming Sortformer tracks are needed only
  as noisy inputs: ~31× realtime on MPS.

## 4. Training plan (each stage has a gate)

**Evaluation protocol for every claim.**
- **Turn:** eot-bench v2 (EOT_BENCH_V2.md): all ≥ 974 AMI dev turns (AMI eval 941 only for the final model);
  label-free `causal_dominant` enrollment (the (e) enrollment clip reported separately) and deployable arming; 6 s
  horizon (2 s also reported); floor strata; cross-fitted ≤ 5 % per-turn FC (per-pause FC second); 1000-sample
  bootstrap CIs and paired differences.
- **Diarization:** pooled frame DER with no collar on word-level labels (our convention), **and** the standard
  0.25 s-collar DER on full sessions against the forced-alignment RTTMs, so the result is comparable to the cards.
- **Speaker:** EER within meeting, with all pairs as secondary.
- **ASR:** the LibriSpeech WER gate (+0.2 absolute max) and AMI segment WER.
- **Latency:** report the encoder chunk and diarizer setting with every latency.

| stage | what changes | data | gate to go on |
|---|---|---|---|
| 1 (done) | heads on the frozen NVIDIA encoder; external Sortformer v2 | 12 AMI (+12 ICSI) meetings | met: VAD 0.92, hybrid 61.9 vs 74.8 % |
| 2 | joint diarization + turn with **in-pass self-conditioning** (IDEAS.md architecture-2) | (a)+(b)+(d) | see below |
| 3 | multi-teacher distillation: Sortformer activity + TitaNet identity into our branch, one encoder pass | (b)+(c)+(e) | DER ≤ teacher + 2 pts; within-meeting EER ≤ TitaNet's on the same trials + 3 pts (est. margins) |
| 4 (optional) | encoder adaptation with the KL anchor | (c) + real transcribed speech | only if stage 3 misses its DER gate by > 3 pts |

- **Stage 2: what changes.** The speaker kernels stop seeing only clean activity (cause 5): a diarization head at layer
  k ≈ 6 of 17 feeds its own primary column to the kernels above it, with scheduled teacher forcing (p(oracle) 1 → 0.3),
  plus injected real-error statistics (d). The layer-k head is supervised by Sortformer's offline soft tracks and the
  (a) labels; binding comes from (e). Arms A′ / B / C / B0 as in IDEAS.md §2; heads only, 5k steps × 0.55 s ≈ 46 min
  per arm (est.).
- **Stage 2 gate.** Causal hybrid miss < 61.9 % (paired CI excluding 0) and floor-open miss < the timeout's 46.1 %;
  IDEAS.md kill rules (B no better than C; diar primary F1 > 8 points below A′). If the frozen encoder still cannot
  diarize (0.38 DER), keep the external Sortformer for diarization output; the in-pass head only drives the kernels.
- **Stage 3: losses.** Activity: BCE to the teacher's soft 4-column probabilities in its arrival order (no permutation
  search), plus word-level labels at weight ~0.3 (est.) to correct the FA-convention misses. Identity: cosine
  distillation to TitaNet-L segment embeddings + AAM-softmax on true ids (190 AMI + 60 ICSI + synthetic). Turn and VAD:
  BCE as before. RNNT: KL anchor (zero while the encoder is frozen).
- **Stage 3: capacity.** The frozen ASR features are not enough (cause 6). Add a trainable side branch of ~10–20M params
  (est.) that reads the all-layer mix. It is still one shared encoder pass, but it adds compute. Check the full-pipeline
  RTF ≤ 0.3 bar (GAME_CHANGER.md §3) against the measured 0.45–0.52 at 2 threads (INTEGRATION.md §1).
- **Stage 3 targets.** Streaming DER within 2 points of v2 at the same latency: 0.252 on 64 windows, 28.3 % on the 6 s
  eot windows at 0.32 s. Beating the teacher is only plausible after (a) aligns the reference.
- **Stage 4.** Any unfreeze must pass the +0.2 WER gate every 250 steps with checkpoint-before-gate (PLAN.md R2). The
  last attempt tripped at step 500, so stage 4 needs real transcribed speech mixed in at every step and a smaller LR.
  Stop if the gate trips twice.

## 5. Budget

Arithmetic as in GAME_CHANGER.md §5.

- **Stage-3 training per audio-hour:** frozen 115M forward, 2N × 45,000 frames ≈ 1.0e13 FLOPs, plus a 20M side branch
  trained at 6N × 1.3 × 45,000 ≈ 7.0e12. Total ≈ **1.7e13 FLOPs per audio-hour** (est.).
- **Ten epochs of 10k h:** 1.7e18 FLOPs ≈ **6 A100-h** at 80 TFLOP/s effective, or 18–30 h at the ×3–5 realistic
  factor.
- **Mac heads:** 0.46–0.55 s/step × batch 6 × ~10.5 s windows ≈ 3,300 audio-h/day at 100 % utilisation. The machine
  is shared, so plan on 50 %: ~79k steps/day.

| tier | budget | feasible | what it can prove |
|---|---|---|---|
| (i) this Mac only | $0 | (a) fully; (b) AMI+ICSI full once the fp16/no-cache change lands (~35 GB); (e) with TitaNet ported; (c) ≤ ~300 h (disk-bound; ~340 h/day labelling, est.); (d) unlimited; stage 2 arms (~45 min each); ~79k head steps/day | whether label fixes + enrollment cut the 37-point gap (gate: ≥ 10 points, CI excludes 0); whether self-conditioning beats frozen kernels; a first stage-3 DER on ~300 h |
| (ii) one rented GPU for a week | A100 168 h = $234–267, H100 $452–586, plus ~$50–150 storage/egress (est.) → **$300–800** | label ~2–5k h of (c) (≈ 10 GPU-h per 10k h for the teachers); stage 3 on 1–2k h × 10 epochs in about a day; 3–5 ablations | whether one-pass distillation reaches teacher DER ± 2 and TitaNet-level within-meeting EER; the answer to "one model ≈ dedicated models" at 120M |
| (iii) 10k-hour run | **$1–3k** (GAME_CHANGER.md §5: 120M / 10k h programme incl. ×5–10 ablation runs) | 10k h labelled (~10 A100-h), stored at ~0.6–1.15 TB; stage 3 at scale; stage 4 unfreeze with the KL anchor and WER gate | whether scale closes any residual DER/EER gap and whether encoder adaptation is worth it; a public eot-bench leaderboard entry |

## 6. Risks and stop rules (updated from PLAN.md §5)

- **Enrollment does not transfer.** If enrollment-clip binding agreement stays at ≤ 75 % (causal is 70 %), or the
  hybrid gains < 10 points, stop the turn-taking claim. Ship the fused front-end with the plain timeout, as today
  (INTEGRATION.md).
- **The floor-open tie persists.** If no stage beats the timeout's 46.1 % on floor-open ends with a CI that excludes 0,
  report the win as taken ends only. Those are the cases where an agent should *not* speak, which is a weaker product
  claim.
- **R2 (forgetting).** Any WER gate trip means restore and stop that arm. Two trips in stage 4 mean the encoder stays
  frozen permanently.
- **Distillation ceiling.** If the stage-3 student is > 5 DER points behind v2 after tier (ii), keep the external
  Sortformer (the current product). "One encoder" stays a research item, as decided in PLAN.md §6.
- **Label-convention trap.** Never compare DER across conventions. Every diarizer row must name its reference (words /
  words∧energy / FA).
- **Pseudo-label bias.** If the student reproduces the teacher's post-end leakage (the next speaker in the user's slot,
  22.5 %), the turn head's inputs do not improve. Check post-end FA on eot-bench before claiming parity.
- **Licence drift.** Re-verify licences at download time. Items marked "to verify" (the NTT RTTMs, Common Voice
  distributor terms, YODAS2) stay out of training until checked. Keep a licence manifest per dataset.
- **Small-n trap.** n ≥ 974 turns per claim, and AMI eval (941) held back for the final number only. n = 64 → n = 200
  already flipped a conclusion once (STAGE1.md).
- **NVIDIA ships the fused model first** (PLAN.md §5). Pivot to evaluation (eot-bench v2) and on-device tooling around
  it.
