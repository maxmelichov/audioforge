# Interim scorecard — everything finished so far (2026-09-27, before E2E_FINAL / FINAL_REPORT land)

This is not the final report. It is every result that is committed and source-verified right now,
while the last piece (the live Pipecat/LiveKit head-to-head, `research/E2E_FINAL.md`) is still running.
`research/FINAL_REPORT.md` will supersede this file once that lands and the report agent folds
everything together. Every number below was read directly from its `runs/*.json` at the time this
was written; sources are listed per section.

## 1. ASR (speech-to-text)

**Read speech — LibriSpeech test-clean, WER, Whisper-normalized text** (`runs/final_asr.json`):

| system | WER | CPU RTF (2 threads) |
|---|---|---|
| ours, streaming @160ms | 2.27% | 0.015 |
| Parakeet-CTC 0.6B, offline | 1.63% | 0.040 |
| Whisper-small, offline | 2.29% | 0.183 |
| Whisper large-v3-turbo, offline | 1.37% | 0.882 |

**Meeting audio — AMI dev, same protocol** (`runs/final_asr.json`, paired 95% CI vs ours):

| system | WER | Δ vs ours [95% CI] | CPU RTF |
|---|---|---|---|
| **ours, streaming @160ms** | **20.63%** | ref | **0.022** |
| Parakeet-CTC 0.6B, offline | 14.14% | +5.11 [+3.4, +7.1] pts worse for ours | 0.049 |
| Whisper-small, offline | 14.43% | +3.27 [+1.2, +5.9] pts worse for ours | 0.280 |
| Whisper large-v3-turbo, offline | 12.94% | +4.80 [+2.5, +7.4] pts worse for ours | 1.469 |

**Verdict: equal on read speech, clearly worse on meetings.** This corrects the earlier scorecard
line, which only had the read-speech comparison. On overlapping meeting audio, every offline
model beats our streaming model by a wide, statistically clear margin — even Parakeet-CTC at only
~2x our compute cost. Ours is 6-65x cheaper, and it streams at 80ms lookahead while the others see
the whole utterance, but the meeting-audio WER gap is real and should not be described as "equal".

## 2. VAD — better than every streaming VAD, with a caveat

(`research/BASELINES.md`, `runs/baselines_sd.json`) — our head F1 0.949 vs Silero 0.915, WebRTC,
MarbleNet (ported) 0.937; only offline pyannote (0.966, ~50x the CPU) beats it. Caveat: trained and
tested on the same word-level AMI label convention as the acoustic VADs were not.

## 3. Speaker verification — fixed this session, now competitive on held-out data

Original stage-1 head: 32.2% (AMI) / 42.3% (ICSI) within-meeting EER — far behind TitaNet-L (8.2% / 1.0%).
Diagnosis: the head read a near-uniform mix of all 17 encoder layers although speaker information
lives only in blocks 1-9. Reading block 4 alone plus a TitaNet-distilled relational loss
(`research/SPK_HEAD.md`, `runs/spk_head.json`) reaches:

| variant | AMI within-meeting EER | ICSI (held-out) within-meeting EER |
|---|---|---|
| original stage-1 head | 32.2% | 42.3% |
| **block-4 + relational distillation (shipped)** | **14.4%** | **5.2%** |
| TitaNet-L, reference | 8.2% | 1.0% |

Now within 5-6 points of TitaNet-L on a corpus never trained on, from 0.5M params on the shared
encoder pass. Shipped in `runs/stage1_served.afm`. Caveat: every embedding tested, including
TitaNet's, degrades to averaging two voices during overlapping speech.

## 4. Diarization — still worse; Sortformer stays in the product

Our head: 0.394 frame DER (AMI) / 0.354 (ICSI). Sortformer v2: 0.201 / 0.209. A full layer-routing
sweep (`research/LAYER_ROUTING.md`) found the best single-block tap (block 6) only closes ~0.02,
short of the pre-registered 0.03 bar, and confirmed on two seeds. **Verdict: worse; keep Sortformer
in the served pipeline.**

## 5. End-of-turn detection — the headline result

**AMI dev, n=974, all systems at matched ≤5% false-cutoff rate, 6s horizon:**

| system | miss %, all ends | miss %, floor-open ends (the ones an agent must answer) |
|---|---|---|
| **ours, hybrid (speaker-aware)** | **61.9%** | 45.9% |
| Silero VAD timeout | 72.7% | **26.7%** |
| Parakeet-Realtime-EOU (NVIDIA) | 70.1% | 42.9% |
| smart-turn v3.2 (Pipecat default) | 72.6% | 26.7% (== Silero) |
| LiveKit turn detector + timeout | ~74% | ~30% |

We beat every dedicated turn model overall; a plain silence timeout still beats us on the specific
ends an agent answers, because label-free primary-speaker selection is wrong 30-50% of the time on
multi-party audio (measured root cause, see §6).

**Held-out ICSI confirmation, n=1312 (`research/BASELINES.md`), frozen AMI thresholds, no refit:**
"head OR any-speaker Silero silence" beats head-OR-nothing by −9.3 [−14.7,−4.1] pts on floor-open
ends at 2.4% false cutoffs. A confidence-gated dynamic version of the same rule beats that by a
further −11.7 pts (CI excludes 0) at 1.8% FC — this is the best turn rule measured to date.

**TurnBench dev (Sesame's official scorer, dyadic, ≤10% FP), no retraining, thresholds frozen from
AMI** (`research/DYADIC.md`):

| system | recall | median latency |
|---|---|---|
| VAP (published) | 0.841 | 463 ms |
| **ours, hybrid head OR Silero per channel** | **0.835** | 1390 ms |
| Kyutai STT-VAD | 0.803 | 1024 ms |
| smart-turn v3 | 0.754 | 1010 ms |
| ours, head alone | 0.752 | 1058 ms |

On two-party audio our head alone equals smart-turn; the hybrid lands one point under VAP. The gap
to VAP is *latency*, not recall — we fire ~0.9s later. A predictive trigger (fires on predicted
future silence rather than waiting for it) trained on real two-party data (otoSpeech) closed part
of that gap: 0.862 recall / 1137ms median, held-out — the highest recall measured on TurnBench dev,
but still ~700ms behind VAP's median because it never fires *before* silence starts (VAP's P10
latency is negative; ours is not).

**Product measurement (Pipecat/LiveKit, live, 5 AMI windows):** the confirmed turn rules eliminate
cut-ins (0 vs 3-5 for a 1s timeout) but cost +0.66-1.2s more median dead air on this sample.
Default stays the 1s timeout; the new rules ship as opt-in.

## 6. Root cause and negative results (why open-end turn detection is hard)

- **Root cause, confirmed:** label-free primary-speaker selection is wrong 30-50% of the time on
  meeting audio; with oracle speaker identity, misses drop from 62% to 29%. Binding the primary
  speaker at the moment the agent stops talking (product-realistic, no labels) recovers most of
  this: 51.6% overall / 34.0% open-end misses, CI excludes 0 vs the label-free baseline.
- **Semantic end-of-turn (transducer YIELD/HOLD tokens): dead.** The token never learns to fire on
  a frozen encoder (new-vocab rows can't move fast enough under the WER-safe learning rate); scored
  by posterior it's far worse than a silence timeout (94% miss). NVIDIA's own end-to-end EOU model
  has the same problem: fires from silence, not from the words (median lag 2.16s after the last word).
- **State contamination:** wrong speaker conditioning leaves measurable, persistent damage in the
  turn head's recurrent state (half-life ~1.5-2s); learned "repair" isn't worth building — bounded
  replay is cheap enough on CPU that it dominates on cost, just not on latency.
- **A 0.6B NVIDIA streaming encoder carries no more speaker information than our 115M one** — ruled
  out as a path to closing the identity gap.
- **A gated/early-exit encoder costs more than it saves** once cache re-priming is counted.

## 7. Language ID — worse, cheaper; ships off by default

Our head: 75.0% (2s) / 89.3% (full utt) on FLEURS-17. NVIDIA AmberNet: 95.1% / 99.5%. On accented
English (EdAcc) our head calls only 19-31% of segments English. **Ships off by default;
`serve --lid ambernet` wires in NVIDIA's model at ~0.02-0.03 added RTF.**

## 8. Still running — not yet in this document

- **`research/E2E_FINAL.md` / `runs/e2e_final.json`**: full live Pipecat-default-stack vs
  LiveKit-default-stack vs ours (product default) vs ours (best rules), on AMI + TurnBench + otoSpeech
  clips, per-component RTF breakdown (VAD/turn/ASR/diarization/LID/framework overhead). This is the
  piece the user is waiting on; ~4 of ~28 run-batches remain as of this writing.
- **`research/FINAL_REPORT.md` / `.html`**: the polished, single-document version of everything above
  plus §8's numbers, written by the report agent once §8 lands.

Sources for every number above: `runs/final_asr.json`, `research/BASELINES.md` +
`runs/baselines_sd.json` + `runs/baselines_turn.json` + `runs/baselines_turn_icsi.json` +
`runs/baselines_turn_icsi_dyn.json`, `research/SPK_HEAD.md` + `runs/spk_head.json`,
`research/LAYER_ROUTING.md` + `runs/layer_routing.json`, `research/EOT_BENCH_V2.md` +
`runs/turn_trail6_hybrid_leakfree.json` + `runs/turn_trail6_primary_titanet.json`,
`research/DYADIC.md` + `runs/dyadic_bench.json` + `runs/turnbench_latency.json` +
`runs/dyadic_train.json`, `research/CONTAMINATION.md` + `runs/contamination.json`,
`research/YIELD_TOKENS.md` + `runs/yield_tokens.json`, `research/ENC_0P6B.md` + `runs/enc_0p6b.json`,
`research/VAD_LAYERS.md` + `runs/vad_layers.json`, `research/LID.md` + `runs/lid.json`,
`research/INTEGRATION.md` + `runs/integration_deadair_shipped.json`.
