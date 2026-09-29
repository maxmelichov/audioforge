# What the two single-model images show

This covers `architecture_single.png` and `results_single.png` in `demo/images/`, each with a `_square` version.
Full-size copies are in `demo_out/images/` on the SSD.

They describe **single-model mode**: the product for a known user who has given a 5-second voice sample. It runs
one frozen NVIDIA streaming speech model (115 M parameters) with our small heads and no diarizer:
`serve --turn-input tsvad --diar-off`. Background is in research/IMPROVE_115M.md Part A and research/IMPROVEMENTS.md
§1.

Every number on the results image is generated from the run files by `demo/images/redesign/export_single.py`
(`numbers_single.json`), and `render.py` checks each one on the rendered page. The pages are
`redesign/arch_single.html` and `redesign/results_single.html`. The layout follows the post images and
`DESIGN_NOTES.md`: white page, one slate block for NVIDIA's frozen model, orange for our heads, grey for everything
else.

## architecture_single.png

**Title and subtitle.** "audioforge: one model, three small heads, following one voice". The subtitle names the
three questions the heads answer: is someone speaking, is it you, and is your turn over. It also says the words
stream out while you talk.

**The slate block: NVIDIA's streaming speech model.** This is NVIDIA's cache-aware FastConformer
(`stt_en_fastconformer_hybrid_large_streaming_multi`). It is drawn as 17 slabs, one per layer, with layer 1 on the
left and 17 on the right. It is frozen: we never change its weights (docs/ARCHITECTURE.md). Audio enters from the
left ("you speak") in 160 ms chunks. Layer 4 is the only orange slab because two of our heads read it.

**"words".** NVIDIA's own decoder turns the output of layer 17 into text. The text streams out as you speak: one
update per 160 ms chunk (`partial` messages), and a `final` at each turn end.

**Top pill, "Is someone speaking?"** This is our VAD head, 33 K parameters. Every 80 ms it gives a speech
probability, read from layer 4 of the model (at a 7.5 % false-alarm rate it misses 10.6 % of speech vs 12.3 % for NVIDIA MarbleNet and 13.8 % for Silero): the same layer the speaker head and the TS-VAD head read
(`runs/stage1_served_v2.afm`, research/VAD_SINGLE.md). That layer does as well as the earlier learned mix of all 17
layers. F1 is 0.951 on AMI and 0.898 on ICSI, against 0.949 and 0.900 for the all-layer head. Language ID, TS-VAD and
the streaming words are unchanged. In architecture_v7 its stem leaves slab 4; earlier images drew the all-layer mix.

**Lower-left pill, "Is it you speaking?"** This is the target-speaker head (TS-VAD, 0.26 M, `runs/tsvad_spk.pt`). It
reads layer 4. Your 5-second voice sample (the small grey box on the far left) is turned into a voice print by our
speaker head on the same layer, and the TS-VAD head is conditioned on that print. Every 80 ms it gives two
probabilities: that you are speaking, and that someone else is (`audioforge/tsvad_stream.py`, `TSVADTrack`). The
sample can be stored at enrolment or taken from the first 5 s of the user's speech.

**Lower-right pill, "Is your turn over?"** This is our turn head (a GRU, 0.32 M). It runs on a second pass of the
same frozen model over the same audio. That pass is conditioned on "your voice track": the TS-VAD head's
probabilities, fed in as the primary speaker's activity (`serve.py` `_act_tsvad`, `run_turn_on_diar`; the speaker
information enters the encoder at blocks 1 and 3). It gives an end-of-turn probability every 80 ms, and a
turn-end event fires when that probability crosses its threshold. The "your voice track →" label between the two
lower pills shows this dependency. It is true in this mode: the "is it you?" output feeds the turn pass.

**"Optional, alongside": Parakeet.** NVIDIA Parakeet-TDT 0.6B v3 can rewrite each finished turn for a better final
transcript (`--final-asr`). It is optional, and it is the only other model. The note on the right says so:
everything else comes from the one model.

**What is not in this mode.** There is no Nemotron-3 or Sortformer diarizer: `--diar-off` replaces the diarizer's
columns with the TS-VAD track. The "Today" row shows the usual default stack for comparison: Silero VAD, a turn
detector, Whisper, one model per question.

**Engineers footer.** Encoder details, and head sizes with the layers each reads: VAD 33 K on all 17 layers;
speaker 0.5 M on layer 4, used to embed the voice sample; TS-VAD 0.26 M on layer 4; turn 0.32 M, a GRU on the
second pass conditioned on your voice track. It ends with the flags that select the mode.

## results_single.png

The legend is orange for audioforge in single-model mode and grey for the system it is compared with. Each panel
says what that system is. Arrows mark the direction: "↑ higher is better" or "↓ lower is better".

**Finds your voice (↑).** How well the system tracks the moments you speak, given the same 5-second sample, measured
as frame F1 on the primary speaker of each meeting. Scores: ICSI 0.88 vs 0.69, AMI 0.74 vs 0.66.
- Grey is **NVIDIA Streaming Sortformer v2** (not Nemotron-3), with one of its speaker columns bound to your voice
  by the same sample. For each set we show the better of the two binders that were tried: the speaker head on ICSI
  (0.690), TitaNet-L on AMI (0.657).
- Source: `runs/improve_115m.json`, `frame > {icsi, ami} > primary > {tsvad_spk_vp5p0, sortformer_vp_spk_vp5p0,
  sortformer_vp_titanet_vp5p0} > all > f1`.
- Offline benchmark: AMI and ICSI dev meetings, with the sample taken from elsewhere in the same meeting.

**Knows when you're done (↓).** Missed turn ends in meetings at the same false-cut-off budget: ICSI 20 vs 69 %, AMI
39 vs 62 %. Each pair stays together.
- Orange is the single model with your sample: the turn head fed the TS-VAD track.
- Grey is our two-model stack without a sample: the same turn head fed a Sortformer column.
- With the sample, false cut-offs stay about the same (AMI 6.5 → 5.2 %, ICSI 5.9 → 6.0 %, in the footer).
- Sources: `runs/improve_115m.json` (`turn_bench > {ami, icsi} > systems > hybrid_tsvad_spk > 6s`),
  `runs/baselines_turn.json` and `runs/baselines_turn_icsi.json`, via `numbers_b.json`.

**Hears speech better (↑).** Speech-detection F1 out of 100: audioforge 94.9, NVIDIA MarbleNet 93.7, Silero 91.5,
on AMI dev windows (`runs/baselines_sd.json`). This is the same VAD head as in the two-model mode.

**Talks over you less (↓).** Times per call the agent started talking while the user was still speaking. All three
systems are pooled over the same 69 live Pipecat 1.12 sessions (37 recorded phone calls, TurnBench calls and AMI
clips; five set and audio conditions).
- Single model: 0.52 per call.
- Our two-model default: 0.93.
- Pipecat's default stack: 1.99.
- The single-model sessions were run on 2026-09-28 (`runs/e2e_tsvad.json`). The other two are the stored records on
  the same clips, protocol and scorer (`runs/e2e_final.json`).

**Half the compute (↓).** The fraction of real time the server spends processing on 2 CPU threads: 0.34 for the
single model vs 0.80 for our two-model stack, where the diarizer takes most of the compute.
- 0.34 is the median over the 69 single-model sessions' `server_stats.rtf`. These are the raw session records on the
  SSD (`scratch/e2e_tsvad/runs/pipecat_T_*.jsonl`); `runs/e2e_tsvad.json` stores no RTF.
- 0.80 is the median of the five per-set medians in `runs/e2e_final.json`.

**Better final transcript, optional (↓).** Word errors on AMI meetings: 9.5 % with the optional Parakeet-TDT pass on
each finished turn, vs 14.4 % for Whisper small (`runs/hybrid_asr.json`, `runs/final_asr.json`). It is labelled
optional because Parakeet is the one model outside the single model.

**Footer.** Where each number comes from, in plain words: meetings offline with a 5-second sample, the live
sessions and clips, the compute measure.

## What the images do not show, and why

- **General room diarization** ("who is who" for everyone in a meeting) is not part of single-model mode. It follows
  one known voice. Labelling every speaker needs the diarizer, which the two-model mode keeps.
- **Answer speed on calls.** In the same 69 live sessions the single model left 45 % of the user's turns unanswered
  within 3 s, against 35 % for our two-model default. That is worse, with a confidence interval that excludes zero
  (research/IMPROVEMENTS.md §1), so it is not presented as a strength.
- **The streaming transcript** from the frozen model is 24 % word errors on AMI meetings (`runs/hybrid_asr.json`).
  The image shows only the optional Parakeet final transcript. The words-as-you-speak stream is described as a
  capability, not ranked.
- **The TurnBench end-of-turn score (85 %)** on the two-model results image was measured with the turn head fed a
  per-channel Silero track (`runs/turnbench_latency.json` `inputs`), not the single-model input. It is not shown
  here and was replaced by the optional-transcript panel.

## Engineering details (moved off architecture_v5)

- **Encoder.** NVIDIA FastConformer, 109 M parameters, frozen and cache-aware (70-frame attention context). It uses
  80 log-mel bins and ×8 subsampling, giving 80 ms frames processed in 160 ms chunks.
- **VAD head.** 33 K parameters, on layer 4 since `runs/stage1_served_v2.afm` (AMI F1 0.951 / ICSI 0.898, vs 0.949 / 0.900 for the earlier all-17-layer mix; research/VAD_SINGLE.md).
- **Voice print.** 192-d, made once by the speaker head (0.5 M) from the 5 s sample's layer-4 features in the same
  frozen model (`tsvad_stream.voiceprint`).
- **TS-VAD head.** 0.26 M, reads layer 4 and outputs [P(you), P(someone else)] every 80 ms.
- **Second pass.** The same encoder weights run again and reuse the first pass's front end. Speaker kernels at blocks 1
  and 3 (`speaker_kernel_layers: [0, 2]` in `runs/stage1_served.afm`) take your voice track.
- **Turn head.** A GRU (0.32 M) reads the top layer of the second pass.
- **Flags.** No diarizer: `--turn-input tsvad --diar-off`. Parakeet-TDT runs only with `--final-asr`.

## results_v7 methodology

`results_v7.png` is audioforge in the shipped single-model mode (`--mode single`) against named external systems
only. Its numbers are the same as in results_v6. Each hero line gives the value, the unit, what is measured and an
arrow for which way is good; the orange audioforge bars carry no value labels because the hero line has them. The
caveats that no longer fit on the image are listed here.

- **Finds your voice.** Frame F1 for the primary speaker of each meeting, offline, AMI dev and held-out ICSI. The grey
  bar is NVIDIA Nemotron-3, given the same stored 5 s sample: one of its columns is bound to the voice, and the better
  of two binders is shown per corpus (the speaker head on ICSI, 0.698; TitaNet-L on AMI, 0.648). AMI is in
  Nemotron-3's training data (`runs/improve_115m.json` frame primary).
- **Hears speech.** Speech frames missed at a false-alarm rate of about 7.5 %, on 64 AMI dev windows. The
  metric is the one in FINAL_REPORT §2.
  - audioforge is the shipped block-4 VAD head: 10.6 % at FPR 0.0749 (`runs/vad_single.json` eval > ami_dev > L3 >
    at_fpr0.075).
  - NVIDIA MarbleNet: 12.3 % at FPR 0.0757 (`runs/baselines_sd.json` vad > marblenet_frame_vad_v2 > sweep > 0.9).
  - Silero: 13.8 % at FPR 0.0751 (sweep > 0.5).
  - "14 % less missed" is relative: 1 − 10.62 / 12.28.
  - At the 0.5 threshold, F1 is 0.951 for the block-4 head against 0.937 for MarbleNet and 0.915 for Silero. F1 is
    no longer shown, because those three bars look identical on a 0–100 axis.
- **Missed turn ends.** AMI dev, 974 turn ends, 6 s horizon, each system at ≈ 5 % per-turn false cut-offs
  (cross-fitted; realised 5.5 % for audioforge, 4.8–5.7 % for the others).
  - audioforge runs hybrid_dyn on the TS-VAD track with a stored 5 s sample.
  - The external detectors have no voice sample: NVIDIA Parakeet-Realtime-EOU, Pipecat smart-turn v3 + Silero
    timeout, Silero timeout, and the LiveKit turn detector (text, en) + timeout.
  - On held-out ICSI (1312 turn ends) audioforge misses 18.7 % against 84.4 % for the Silero timeout.
  - "51 % fewer missed" is relative: 1 − 34.24 / 70.07.
- **Responds after your turn** and **Talks over you less.** Live, all 69 sessions through Pipecat 1.12 and LiveKit
  Agents 1.8 with default settings.
  - The sessions are 16 real phone calls and 16 TurnBench calls, each run as the mixed call and the user channel,
    plus 5 AMI windows.
  - audioforge is the shipped `--mode single` rule (single_S in `runs/single_model.json` table live_69).
  - "40 % fewer" is relative: 1 − 0.667 / 1.116.
- **Words on screen sooner.** Median time from the user's first speech onset to the first words on screen, over the
  64 two-party sessions (research/SINGLE_MODEL.md). Every system ran on the same Mac at 1x, on CPU with 2 threads per
  model process and no GPU; the defaults' faster-whisper small ran in-process (research/E2E_FINAL.md). "2.6× faster" is 2.812 / 1.081 s.
- **Not shown.**
  - General room diarization: single-model mode follows one enrolled voice.
  - Transcript accuracy: live streaming WER is 23.2 % against Pipecat 23.5 % and LiveKit 19.3 %, which is not a
    strength. The optional Parakeet pass is not live.
