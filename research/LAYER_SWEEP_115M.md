# Layer sweep: which block of the 115M each head should read

2026-10-01. This is the same protocol as research/LAYER_SWEEP_0P6B.md, run on the default 115M core (17 blocks × 512,
the frozen encoder of `stage1_served_v3.afm`). Script: `scripts/research/core_0p6b_heads.py --core 115m` (stages
`vad_feats`, `sweep_vad`, `sweep_spk_feats` / `sweep_spk`, `sweep_tsvad`, `sweep_seg_feats` / `sweep_seg`,
`sweep_report`). Results: `runs/core_115m.json` (`sweep_*`). Caches: `scratch/core_115m/` on the SSD. Every job ran
through `scripts/dev/gate.sh` on MPS, one at a time.

## STATUS (stopped 2026-10-01 09:14 for the 0.6B turn-taking work)

- **Done:** the VAD, speaker and TS-VAD sweeps are complete: all 17 blocks, the softmax mix of all 17, and the concat
  of the top-3 singles.
- **Partial:** the v5 classifier sweep has blocks 1-4 only. Blocks 5-17, the mix (odd blocks 1, 3, .., 17) and the
  top-3 concat have not run. Resume with `sweep_seg --core 115m --steps 500 --budget 480`. The 17-block cache of the
  1000-clip subset and the 399 smart-turn test clips is already built (`scratch/core_115m/sweep/blk17`).
- **Not started:** steps 2-6 of the brief (VAD recipe, TS-VAD v2, speaker head, v5 retrain, served v0.4). Their code
  is in place in the same script, but none of it has been run:
  - `build115`: a candidate `.afm` with the heads replaced; the frozen tensors are checked equal.
  - `eot_dump --core 115m --tag T`: the served-engine dumps of each candidate.
  - `tswer115`: target-speaker WER of any TS-VAD head on AMI / ICSI / live, paired against the shipped head.
  - the 115M branch of `vad_feats` / `vad2` / `tsvad_sim` / `tsvad_train` / `tsvad_eval`.
  - The queued chain (VAD training-window caches with SpecAugment views, then the GRU-64 VAD variants on blocks 4 / 3
    / 3+4 / mix 1-6 / mix of the odd blocks, plus an MLP control) was cancelled before it started.
- **Nothing ships from this pass.** No served head, asset or hub pin changed.

## Summary (what the sweep says so far)

| head | served on the 115M today | sweep says (held-out data only) | change the block? |
|---|---|---|---|
| VAD | block 4 MLP | The blocks 3 and 4 singles tie at the top (held-out mean AUC 0.9694). The mix of all 17 is 0.9710 (selected); concat 3+4+7 is 0.9674. Room-tone p95: b4 0.034, mix 0.054. | Not on the probe alone. The +0.0016 for the mix is under the ~0.01 noise floor of these sets. Whether the mix helps under the full recipe (GRU, SpecAugment views, room tone) is part of step 2, which was not run. |
| speaker | block 4 | **Block 4**: within-meeting EER AMI 20.0 % / ICSI 6.9 %. Next best: block 6 (21.1 / 10.4), block 5 (22.4 / 7.9), mix (21.1 / 10.2). | **No.** Block 4 is already the tap. |
| TS-VAD | block 4 (shares the speaker head's tap) | Concat 3+4+5 scores 0.7409 (selected) and block 5 0.7407, against block 4 at 0.7342 (mean held-out target F1). | **No, for now.** The gap is 0.007, under the noise floor. The server's TS-VAD reads the speaker head's tap, and block 5 is worse for the speaker head (22.4 vs 20.0 % AMI EER). Moving TS-VAD alone would need a second tap in the server and a full-recipe check at block 5. |
| v5 turn classifier | block 8 (c5) | Partial, blocks 1-4: end-vs-pause AUC 0.749 / 0.763 / 0.780 / 0.783. Smart-turn test 83.5-90.0 %. | Open: blocks 5-17 have not run. |

**Depth profile.** It matches the 0.6B's, scaled to 17 blocks:
- Speaker identity and target-speaker tracking peak early (blocks 4-5) and fall to chance at the top (block 17: 45.9
  / 43.2 % EER, target F1 0.43).
- VAD is flat over blocks 3-10 on AMI (held-out AUC 0.993-0.994).
- On ICSI, VAD falls from block 7 on (0.944 → 0.891 at block 17), and room-tone false alarms rise with depth (p95
  0.03-0.05 at blocks 3-7, 0.25-0.33 at blocks 15-17).
- So the early blocks transfer across rooms, as on the 0.6B.

**Cost.** A head reading another block costs no encoder compute: the served pass runs all 17 blocks for the
transcript. A concat input triples a head's input width.

## Protocol

This is the protocol of research/LAYER_SWEEP_0P6B.md, with the 115M's dimensions. Selection is on held-out data
only:

- **VAD:** the served FrameHead probe (Linear(512, 64)-SiLU-Linear), 1500 × 2048 frames.
  - Train: the 300 AMI train windows minus 2 held-out meetings (TS3011b, ES2015c), plus 15 % room tone.
  - Scored on the held-out AMI meetings and on 150 ICSI train windows, each with room-tone validation negatives.
- **Speaker:** pooled mean + std of the block → Linear(1024, 192), distilled to TitaNet-L (0.5 × cos + relational)
  on 200 LibriSpeech-100 speakers (3589 utterances).
  - Scored by within-meeting EER on 928 enrollment clips of the 24 AMI / ICSI train meetings, never seen by the probe.
- **TS-VAD:** TSVADHead (hidden 128, prenet), 1000 × 32 windows, overlap weight 2.5.
  - Enrollment: a clip from elsewhere in the meeting, embedded by the shipped 115M speaker head.
  - Train: AMI / ICSI train windows minus the held-out meetings (AMI TS3011b, ES2015c; ICSI Bro026, Bmr022).
    Scored on those held-out meetings.
- **v5 turn classifier:** SegTurn (2 × 256, the c5 architecture), 500 × 256 steps, on 1000 turn clips (50 % oto /
  AMI / ICSI, 30 % smart-turn, 20 % cuts).
  - Inputs: the 115M's own served turn inputs (turn_v4 / turn_v5 caches) and every block re-encoded.
  - Selection: turn_v5's held-out end-vs-pause AUC, with the held-out smart-turn clip accuracy.
  - The smart-turn test accuracy (399 clips) is shown for information only.

The held-out sets are small, so differences under ~0.01 AUC / F1 are noise. A block is changed only where the full
recipe confirms the probe.

## Tables

### VAD

| config | AMI held-out AUC | ICSI held-out AUC | AMI F1 | ICSI F1 | room-tone p95 | selection (mean AUC) |
|---|---:|---:|---:|---:|---:|---:|
| b1 | 0.992 | 0.9358 | 0.955 | 0.8817 | 0.0816 | 0.9639 |
| b2 | 0.9924 | 0.9406 | 0.9551 | 0.8849 | 0.0674 | 0.9665 |
| b3 | 0.9939 | 0.945 | 0.9596 | 0.8863 | 0.0451 | 0.9694 |
| b4 | 0.9943 | 0.9446 | 0.9623 | 0.8866 | 0.034 | 0.9694 |
| b5 | 0.9937 | 0.9419 | 0.9618 | 0.8884 | 0.0393 | 0.9678 |
| b6 | 0.9938 | 0.9435 | 0.9605 | 0.885 | 0.0356 | 0.9687 |
| b7 | 0.9943 | 0.9437 | 0.9639 | 0.8809 | 0.0316 | 0.969 |
| b8 | 0.9939 | 0.9406 | 0.9618 | 0.8859 | 0.046 | 0.9672 |
| b9 | 0.9936 | 0.9382 | 0.9637 | 0.8865 | 0.0825 | 0.9659 |
| b10 | 0.9938 | 0.9341 | 0.9638 | 0.8866 | 0.0855 | 0.964 |
| b11 | 0.9932 | 0.9333 | 0.962 | 0.8816 | 0.146 | 0.9632 |
| b12 | 0.9928 | 0.9266 | 0.9607 | 0.8785 | 0.1616 | 0.9597 |
| b13 | 0.9925 | 0.9192 | 0.9614 | 0.8798 | 0.2064 | 0.9559 |
| b14 | 0.9913 | 0.9087 | 0.9585 | 0.8788 | 0.2065 | 0.95 |
| b15 | 0.9894 | 0.8927 | 0.9584 | 0.8781 | 0.3266 | 0.941 |
| b16 | 0.9861 | 0.8837 | 0.953 | 0.878 | 0.3191 | 0.9349 |
| b17 | 0.9899 | 0.8913 | 0.9608 | 0.8702 | 0.2503 | 0.9406 |
| **mix_all** | 0.9947 | 0.9473 | 0.9653 | 0.8863 | 0.0541 | 0.971 |
| concat 3+4+7 | 0.9941 | 0.9407 | 0.9642 | 0.8866 | 0.0092 | 0.9674 |

selected by `val_auc`: **mix_all**

### speaker

| config | AMI within-meeting EER | ICSI within-meeting EER |
|---|---:|---:|
| b1 | 29.9 | 14.1 |
| b2 | 24.8 | 11.8 |
| b3 | 23.2 | 9.1 |
| **b4** | 20.0 | 6.9 |
| b5 | 22.4 | 7.9 |
| b6 | 21.1 | 10.4 |
| b7 | 26.8 | 13.6 |
| b8 | 32.2 | 17.8 |
| b9 | 36.9 | 24.0 |
| b10 | 38.4 | 29.7 |
| b11 | 40.9 | 35.1 |
| b12 | 43.2 | 37.7 |
| b13 | 43.1 | 40.3 |
| b14 | 44.8 | 41.4 |
| b15 | 43.9 | 43.3 |
| b16 | 45.3 | 43.7 |
| b17 | 45.9 | 43.2 |
| mix_all | 21.1 | 10.2 |
| concat 4+5+6 | 21.7 | 9.0 |

selected by `score`: **b4**

### TS-VAD

| config | AMI target F1 | AMI overlap recall | ICSI target F1 | ICSI overlap recall | selection (mean F1) |
|---|---:|---:|---:|---:|---:|
| b1 | 0.6087 | 0.8518 | 0.8094 | 0.6152 | 0.709 |
| b2 | 0.6069 | 0.9012 | 0.8087 | 0.6517 | 0.7078 |
| b3 | 0.6016 | 0.8841 | 0.8325 | 0.6795 | 0.717 |
| b4 | 0.6303 | 0.8989 | 0.8382 | 0.6688 | 0.7342 |
| b5 | 0.626 | 0.8757 | 0.8553 | 0.671 | 0.7407 |
| b6 | 0.5988 | 0.8198 | 0.8327 | 0.6292 | 0.7157 |
| b7 | 0.5776 | 0.8198 | 0.8161 | 0.6559 | 0.6968 |
| b8 | 0.5473 | 0.8121 | 0.7749 | 0.6249 | 0.6611 |
| b9 | 0.5039 | 0.7433 | 0.7231 | 0.6152 | 0.6135 |
| b10 | 0.4722 | 0.6881 | 0.682 | 0.6217 | 0.5771 |
| b11 | 0.4559 | 0.67 | 0.6106 | 0.5691 | 0.5333 |
| b12 | 0.44 | 0.576 | 0.5264 | 0.4984 | 0.4832 |
| b13 | 0.4745 | 0.5806 | 0.5053 | 0.478 | 0.4899 |
| b14 | 0.377 | 0.475 | 0.4671 | 0.3794 | 0.4221 |
| b15 | 0.4544 | 0.5347 | 0.4184 | 0.4191 | 0.4364 |
| b16 | 0.4516 | 0.5124 | 0.407 | 0.2958 | 0.4293 |
| b17 | 0.4153 | 0.4333 | 0.4349 | 0.3408 | 0.4251 |
| mix_all | 0.6025 | 0.8721 | 0.8114 | 0.6731 | 0.7069 |
| **concat 3+4+5** | 0.628 | 0.9012 | 0.8538 | 0.6613 | 0.7409 |

selected by `score`: **concat_top3**

### v5 turn classifier

| config | end-vs-pause AUC | AUC, first quiet frames | held-out smart-turn clip acc % | end vs cut AUC | smart-turn test acc % |
|---|---:|---:|---:|---:|---:|
| b1 | 0.749 | 0.7236 | 78.1 | 0.6983 | 83.5 |
| b2 | 0.7625 | 0.7524 | 84.0 | 0.7192 | 86.7 |
| b3 | 0.7798 | 0.7734 | 81.2 | 0.7586 | 87.5 |
| **b4** | 0.7833 | 0.7723 | 84.4 | 0.7435 | 90.0 |

selected by `auc`: **b4** (of blocks 1-4 only: the v5 sweep is partial, see STATUS)
