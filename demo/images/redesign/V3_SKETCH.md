# V3 hierarchy sketch (before HTML)

Source of truth: demo/images/EXPLAINER.md. Code checks behind the architecture:
- The voice print is made once: block-4 frames of the sample go through the speaker head, giving a unit-norm 192-d
  print (`tsvad_stream.voiceprint` / `embed_frames`).
- TS-VAD reads block 4 of pass 1 and gives [P(you), P(other)] every 80 ms.
- The turn head reads the TOP layer of a second, speaker-conditioned pass (`server/streams.py`, kernel path). That
  pass reuses pass 1's subsampled frames, and the speaker kernels sit at `speaker_kernel_layers: [0, 2]` (blocks 1
  and 3 of 17, read from `runs/stage1_served.afm`).
- Parakeet runs only with `--final-asr`, on the finished turn.

## Image 1: architecture_v3 (computation graph, left to right, two passes of ONE model)

```
 Audio ~~~~~~~┬───────────────▶ ┌ Frozen FastConformer · 17 layers ─────────────┐        ▲ Speech?
              │     ╭──── all layers · 80 ms · 33K ────╮  (bracket over every slab)──┘
              │     ┃1┃2┃3┃4┃5┃ … ┃17┃  PASS 1 ──────────▶ [RNNT decoder] ──▶ Streaming words · 160 ms
              │         │ layer 4                                    ║
              │         ▼                                            ║ same weights
 [5 s voice sample]─▶[speaker head · once]─▶ print ─▶[TS-VAD 0.26M]─▶ Your voice track ──▶ Is it you?
              │                                          │ you / other / no one · 80 ms
              │                ┌─────────────────────────┘ (into blocks 1 and 3)
              │         ▼  ▼                                         ║
              └───────▶ ┃1┃2┃3┃4┃5┃ … ┃17┃  PASS 2 ──▶ [Turn head · GRU 0.32M] ──▶ Turn over?  ⌇⌇ threshold ─▶ Turn end
                                                                                                    │
                                                                    ┌ ─ ─ Optional ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ┐
                                                                    │  [Parakeet · finished turn] ─▶ Final transcript │
                                                                    └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
 footer (tiny): engineering details
```

Hierarchy: Level 1 (≈32 px bold): Audio, Frozen FastConformer, Pass 1, Pass 2, Speech?, Is it you?, Your voice
track, Streaming words, Turn over?, Turn end, Optional, Final transcript. Level 2 (22 px grey): 17 layers, layer 4,
all layers, every 80 ms, 160 ms, 33K, 0.26M, 0.32M GRU, blocks 1 and 3. Level 3: the footer.

Reuse treatment: the two passes are two identical slate lanes, column-aligned layer for layer, joined by a vertical
"same weights" double rule. Pass 2 is a second run of the same weights, not a second model.

Colours: slate = NVIDIA frozen (encoder, RNNT decoder); orange = audioforge heads and their arrows; grey = inputs,
audio and optional; a dashed grey boundary only around the Optional zone.

## Image 2: results_v3 (six cards, each with its own scale and the best encoding)

```
┌ Finds your voice ─────────┐ ┌ Knows when you're done ───┐ ┌ Hears speech better ──────┐
│ ICSI  ○──────────●        │ │ ICSI  ●◀─────────────○     │ │ ████████████████████ 94.9 │
│ AMI     ○───●             │ │ AMI     ●◀──────○          │ │ ███████████████████▏ 93.7 │
│ 0         0.5          1  │ │ 0 %            50 %   100 %│ │ ███████████████████  91.5 │
│ 0.88   speaker tracking F1│ │ 20 %  missed turn ends     │ │ 94.9  speech detection F1 │
└───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
┌ Talks over you less ──────┐ ┌ Uses less compute ────────┐ ┌ Better final transcript ──┐
│ ██ 0.52                   │ │ ████ 0.34      ┆ real time │ │ OPTIONAL FINAL TRANSCRIPT │
│ ████ 0.93                 │ │ █████████ 0.80 ┆ = 1.0     │ │   ●◀──────○  (0–20 %)     │
│ █████████ 1.99            │ │                            │ │ 9.5 %  WER                 │
│ 0.52  interruptions / call│ │ 0.34  server RTF           │ │ 34 % fewer word errors     │
└───────────────────────────┘ └───────────────────────────┘ └───────────────────────────┘
```

Direction is drawn, not written:
- Dumbbells point toward the better end: toward 1 for F1, toward 0 for misses and WER.
- The bars start at zero, so "shorter is better" reads as distance to zero.
- Compute has a "real time = 1.0" rule.
- Each card has its own axis and unit.
- Deltas are small grey text; measured values are big.

Footer: the fairness context from EXPLAINER.md.
