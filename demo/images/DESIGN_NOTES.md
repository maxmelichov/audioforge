# Design notes: redesigned architecture and results images for the post

The goal: make the two post images look like the figures published by NVIDIA, Google, Meta, Anthropic and Datawrapper,
and not like generated slides. References (34 images at source resolution, with URL, author and a note on each):
`/Volumes/ExternalSSD/nvidia-audio-models/demo_out/refs_images/` (`sources.md`, `board.png`). Sources:
`demo/images/redesign/` (`arch.html`, `results.html`, `style.css`, `render.py`, `contact.py`).

## What the good references have in common

1. **Paper-white background.** Every architecture figure and every results chart from NVIDIA's Nemotron-3 blogs,
   Meta's Llama 4 post, Google Research, Anthropic, Sesame, Kyutai and Datawrapper is on white or very light grey
   (#F5F6F8 to #FAFAF9). Only two good ones are dark: Apple's AFM figure and the dark copy of DeepMind's Gemini Live
   chart. Both use near-black with dark-grey marks and **one** light accent. The dark ones that fail (Deepgram's
   rainbow bars, VoiceArena's vendor colours) fail on colour, not on darkness.
2. **One accent.** The accent marks the thing the post is about: Anthropic's own models in clay, DeepMind's model in
   one blue, NVIDIA's model in green, Llama 4's "Shared Expert" as the only blue box. Everything else is grey. There is
   no second hue for decoration.
3. **Thin grey outlines and pale container panels.** Groups are pale panels with 1-2 px borders (Nemotron-3 panels
   01/02/03, the Llama 4 "Mixture of Experts" panel). The component being described is either the only dark block
   (Nemotron-3's "31-layer Transformer", Llama 4's slate "Attention") or the only accent block.
4. **Generous whitespace, consistent rounded rectangles.** One corner radius per figure (about 12-18 px at 1080p),
   the same padding in every box, wide gutters.
5. **Sans type in two weights.** Bold for names and titles, regular for everything else, and a grey for secondary
   text. Labels sit inside or right on the shapes: "8 arrival-ordered channels", "Expert 0", "Model dimension: 7,168"
   (Alammar's dotted leaders).
6. **Few thin arrows.** 1.5-2.5 px lines, small open chevrons (Llama 4), right angles, no curves unless they carry
   meaning.
7. **No effects.** No glows, gradients, drop shadows, glass or "dark hacker" palette in any of the good references.
8. **Charts: ranked bars with direct labels.** The best charts sort the bars (FT Visual Vocabulary "Ranking",
   Observable's sorted-bars example), print the value at the end of each bar (NVIDIA Nemotron-3, Anthropic, DeepMind),
   give the lead model the accent and the rest one grey (DeepMind τ³-banking, Anthropic SWE-bench, Datawrapper
   "highlight one"), use light gridlines or none, put the unit in the title or subtitle (Datawrapper, "Diarization
   error rate (%) — lower is better") and end with one small source line (DeepMind's "Methodology: ...").
9. **The title says what the chart shows, the subtitle says how it is measured** (Datawrapper text guide; Anthropic
   "Software engineering / SWE-bench verified").

Counter-examples in the folder: `res_11` (one colour per vendor plus logos), `res_12` (rainbow on black), `res_03`
(eight colours for eight models), `arch_04` (heavy saturated green blocks, NVIDIA's older style).

## The choices copied, and from where

| choice | taken from |
|---|---|
| white canvas `#FFFFFF`, pale panel `#F5F6F8`, 1.5 px borders `#C3C9D1` / `#DDE1E6` | NVIDIA Nemotron-3 diarization pipeline (`arch_01`), Llama 4 MoE (`arch_05`) |
| the frozen NVIDIA model drawn as the **only dark block** (slate `#2F3B4A`), white bold name inside, one grey line under it | Nemotron-3 "31-layer Transformer" block (`arch_01`), Llama 4 slate "Attention" (`arch_05`) |
| the 17 layers drawn as slabs **inside** the block, the tapped layer 4 as the only accent slab | Alammar's DeepSeek-R1 block stack with the routed experts coloured in place (`arch_10`) |
| one accent, orange `#E8640C`, only for audioforge's parts (the three heads, the tap lines, our bars); NVIDIA parts and competitors grey or slate | DeepMind τ³-banking (`res_04`), Anthropic SWE-bench (`res_02`), Llama 4 "Shared Expert" (`arch_05`) |
| heads as white rounded cards with an accent outline, the answer as a pale-tint pill inside ("yes", "Person 2", "not yet") | Nemotron-3 white inner cards (`arch_01`); the pill from Nemotron-3's "01 / 02 / 03" chips |
| A1: each output on one short solid stem at its layer, no dashed or dotted lines; A2 keeps side leaders and a bracket | Llama 4 connectors (`arch_05`), Alammar bracket/leader (`arch_10`) |
| A2: model as a vertical stack (layer 1 at the bottom, 17 at the top), heads to the side at the height of the layer they read, leader lines | Alammar DeepSeek-R1 (`arch_10`), Llama 4 left column (`arch_05`) |
| "Today: Silero VAD → turn detector → Whisper" as three small grey outlined boxes, no accent | Parakeet-TDT plain outlined boxes (`arch_13`) |
| results as horizontal bars, sorted best first, value printed at the bar end (bold for ours), zero baseline as a thin grey rule, no gridlines | NVIDIA Nemotron-3 diarization bars (`res_01`), FT "Ranking" (`guide_01`), Observable sorted bars (`guide_06`) |
| ours in the accent, every other system in one grey `#C8CDD4` | Anthropic SWE-bench (`res_02`), DeepMind (`res_04`), Datawrapper "highlight" (`guide_05`) |
| panel title = the finding in plain words ("Better transcript"), subtitle = the measure with its unit ("Word errors in meetings (%)"); no unit on the bars | Datawrapper title/subtitle (`guide_02`), Anthropic title + subtitle (`res_02`) |
| colour legend as one line of two swatches under the headline, no legend boxes in the panels | Datawrapper direct labels (`guide_03`) |
| one small source line at the bottom, in the secondary grey | DeepMind methodology line (`res_04`), Datawrapper notes (`guide_04`) |
| B1: 3 × 2 small multiples, same bar height and scale rules in every panel | Anthropic's per-domain panels (`res_03`) with its colours removed |
| B2: one ranked list per metric, stacked, metric name + subtitle on the left, bars on the right, thin rules between | NVIDIA Nemotron-3 per-dataset bars (`res_01`), Meta's benchmark table rows (`res_09`) |
| type: system sans (SF Pro), bold and regular only; title 44 px, panel titles 28-30 px, body 22-27 px, footnote 22 px at 1x | all of the above |

**Dark variant.** Two top references do use dark well (Apple AFM, DeepMind's dark chart), but both are brand pages
with a dark site around them, and the post goes into LinkedIn's light feed. The current images are dark, and the aim
is to move away from them. So both architecture variants are light and differ in layout instead: A1 is a horizontal
flow, A2 a vertical stack. A dark copy only needs new values for the `style.css` tokens (`--bg` `#1A1B1F`, grey
bars `#2F3136`, the same single accent) if it is wanted later.

## The four variants

- **A1 `A1_flow`** (the post image) is drawn picture first, with words added after. Data flows left to right.
  - Row 1: voice icon → the frozen NVIDIA model as 17 vertical slabs (layer 1 on the left, 17 on the right, layer 4
    the only orange slab) → decoder → "Words" (with a "Parakeet" tag).
  - Under row 1: a brace over all 17 slabs, with one arrow from its centre to "Speaking?" (the VAD head reads a
    learned mix of all 17 layers).
  - A line straight down from slab 4 → "Who?" (speaker head, block 4).
  - Row 2: a lighter copy of the same model ("Second pass"), fed by the voice from the left and by Nemotron-3 (the
    diarizer's who-is-talking columns, the true conditioning input) from above-left → "Turn over?".
  - Every output sits in one column: a flat icon plus at most three words. The icons are orange only for our heads.
    A "Today" row of three grey boxes (VAD → Turn → Whisper) closes the image.
  - No subtitle, footer, dashed lines or example values. The engineering details moved to `demo/archive/README.md`.
  - `render.py` also writes `demo_out/images/architecture_notext.png`, the same render with every word hidden, to
    check that the flow reads without text.
- **A2 `A2_stack`**: the model as a vertical stack with the same heads annotated at their layer heights: "speaking?"
  on the left bracket, "who?" at layer 4, "turn over?" on the second pass. The "Today" chain runs bottom-up in a side
  column.
- **B1 `B1_grid`** (the post image): six panels in a 3 × 2 grid (2 × 3 in the square format). Every panel has a
  direction tag in the subtitle grey, "↑ higher is better" or "↓ lower is better" (the Datawrapper convention). It
  closes the subtitle line in the wide image and the title line in the narrower square panels. Bars are sorted
  best-first in panels with independent rows. In "Follows your voice" each before → voice-print pair stays adjacent,
  and the pairs are ordered by the voice-print value (ICSI 69 → 20 first, then AMI 62 → 39).
- **B2 `B2_list`**: the six metrics stacked as ranked lists with thin rules between them.

Each variant comes as 3840 × 2160 and 2160 × 2160 on the SSD (`demo_out/images_redesign/`) and as half-size copies
in `demo/images/redesign/`.

## Honesty rules kept

- Every number on the results images is a `numbers_b.json` "shown" string: the page reads `demo/v5/shots/numbers_b.js`
  (both generated by `export_b.py` from `runs/*.json`). Each number is a `[data-key]` element, and `render.py` checks
  it against the JSON. Definitions, baselines and wording are the same as in the reviewed v5 results image: VAD "edges
  NVIDIA's MarbleNet, clearly ahead of Silero"; voice print "with about the same false cut-offs", with the
  false-cut-off numbers in the footer; "nearly 40 % fewer" from `live/cutins_less`; first words 1.2 s vs 2.9 s
  (Pipecat) and 3.4 s (LiveKit). The footer is the v5 footer.
- In "Follows your voice" the grey bars are audioforge *without* the voice print. The subtitle reads "Give it a 5-second sample of your voice and it misses fewer of your turn ends (%)". Row labels
  ("AMI meetings, no sample" / "with sample") and the legend line both say so, and each pair stays adjacent. The
  footer keeps "false cut-offs stay about the same" with both pairs of numbers; it does not say "did not raise",
  because ICSI goes from 5.9 to 6.0 %.
- The architecture images contain structural facts only (17 layers, layer 4, the second pass, the heads' parameter
  counts in the footnote), taken from the reviewed v5 figure and `research/FINAL_REPORT.md` / `VAD_LAYERS.md`. The
  "speaking?" head is drawn reading all 17 layers because the served head is a learned mix of all 17 blocks
  (`research/archive/VAD_LAYERS.md`). The v5 "Why it matters" strip (speaker errors 32 → 14 %, 38 % less compute) is left out
  because those numbers are not in `numbers_b.json`.

## Checks (`render.py`, all eight PNGs pass; `demo/images/redesign/checks.json`)

Text at least 22 px at 1x; every text at least 64 px from the frame edge; contrast at least 7:1 against the nearest
painted background (minimum 7.56:1, the secondary grey `#4B5563` on white); no intersecting text runs; no clipped or
off-frame text; every value equal to its `numbers_b.json` "shown" string. Contact sheets for judging the resemblance
(references | ours): `demo_out/images_redesign/contact_{A1_flow,A2_stack,B1_grid,B2_list}.png`.
