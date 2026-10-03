# Head sizes: where each is set and whether it was measured (2026-10-03)

docs/PROJECT.md "Sizing": a size that was never swept is a placeholder. "Set in" is where the shipped value lives (the
head class default is used only when the shipped config omits the key). Searched: research/LAYER_SWEEP_115M.md,
LAYER_SWEEP_0P6B.md (they sweep encoder blocks, not widths), FIXALL.md, TURN_V5.md, HEAD_ARCHITECTURES.md (proposals,
no width runs), CORE_0P6B.md, CORE_0P6B_TURN.md (GRU-64 vs MLP is an architecture choice, not a width sweep),
IMPROVE_115M.md, TURN_DATA.md, BARGEIN.md, archive/LID.md.

| head | core | size | value | set in | swept? | record |
|---|---|---|---|---|---|---|
| speech detector `speech` | 115M | FrameHead hidden | 64 | served_heads_v0.4.pt cfg | yes: 64 vs 256 (equal steps); 16-256 at equal wall-clock | speech_115m_2026-10-01.md, speech_115m_2026-10-03.md |
| speech detector `speech` | 0.6B | FrameHead hidden | 64 | served_heads_0p6b_v0.4.pt cfg | no | placeholder (audio.py FrameHead) |
| VAD `vad` (turn rules) | 115M | FrameHead hidden | 64 | served_heads_v0.4.pt cfg | no (VAD_SINGLE swept blocks only) | placeholder |
| VAD `vad` | 0.6B | FrameGRUHead hidden | 64 | served_heads_0p6b_v0.4.pt cfg | no | placeholder |
| end of utterance `eou` | 115M | FrameHead hidden | 64 | served_heads_v0.4.pt cfg | no | placeholder |
| turn VAD `turn_vad` | 0.6B | FrameHead hidden | 64 | served_heads_0p6b_v0.4.pt cfg | no | placeholder |
| speaker `spk` | both | emb_dim | 192 | class default | fixed by the TitaNet-L print format | – |
| speaker `spk` | 0.6B | projection hidden | 0 (single linear) | class default | yes: 0 vs 512 | speaker_0p6b_2026-10-02.md |
| speaker `spk` | 115M | projection hidden | 0 | class default | no (only 512 trained) | placeholder |
| speaker `spk` | both | AttentiveStatsPool bottleneck | 128 | class default | no | placeholder |
| TS-VAD | 115M | hidden | 128 | assets/tsvad_spk.pt cfg | yes: 128 vs 192 | tsvad_115m_2026-09-28.md |
| TS-VAD | 0.6B | hidden | 128 | assets/tsvad_0p6b.pt cfg | no | placeholder |
| TS-VAD | both | prenet_dim | 64 | class default | no | placeholder |
| turn GRU `turn` | both | hidden / n_layers / history / k_tokens / text_dim | 96 / 1 / 8 / 4 / 64 | heads cfg + class default | no | placeholder (turn.py) |
| v5 classifier `turn_seg` | 115M | n_layers | 2 | served_heads_v0.4.pt cfg | yes: 0 / 2 / 4 | turn_seg_115m_2026-09-30.md |
| v5 classifier `turn_seg`, `turn_seg_a` | 0.6B | n_layers | 2 | served_heads_0p6b_v0.4.pt cfg | no (115M recipe copied) | placeholder |
| v5 classifier | both | d / heads / ff / d_text / text_layers / cls MLP / AttnPool h | 256 / 4 / 1024 / 128 / 2 / 256,64 / 128 | heads cfg + class code | no | placeholder (turn_seg.py) |
| LID | 115M | hidden | 1024 | assets/lid_115m_v2.pt cfg | yes: 256 / 512 / 1024 (equal steps; not bracketed upward) | lid_2026-10-02.md |
| LID | 0.6B | hidden | 1024 | assets/lid_0p6b_v2.pt cfg | yes: 512 / 1024 (not bracketed upward) | lid_2026-10-02.md |
| LID | both | att_hidden / cls_hidden | 128 / 256 | assets/lid_*_v2.pt cfg | no | placeholder |
| diarizer `diar` (Sortformer head) | 115M | d_hidden / n_layers / n_heads | 192 / 4 / 4 | class default | no | placeholder |
| RNNT / CTC | both | pred / joint hidden | 640 | imported NVIDIA .nemo config | fixed by NVIDIA's weights | – |
| barge-in probe (not served) | 115M | hid | 32 | scripts/research/bargein.py make_head | no | placeholder |
| completeness, codec tokens, AED (not shipped) | – | hidden / layers | class defaults | class default | no | placeholder |

None of the recorded measurements before 2026-10-03 used an equal wall-clock budget or logged an eval loss; each
record says what it used instead. `scripts/sweep_capacity.py` is the procedure from now on.
