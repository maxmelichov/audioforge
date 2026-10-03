# Heads over the 700-line limit (AGENTS.md "File size"): proposed splits, not done (rule: ask first)

Counted 2026-10-03 at 4895faf + the sizing comments. Only one file in `audioforge/heads/` is over the limit:

| file | lines | over by |
|---|---|---|
| audioforge/heads/turn.py | 966 | 266 |

The other heads files are under it: audio.py 540, asr.py 393, turn_seg.py 184, completeness.py 164, prosody.py 126,
tsvad.py 116.

## Proposed split of turn.py (by responsibility, two halves)

turn.py does two jobs: the per-frame end-of-turn head (`TurnHead` and the target / feature builders it trains on), and
the offline scoring driver that runs a whole model over conversations (`turn_scores` and its helpers).

- Keep in `audioforge/heads/turn.py`: the targets and features (`eot_targets` .. `multi_horizon_targets`, lines
  86-231), token alignment (`align_chunk` .. `pad_tokens`, 232-366) and `TurnStreamState` / `TurnHead` (367-822).
  About 820 lines, still over: move the token alignment block (about 135 lines) to `audioforge/heads/turn_align.py`
  as a second step, re-exported from turn.py so `from audioforge.heads.turn import greedy_align` keeps working.
- Move to `audioforge/heads/turn_scoring.py`: `_flat`, `_onset_end`, `_diar_name`, `primary_column`,
  `streaming_diar_act`, `streaming_diar_probs`, `decoded_text_state`, `turn_scores` (lines 823-966, about 144
  lines). They drive a whole SpeechModel (encoder, diarizer, ASR) and are not part of the head; re-export them from
  turn.py for existing callers.
- Tests move with their code (tests that only cover `turn_scores` go with turn_scoring.py).
