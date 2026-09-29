# REVIEW_v5: finding -> fix

| # | finding | fix (who) |
|---|---|---|
| 1 | "misses 61.9 %" reads as failure | AMI meeting chart dropped (coordinator decision); results shown as positives (answered, caught) (charts) |
| 2 | false "≤ 5 % budget" (ours 6.47 %) | AMI chart and its scorecard row removed (charts, builder) |
| 3 | AMI 61.9 vs 20 % contradiction | both removed; one definition per metric: "answers when you finish" = no reply within 3 s, live, caller's microphone (charts) |
| 4 | phone calls 7.5 vs 42 % | oracle-timeout row removed; phone calls only as live "unanswered within 3 s" (charts) |
| 5 | TurnBench 85 % caught vs 68 % missed | chart_live mono panel dropped; TurnBench live = unanswered 23 / 36 / 54 %, detector chart = "turn ends caught at the same false-alarm rate" (charts) |
| 6 | ICSI shown as a miss rate with a CI badge | "caught 21 vs 13 %, people cut off 1.8 vs 3.6 %", CI in the fine print (charts) |
| 7 | scorecard rows not shown / not comparisons | seven rows, each one shown earlier with the same number; no RTF, no 6 of 6, no Kyutai, no oracle (charts) |
| 8 | "One model, not a chain" contradicted | claim and comparison box removed; arch ends on "Our heads on NVIDIA's models"; narration "NVIDIA's speaker model and Parakeet complete the stack" (builder) |
| 9 | RTF 0.64 scope / threads | compute claim removed from screen, narration and scorecard (builder, charts) |
| 10 | count-ups show intermediate values | labels appear at their final value when bars land (charts); counters show recorded counts only (examples) |
| 11 | "0.0 s" LiveKit timer in the pre-roll | "–" before onset, then "waiting…" with the running time from speech onset, frozen at the first word (coordinator decision) (examples) |
| 12 | placeholder text | removed from split, room and phone pages (examples, builder) |
| 13 | idle playhead | removed (examples, builder) |
| 14 | "playing" pill over the counter label | pill in the status row after the chip (examples) |
| 15 | waveform through "Who is speaking?" | text below the waveform box (examples) |
| 16 | caption over the Parakeet quote | 3D arch replaced by the flat build; no quote card under the caption (builder) |
| 17 | new labels over fading old labels | flat build, one idea per beat, elements never share a spot (builder) |
| 18 | camera punch-in | no camera in the flat build (builder) |
| 19 | layers clipped at the top | no 3D stack (builder) |
| 20 | 14 s without a title | a title per beat, held ≥ 2 s (builder) |
| 21 | label overload | plain questions with live answers; sizes and tap points in the fine print only (builder) |
| 22 | "silence 0.48" | "yes" / "no" chip (builder) |
| 23 | 99 vs 100 M | diarizer size moved to the fine print as "Nemotron-3" (builder) |
| 24 | "A real call" vs a meeting, "you" vs "her" | the opening is now a real phone call (otoSpeech 5f9148c5, 39.9-46.2 s); titles and narration say "a real call"; neutral wording, no gender inferred from voice (builder, examples) |
| 25 | hard cuts, chopped agent voice | 150 ms fades on every clip edge; agent lines chosen to fit before the next event, never cut mid-word ("Sure," ends at a word boundary) (audio) |
| 26 | music at -44 dB on the last sample | 1.5 s fade of picture and sound to silence at the end (audio, builder) |
| 27 | loudness jumps | clips processed to -16 LUFS / -1 dBTP; narration lines at -16 LUFS; audio QA reports each (audio) |
| 28 | music swells between lines | constant bed, muted per example block with raised-cosine ramps (audio) |
| 29 | "the same audio" was not | both sides play the identical span from the clip record (examples, builder) |
| 30 | ex_words clip where ours cuts in later | new clip tb_160 (no cut-in by ours in or after the span shown) (audio) |
| 31 | empty transcript area on ex_phone | phone/interrupt panels drop the transcript area; subtitles carry the words (examples) |
| 32 | chip fades in over 0.5 s | chip and bubble pop in within 2 frames (examples, builder) |
| 33 | room people lit before speaking | a person lights on their first attributed word; count starts at 0 of 4 (examples) |
| 34 | near-empty first frames | every chart/scorecard/end page has its title and layout at frame 0 (charts, builder) |
| 35 | TurnBench title grammar, Kyutai | "Catches more turn ends than the detectors in voice-agent frameworks"; Kyutai removed (charts) |
| 36 | TurnBench decimals and jargon | whole percents, "same false-alarm rate" (charts) |
| 37 | VAD F1 jargon, tiny margin | VAD chart dropped (charts) |
| 38 | WER win belongs to Parakeet | bar labelled "NVIDIA Parakeet in our stack", narration says so (charts, builder) |
| 39 | chart_live ranges | one number per system, split into three frames (charts) |
| 40 | precision differs | one rounding per value, shared by chart and scorecard (charts) |
| 41 | first-word numbers differ | one definition (median first words, live, caller's microphone) on chart and scorecard; the example's fine print says it is one call (charts, builder) |
| 42 | names change shot to shot | "Pipecat default" / "LiveKit default" everywhere; components in the fine print (charts, examples) |
| 43 | LiveKit colour inconsistent | one colour per system (charts, examples) |
| 44 | brand casing, end caption | lowercase "audioforge"; end narration no longer a one-word caption (builder) |
| 45 | animated dots change pill width | static "waiting…" / "listening…" (builder, examples) |
| 46 | RNNT quote unreadable | the flat build shows one live line of words in a card (builder) |
| 47 | clip caption repeats narration | captions during clips are short labels ("Left: Pipecat default · Right: audioforge") (builder) |
| 48 | script.md out of sync | script.md rewritten for the cut; orphaned lines removed (builder) |
| 49 | pacing, results block | examples re-cut, results = 6 short charts + scorecard; total 183.6 s (the examples now play real audio alone, which sets the floor) (builder) |
