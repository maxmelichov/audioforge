# Example audio

## Single-model mode (the default): `two_party_call_16s.wav`

`two_party_call_16s.wav` (16 kHz mono, 16.0 s) is a real two-party call: a 16 s excerpt of one conversation of
otoSpeech-full-duplex-processed-141h, with the user's channel and the other party's channel averaged to mono. This
is how research/E2E_FINAL.md built its clips; this conversation is one of its 16 otoSpeech clips. The user asks a
question (speech 3.9-10.8 s) and the other party starts to answer (12.9-15.0 s). `two_party_call_16s.json` holds the
source, the labelled speech intervals of both parties and the user's turn end. otoSpeech has no transcripts.

`two_party_call_16s.voiceprint.json` is the user's stored voice print: 192 numbers from `audioforge.voiceprint`
over 10 s of the user's own channel, with the other party silent, taken from elsewhere in the same call (outside
this excerpt ± 2 s). `examples/quickstart_client.py` and `examples/python_api.py` send it as
`{"type": "enroll", "embedding": [...]}` right after the config.

Licence: CC BY 4.0, otoSpeech-full-duplex-processed-141h by otoearth
(https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-processed-141h). Changes: the excerpt was cut, the
two channels were averaged, and the audio was resampled to 16 kHz. The dataset card prohibits attempts to identify
the speakers; do not try.

## Room mode: `two_speakers_10s.wav`


`two_speakers_10s.wav` (16 kHz mono, 10.2 s): two LibriSpeech test-clean utterances by different speakers with a
1.4 s gap, so a client sees partials, one speaker change, a `turn_end` and two finals.

| segment | speaker | start-end (s) | text | source |
|---|---|---|---|---|
| 1 | A | 0.200-4.475 | a cold lucid indifference reigned in his soul | LibriSpeech test-clean 1089-134686-0007 |
| 2 | B | 5.875-9.895 | heaven a good place to be raised to | LibriSpeech test-clean 121-121726-0004 |

`two_speakers_10s.json` holds the same reference (usable as `scripts/stream_client.py --ref`).

Licence: LibriSpeech is CC-BY-4.0 (V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based
on public domain audio books", ICASSP 2015; https://www.openslr.org/12). The clip was made by
concatenating the two utterances with silence; no other processing.
