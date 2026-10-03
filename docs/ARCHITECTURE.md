# Architecture

One page: the model that runs, the server around it, where each piece lives in the code, and the model files with
their licences. The measured comparisons are in [`RESULTS.md`](RESULTS.md).

## The model

```
 16 kHz PCM ──► 160 ms chunk ──► 80-bin log-mel ──► 17 x FastConformer block (115M, frozen, NVIDIA)
                                                     cache-aware streaming, 80 ms lookahead
                                                        │
        ┌───────────────────────┬───────────────────────┼───────────────────────┬─────────────────────────┐
        ▼                       ▼                       ▼                       ▼                         ▼
   RNNT decoder            VAD head               speaker head           turn head (GRU)           TS-VAD head
   "what was said"         "is anyone             "who is this?"         "is the turn over?"       "is it the user?"
   partial transcripts     speaking?"             block 4, 0.50 M        a second pass of the      block 4, 0.26 M,
                           block 4, 33 K          distilled from         same encoder, told who    conditioned on a
                                                  TitaNet-L              is speaking               voice print (flags)

 also on the same encoder:  speech detector (learned mix of blocks 2-6, 33 K)          "is anyone speaking?" (the client's
                            per-frame speech probability; the block-4 VAD head above feeds the turn rules)
                            end-of-turn classifier v5 (block 8 + VAD / TS-VAD + words, 2.5 M)   `fast` / `assistant`
                            LID head v2 (blocks 8-12, 2.37 M, distilled from AmberNet, optional)   "which language?"
                            voice-gender head (block 4, 22 K, optional, off by default)   perceived female / male voice

 room mode only (--mode room):
              NVIDIA Nemotron-3-Diarization (or Streaming Sortformer v2)   "who is in the room": 4-8 activity columns
              NVIDIA Parakeet-TDT 0.6B v3 on each finished turn (optional)   rewrites the turn's transcript
              TitaNet-L (voice enrollment), AmberNet (language ID) (optional)
```

**The heads at a glance** (block numbers for the 115M; the 0.6B's in brackets):

| head | reads | size | job |
|---|---|---|---|
| speech detector | a learned mix of blocks 2-6 [8-16] | 33K [66K] | is anyone speaking? (the client's speech probability, every 80 ms) |
| VAD (turn rules) | block 4 | 33K | the quiet gate the turn rules, TS-VAD and LID read |
| speaker | block 4 [5] | 0.5M | a 192-number voice print |
| TS-VAD | block 4 [5] + the stored print | 0.26M | is it the user or someone else? |
| turn | a second, speaker-conditioned pass (GRU; 115M only) | 0.32M | is the user's turn over? (`balanced`, `steady`) |
| turn classifier v5 | block 8 [12] of the first pass + VAD / TS-VAD + the words so far, at each quiet frame | 2.5M [2.6M for `assistant`] | is the user's turn over? (`fast`, `assistant`) |
| LID v2 (optional) | blocks 8-12 [16-20] | 2.4M [2.9M] | which language? |
| voice gender (optional, off by default) | block 4 [5] | 22K [21K] | perceived female / male voice probabilities, not gender identity |

**Two modes.** In **single-model mode**, the default since 2026-09-29 ([`CONFIGURATION.md`](CONFIGURATION.md) §13), only the model
above runs. The user's stored voice print (192 numbers from the speaker head; ≥ 5 s of clean speech, 10 s for
meetings) conditions the TS-VAD head. Its track [P(user), P(someone else), 0, 0] is `frame.speakers` and is what the
turn head reads. No diarizer is loaded, and server RTF is 0.33 on 2 threads. **Room mode**
(`--mode room`) adds NVIDIA's diarizer to label everyone in a room, and optionally Parakeet-TDT v3 for the final
transcript. The rest of this page notes where the two differ.

- **The encoder** is NVIDIA's `stt_en_fastconformer_hybrid_large_streaming_multi`: 17 FastConformer blocks,
  cache-aware, run at attention context [70, 1], which gives 160 ms chunks and 80 ms lookahead. It is imported without
  NeMo (`audioforge/nemo_import.py`) and **never fine-tuned**: every fine-tuning attempt raised LibriSpeech WER (2.05 to
  4.31 % in 500 steps), so everything this project learned lives in the heads.
- **The heads** read the encoder's frames at no extra encoder cost. The shipped build is `stage1_served_v4.afm`
  (heads v0.4). The VAD head that gates the turn rules reads block 4; the `speech` detector
  head added in v0.4 reads a learned mix of blocks 2-6 and gives the client's speech probability.
  The `--core 0.6b` build (`served_0p6b_v0.4.afm`) has the same heads on its own blocks (speaker and TS-VAD on block 5). The speaker head reads block 4 too, where speaker information lives; the top blocks are speaker-blind. The turn head
  runs on a second, speaker-conditioned pass of the same encoder, fed the primary speaker's diarizer column. The
  TS-VAD head replaces that column with the enrolled user's activity (single-model mode; `--turn-input tsvad`).
- **The diarizer** (room mode only) is NVIDIA's, because our own general diarization head is not good enough (0.394
  vs 0.232 DER). It runs on its own front end on the same 80 ms clock. Its columns arrive 80-240 ms after their audio with the default 0.32 s
  setting.
- **End of turn** is a policy over these signals, chosen per session. Single mode's default is `vad_head`: the VAD
  head's silence AND the turn head's probability, a VAD-silence fallback, and the user's TS-VAD silence while someone
  else has the floor; no Silero. Room mode's default is a silence timeout on the primary
  speaker's column (`timeout`). Others: the head's probability OR a Silero silence with a dynamic wait (`hybrid_dyn`),
  any speaker's silence (`timeout_any`), and more ([`CONFIGURATION.md`](CONFIGURATION.md) §4).
- **Speed** comes from exact CPU fast paths (`audioforge/perf.py`, on by default): column-major weights, cached
  position projections, one subsampling shared by both encoder passes, and a cached RNNT joint. Decisions are
  unchanged and compute drops to 0.62x. Conformer convolutions run as unfold + linear (`fast_conv`).

## The server

```
 client (Pipecat / LiveKit adapter, audioforge-client, any WebSocket)
   │  config (JSON) · PCM (binary) · agent_end / enroll · end
   ▼
 connection loop            audioforge/serve.py: handle(), serve(), GET /health
   │  decode + resample, backpressure, idle / session limits, final-ASR delivery
   ▼
 Session (one per connection)   audioforge/serve.py: Session.process() -> messages
   │  ASRStream (encoder pass, RNNT, VAD, turn, speaker)   audioforge/server/streams.py
   │  speaker columns: TS-VAD track (single) or diarizer step + binding (room)   tsvad_stream.py, server/binding.py
   │  turn policies                                        audioforge/server/policies.py
   │  load shedding, NaN guards, segment cap
   ▼
 Engine (shared)            audioforge/serve.py: models, the compute thread, Silero, the final-ASR worker process
   │
   ▼
 ready · frame (80 ms) · partial · turn_end · final · stats · error    audioforge/server/protocol.py (validate)
```

- **One compute thread** serves every session, so sessions never race on torch state. The asyncio loop only moves
  bytes. One session per 2 CPU threads is what one worker sustains in real time.
- **The final-ASR worker** (Parakeet-TDT v3) runs in a child process with its own torch threads, so it never blocks
  the streaming thread. It is restarted if it hangs or dies, and the streaming final stands in meanwhile.
- **Under load** the server sheds work in levels instead of falling behind: first the diarizer runs at half cadence
  and holds the last speaker, then partials are dropped. Every degradation is reported as a non-fatal `error`
  message and counted in `stats` and `/health`.
- **Configuration** is one flag table (`audioforge/server/cli.py`). The same table produces `--help`, the config-file
  keys and the reference in [`CONFIGURATION.md`](CONFIGURATION.md).
- **In-process use** (`audioforge.load()`, `audioforge-bench`) drives the same `Engine` / `Session` without the
  socket.

## Where things live

| concern | code |
|---|---|
| in-process API | `audioforge/api.py` |
| server core (engine, session, connection loop) | `audioforge/serve.py` |
| protocol, policies, binding, streams, flags | `audioforge/server/` |
| launcher and model download | `audioforge/launch.py`, `audioforge/hub.py` |
| framework adapters | `audioforge/integrations/pipecat.py`, `audioforge/integrations/livekit.py` |
| model, encoder, heads, losses | `audioforge/model.py`, `audioforge/modules/`, `audioforge/heads/`, `audioforge/losses/` |
| NVIDIA checkpoint import, diarizer | `audioforge/nemo_import.py`, `audioforge/streaming_diar.py` |
| voice print, speaker registry, enrollment | `audioforge/tsvad_stream.py`, `audioforge/speaker_registry.py`, `audioforge/enrollment.py` |
| training library and recipes | `audioforge/train.py`, `audioforge/datasets/`, `audioforge/baselines/`, `recipes/` |

## Models and files

audioforge's only weights of its own are the small heads in `assets/`. Everything else is NVIDIA's (or Silero's),
fetched from the publisher by `audioforge-download` (`audioforge/hub.py`) and kept under its own licence. The command
shows the licences of what it is about to fetch and asks you to accept them (`--yes` or
`AUDIOFORGE_ACCEPT_LICENSES=1` non-interactively).

```
audioforge-download                          # asr + tsvad + lid: single-model mode (the default)
audioforge-download --core 0.6b              # the 0.6B core and its heads (2.5 GB download, ~2.3 GB on disk)
audioforge-download --diarizer nemotron3     # + nemotron3 for room mode (--diarizer sortformer: Sortformer v2)
audioforge-download --heads-version 0.1      # the measured 2026-09-27 build (stage1_served.afm)
audioforge-download --with tdt_v3 titanet ambernet silero      # or --with all
audioforge-download --only silero
audioforge-download --list
```

### The two cores

Both cores are frozen NVIDIA streaming encoders. Every NVIDIA tensor is used unchanged; audioforge adds its heads.

| core | NVIDIA model | licence | served file | heads file |
|---|---|---|---|---|
| `115m` (default) | [stt_en_fastconformer_hybrid_large_streaming_multi](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi) (114M) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | `stage1_served_v4.afm` (453 MB) | `served_heads_v0.4.pt` |
| `0.6b` | [nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) (618M) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | `served_0p6b_v0.4.afm` | `served_heads_0p6b_v0.4.pt` |

### The shipped heads (`assets/`)

- `served_heads_v0.4.pt`: the heads merged into the 115M served model (195 tensors, 29.5 MB). It is v0.3's heads plus
  `speech`, a stateless speech detector on a learned mix of blocks 2-6, trained on AMI + ICSI train + oto user
  channels. It gives the client's per-frame speech probability (AMI test F1 0.959). The turn rules, TS-VAD, LID gating
  and the v5 classifier keep reading the block-4 `vad` head, so turn taking is unchanged. Older builds stay selectable:
  `--heads-version 0.3` (v0.2 + `turn_seg`, the turn head v5 segment classifier), `0.2` (104 tensors, 19.6 MB) and
  `0.1` (the 2026-09-27 measured heads).
- `tsvad_spk.pt`: the target-speaker head (1.0 MB).
- `lid_115m_v2.pt`: the language-ID head (2.37 M parameters, 9.48 MB; FLEURS-17 test 92.4 % at 2 s, 98.2 % full clip).
  `lid_distill.pt` (v1, 0.92 M, 90.9 / 97.8 %) is kept for reproduction.
- `served_heads_0p6b_v0.4.pt` (`--core 0.6b`): the heads merged into nemotron-speech-streaming-en-0.6b (213 tensors,
  27.0 MB). It is v0.3 plus `turn_seg_a`, the `assistant` preset's own v5 turn classifier (2.59 M parameters, trained
  with real two-party channels: AMI individual headsets and otoSpeech-280h raw channels), and `turn_vad`, the
  stateless block-12 VAD that classifier reads (65.7 K). `balanced` / `fast` and every other head are unchanged. v0.3
  (123 tensors, 16.3 MB: v0.2 + the same kind of `speech` head on blocks 8-16; AMI test F1 0.957) and v0.2 (v0.1's
  tensors with the turn presets' constants re-picked on held-out data) stay selectable.
- `tsvad_0p6b.pt` and `lid_0p6b_v2.pt` (2.89 M; FLEURS-17 test 92.7 / 98.6 %, +5.1 / +3.1 points over `lid_0p6b.pt`):
  the 0.6B core's TS-VAD and LID heads.
- `voice_gender_115m.pt` / `voice_gender_0p6b.pt` (optional, off by default; `serve --voice-gender head`): a perceived
  voice-gender head per core (21.8 K / 20.6 K parameters, 92 / 87 KB; causal attentive statistics pooling and a
  linear classifier on the speaker head's tap). It outputs probabilities for the two classes its training data
  annotate, female voice and male voice. It is a perceived vocal characteristic estimated from audio, not a person's
  gender identity, and it can be wrong for any individual. LibriSpeech test-clean balanced accuracy 97.8 / 97.7 % at
  2 s of speech, FLEURS-17 test 96.3 / 96.7 %.

**The served VAD head reads block 4** (`vad_layer: 3`, zero-based, the speaker head's tap) since
`stage1_served_v2.afm`. In an AMI dev selection experiment (not a test result) it matched the learned mix of all 17
blocks, with no downstream regression. Apart from the VAD head, `stage1_served_v2.afm` is identical to
`stage1_served.afm`, the 2026-09-27 checkpoint that reads the all-block mix (`vad_layer: all`;
`--heads-version 0.1` rebuilds it).

### Components

| key | model | source (pinned) | download | on disk (`$AUDIOFORGE_HOME`) | licence | used for |
|---|---|---|---|---|---|---|
| `asr` | NVIDIA FastConformer hybrid streaming (114M) + audioforge heads | [nvidia/stt_en_fastconformer_hybrid_large_streaming_multi](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi) @ `ae9814333369` | `stt_en_fastconformer_hybrid_large_streaming_multi.nemo`, 460 MB | `stage1_served_v4.afm` (453 MB; `--heads-version 0.3`: `stage1_served_v3.afm`, `0.2`: `stage1_served_v2.afm`, `0.1`: `stage1_served.afm`) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | streaming ASR, VAD, turn and speaker heads (required) |
| `asr_0p6b` | NVIDIA Nemotron Speech Streaming EN 0.6B + audioforge heads | [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) @ `ebe59e5a8171` | `nemotron-speech-streaming-en-0.6b.nemo` | `served_0p6b_v0.4.afm` | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | `--core 0.6b` |
| `sortformer` | NVIDIA Streaming Sortformer 4spk v2 (117M) | [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) @ `84edd514b8ef` | `diar_streaming_sortformer_4spk-v2.nemo`, 471 MB | `nemo_sortformer_v2.afm` (436 MB) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | speaker activity / diarizer (default --diar) |
| `nemotron3` | NVIDIA Nemotron-3-Diarization | [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) @ `f667ed73aee5` | `Nemotron-3-Diarization.nemo`, 199 MB | `nemo_nemotron3_diar.afm` (177 MB) | [OpenMDW-1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization) | alternative diarizer (--diarizer nemotron3) |
| `tdt_v3` | NVIDIA Parakeet-TDT 0.6B v3 | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) @ `541d1f99c6b0` | `parakeet-tdt-0.6b-v3.nemo`, 2509 MB | `nemo/parakeet-tdt-0.6b-v3.nemo` (2393 MB) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | offline per-turn final ASR (serve --final-asr tdt_v3) |
| `titanet` | NVIDIA TitaNet-Large speaker embeddings (23M) | [nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large) @ `0dc382f40121` | `speakerverification_en_titanet_large.nemo`, 102 MB | `nemo/speakerverification_en_titanet_large.nemo` (97 MB) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | voice enrollment (serve --enroll after_agent / explicit) |
| `ambernet` | NVIDIA AmberNet spoken language ID (NGC) | [fixed URL](https://api.ngc.nvidia.com/v2/models/nvidia/nemo/langid_ambernet/versions/1.12.0/files/ambernet.nemo) | `langid_ambernet.nemo`, 116 MB | `nemo/langid_ambernet.nemo` (111 MB) | [NGC Terms of Use](https://ngc.nvidia.com/legal/terms) | language ID (serve --lid ambernet) |
| `tsvad` | audioforge TS-VAD head (target speaker, 0.26 M) | `assets/tsvad_spk.pt`, else the v0.1.0 release | `tsvad_spk.pt`, 1 MB | `tsvad_spk.pt` | Apache-2.0 (audioforge heads) | single-model mode: the user's track |
| `lid` | audioforge LID head v2 (2.37 M, distilled from AmberNet) | `assets/lid_115m_v2.pt` (not yet on a release) | `lid_115m_v2.pt`, 9.48 MB | `lid_115m_v2.pt` | Apache-2.0 (audioforge heads; trained on AmberNet outputs, NGC Terms of Use) | single-model mode: language ID |
| `voice_gender` | audioforge perceived voice-gender head (optional, 21.8 K) | `assets/voice_gender_115m.pt` (not yet on a release) | `voice_gender_115m.pt`, 92 KB | `voice_gender_115m.pt` | Apache-2.0 (audioforge heads; trained on FLEURS and LibriSpeech, CC BY 4.0) | optional: perceived voice-gender probabilities (serve --voice-gender head) |
| `voice_gender_0p6b` | the same for the 0.6B core (optional, 20.6 K) | `assets/voice_gender_0p6b.pt` (not yet on a release) | `voice_gender_0p6b.pt`, 87 KB | `voice_gender_0p6b.pt` | Apache-2.0 (audioforge heads; trained on FLEURS and LibriSpeech, CC BY 4.0) | --core 0.6b, optional: perceived voice-gender probabilities |
| `silero` | Silero VAD v5 (v5.1.2 ONNX) | [fixed URL](https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx) | `silero_vad.onnx`, 2 MB | `silero_vad_v5.onnx` (2 MB) | [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE) | turn policies hybrid_silero / hybrid_dyn |

The default set is `asr` + `tsvad` + `lid`. Silero is optional, since single mode's default turn rule (`vad_head`)
reads the model's own heads; add `--with silero` for `hybrid_silero` / `hybrid_dyn`. Peak disk during a default
install is about 0.9 GB: the `.nemo` files are deleted after conversion unless `--keep-nemo`. `--from-local DIR` and
`--data-root DIR` (default `$AUDIOFORGE_DATA`, i.e. `<repo>/data`) reuse `.nemo` / `.onnx` files you already have.
They are sha256-checked and used in place.

### Integrity

Every download is checked against the size and sha256 below. A mismatch deletes the file and fails the command. The
served model is not downloaded as such: `build_served` merges NVIDIA's encoder checkpoint (all of its tensors
unchanged) with the heads file and asserts the merged model's tensor hash against the hash recorded when the heads
were exported. The model you run is therefore bit-identical to the checkpoint the heads came from.

The heads file (`--heads-version`, default 0.4) is looked up in the checkout's `assets/` first, then in the models
directory, then fetched from `https://github.com/maxmelichov/audioforge/releases/download/v0.1.0/<file>`. Its size
and sha256 are checked.

| key | file | bytes | sha256 |
|---|---|---:|---|
| `asr` | `stt_en_fastconformer_hybrid_large_streaming_multi.nemo` | 459673600 | `8db5289d5238aca839b84f5afdd66eb1aa64413feb9c2a915940b291012aff8b` |
| `asr_0p6b` | `nemotron-speech-streaming-en-0.6b.nemo` | 2473041920 | `283638054c44f6794e74fe9af9048d78a6d9d6c058c12131856c7859a62ac9cd` |
| `sortformer` | `diar_streaming_sortformer_4spk-v2.nemo` | 471367680 | `b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329` |
| `nemotron3` | `Nemotron-3-Diarization.nemo` | 198676480 | `867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d` |
| `tdt_v3` | `parakeet-tdt-0.6b-v3.nemo` | 2509332480 | `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d` |
| `titanet` | `speakerverification_en_titanet_large.nemo` | 101621760 | `e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3` |
| `ambernet` | `langid_ambernet.nemo` | 116049920 | `2f92d645b9ea5824d7663584fecb9ecc52557d0d700e24266747f38a61ba1681` |
| `silero` | `silero_vad.onnx` | 2327524 | `2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f` |
| `tsvad` | `tsvad_spk.pt` | 1048542 | `dbc6230d8d722bad65aaf598dce69569995bd96bc002da40a2069d664c427683` |
| `lid` | `lid_115m_v2.pt` | 9478879 | `6e9586354538d298f90cffb3dd0c2439a97107f3ab969d7286132fbcd6570e27` |
| `lid_0p6b` | `lid_0p6b_v2.pt` | 11580148 | `63acfa9d5b83986c11221a234a5ffb6e24e19b7761c19490cc8e60bcc06d8d01` |
| `voice_gender` | `voice_gender_115m.pt` | 91813 | `e1e1b82416bfeb3a83f0fcceebe3af87ab5309e67f64cb01a758bf32e0cc1f54` |
| `voice_gender_0p6b` | `voice_gender_0p6b.pt` | 87333 | `7a38cf6bdef36c33eb03f319a0e719de80fa4423f75935a876923713220868fe` |
| heads v0.4 (ships) | `served_heads_v0.4.pt` | 29614995 | `c3301b453f7f3e0b7e85da2c34471ce3c8f604e8c2d201c405cb235c64cd6d99` |
| heads v0.3 (previous; v0.4 is current) | `served_heads_v0.3.pt` | 29481643 | `ea1e8331fa9b9efdee76f1b44d4352f9e1660f34e3da6491b5655ce56ab18848` |
| heads v0.2 (previous; v0.4 is current) | `served_heads_v0.2.pt` | 19629563 | `cb5aa06974f27576c0f66b9453868701106969dea5d5ad100b06b2b779f121d2` |
| heads v0.1 (measured) | `served_heads_v0.1.pt` | 19629955 | `834f3e94467bc4110555d8d4cbdbe0ce75254a80d8ca203ecd975d4f007286f5` |
| 0.6B heads v0.4 (ships) | `served_heads_0p6b_v0.4.pt` | 26952298 | `2062496c631372387356bacc93241d5a2a30f5de751c7210fe8cf03e7e0b0ed1` |
| 0.6B heads v0.3 (previous; v0.4 is current) | `served_heads_0p6b_v0.3.pt` | 16309976 | `3245e5ee5bc05beb5f2bc9f412c89b455c0b2faa6ab592ddd7c6bab3aa4568f1` |
| 0.6B heads v0.2 (previous; v0.4 is current) | `served_heads_0p6b_v0.2.pt` | 16045591 | `ceff8c8912640e67500ca796d7c983d220ddf581845134e3e7ca68d7c32db3ff` |
| 0.6B heads v0.1 | `served_heads_0p6b_v0.1.pt` | 16045143 | `664be5a0e498b9268d088ccf2fa079d909cd70311c4096f8d28f94258adc4e6e` |

### Licences of the weights

The code licence (`LICENSE`, Apache-2.0) does not cover the weights. [`NOTICE`](../NOTICE) has the full statement.

- CC-BY-4.0 (NVIDIA): the 115M served encoder, Streaming Sortformer v2, Parakeet-TDT 0.6B v3, TitaNet-Large.
  Attribution to NVIDIA is required when you redistribute the weights or models derived from them. The merged served
  model is such a derivative.
- NVIDIA Open Model License: nemotron-speech-streaming-en-0.6b (the 0.6B core).
- OpenMDW-1.1 (NVIDIA): Nemotron-3-Diarization.
- NGC Terms of Use (NVIDIA): AmberNet (`langid_ambernet`, downloaded from NGC, not Hugging Face).
- MIT: Silero VAD.
- Apache-2.0: the audioforge heads in `assets/`.

The benchmarks also compare against models this tool never installs: parakeet_realtime_eou_120m-v1, Sortformer v2.1
and MarbleNet (NVIDIA Open Model License), parakeet-ctc-0.6b / 1.1b (CC-BY-4.0), smart-turn (BSD-2-Clause), the
LiveKit turn detectors (LiveKit Model License, evaluation only), pyannote, SpeechBrain, Whisper / faster-whisper and
WebRTC VAD.

### Training data (not shipped)

| dataset | licence | used for | notes |
|---|---|---|---|
| AMI Meeting Corpus | CC BY 4.0 | turn / speaker / VAD heads, meeting ASR adaptation and the 0.6B -> 115M distillation | headset mix |
| ICSI Meeting Corpus | ICSI release terms (research) | same | headset mix |
| LibriSpeech | CC BY 4.0 | anchor / WER gate | |
| LibriSpeech train-clean-100 (251 readers; dev-clean for selection, test-clean for test; gender from SPEAKERS.TXT) and FLEURS train / dev / test (17 languages; the manifest's gender field) | CC BY 4.0 | the optional voice-gender heads | speaker-disjoint for LibriSpeech (checked); FLEURS has no speaker ids |
| AMI Meeting Corpus, individual headsets (45 train meetings; 18 dev meetings held out; the 16 test meetings never touched) | CC BY 4.0 | 0.6B heads v0.4: `turn_vad` / `turn_seg_a` training, held-out selection | one participant's headset = the user channel |
| otoSpeech-full-duplex-280h (raw channels; 64 train / 32 held-out / 32 never-touched sessions) | CC BY 4.0; no speaker identification, redactions respected | same | labels from Silero VAD v5 per channel minus bleed |
| AppTek Call-Center Dialogues (48 held-out / 48 never-touched calls) | CC BY-SA 4.0; the card excludes training | held-out selection and never-touched reporting only, never trained on | one file per party |
