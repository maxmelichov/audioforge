# Models

Everything `audioforge-download` (`audioforge/hub.py`) can fetch, with the pinned source, size, what it becomes on
disk, and its licence. audioforge's only model weights of its own are in `assets/`:
- `served_heads_v0.2.pt`: the heads merged into the served model (104 tensors, 19.6 MB; the VAD head reads block 4).
- `tsvad_spk.pt`: the target-speaker head (1.0 MB).
- `lid_distill.pt`: the language-ID head (3.7 MB).
- `served_heads_v0.1.pt`: the 2026-09-27 measured heads, kept for reproducibility. The download command shows the licences of the components it is
about to fetch and asks you to accept them (`--yes` / `AUDIOFORGE_ACCEPT_LICENSES=1` non-interactively).

```
audioforge-download                          # asr + tsvad + lid: single-model mode (the default)
audioforge-download --diarizer nemotron3     # + nemotron3 for room mode (--diarizer sortformer: Sortformer v2)
audioforge-download --heads-version 0.1      # the measured 2026-09-27 build (stage1_served.afm), for research/
audioforge-download --with tdt_v3 titanet ambernet silero      # or --with all
audioforge-download --only silero
audioforge-download --list
```

## Components

| key | model | source (pinned) | download | on disk (`$AUDIOFORGE_HOME`) | licence | used for |
|---|---|---|---|---|---|---|
| `asr` | NVIDIA FastConformer hybrid streaming (114M) + audioforge heads | [nvidia/stt_en_fastconformer_hybrid_large_streaming_multi](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi) @ `ae9814333369` | `stt_en_fastconformer_hybrid_large_streaming_multi.nemo`, 460 MB | `stage1_served_v2.afm` (443 MB; `--heads-version 0.1`: `stage1_served.afm`) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | streaming ASR, VAD, turn and speaker heads (required) |
| `sortformer` | NVIDIA Streaming Sortformer 4spk v2 (117M) | [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) @ `84edd514b8ef` | `diar_streaming_sortformer_4spk-v2.nemo`, 471 MB | `nemo_sortformer_v2.afm` (436 MB) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | speaker activity / diarizer (default --diar) |
| `nemotron3` | NVIDIA Nemotron-3-Diarization | [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) @ `f667ed73aee5` | `Nemotron-3-Diarization.nemo`, 199 MB | `nemo_nemotron3_diar.afm` (177 MB) | [OpenMDW-1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization) | alternative diarizer (--diarizer nemotron3) |
| `tdt_v3` | NVIDIA Parakeet-TDT 0.6B v3 | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) @ `541d1f99c6b0` | `parakeet-tdt-0.6b-v3.nemo`, 2509 MB | `nemo/parakeet-tdt-0.6b-v3.nemo` (2393 MB) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | offline per-turn final ASR (serve --final-asr tdt_v3) |
| `titanet` | NVIDIA TitaNet-Large speaker embeddings (23M) | [nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large) @ `0dc382f40121` | `speakerverification_en_titanet_large.nemo`, 102 MB | `nemo/speakerverification_en_titanet_large.nemo` (97 MB) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | voice enrollment (serve --enroll after_agent / explicit) |
| `ambernet` | NVIDIA AmberNet spoken language ID (NGC) | [fixed URL](https://api.ngc.nvidia.com/v2/models/nvidia/nemo/langid_ambernet/versions/1.12.0/files/ambernet.nemo) | `langid_ambernet.nemo`, 116 MB | `nemo/langid_ambernet.nemo` (111 MB) | [NGC Terms of Use](https://ngc.nvidia.com/legal/terms) | language ID (serve --lid ambernet) |
| `tsvad` | audioforge TS-VAD head (target speaker, 0.26 M) | `assets/tsvad_spk.pt`, else the v0.1.0 release | `tsvad_spk.pt`, 1 MB | `tsvad_spk.pt` | Apache-2.0 (audioforge heads) | single-model mode: the user's track |
| `lid` | audioforge LID head (0.92 M, distilled from AmberNet) | `assets/lid_distill.pt`, else the v0.1.0 release | `lid_distill.pt`, 4 MB | `lid_distill.pt` | Apache-2.0 (audioforge heads; trained on AmberNet outputs, NGC Terms of Use) | single-model mode: language ID |
| `silero` | Silero VAD v5 (v5.1.2 ONNX) | [fixed URL](https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx) | `silero_vad.onnx`, 2 MB | `silero_vad_v5.onnx` (2 MB) | [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE) | turn policies hybrid_silero / hybrid_dyn |

**The served VAD head reads block 4** (`vad_layer: 3`, zero-based, the speaker head's tap). That is the shipped
`stage1_served_v2.afm`, built from `served_heads_v0.2.pt`: AMI VAD F1 0.951 against 0.949 for the learned mix of all
17 blocks, with no downstream regression (research/VAD_SINGLE.md). Every other tensor is identical to
`stage1_served.afm`, the 2026-09-27 checkpoint that reads the all-block mix (`vad_layer: all`) and that most of
`research/` was measured with (`--heads-version 0.1` rebuilds it).

Default set: `asr` + `tsvad` + `lid`. Silero is optional since single mode's default turn rule (`vad_head`,
research/EOT_LATENCY.md) reads the model's own heads; `--with silero` for `hybrid_silero` / `hybrid_dyn`. Peak disk during a default install is about 0.9 GB (the `.nemo` files are deleted
after conversion unless `--keep-nemo`). `--from-local DIR` and `--data-root DIR` (default `$AUDIOFORGE_DATA`, i.e.
`<repo>/data`) reuse `.nemo` / `.onnx` files you already have; they are sha256-checked and used in place.

## Integrity

Every download is checked against the size and sha256 below; a mismatch deletes the file and fails the command. The
served model is not downloaded as such: `build_served` merges NVIDIA's `stt_en_fastconformer_hybrid_large_streaming_multi`
(all 654 tensors unchanged) with the heads file and asserts the merged model's tensor hash against the hash recorded
when the heads were exported from `runs/stage1_served_v2.afm` (v0.2) or `runs/stage1_served.afm` (v0.1). The model
you run is therefore bit-identical to the checkpoint the heads came from.

| key | file | bytes | sha256 |
|---|---|---:|---|
| `asr` | `stt_en_fastconformer_hybrid_large_streaming_multi.nemo` | 459673600 | `8db5289d5238aca839b84f5afdd66eb1aa64413feb9c2a915940b291012aff8b` |
| `sortformer` | `diar_streaming_sortformer_4spk-v2.nemo` | 471367680 | `b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329` |
| `nemotron3` | `Nemotron-3-Diarization.nemo` | 198676480 | `867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d` |
| `tdt_v3` | `parakeet-tdt-0.6b-v3.nemo` | 2509332480 | `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d` |
| `titanet` | `speakerverification_en_titanet_large.nemo` | 101621760 | `e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3` |
| `ambernet` | `langid_ambernet.nemo` | 116049920 | `2f92d645b9ea5824d7663584fecb9ecc52557d0d700e24266747f38a61ba1681` |
| `silero` | `silero_vad.onnx` | 2327524 | `2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f` |
| `tsvad` | `tsvad_spk.pt` | 1048542 | `dbc6230d8d722bad65aaf598dce69569995bd96bc002da40a2069d664c427683` |
| `lid` | `lid_distill.pt` | 3704886 | `07de4e4da5444ecae250b4c5ef0172372451ba2da9ea7ae492e42b1759f1d487` |
| heads v0.2 (ships) | `served_heads_v0.2.pt` | 19629563 | `cb5aa06974f27576c0f66b9453868701106969dea5d5ad100b06b2b779f121d2` |
| heads v0.1 (measured) | `served_heads_v0.1.pt` | 19629955 | `834f3e94467bc4110555d8d4cbdbe0ce75254a80d8ca203ecd975d4f007286f5` |

The heads file (`--heads-version`, default 0.2) is looked up in the checkout's `assets/` first, then in the models
directory, then fetched from `https://github.com/maxmelichov/audioforge/releases/download/v0.1.0/<file>`.
Its size and sha256 are checked.

## Licences of the weights

The code licence (`LICENSE`) does not cover the weights. In short (`NOTICE` has the full statement):

- CC-BY-4.0 (NVIDIA): the served encoder, Streaming Sortformer v2, Parakeet-TDT 0.6B v3, TitaNet-Large. Attribution
  to NVIDIA is required when you redistribute the weights or models derived from them; the merged served model is
  such a derivative.
- OpenMDW-1.1 (NVIDIA): Nemotron-3-Diarization.
- NGC Terms of Use (NVIDIA): AmberNet (`langid_ambernet`, downloaded from NGC, not Hugging Face).
- MIT: Silero VAD.

Models that the research benchmarks compare against but that are never installed by this tool (their licences are in
`research/archive/BASELINES.md`): nemotron-speech-streaming-en-0.6b, parakeet_realtime_eou_120m-v1, Sortformer v2.1 and
MarbleNet (NVIDIA Open Model License), parakeet-ctc-0.6b / 1.1b (CC-BY-4.0), smart-turn (BSD-2-Clause), the LiveKit
turn detectors (LiveKit Model License, evaluation only), pyannote, SpeechBrain, Whisper / faster-whisper, WebRTC VAD.

## Datasets used for training research (not shipped)

| dataset | licence | used for | notes |
|---|---|---|---|
| AMI Meeting Corpus | CC BY 4.0 | turn / speaker / VAD heads, meeting ASR adaptation and the 0.6B -> 115M distillation (`scripts/research/distill_0p6b_to_115m/`) | headset mix |
| ICSI Meeting Corpus | ICSI release terms (research) | same | headset mix |
| LibriSpeech | CC BY 4.0 | anchor / WER gate | |
