"""Edge cases of audioforge.teachers: pseudo_label WAV naming, NeMo offset/duration segments, and
RNNT batched decoding (a short clip batched with a long one must decode as if alone)."""
import json
import os

import numpy as np
import pytest
import torch

from audioforge.data import load_wav, save_wav
from audioforge.teachers import SAMPLE_RATE, pseudo_label


class _LenStub:
    """Labels each clip with its length in samples, so clips can be told apart."""
    name = "stub"

    def transcribe(self, audios, batch_size=8, timestamps=False):
        return [f"len {len(a)}" for a in audios]


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _check_consistent(rows):
    names = [os.path.basename(r["audio_filepath"]).lower() for r in rows]
    assert len(set(names)) == len(names), names
    for r in rows:
        n = len(load_wav(r["audio_filepath"]))
        assert r["text"] == f"len {n}", (r["audio_filepath"], r["text"], n)


def test_pseudo_label_wav_names_never_collide(tmp_path):
    layout = (("d1/a.wav", 16000), ("d1/a_2.wav", 24000), ("d2/a.wav", 32000), ("d3/B.wav", 8000),
              ("d4/b.wav", 4000))
    for rel, n in layout:
        (tmp_path / "src" / rel).parent.mkdir(parents=True, exist_ok=True)
        save_wav(tmp_path / "src" / rel, np.zeros(n, np.float32))
    wav = tmp_path / "wav"
    pseudo_label(tmp_path / "src", tmp_path / "out1.jsonl", _LenStub(), wav_dir=wav, log_every=0)
    first = _rows(tmp_path / "out1.jsonl")
    assert len(first) == len(layout)
    _check_consistent(first)
    # a second run into the same wav_dir must not overwrite the first run's copies
    pseudo_label(tmp_path / "src", tmp_path / "out2.jsonl", _LenStub(), wav_dir=wav, log_every=0)
    second = _rows(tmp_path / "out2.jsonl")
    _check_consistent(first)
    _check_consistent(second)
    _check_consistent(first + second)


def test_pseudo_label_honors_offset_duration_segments(tmp_path):
    x = np.zeros(SAMPLE_RATE * 20, np.float32)
    save_wav(tmp_path / "long.wav", x)
    src = [{"audio_filepath": str(tmp_path / "long.wav"), "offset": 5.0, "duration": 2.5, "text": "one"},
           {"audio_filepath": str(tmp_path / "long.wav"), "offset": 12.0, "duration": 4.0, "text": "two"},
           {"audio_filepath": str(tmp_path / "long.wav"), "duration": 20.0, "text": "whole"}]
    (tmp_path / "in.jsonl").write_text("".join(json.dumps(r) + "\n" for r in src))

    pseudo_label(tmp_path / "in.jsonl", tmp_path / "out.jsonl", _LenStub(), log_every=0)
    rows = _rows(tmp_path / "out.jsonl")
    assert [r["text"] for r in rows] == ["len 40000", "len 64000", f"len {len(x)}"]
    assert [(r.get("offset"), r["duration"]) for r in rows] == [(5.0, 2.5), (12.0, 4.0), (None, 20.0)]
    assert [r["ref_text"] for r in rows] == ["one", "two", "whole"]

    # with wav_dir each segment is written as its own WAV and the offset no longer applies
    pseudo_label(tmp_path / "in.jsonl", tmp_path / "out_wav.jsonl", _LenStub(), wav_dir=tmp_path / "wav",
                 log_every=0)
    rows = _rows(tmp_path / "out_wav.jsonl")
    assert all("offset" not in r for r in rows)
    assert [r["duration"] for r in rows] == [2.5, 4.0, 20.0]
    _check_consistent(rows)


def _tiny_teacher(repo, auto_name, kind):
    """Random-init tiny model with the real config / processor / generation config (HF cache only)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # restored after each test by tests/conftest.py
    transformers = pytest.importorskip("transformers")
    from audioforge.teachers import Teacher, _patch_librosa
    _patch_librosa()
    try:
        cfg = transformers.AutoConfig.from_pretrained(repo)
        processor = transformers.AutoProcessor.from_pretrained(repo)
        gen_cfg = transformers.GenerationConfig.from_pretrained(repo)
    except OSError as ex:  # not in the local HF cache (LocalEntryNotFoundError / OSError from from_pretrained)
        pytest.skip(f"{repo} not cached: {ex}")
    e = cfg.encoder_config
    e.hidden_size, e.intermediate_size, e.num_hidden_layers, e.subsampling_conv_channels = 32, 64, 1, 8
    cfg.decoder_hidden_size = 16
    torch.manual_seed(0)
    m = getattr(transformers, auto_name).from_config(cfg).eval()
    # a random model emits max_symbols_per_step tokens on every frame, which overflows HF's default output
    # buffer (max_symbols_per_step * encoder frames, incl. the start token) before the last frame
    gen_cfg.max_new_tokens = 4096
    m.generation_config = gen_cfg
    return Teacher(f"tiny-{kind}", m, processor, kind, torch.device("cpu"))


@pytest.mark.parametrize("repo,auto_name,kind", [
    ("nvidia/nemotron-speech-streaming-en-0.6b", "AutoModelForRNNT", "rnnt"),
    ("nvidia/parakeet-tdt-0.6b-v3", "AutoModelForTDT", "tdt"),
])
def test_transducer_batched_equals_solo(repo, auto_name, kind):
    torch.set_num_threads(1)
    t = _tiny_teacher(repo, auto_name, kind)
    rng = np.random.default_rng(0)
    short, long_ = [(0.1 * rng.standard_normal(int(SAMPLE_RATE * s))).astype(np.float32) for s in (0.6, 3.0)]
    solo = t.transcribe([short], timestamps=True)[0]
    batched = t.transcribe([short, long_], timestamps=True)[0]
    assert batched["text"] == solo["text"]
    assert batched["tokens"] == solo["tokens"]
    assert batched["words"] == solo["words"]
    assert t.transcribe([short, long_])[0] == t.transcribe([short])[0]
    if solo["tokens"]:
        assert solo["tokens"][-1]["end"] <= len(short) / SAMPLE_RATE + 2 * t.frame_sec
