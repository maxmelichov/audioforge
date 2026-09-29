"""End-of-file flush fix of the cached streaming Sortformer tracks (scripts/research/make_sortformer_tracks.py --pad-mode).

Fed only the window audio, StreamingDiarizer.feed(final=True) flushes the last chunks with a shrinking right
context, so the last frames of a window differ from what a live stream (which keeps receiving audio) computes.
The fix feeds PAD_FRAMES frames beyond the window end and crops to the window's T frames. The diarizer here is a
tiny random SpeechModel (tests/test_streaming_diar._model) run through the real StreamingDiarizer code path with
the card low-latency config of the script (chunk 6, right context 7)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from audioforge.data import ToneLanguage
from audioforge.datasets import ext_tracks as xt

sys.path.insert(0, str(Path(__file__).parent))
from test_streaming_diar import _model  # noqa: E402

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def mk():
    spec = importlib.util.spec_from_file_location("make_sortformer_tracks", ROOT / "scripts" / "research" / "make_sortformer_tracks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dm():
    return _model(causal=False, seed=1)


def _meeting(n=16000 * 9, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)


def test_pad_constants(mk):
    C, R = mk.SORTFORMER_LOW_LATENCY["chunk_len"], mk.SORTFORMER_LOW_LATENCY["chunk_right_context"]
    assert mk.PAD_FRAMES == C + R + 1 == 14 and mk.FRAME_SAMPLES == 1280
    t = mk.pad_tail(np.zeros(100), np.ones(10 ** 5, np.float32), "audio")
    assert len(t) == 14 * 1280 and (t == 1).all()
    t = mk.pad_tail(np.zeros(100), np.ones(5000, np.float32), "audio")  # meeting end: what is left, no silence
    assert len(t) == 5000 and (t == 1).all()
    t = mk.pad_tail(np.zeros(100), np.ones(10 ** 5), "silence")
    assert len(t) == 14 * 1280 and (t == 0).all()


# window lengths: odd sample counts, last chunk full / partial (T % 6 in {0, 1, 5}), long enough to fill FIFO + cache
@pytest.mark.parametrize("n", [16000 * 5 + 333, 1280 * 60 - 7, 1280 * 61 + 640])
def test_padded_track_equals_long_context_stream(mk, dm, n):
    x = _meeting()
    win, tail = x[:n], x[n:]
    T = ToneLanguage.n_frames(n)
    long = mk.stream_track(dm, "diar", x)[:T]            # the stream keeps going: every window frame has full rc
    old = mk.stream_track(dm, "diar", win)                # v1: flush at the window end
    new = mk.stream_track_padded(dm, "diar", win, tail, "audio")
    assert old.shape == new.shape == long.shape == (T, 4)
    np.testing.assert_array_equal(new, long)              # the fix: bit-identical to the long-context computation
    C, R = 6, 7
    k = T - (T - 1) // C * C + C + R                      # frames whose chunk or right context reaches the flush
    np.testing.assert_array_equal(old[: T - k], long[: T - k])   # the artifact is confined to the end ...
    assert np.abs(old[T - k:] - long[T - k:]).max() > 1e-4       # ... and is there
    # none = v1; silence padding also removes the flush (the last frames see zeros instead of the meeting)
    np.testing.assert_array_equal(mk.stream_track_padded(dm, "diar", win, tail, "none"), old)
    sil = mk.stream_track_padded(dm, "diar", win, tail, "silence")
    assert sil.shape == (T, 4)
    np.testing.assert_array_equal(sil[: T - k], long[: T - k])


def test_padding_is_enough(mk, dm):
    """More audio than PAD_FRAMES changes nothing inside the window (the pad covers the last chunk's full rc)."""
    x = _meeting(seed=3)
    n = 1280 * 55 + 1  # T % 6 == 1: the last chunk holds one window frame and 5 + 7 padded ones
    a = mk.stream_track_padded(dm, "diar", x[:n], x[n:], "audio")
    b = mk.stream_track_padded(dm, "diar", x[:n], x[n: n + mk.PAD_FRAMES * 1280], "audio")
    np.testing.assert_array_equal(a, b)


def test_meeting_tail_is_the_continuation_of_the_clip(mk):
    from types import SimpleNamespace
    x = _meeting(16000 * 3)
    ds = SimpleNamespace(_audio={"M": x})
    ex = {"meeting": "M", "start": 0.5, "audio": x[8000: 8000 + 12345]}
    np.testing.assert_array_equal(mk.meeting_tail(ds, ex), x[8000 + 12345: 8000 + 12345 + 14 * 1280])
    ex = {"meeting": "M", "start": 2.5, "audio": x[40000: 47000]}  # meeting end: shorter tail
    assert len(mk.meeting_tail(ds, ex)) == 1000


def test_main_feeds_the_meeting_tail_and_keeps_v1(mk, monkeypatch, tmp_path):
    """main(): stream tracks get the meeting audio after the window; --regen-v1 keeps the old file as .stream_v1.npy."""
    import json
    from types import SimpleNamespace

    import audioforge.heads.turn as turn
    import audioforge.train as train
    x = _meeting(16000 * 30)
    ds = SimpleNamespace(_audio={"M": x})
    exs = [{"meeting": "M", "start": s, "audio": x[int(s * 16000): int(s * 16000) + 16000 * 4 + 7]}
           for s in (1.0, 10.0, 22.0)]
    exs.append({"meeting": "M", "start": 27.0, "audio": x[27 * 16000: 30 * 16000]})  # ends at the meeting end
    seen, calls = [], []

    def fake_stream(dm_, dn, audio):
        seen.append(np.asarray(audio).copy())
        calls.append(1)
        return np.full((ToneLanguage.n_frames(len(audio)), 4), len(calls), np.float32)

    monkeypatch.setattr(mk, "turn_windows", lambda *a, **k: (exs, ds))
    monkeypatch.setattr(mk.xt, "cache_dir", lambda root, split, dataset="ami": tmp_path)
    monkeypatch.setattr(mk.xt, "manifest", lambda root=None, split="train", directory=None, dataset="ami":
                        json.loads((tmp_path / "manifest.json").read_text()) if (tmp_path / "manifest.json").exists()
                        else {})
    monkeypatch.setattr(train, "load_model", lambda *a, **k: SimpleNamespace(preprocessor=SimpleNamespace(n_mels=128)))
    monkeypatch.setattr(turn, "_diar_name", lambda dm_: "diar")
    monkeypatch.setattr(mk, "stream_track", fake_stream)
    keys = [xt.example_key(e) for e in exs]
    monkeypatch.setattr(sys, "argv", ["x", "--split", "dev", "--track-source", "stream", "--pad-mode", "none"])
    mk.main()  # v1 procedure: window audio only
    assert all(len(a) == len(e["audio"]) for a, e in zip(seen, exs))
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--split", "dev", "--track-source", "stream", "--regen-v1"])
    mk.main()
    need = 14 * 1280
    for a, e in zip(seen, exs):
        i0 = int(round(e["start"] * 16000))
        np.testing.assert_array_equal(a, x[i0: i0 + len(e["audio"]) + need])  # window + the real continuation
    assert [len(a) - len(e["audio"]) for a, e in zip(seen, exs)] == [need] * 3 + [0]  # the last one ends the meeting
    for k, e in zip(keys, exs):
        old, new = np.load(tmp_path / f"{k}.stream_v1.npy"), np.load(xt.track_path(tmp_path, k, "stream"))
        assert old.shape == new.shape == (ToneLanguage.n_frames(len(e["audio"])), 4)
        assert old.max() <= 4 < new.min()  # v1 = the first run's tracks, the new ones the regen run's
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["stream"]["flush_fix"]["pad_mode"] == "audio" and man["stream_v1"]["n"] == 4
    seen.clear()
    mk.main()  # resumable: nothing left to migrate
    assert seen == []
