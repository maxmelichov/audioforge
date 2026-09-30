"""Spoken language ID (research/archive/LID.md): LanguageHead running posterior == streaming steps, padding invariance,
loss, the model registry / single-block tap, the head file round trip, the serving confidence rule and the AmberNet
x-vector decoder."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audioforge.heads.audio import LanguageHead  # noqa: E402


def _head(D=16, L=5):
    torch.manual_seed(0)
    return LanguageHead(D, L, hidden=12, att_hidden=8, cls_hidden=10, labels=[f"l{i}" for i in range(L)]).eval()


def test_running_equals_streaming_and_forward():
    h = _head()
    x = torch.randn(2, 23, 16)
    lens = torch.tensor([23, 17])
    z = h.running_logits(x, lens)
    # forward == running posterior at the last valid frame
    assert torch.allclose(h(x, lens), z[torch.arange(2), lens - 1], atol=1e-5)
    # streaming in uneven chunks == running posterior
    st = h.init_stream(1)
    outs = [h.step(x[:1, a:b], st) for a, b in ((0, 2), (2, 3), (3, 11), (11, 23))]
    assert torch.allclose(torch.cat(outs, 1), z[:1], atol=1e-5)


def test_padding_does_not_change_valid_frames():
    h = _head()
    x = torch.randn(1, 10, 16)
    a = h.running_logits(x, torch.tensor([10]))
    xp = torch.cat([x, 100 * torch.randn(1, 5, 16)], 1)
    b = h.running_logits(xp, torch.tensor([10]))
    assert torch.allclose(a, b[:, :10], atol=1e-5)
    assert torch.allclose(h(x, torch.tensor([10])), h(xp, torch.tensor([10])), atol=1e-5)


def test_causal():
    h = _head()
    x = torch.randn(1, 12, 16)
    y = x.clone()
    y[:, 8:] = torch.randn(1, 4, 16)
    za, zb = h.running_logits(x, torch.tensor([12])), h.running_logits(y, torch.tensor([12]))
    assert torch.allclose(za[:, :8], zb[:, :8], atol=1e-6) and not torch.allclose(za[:, 8:], zb[:, 8:])


def test_loss_trains():
    h = _head().train()
    x = torch.randn(8, 20, 16)
    lens = torch.tensor([20, 20, 3, 15, 20, 1, 20, 9])
    y = torch.arange(8) % 5
    x = x + 2 * torch.nn.functional.one_hot(y, 16)[:, None].float()
    opt = torch.optim.Adam(h.parameters(), 1e-2)
    l0 = None
    for _ in range(60):
        l = h.loss(x, lens, {"lang": y})
        assert torch.isfinite(l)
        l0 = l0 if l0 is not None else float(l.detach())
        opt.zero_grad()
        l.backward()
        opt.step()
    assert float(l.detach()) < 0.5 * l0


def _tiny_model():
    from audioforge.model import SpeechModel
    cfg = {"encoder": {"d_model": 32, "n_layers": 3, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, 1]},
           "heads": {"vad": {"type": "frame", "key": "vad"},
                     "lid": {"type": "language", "num_languages": 3, "hidden": 8, "att_hidden": 4, "cls_hidden": 8,
                             "labels": ["en", "he", "ar"], "from_layers": [1]}}}
    torch.manual_seed(0)
    return SpeechModel(cfg).eval()


def test_model_registry_and_tap():
    m = _tiny_model()
    audio = torch.randn(2, 16000)
    enc, elen, hid = m.encode(audio, torch.tensor([16000, 12000]), return_hidden=True)
    assert m.head_input("lid", enc, hid) is hid[1]
    p = m.analyze([np.random.randn(8000).astype(np.float32)])["lid"]
    assert p.shape == (1, 3) and abs(float(p.sum()) - 1) < 1e-5
    out = m({"audio": audio, "audio_len": torch.tensor([16000, 12000]), "lang": torch.tensor([0, 2])})
    assert "loss_lid" in out and torch.isfinite(out["loss_lid"])


def test_head_file_roundtrip_and_attach(tmp_path):
    from audioforge.lid import attach_head, load_head, save_head
    m = _tiny_model()
    p = tmp_path / "lid.pt"
    save_head(m, "lid", p, meta={"note": "x"})
    blob = load_head(p)
    assert blob["labels"] == ["en", "he", "ar"] and blob["from_layers"] == [1] and blob["meta"]["note"] == "x"
    m2 = _tiny_model()
    del m2.heads["lid"], m2.head_cfg["lid"], m2.layer_tap["lid"]
    name = attach_head(m2, p)
    for k, v in m.heads["lid"].state_dict().items():
        assert torch.equal(v, m2.heads[name].state_dict()[k])
    assert m2.layer_tap[name] == [1]


def test_stream_decider():
    from audioforge.lid import LangDecider
    d = LangDecider(["en", "he"], threshold=0.8, min_ms=400, frame_ms=80)
    ev = []
    for t in range(1, 30):
        p = np.array([0.9, 0.1]) if t < 15 else np.array([0.05, 0.95])
        e = d.update(p, t * 0.08)
        if e:
            ev.append(e)
    assert [e["language"] for e in ev] == ["en", "he"]
    assert ev[0]["t"] >= 0.4 and all(0 <= e["confidence"] <= 1 for e in ev)
    # no event below the threshold
    d2 = LangDecider(["en", "he"], threshold=0.8, min_ms=0)
    assert all(d2.update(np.array([0.6, 0.4]), t * 0.08) is None for t in range(1, 10))


def test_xvector_decoder_matches_manual():
    from audioforge.baselines.lid import XVectorDecoder
    torch.manual_seed(0)
    dec = XVectorDecoder(6, 4, 3).eval()
    x = torch.randn(1, 6, 11)
    z = dec(x, torch.tensor([11]))
    pooled = torch.cat([x.mean(-1), x.std(-1)], -1)
    assert torch.allclose(z, dec.final(dec.emb_layers[0](pooled)), atol=1e-5)


# --------------------------------------------------------------------------- server (--lid)
def _serve_engine(tmp_path, with_lid=True):
    import pytest
    pytest.importorskip("websockets")
    from audioforge.lid import save_head
    from audioforge.model import SpeechModel
    from audioforge.serve import Engine
    from audioforge.tokenizer import CharTokenizer
    sys.path.insert(0, str(ROOT / "tests"))
    from test_serve import _asr_model, _diar_model
    asr = _asr_model()
    with torch.no_grad():  # VAD always "speech" so every frame is pooled
        asr.heads["vad"].net[-1].bias.fill_(10.0)
    path = None
    if with_lid:
        cfg = {**asr.cfg, "heads": {**asr.cfg["heads"], "lid": {"type": "language", "num_languages": 2, "hidden": 8,
                                                                  "att_hidden": 4, "cls_hidden": 8,
                                                                  "labels": ["en", "he"], "from_layers": [0]}}}
        donor = SpeechModel(cfg, CharTokenizer(list("abc ")))
        with torch.no_grad():
            donor.heads["lid"].cls[-1].bias.copy_(torch.tensor([8.0, 0.0]))  # confidently "en"
        path = tmp_path / "lid.pt"
        save_head(donor, "lid", path)
    return Engine(asr, _diar_model(), name="tiny", threads=1, debug=False, lid=path, lid_threshold=0.9,
                  lid_min_ms=400)


def test_serve_lid_messages(tmp_path):
    from audioforge.serve import Session, SessionConfig, validate
    eng = _serve_engine(tmp_path)
    x = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.1).astype(np.float32)
    s = Session(eng, SessionConfig())
    msgs = []
    for i in range(0, len(x), 320):
        msgs += s.process(x[i:i + 320])
    msgs += s.finish()
    for m in msgs:
        validate(m)
    lang = [m for m in msgs if m["type"] == "language"]
    assert len(lang) == 1 and lang[0]["language"] == "en" and lang[0]["confidence"] >= 0.9
    assert lang[0]["t"] >= 0.4  # min_ms of pooled speech
    assert msgs[-1]["type"] == "stats" and msgs[-1]["lang"] == "en"


def test_serve_default_protocol_unchanged(tmp_path):
    from audioforge.serve import Session, SessionConfig, validate
    eng = _serve_engine(tmp_path, with_lid=False)
    x = (np.random.default_rng(0).standard_normal(16000) * 0.1).astype(np.float32)
    s = Session(eng, SessionConfig())
    msgs = s.process(x) + s.finish()
    for m in msgs:
        validate(m)
    assert not any(m["type"] == "language" for m in msgs) and "lang" not in msgs[-1]
    assert "lid" not in eng.asr.heads


def test_validate_language_message():
    import pytest

    from audioforge.serve import validate
    validate({"type": "language", "t": 1.2, "language": "he", "confidence": 0.93})
    with pytest.raises(ValueError):
        validate({"type": "language", "t": 1.2, "language": "he", "confidence": 1.5})
    with pytest.raises(ValueError):
        validate({"type": "language", "t": 1.2, "language": "he"})
    validate({"type": "stats", "rtf": 0.3, "chunk_ms_p50": 1.0, "chunk_ms_p95": 2.0, "first_partial_ms": None,
              "peak_rss_mb": 1.0, "lang": None})


class _FakeAmber:
    def __init__(self):
        self.calls = []

    def probs(self, audio):
        self.calls.append(len(audio))
        return {"en": 0.05, "iw": 0.0, "he": 0.9, "fr": 0.05}


def test_ambernet_stream_rule():
    from audioforge.lid import AmberNetLIDStream
    m = _FakeAmber()
    s = AmberNetLIDStream(m, ["en", "he"], threshold=0.9, min_ms=800, period_ms=480, window_s=2.0,
                          checkpoints_ms=(800, 1200))
    ev = []
    for v in range(0, 40, 2):  # 2 frames per chunk
        s.feed_audio(np.zeros(2560, np.float32))
        ev += s.feed(None, None, [0.9, 0.2 if v == 10 else 0.9], [(v + 1) * 0.08, (v + 2) * 0.08])
    assert [e["language"] for e in ev] == ["he"] and ev[0]["confidence"] > 0.9 and s.current == "he"
    # runs at 800 and 1200 ms of speech, then every 480 ms of new speech; windows capped at 2 s
    assert s.runs == len(m.calls) >= 4 and max(m.calls) <= 32000
    assert m.calls[0] == 10 * 1280 and m.calls[1] == 15 * 1280
    assert ev[0]["t"] >= 0.8


def test_serve_ambernet_backend(tmp_path):
    from audioforge.serve import Session, SessionConfig, validate
    eng = _serve_engine(tmp_path, with_lid=False)
    eng.lid_model, eng.lid_name, eng.lid_langs = _FakeAmber(), "ambernet", ["en", "he"]
    x = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.1).astype(np.float32)
    s = Session(eng, SessionConfig())
    msgs = []
    for i in range(0, len(x), 320):
        msgs += s.process(x[i:i + 320])
    msgs += s.finish()
    for m in msgs:
        validate(m)
    assert [m["language"] for m in msgs if m["type"] == "language"] == ["he"] and msgs[-1]["lang"] == "he"


def test_fast_depthwise_matches_conv():
    from audioforge.baselines.lid import fast_depthwise
    from audioforge.baselines.sd import ConvEncoder
    torch.manual_seed(0)
    jasper = [{"filters": 16, "repeat": 1, "kernel": [3], "stride": [1], "dilation": [1], "residual": False,
               "separable": True, "se": True},
              {"filters": 16, "repeat": 2, "kernel": [7], "stride": [1], "dilation": [1], "residual": True,
               "separable": True, "se": True}]
    a, b = ConvEncoder(8, jasper).eval(), ConvEncoder(8, jasper).eval()
    b.load_state_dict(a.state_dict())
    assert fast_depthwise(b) == 3
    x, lens = torch.randn(2, 8, 30), torch.tensor([30, 21])
    with torch.no_grad():
        (ya, la), (yb, lb) = a(x, lens), b(x, lens)
    assert torch.equal(la, lb) and torch.allclose(ya, yb, atol=1e-5)


# --------------------------------------------------------------------------- fix pass (research/archive/LID.md, 2026-09-29)
def test_decider_timeout_rule():
    """Evidence threshold or timeout: announce at p >= threshold, or the top language after max_ms of pooled speech
    if nothing was confident; max_ms=None is the previous rule."""
    from audioforge.lid import LangDecider
    p = np.array([0.6, 0.4])
    d = LangDecider(["en", "he"], threshold=0.9, min_ms=1000, frame_ms=80, max_ms=3000)
    ev = [e for t in range(1, 60) for e in [d.update(p, t * 0.08)] if e]
    assert len(ev) == 1 and ev[0]["language"] == "en" and abs(ev[0]["t"] - 38 * 0.08) < 1e-6  # 38 frames >= 3 s
    assert 0 <= ev[0]["confidence"] <= 1
    # a confident call before the timeout wins, and the timeout never fires after something was announced
    d = LangDecider(["en", "he"], threshold=0.9, min_ms=1000, frame_ms=80, max_ms=3000)
    ev = [e for t in range(1, 60) for e in [d.update(np.array([0.05, 0.95]) if t < 20 else p, t * 0.08)] if e]
    assert [e["language"] for e in ev] == ["he"] and ev[0]["t"] < 3.0
    # default: no timeout
    d = LangDecider(["en", "he"], threshold=0.9, min_ms=1000, frame_ms=80)
    assert all(d.update(p, t * 0.08) is None for t in range(1, 100))


import pytest  # noqa: E402


@pytest.mark.parametrize("context,rnn", [(0, 0), (3, 0), (0, 16), (2, 16)])
def test_fix_head_shape_and_stream(context, rnn):
    """The fix-pass head layout (LanguageHead, 17 languages, optional causal left context): logits (B, 17), the
    streaming step equals the offline running posterior (also with VAD gating), padding and future frames do not
    change earlier posteriors."""
    from audioforge.model import build_head
    labels = ["en", "he", "ar", "ru", "es", "fr", "de", "pt", "it", "nl", "pl", "uk", "tr", "fa", "hi", "zh", "ja"]
    cfg = {"type": "language", "num_languages": 17, "hidden": 32, "att_hidden": 8, "cls_hidden": 16, "dropout": 0.2,
           "min_frames": 6, "labels": labels, "context": context, "rnn": rnn}
    torch.manual_seed(0)
    h = build_head(dict(cfg), 24).eval()
    x, lens = torch.randn(3, 40, 24), torch.tensor([40, 26, 7])
    assert h(x, lens).shape == (3, 17)
    z = h.running_logits(x, lens)
    st = h.init_stream(1)
    s = torch.cat([h.step(x[:1, a:b], st) for a, b in ((0, 2), (2, 4), (4, 40))], 1)
    assert torch.allclose(s, z[:1], atol=1e-5)
    # padding / future frames
    xp = x[1:2].clone()
    xp[:, 26:] = 50 * torch.randn(1, 14, 24)
    assert torch.allclose(h.running_logits(xp, torch.tensor([40]))[:, :26], z[1:2, :26], atol=1e-5)
    # gated streaming == offline gated sums
    keep = torch.rand(1, 40) > 0.3
    st = h.init_stream(1)
    g = torch.cat([h.step(x[:1, a:b], st, keep=keep[:, a:b]) for a, b in ((0, 5), (5, 40))], 1)
    w, wx, wxx = h._terms(x[:1])
    k = keep.float()[..., None]
    assert torch.allclose(g, h._classify((w * k).cumsum(1), (wx * k).cumsum(1), (wxx * k).cumsum(1)), atol=1e-5)
    assert len(h.init_stream(1)) == 3 + bool(context) + bool(rnn)


def test_serve_lid_does_not_change_the_stream(tmp_path):
    """Streaming with and without --lid is bit-identical: every frame / partial / turn / final message is the same;
    LID only adds 'language' messages and stats.lang."""
    from audioforge.serve import Session, SessionConfig
    x = (np.random.default_rng(1).standard_normal(16000 * 3) * 0.1).astype(np.float32)
    outs = []
    for with_lid in (False, True):
        eng = _serve_engine(tmp_path, with_lid=with_lid)
        s = Session(eng, SessionConfig())
        msgs = []
        for i in range(0, len(x), 640):
            msgs += s.process(x[i:i + 640])
        msgs += s.finish()
        outs.append(msgs)
    strip = [[m for m in ms if m["type"] not in ("language", "stats")] for ms in outs]
    assert strip[0] == strip[1] and len(strip[0]) > 10
    assert any(m["type"] == "language" for m in outs[1]) and not any(m["type"] == "language" for m in outs[0])


def test_resolve_head_alias(tmp_path, monkeypatch):
    from audioforge import lid
    p = tmp_path / lid.HEAD_FILE
    p.write_bytes(b"x")
    monkeypatch.setenv("AUDIOFORGE_HOME", str(tmp_path))
    assert lid.resolve_head("head") == str(p)
    assert lid.resolve_head("some/other.pt") == "some/other.pt"


def test_serve_head_alias_sets_timeout(tmp_path, monkeypatch):
    """--lid head: the shipped head file from the models directory, with the fix-pass rule (max_ms 3000)."""
    import shutil

    from audioforge import lid
    eng0 = _serve_engine(tmp_path, with_lid=True)  # writes tmp_path/lid.pt (a tiny head for the tiny encoder)
    shutil.copyfile(tmp_path / "lid.pt", tmp_path / lid.HEAD_FILE)
    monkeypatch.setenv("AUDIOFORGE_HOME", str(tmp_path))
    from audioforge.serve import Engine
    eng = Engine(eng0.asr, eng0.diar, name="tiny", threads=1, debug=False, lid="head")
    assert eng.lid_name == "lid" and eng.lid_max_ms == lid.HEAD_MAX_MS
    eng = Engine(eng0.asr, eng0.diar, name="tiny", threads=1, debug=False, lid="head", lid_max_ms=0)
    assert eng.lid_max_ms is None
