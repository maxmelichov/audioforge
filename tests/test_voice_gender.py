"""Optional perceived voice-gender head: running posterior == streaming steps == the
served stream's readings, VAD gating, the head file round trip and attach, the protocol fields, off by default, and
the shipped files pinned in the hub as OPTIONAL."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audioforge import hub  # noqa: E402
from audioforge.heads.voice_gender import LABELS, VoiceGenderHead  # noqa: E402
from audioforge.model import build_head  # noqa: E402
from audioforge.server.protocol import validate  # noqa: E402
from audioforge.voice_gender import (  # noqa: E402
    NAME,
    VoiceGenderStream,
    attach_head,
    build_from_file,
    load_head,
    save_head_state,
)


def _head(D=16, H=8):
    torch.manual_seed(0)
    return VoiceGenderHead(D, hidden=H, att_hidden=4).eval()


def test_running_equals_streaming_and_forward():
    h = _head()
    x = torch.randn(2, 23, 16)
    lens = torch.tensor([23, 17])
    keep = torch.rand(2, 23) > 0.3
    keep[:, 0] = True
    z = h.running_logits(x, lens, keep)
    assert torch.allclose(h(x, lens, keep), z[torch.arange(2), lens - 1], atol=1e-5)
    st = h.init_stream(1)
    outs = [h.step(x[:1, a:b], st, keep[:1, a:b]) for a, b in ((0, 2), (2, 3), (3, 11), (11, 23))]
    assert torch.allclose(torch.cat(outs, 1), z[:1], atol=1e-5)


def test_gated_frames_do_not_count_and_padding_is_ignored():
    h = _head()
    x = torch.randn(1, 10, 16)
    keep = torch.ones(1, 10, dtype=torch.bool)
    keep[0, 4:7] = False
    a = h(x, torch.tensor([10]), keep)
    b = h(torch.cat([x[:, :4], x[:, 7:]], 1), torch.tensor([7]))
    assert torch.allclose(a, b, atol=1e-5)
    xp = torch.cat([x, 100 * torch.randn(1, 5, 16)], 1)
    assert torch.allclose(h(x, torch.tensor([10])), h(xp, torch.tensor([10])), atol=1e-5)


def test_loss_and_registry():
    h = build_head({"type": "voice_gender", "hidden": 8, "labels": list(LABELS)}, 16)
    assert isinstance(h, VoiceGenderHead) and h.labels == ["female", "male"]
    x = torch.randn(3, 12, 16, requires_grad=True)
    loss = h.loss(x, torch.tensor([12, 9, 5]), {"voice_gender": torch.tensor([0, 1, 1])})
    loss.backward()
    assert torch.isfinite(loss) and x.grad is not None


class _Enc(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.d_model = d
        self.layers = nn.ModuleList(nn.Identity() for _ in range(n))


class _Model(nn.Module):  # the parts of SpeechModel attach_head / VoiceGenderStream use
    def __init__(self, d=16, n=6):
        super().__init__()
        self.encoder, self.heads, self.head_cfg, self.layer_tap = _Enc(d, n), nn.ModuleDict(), {}, {}
        self.p = nn.Parameter(torch.zeros(1))

    def head_input(self, name, enc, hidden):
        return hidden[self.layer_tap[name][0]] if name in self.layer_tap else enc


def test_file_round_trip_attach_and_stream(tmp_path):
    h = _head()
    f = tmp_path / "vg.pt"
    save_head_state(h.state_dict(), {"type": "voice_gender", "hidden": 8, "att_hidden": 4, "labels": list(LABELS)},
                    [3], 16, f, meta={"core": "test"})
    assert load_head(f)["meta"]["core"] == "test"
    h2, _ = build_from_file(f)
    m = _Model()
    assert attach_head(m, f) == NAME and m.layer_tap[NAME] == [3]
    torch.manual_seed(1)
    hid = [torch.randn(1, 20, 16) for _ in range(6)]
    vad = (torch.rand(20) > 0.3).float().tolist()
    s = VoiceGenderStream(m)
    assert s.session() is None
    for a, b in ((0, 2), (2, 9), (9, 20)):
        s.feed(None, [x[:, a:b] for x in hid], vad[a:b])
    keep = torch.tensor([v > 0.5 for v in vad])[None]
    with torch.no_grad():
        want = h2(hid[3], torch.tensor([20]), keep).softmax(-1)[0]
    got = s.session()
    assert abs(got["male"] - float(want[1])) < 1e-3 and abs(got["female"] + got["male"] - 1) < 1e-3
    assert got["speech_ms"] == 80 * int(keep.sum())
    seg = s.segment(5, 14)  # frames 5..13 only, then the buffer drops them
    k2 = keep.clone()
    k2[:, :5] = False
    k2[:, 14:] = False
    with torch.no_grad():
        want = h2(hid[3], torch.tensor([20]), k2).softmax(-1)[0]
    assert abs(seg["male"] - float(want[1])) < 1e-3
    assert s.segment(0, 5) is None  # already released
    s.feed(None, [x[:, :3] for x in hid], [1.0, 1.0, 0.0])  # 3 more frames after the release
    keep3 = torch.cat([keep, torch.tensor([[True, True, False]])], 1)
    with torch.no_grad():
        want = h2(torch.cat([hid[3], hid[3][:, :3]], 1), torch.tensor([23]), keep3).softmax(-1)[0]
    got = s.session()  # every frame pooled exactly once across releases
    assert abs(got["male"] - float(want[1])) < 1e-3 and got["speech_ms"] == 80 * int(keep3.sum())


def test_attach_rejects_other_encoder(tmp_path):
    h = _head()
    f = tmp_path / "vg.pt"
    save_head_state(h.state_dict(), {"type": "voice_gender", "hidden": 8, "att_hidden": 4, "labels": list(LABELS)},
                    [3], 16, f)
    with pytest.raises(AssertionError):
        attach_head(_Model(d=32), f)


def test_protocol_fields_are_optional():
    fin = {"type": "final", "t": 1.0, "text": "hi", "speaker": None}
    validate(fin)
    validate({**fin, "voice_gender": {"female": 0.2, "male": 0.8, "speech_ms": 960}})
    validate({**fin, "voice_gender": None})
    with pytest.raises(ValueError):
        validate({**fin, "voice_gender": 0.8})


def test_off_by_default():
    from audioforge.server.cli import MODES, parse_args
    assert all("voice_gender" not in m for m in MODES.values())
    assert parse_args(["--asr", "a.afm", "--diar", "d.afm"]).voice_gender is None
    assert "voice_gender" not in hub.SINGLE and "voice_gender_0p6b" not in hub.SINGLE


@pytest.mark.parametrize("key,d", [("voice_gender", 512), ("voice_gender_0p6b", 1024)])
def test_shipped_head_pinned(key, d):
    c = hub.COMPONENTS[key]
    assert key in hub.OPTIONAL
    p = ROOT / "assets" / c.output
    if not p.exists():
        pytest.skip(f"{p} not in this checkout")
    assert p.stat().st_size == c.size and hub.sha256_file(p) == c.sha256
    head, blob = build_from_file(p)
    assert blob["encoder"]["d_model"] == d and head.labels == ["female", "male"]
    assert blob["from_layers"] == [3 if d == 512 else 4]  # the speaker head's tap (block 4 / block 5)
