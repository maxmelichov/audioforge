"""The classifier's clock on a second (stateless) VAD (VadHeadPolicy ``vad_m``, served as a
preset's ``model_clock`` with the model's ``turn_vad`` head). Tiny random models, CPU, no checkpoints."""
import numpy as np
import torch

from audioforge.model import SpeechModel
from audioforge.server.policies import VadHeadPolicy
from audioforge.server.streams import ASRStream
from audioforge.tokenizer import CharTokenizer


def _run(pol, vad, vm=None, p=0.9):
    out = []
    for v in range(len(vad)):
        ev = pol.update(p, vad[v], None, None, None, None if vm is None else vm[v])
        if ev is not None:
            out.append((v, ev["path"]))
    return out


def _pol(**kw):
    return VadHeadPolicy(0.99, 3, 12, 0.4, others=None, turn_model=lambda v, o: (0.9, 0.0), model_p=0.5,
                         model_vad_thr=0.4, **kw)


def test_model_clock_on_second_vad():
    # speech, then a pause where the (GRU) VAD flickers around 0.5 but the stateless VAD is quiet
    vad = [0.9] * 10 + [0.5, 0.3] * 10
    vm = [0.9] * 10 + [0.05] * 20
    assert _run(_pol(), vad) == []  # the flicker restarts the classifier's clock: never asked, fallback never reached
    got = _run(_pol(), vad, vm)
    assert got == [(12, "model")]  # asked at 3 quiet frames of the stateless clock
    # without vad_m the policy is unchanged (vad_m=None is the default)
    assert _run(_pol(), vad, None) == _run(_pol(), vad)


def test_reset_thr_unchanged_without_vad_m():
    vad = [0.9] * 5 + [0.3] * 3 + [0.1] * 10
    a = _run(_pol(reset_thr=0.25), vad)
    b = _run(_pol(reset_thr=0.25), vad, vad)  # vad_m == vad: identical clocks
    assert a == b and a == [(10, "model")]


def _model(turn_vad=True, seg_input=None):
    torch.manual_seed(0)
    heads = {"rnnt": {"type": "rnnt", "pred_hidden": 16, "joint_hidden": 16},
             "vad": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": [0]}}
    if turn_vad:
        heads["turn_vad"] = {"type": "frame", "key": "turn_vad", "hidden": 8, "from_layers": [1], "weight": 0.0}
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, 1], "dropout": 0.0},
           "heads": heads}
    return SpeechModel(cfg, CharTokenizer(list("abc "))).eval()


def test_turn_vad_streams_per_frame_and_vad_unchanged():
    x = (np.random.default_rng(1).standard_normal(16000 * 2) * 0.1).astype(np.float32)
    res = {}
    for tv in (True, False):
        s = ASRStream(_model(turn_vad=tv), turn=None)
        fr = []
        for i in range(0, len(x), 777):
            fr += s.feed_frames(x[i:i + 777])
        fr += s.feed_frames(np.zeros(0, np.float32), final=True)
        res[tv] = (fr, s)
    fr, s = res[True]
    assert len(s.turn_vad_p) == len(fr) and len(res[False][1].turn_vad_p) == 0
    assert [round(f["vad"], 6) for f in fr] == [round(f["vad"], 6) for f in res[False][0]]
    # streamed turn VAD equals the whole-utterance masked forward of the same head
    m = _model(turn_vad=True)
    with torch.no_grad():
        xx = torch.from_numpy(x)[None]
        enc, elen, hid = m.encode(xx, torch.tensor([len(x)]), s.att, return_hidden=True)
        ref = m.heads["turn_vad"](m.head_input("turn_vad", enc, hid)).sigmoid()[0, : len(fr)].numpy()
    got = np.array([s.turn_vad_p[v] for v in range(len(fr))])
    assert np.allclose(got, ref[: len(got)], atol=2e-4)


def _model2():
    torch.manual_seed(0)
    seg = {"d_in": 32, "n_extra": 3, "d": 16, "n_layers": 1, "heads": 2, "ff": 32, "win": 12, "d_text": 16,
           "text_layers": 1, "max_tok": 6, "vocab": 1024}
    heads = {"rnnt": {"type": "rnnt", "pred_hidden": 16, "joint_hidden": 16},
             "vad": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": [0]},
             "turn_vad": {"type": "frame", "key": "turn_vad", "hidden": 8, "from_layers": [1], "weight": 0.0},
             "turn_seg": {"type": "turn_seg", "weight": 0.0, **seg, "block": "1"},
             "turn_seg_a": {"type": "turn_seg", "weight": 0.0, **seg, "block": "2", "vad_input": "turn_vad"}}
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, 1], "dropout": 0.0},
           "heads": heads}
    return SpeechModel(cfg, CharTokenizer(list("abc "))).eval()


def test_second_classifier_reads_its_block_and_turn_vad():
    """attach_seg2: the preset's own classifier sees its own block and (vad_input turn_vad) the stateless turn VAD;
    the primary classifier keeps heads.vad; both equal a classifier fed the same inputs offline."""
    from audioforge.heads.turn_seg import SegTurnStream
    m = _model2()
    x = (np.random.default_rng(2).standard_normal(16000 * 2) * 0.1).astype(np.float32)
    s = ASRStream(m, turn=None)
    s.attach_seg(m.heads["turn_seg"])
    s.attach_seg2("turn_seg_a")
    fr = []
    for i in range(0, len(x), 640):
        fr += s.feed_frames(x[i:i + 640])
    with torch.no_grad():
        enc, elen, hid = m.encode(torch.from_numpy(x)[None], torch.tensor([len(x)]), s.att, return_hidden=True)
    T = len(fr)
    h2 = hid[1][0, :T].numpy()
    tv = np.array([s.turn_vad_p[v] for v in range(T)])
    ref = SegTurnStream(m.heads["turn_seg_a"])
    for v in range(T):
        ref.push(h2[v], float(tv[v]), 0.0, 0.0, s.tok_at[v])
    for v in (T // 2, T - 1):
        assert abs(s.seg2_prob(v) - ref.prob(v, s.tokens)) < 2e-4
    # the primary classifier is fed heads.vad (not the turn VAD)
    vad = np.array([f["vad"] for f in fr])
    assert np.allclose([s.seg.ex[v][0] for v in range(T - 5, T)], vad[T - 5:], atol=1e-6)
