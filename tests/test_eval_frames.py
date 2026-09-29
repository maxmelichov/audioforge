"""Label frame count vs encoder frame count in Trainer.evaluate (stage-1 eval crash).

Labels (datasets/ami.py, conversation.py, ...) have T = ToneLanguage.n_frames(len(audio)): mel frames
n // 160 + 1, then ceil(l / 2) three times. The encoder's DWStridingSubsampling with
``subsampling_padding: nemo`` and ``causal: true`` (NeMo CausalConv2D, what nemo_import sets for the ported
streaming model) gives floor(l / 2) + 1 per stage, i.e. ceil((M + 7) / 8) vs ceil(M / 8) for M mel frames:
the encoder has exactly ONE frame more than the labels unless M % 8 == 1 (then equal). Never fewer, never +2.
Non-causal nemo and "ours" padding match n_frames exactly. n_frames and the subsampling are left alone;
every evaluation consumer must crop/pad to a common length (as the training losses do via _match_len).
"""
import numpy as np
import pytest
import torch

from audioforge.data import ToneLanguage
from audioforge.features import LogMel
from audioforge.model import SpeechModel
from audioforge.modules.fastconformer import DWStridingSubsampling
from audioforge.train import Trainer

PRE = {"n_mels": 40, "normalize": "fixed", "dither": 0.0}
# the ported-NeMo encoder flavour (nemo_import: pos_emb rel_pos, xscaling, relu, subsampling_padding nemo)
ENC = {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 8, "causal": True,
       "att_context_size": [70, 1], "dropout": 0.0, "pos_emb": "rel_pos", "xscaling": True,
       "subsampling_activation": "relu"}
HEADS = {"hes": {"type": "frame", "key": "hes"},
         "eot": {"type": "frame", "key": "eot"},
         "diar": {"type": "sortformer", "num_spks": 4, "d_hidden": 32, "n_layers": 1, "n_heads": 2},
         "turn": {"type": "turn", "mode": "concat", "hidden": 16}}


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _enc_frames(n_samples: int, padding: str, causal: bool) -> int:
    sub = DWStridingSubsampling(40, 8, channels=2, causal=causal, padding=padding)
    return int(sub.out_lengths(LogMel().num_frames(torch.tensor([n_samples])))[0])


# --------------------------------------------------------------------------- pure length arithmetic
@pytest.mark.parametrize("padding,causal", [("ours", True), ("ours", False), ("nemo", True), ("nemo", False)])
def test_n_frames_vs_subsampling_lengths(padding, causal):
    lens = range(1, 3001, 7)
    diff = {n: _enc_frames(n, padding, causal) - ToneLanguage.n_frames(n) for n in lens}
    if padding == "nemo" and causal:
        # +1 unless the mel count M = n // 160 + 1 is 1 mod 8 (n // 160 a multiple of 8)
        assert all(d == (0 if (n // 160) % 8 == 0 else 1) for n, d in diff.items()), diff
        assert set(diff.values()) == {0, 1}
    else:
        assert set(diff.values()) == {0}


@pytest.mark.parametrize("padding", ["ours", "nemo"])
def test_out_lengths_match_conv_output(padding):
    """out_lengths is what the encoder reports as elen: check it against the actual conv output."""
    torch.manual_seed(0)
    sub = DWStridingSubsampling(40, 8, channels=2, causal=True, padding=padding).eval()
    for n in range(1, 3001, 97):
        m = int(LogMel().num_frames(torch.tensor([n]))[0])
        y, yl = sub(torch.zeros(1, 40, m), torch.tensor([m]))
        assert y.shape[1] == int(yl[0]) == _enc_frames(n, padding, True)


# --------------------------------------------------------------------------- evaluate on off-by-one labels
def _item(n_samples: int, T: int, rng: np.random.Generator) -> dict:
    """AMI turn-window-like item (datasets/ami.py turn_examples keys) with T label frames."""
    tgt = np.zeros((T, 4), np.float32)
    on, end = T // 5, (3 * T) // 5
    tgt[on:end, 0] = 1
    tgt[end + 1: T - 2, 1] = 1
    act = tgt[:, 0].copy()
    eot = np.zeros(T, np.float32)
    eot[end:] = 1
    hes = np.zeros(T, np.float32)
    hes[on + 2: on + 4] = 1
    return dict(audio=(0.01 * rng.standard_normal(n_samples)).astype(np.float32), spk_targets=tgt,
                spk_act=act, eot=eot, hes=hes, speaker=0, onset_frame=on, turn_end_frame=end)


def _model(padding, turn_mode="concat"):
    torch.manual_seed(0)
    enc, heads = {**ENC, "subsampling_padding": padding}, dict(HEADS)
    if turn_mode == "kernel":  # the stage-1 recipe's speaker-conditioned turn head
        enc["speaker_kernel_layers"] = [0]
        heads["turn"] = {**HEADS["turn"], "mode": "kernel", "condition_on_speaker": True}
    cfg = {"preprocessor": PRE, "encoder": enc, "heads": heads,
           "trainer": {"max_steps": 1, "batch_size": 2, "lr": 1e-3, "device": "cpu"}}
    return SpeechModel(cfg), cfg


# 4.2 s (the AMI fixture's first turn window: M = 421, nemo +1) and neighbours incl. M % 8 == 1 (equal)
LENS = [67200, 32000, 32160, 20000, 45440]


@pytest.mark.parametrize("padding", ["ours", "nemo"])
@pytest.mark.parametrize("label_offset", ["n_frames", "minus1", "plus1"])
@pytest.mark.parametrize("turn_mode", ["concat", "kernel"])
def test_evaluate_tolerates_label_frame_offset(padding, label_offset, turn_mode):
    """Frame heads, Sortformer DER and the turn head's eot-bench all run when labels have one frame fewer
    (the real nemo case, 170 vs 169 in runs/stage1_heads_pretrained.log) or one more than the encoder."""
    m, cfg = _model(padding, turn_mode)
    rng = np.random.default_rng(0)
    val = []
    for n in LENS:
        enc_T = _enc_frames(n, padding, True)
        T = {"n_frames": ToneLanguage.n_frames(n), "minus1": enc_T - 1, "plus1": enc_T + 1}[label_offset]
        val.append(_item(n, T, rng))
    if padding == "nemo" and label_offset == "n_frames":
        assert any(_enc_frames(n, padding, True) != len(v["spk_act"]) for n, v in zip(LENS, val))
    res = Trainer(m, cfg).evaluate(val, chunk=3)
    for k in ("acc_hes", "acc_eot", "recall_eot", "der_diar", "eot_turn_p50_ms@5fc", "eot_turn_diar_p50_ms@5fc",
              "eot_timeout_p50_ms@5fc"):
        assert k in res, (k, res)
    assert 0.0 <= res["acc_eot"] <= 1.0 and res["der_diar"] >= 0.0


def test_sortformer_der_crops_to_label_length():
    """DER is computed over min(encoder frames, label frames) per item, identical to cropping by hand."""
    from audioforge.metrics import frame_der
    m, cfg = _model("nemo")
    rng = np.random.default_rng(1)
    n = 67200
    enc_T = _enc_frames(n, "nemo", True)
    val = [_item(n, enc_T - 1, rng)]
    res = Trainer(m, cfg).evaluate(val)
    from audioforge.data import Collate
    b = Collate(None)(val)
    enc, elen, hidden = m.encode(b["audio"], b["audio_len"], return_hidden=True)
    pred = m.heads["diar"].decode(m.head_input("diar", enc, hidden), elen)
    assert pred.shape[1] == enc_T
    exp = frame_der(pred[0, : enc_T - 1], b["spk_targets"][0])
    assert res["der_diar"] == pytest.approx(round(exp, 4))
