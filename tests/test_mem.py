"""Memory guards for the pretrained stage recipes (stage-1 OOM: ``Tried to allocate 1.94 GiB``).

1. SpeechModel.forward skips a head's loss when it cannot produce a gradient (weight 0, or the head
   and everything upstream of it frozen).
2. greedy_align (TurnHead forced alignment) never builds more than one item's lattice at a time and
   chunks it over U above a byte budget; results identical to the previous implementation.
3. Peak-temporary arithmetic for B=8, T=250, U=150, V=1025 (printed as info, nothing allocated).
"""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from audioforge.data import Collate, ToneLanguage, synthetic_dataset
from audioforge.heads.asr import RNNTHead
from audioforge.heads.turn import align_chunk, greedy_align
from audioforge.model import SpeechModel
from audioforge.tokenizer import CharTokenizer


@pytest.fixture(autouse=True, scope="module")
def _threads_1():
    """1 torch thread for this module only (restored after it, not at import time for the session)."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


ROOT = Path(__file__).parent.parent

CFG = {
    "preprocessor": {"n_mels": 40, "normalize": "fixed", "dither": 0.0},
    "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 8, "causal": True,
                "att_context_size": [8, 1], "dropout": 0.0},
    "heads": {"rnnt": {"type": "rnnt", "pred_hidden": 32, "joint_hidden": 32},
              "ctc": {"type": "ctc", "weight": 0.3},
              "vad": {"type": "frame", "key": "vad", "weight": 0.5}},
}


def _model_batch(cfg=CFG):
    torch.manual_seed(0)
    lang = ToneLanguage(seed=0)
    tok = CharTokenizer(sorted(set(" " + "".join(lang.LEXICON))))
    model = SpeechModel(cfg, tok).eval()  # eval: deterministic, but autograd stays on
    batch = Collate(tok)(synthetic_dataset("multitask", 4, seed=0, lang=lang))
    assert "vad" in batch and "text" in batch
    return model, batch


# --------------------------------------------------------------------------- 1. skipped losses
def test_nothing_frozen_unchanged():
    model, batch = _model_batch()
    out = model(batch)
    assert {"loss_rnnt", "loss_ctc", "loss_vad"} <= set(out)
    w = {k: model.head_cfg[k].get("weight", 1.0) for k in ("rnnt", "ctc", "vad")}
    ref = sum(w[k] * out[f"loss_{k}"] for k in w)
    assert torch.allclose(out["loss"], ref) and out["loss"].requires_grad


def test_frozen_head_and_encoder_skipped(capsys):
    model, batch = _model_batch()
    model.encoder.requires_grad_(False)
    model.heads["rnnt"].requires_grad_(False)
    out = model(batch)
    assert "loss_rnnt" not in out and {"loss_ctc", "loss_vad"} <= set(out)
    ref = 0.3 * out["loss_ctc"] + 0.5 * out["loss_vad"]
    assert torch.allclose(out["loss"], ref)
    out["loss"].backward()  # still trains the trainable heads
    assert model.heads["ctc"].proj.weight.grad is not None
    model(batch)  # logged once
    assert capsys.readouterr().out.count("skipping loss") == 1


def test_frozen_head_trainable_encoder_kept():
    model, batch = _model_batch()
    model.heads["rnnt"].requires_grad_(False)  # gradient still reaches the encoder through it
    assert "loss_rnnt" in model(batch)


def test_weight_zero_skipped():
    cfg = {**CFG, "heads": {**CFG["heads"], "ctc": {"type": "ctc", "weight": 0},
                            "rnnt": {"type": "rnnt", "pred_hidden": 32, "joint_hidden": 32, "weight": 0.0}}}
    model, batch = _model_batch(cfg)
    out = model(batch)
    assert "loss_ctc" not in out and "loss_rnnt" not in out
    assert torch.allclose(out["loss"], 0.5 * out["loss_vad"])


# --------------------------------------------------------------------------- 2. forced alignment
@torch.no_grad()
def greedy_align_reference(asr, enc, enc_len, y, yl, act=None, max_per_frame: int = 4, slack: int = 3,
                           min_inside: float = 0.8):
    """The pre-fix implementation (heads/turn.py before the memory fix), verbatim."""
    B, T, _ = enc.shape
    U = y.shape[1]
    g, _ = asr.pred(asr.pred.prepend_sos(y))
    fe, gp, lin = asr.joint.enc(enc), asr.joint.pred(g), asr.joint.out[-1]
    emit = torch.full((B, U), -1, dtype=torch.long, device=enc.device)
    ok = torch.zeros(B, dtype=torch.bool)
    for b in range(B):
        Tb, Ub = int(enc_len[b]), int(yl[b])
        if Ub == 0:
            ok[b] = True
            continue
        h = torch.relu(fe[b, :Tb, None] + gp[b, None, : Ub + 1])
        blank = h[:, :Ub] @ lin.weight[asr.blank] + lin.bias[asr.blank]
        lab = (h[:, :Ub] * lin.weight[y[b, :Ub]][None]).sum(-1) + lin.bias[y[b, :Ub]]
        go = (lab > blank).tolist()
        t = u = per = 0
        e = []
        while t < Tb and u < Ub:
            if go[t][u] and per < max_per_frame:
                e.append(t)
                u, per = u + 1, per + 1
            else:
                t, per = t + 1, 0
        if u < Ub:
            continue
        emit[b, :Ub] = torch.tensor(e, device=enc.device)
        ok[b] = True
        if act is not None and (act[b, :Tb] > 0.5).any():
            sp = (act[b, :Tb] > 0.5).float()
            nz = torch.nonzero(sp)[:, 0]
            near = F.max_pool1d(sp[None, None], 2 * slack + 1, 1, slack)[0, 0] > 0
            inside = near[emit[b, :Ub]].float().mean()
            ok[b] = bool(e[0] >= int(nz[0]) - slack and e[-1] >= int(nz[-1]) - slack and inside >= min_inside)
    return emit, ok


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("budget", [None, 1, 4096])  # default, one token per chunk, a few tokens per chunk
def test_greedy_align_identical_to_reference(seed, budget):
    torch.manual_seed(seed)
    V, B, T = 12, 5, 40
    asr = RNNTHead(16, V, pred_hidden=16, joint_hidden=24).eval()
    with torch.no_grad():
        asr.joint.out[-1].bias[V] -= 1.0 + seed  # vary how often labels beat blank
    enc = torch.randn(B, T, 16)
    enc_len = torch.tensor([40, 31, 17, 25, 8])
    yl = torch.tensor([6, 11, 0, 25, 3])
    y = torch.randint(0, V, (B, int(yl.max())))
    act = (torch.rand(B, T) > 0.4).float()
    kw = {} if budget is None else {"max_bytes": budget}
    for a in (None, act):
        e_ref, ok_ref = greedy_align_reference(asr, enc, enc_len, y, yl, a)
        e_new, ok_new = greedy_align(asr, enc, enc_len, y, yl, a, **kw)
        assert torch.equal(e_ref, e_new) and torch.equal(ok_ref, ok_new)
    assert ok_ref.any() or seed  # the lattice walk actually places tokens for some item


def test_align_chunk_bounds():
    Dj = 640
    assert align_chunk(250, 150, Dj) == 150  # typical item fits the 256 MB budget in one chunk
    uc = align_chunk(250, 1626, Dj)  # the outlier AMI item (910 words, ~1.6k tokens)
    assert uc < 1626 and 2 * 250 * uc * Dj * 4 <= 256 * 2 ** 20
    assert align_chunk(250, 1626, Dj, max_bytes=1) == 1


# --------------------------------------------------------------------------- 3. arithmetic (info)
def _gib(n):
    return n / 2 ** 30


def test_peak_estimate_info(capsys):
    B, T, U, V, Dj, f32 = 8, 250, 150, 1025, 640, 4
    U1 = U + 1
    # RNNT loss joint per sub-batch (fused_batch_size fbs): enc+pred sum, ReLU, Dropout out (+ mask),
    # logits, log-softmax in the loss -> 4 Dj-sized + 2 V-sized temporaries (forward; recomputed in backward)
    def rnnt(fbs):
        return fbs * T * U1 * (4 * Dj + 2 * V) * f32
    # TurnHead greedy_align per item: h (T,U+1,Dj) + (h * W_y) product (T,U,Dj)
    align_before = T * U1 * Dj * f32 + T * U * Dj * f32
    align_after = 2 * T * align_chunk(T, U, Dj) * Dj * f32
    before = {"rnnt fbs=2": rnnt(2), "rnnt whole batch": rnnt(B), "greedy_align/item": align_before}
    after = {"rnnt (stage 1: weight 0, skipped)": 0, "greedy_align/item (<=256 MiB)": align_after}
    # the real OOM: one AMI turn window of 20 s (T=250) whose transcript is the whole turn (910 words ~ U 1626)
    oom = 2 * T * 1627 * Dj * f32
    with capsys.disabled():
        print(f"\n[mem] B={B} T={T} U={U} V={V} Dj={Dj} fp32")
        for k, v in before.items():
            print(f"[mem] before {k:34s} {_gib(v):6.3f} GiB")
        for k, v in after.items():
            print(f"[mem] after  {k:34s} {_gib(v):6.3f} GiB")
        print(f"[mem] OOM tensor: 2 x 250 x 1627 x 640 x 4 B = {_gib(oom):.3f} GiB (one Dj-sized joint temporary)")
        print(f"[mem] greedy_align on that item: before {_gib(2 * T * 1627 * Dj * f32):.3f} GiB, after "
              f"{_gib(2 * T * align_chunk(T, 1626, Dj) * Dj * f32):.3f} GiB")
    assert abs(_gib(oom) - 1.94) < 0.01
    assert max(after.values()) < min(before.values())


# --------------------------------------------------------------------------- 4. recipes
def test_stage_recipes_memory_keys():
    s1 = yaml.safe_load((ROOT / "recipes/stage1_heads_pretrained.yaml").read_text())
    s2 = yaml.safe_load((ROOT / "recipes/stage2_unfreeze_pretrained.yaml").read_text())
    assert s1["heads"]["rnnt"]["weight"] == 0 and s1["heads"]["ctc"]["weight"] == 0
    assert s1["heads"]["rnnt"]["fused_batch_size"] == 2
    # stage 2 inherits stage 1's head cfg via init.from, so the weights must be explicit
    assert s2["heads"]["rnnt"]["weight"] == 1.0 and s2["heads"]["ctc"]["weight"] == 0.1
    for s in (s1, s2):
        assert s["trainer"]["batch_size"] <= 6  # stage 1: 6; stage 2 lowered to 3 after MPS OOMs (unfrozen encoder)
        for src in s["data"]["mix"]:
            if "ami" in src:
                assert src["ami"]["window_sec"] == 16
