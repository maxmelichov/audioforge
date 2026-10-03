"""Speaker-aware end-of-turn: conversation generator (v2: FastMSS moves, disfluencies), TurnHead (v2: text state),
eot-bench metric, recipe smoke runs (4-way ablation)."""
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from audioforge.conversation import (
    MOVES,
    PRESETS,
    ConversationCollate,
    TurnTakingConfig,
    baselines,
    conversation,
    conversation_dataset,
    eot_bench,
    event_stats,
    silence_scores,
)
from audioforge.data import Collate, ToneLanguage
from audioforge.heads.asr import RNNTHead
from audioforge.heads.turn import TurnHead, eot_targets, greedy_align, primary_column, token_counts, uniform_align
from audioforge.train import load_model, run_recipe

RECIPE = Path(__file__).parent.parent / "recipes" / "speaker_aware_turn.yaml"


def test_conversation_labels():
    lang = ToneLanguage(seed=0)
    rng = random.Random(3)
    for _ in range(20):
        c = conversation(lang, rng)
        T = ToneLanguage.n_frames(len(c["audio"]))
        assert c["spk_targets"].shape == (T, 4) and c["eot"].shape == (T,) and c["primary_act"].shape == (T,)
        end, on = c["turn_end_frame"], c["onset_frame"]
        assert 0 <= on < end < T and c["primary_act"][end:].sum() == 0 and c["primary_act"][end - 1] == 1
        assert c["eot"][:end].sum() == 0 and c["eot"][end:].min() == 1
        # the label is recoverable from the activity alone (so the stock Collate is enough)
        assert torch.equal(eot_targets(torch.tensor(c["spk_act"])[None])[0], torch.tensor(c["eot"]))
        assert c["text"] and set(c["text"]) <= set(" " + "".join(lang.LEXICON))
    st = event_stats(conversation_dataset(200, seed=1))
    assert 0.5 < st["with_hesitation"] < 0.8 and st["with_bystander"] > 0.2 and st["with_backchannel"] > 0.2


def test_collate_keys():
    convs = conversation_dataset(3, seed=2)
    b = ConversationCollate(None)(convs)
    assert b["eot"].shape == b["spk_act"].shape == b["spk_targets"].shape[:2]
    assert b["turn_end_frame"].tolist() == [c["turn_end_frame"] for c in convs]
    b2 = Collate(None)(convs)
    assert torch.equal(eot_targets(b2["spk_act"], b2["spk_act_len"]) * (
        torch.arange(b["eot"].shape[1])[None] < b["eot_len"][:, None]), b["eot"])


def test_primary_column():
    t = torch.zeros(1, 10, 4)
    t[0, 5:, 0] = 1  # primary arrives second
    t[0, 2:4, 1] = 1
    assert primary_column(t).tolist() == [1]


@pytest.mark.parametrize("mode", ["none", "concat", "kernel", "kernel+concat"])
def test_turn_head_shapes_batch_invariance_causality(mode):
    torch.manual_seed(0)
    head = TurnHead(32, mode=mode, hidden=24, history=4, dropout=0.0).eval()
    lens = [17, 30, 9]
    enc = torch.randn(3, 30, 32)
    act = (torch.rand(3, 30) > 0.5).float()
    need = "concat" in mode
    z = head(enc, torch.tensor(lens), act if need else None)
    assert z.shape == (3, 30)
    for i, n in enumerate(lens):  # padded batch == alone, valid frames
        one = head(enc[i:i + 1, :n], torch.tensor([n]), act[i:i + 1, :n] if need else None)
        assert torch.allclose(one[0], z[i, :n], atol=1e-5)
    enc2, act2 = enc.clone(), act.clone()
    enc2[:, 12:] += 1.0
    act2[:, 12:] = 1 - act2[:, 12:]
    z2 = head(enc2, torch.tensor(lens), act2 if need else None)
    assert torch.allclose(z[:, :12], z2[:, :12], atol=1e-6)  # causal: no future leakage
    batch = {"spk_act": act, "spk_act_len": torch.tensor(lens)}
    l = head.loss(enc, torch.tensor(lens), batch)
    assert torch.isfinite(l) and l > 0
    p = head.decode(enc, torch.tensor(lens), spk_act=act)
    assert p.min() >= 0 and p.max() <= 1
    if need:
        assert head.decode(enc, torch.tensor(lens)) is None


def test_eot_bench_hand_made():
    # three conversations, turn ends at frame 5 (onset 0)
    A = [0.1, 0.3, 0.2, 0.1, 0.1, 0.2, 0.6, 0.9, 0.9]
    B = [0.1, 0.7, 0.1, 0.2, 0.1, 0.8, 0.9, 0.9, 0.9]  # 0.7 spike during a hesitation
    C = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.95]
    sc = [np.array(x) for x in (A, B, C)]
    r = eot_bench(sc, [0] * 3, [5] * 3, frame_ms=80, thresholds=[0.5, 0.75], max_fc=0.4, fixed_latency_ms=200)
    p = r["at_max_fc"]
    # θ=0.5: B is a false cutoff; A fires at frame 6 -> (6+1-5)*80 = 160 ms; C at frame 8 -> 320 ms
    assert p["threshold"] == 0.5 and p["fc_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert p["p50_ms"] == 160 and p["p90_ms"] == 320
    r = eot_bench(sc, [0] * 3, [5] * 3, frame_ms=80, thresholds=[0.5, 0.75], max_fc=0.05)
    p = r["at_max_fc"]  # θ=0.75: no cutoffs; latencies A 240, B 80, C 320
    assert p["threshold"] == 0.75 and p["fc_rate"] == 0 and p["p50_ms"] == 240 and p["p90_ms"] == 320
    # chunked look-ahead: frame 6 is emitted at the end of chunk [6,7] -> (8-5)*80
    r = eot_bench([sc[0]], [0], [5], frame_ms=80, thresholds=[0.5], chunk=2)
    assert r["at_max_fc"]["p50_ms"] == 240
    # never firing after the end = miss (inf latency)
    r = eot_bench([np.zeros(9)], [0], [5], thresholds=[0.5])
    assert r["at_max_fc"]["p50_ms"] == float("inf") and r["at_max_fc"]["miss_rate"] == 1.0


def test_silence_timeout_baseline():
    act = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], np.float32)  # 2-frame hesitation, end at 8
    s = silence_scores(act, onset=2)
    assert s.tolist() == [0, 0, 0, 0, 1, 2, 0, 0, 1, 2, 3, 4, 5, 6]
    r = eot_bench([s], [2], [8], frame_ms=80, max_fc=0.0)
    # the shortest timeout that does not fire in the hesitation is 3 frames -> 240 ms of dead air
    assert r["at_max_fc"]["p50_ms"] == 240 and r["at_max_fc"]["fc_rate"] == 0
    convs = conversation_dataset(40, seed=4)
    b = baselines(convs)
    assert b["timeout_primary_oracle"]["at_max_fc"]["fc_rate"] <= 0.05


@pytest.mark.parametrize("mode", ["concat"])
def test_recipe_smoke(mode, tmp_path):
    model, metrics = run_recipe(str(RECIPE), _smoke_overrides(mode, False), out=str(tmp_path / "m.afm"))
    assert "eot_turn_p50_ms@5fc" in metrics and "eot_timeout_p50_ms@5fc" in metrics and "der_diar" in metrics


# --------------------------------------------------------------------------- v2: text state
def _text_head(mode="none", seed=0):
    torch.manual_seed(seed)
    asr = RNNTHead(32, vocab_size=12, pred_hidden=16, joint_hidden=16).eval()
    head = TurnHead(32, mode=mode, hidden=24, history=4, dropout=0.0, use_text=True, k_tokens=4, text_dim=8).eval()
    head.bind_asr(asr)
    return head, asr


def test_alignment_shapes_and_counts():
    head, asr = _text_head()
    B, T = 3, 20
    enc, elen = torch.randn(B, T, 32), torch.tensor([20, 14, 9])
    y, yl = torch.randint(0, 12, (B, 6)), torch.tensor([6, 3, 0])
    act = torch.zeros(B, T)
    act[0, 2:12] = 1
    act[1, 5:8] = 1
    act[2, 1:3] = 1
    uni = uniform_align(yl, T, elen, act, U=6)
    assert uni.shape == (B, 6) and uni[0].tolist() == [3, 5, 6, 8, 10, 11]  # ends on the last speech frame
    assert uni[1, :3].tolist() == [5, 6, 7] and (uni[2] == -1).all()
    n = token_counts(uni, yl, T)
    assert n.shape == (B, T) and n[0, -1] == 6 and n[1, -1] == 3 and n[2].max() == 0
    assert (n[:, 1:] >= n[:, :-1]).all()  # non-decreasing
    emit, ok = greedy_align(asr, enc, elen, y, yl, max_per_frame=2)
    assert emit.shape == (B, 6) and ok.shape == (B,)
    for b in range(B):  # placed tokens are monotonic, inside the valid frames, <= 2 per frame
        e = emit[b, : int(yl[b])]
        if ok[b] and len(e):
            assert (e[1:] >= e[:-1]).all() and int(e.max()) < int(elen[b])
            assert max(e.tolist().count(t) for t in set(e.tolist())) <= 2
    # a planted alignment is recovered: joint = identity on one-hot frames (token y_u at its frame, blank elsewhere)
    planted = RNNTHead(13, vocab_size=12, pred_hidden=8, joint_hidden=13).eval()
    with torch.no_grad():
        planted.joint.enc.weight.copy_(torch.eye(13))
        planted.joint.enc.bias.zero_()
        planted.joint.pred.weight.zero_()
        planted.joint.pred.bias.zero_()
        planted.joint.out[-1].weight.copy_(10 * torch.eye(13))
        planted.joint.out[-1].bias.zero_()
    yp, fp = torch.tensor([[3, 5, 7, 2]]), [2, 4, 5, 9]
    ep = F.one_hot(torch.full((1, 12), 12), 13).float()
    ep[0, fp] = F.one_hot(yp[0], 13).float()
    ap = torch.zeros(1, 12)
    ap[0, 2:10] = 1
    e, ok = greedy_align(planted, ep, torch.tensor([12]), yp, torch.tensor([4]), ap)
    assert e[0].tolist() == fp and ok.tolist() == [True]
    late = torch.zeros(1, 12)
    late[0, 7:12] = 1  # speech starts long after the first emission
    e, ok = greedy_align(planted, ep, torch.tensor([12]), yp, torch.tensor([4]), late)
    assert ok.tolist() == [False]  # implausible w.r.t. the speech frames -> caller falls back to uniform
    em = head.align(enc, elen, y, yl, act)  # greedy with uniform fallback: every token placed
    assert all((em[b, : int(yl[b])] >= 0).all() for b in range(B))
    assert sum(head.align_counts.values()) == B


@pytest.mark.parametrize("mode", ["none", "concat"])
def test_text_state_causality_and_streaming(mode):
    head, asr = _text_head(mode)
    B, T, U = 2, 16, 5
    enc, elen = torch.randn(B, T, 32), torch.tensor([16, 16])
    act = (torch.rand(B, T) > 0.4).float()
    y, yl = torch.randint(0, 12, (B, U)), torch.tensor([U, U])
    emit = torch.tensor([[1, 3, 6, 9, 12], [0, 2, 4, 10, 11]])
    n = token_counts(emit, yl, T)
    sa = act if head.concat else None
    z = head(enc, elen, sa, text=(y, n))
    assert z.shape == (B, T)
    # changing a future token (index 3, emitted at frames 9 / 10) must not change earlier frames
    y2 = y.clone()
    y2[:, 3] = (y2[:, 3] + 1) % 12
    z2 = head(enc, elen, sa, text=(y2, n))
    assert torch.allclose(z[0, :9], z2[0, :9], atol=1e-6) and torch.allclose(z[1, :10], z2[1, :10], atol=1e-6)
    assert not torch.allclose(z[0, 9:], z2[0, 9:])
    # same for a greedy re-alignment: frames before the changed token's emission are unchanged
    e1, _ = greedy_align(asr, enc, elen, y, yl)
    e2, _ = greedy_align(asr, enc, elen, y2, yl)
    assert torch.equal(e1[:, :3], e2[:, :3])
    # frame-by-frame step() with the prefix known at each frame == offline forward
    g, _ = asr.pred(asr.pred.prepend_sos(y))
    st = head.init_stream(B)
    probs = []
    for t in range(T):
        nt = n[:, t]
        gt = torch.stack([g[b, nt[b]] for b in range(B)])
        prefix = [y[b, : nt[b]].tolist() for b in range(B)]
        probs.append(head.step(enc[:, t:t + 1], (gt[:, None], None), prefix, act[:, t:t + 1] if head.concat else None,
                               state=st))
    assert torch.allclose(torch.cat(probs, 1), z.sigmoid(), atol=1e-5)
    # decode needs the text; loss runs on a batch with text (alignment inside)
    assert head.decode(enc, elen, spk_act=act) is None
    l = head.loss(enc, elen, {"spk_act": act, "spk_act_len": elen, "text": y, "text_len": yl})
    assert torch.isfinite(l) and l > 0


def test_text_branch_no_grad_into_asr():
    head, asr = _text_head()
    asr.train()
    head.train()
    enc = torch.randn(2, 12, 32)
    act = torch.zeros(2, 12)
    act[:, 2:8] = 1
    b = {"spk_act": act, "text": torch.randint(0, 12, (2, 4)), "text_len": torch.tensor([4, 2])}
    head.loss(enc, torch.tensor([12, 12]), b).backward()
    assert all(p.grad is None for p in asr.parameters())
    assert head.text_proj[0].weight.grad is not None
    assert "_asr" not in dict(head.named_modules()) and not any(k.startswith("_asr") for k in head.state_dict())


# --------------------------------------------------------------------------- v2: generator
def test_presets_turn_taking_statistics():
    assert set(PRESETS) >= {"default", "asr", "diar"}
    assert TurnTakingConfig.preset("asr").p_moves == pytest.approx((0.09, 0.13, 0.54, 0.24), abs=0.005)
    with pytest.raises(ValueError):
        TurnTakingConfig.preset("nope")
    P = TurnTakingConfig.preset("default").matrix()
    assert P.shape == (4, 4) and np.allclose(P.sum(1), 1)
    st = {p: event_stats(list(conversation_dataset(200, seed=1, preset=p))) for p in ("asr", "default", "diar")}
    assert st["asr"]["overlap_fraction"] > st["default"]["overlap_fraction"] > st["diar"]["overlap_fraction"] > 0
    for p, s in st.items():  # the chain draws its moves with the promised frequencies
        want = TurnTakingConfig.preset(p).p_moves
        assert all(abs(s["moves_drawn"][z] - w) < 0.06 for z, w in zip(MOVES, want)), (p, s["moves_drawn"])
        assert 0.55 < s["with_hesitation"] < 0.75 and s["with_backchannel"] > 0.2 and s["with_bystander"] > 0.3
    ov = [s["moves_drawn"]["IR"] + s["moves_drawn"]["BC"] for s in st.values()]
    assert ov[0] > ov[1] > ov[2]


def test_no_self_overlap_and_move_semantics():
    lang = ToneLanguage(seed=0)
    rng = random.Random(5)
    for _ in range(40):
        c = conversation(lang, rng, preset="asr")
        ev = c["events"]
        assert set(ev["moves"]) <= set(MOVES) and len(ev["moves_drawn"]) == len(ev["moves"])
        assert ev["self_overlaps"] == 0
        if ev["n_others"] == 0:
            assert set(ev["moves"]) <= {"TH"} and c["spk_targets"][:, 1:].sum() == 0
        assert ev["backchannels"] == ev["moves"].count("BC")
    # a primary with the whole floor: only holds, exponential pauses within the clip range
    cfg = TurnTakingConfig.preset("default", p_others=(1.0,), p_hes=(0, 0, 1.0))
    cs = [conversation(lang, rng, cfg) for _ in range(30)]
    ps = [p for c in cs for p in c["events"]["th_pauses"]]
    assert len(ps) == 60 and min(ps) >= cfg.pause_clip[0] and max(ps) <= cfg.pause_clip[1]
    assert 0.35 < np.mean(ps) < 0.95  # Exp(1.6): mean 0.625 s


def test_disfluency_rates():
    kw = dict(p_filler=0.5, p_cutoff=0.3, p_repeat=0.1, p_correct=0.08, p_decoy=0.0)
    st = event_stats(list(conversation_dataset(600, seed=3, **kw)))
    assert abs(st["hesitation_with_filler"] - 0.5) < 0.06 and abs(st["hesitation_cutoff"] - 0.3) < 0.06
    assert abs(st["repetition_per_word"] - 0.1) < 0.025 and abs(st["correction_per_word"] - 0.08) < 0.025
    assert st["hesitation_lowered_decoy"] == 0
    off = event_stats(list(conversation_dataset(100, seed=3, p_filler=0, p_cutoff=0, p_repeat=0, p_correct=0)))
    assert off["hesitation_with_filler"] == off["hesitation_cutoff"] == off["repetition_per_word"] == 0
    lang = ToneLanguage(seed=0)
    rng = random.Random(1)
    c = conversation(lang, rng, p_hes=(0, 1.0), p_cutoff=1.0, p_filler=1.0, p_repeat=0, p_correct=0)
    w = c["text"].split()
    # verbatim: <cut word> e <restart of the same word> ...; the clean text says every word once
    i = w.index("e")
    assert c["text_clean"].split()[i - 1] == w[i + 1] and w[i + 1].startswith(w[i - 1]) and w[i - 1] != w[i + 1]
    assert len(c["text_clean"].split()) == len(w) - 2


def test_conversation_dataset_backward_compatible():
    a, b = conversation_dataset(3, seed=2), conversation_dataset(3, 2, None, "default")
    assert np.array_equal(a[1]["audio"], b[1]["audio"]) and a[1]["text"] == b[1]["text"]
    assert conversation_dataset(3, seed=2, preset="asr").cfg.p_moves != a.cfg.p_moves


def _smoke_overrides(mode, use_text):
    return ["trainer.max_steps=3", "trainer.batch_size=4", "trainer.device=cpu", "data.synthetic.n_train=8",
            "data.synthetic.n_val=4", "encoder.n_layers=2", "encoder.d_model=64", "encoder.subsampling_channels=16",
            f"heads.turn.mode={mode}", f"heads.turn.condition_on_speaker={str(mode == 'kernel').lower()}",
            f"heads.turn.use_text={str(use_text).lower()}"]


@pytest.mark.parametrize("mode,use_text", [("none", False), ("kernel", False), ("none", True), ("kernel", True)],
                         ids=["acoustic", "speaker", "text", "speaker+text"])
def test_recipe_smoke_ablation(mode, use_text, tmp_path):
    model, metrics = run_recipe(str(RECIPE), _smoke_overrides(mode, use_text), out=str(tmp_path / "m.afm"))
    assert "eot_turn_p50_ms@5fc" in metrics and "eot_timeout_p50_ms@5fc" in metrics and "der_diar" in metrics
    assert model.heads["turn"].use_text == use_text and (model.heads["turn"].text_proj is not None) == use_text
    loaded = load_model(tmp_path / "m.afm")
    assert all(torch.equal(v.cpu(), loaded.state_dict()[k]) for k, v in model.state_dict().items())
    if use_text:
        assert loaded.heads["turn"]._asr is loaded.heads["tdt"]
