"""Layer routing for the diar / turn heads (research/LAYER_ROUTING.md): init.from with a re-sized layer mix, weight-0
speaker-conditioned heads skip the conditioned pass, from_layers on a condition_on_speaker head reads the CONDITIONED
encoder pass (training, turn_scores, eval_stage1.turn_scores_given_act), and the driver's diarization scoring."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from audioforge.heads.turn import turn_scores
from audioforge.model import SpeechModel

sys.path.insert(0, str(Path(__file__).parent))
from test_turn_v3 import V3, _model, _tiny_stage1_afm  # noqa: E402

ROOT = Path(__file__).parent.parent


def _script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "research" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recipe_init(tmp_path, diar_layers):
    from audioforge.train import build_model
    init = tmp_path / "init.afm"
    if not init.exists():
        _tiny_stage1_afm(init)
    recipe = {"name": "t", "init": {"from": str(init), "pretrained_lr_mult": 0, "train_only": ["heads.diar", "layer_mix.diar"]},
              "heads": {"diar": {"from_layers": diar_layers}},
              "trainer": {"max_steps": 1, "batch_size": 2, "device": "cpu"}, "data": {}}
    return init, build_model(recipe, [])[0]


def test_init_from_resized_and_dropped_layer_mix(tmp_path):
    from audioforge.train import load_model
    init, sub = _recipe_init(tmp_path, [0, 1])  # 3-layer stand-in: all -> 2-block sub-mix (fresh, uniform)
    src = load_model(init).state_dict()
    assert sub.layer_mix["diar"].shape == (2,) and torch.equal(sub.layer_mix["diar"].detach(), torch.zeros(2))
    w = sub.layer_weights("diar")
    assert torch.allclose(w, torch.tensor([0.5, 0.5, 0.0]))
    sd = sub.state_dict()
    for k in src:  # every other tensor loaded unchanged, including the diar head itself (warm start)
        if k != "layer_mix.diar":
            assert torch.equal(src[k], sd[k]), k
    _, tap = _recipe_init(tmp_path, [2])  # single tap: no mix tensor at all
    assert "diar" not in tap.layer_mix and torch.equal(tap.layer_weights("diar"), torch.tensor([0.0, 0.0, 1.0]))
    _, ctl = _recipe_init(tmp_path, "all")  # the control keeps the source's mix
    assert torch.equal(ctl.layer_mix["diar"].detach(), src["layer_mix.diar"])


def test_weight0_conditioned_head_skips_the_conditioned_pass():
    model, b = _model()
    cond = [k for k, v in model.head_cfg.items() if v.get("condition_on_speaker")]
    assert "turn" in cond
    for k in cond:  # the stand-in's ASR head is speaker-conditioned too
        model.head_cfg[k]["weight"] = 0
    calls = []
    orig = model.encoder.forward

    def spy(*a, **kw):
        calls.append(kw.get("spk_act") is not None)
        return orig(*a, **kw)
    model.encoder.forward = spy
    model.train()
    out = model(b)
    assert calls == [False] and not any(f"loss_{k}" in out for k in cond)


def _tapped(layers):
    model, b = _model(turn=V3)
    if layers is not None:
        cfg = {**model.cfg, "heads": {**model.cfg["heads"], "turn": {**model.cfg["heads"]["turn"], "from_layers": layers}}}
        m2 = SpeechModel(cfg, model.tokenizer)
        m2.load_state_dict(model.state_dict(), strict=False)
        model = m2
    return model.eval(), b


def test_cond_head_input_top_is_unchanged_and_mix_reads_the_conditioned_pass():
    model, b = _tapped(None)
    act = b["spk_act"].float()
    with torch.no_grad():
        ref = model.encode(b["audio"], b["audio_len"], spk_act=act)[0]
        assert torch.equal(model.cond_head_input("turn", b["audio"], b["audio_len"], act), ref)
    mixm, _ = _tapped([0, 1])
    with torch.no_grad():
        mixm.layer_mix["turn"].copy_(torch.tensor([0.3, -0.2]))
        enc, _, hid = mixm.encode(b["audio"], b["audio_len"], spk_act=act, return_hidden=True)
        w = torch.tensor([0.3, -0.2]).softmax(0)
        got = mixm.cond_head_input("turn", b["audio"], b["audio_len"], act)
        assert torch.allclose(got, w[0] * hid[0] + w[1] * hid[1], atol=1e-6)
        unc = mixm.encode(b["audio"], b["audio_len"], return_hidden=True)[2]
        assert not torch.allclose(got, w[0] * unc[0] + w[1] * unc[1])  # the kernels act on the tapped blocks
        assert torch.allclose(mixm.layer_weights("turn"), w)


def test_forward_and_scores_use_the_tap():
    """Training loss and turn_scores change when the turn head reads blocks 1-2 instead of the top of the 2-layer
    stand-in, and a single tap of the TOP block reproduces the untapped model exactly."""
    top, b = _tapped(None)
    last, _ = _tapped([1])  # = the top block (2 layers): identical to reading enc
    mix, _ = _tapped([0, 1])
    torch.manual_seed(0)
    lt = top(b)["loss_turn"]
    torch.manual_seed(0)
    ll = last(b)["loss_turn"]
    torch.manual_seed(0)
    lm = mix(b)["loss_turn"]
    assert torch.allclose(lt, ll) and not torch.allclose(lt, lm)
    convs = [{k: b[k][i, : int(b["audio_len"][i])] if k == "audio" else b[k][i] for k in ("audio", "spk_act", "spk_targets")}
             for i in range(2)]
    convs = [{k: np.asarray(v) for k, v in c.items()} for c in convs]
    s_top, s_last, s_mix = (turn_scores(m, "turn", convs) for m in (top, last, mix))
    assert all(np.allclose(a, c) for a, c in zip(s_top, s_last))
    assert not all(np.allclose(a, c) for a, c in zip(s_top, s_mix))
    ev = _script("eval_stage1")
    g = ev.turn_scores_given_act(mix, "turn", convs, [c["spk_act"] for c in convs], 2,
                                 cols=[c["spk_targets"] for c in convs], prims=[0, 0])
    assert all(np.allclose(a, c, atol=1e-5) for a, c in zip(g, s_mix))


def test_score_diar_perfect_and_half():
    lr = _script("layer_routing")
    rng = np.random.default_rng(0)
    refs = [(rng.random((50, 4)) > 0.7).astype(np.float32) for _ in range(3)]
    val = [{"spk_targets": r} for r in refs]
    r = lr.score_diar([x.copy() for x in refs], val)
    assert r["frame_der"]["der_pooled"] == 0 and r["pyannote"]["collar_0.0"]["der"] == 0
    r = lr.score_diar([np.zeros_like(x) for x in refs], val)  # all silence = all miss
    assert r["frame_der"]["der_pooled"] == 1.0 and r["frame_der"]["miss"] == 1.0


def test_recipes_parse():
    for name in ("diar_layer_route", "turn_layer_route_mid"):
        cfg = yaml.safe_load((ROOT / "research" / "recipes" / f"{name}.yaml").read_text())
        assert cfg["init"]["train_only"]
    t = yaml.safe_load((ROOT / "research" / "recipes" / "turn_layer_route_mid.yaml").read_text())
    assert t["heads"]["turn"]["from_layers"] == list(range(3, 12))
    trail6 = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_v3_trail6.yaml").read_text())
    same = {k: v for k, v in t["heads"]["turn"].items() if k != "from_layers"}
    assert same == trail6["heads"]["turn"] and t["data"] == trail6["data"]
    assert t["trainer"]["conditioning"] == trail6["trainer"]["conditioning"]
    assert t["trainer"]["max_steps"] == trail6["trainer"]["max_steps"]
