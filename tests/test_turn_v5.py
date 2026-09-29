"""TurnHead v5: the diarizer track reaches ONLY the head; the encoder is never speaker-conditioned
(research/TURN_ERRORS.md: kernel input clean + head inputs from the Sortformer track -> 9.5 % misses vs 62 % with the
track in the kernels; v4 retraining the kernels on noisy tracks did not help).

Under test: SpeechModel.forward runs the train-time conditioning (p_ext / ext_noise / flip) also when no head is
speaker-conditioned but a head reads the track itself (head-only path), with a single encoder pass; recipes with a
condition_on_speaker head are bit-identical; TurnHead(init_partial) warm-starts the reshaped head from the stage-1
checkpoint; the eval paths feed the override / oracle columns to this head.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from audioforge.conversation import conversation_dataset
from audioforge.data import Collate
from audioforge.datasets import ext_tracks as xt
from audioforge.heads.turn import turn_scores
from audioforge.model import SpeechModel

sys.path.insert(0, str(Path(__file__).parent))
from test_cond import _cfg, _spy_turn_batch  # noqa: E402
from test_turn_v3 import V3, _perturbed, _script, _tiny_stage1_afm, _with_ext  # noqa: E402
from test_turn_v3 import _model as _v3_model

ROOT = Path(__file__).parent.parent
RECIPE = ROOT / "research" / "recipes" / "stage1_turn_v5_headonly.yaml"
NOISE = {"p": 1.0, "swap_max": 12, "drop": 1.0, "drop_max": 8, "at_turn_end": 0.6}

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def ev():
    return _script("eval_stage1")


def _v5_model(conditioning=None, turn=None):
    """test_cond's tiny model with NO speaker-conditioned head: TDT / CTC plain, turn head concat + v3 inputs."""
    from audioforge.tokenizer import train_tokenizer
    convs = conversation_dataset(4, seed=9)
    tok = train_tokenizer("char", [c["text"] for c in convs])
    cfg = _cfg(conditioning, turn_mode="concat")
    for k in ("tdt", "ctc"):
        cfg["heads"][k]["condition_on_speaker"] = False
    cfg["heads"]["turn"].update(dict(V3, condition_on_speaker=False), **(turn or {}))
    torch.manual_seed(0)
    return _perturbed(SpeechModel(cfg, tok)), Collate(tok)(convs)


def _count_encoder(model):
    calls = []
    orig = model.encoder.forward

    def spy(feats, flen, att=None, spk_act=None, **k):
        calls.append(spk_act is not None)
        return orig(feats, flen, att, spk_act=spk_act, **k)

    model.encoder.forward = spy
    return calls


# --------------------------------------------------------------------------- head-only conditioning
def test_head_only_conditioning_feeds_the_head():
    model, b = _v5_model({"p_ext": 1.0, "flip": 0.01, "ext_noise": NOISE})
    assert model.head_reads_act and model.cond_on and model.n_act_columns == 4
    assert not any(hc.get("condition_on_speaker") for hc in model.head_cfg.values())
    model.train()
    b, prim = _with_ext(b, missing=(1,))
    batches, calls = [], _count_encoder(model)
    _spy_turn_batch(model, batches)
    torch.manual_seed(3)
    out = model(b)
    assert torch.isfinite(out["loss"])
    assert calls == [False]  # one encoder pass, never speaker-conditioned
    tb = batches[0]
    act, cols, pr = tb["spk_act"], tb["spk_cols"], tb["spk_prim"]
    B, T, S = cols.shape
    assert torch.equal(tb["spk_act_oracle"], b["spk_act"])  # labels stay clean
    assert not torch.equal(act[:, :b["spk_act"].shape[1]], b["spk_act"][:, :T].float())  # the head sees the track
    assert torch.allclose(cols.gather(2, pr[:, None, None].expand(B, T, 1))[..., 0], act)
    ext_items = [i for i in range(B) if i != 1]
    assert all(int(pr[i]) == int(prim[i]) for i in ext_items)  # the ext track's enrollment column
    L = [min(int(b["spk_act_len"][i]), T) for i in range(B)]
    changed = [not torch.allclose(cols[i, :L[i]], b["spk_targets_ext"][i, :L[i]]) for i in ext_items]
    assert any(changed)  # ext_noise p = 1 corrupted the ext items' columns
    # item 1 has no cached track -> the oracle columns, permuted
    orc = b["spk_targets"][1, :T].float()
    got = sorted(cols[1, :L[1], k].tolist() for k in range(S) if k != int(pr[1]))
    assert got == sorted(orc[:L[1], k].tolist() for k in range(1, orc.shape[1]))
    # the head consumes these columns (batch_columns), and they reach its gradient
    assert model.heads["turn"].batch_columns(tb)[0] is cols
    out["loss"].backward()
    assert model.heads["turn"].v3_in.weight.grad is not None

    # no noise: the ext items' columns are the cached track as is
    m0, b0 = _v5_model({"p_ext": 1.0})
    m0.train()
    b0, _ = _with_ext(b0, missing=(1,))
    bb = []
    _spy_turn_batch(m0, bb)
    m0(b0)
    c0 = bb[0]["spk_cols"]
    assert all(torch.allclose(c0[i, :L[i]], b0["spk_targets_ext"][i, :L[i]]) for i in ext_items)


def test_single_encoder_pass_vs_kernel_recipe():
    """v5: one encoder call per training step (the text alignment reuses enc). The v3 kernel head: two."""
    model, b = _v5_model({"p_ext": 1.0, "ext_noise": NOISE})
    model.train()
    b, _ = _with_ext(b)
    calls = _count_encoder(model)
    model(b)
    assert calls == [False]
    k, kb = _v3_model({"p_ext": 1.0, "ext_noise": NOISE}, turn=V3)
    k.train()
    kb, _ = _with_ext(kb)
    kc = _count_encoder(k)
    k(kb)
    assert kc == [False, True]


def test_eval_forward_untouched():
    model, b = _v5_model({"p_ext": 1.0, "ext_noise": NOISE})
    b, _ = _with_ext(b)
    model.eval()
    batches = []
    _spy_turn_batch(model, batches)
    with torch.no_grad():
        model(b)
    assert "spk_cols" not in batches[0] and "spk_act_oracle" not in batches[0]


# --------------------------------------------------------------------------- bit identity
def _loss(model, b, seed=5):
    torch.manual_seed(seed), random.seed(seed)
    return model(b, att_context_size=model.encoder.att_context_size)["loss"]


@pytest.mark.parametrize("arm", ["v3_kernel_ext", "concat_turn_with_conditioned_tdt", "cond_off_concat",
                                 "cond_on_mode_none"])
def test_existing_configs_bit_identical(arm, monkeypatch):
    """The new trigger (head_reads_act) never changes a model that already conditioned (a condition_on_speaker head)
    or had conditioning off / no head reading the track: same loss as with the trigger disabled."""
    if arm == "v3_kernel_ext":
        model, b = _v3_model({"p_ext": 0.9, "flip": 0.01, "ext_noise": NOISE}, turn=V3)
        b, _ = _with_ext(b, missing=(2,))
    elif arm == "concat_turn_with_conditioned_tdt":  # speaker_aware_turn "+speaker concat" arm
        model, b = _v3_model({"p_diar": 0.5, "flip": 0.05, "jitter": 2}, turn_mode="concat")
    elif arm == "cond_off_concat":
        model, b = _v5_model(None)
    else:
        model, b = _v5_model({"flip": 0.05}, turn={"mode": "none", "act_columns": 1, "duration_feats": False})
    model.train()
    new = _loss(model, b)
    monkeypatch.setattr(SpeechModel, "head_reads_act", property(lambda self: False))
    ref = _loss(model, b)
    assert torch.equal(new, ref)


def test_every_existing_recipe_already_conditions_through_a_kernel_head():
    """Recipes with a conditioning block all have a condition_on_speaker head (so the head-only branch cannot change
    them); only v5 relies on it."""
    for p in sorted((ROOT / "research" / "recipes").glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text())
        cond = cfg.get("conditioning") or (cfg.get("trainer") or {}).get("conditioning")
        if not cond or p == RECIPE:
            continue
        assert any((h or {}).get("condition_on_speaker") for h in cfg.get("heads", {}).values()), p.name


# --------------------------------------------------------------------------- recipe + warm start
def test_recipe_v5_yaml():
    cfg = yaml.safe_load(RECIPE.read_text())
    v3 = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_v3.yaml").read_text())
    assert cfg["init"] == v3["init"] and cfg["encoder"] == v3["encoder"] and cfg["data"] == v3["data"]
    t = cfg["heads"]["turn"]
    assert t["mode"] == "concat" and t["condition_on_speaker"] is False and t["act_columns"] == 4
    assert t["duration_feats"] is True and t["use_text"] is True and t["hidden"] == 128 and t["history"] == 8
    assert t["future_act_aux"] == {"horizons": [6, 12, 25], "weight": 0.3} and t["init_partial"] is True
    tr = cfg["trainer"]
    assert (tr["max_steps"], tr["batch_size"], tr["lr"], tr["checkpoint_every"]) == (2000, 6, 1e-3, 500)
    assert tr["wer_gate"] == v3["trainer"]["wer_gate"]
    assert tr["conditioning"] == {"p_ext": 0.9, "p_diar": 0, "flip": 0.01,
                                  "ext_noise": {"p": 0.4, "swap_max": 12, "drop": 0.3, "drop_max": 8,
                                                "at_turn_end": 0.6}}


def test_init_from_accepts_v5_without_kernels(tmp_path):
    from audioforge.train import init_from_afm, load_model
    init = tmp_path / "init.afm"
    _tiny_stage1_afm(init)
    cfg = yaml.safe_load(RECIPE.read_text())
    model, new_cfg = init_from_afm(cfg, str(init))
    hc, head = new_cfg["heads"]["turn"], model.heads["turn"]
    assert hc["condition_on_speaker"] is False and head.mode == "concat" and head.init_partial
    assert not any(v.get("condition_on_speaker") for v in new_cfg["heads"].values())
    assert model.head_reads_act and model.cond["p_ext"] == 0.9
    src = load_model(init).state_dict()
    sd = model.state_dict()
    for k in ("heads.turn.text_proj.0.weight", "encoder.speaker_kernels.0.weight"):  # same shape: loaded
        if k in src:
            assert torch.equal(sd[k], src[k])
    assert sd["heads.turn.inp.0.weight"].shape != src["heads.turn.inp.0.weight"].shape  # reshaped: fresh
    # without init_partial the reshaped head cannot load
    bad = yaml.safe_load(RECIPE.read_text())
    bad["heads"]["turn"].pop("init_partial")
    with pytest.raises(RuntimeError):
        init_from_afm(bad, str(init))


@pytest.mark.skipif(not xt.has_tracks(split="train", source="stream"), reason="AMI train streaming tracks not cached")
def test_recipe_v5_smoke_on_ami(tmp_path, monkeypatch):
    """3 steps of research/recipes/stage1_turn_v5_headonly.yaml on CPU: tiny stand-in init model, 8 AMI turn items with their
    cached streaming Sortformer tracks; only heads.turn trains; the encoder is never speaker-conditioned."""
    from audioforge.modules.fastconformer import FastConformerEncoder
    from audioforge.train import load_model, run_recipe
    cfg = yaml.safe_load(RECIPE.read_text())
    cfg["data"]["mix"][0]["ami"].update(n_train=8, seed=0)
    cfg["data"]["mix"][0]["ami"]["ext_tracks"]["require"] = True
    cfg["data"]["mix"][1]["synthetic"]["n_train"] = 8
    cfg["data"]["val"]["ami"]["n_val"] = 3
    rp = tmp_path / "v5_smoke.yaml"
    rp.write_text(yaml.safe_dump(cfg))
    init = tmp_path / "init.afm"
    _tiny_stage1_afm(init)
    spk_calls, orig = [], FastConformerEncoder.forward

    def spy(self, feats, flen, att=None, spk_act=None, **k):
        spk_calls.append(spk_act is not None)
        return orig(self, feats, flen, att, spk_act=spk_act, **k)

    monkeypatch.setattr(FastConformerEncoder, "forward", spy)
    ov = [f"init.from={init}", "trainer.max_steps=3", "trainer.device=cpu", "trainer.wer_gate=null",
          "trainer.log_every=1", "trainer.warmup_steps=1"]
    before = {k: v.clone() for k, v in load_model(init).state_dict().items()}
    model, metrics = run_recipe(str(rp), ov, out=str(tmp_path / "m.afm"))
    assert spk_calls and not any(spk_calls)  # training + eval: the encoder never gets a conditioning track
    head = model.heads["turn"]
    assert head.mode == "concat" and head.act_columns == 4 and head.duration_feats and model.head_reads_act
    assert model.cond["p_ext"] == 0.9 and "eot_turn_p50_ms@5fc" in metrics
    sd = model.state_dict()
    assert all(torch.equal(sd[k].cpu(), v) for k, v in before.items() if not k.startswith("heads.turn."))  # frozen
    k = "heads.turn.text_proj.0.weight"  # warm-started (same shape) and trained
    assert sd[k].shape == before[k].shape and not torch.equal(sd[k].cpu(), before[k])
    assert sd["heads.turn.out.weight"].shape == (1, 128)
    loaded = load_model(tmp_path / "m.afm")
    assert loaded.heads["turn"].mode == "concat" and not loaded.head_cfg["turn"]["condition_on_speaker"]


# --------------------------------------------------------------------------- evaluation
def test_eval_feeds_override_and_oracle_columns(ev):
    model, _ = _v5_model(None)
    model.eval()
    convs = conversation_dataset(5, seed=9)
    rng = np.random.default_rng(4)
    cols = [rng.random((len(c["spk_act"]) + int(rng.integers(-2, 3)), 4)).astype(np.float32) for c in convs]
    prims = [int(rng.integers(0, 4)) for _ in convs]
    acts = [c_[:, p] for c_, p in zip(cols, prims)]
    got = turn_scores(model, "turn", convs, batch_size=2, act_override=acts, cols_override=cols, prim_override=prims)
    ref = ev.turn_scores_given_act(model, "turn", convs, acts, 2, cols, prims)  # the Sortformer rows' path
    assert all(np.array_equal(a, b) for a, b in zip(got, ref))
    # the oracle row = oracle columns (spk_targets, primary column 0) + the oracle primary track
    orc = turn_scores(model, "turn", convs, batch_size=2)
    via = turn_scores(model, "turn", convs, batch_size=2, act_override=[c["spk_act"] for c in convs],
                      cols_override=[c["spk_targets"] for c in convs], prim_override=[0] * len(convs))
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(orc, via))
    # the override changes this (non-conditioned) head's output: primary track, columns and primary index all count
    assert any(not np.allclose(a, b) for a, b in zip(got, orc))
    prims2 = [(p + 1) % 4 for p in prims]
    alt = ev.turn_scores_given_act(model, "turn", convs, [c_[:, p] for c_, p in zip(cols, prims2)], 2, cols, prims2)
    assert any(not np.allclose(a, b) for a, b in zip(got, alt))
    same_act = ev.turn_scores_given_act(model, "turn", convs, acts, 2, [np.zeros_like(c_) for c_ in cols], prims)
    assert any(not np.allclose(a, b) for a, b in zip(got, same_act))
