"""Turn head v3 trained on 6 s post-end trails (research/recipes/stage1_turn_v3_trail6.yaml; research/STAGE1.md "Turn head v3
trained on 6 s trails"; motivation research/EOT_BENCH_V2.md: v3 saw only 2 s after a turn end).

Under test:
  - AMI turn windows with trail_sec 6 / window_sec 20: >= 75 post-end frames unless the trail stops at the primary's
    next turn or the meeting end; the window cap only clips the lead; the same turns / turn ends as the 2 s set;
  - their cache lives in <split>_trail6/ (ext_tracks.split_dirname), selected by the recipe's trail_sec, never the
    default cache; make_sortformer_tracks --trail-sec writes there;
  - the recipe loads (yaml + a 3-step CPU smoke run with a tiny stand-in init model);
  - eot-bench v2 --v2-tag keeps a second checkpoint's head scores apart from the untagged (v3) ones.
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from audioforge.data import ToneLanguage
from audioforge.datasets import ami
from audioforge.datasets import ext_tracks as xt

ROOT = Path(__file__).parent.parent
SR = 16000


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "research" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mk():
    return _script("make_sortformer_tracks")


@pytest.fixture(scope="module")
def ev():
    return _script("eval_stage1")


def _have_ami_train():
    try:
        m = ami.DEFAULT_MEETINGS["train"][0]
        return (ami.DEFAULT_ROOT / "cache" / f"{m}.f32.npy").exists()
    except Exception:
        return False


needs_ami = pytest.mark.skipif(not _have_ami_train(), reason="AMI train audio / annotations not cached")


# --------------------------------------------------------------------------- windows
@pytest.fixture(scope="module")
def ami_train():
    torch.set_num_threads(2)
    ds = ami.AMI(list(ami.DEFAULT_MEETINGS["train"]), verbose=False)
    e2 = ds.turn_examples(16.0)
    e6 = ds.turn_examples(20.0, 4.0, 6.0)  # sets t["_next_start"] on ds.turns (the same-speaker bound)
    return ds, e2, e6


@needs_ami
def test_trail6_windows_have_6s_after_the_end_unless_clipped(ami_train):
    ds, e2, e6 = ami_train
    assert len(e6) == len(e2) == 3274  # the same turns (min_trail 1 s passes for exactly the same ones)
    full = clipped_resume = clipped_end = 0
    for e in e6:
        m = e["meeting"]
        T, end = len(e["spk_act"]), e["turn_end_frame"]
        assert T == ToneLanguage.n_frames(len(e["audio"])) and len(e["audio"]) <= 20 * SR + 1
        b = e["start"] + len(e["audio"]) / SR
        # the turn of this window: the primary's turn whose end lands on turn_end_frame
        t = min((t for t in ds.turns[m] if not t["bc"] and ds.gid(t["speaker"]) == e["speaker"]),
                key=lambda t: abs(t["end"] - (e["start"] + end * 0.08)))
        if b - t["end"] >= 6.0 - 1e-6:
            assert T - end >= 75, (m, e["start"], T - end)  # the full 6 s trail = >= 75 post-end frames
            full += 1
        elif abs(b - ds.duration(m)) < 1e-3:
            clipped_end += 1
        else:  # stopped where the same speaker's next turn starts
            assert abs(b - t["_next_start"]) < 1e-3, (m, e["start"], b, t["_next_start"])
            assert T - end < 76
            clipped_resume += 1
    assert (full, clipped_resume, clipped_end) == (1344, 1929, 1)  # time-exact 6.0 s trails
    assert sum(len(e["spk_act"]) - e["turn_end_frame"] >= 75 for e in e6) == 1353  # + 9 resume < 1 frame short of 6 s


@needs_ami
def test_trail6_same_turns_longer_trail_window_cap_clips_only_the_lead(ami_train):
    _, e2, e6 = ami_train
    for e, u in zip(e6, e2):  # the same turns in the same order
        assert (e["meeting"], e["speaker"]) == (u["meeting"], u["speaker"])
        assert abs((e["start"] + e["turn_end_frame"] * 0.08) - (u["start"] + u["turn_end_frame"] * 0.08)) <= 0.081
        assert e["start"] + len(e["audio"]) / SR >= u["start"] + len(u["audio"]) / SR - 1e-6  # trail >= the 2 s one
        assert e["text"].endswith(u["text"].split()[-1])
        post6, post2 = len(e["spk_act"]) - e["turn_end_frame"], len(u["spk_act"]) - u["turn_end_frame"]
        assert post6 >= post2 - 1  # a later start (lead clipped by the 20 s cap) can shift the 80 ms grid by 1 frame
        assert (e["eot"][e["turn_end_frame"]:] == 1).all() and (e["eot"][: e["turn_end_frame"]] == 0).all()


# --------------------------------------------------------------------------- keys / dirs
def test_split_dirname_and_cache_dir(tmp_path):
    assert xt.split_dirname("train") == xt.split_dirname("train", None) == xt.split_dirname("train", 2.0) == "train"
    assert xt.split_dirname("train", 6.0) == xt.split_dirname("train", 6) == "train_trail6"
    assert xt.split_dirname("dev", 3.5) == "dev_trail3.5"
    assert xt.cache_dir(tmp_path, "train", trail_sec=6.0) == tmp_path / "cache" / "sortformer" / "train_trail6"
    assert xt.cache_dir(tmp_path, "train", trail_sec=2.0) == xt.cache_dir(tmp_path, "train")
    assert xt.cache_dir(None, "train", trail_sec=6.0) == ami.DEFAULT_ROOT / "cache" / "sortformer" / "train_trail6"
    assert xt.cache_dir(tmp_path, "train", "icsi", 6.0) == tmp_path / "cache" / "sortformer" / "train_trail6"


def _ex(T=40, start=12.34, meeting="ES2002a", n=None):
    act = np.zeros(T, np.float32)
    act[5:20] = 1
    return {"meeting": meeting, "start": start, "audio": np.zeros(n or T * 1280, np.float32), "spk_act": act,
            "onset_frame": 5, "turn_end_frame": 20}


def test_attach_resolves_the_trail_dir_only(tmp_path):
    d6 = xt.cache_dir(tmp_path, "train", trail_sec=6.0)
    d6.mkdir(parents=True)
    e = _ex()
    tr = np.zeros((40, 4), np.float32)
    tr[5:20, 2] = 0.9
    np.save(xt.track_path(d6, xt.example_key(e), "stream"), tr)
    assert xt.has_tracks(tmp_path, "train", "stream", trail_sec=6.0)
    assert not xt.has_tracks(tmp_path, "train", "stream")
    assert xt.attach([_ex()], tmp_path, "train", "stream") == {"found": 0, "missing": 1}  # default dir: nothing
    got = [_ex()]
    assert xt.attach(got, tmp_path, "train", "stream", trail_sec=6.0) == {"found": 1, "missing": 0}
    assert got[0]["spk_prim_ext"].argmax() == 2
    # the recipe hook: trail_sec selects the dir, fallback null = streaming only; an explicit dir wins
    r = ami.attach_ext_tracks([_ex()], {"source": "stream", "fallback": None, "require": True}, "train", tmp_path, 6.0)
    assert r["found"] == 1 and r["missing"] == 0
    assert ami.attach_ext_tracks([_ex()], {"source": "stream"}, "train", tmp_path)["found"] == 0  # 2 s set: default dir
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert ami.attach_ext_tracks([_ex()], {"source": "stream", "dir": str(other)}, "train", tmp_path, 6.0)["found"] == 0
    (d6 / "manifest.json").write_text(json.dumps({"trail_sec": 6.0}))
    assert xt.manifest(tmp_path, "train", trail_sec=6.0) == {"trail_sec": 6.0} and xt.manifest(tmp_path, "train") == {}


@needs_ami
def test_recipe_data_passes_the_recipes_trail_to_the_turn_windows_and_the_cache(monkeypatch):
    seen = {}

    def fake_attach(data, root, split, src, directory, require=False, fallback=None, trail_sec=None, **kw):
        seen.update(split=split, trail_sec=trail_sec, directory=directory)
        return {"found": len(data), "missing": 0}
    monkeypatch.setattr(xt, "attach", fake_attach)
    cfg = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_v3_trail6.yaml").read_text())
    src = dict(cfg["data"]["mix"][0]["ami"], n_train=6, seed=0)
    data = ami.recipe_data({"data": {"ami": src}}, "train")
    assert len(data) == 6 and seen == {"split": "train", "trail_sec": 6.0, "directory": None}
    assert xt.cache_dir(None, seen["split"], trail_sec=seen["trail_sec"]).name == "train_trail6"
    assert max(len(e["spk_act"]) - e["turn_end_frame"] for e in data) >= 75


def test_make_tracks_trail_sec_writes_the_trail_dir(mk, monkeypatch, tmp_path):
    """--trail-sec 6: windows built with trail 6, tracks + manifest in <root>/cache/sortformer/train_trail6/, the
    default train/ dir untouched."""
    import audioforge.heads.turn as turn
    import audioforge.train as train
    x = (np.random.default_rng(0).standard_normal(SR * 30) * 0.1).astype(np.float32)
    ds = SimpleNamespace(_audio={"M": x})
    exs = [{"meeting": "M", "start": s, "audio": x[int(s * SR): int(s * SR) + SR * 8]} for s in (1.0, 12.0)]
    got = {}

    def fake_windows(split, ws, meetings=None, dataset="ami", return_ds=False, trail_sec=2.0):
        got.update(ws=ws, trail_sec=trail_sec)
        return exs, ds
    monkeypatch.setattr(mk, "turn_windows", fake_windows)
    monkeypatch.setattr(ami, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(ami, "_root", lambda root=None: Path(root) if root else tmp_path)
    monkeypatch.setattr(train, "load_model", lambda *a, **k: SimpleNamespace(preprocessor=SimpleNamespace(n_mels=128)))
    monkeypatch.setattr(turn, "_diar_name", lambda dm_: "diar")
    monkeypatch.setattr(mk, "stream_track", lambda dm_, dn, audio: np.zeros((ToneLanguage.n_frames(len(audio)), 4),
                                                                             np.float32))
    monkeypatch.setattr(sys, "argv", ["x", "--split", "train", "--track-source", "stream", "--trail-sec", "6",
                                      "--window-sec", "20"])
    mk.main()
    d6 = tmp_path / "cache" / "sortformer" / "train_trail6"
    assert got == {"ws": 20.0, "trail_sec": 6.0}
    assert sorted(p.name for p in d6.glob("*.stream.npy")) == sorted(f"{xt.example_key(e)}.stream.npy" for e in exs)
    man = json.loads((d6 / "manifest.json").read_text())
    assert man["trail_sec"] == 6.0 and man["window_sec"] == 20.0 and man["stream"]["n"] == 2
    assert man["stream"]["flush_fix"]["pad_mode"] == "audio"
    assert not (tmp_path / "cache" / "sortformer" / "train").exists()


D6 = xt.cache_dir(None, "train", trail_sec=6.0)


@pytest.mark.skipif(not xt.has_tracks(split="train", source="stream", trail_sec=6.0),
                    reason="trail-6 train streaming tracks not cached")
@needs_ami
def test_real_trail6_cache_matches_the_windows(ami_train):
    _, e2, e6 = ami_train
    man = xt.manifest(split="train", trail_sec=6.0)
    assert man["trail_sec"] == 6.0 and man["window_sec"] == 20.0 and man["n_examples"] == len(e6)
    assert man["stream"]["flush_fix"]["pad_mode"] == "audio"
    have = [e for e in e6 if xt.track_path(D6, xt.example_key(e), "stream").exists()]
    if man["stream"]["n"] == len(e6):
        assert len(have) == len(e6)
    for e in have[:: max(1, len(have) // 20)]:
        p = np.load(xt.track_path(D6, xt.example_key(e), "stream"))
        assert p.shape == (len(e["spk_act"]), 4) and np.isfinite(p).all()
    # the default 2 s cache is a different window set: it is not read for the trail-6 windows
    assert xt.cache_dir(None, "train") != D6


# --------------------------------------------------------------------------- recipe
def test_recipe_trail6_yaml_is_v3_with_the_trail6_source():
    cfg = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_v3_trail6.yaml").read_text())
    v3 = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_v3.yaml").read_text())
    assert cfg["name"] == "stage1_turn_v3_trail6"
    for k in ("init", "encoder", "heads", "trainer"):
        assert cfg[k] == v3[k], k
    assert cfg["data"]["val"] == v3["data"]["val"] and cfg["data"]["mix"][1] == v3["data"]["mix"][1]
    src = cfg["data"]["mix"][0]["ami"]
    assert (src["mode"], src["window_sec"], src["lead_sec"], src["trail_sec"]) == ("turn", 20, 4.0, 6.0)
    assert src["ext_tracks"] == {"source": "stream", "fallback": None, "require": True}
    assert cfg["trainer"]["max_steps"] == 2000


@pytest.mark.skipif(not xt.has_tracks(split="train", source="stream", trail_sec=6.0),
                    reason="trail-6 train streaming tracks not cached")
@needs_ami
def test_recipe_trail6_smoke(tmp_path):
    """3 CPU steps of the recipe with a tiny stand-in init model and 8 trail-6 AMI items (+ 8 synthetic)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_turn_v3 import _tiny_stage1_afm

    from audioforge.train import run_recipe
    torch.set_num_threads(2)
    cfg = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_v3_trail6.yaml").read_text())
    have = {p.name[: -len(".stream.npy")] for p in D6.glob("*.stream.npy")}
    probe = ami.recipe_data({"data": {"ami": dict(cfg["data"]["mix"][0]["ami"], ext_tracks=None)}}, "train")
    idx = [i for i, e in enumerate(probe) if xt.example_key(e) in have]
    if len(idx) < 8:
        pytest.skip("fewer than 8 trail-6 tracks cached")
    ms = sorted({probe[i]["meeting"] for i in idx[:8]})
    cfg["data"]["mix"][0]["ami"].update(n_train=8, seed=0, meetings={"train": ms[:1]})
    first = [e for e in probe if e["meeting"] == ms[0]]
    if not all(xt.example_key(e) in have for e in first):
        pytest.skip("first meeting's trail-6 tracks not all cached yet")
    cfg["data"]["mix"][1]["synthetic"]["n_train"] = 8
    cfg["data"]["val"]["ami"]["n_val"] = 3
    rp = tmp_path / "t6.yaml"
    rp.write_text(yaml.safe_dump(cfg))
    init = tmp_path / "init.afm"
    _tiny_stage1_afm(init)
    ov = [f"init.from={init}", "trainer.max_steps=3", "trainer.device=cpu", "trainer.wer_gate=null",
          "trainer.log_every=1", "trainer.warmup_steps=1"]
    model, metrics = run_recipe(str(rp), ov, out=str(tmp_path / "m.afm"))
    assert model.heads["turn"].act_columns == 4 and "eot_turn_p50_ms@5fc" in metrics


# --------------------------------------------------------------------------- eot-bench v2 tag isolation
def test_v2_tag_isolates_score_dirs(ev, monkeypatch, tmp_path):
    ex = _ex()
    assert ev.v2_score_path(tmp_path, "oracle@2s", ex) == tmp_path / "scores_oracle_2s" / f"{xt.example_key(ex)}.npy"
    assert ev.v2_score_path(tmp_path, "oracle@2s", ex, "trail6") == \
        tmp_path / "scores_oracle_2s__trail6" / f"{xt.example_key(ex)}.npy"
    assert ev.v2_score_path(tmp_path, "causal_dominant", ex, "trail6").parent.name == "scores_causal_dominant__trail6"
    # v2_scores with a tag writes only the tagged dir and leaves the untagged (v3) scores alone
    base = [_ex(start=1.0), _ex(start=2.0)]
    ext = [_ex(start=1.0, T=60), _ex(start=2.0, T=60)]
    (tmp_path / "tracks").mkdir()
    for v in ext:
        tr = np.zeros((60, 4), np.float32)
        tr[5:20, 1] = 0.9
        np.save(ev.v2_track_path(tmp_path, v), tr)
    v3dir = tmp_path / "scores_oracle"
    v3dir.mkdir()
    for v in ext:
        np.save(ev.v2_score_path(tmp_path, "oracle", v), np.full(60, 0.25, np.float32))
    monkeypatch.setattr(ev, "turn_scores_given_act", lambda model, name, cc, acts, bs, cols=None, prims=None:
                        [np.full(len(a), 0.75, np.float32) for a in acts])
    model = SimpleNamespace(head_cfg={"turn": {"type": "turn"}})
    left = ev.v2_scores(model, base, ext, tmp_path, ["oracle"], 60.0, 4, tag="trail6")
    assert left == {"oracle": 0}
    for v in ext:
        assert np.load(ev.v2_score_path(tmp_path, "oracle", v)).max() == 0.25  # v3 untouched
        assert np.load(ev.v2_score_path(tmp_path, "oracle", v, "trail6")).min() == 0.75
    assert sorted(p.name for p in tmp_path.iterdir()) == ["scores_oracle", "scores_oracle__trail6", "tracks"]


def test_v2_block_reads_the_tagged_head_scores(ev, tmp_path):
    """The report block loads head rows from scores_<binding>__<tag> (and not the untagged v3 dir) when tagged."""
    rng = np.random.default_rng(0)
    convs, meta, tracks = [], [], []
    for i, m in enumerate(["IS1008b", "ES2011b", "TS3004b", "IB4002"] * 2):
        T = 60
        y = np.zeros((T, 4), np.float32)
        y[5:30, 0] = 1
        y[40:50, 1] = float(i % 2)
        convs.append({"meeting": m, "start": float(i), "audio": np.zeros(T * 1280, np.float32), "spk_act": y[:, 0].copy(),
                      "spk_targets": y, "onset_frame": 5, "turn_end_frame": 30, "hes": np.zeros(T, np.float32)})
        meta.append({"post_avail": 30, "end_reason": "trail"})
        tracks.append(np.clip(y + 0.05 * rng.standard_normal(y.shape), 0, 1).astype(np.float32))
    for mode in ("oracle", "causal_dominant"):
        for v in convs:
            q = ev.v2_score_path(tmp_path, mode, v, "t6")
            q.parent.mkdir(parents=True, exist_ok=True)
            s = np.zeros(60, np.float32)
            s[33:] = 1.0
            np.save(q, s)
    r = ev.v2_block(convs, meta, tracks, tmp_path, "", {"2s": 25}, 2, 10, score_dir_tag="t6")
    assert {"head_v3_stream_oracle", "head_v3_stream_causal_dominant"} <= set(r["systems"])
    assert "head_v3_stream_first_active" not in r["systems"]  # no tagged scores for it
    r0 = ev.v2_block(convs, meta, tracks, tmp_path, "", {"2s": 25}, 2, 10)  # untagged: nothing stored there
    assert not any(k.startswith("head_") for k in r0["systems"])
