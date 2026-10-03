"""Hybrid (head OR timeout) under eot-bench v2 and the 0.32 s streaming Sortformer preset.

Under test:
  - conversation.or_outcomes / eot_outcomes_or: the OR of two detectors' per-conversation outcomes (hand-made and
    against a brute-force hybrid_fire_frame);
  - the AOSCConfig presets (card 1.04 s and 0.32 s rows), defaults unchanged;
  - eval_stage1 v2 tracks stage with another config writes to its own dir with the same keys (and the padding and
    config of that preset); make_sortformer_tracks --diar-config never writes the default cache;
  - v2 scores with a separate track set + tag write only the tagged dir; the default windows use the cropped track;
  - v2_block --v2-hybrid: the joint (θ, k) cross-fit, reported per fold.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audioforge import streaming_diar as sdm
from audioforge.conversation import (
    eot_outcomes,
    eot_outcomes_or,
    hybrid_fire_frame,
    or_outcomes,
    pause_fire,
    silence_scores,
)
from audioforge.datasets import ext_tracks as xt

ROOT = Path(__file__).parent.parent


def _script(name):
    path = ROOT / "scripts" / "research" / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{path.name} is a research driver, not part of this checkout")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ev():
    return _script("eval_stage1")


@pytest.fixture(scope="module")
def mk():
    return _script("make_sortformer_tracks")


# --------------------------------------------------------------------------- combinator
def test_or_outcomes_hand_made():
    inf = np.inf
    oa = dict(ths=np.array([0.5, inf]), fc=np.array([[True, False], [False, False]]),
              lat=np.array([[80., inf], [160., inf]]), pf=np.zeros((2, 2), np.int64), npause=np.array([1, 0]))
    ob = dict(ths=np.array([3., 9., inf]), fc=np.array([[False, False, False], [True, False, False]]),
              lat=np.array([[320., 800., inf], [240., inf, inf]]), pf=np.zeros((2, 3), np.int64),
              npause=np.array([1, 0]))
    pa = np.array([[True, False]])          # the one pause (conv 0): a fires at θ_a = 0.5 only
    pb = np.array([[False, False, False]])  # b never fires on it
    o = or_outcomes(oa, ob, pa, pb, np.array([0]))
    assert o["fc"].shape == (2, 6)
    # column j = jb * 2 + ja
    np.testing.assert_array_equal(o["ths"][:, 0], [0.5, inf] * 3)
    np.testing.assert_array_equal(o["ths"][:, 1], [3, 3, 9, 9, inf, inf])
    np.testing.assert_array_equal(o["fc"][0], [True, False, True, False, True, False])
    np.testing.assert_array_equal(o["fc"][1], [True, True, False, False, False, False])
    np.testing.assert_array_equal(o["lat"][0], [80, 320, 80, 800, 80, inf])
    np.testing.assert_array_equal(o["lat"][1], [160, 240, 160, inf, 160, inf])
    np.testing.assert_array_equal(o["pf"][0], [1, 0, 1, 0, 1, 0])
    assert (o["pf"][1] == 0).all() and list(o["npause"]) == [1, 0]
    # (θ_a = inf, θ_b = inf) never fires: no cutoffs, all misses
    assert not o["fc"][:, -1].any() and np.isinf(o["lat"][:, -1]).all()


def test_eot_outcomes_or_equals_brute_force_hybrid():
    rng = np.random.default_rng(0)
    heads, sils, on, en, pauses = [], [], [], [], []
    for i in range(12):
        T = 70
        act = (rng.random(T) < 0.7).astype(np.float32)
        o, e = 3, 40
        act[:o] = 0
        act[e:] = (rng.random(T - e) < 0.15)
        heads.append(rng.random(T).astype(np.float64))
        sils.append(silence_scores(act, o))
        on.append(o)
        en.append(e)
        pauses.append([(10, 13), (20, 22)] if i % 2 else [])
    ta, tb = np.array([0.9, 0.97, np.inf]), np.array([2., 5., np.inf])
    oc = eot_outcomes_or(heads, sils, on, en, ta, tb, post_end_frames=25, pauses=pauses)
    for j, (th, k) in enumerate(oc["ths"]):
        for i in range(12):
            # hybrid_fire_frame: p >= θ OR silence >= k; eot_outcomes: score > θ  -> use the next float up / k + 1
            f = hybrid_fire_frame(heads[i], sils[i], np.nextafter(th, np.inf), k + 1, start=on[i])
            fc = f < en[i]
            assert oc["fc"][i, j] == fc
            if not fc:
                # the first post-end firing (within 25 frames) - the brute force from the end frame
                g = hybrid_fire_frame(heads[i], sils[i], np.nextafter(th, np.inf), k + 1, start=en[i])
                want = (g - en[i] + 1) * 80.0 if g < min(len(heads[i]), en[i] + 25) else np.inf
                assert oc["lat"][i, j] == want
        # per pause: fired by either detector
        n_fire = sum(int((heads[i][a:b].max() > th) or (sils[i][a:b].max() > k)) for i in range(12)
                     for a, b in pauses[i])
        assert oc["pf"][:, j].sum() == n_fire
    # the pure slices reproduce the single-detector outcomes
    a_only = eot_outcomes(heads, on, en, ta, post_end_frames=25)
    ja = np.nonzero(np.isinf(oc["ths"][:, 1]))[0]
    np.testing.assert_array_equal(oc["lat"][:, ja], a_only["lat"])
    fire, cv = pause_fire(heads, pauses, ta)
    assert fire.shape == (12, 3) and list(np.unique(cv)) == [1, 3, 5, 7, 9, 11]


# --------------------------------------------------------------------------- presets
def test_presets(ev, mk):
    c = sdm.AOSCConfig.preset("low_latency_032")
    assert (c.chunk_len, c.chunk_right_context, c.fifo_len, c.spkcache_update_period, c.spkcache_len) == \
        (3, 1, 188, 144, 188)
    assert c.latency_ms == 320.0
    lo = sdm.AOSCConfig.preset("low_latency")
    assert lo.latency_ms == 1040.0
    assert sdm.SORTFORMER_PRESETS["low_latency"] == ev.SORTFORMER_LOW_LATENCY == mk.SORTFORMER_LOW_LATENCY
    assert ev.v2_diar_cfg("low_latency_032") == mk.diar_config("low_latency_032") == \
        sdm.SORTFORMER_PRESETS["low_latency_032"]
    d = sdm.AOSCConfig()  # dataclass defaults untouched
    assert (d.chunk_len, d.chunk_right_context, d.fifo_len, d.spkcache_len, d.spkcache_update_period) == (6, 2, 20, 40, 10)
    assert mk.pad_frames() == mk.PAD_FRAMES == 14 and mk.pad_frames(mk.diar_config("low_latency_032")) == 5
    assert mk.DIAR_CONFIG_SUFFIX["low_latency"] == "" and mk.DIAR_CONFIG_SUFFIX["low_latency_032"] == "_ll032"


# --------------------------------------------------------------------------- tracks stage: separate dir, same keys
def _ex(T=40, start=12.34, meeting="ES2011b"):
    act = np.zeros(T, np.float32)
    act[5:25] = 1
    y = np.zeros((T, 4), np.float32)
    y[:, 0] = act
    return {"meeting": meeting, "start": float(start), "audio": np.zeros(T * 1280, np.float32), "spk_act": act,
            "spk_targets": y, "onset_frame": 5, "turn_end_frame": 25, "hes": np.zeros(T, np.float32)}


def test_v2_tracks_other_config_writes_its_own_dir(ev, monkeypatch, tmp_path):
    seen = []

    class FakeSD:
        def __init__(self, dm, diar_head, mode, enc_left_context, **cfg):
            seen.append(cfg)

        def feed(self, x, final):
            self.n = len(x) // 1280
            seen.append(len(x))

        @property
        def all_probs(self):
            import torch
            return torch.full((self.n, 4), 0.25)

    clips = []
    ds = SimpleNamespace(_clip=lambda m, a, b: clips.append((a, b)) or np.zeros(int(round((b - a) * 16000)), np.float32),
                         duration=lambda m: 1000.0)
    monkeypatch.setattr(sdm, "StreamingDiarizer", FakeSD)
    monkeypatch.setattr(ev, "load_model", lambda ck, dev: SimpleNamespace())
    monkeypatch.setattr(ev, "_diar_name", lambda dm: "diar")
    ext = [_ex(start=1.0), _ex(start=5.0, T=50)]
    work, td = tmp_path / "work", tmp_path / "dev_ll032"
    (work / "tracks").mkdir(parents=True)
    r = ev.v2_tracks(ext, ds, work, 60.0, order=[1, 0], cfg_name="low_latency_032", tracks_dir=td)
    assert r == {"done": 2, "todo": 0}
    assert list((work / "tracks").iterdir()) == []  # the default track set is untouched
    names = sorted(p.name for p in td.iterdir())
    assert names == sorted(ev.v2_track_path(work, v).name for v in ext)  # same keys as the 1.04 s set
    assert names == sorted(f"{xt.example_key(v)}.stream_rc.npy" for v in ext)
    assert seen[0] == {"chunk_len": 3, "chunk_right_context": 1, "fifo_len": 188, "spkcache_update_period": 144,
                       "spkcache_len": 188}
    # padding = C + R + 1 = 5 frames of the following meeting audio
    assert abs((clips[0][1] - clips[0][0]) - (50 * 0.08 + 5 * 0.08)) < 1e-9
    for v in ext:
        assert np.load(ev.v2_track_path(work, v, td)).shape == (len(v["spk_act"]), 4)
    assert ev.v2_track_path(work, ext[0], td).parent == td
    assert ev.v2_track_path(work, ext[0]).parent == work / "tracks"


def test_make_sortformer_tracks_diar_config_dir(mk, monkeypatch, tmp_path):
    import json

    import audioforge.heads.turn as turn
    import audioforge.train as train
    x = (np.random.default_rng(0).standard_normal(16000 * 20) * 0.1).astype(np.float32)
    ds = SimpleNamespace(_audio={"M": x})
    exs = [{"meeting": "M", "start": s, "audio": x[int(s * 16000): int(s * 16000) + 16000 * 4]} for s in (1.0, 8.0)]
    cfgs = []

    def fake_stream(dm_, dn, audio, cfg=None):
        cfgs.append((cfg, len(audio)))
        from audioforge.data import ToneLanguage
        return np.zeros((ToneLanguage.n_frames(len(audio)), 4), np.float32)

    base = tmp_path / "dev"
    monkeypatch.setattr(mk, "turn_windows", lambda *a, **k: (exs, ds))
    monkeypatch.setattr(mk.xt, "cache_dir", lambda root, split, dataset="ami": base)
    monkeypatch.setattr(train, "load_model", lambda *a, **k: SimpleNamespace(preprocessor=SimpleNamespace(n_mels=128)))
    monkeypatch.setattr(turn, "_diar_name", lambda dm_: "diar")
    monkeypatch.setattr(mk, "stream_track", fake_stream)
    monkeypatch.setattr(sys, "argv", ["x", "--split", "dev", "--track-source", "stream", "--diar-config",
                                      "low_latency_032"])
    mk.main()
    d = tmp_path / "dev_ll032"
    assert not base.exists() or not any(base.iterdir())
    assert sorted(p.name for p in d.glob("*.stream.npy")) == sorted(f"{xt.example_key(e)}.stream.npy" for e in exs)
    assert all(c == mk.diar_config("low_latency_032") and n == 4 * 16000 + 5 * 1280 for c, n in cfgs)
    man = json.loads((d / "manifest.json").read_text())
    assert man["stream"]["config"]["chunk_len"] == 3 and man["stream"]["input_buffer_ms"] == 320.0


# --------------------------------------------------------------------------- scores: tag + separate track set
def test_v2_scores_tag_and_tracks_dir_isolation(ev, monkeypatch, tmp_path):
    base = [_ex(start=1.0), _ex(start=2.0)]
    ext = [_ex(start=1.0, T=60), _ex(start=2.0, T=60)]
    td = tmp_path / "ll032"
    td.mkdir()
    for v in ext:
        tr = np.zeros((60, 4), np.float32)
        tr[5:24, 2] = 0.9
        tr[:, 3] = np.arange(60) / 100.0  # marks the frame index: the crop must keep the first 40 frames
        np.save(ev.v2_track_path(tmp_path, v, td), tr)
    got = []

    def fake_scores(model, name, cc, acts, bs, cols=None, prims=None):
        got.append([c.copy() for c in cols])
        return [np.full(len(a), 0.75, np.float32) for a in acts]

    monkeypatch.setattr(ev, "turn_scores_given_act", fake_scores)
    def no_dev_cache(*a, **k):
        raise AssertionError("the dev cache must not be read with a separate track set")

    monkeypatch.setattr(xt, "cache_dir", no_dev_cache)
    model = SimpleNamespace(head_cfg={"turn": {"type": "turn"}})
    left = ev.v2_scores(model, base, ext, tmp_path, ["oracle@2s", "causal_dominant"], 60.0, 4, tag="ll", tracks_dir=td)
    assert left == {"oracle@2s": 0, "causal_dominant": 0}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ll032", "scores_causal_dominant__ll", "scores_oracle_2s__ll"]
    for v in base:
        assert np.load(ev.v2_score_path(tmp_path, "oracle@2s", v, "ll")).shape == (40,)
    c2 = got[0][0]  # oracle@2s: the extended track cropped to the default window's 40 frames
    assert c2.shape == (40, 4) and np.allclose(c2[:, 3], np.arange(40) / 100.0)


# --------------------------------------------------------------------------- report block with the hybrid
def test_v2_block_hybrid_rows(ev, tmp_path):
    rng = np.random.default_rng(0)
    convs, meta, tracks = [], [], []
    for i, m in enumerate(["IS1008b", "ES2011b", "TS3004b", "IB4002"] * 6):
        T = 90
        y = np.zeros((T, 4), np.float32)
        y[5:30, 0] = 1
        y[40:50, 1] = float(i % 2)
        convs.append({"meeting": m, "start": float(i), "audio": np.zeros(T * 1280, np.float32), "spk_act": y[:, 0].copy(),
                      "spk_targets": y, "onset_frame": 5, "turn_end_frame": 30, "hes": np.zeros(T, np.float32)})
        meta.append({"post_avail": 60, "end_reason": "trail"})
        tracks.append(np.clip(y + 0.05 * rng.standard_normal(y.shape), 0, 1).astype(np.float32))
    for mode in ("oracle", "causal_dominant"):
        for i, v in enumerate(convs):
            q = ev.v2_score_path(tmp_path, mode, v, "t6")
            q.parent.mkdir(parents=True, exist_ok=True)
            s = rng.random(90).astype(np.float32) * 0.5
            s[33 + i % 20:] = 1.0  # the head fires late on some turns: the timeout (k ~ 3) wins there
            np.save(q, s)
    r = ev.v2_block(convs, meta, tracks, tmp_path, "", {"2s": 25, "6s": 75}, 2, 20, score_dir_tag="t6",
                    cfg_name="low_latency_032", hybrid=True)
    assert {"hybrid_stream_oracle", "hybrid_stream_causal_dominant"} <= set(r["systems"])
    h = r["systems"]["hybrid_stream_causal_dominant"]["6s"]
    th = h["fixed_5pct_turn_fc"]["thresholds_by_fold"]
    assert set(th) == {0, 1} and all(set(x) == {"theta", "k_frames"} for x in th.values())
    t = r["systems"]["timeout_stream_causal_dominant"]["6s"]["fixed_5pct_turn_fc"]
    hd = r["systems"]["head_v3_stream_causal_dominant"]["6s"]["fixed_5pct_turn_fc"]
    assert h["fixed_5pct_turn_fc"]["p50_ms"] <= max(t["p50_ms"], hd["p50_ms"])
    assert "hybrid_stream_causal_dominant - timeout_stream_causal_dominant | 6s | fixed_5pct_turn_fc" in r["paired"]
    q = r["track_quality"]
    assert q["input_buffer_ms"] == 320.0 and q["emission_delay_frames_mean"] == 3.0  # (3 + 2 + 1) / 3 + 1
    assert 0 <= q["der"] < 0.2 and "offset_lag_frames_median" in q["causal_dominant"]
    r0 = ev.v2_block(convs, meta, tracks, tmp_path, "", {"2s": 25}, 2, 10, score_dir_tag="t6")  # default: no hybrid
    assert not any(k.startswith("hybrid_") for k in r0["systems"]) and "track_quality" not in r0
