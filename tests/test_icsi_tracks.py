"""ICSI external-diarizer tracks: scripts/research/make_sortformer_tracks.py --dataset icsi, datasets/ext_tracks.py (dataset=),
icsi.recipe_data ext_tracks (ICSI's own cache dir by default) and research/recipes/stage1_turn_v3_ami_icsi.yaml.

Hermetic tests run on test_icsi's hand-written NXT meeting TST001 (the diarizer is stubbed); the real-data smoke
(4 ICSI turn items through data.mix with their cached tracks) is skipped when the ICSI tracks are not cached."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from audioforge.data import Collate
from audioforge.datasets import ami, icsi
from audioforge.datasets import ext_tracks as xt

sys.path.insert(0, str(Path(__file__).parent))
from test_icsi import root  # noqa: E402,F401,F811  (fixture: meeting TST001 under a tmp ICSI root)

ROOT = Path(__file__).parent.parent
RECIPE = ROOT / "research" / "recipes" / "stage1_turn_v3_ami_icsi.yaml"


@pytest.fixture(scope="module")
def mk():
    spec = importlib.util.spec_from_file_location("make_sortformer_tracks", ROOT / "scripts" / "research" / "make_sortformer_tracks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _icsi_cfg(root, split="train", **ext):  # noqa: F811
    key = "train" if split == "train" else "val"
    src = {"mode": "turn", "window_sec": 16, "root": str(root), "meetings": {key: ["TST001"]}}
    if ext:
        src["ext_tracks"] = ext
    return {"data": {"icsi": src}}


# --------------------------------------------------------------------------- dirs / keys
def test_cache_dir_per_dataset(tmp_path):
    assert xt.DATASETS == ("ami", "icsi")
    assert xt.cache_dir(None, "train") == ami.DEFAULT_ROOT / "cache" / "sortformer" / "train"  # AMI unchanged
    assert xt.cache_dir(None, "dev", "icsi") == icsi.DEFAULT_ROOT / "cache" / "sortformer" / "dev"
    assert xt.cache_dir(tmp_path, "train", "icsi") == xt.cache_dir(tmp_path, "train") == tmp_path / "cache" / "sortformer" / "train"
    with pytest.raises(AssertionError):
        xt.cache_dir(None, "train", "voxconverse")


def test_script_builds_the_recipe_windows_for_icsi(root, mk, monkeypatch):  # noqa: F811
    assert mk.default_meetings("icsi", "train") == icsi.DEFAULT_MEETINGS["train"]
    assert mk.default_meetings("icsi", "dev") == icsi.DEFAULT_MEETINGS["dev"] == ["Bmr021", "Bns001"]
    assert mk.default_meetings("ami", "train") == ami.DEFAULT_MEETINGS["train"]
    monkeypatch.setattr(icsi, "DEFAULT_ROOT", root)  # icsi._root(None) -> the fixture root
    exs = mk.turn_windows("train", 16.0, meetings=["TST001"], dataset="icsi")
    rec = icsi.recipe_data(_icsi_cfg(root), "train")
    assert len(exs) == 3 and [xt.example_key(e) for e in exs] == [xt.example_key(e) for e in rec]
    assert xt.example_key(exs[0]) == f"TST001_00000000_{len(exs[0]['audio'])}"
    assert xt.cache_dir(None, "train", "icsi") == root / "cache" / "sortformer" / "train"


def _stub_diarizer(mk, monkeypatch, seed=0):
    """Replace the Sortformer checkpoint by random (T, 4) tracks (main() otherwise loads runs/nemo_sortformer_v2.afm)."""
    import audioforge.heads.turn as turn
    import audioforge.train as train
    rng = np.random.default_rng(seed)
    monkeypatch.setattr(train, "load_model", lambda *a, **k: SimpleNamespace(preprocessor=SimpleNamespace(n_mels=128)))
    monkeypatch.setattr(turn, "_diar_name", lambda dm: "diar")
    track = lambda audio: rng.random((len(audio) // 1280, 4)).astype(np.float32)  # noqa: E731
    monkeypatch.setattr(mk, "offline_tracks", lambda dm, dn, audios, device="cpu": [track(a) for a in audios])
    monkeypatch.setattr(mk, "stream_track", lambda dm, dn, audio: track(audio))


def test_script_main_writes_icsi_cache_and_recipe_attaches_it(root, mk, monkeypatch):  # noqa: F811
    monkeypatch.setattr(icsi, "DEFAULT_ROOT", root)
    monkeypatch.setattr(mk.icsi, "DEFAULT_MEETINGS", {**icsi.DEFAULT_MEETINGS, "train": ["TST001"]})
    _stub_diarizer(mk, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "icsi", "--split", "train", "--track-source", "offline"])
    mk.main()
    d = root / "cache" / "sortformer" / "train"
    keys = [xt.example_key(e) for e in mk.turn_windows("train", 16.0, dataset="icsi")]
    assert all(xt.track_path(d, k, "offline").exists() for k in keys) and len(keys) == 3
    man = json.loads((d / "manifest.json").read_text())
    assert man["dataset"] == "icsi" and man["meetings"] == ["TST001"] and man["offline"]["n"] == 3
    assert man["window_sec"] == 16.0 and man["stream"]["n"] == 0
    assert not (ami.DEFAULT_ROOT / "cache" / "sortformer" / "train" / f"{keys[0]}.npy").exists()  # not AMI's dir
    # stream tracks for the first 2 windows (resumable --limit), then the recipe hook: stream, offline fallback
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", "icsi", "--split", "train", "--track-source", "stream",
                                      "--limit", "2"])
    mk.main()
    assert [xt.track_path(d, k, "stream").exists() for k in keys] == [True, True, False]
    data = icsi.recipe_data(_icsi_cfg(root, source="stream", fallback="offline", require=True), "train")
    assert len(data) == 3
    for e, k, src in zip(data, keys, ("stream", "stream", "offline")):
        tr = np.load(xt.track_path(d, k, src))
        np.testing.assert_array_equal(e["spk_targets_ext"], xt.fit(tr, len(e["spk_act"])))
        assert e["spk_prim_ext"].sum() == 1 and e["spk_act_ext"].shape == e["spk_act"].shape
    # string spec, default dir from the recipe root (no DEFAULT_ROOT patch needed)
    monkeypatch.setattr(icsi, "DEFAULT_ROOT", ROOT / "nonexistent")
    data = icsi.recipe_data(_icsi_cfg(root), "train")
    assert icsi.attach_ext_tracks(data, "offline", "train", str(root)) == {"ext_tracks": "offline", "found": 3,
                                                                           "missing": 0}
    # explicit dir wins; val split reads cache/sortformer/dev (empty here -> -1, never used)
    assert icsi.attach_ext_tracks(data, {"source": "offline", "dir": str(root / "elsewhere")}, "train",
                                  str(root))["missing"] == 3
    val = icsi.recipe_data(_icsi_cfg(root, "val", source="offline"), "val")
    assert all((e["spk_act_ext"] == xt.MISSING).all() for e in val)


# --------------------------------------------------------------------------- recipe
def test_recipe_ami_icsi_keys(mk):
    cfg = yaml.safe_load(RECIPE.read_text())
    v3 = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_v3.yaml").read_text())
    assert cfg["name"] == "stage1_turn_v3_ami_icsi"
    for k in ("init", "encoder", "heads"):
        assert cfg[k] == v3[k], k
    drop = lambda t: {k: v for k, v in t.items() if k != "max_steps"}  # noqa: E731
    assert drop(cfg["trainer"]) == drop(v3["trainer"]) and cfg["trainer"]["max_steps"] == 3000
    mix = {e["name"]: e for e in cfg["data"]["mix"]}
    assert list(mix) == ["ami_turn", "icsi_turn", "conv"]
    assert mix["ami_turn"] == v3["data"]["mix"][0] and mix["ami_turn"]["ami"]["ext_tracks"]["source"] == "stream"
    ic = mix["icsi_turn"]
    assert ic["icsi"] == {"mode": "turn", "window_sec": 16, "ext_tracks": {"source": "stream", "fallback": "offline"}}
    assert ic["icsi"]["window_sec"] == mk.WINDOW_SEC["train"]  # the windows the tracks were cached for
    assert ic["weight"] == 1.0 and ic["drop_keys"] == ["speaker"]  # ICSI ids 1000+ > the 190-class speaker head
    assert mix["conv"] == v3["data"]["mix"][1] and mix["conv"]["weight"] == 0.3
    assert cfg["data"]["val"] == v3["data"]["val"] == {"ami": {"mode": "turn", "val_split": "dev", "n_val": 64}}
    assert cfg["data"]["batching"] == "source" and cfg["data"]["derive"] == ["vad", "eou"]


def test_recipe_ami_icsi_routes_through_load_mix():
    from audioforge.train import load_data
    cfg = yaml.safe_load(RECIPE.read_text())
    cfg["data"]["standin"] = {"n_train": 3, "n_val": 2}  # every source returns its synthetic stand-in
    tr, va = load_data(cfg, "train"), load_data(cfg, "val")
    assert tr.counts == {"ami_turn": 3, "icsi_turn": 3, "conv": 1} and len(va) == 2
    assert all("speaker" not in tr[i] for i in range(3, 6))  # icsi_turn drop_keys


# --------------------------------------------------------------------------- real data
@pytest.mark.skipif(not (xt.has_tracks(split="train", dataset="icsi") and xt.has_tracks(split="dev", dataset="icsi")),
                    reason="ICSI Sortformer tracks not cached (scripts/research/make_sortformer_tracks.py --dataset icsi)")
def test_recipe_data_ext_tracks_real_4_items(mk):
    from audioforge.train import load_data
    cfg = yaml.safe_load(RECIPE.read_text())
    entry = copy.deepcopy(next(e for e in cfg["data"]["mix"] if e["name"] == "icsi_turn"))
    entry["icsi"].update(n_train=4, n_val=4, seed=0)
    entry["icsi"]["ext_tracks"]["require"] = True
    data = load_data({"data": {"mix": [entry], "derive": ["vad", "eou"]}}, "train")
    items = [data[i] for i in range(len(data))]
    assert len(items) == 4
    d = xt.cache_dir(None, "train", "icsi")
    stream_complete = xt.manifest(split="train", dataset="icsi").get("stream", {}).get("n") == \
        xt.manifest(split="train", dataset="icsi").get("n_examples")
    for e in items:
        T = len(e["spk_act"])
        assert "speaker" not in e and e["meeting"] in icsi.DEFAULT_MEETINGS["train"]
        assert e["spk_targets_ext"].shape == (T, 4) and e["spk_act_ext"].shape == (T,) and e["spk_prim_ext"].sum() == 1
        assert xt.track_path(d, xt.example_key(e), "offline").exists()
        if stream_complete:
            src = "stream"
            np.testing.assert_array_equal(e["spk_targets_ext"], xt.fit(np.load(xt.track_path(d, xt.example_key(e), src)), T))
    b = Collate(None)(items)
    assert b["spk_targets_ext"].shape[0] == 4 and b["spk_prim_ext"].shape == (4, 4)
    # val split (dev meetings, 20 s windows) finds its tracks in data/icsi/cache/sortformer/dev
    val = icsi.recipe_data({"data": {"icsi": {**entry["icsi"], "window_sec": 20}}}, "val")
    assert len(val) == 4 and all(e["spk_prim_ext"].sum() == 1 for e in val)
    # the script's dev windows are exactly the recipe's (all of them)
    allv = icsi.recipe_data({"data": {"icsi": {"mode": "turn", "window_sec": 20}}}, "val")
    assert [xt.example_key(e) for e in mk.turn_windows("dev", 20.0, dataset="icsi")] == [xt.example_key(e) for e in allv]
