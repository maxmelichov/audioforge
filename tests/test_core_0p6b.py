"""The second core (--core 0.6b, research/CORE_0P6B.md): registry, download set, launcher wiring and clear errors run
everywhere; the served-model checks run only when the 0.6B has been downloaded / built (skipped otherwise)."""
from pathlib import Path

import pytest

from audioforge import hub, launch

ROOT = Path(__file__).resolve().parents[1]


def test_registry_pins_the_0p6b_core():
    c = hub.COMPONENTS["asr_0p6b"]
    assert c.repo == "nvidia/nemotron-speech-streaming-en-0.6b" and len(c.revision) == 40
    assert c.sha256 == "283638054c44f6794e74fe9af9048d78a6d9d6c058c12131856c7859a62ac9cd"
    assert c.license == "NVIDIA Open Model License" and "nvidia-open-model-license" in c.license_url
    assert c.convert == "served" and c.output == hub.SERVED_0P6B
    assert hub.core_keys("0.6b") == ("asr_0p6b", "tsvad_0p6b", "lid_0p6b")
    assert hub.core_keys("115m") == ("asr", "tsvad", "lid") and hub.CORE_DEFAULT == "115m"
    with pytest.raises(ValueError, match="unknown core"):
        hub.core_keys("1b")


def test_shipped_0p6b_heads_match_their_pins():
    name, size, sha, out = hub.HEADS_0P6B[hub.HEADS_0P6B_VERSION]
    if not sha:
        pytest.skip("0.6B heads not pinned yet in this revision")
    p = ROOT / "assets" / name
    assert p.exists(), f"{p} ships in the repository"
    assert p.stat().st_size == size and hub.sha256_file(p) == sha
    assert hub.heads_path(None, ROOT / "models", hub.HEADS_0P6B_VERSION, hub.HEADS_0P6B) == p
    for key in ("tsvad_0p6b", "lid_0p6b"):
        c = hub.COMPONENTS[key]
        f = ROOT / "assets" / c.filename
        if key == "lid_0p6b" and not f.exists():
            continue  # the LID head is optional and not in every checkout (licence, see docs/MODELS.md)
        assert f.stat().st_size == c.size and hub.sha256_file(f) == c.sha256


def test_download_core_0p6b_set(tmp_path, monkeypatch):
    seen = {}

    def fake_install(keys, directory, *a, **k):
        seen["keys"] = keys
        for key in keys:
            (directory / hub.COMPONENTS[key].output).write_bytes(b"x")
        return {key: directory / hub.COMPONENTS[key].output for key in keys}
    monkeypatch.setattr(hub, "install", fake_install)
    hub.main(["--dir", str(tmp_path), "--yes", "--core", "0.6b"])
    assert seen["keys"] == ["asr_0p6b", "tsvad_0p6b", "lid_0p6b"]
    hub.main(["--dir", str(tmp_path), "--yes"])
    assert seen["keys"] == ["asr", "tsvad"]  # the 115M default set is unchanged


def test_launcher_core_0p6b_resolves_its_own_files(tmp_path, capsys):
    for name in (hub.SERVED_0P6B, "tsvad_0p6b.pt", "lid_0p6b.pt"):
        (tmp_path / name).write_bytes(b"x")
    argv = launch.resolve_models(["--device", "mps"], str(tmp_path), mode="single", core="0.6b")
    val = lambda f: argv[argv.index(f) + 1]  # noqa: E731
    assert val("--asr") == str(tmp_path / hub.SERVED_0P6B)
    assert val("--tsvad") == str(tmp_path / "tsvad_0p6b.pt")
    assert val("--lid") == str(tmp_path / "lid_0p6b.pt")
    assert "one real-time stream" not in capsys.readouterr().err  # a GPU device was given
    launch.resolve_models([], str(tmp_path), mode="single", core="0.6b")
    assert "one real-time stream" in capsys.readouterr().err  # CPU: the cost is stated


def test_launcher_core_0p6b_missing_models_say_how_to_get_them(tmp_path, monkeypatch):
    # only the (empty) models directory counts: this checkout's assets/ and runs/ may hold the 0.6B files
    monkeypatch.setattr(launch, "find_head", lambda name, d=None: next(
        (p for p in [tmp_path / name] if p.exists()), None))
    monkeypatch.setattr(hub, "find_model", lambda key, d=None: next(
        (p for p in [tmp_path / hub.COMPONENTS[key].output] if p.exists()), None))
    with pytest.raises(SystemExit, match="audioforge-download --core 0.6b"):
        launch.resolve_models(["--device", "mps"], str(tmp_path), mode="single", core="0.6b")
    (tmp_path / "tsvad_0p6b.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="'asr_0p6b' model, not found.*\n.*audioforge-download --core 0.6b"):
        launch.resolve_models(["--device", "mps"], str(tmp_path), mode="single", core="0.6b")
    with pytest.raises(SystemExit, match="--core '1b'"):
        launch.resolve_models([], str(tmp_path), mode="single", core="1b")


def test_serve_help_lists_core(capsys):
    from audioforge.server import cli
    with pytest.raises(SystemExit):
        cli.parse_args(["--help"], prog="audioforge-serve", launcher=True)
    assert "--core" in capsys.readouterr().out


def _served_0p6b():
    p = hub.find_model("asr_0p6b")
    t = hub.find_model("tsvad_0p6b")
    if p is None or t is None:
        pytest.skip("the 0.6B core is not downloaded (audioforge-download --core 0.6b)")
    return p, t


def test_served_0p6b_model_streams_like_the_offline_forward():
    import numpy as np
    import torch

    from audioforge.model import StreamingSession
    from audioforge.train import load_model
    p, _ = _served_0p6b()
    m = load_model(str(p), "cpu")
    assert m.encoder.d_model == 1024 and len(m.encoder.layers) == 24
    assert m.encoder.att_context_size == [70, 1]
    assert {"rnnt", "vad", "spk", "turn", "turn_seg"} <= set(m.heads)
    assert m.layer_tap["spk"] == [4] and not m.head_cfg["turn"].get("condition_on_speaker")  # block 5
    rng = np.random.default_rng(0)
    x = (0.05 * rng.standard_normal(16000 * 3)).astype(np.float32)
    feats, fl = m.preprocessor(torch.tensor(x)[None], torch.tensor([len(x)]))
    with torch.no_grad():
        off, olen = m.encoder(feats, fl, [70, 1])
        cs, T, st, outs = m.encoder.stream_chunk_frames([70, 1]), int(fl), None, []
        for s in range(0, T, cs):
            y, st = m.encoder.stream_step(feats[..., s:s + cs], st, [70, 1], final=s + cs >= T)
            outs.append(y)
    on = torch.cat(outs, 1)
    assert on.shape[1] == int(olen) and float((on - off[:, : on.shape[1]]).abs().max()) < 1e-4
    sess = StreamingSession(m, "rnnt", [70, 1])
    for i in range(0, len(x), 2560):
        sess.feed(x[i:i + 2560])
    assert sess.feed(x[:0], final=True) == m.transcribe([x], head="rnnt", att_context_size=[70, 1])[0]


def test_single_mode_engine_on_the_0p6b_emits_events():
    import audioforge
    p, t = _served_0p6b()
    fe = audioforge.load(core="0.6b", warmup=False)
    assert fe.engine.asr.encoder.d_model == 1024
    ev = fe.run_file(ROOT / "examples" / "audio" / "two_party_call_16s.wav")
    assert any(e["type"] == "final" for e in ev) and any(e["type"] == "turn_end" for e in ev)


def test_frame_gru_head_streams_like_the_whole_sequence():
    import torch

    from audioforge.model import build_head
    torch.manual_seed(0)
    h = build_head({"type": "frame_gru", "key": "vad", "hidden": 8, "from_layers": [0, 1], "weight": 0.5}, 16)
    x = torch.randn(1, 11, 16)
    full = h(x)
    st = h.init_stream(1)
    parts = torch.cat([h.step(x[:, i:i + 2], st) for i in range(0, 11, 2)], 1)
    assert full.shape == (1, 11) and torch.allclose(full, parts, atol=1e-6)
    loss = h.loss(x, torch.tensor([11]), {"vad": torch.ones(1, 11)})
    assert torch.isfinite(loss)
