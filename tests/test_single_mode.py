"""audioforge-serve --mode single (research/SINGLE_MODEL.md, docs/CONFIGURATION.md section 13): the preset resolves to
the single-model flags, refuses a second model, loads no diarizer / TitaNet / Silero / final-ASR worker, and streams
the bundled clip end to end over the WebSocket on tiny models."""
import asyncio
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge import launch
from audioforge.server import cli

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "examples" / "audio" / "two_party_call_16s.wav"  # the single-mode quickstart clip
EMB = 192  # the protocol's voice print size


def hub_served():
    from audioforge.hub import SERVED
    return SERVED  # stage1_served_v4.afm: what audioforge-download builds


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tiny_models_dir(tmp_path: Path) -> Path:
    """A models directory holding a tiny served .afm (hub.SERVED) (ASR + VAD + turn + a single-tap speaker head),
    tsvad_spk.pt and lid_115m_v2.pt for it; no diarizer, TitaNet, AmberNet, TDT or Silero."""
    from audioforge.heads.tsvad import TSVADHead
    from audioforge.lid import save_head
    from audioforge.model import SpeechModel
    from audioforge.tokenizer import CharTokenizer
    from audioforge.train import save_model
    H = _load("test_serve")
    torch.manual_seed(0)
    base = H._asr_model()
    cfg = dict(base.cfg)
    cfg["heads"] = {**cfg["heads"], "spk": {"type": "speaker", "num_speakers": 5, "emb_dim": EMB, "from_layers": [1]}}
    m = SpeechModel(cfg, CharTokenizer(list("abc ")))
    m.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        m.heads["vad"].net[-1].bias.fill_(5.0)  # always speech: the print is taken and the LID pools every frame
    d = tmp_path / "models"
    d.mkdir()
    save_model(m, d / hub_served())
    h = TSVADHead(32, emb_dim=EMB, hidden=16)
    torch.save({"cfg": {"type": "tsvad", "emb_dim": EMB, "hidden": 16}, "state_dict": h.state_dict()},
               d / launch.TSVAD_FILE)
    lcfg = {**cfg, "heads": {**cfg["heads"], "lid": {"type": "language", "num_languages": 2, "hidden": 8,
                                                     "att_hidden": 4, "cls_hidden": 8, "labels": ["en", "he"],
                                                     "from_layers": [0]}}}
    donor = SpeechModel(lcfg, CharTokenizer(list("abc ")))
    with torch.no_grad():
        donor.heads["lid"].cls[-1].bias.copy_(torch.tensor([8.0, 0.0]))  # confidently "en"
    save_head(donor, "lid", d / "lid_115m_v2.pt")
    return d


def _value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


# --------------------------------------------------------------------------- the preset
def test_preset_resolves_to_the_single_model_flags(tmp_path, monkeypatch):
    d = _tiny_models_dir(tmp_path)
    monkeypatch.setenv("AUDIOFORGE_HOME", str(d))
    argv = launch.resolve_models([], str(d), mode="single")
    assert _value(argv, "--turn-input") == "tsvad" and "--diar-off" in argv
    assert _value(argv, "--lid") == "head" and _value(argv, "--enroll") == "after_agent_arm"
    assert _value(argv, "--dyn-wait-ms") == "2000,960" and _value(argv, "--turn-policy") == "vad_head"
    assert "--silero" not in argv  # the default turn rule reads the model's own heads (research/EOT_LATENCY.md)
    assert _value(argv, "--asr") == str(d / hub_served()) and _value(argv, "--tsvad") == str(d / launch.TSVAD_FILE)
    for flag in ("--diar", "--final-asr", "--titanet", "--shed-diar", "--diar-pool"):
        assert flag not in argv, flag
    # flags given on the command line win over the preset
    argv = launch.resolve_models(["--enroll", "explicit", "--port", "9000"], str(d), mode="single")
    assert _value(argv, "--enroll") == "explicit" and argv.count("--enroll") == 1 and "--titanet" not in argv
    a = cli.parse_args(argv)  # python -m audioforge.serve accepts it without --diar
    assert a.diar is None and a.diar_off and a.turn_input == "tsvad" and a.lid == "head" and a.final_asr is None
    assert cli.MODES["room"] == {}  # room mode adds nothing: the previous two-model stack


def test_mode_from_the_config_file_and_serve_main(tmp_path, monkeypatch):
    d = _tiny_models_dir(tmp_path)
    cfg = tmp_path / "serve.yaml"
    cfg.write_text("mode: single\nport: 9001\n")
    assert cli.parse_args(["--config", str(cfg)], prog="audioforge-serve", launcher=True).mode == "single"
    seen = {}
    import audioforge.serve as serve_mod
    monkeypatch.setattr(serve_mod, "main", lambda argv: seen.setdefault("argv", argv))
    launch.serve_main(["--config", str(cfg), "--models-dir", str(d)])
    argv = seen["argv"]
    assert "--mode" not in argv and "--diar" not in argv and "--diar-off" in argv and _value(argv, "--lid") == "head"
    a = cli.parse_args(argv)  # the non-launcher parser ignores the launcher-only `mode` key of the same file
    assert a.port == 9001 and a.diar_off


def test_single_is_the_default_and_a_diarizer_selects_room(tmp_path, monkeypatch):
    d = _tiny_models_dir(tmp_path)
    (d / "nemo_nemotron3_diar.afm").write_bytes(b"x")
    seen = []
    import audioforge.serve as serve_mod
    monkeypatch.setattr(serve_mod, "main", lambda argv: seen.append(argv))
    launch.serve_main(["--models-dir", str(d)])  # no --mode: single
    assert "--diar-off" in seen[-1] and "--diar" not in seen[-1]
    launch.serve_main(["--models-dir", str(d), "--diarizer", "nemotron3"])  # a diarizer without --mode: room
    assert _value(seen[-1], "--diar") == str(d / "nemo_nemotron3_diar.afm") and "--diar-off" not in seen[-1]
    launch.serve_main(["--models-dir", str(d), "--mode", "room"])  # room defaults to Nemotron-3
    assert _value(seen[-1], "--diar") == str(d / "nemo_nemotron3_diar.afm") and _value(seen[-1], "--diar-pool") == "max"
    assert launch.pick_mode([], {}, None, None) == "single" and launch.pick_mode(["--final-asr", "x"], {}, None, None) == "room"
    with pytest.raises(SystemExit):
        cli.parse_args(["--mode", "default"], prog="audioforge-serve", launcher=True)  # renamed to room


def test_download_default_set_is_single_model(tmp_path, monkeypatch):
    from audioforge import hub
    seen = {}

    def fake_install(keys, directory, *a, **k):
        seen["keys"] = keys
        for key in keys:
            (directory / hub.COMPONENTS[key].output).parent.mkdir(parents=True, exist_ok=True)
            (directory / hub.COMPONENTS[key].output).write_bytes(b"x")
        return {key: directory / hub.COMPONENTS[key].output for key in keys}
    monkeypatch.setattr(hub, "install", fake_install)
    hub.main(["--dir", str(tmp_path), "--yes"])
    assert seen["keys"] == ["asr", "tsvad"]  # the LID head is optional (--with lid); Silero only --with silero
    hub.main(["--dir", str(tmp_path), "--yes", "--diarizer", "nemotron3"])
    assert seen["keys"] == ["asr", "tsvad", "nemotron3"]
    for ver, (name, size, sha, _out) in hub.HEADS.items():  # heads assets: v0.4 ships, v0.3 / v0.2 / v0.1 kept
        p = ROOT / "assets" / name
        assert p.stat().st_size == size and hub.sha256_file(p) == sha and hub.heads_path(None, tmp_path, ver) == p
    assert hub.SERVED == hub.HEADS[hub.HEADS_VERSION][3] == "stage1_served_v4.afm"
    for key in ("tsvad", "lid"):  # the shipped head files match their pinned sha256 (the LID head may be absent)
        c = hub.COMPONENTS[key]
        p = ROOT / "assets" / c.filename
        if key == "lid" and not p.exists():
            continue
        assert p.stat().st_size == c.size and hub.sha256_file(p) == c.sha256


def test_download_of_a_missing_head_fails_clearly(tmp_path, monkeypatch):
    from audioforge import hub

    def boom(url, dst):
        raise OSError("HTTP Error 404: Not Found")
    monkeypatch.setattr(hub, "_http_get", boom)
    with pytest.raises(RuntimeError, match="could not download .*tsvad_spk.pt"):
        hub.fetch(hub.COMPONENTS["tsvad"], tmp_path, [])


@pytest.mark.parametrize("extra", [["--final-asr", "tdt_v3"], ["--diarizer", "nemotron3"], ["--diar", "x.afm"],
                                   ["--lid", "ambernet"], ["--diar-embed", "titanet"]])
def test_single_refuses_a_second_model(tmp_path, monkeypatch, extra):
    d = _tiny_models_dir(tmp_path)
    import audioforge.serve as serve_mod
    monkeypatch.setattr(serve_mod, "main", lambda argv: pytest.fail("must not start"))
    with pytest.raises(SystemExit) as e:
        launch.serve_main(["--mode", "single", "--models-dir", str(d)] + extra)
    assert "--mode single" in str(e.value)


def test_help_lists_the_mode_flag(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["--help"], prog="audioforge-serve", launcher=True)
    out = capsys.readouterr().out
    assert "--mode {single,room}" in out and "single (default: the one 115M model, known user)" in out


# --------------------------------------------------------------------------- one model loaded
def _single_engine(tmp_path, monkeypatch, **over):
    d = _tiny_models_dir(tmp_path)
    monkeypatch.setenv("AUDIOFORGE_HOME", str(d))
    import audioforge.nemo_import as ni
    monkeypatch.setattr(ni, "load_any", lambda *a, **k: pytest.fail("a diarizer was loaded"))
    import audioforge.enrollment as en
    monkeypatch.setattr(en, "TitaNetEmbedder", lambda *a, **k: pytest.fail("TitaNet was loaded"))
    argv = launch.resolve_models([f"--{k.replace('_', '-')}={v}" for k, v in over.items()] + ["--threads", "1"],
                                 str(d), mode="single")
    a = cli.parse_args(argv)
    cli._startup_checks(a)
    return cli.load_engine(a)


def test_single_engine_loads_no_second_model(tmp_path, monkeypatch):
    eng = _single_engine(tmp_path, monkeypatch)
    assert eng.diar is None and eng.diar_mode == "off" and eng.num_spks == 4
    assert eng.embedder is None and eng.final_asr is None and eng.lid_model is None
    assert eng.dyn_t0 == 25.0 and eng.dyn_a == 13.0  # --dyn-wait-ms 2000,960 (research/SINGLE_MODEL.md A1)
    # the default turn rule: vad_head with the others path (research/EOT_LATENCY.md), no Silero at start or per session
    assert eng.turn_policy == "vad_head" and (eng.vad_head_k, eng.vad_head_fb, eng.vad_head_others) == (2, 8, (12, 8))
    from audioforge.serve import Session, SessionConfig
    assert Session(eng, SessionConfig(turn_policy=eng.turn_policy)).sil is None and eng.silero_model is None
    assert eng.registry_embedder is None and eng.lid_name == "lid" and eng.tsvad is not None
    assert eng.name == Path(hub_served()).stem and eng.turn_input == "tsvad" and eng.diar_off
    r = eng.ready_msg()
    assert r["diar_config"] == "off" and r["model"] == Path(hub_served()).stem and r["column_lag_ms"] == 40.0
    assert "final_asr" not in r and r["enroll"] == "after_agent_arm"
    mods = {id(m) for m in (eng.asr, eng.tsvad)}
    assert all(not isinstance(v, torch.nn.Module) or id(v) in mods for v in vars(eng).values())
    from audioforge.serve import Engine
    with pytest.raises(ValueError):  # without --diar-off a diarizer stays required
        Engine(eng.asr, None, name="x", threads=1)


# --------------------------------------------------------------------------- end to end over the socket
async def _client(url, pcm_i16: bytes, sr: int, first: dict | None):
    from websockets.asyncio.client import connect
    out = []
    async with connect(url) as ws:
        out.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "config", "turn_policy": "timeout", "sample_rate": sr}))
        if first is not None:
            await ws.send(json.dumps(first))
        step = sr // 50 * 2  # 20 ms blocks, like a microphone
        for i in range(0, len(pcm_i16), step):
            await ws.send(pcm_i16[i:i + step])
        await ws.send(json.dumps({"type": "end"}))
        async for m in ws:
            out.append(json.loads(m))
    return out


@pytest.mark.parametrize("how", ["enroll_embedding", "agent_end"])
def test_single_streams_the_bundled_clip_end_to_end(tmp_path, monkeypatch, how):
    pytest.importorskip("websockets")
    import soundfile as sf

    from audioforge.serve import validate
    H = _load("test_serve")
    eng = _single_engine(tmp_path, monkeypatch, lid_min_ms=400)
    pcm, sr = sf.read(str(CLIP), dtype="int16")
    first = ({"type": "enroll", "embedding": np.random.default_rng(0).standard_normal(EMB).round(4).tolist()}
             if how == "enroll_embedding" else {"type": "agent_end"})
    msgs = asyncio.run(H._with_server(eng, lambda url: _client(url, pcm.astype("<i2").tobytes(), sr, first)))
    for m in msgs:
        validate(m)
    types = [m["type"] for m in msgs]
    assert types[0] == "ready" and msgs[0]["diar_config"] == "off" and types[-1] == "stats"
    assert "error" not in types, [m for m in msgs if m["type"] == "error"]
    vp = [m for m in msgs if m["type"] == "voiceprint"]
    assert len(vp) == 1 and vp[0]["source"] == ("explicit" if how == "enroll_embedding" else "arm")
    lang = [m for m in msgs if m["type"] == "language"]
    assert lang and lang[0]["language"] == "en"
    st = msgs[-1]
    assert st["enrolled"] is True and st["lang"] == "en" and "final_asr" not in st
