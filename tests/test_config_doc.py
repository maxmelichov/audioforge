"""The server's one config surface: flags, the --config file, and the generated docs/CONFIGURATION.md block."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from audioforge.server import cli

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_doc_is_generated_from_the_flag_table():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "dev" / "gen_config_doc.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + "\nrun: python scripts/dev/gen_config_doc.py"


def test_every_parser_flag_is_in_the_table_and_documented():
    ap = cli.build_parser(advanced=True, launcher=True)
    flags = {s for a in ap._actions for s in a.option_strings if s.startswith("--")} - {"--help", "--help-advanced"}
    assert flags == {f.flags[0] for f in cli.FLAGS}
    doc = (ROOT / "docs" / "CONFIGURATION.md").read_text()
    for f in cli.FLAGS:
        assert f"`{f.flags[0]}" in doc, f.flags[0]
        assert f.help and "\n" not in f.help


def test_help_fits_one_screen(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.parse_args(["--help"], prog="audioforge-serve", launcher=True)
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert len(out.splitlines()) <= 45 and "--help-advanced" in out and "--perf" not in out
    with pytest.raises(SystemExit):
        cli.parse_args(["--help-advanced"], launcher=True)
    assert "--perf" in capsys.readouterr().out


def test_config_file_sets_options_and_flags_win(tmp_path):
    y = tmp_path / "serve.yaml"
    y.write_text("asr: a.afm\ndiar: d.afm\nport: 9000\nfinal-asr: tdt_v3\nlog_json: true\ndiar-set: [chunk_len=2]\n")
    a = cli.parse_args(["--config", str(y)])
    assert (a.asr, a.diar, a.port, a.final_asr, a.log_json, a.diar_set) == \
        ("a.afm", "d.afm", 9000, "tdt_v3", True, ["chunk_len=2"])
    a = cli.parse_args(["--config", str(y), "--port", "9001"])
    assert a.port == 9001 and a.threads == 2
    j = tmp_path / "serve.json"
    j.write_text(json.dumps({"asr": "x", "diar": "y", "diarizer": "nemotron3", "threads": 4}))
    a = cli.parse_args(["--config", str(j)])  # the launcher-only key is ignored by the module's own parser
    assert a.threads == 4 and not hasattr(a, "diarizer")
    assert cli.parse_args(["--config", str(j)], launcher=True).diarizer == "nemotron3"


def test_config_file_env_var_and_errors(tmp_path, monkeypatch, capsys):
    y = tmp_path / "c.yaml"
    y.write_text("asr: a\ndiar: d\nenroll: after_agent_arm\n")
    monkeypatch.setenv("AUDIOFORGE_CONFIG", str(y))
    assert cli.parse_args([]).enroll == "after_agent_arm"
    for bad in ("prot: 1\n", "enroll: nobody\n", "port: many\n", "log-json: 3\n"):
        y.write_text(bad)
        with pytest.raises(SystemExit) as ei:
            cli.parse_args(["--asr", "a", "--diar", "d"])
        assert ei.value.code == 2
    assert "unknown option 'prot'" in capsys.readouterr().err
    monkeypatch.delenv("AUDIOFORGE_CONFIG")
    with pytest.raises(SystemExit) as ei:  # --asr / --diar are required (from the line or the file)
        cli.parse_args(["--port", "1"])
    assert ei.value.code == 2


def test_launcher_counts_config_options_as_given(tmp_path, monkeypatch):
    from audioforge import launch

    monkeypatch.setattr(launch.hub, "find_model", lambda key, d=None: Path(f"/m/{key}"))
    argv = launch.resolve_models([], None, "nemotron3", {"asr": "mine.afm", "shed_diar": "vad"})
    assert "--asr" not in argv and argv[argv.index("--diar") + 1] == "/m/nemotron3"
    assert "--shed-diar" not in argv and argv[argv.index("--diar-pool") + 1] == "max"
    assert launch._strip(["--models-dir", "x", "--port", "1", "--diarizer=nemotron3"],
                         ("--models-dir", "--diarizer")) == ["--port", "1"]
