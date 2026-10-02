"""``audioforge-serve``: start the server with the models ``audioforge-download`` installed.

    audioforge-serve                                  # single-model mode (default): the one 115M model, known user
    audioforge-serve --core 0.6b --device mps         # the second core: nemotron-speech-streaming-en-0.6b + its heads
    audioforge-serve --mode room                      # + Nemotron-3-Diarization (max pooling, all 8 columns)
    audioforge-serve --mode room --diarizer sortformer  # + Streaming Sortformer v2
    audioforge-serve --config serve.yaml              # any flag from a YAML / JSON file (flags on the line win)
    audioforge-serve --help-advanced                  # every flag

On top of ``python -m audioforge.serve`` it adds ``--models-dir`` and ``--diarizer``: ``--asr`` / ``--diar`` are filled
in from the models directory when absent (``--shed-diar hold`` is added unless given, research/archive/DIARIZATION_FIX.md), as
are the optional model paths a flag needs (TitaNet for ``--enroll after_agent|explicit``, AmberNet for
``--lid ambernet``, Silero for ``hybrid_silero``/``hybrid_dyn`` in room mode, Parakeet-TDT v3 for ``--final-asr tdt_v3``). Every
other flag goes to ``audioforge.serve`` unchanged (docs/CONFIGURATION.md explains them).

Without ``--mode``, ``--diarizer`` / ``--diar`` / ``--final-asr`` select room mode (the paragraph above); otherwise the
default ``--mode single`` (docs/CONFIGURATION.md section 13, research/SINGLE_MODEL.md) adds ``cli.MODES["single"]``
(``--turn-input tsvad --diar-off --lid head --enroll after_agent_arm --turn-policy vad_head --dyn-wait-ms 2000,960``)
and the TS-VAD head file, loads no diarizer and no Silero (the default turn rule ``vad_head`` reads the model's own VAD,
turn and TS-VAD heads, research/EOT_LATENCY.md), and refuses the options that would load a second model
(``cli.SINGLE_CONFLICTS``, ``--diarizer``, ``--lid ambernet``).
"""
from __future__ import annotations

import os
import sys

from . import hub
from .server import cli

__all__ = ["CORE_FILES", "TSVAD_FILE", "core_files", "find_head", "pick_mode", "resolve_models", "serve_main"]

TSVAD_FILE = "tsvad_spk.pt"  # the TS-VAD head of research/IMPROVE_115M.md part A (serve --turn-input tsvad)
LID_FILE = "lid_115m_v2.pt"  # the distilled language-ID head (serve --lid head); optional, see docs/MODELS.md
# --core 0.6b (research/CORE_0P6B.md): the same roles on nemotron-speech-streaming-en-0.6b (heads retrained on it)
CORE_FILES = {"115m": (TSVAD_FILE, LID_FILE), "0.6b": ("tsvad_0p6b.pt", "lid_0p6b_v2.pt")}


def core_files(core: str) -> tuple[str, str]:
    """(TS-VAD head file, LID head file) of a core; exits with the choices on an unknown core."""
    if core not in CORE_FILES:
        sys.exit(f"audioforge-serve: --core {core!r}: one of {', '.join(CORE_FILES)}")
    return CORE_FILES[core]


def _dl_hint(core: str, models_dir: str | None) -> str:
    return ("audioforge-download" + ("" if core == hub.CORE_DEFAULT else f" --core {core}")
            + (f" --dir {models_dir}" if models_dir else ""))


def _in_argv(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _argv_value(argv: list[str], flag: str) -> str | None:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _strip(argv: list[str], flags: tuple[str, ...]) -> list[str]:
    """argv without the given value-taking flags (``--flag V`` and ``--flag=V``)."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
        elif a in flags:
            skip = True
        elif not any(a.startswith(f + "=") for f in flags):
            out.append(a)
    return out


def find_head(name: str, models_dir: str | None = None):
    """A small head file (``tsvad_spk.pt``, ``lid_115m_v2.pt``): the models directory, else this checkout's assets/,
    else its runs/."""
    from .paths import ROOT
    return next((p for p in (hub.models_dir(models_dir) / name, ROOT / "assets" / name, ROOT / "runs" / name)
                 if p.exists()), None)


def _single(argv: list[str], models_dir: str | None, config: dict, diarizer_given: bool,
            core: str = "115m") -> list[str]:
    """--mode single: the preset's options unless given; exits on an option that would load a second model."""
    def given(dest: str):
        v = _argv_value(argv, "--" + dest.replace("_", "-"))
        return v if v is not None else config.get(dest)
    bad = [f"--{d.replace('_', '-')} ({why})" for d, why in cli.SINGLE_CONFLICTS
           if given(d) not in (None, False) and not (d == "diar_embed" and given(d) != "titanet")]
    if diarizer_given:
        bad.append("--diarizer (a diarizer model)")
    lid = given("lid")
    if lid is not None and (lid == "ambernet" or str(lid).endswith(".nemo")):
        bad.append("--lid ambernet (AmberNet)")
    tsvad_file, lid_file = core_files(core)
    if bad:
        sys.exit("audioforge-serve: --mode single runs the one core model; drop " + ", ".join(bad)
                 + " (or use --mode room)")
    for dest, val in cli.MODES["single"].items():
        flag = "--" + dest.replace("_", "-")
        if _in_argv(argv, flag) or dest in config:
            continue
        if dest == "lid" and val == "head":
            p = find_head(lid_file, models_dir)
            if p is None:
                print(f"audioforge-serve: {lid_file} not found, language ID is off "
                      f"(audioforge-download --only {hub.core_keys(core)[2]}, or --lid ambernet in --mode room)",
                      file=sys.stderr)
                continue
            if core != "115m":  # serve's "head" means the 115M file; name the core's own head explicitly
                argv += [flag, str(p)]
                continue
        argv += [flag] if val is True else [flag, str(val)]
    if not (_in_argv(argv, "--tsvad") or "tsvad" in config):
        p = find_head(tsvad_file, models_dir)
        if p is None:
            sys.exit(f"audioforge-serve: single-model mode needs the TS-VAD head {tsvad_file}, not found in "
                     f"{hub.models_dir(models_dir)}.\n  run: "
                     + (f"audioforge-download --only tsvad lid" + (f" --dir {models_dir}" if models_dir else "")
                        if core == "115m" else _dl_hint(core, models_dir))
                     + "   (or pass --tsvad PATH, or --mode room)")
        argv += ["--tsvad", str(p)]
    return argv


def pick_mode(argv: list[str], config: dict, mode: str | None, diarizer: str | None) -> str:
    """``--mode`` if given, else ``room`` when a diarizer, ``--diar`` or ``--final-asr`` is asked for, else ``single``."""
    if mode:
        return mode
    two_model = diarizer is not None or any(_in_argv(argv, f) or f[2:].replace("-", "_") in config
                                            for f in ("--diar", "--final-asr"))
    return "room" if two_model else "single"


def resolve_models(argv: list[str], models_dir: str | None = None, diarizer: str | None = None,
                   config: dict | None = None, mode: str | None = None, diarizer_given: bool = False,
                   core: str = "115m") -> list[str]:
    """argv for audioforge.serve with the model paths filled in; exits with a hint when a model is missing.
    ``config`` holds the options of the ``--config`` file: an option set there counts as given. ``mode`` is the
    ``--mode`` preset (None: ``pick_mode``; ``single``: no diarizer, see ``_single``; ``room``: + a diarizer,
    ``diarizer`` or else Nemotron-3 if downloaded, else Sortformer v2)."""
    argv = list(argv)
    config = config or {}
    diarizer_given = diarizer_given or diarizer is not None
    mode = pick_mode(argv, config, mode, diarizer)
    core_files(core)  # exits on an unknown core
    asr_key = hub.core_keys(core)[0]
    if core != "115m" and not (_in_argv(argv, "--device") or "device" in config):
        print(f"[serve] --core {core} on CPU: ~96 ms of compute per 160 ms chunk on 2 threads, one real-time stream "
              "per process (the 115M holds 4; research/CORE_0P6B.md); --device mps / cuda for more", file=sys.stderr,
              flush=True)
    if mode == "single":
        argv = _single(argv, models_dir, config, diarizer_given, core)
        if not (_in_argv(argv, "--asr") or "asr" in config):
            p = hub.find_model(asr_key, models_dir)
            if p is None:
                sys.exit(f"audioforge-serve: the server needs the '{asr_key}' model, not found in "
                         f"{hub.models_dir(models_dir)}.\n  run: {_dl_hint(core, models_dir)}")
            argv += ["--asr", str(p)]
        # no Silero: the default turn rule (vad_head) reads the model's own heads; a session that asks for
        # hybrid_silero / hybrid_dyn loads serve's default Silero path lazily, or pass --silero
        return argv  # no diarizer, TitaNet, AmberNet, TDT or Silero

    def _has(argv: list[str], flag: str) -> bool:
        return _in_argv(argv, flag) or flag[2:].replace("-", "_") in config

    def _value(argv: list[str], flag: str) -> str | None:
        v = _argv_value(argv, flag)
        return v if v is not None else config.get(flag[2:].replace("-", "_"))

    def need(key: str, why: str):
        p = hub.find_model(key, models_dir)
        if p is None:
            extra = "" if key == "asr" else f" --diarizer {key}" if key in hub.DIARIZERS else f" --only {key}"
            sys.exit(f"audioforge-serve: {why} needs the '{key}' model, not found in {hub.models_dir(models_dir)}.\n"
                     f"  run: audioforge-download{extra}" + (f" --dir {models_dir}" if models_dir else ""))
        return str(p)

    if not _has(argv, "--asr"):
        if core != "115m" and hub.find_model(asr_key, models_dir) is None:
            sys.exit(f"audioforge-serve: --core {core} needs the '{asr_key}' model, not found in "
                     f"{hub.models_dir(models_dir)}.\n  run: {_dl_hint(core, models_dir)}")
        argv += ["--asr", need(asr_key, "the server")]
    if diarizer is None:  # room mode: Nemotron-3 (the measured room / product default), else Sortformer v2
        diarizer = "nemotron3" if hub.find_model("nemotron3", models_dir) or not hub.find_model("sortformer", models_dir) \
            else "sortformer"
    if not _has(argv, "--diar"):
        argv += ["--diar", need(diarizer, "the server")]
        product = hub.diarizer_defaults(diarizer)  # Nemotron-3: max pooling, frame-local encoder, all 8 columns
    else:
        product = {"shed_diar": "hold"}  # research/archive/DIARIZATION_FIX.md section 4: no speaker-0 collapse under load
    for key, val in product.items():
        flag = "--" + key.replace("_", "-")
        if not _has(argv, flag):
            argv += [flag, str(val)]
    if _value(argv, "--enroll") in ("after_agent", "explicit") and not _has(argv, "--titanet"):
        argv += ["--titanet", need("titanet", "--enroll after_agent/explicit")]
    if _value(argv, "--lid") == "ambernet":
        i = argv.index("--lid") if "--lid" in argv else None
        path = need("ambernet", "--lid ambernet")
        if i is not None:
            argv[i + 1] = path
        else:
            argv = [a for a in argv if not a.startswith("--lid=")] + ["--lid", path]
    if _value(argv, "--final-asr") == "tdt_v3" and "AUDIOFORGE_TDT_V3" not in os.environ:
        os.environ["AUDIOFORGE_TDT_V3"] = need("tdt_v3", "--final-asr tdt_v3")  # read by final_asr (also in its worker)
    if not _has(argv, "--silero"):
        p = hub.find_model("silero", models_dir)  # models dir, else $AUDIOFORGE_DATA/silero (serve's own default is <repo>/data)
        if p is not None:
            argv += ["--silero", str(p)]
    return argv


def serve_main(argv: list[str] | None = None) -> int:
    """Entry point of ``audioforge-serve``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = ["--help-advanced" if x == "--help-serve" else x for x in argv]  # the flag's name before 2026-09-29
    a = cli.parse_args(argv, prog="audioforge-serve", launcher=True)  # help, config file and value checks
    config = cli.load_config(a.config) if a.config else {}
    from . import serve
    rest = _strip(argv, ("--models-dir", "--diarizer", "--mode", "--core"))
    mode = pick_mode(rest, config, a.mode, a.diarizer)
    if a.mode is None and mode == "room":
        print("[serve] room mode (a diarizer / --diar / --final-asr was given); the default is single-model mode",
              file=sys.stderr, flush=True)
    core = a.core or config.get("core") or hub.CORE_DEFAULT
    serve.main(resolve_models(rest, a.models_dir, a.diarizer, config, mode=mode, diarizer_given=a.diarizer is not None,
                              core=core))
    return 0


if __name__ == "__main__":
    sys.exit(serve_main())
