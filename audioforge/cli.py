"""audioforge CLI.

  python -m audioforge.cli catalog fetch               # refresh from Hugging Face
  python -m audioforge.cli catalog summary
  python -m audioforge.cli catalog search "streaming" --decoder rnnt
  python -m audioforge.cli train research/recipes/parakeet_tdt_ctc.yaml -o out.afm [key=value ...]
  python -m audioforge.cli transcribe out.afm audio.wav [--head ctc] [--prompt "<|en|>..."]
  python -m audioforge.cli stream out.afm audio.wav    # cache-aware streaming, prints partials
  python -m audioforge.cli export out.afm encoder.onnx
  python -m audioforge.cli train-codec --mel -o runs/codec.pt --steps 1500
  python -m audioforge.cli train-salm --steps 800 --encoder-from runs/parakeet_tdt_ctc.afm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "research" / "catalog.json"
RAW = ROOT / "research" / "archive" / "raw"


def cmd_catalog(a):
    from . import catalog as c
    if a.action == "fetch":
        entries = c.fetch(RAW)
        c.save(entries, CATALOG)
        print(f"fetched {len(entries)} audio models -> {CATALOG}")
        return
    entries = c.load(CATALOG) if CATALOG.exists() else c.build_from_raw(RAW)
    if a.action == "summary":
        print(json.dumps(c.summarize(entries), indent=1))
        return
    res = c.search(entries, a.query or "", a.family or "", a.encoder or "", a.decoder or "",
                   True if a.streaming else None, a.sort)
    for e in res[: a.limit]:
        print(f"{e.id:<62} {e.family:<30} {e.encoder:<16} {e.decoder:<22} {e.params:>6} {e.downloads:>9,}")
    print(f"-- {len(res)} match(es)")


def cmd_train(a):
    from .train import run_recipe
    run_recipe(a.recipe, a.overrides, a.out)


def _audio(path, sr):
    from .data import load_wav
    return load_wav(path, sr)


def cmd_transcribe(a):
    from .train import load_model
    m = load_model(a.model)
    for p in a.audio:
        print(p, "->", m.transcribe([_audio(p, m.preprocessor.sample_rate)], head=a.head, prompt=a.prompt)[0])


def cmd_stream(a):
    from .model import StreamingSession
    from .train import load_model
    m = load_model(a.model)
    x = _audio(a.audio, m.preprocessor.sample_rate)
    s = StreamingSession(m, a.head)
    step = int(a.chunk_ms * m.preprocessor.sample_rate / 1000)
    last = None
    for i in range(0, len(x), step):
        t = s.feed(x[i:i + step])
        if t != last:
            print(f"{(i + step) / m.preprocessor.sample_rate:6.2f}s  {t}")
            last = t
    print("final:", s.feed([], final=True))


def cmd_export(a):
    from .train import export_onnx, load_model
    print("wrote", export_onnx(load_model(a.model), a.out))


def cmd_train_codec(a):
    import torch

    from .train import train_codec
    codec, _ = train_codec(a.steps, gan=a.gan, codec_cfg={"type": "mel"} if a.mel else None)
    torch.save(codec.state_dict(), a.out)
    print("saved", a.out)


def cmd_train_salm(a):
    from .train import train_speech_llm
    _, _, w = train_speech_llm(a.steps, encoder_from=a.encoder_from)
    print("SALM WER:", w)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="audioforge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("catalog")
    c.add_argument("action", choices=["fetch", "summary", "search"])
    c.add_argument("query", nargs="?")
    c.add_argument("--family"); c.add_argument("--encoder"); c.add_argument("--decoder")
    c.add_argument("--streaming", action="store_true")
    c.add_argument("--sort", default="downloads", choices=["downloads", "likes", "created", "id"])
    c.add_argument("--limit", type=int, default=50)
    c.set_defaults(fn=cmd_catalog)
    t = sub.add_parser("train"); t.add_argument("recipe"); t.add_argument("overrides", nargs="*")
    t.add_argument("-o", "--out"); t.set_defaults(fn=cmd_train)
    tr = sub.add_parser("transcribe"); tr.add_argument("model"); tr.add_argument("audio", nargs="+")
    tr.add_argument("--head"); tr.add_argument("--prompt"); tr.set_defaults(fn=cmd_transcribe)
    s = sub.add_parser("stream"); s.add_argument("model"); s.add_argument("audio"); s.add_argument("--head")
    s.add_argument("--chunk-ms", type=int, default=160); s.set_defaults(fn=cmd_stream)
    e = sub.add_parser("export"); e.add_argument("model"); e.add_argument("out"); e.set_defaults(fn=cmd_export)
    tc = sub.add_parser("train-codec"); tc.add_argument("-o", "--out", default="codec.pt")
    tc.add_argument("--steps", type=int, default=1000); tc.add_argument("--gan", action="store_true")
    tc.add_argument("--mel", action="store_true", help="train the MelCodec instead of the waveform codec")
    tc.set_defaults(fn=cmd_train_codec)
    ts = sub.add_parser("train-salm"); ts.add_argument("--steps", type=int, default=300)
    ts.add_argument("--encoder-from", help="trained .afm to initialize the speech encoder")
    ts.set_defaults(fn=cmd_train_salm)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
