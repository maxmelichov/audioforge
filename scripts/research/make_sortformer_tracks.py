"""Cache NVIDIA streaming Sortformer v2 activity tracks for every AMI / ICSI turn example (train + dev), so the turn
head can TRAIN on real diarizer output (audioforge/datasets/ext_tracks.py; research/recipes/stage1_turn_on_sortformer.yaml).

--dataset ami (default) | icsi: ICSI (audioforge/datasets/icsi.py) subclasses ami.AMI, so its turn windows and keys
come from the same code (its 12 / 2 default train / dev meetings, untimed-zone and >4-speaker windows dropped exactly
as icsi.recipe_data drops them); its cache is data/icsi/cache/sortformer/<split>/ (ext_tracks.cache_dir(dataset=)).
  TMPDIR=<scratch> .venv/bin/python scripts/research/make_sortformer_tracks.py --dataset icsi --split train --track-source offline

For each turn window (built exactly as ami.recipe_data does: train = 12 default meetings, window_sec 16; dev =
4 default meetings, window_sec 20) the external diarizer (runs/nemo_sortformer_v2.afm, its own 128-mel
preprocessor + NEST encoder + Sortformer head) is run and its full (T, 4) sigmoid output is saved to
data/ami/cache/sortformer/<split>/<key>.npy (key = ext_tracks.example_key: meeting + start ms + samples).

  --track-source offline  one head pass over the whole window (batched, fast: ~9 s / 64 windows on 2 threads)
  --track-source stream   StreamingDiarizer, window mode, card low-latency config (chunk 6, right context 7,
                          FIFO 188, update 144, cache 188, encoder left context 188) - the same procedure as
                          scripts/research/eval_stage1.py; slow (~2.4 s / window) -> <key>.stream.npy

End-of-file flush fix (research/EOT_BENCH_V2.md, "flush artifact"): fed only the window audio, the diarizer flushes
at the window end with a shrinking right context, so the last ~13 frames (exactly the post-turn-end frames every turn
decision is scored / trained on) differ from what a live stream computes (|dp| up to ~0.6). Streaming tracks are now
computed on the window audio plus PAD_FRAMES = C + R + 1 frames (1.12 s) beyond the window end and cropped to the
window's T frames, so every window frame gets its full streaming right context:
  --pad-mode audio    (default) the real meeting audio that follows the window (a live stream has it: the right
                      context is part of the diarizer's latency); a window ending within 1.12 s of the meeting end
                      gets what is left, so it flushes where the live stream also ends: at the end of the meeting
  --pad-mode silence  zeros instead of the meeting audio (measured on 90 windows it is NOT a good stand-in: its
                      last-16-frame |dp| vs real audio is as large as the flush's - use it only without meeting audio)
  --pad-mode none     the pre-fix v1 procedure (flush inside the window; provenance / tests only)
With pad-mode audio the window's frames are bit-identical to a stream over the whole meeting's audio from the window
start (tests/test_track_flush.py). --regen-v1 migrates an existing cache: each old <key>.stream.npy is kept as
<key>.stream_v1.npy and recomputed with the fix (resumable like everything else).

Resumable: existing tracks are skipped; the run stops cleanly after --budget seconds (re-run to continue).
One python process at a time; TMPDIR must be outside /tmp (load_model extracts the archive via tempfile):
  TMPDIR=<scratch> .venv/bin/python scripts/research/make_sortformer_tracks.py --split train --track-source offline
  TMPDIR=<scratch> .venv/bin/python scripts/research/make_sortformer_tracks.py --split dev --track-source stream --budget 540

--device mps runs the diarizer on the Apple GPU (the stream source is ~10x cheaper there; tracks are written as
float32 CPU arrays, identical layout). PYTORCH_MPS_HIGH_WATERMARK_RATIO defaults to 0.5 and the low ratio to 0.4
(set before torch loads; the low ratio must not exceed the high one).
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 TMPDIR=<scratch> .venv/bin/python scripts/research/make_sortformer_tracks.py \
      --split train --track-source stream --device mps --budget 540       # repeat until "3274/3274 tracks cached"
--trail-sec X (default 2 = the library default): build the turn windows with an X s post-end trail (still stopped at
the primary's next turn / the meeting end) and cache them in <split>_trail<X>/ (ext_tracks.split_dirname), never in
the default <split>/ cache. The 6 s set of research/recipes/stage1_turn_v3_trail6.yaml (window 20 s so turn + 6 s fit):
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 TMPDIR=<scratch> .venv/bin/python \
      scripts/research/make_sortformer_tracks.py --split train --track-source stream --device mps --trail-sec 6 --window-sec 20
--verify-cpu N (no writes): recompute N cached tracks (evenly spread) of --track-source on the CPU and print the max
abs difference to the cached ones (checks an MPS-generated cache against the CPU reference).
  TMPDIR=<scratch> .venv/bin/python scripts/research/make_sortformer_tracks.py --split train --track-source stream --verify-cpu 6
--diar-config NAME (stream source; default low_latency = the card's 1.04 s row, unchanged): another
audioforge.streaming_diar.SORTFORMER_PRESETS entry, e.g. low_latency_032 = the card's 0.32 s "ultra low latency" row
(chunk 3, right context 1, FIFO 188, update 144, cache 188). Its padding is that config's C + R + 1 frames and its
tracks go to <split>[_trail<x>]_ll032/ (DIAR_CONFIG_SUFFIX), never the default cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")  # before torch initialises the MPS allocator
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO",          # must be <= the high ratio (default 1.4)
                      str(min(0.4, 0.8 * float(os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audioforge.data import Collate  # noqa: E402
from audioforge.datasets import ext_tracks as xt  # noqa: E402
from audioforge.datasets import icsi  # noqa: E402
from audioforge.datasets.ami import AMI, DEFAULT_MEETINGS  # noqa: E402

# must equal scripts/research/eval_stage1.py SORTFORMER_LOW_LATENCY / SORTFORMER_ENC_LEFT (checked in tests)
SORTFORMER_LOW_LATENCY = dict(chunk_len=6, chunk_right_context=7, fifo_len=188, spkcache_update_period=144,
                              spkcache_len=188)
SORTFORMER_ENC_LEFT = 188
DIAR_CONFIGS = ("low_latency", "low_latency_032")  # audioforge.streaming_diar.SORTFORMER_PRESETS names
DIAR_CONFIG_SUFFIX = {"low_latency": "", "low_latency_032": "_ll032"}  # cache dir suffix (default: none)
WINDOW_SEC = {"train": 16.0, "dev": 20.0, "eval": 20.0}  # recipe_data: stage-1 train mix 16 s, val default 20 s


def default_meetings(dataset: str, split: str) -> list[str]:
    """The recipe_data default meetings of a split (ami.DEFAULT_MEETINGS / icsi.DEFAULT_MEETINGS)."""
    return list((icsi.DEFAULT_MEETINGS if dataset == "icsi" else DEFAULT_MEETINGS)[split])


def turn_windows(split: str, window_sec: float, meetings=None, dataset: str = "ami", return_ds: bool = False,
                 trail_sec: float = xt.DEFAULT_TRAIL_SEC):
    """The turn examples of ami.recipe_data / icsi.recipe_data (default label rules and lead; ``trail_sec`` as the
    recipe's); with return_ds also the dataset object (its meeting audio gives the pad-mode audio tails)."""
    cls = icsi.ICSI if dataset == "icsi" else AMI
    ds = cls(meetings or default_meetings(dataset, split), verbose=False)
    exs = ds.turn_examples(window_sec, trail_sec=trail_sec)
    return (exs, ds) if return_ds else exs


FRAME_SAMPLES = 1280  # 16 kHz samples per 80 ms diarizer frame
PAD_MODES = ("audio", "silence", "none")
# frames fed beyond the window end: the chunk holding the last window frame ends <= C - 1 frames after it and then
# needs R right-context frames; + 1 frame of margin (the same padding as eval_stage1.v2_tracks)
PAD_FRAMES = SORTFORMER_LOW_LATENCY["chunk_len"] + SORTFORMER_LOW_LATENCY["chunk_right_context"] + 1


def diar_config(name: str = "low_latency") -> dict:
    """The StreamingDiarizer config of a --diar-config name (low_latency == SORTFORMER_LOW_LATENCY)."""
    from audioforge.streaming_diar import SORTFORMER_PRESETS
    assert name in DIAR_CONFIGS, name
    return dict(SORTFORMER_PRESETS[name])


def pad_frames(cfg: dict | None = None) -> int:
    """Frames fed beyond the window end for a config: C + R + 1 (PAD_FRAMES for the default)."""
    cfg = cfg or SORTFORMER_LOW_LATENCY
    return cfg["chunk_len"] + cfg["chunk_right_context"] + 1


def stream_track(dm, dname: str, audio, cfg: dict | None = None) -> np.ndarray:
    """(T, S) float32 streaming track of one window (StreamingDiarizer, window mode, card low-latency config or
    ``cfg``)."""
    from audioforge.streaming_diar import StreamingDiarizer
    sd = StreamingDiarizer(dm, diar_head=dname, mode="window", enc_left_context=SORTFORMER_ENC_LEFT,
                           **(cfg or SORTFORMER_LOW_LATENCY))
    sd.feed(np.asarray(audio, np.float32), final=True)
    return sd.all_probs.float().cpu().numpy()


def pad_tail(audio, tail=None, pad_mode: str = "audio", n_frames: int = PAD_FRAMES) -> np.ndarray:
    """The samples fed after the window: pad-mode audio = the following real audio ``tail`` (up to PAD_FRAMES *
    FRAME_SAMPLES samples; shorter at the meeting end, where the live stream ends too - no silence is invented);
    pad-mode silence = PAD_FRAMES * FRAME_SAMPLES zeros."""
    assert pad_mode in ("audio", "silence"), pad_mode
    need = n_frames * FRAME_SAMPLES
    if pad_mode == "silence":
        return np.zeros(need, np.float32)
    assert tail is not None, "pad-mode audio needs the meeting audio after the window"
    return np.asarray(tail[:need], np.float32)


def stream_track_padded(dm, dname: str, audio, tail=None, pad_mode: str = "audio", cfg: dict | None = None) -> np.ndarray:
    """(T, S) streaming track of a window without an end-of-file flush inside it: stream_track over the window audio plus
    pad_tail(...), cropped to the window's T = n_frames(len(audio)) frames. pad_mode 'none' = plain stream_track."""
    run = (lambda x: stream_track(dm, dname, x)) if cfg is None else (lambda x: stream_track(dm, dname, x, cfg))
    if pad_mode == "none":
        return run(audio)
    from audioforge.data import ToneLanguage
    T = ToneLanguage.n_frames(len(audio))
    p = run(np.concatenate([np.asarray(audio, np.float32), pad_tail(audio, tail, pad_mode, pad_frames(cfg))]))
    assert len(p) >= T, (len(p), T)
    return p[:T]


def meeting_tail(ds, ex: dict, n: int = PAD_FRAMES * FRAME_SAMPLES) -> np.ndarray:
    """The (up to) ``n`` meeting samples right after a window (sample-exact continuation of ds._clip)."""
    from audioforge.datasets.ami import SR
    x = ds._audio[ex["meeting"]]
    i = int(round(float(ex["start"]) * SR)) + len(ex["audio"])
    return np.asarray(x[i: i + n], np.float32)


@torch.no_grad()
def offline_tracks(dm, dname: str, audios: list, device="cpu") -> list[np.ndarray]:
    """(T_i, S) float32 offline tracks (one head pass per window) of a batch of windows."""
    b = Collate(None)([{"audio": a} for a in audios])
    enc, elen = dm.encode(b["audio"].to(device), b["audio_len"].to(device))
    p = dm.heads[dname](enc, elen).sigmoid().float().cpu()
    return [p[j, : int(elen[j])].numpy() for j in range(len(audios))]


@torch.no_grad()
def verify_cpu(a, exs, keys, d, ds=None) -> dict:
    """Recompute ``a.verify_cpu`` cached tracks (evenly spread) of ``a.track_source`` on the CPU; max abs difference."""
    from audioforge.heads.turn import _diar_name
    from audioforge.train import load_model
    have = [i for i, k in enumerate(keys) if xt.track_path(d, k, a.track_source).exists()]
    have = have[:: max(1, len(have) // a.verify_cpu)][: a.verify_cpu]  # evenly spread over the cached windows
    dm = load_model(a.diar_ckpt, "cpu")
    dname = _diar_name(dm)
    diffs = []
    for i in have:
        ref = (stream_track_padded(dm, dname, exs[i]["audio"], meeting_tail(ds, exs[i]) if ds is not None else None,
                                   a.pad_mode, None if getattr(a, "diar_config", "low_latency") == "low_latency"
                                   else diar_config(a.diar_config))
               if a.track_source == "stream"
               else offline_tracks(dm, dname, [exs[i]["audio"]])[0])
        got = np.load(xt.track_path(d, keys[i], a.track_source))
        assert got.shape == ref.shape, (keys[i], got.shape, ref.shape)
        diffs.append(float(np.abs(got - ref).max()))
        print(f"  {keys[i]} T={len(ref)} max|cached - cpu| = {diffs[-1]:.2e}", flush=True)
    res = {"n": len(diffs), "max_abs_diff": max(diffs) if diffs else None,
           "mean_max_abs_diff": float(np.mean(diffs)) if diffs else None}
    print(json.dumps({"verify_cpu": a.track_source, **res}), flush=True)
    return res


def _save(path: Path, arr: np.ndarray):
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, np.ascontiguousarray(arr, np.float32))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ami", choices=xt.DATASETS)
    ap.add_argument("--split", default="train", choices=["train", "dev", "eval"])
    ap.add_argument("--track-source", default="offline", choices=xt.SOURCES)
    ap.add_argument("--diar-ckpt", default="runs/nemo_sortformer_v2.afm")
    ap.add_argument("--window-sec", type=float, default=None, help="default: 16 (train) / 20 (dev, eval)")
    ap.add_argument("--trail-sec", type=float, default=xt.DEFAULT_TRAIL_SEC,
                    help="post-end trail of the turn windows (!= 2: own cache dir <split>_trail<x>)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N examples (in example order)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--budget", type=float, default=540.0, help="stop after this many seconds (resume later)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda: where the diarizer runs (tracks saved as CPU "
                                                     "float32 either way)")
    ap.add_argument("--pad-mode", default="audio", choices=PAD_MODES,
                    help="stream source: what is fed after the window end before cropping (end-of-file flush fix): "
                         "audio = the meeting's following audio (default), silence = zeros, none = pre-fix v1")
    ap.add_argument("--regen-v1", action="store_true",
                    help="stream source: keep each existing <key>.stream.npy as <key>.stream_v1.npy and recompute it "
                         "with --pad-mode (resumable)")
    ap.add_argument("--diar-config", default="low_latency", choices=DIAR_CONFIGS,
                    help="stream source: the StreamingDiarizer card config (low_latency = 1.04 s, default; "
                         "low_latency_032 = 0.32 s, cached in <split>_ll032/)")
    ap.add_argument("--verify-cpu", type=int, default=0,
                    help="no writes: recompute N cached tracks (evenly spread) on the CPU, print the max abs difference")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    ws = a.window_sec or WINDOW_SEC[a.split]
    exs, ds = turn_windows(a.split, ws, dataset=a.dataset, return_ds=True, trail_sec=a.trail_sec)
    n_all = len(exs)
    if a.limit:
        exs = exs[: a.limit]
    # the trail kwarg only for a non-default window set: the default call (and its signature) stays as before
    tkw = {} if xt.split_dirname(a.split, a.trail_sec) == a.split else {"trail_sec": a.trail_sec}
    d = xt.cache_dir(None, a.split, a.dataset, **tkw)
    if a.diar_config != "low_latency":  # a non-default diarizer config never writes into the default cache
        assert a.track_source == "stream", "--diar-config applies to the stream source"
        d = d.with_name(d.name + DIAR_CONFIG_SUFFIX[a.diar_config])
    cfg = diar_config(a.diar_config)
    d.mkdir(parents=True, exist_ok=True)
    keys = [xt.example_key(e) for e in exs]
    assert len(set(keys)) == len(keys), "example keys are not unique"
    if a.verify_cpu:
        verify_cpu(a, exs, keys, d, ds)
        return
    v1 = lambda k: d / f"{k}.stream_v1.npy"  # noqa: E731  (pre-fix track kept for provenance by --regen-v1)
    if a.regen_v1:
        assert a.track_source == "stream" and a.pad_mode != "none", "--regen-v1 recomputes streaming tracks with a fix"
        todo = [i for i, k in enumerate(keys) if not (xt.track_path(d, k, "stream").exists() and v1(k).exists())]
        bare = [keys[i] for i in todo if not xt.track_path(d, keys[i], "stream").exists() and not v1(keys[i]).exists()]
        assert not bare, f"{len(bare)} windows have no track to migrate (e.g. {bare[0]}): run without --regen-v1 first"
    else:
        todo = [i for i, k in enumerate(keys) if not xt.track_path(d, k, a.track_source).exists()]
    print(f"[{a.dataset} {a.split}/{a.track_source}] {len(exs)} turn windows (window_sec {ws}, trail_sec "
          f"{a.trail_sec:g}) -> {d}, {len(todo)} to do "
          f"({time.time() - t0:.0f}s)", flush=True)
    stopped = False
    if todo:
        from audioforge.heads.turn import _diar_name
        from audioforge.train import load_model
        dm = load_model(a.diar_ckpt, a.device)
        dname = _diar_name(dm)
        print(f"  loaded {a.diar_ckpt} head={dname} n_mels={dm.preprocessor.n_mels} on {a.device} "
              f"({time.time() - t0:.0f}s)", flush=True)
        done = 0
        with torch.no_grad():
            if a.track_source == "offline":
                for s in range(0, len(todo), a.batch_size):
                    idx = todo[s: s + a.batch_size]
                    for i, p in zip(idx, offline_tracks(dm, dname, [exs[i]["audio"] for i in idx], a.device)):
                        _save(xt.track_path(d, keys[i], "offline"), p)
                    done += len(idx)
                    if time.time() - t0 > a.budget and done < len(todo):
                        stopped = True
                        break
            else:
                for i in todo:
                    q = xt.track_path(d, keys[i], "stream")
                    if a.regen_v1 and q.exists() and not v1(keys[i]).exists():
                        os.replace(q, v1(keys[i]))
                    tail = meeting_tail(ds, exs[i]) if a.pad_mode == "audio" else None
                    _save(q, stream_track_padded(dm, dname, exs[i]["audio"], tail, a.pad_mode,
                                                 None if a.diar_config == "low_latency" else cfg))
                    done += 1
                    if time.time() - t0 > a.budget and done < len(todo):
                        stopped = True
                        break
        print(f"  wrote {done} tracks in {time.time() - t0:.0f}s", flush=True)
    # manifest (merged across runs / sources)
    man = xt.manifest(None, a.split, directory=d, dataset=a.dataset, **tkw) if a.diar_config != "low_latency" \
        else xt.manifest(None, a.split, dataset=a.dataset, **tkw)
    flush_fix = man.get("stream", {}).get("flush_fix")  # recorded only by a stream run that wrote / migrated tracks
    if a.track_source == "stream" and (todo or a.regen_v1):
        flush_fix = None if a.pad_mode == "none" else {
            "pad_mode": a.pad_mode, "pad_frames": pad_frames(cfg), "regen_v1": bool(a.regen_v1 or man.get("stream_v1")),
            "what": "window audio + pad frames beyond the end (the following meeting audio; at the meeting end what is left), cropped "
                    "to T: no end-of-file flush inside the window (research/EOT_BENCH_V2.md)"}
    have = {s: [k for k in keys if xt.track_path(d, k, s).exists()] for s in xt.SOURCES}
    man.update({"dataset": a.dataset, "diar_ckpt": a.diar_ckpt, "split": a.split, "window_sec": ws, "trail_sec": a.trail_sec,
                "meetings": default_meetings(a.dataset, a.split), "n_examples": n_all,
                "key": "meeting_startms_nsamples (audioforge.datasets.ext_tracks.example_key)",
                "offline": {"n": len(have["offline"]), "what": "one Sortformer head pass over the window, sigmoid (T,4)"},
                "stream": {"n": len(have["stream"]), "what": "StreamingDiarizer window mode, card low latency"
                           + ("" if a.diar_config == "low_latency" else f" ({a.diar_config})"),
                           "config": dict(cfg, enc_left_context=SORTFORMER_ENC_LEFT),
                           "input_buffer_ms": (cfg["chunk_len"] + cfg["chunk_right_context"]) * 80.0,
                           "flush_fix": flush_fix}})
    n_v1 = sum(v1(k).exists() for k in keys)
    if n_v1:
        man["stream_v1"] = {"n": n_v1, "file": "<key>.stream_v1.npy",
                            "what": "pre-fix streaming tracks (window audio only: end-of-file flush in the last "
                                    "~13 frames); kept for provenance"}
    man.setdefault("devices", {})[a.track_source] = sorted(set(man.get("devices", {}).get(a.track_source, [])) | {a.device})
    man.setdefault("seconds", {})
    man["seconds"][a.track_source] = round(man["seconds"].get(a.track_source, 0.0) + time.time() - t0, 1)
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    msg = f"[{a.dataset} {a.split}/{a.track_source}] {len(have[a.track_source])}/{len(exs)} tracks cached in {d}"
    if stopped:
        msg += f"; budget {a.budget:.0f}s reached - re-run the same command to resume"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
