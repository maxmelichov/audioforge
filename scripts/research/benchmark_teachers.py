"""Benchmark NVIDIA teachers (and our students) on the first N LibriSpeech test-clean utterances.

    .venv/bin/python scripts/research/benchmark_teachers.py                         # all teachers, 200 utterances
    .venv/bin/python scripts/research/benchmark_teachers.py --teachers parakeet-ctc-0.6b --n 50
    .venv/bin/python scripts/research/benchmark_teachers.py --teachers none --student runs/x.afm [--student runs/y.afm]

Utterances: ids sorted as strings, first N; references lowercased. The list is written once to
data/librispeech/test-clean-first{N}.jsonl (NeMo keys; audio_filepath points at the FLAC).
WER: both sides go through audioforge.teachers.normalize_text (lowercase, punctuation stripped).
RTFx: total audio seconds / wall seconds of transcription. Model loading and one warm-up batch are
excluded; the device is synchronized before the clock stops. Peak memory: highest MPS driver
allocation (CUDA: max_memory_allocated) seen while transcribing (weights included); on CPU the process max RSS. Each model runs in
its own subprocess (unless --in-process) so memory numbers don't carry over between models.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from audioforge.metrics import wer  # noqa: E402
from audioforge.teachers import TEACHERS, Teacher, load_audio, normalize_text  # noqa: E402

LS = ROOT / "data" / "librispeech" / "LibriSpeech" / "test-clean"
MARK = "@@result "


def utterances(n: int) -> list[dict]:
    out = ROOT / "data" / "librispeech" / f"test-clean-first{n}.jsonl"
    if out.exists():
        return [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    if not LS.exists():
        sys.exit(f"missing {LS}: download https://www.openslr.org/resources/12/test-clean.tar.gz and extract "
                 f"it into {LS.parent.parent}")
    trans = {}
    for f in LS.glob("*/*/*.trans.txt"):
        for line in f.read_text().splitlines():
            k, text = line.split(" ", 1)
            trans[k] = text.lower()
    rows = []
    for k in sorted(trans)[:n]:
        spk, chap, _ = k.split("-")
        path = LS / spk / chap / f"{k}.flac"
        rows.append({"id": k, "audio_filepath": str(path.relative_to(ROOT)),
                     "duration": round(len(load_audio(path)) / 16000, 3), "text": trans[k]})
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


class PeakMemory:
    """Poll MPS driver memory in a thread (torch.mps has no max_memory_allocated)."""

    def __init__(self, device: torch.device):
        self.device, self.peak, self._stop = device, 0, threading.Event()

    def _poll(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, torch.mps.driver_allocated_memory())
            time.sleep(0.01)

    def __enter__(self):
        if self.device.type == "mps":
            self._t = threading.Thread(target=self._poll, daemon=True)
            self._t.start()
        elif self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.device.type == "mps":
            self._t.join()
        elif self.device.type == "cuda":
            self.peak = torch.cuda.max_memory_allocated()
        else:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.peak = rss if sys.platform == "darwin" else rss * 1024  # bytes on macOS, KiB on Linux

    @property
    def gb(self) -> float:
        return self.peak / 2 ** 30


def sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, audios, device, batch_size):
    """Warm up on one batch, then time fn over everything."""
    fn(audios[:batch_size])
    sync(device)
    with PeakMemory(device) as mem:
        t0 = time.perf_counter()
        hyps = fn(audios)
        sync(device)
        wall = time.perf_counter() - t0
    return hyps, wall, mem.gb


def free(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def student_fn(model, batch_size):
    def fn(audios):  # length-sorted batches, like Teacher.transcribe
        order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
        out = [None] * len(audios)
        for s in range(0, len(order), batch_size):
            idx = order[s: s + batch_size]
            for i, h in zip(idx, model.transcribe([audios[i] for i in idx])):
                out[i] = h
        return out
    return fn


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--teachers", default=",".join(TEACHERS), help="comma list of TEACHERS keys / repo ids, or 'none'")
    p.add_argument("--student", action="append", default=[], help="our .afm model (repeatable)")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--save-hyps", default=None, help="directory for per-model hypothesis JSONL")
    p.add_argument("--json", default=None, help="write the results table as JSON here")
    p.add_argument("--in-process", action="store_true", help="run all models in this process")
    p.add_argument("--no-table", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()

    rows = utterances(a.n)
    total = sum(r["duration"] for r in rows)
    names = [t for t in a.teachers.split(",") if t and t != "none"]
    jobs = [("teacher", n) for n in names] + [("student", s) for s in a.student]
    if a.in_process or len(jobs) == 1:
        results = run_jobs(jobs, rows, a)
    else:
        print(f"{len(rows)} utterances, {total / 60:.1f} min of audio ({rows[0]['id']} .. {rows[-1]['id']})", flush=True)
        results = [run_isolated(kind, name, a) for kind, name in jobs]
    print_table(results, rows, total, a)


def run_isolated(kind, name, a) -> dict:
    """Run one model in a child process; its result comes back as a marked JSON line on stdout."""
    cmd = [sys.executable, __file__, "--n", str(a.n), "--device", a.device, "--dtype", a.dtype,
           "--batch-size", str(a.batch_size), "--in-process", "--no-table"]
    cmd += ["--teachers", name] if kind == "teacher" else ["--teachers", "none", "--student", name]
    if a.save_hyps:
        cmd += ["--save-hyps", a.save_hyps]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    result = dict(model=name, kind=kind, error="child process produced no result")
    for line in proc.stdout:
        if line.startswith(MARK):
            result = json.loads(line[len(MARK):])
        else:
            print(line, end="", flush=True)
    proc.wait()
    return result


def run_jobs(jobs, rows, a) -> list[dict]:
    audios = [load_audio(ROOT / r["audio_filepath"]) for r in rows]
    refs = [normalize_text(r["text"]) for r in rows]
    total = sum(len(x) for x in audios) / 16000
    if not a.no_table:
        print(f"{len(rows)} utterances, {total / 60:.1f} min of audio ({rows[0]['id']} .. {rows[-1]['id']})", flush=True)
    results = []
    for kind, name in jobs:
        t0 = time.time()
        try:
            if kind == "teacher":
                t = Teacher.from_name(name, a.device, a.dtype)
                dev, fn = t.device, (lambda xs, t=t: t.transcribe(xs, batch_size=a.batch_size))
                info = dict(decoder=t.kind, params=t.num_params, dtype=str(t.model.dtype).replace("torch.", ""),
                            license=t.license)
            else:
                from audioforge.train import load_model, pick_device
                dev = pick_device(a.device)
                t = load_model(name, dev)
                fn = student_fn(t, a.batch_size)
                info = dict(decoder=t.primary, params=t.num_params(), dtype="float32", license="ours")
            load_sec = time.time() - t0
            hyps, wall, mem = timed(fn, audios, dev, a.batch_size)
        except Exception as e:  # report and keep going: one broken model shouldn't sink the table
            print(f"!! {name}: {type(e).__name__}: {e}", flush=True)
            results.append(dict(model=name, kind=kind, error=f"{type(e).__name__}: {e}"))
            if a.no_table:  # child of run_isolated
                print(MARK + json.dumps(results[-1]), flush=True)
            continue
        norm = [normalize_text(h) for h in hyps]
        r = dict(model=Path(name).name if kind == "student" else name, kind=kind, **info, device=dev.type,
                 wer=100 * wer(refs, norm), rtfx=total / wall, wall_sec=wall, load_sec=load_sec, peak_mem_gb=mem,
                 n=len(rows), audio_sec=total, batch_size=a.batch_size)
        results.append(r)
        print(f"   {r['model']}: WER {r['wer']:.2f}%  RTFx {r['rtfx']:.1f}  ({wall:.1f}s wall, "
              f"peak {mem:.2f} GB, load {load_sec:.1f}s)", flush=True)
        if a.save_hyps:
            d = Path(a.save_hyps)
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{r['model'].replace('/', '_')}.jsonl", "w") as f:
                for row, h, n_ in zip(rows, hyps, norm):
                    f.write(json.dumps({"id": row["id"], "ref": row["text"], "hyp": h, "hyp_norm": n_}) + "\n")
        if a.no_table:
            print(MARK + json.dumps(r), flush=True)
        del t, fn
        free(dev)
    return results


def print_table(results, rows, total, a):
    if a.no_table:
        return
    print(f"\nLibriSpeech test-clean, first {len(rows)} utterances ({total / 60:.1f} min), greedy decoding, "
          f"batch {a.batch_size}\n")
    print("| model | kind | decoder | params | device | dtype | WER % | RTFx | peak mem GB | license |")
    print("|---|---|---|---:|---|---|---:|---:|---:|---|")
    for r in results:
        if "error" in r:
            print(f"| {r['model']} | {r['kind']} | – | – | – | – | failed: {r['error'][:80]} | – | – | – |")
        else:
            print(f"| {r['model']} | {r['kind']} | {r['decoder']} | {r['params'] / 1e6:.0f}M | {r['device']} | "
                  f"{r['dtype']} | {r['wer']:.2f} | {r['rtfx']:.1f} | {r['peak_mem_gb']:.2f} | {r['license']} |")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
