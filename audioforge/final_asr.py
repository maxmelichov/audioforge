"""Per-turn final ASR for the hybrid front end (``serve --final-asr``; research/archive/HYBRID_ASR.md).

The streaming 115M model keeps every live head (partials, VAD, turn, speaker, enrollment); a second, larger offline
model transcribes each finished user turn once for the accurate final transcript. This module is that second model
behind a small asynchronous interface, so it never runs on the server's streaming worker thread:

    w = FinalASRWorker("tdt_v3", mode="process", threads=2, device="cpu")
    fut = w.submit(audio_16k_float32)       # concurrent.futures.Future -> FinalResult(text, compute_ms, rss_mb)
    w.close()

Specs: ``tdt_v3`` (nvidia/parakeet-tdt-0.6b-v3 from data/nemo/, imported by audioforge.nemo_import, greedy TDT),
any ``.nemo`` path that nemo_import can load, or ``fake:<ms>[:busy]`` (sleeps, or spins the CPU holding the GIL, for
``ms`` and returns "<N.NNs>" = the span's duration; tests).
Modes: ``process`` (default; a spawned child process with its own torch threads and its own GIL, so the TDT decode
loop never holds the server's GIL) or ``thread`` (one worker thread in the server process; shares the GIL and the
torch intra-op pool with the streaming loop). Jobs run one at a time, in submission order.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import peak_rss_mb
from .paths import DATA_ROOT

SR = 16000
ROOT = Path(__file__).resolve().parent.parent
# AUDIOFORGE_TDT_V3 (set by audioforge-serve from the models directory) overrides the checkout path
SPECS = {"tdt_v3": Path(os.environ.get("AUDIOFORGE_TDT_V3") or DATA_ROOT / "nemo" / "parakeet-tdt-0.6b-v3.nemo")}
NAMES = {"tdt_v3": "tdt_v3"}


@dataclass
class FinalResult:
    text: str
    compute_ms: float
    rss_mb: float | None = None


def source_name(spec: str) -> str:
    """The ``source`` tag of the finals this spec produces."""
    if spec in NAMES:
        return NAMES[spec]
    if spec.startswith("fake"):
        return "fake"
    return Path(spec).name.removesuffix(".nemo")


def load_transcriber(spec: str, threads: int | None = 2, device: str = "cpu"):
    """-> fn(audio float32 16 kHz) -> text."""
    if spec.startswith("fake"):  # fake:<ms>[:busy] - sleep (or spin the CPU in Python, holding the GIL) for ms
        parts = spec.split(":")
        ms = float(parts[1]) if len(parts) > 1 else 0.0
        busy = len(parts) > 2 and parts[2] == "busy"

        def fake(x):
            if busy:
                t_end = time.perf_counter() + ms / 1000
                while time.perf_counter() < t_end:
                    sum(i * i for i in range(1000))
            else:
                time.sleep(ms / 1000)
            return f"<{len(x) / SR:.2f}s>"
        return fake
    import torch
    if threads:
        torch.set_num_threads(threads)
    from .nemo_import import import_nemo
    path = SPECS.get(spec, spec)
    m = import_nemo(path).to(device).eval()
    head = m.primary

    def fn(x):
        with torch.inference_mode():
            a = torch.as_tensor(np.asarray(x, np.float32), device=device)[None]
            enc, elen = m.encode(a, torch.tensor([a.shape[1]], device=device))
            ids = m.heads[head].decode(enc, elen)[0]
        return m.tokenizer.decode(ids)
    fn(np.zeros(SR, np.float32))  # warm-up
    return fn


def _child(conn, spec, threads, device):
    fn = load_transcriber(spec, threads, device)
    conn.send(("ready", peak_rss_mb()))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        jid, audio = msg
        t0 = time.perf_counter()
        try:
            text, err = fn(audio), None
        except Exception as e:  # noqa: BLE001 - reported to the parent
            text, err = "", f"{type(e).__name__}: {e}"
        conn.send((jid, text, (time.perf_counter() - t0) * 1000, peak_rss_mb(), err))


class FinalASRWorker:
    def __init__(self, spec: str = "tdt_v3", mode: str = "process", threads: int | None = 2, device: str = "cpu"):
        if mode not in ("process", "thread"):
            raise ValueError(f"mode must be process|thread, not {mode!r}")
        self.spec, self.mode, self.device, self.threads = spec, mode, device, threads
        self.source = source_name(spec)
        self.rss_mb: float | None = None
        t0 = time.perf_counter()
        if mode == "thread":
            self._fn = load_transcriber(spec, threads, device)
            self._ex = ThreadPoolExecutor(1, thread_name_prefix="final-asr")
        else:
            ctx = mp.get_context("spawn")
            self._conn, child = ctx.Pipe()
            self._proc = ctx.Process(target=_child, args=(child, spec, threads, device), daemon=True)
            self._proc.start()
            tag, rss = self._conn.recv()  # blocks until the child has loaded the model
            assert tag == "ready", tag
            self.rss_mb = rss
            self._lock = threading.Lock()
            self._jobs: dict[int, Future] = {}
            self._next = 0
            self._reader = threading.Thread(target=self._read, name="final-asr-reader", daemon=True)
            self._reader.start()
        self.load_s = round(time.perf_counter() - t0, 2)

    @property
    def alive(self) -> bool:
        """False once the child process has exited (the server then falls back to the streaming final and may
        ``restart`` it)."""
        return self.mode == "thread" or self._proc.is_alive()

    def restart(self):
        """Replace a dead child process (or thread pool) with a fresh one; pending jobs of the old one fail."""
        self.close()
        self.__init__(self.spec, self.mode, self.threads, self.device)

    def submit(self, audio: np.ndarray) -> Future:
        x = np.ascontiguousarray(audio, dtype=np.float32)
        if self.mode == "process" and not self._proc.is_alive():
            raise RuntimeError("final-ASR worker process is not alive")
        if self.mode == "thread":
            def run():
                t0 = time.perf_counter()
                text = self._fn(x)
                return FinalResult(text, (time.perf_counter() - t0) * 1000, None)
            return self._ex.submit(run)
        fut: Future = Future()
        with self._lock:
            jid = self._next
            self._next += 1
            self._jobs[jid] = fut
            self._conn.send((jid, x))
        return fut

    def _read(self):
        while True:
            try:
                jid, text, ms, rss, err = self._conn.recv()
            except (EOFError, OSError):
                with self._lock:
                    for f in self._jobs.values():
                        if not f.done():
                            f.set_exception(RuntimeError("final-ASR worker process exited"))
                    self._jobs.clear()
                return
            self.rss_mb = rss
            with self._lock:
                fut = self._jobs.pop(jid, None)
            if fut is not None:
                if err:
                    fut.set_exception(RuntimeError(err))
                else:
                    fut.set_result(FinalResult(text, ms, rss))

    def close(self):
        if self.mode == "thread":
            self._ex.shutdown(wait=False, cancel_futures=True)
            return
        try:
            self._conn.send(None)
        except (OSError, BrokenPipeError):
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
