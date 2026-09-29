"""ASR on real meeting audio for the final report (research/FINAL_REPORT.md): WER + CPU RTF, same sets and normalisation.

Sets
  ami   AMI dev single-speaker segments, the 200-segment set of scripts/research/bench_yield_tokens.py `wer` stage
        (AMI(subset({"dev": 4})["dev"]).asr(1.0, 15.0), random.Random(0).sample(..., 200), sorted)
  libri LibriSpeech test-clean, first 200 utterances (data/librispeech/test-clean-first200.jsonl)

Systems (each run in its own process; torch/ctranslate2 at 2 CPU threads)
  served          runs/stage1_served.afm, RNNT head, encoder att_context [70, 1] = 160 ms chunks. m.transcribe uses the
                  masked offline forward, which equals chunk-by-chunk cache-aware streaming (tests/test_streaming.py).
  parakeet        runs/nemo_parakeet_ctc_0.6b.afm (NVIDIA parakeet-ctc-0.6b imported, offline, full context). ALONE.
  whisper_small   OpenAI Whisper small (244M) via faster-whisper / CTranslate2 (Systran/faster-whisper-small), int8,
                  greedy (beam 1), language en, no timestamps, no VAD filter.
  whisper_turbo   Whisper large-v3-turbo (809M) via faster-whisper (mobiuslabsgmbh/faster-whisper-large-v3-turbo), same.

Scoring: audioforge.teachers.normalize_text on ref and hyp (primary). Two diagnostics: `nofill` also drops filler words
(um, uh, mm, hmm, ...) from both sides; `whisper_norm` applies Whisper's EnglishTextNormalizer to both sides.
RTF = decode wall time / audio seconds (model load excluded), CPU, 2 threads.

Usage (resumable; each call <= --budget s; re-run until it prints "done"):
  PYTHONPATH=. .venv/bin/python scripts/research/final_asr.py prep
  PYTHONPATH=. .venv/bin/python scripts/research/final_asr.py run --system served --set ami
  PYTHONPATH=. .venv/bin/python scripts/research/final_asr.py report      # -> runs/final_asr.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORK = Path(os.environ.get("AUDIOFORGE_SCRATCH", "/Volumes/ExternalSSD/nvidia-audio-models/scratch")) / "final_asr"  # SSD scratch
OUT = ROOT / "runs" / "final_asr.json"
HUB = Path.home() / ".cache/huggingface/hub"
FILLERS = {"um", "uh", "mm", "hmm", "mhm", "uhm", "erm", "er", "ah", "eh", "oh", "huh", "mm hmm", "uh huh", "hm"}
PARAMS = {"served": "114.6M encoder + heads (shared)", "parakeet": "~600M", "whisper_small": "244M", "whisper_turbo": "809M"}


def _snap(repo: str) -> str:
    return str(next((HUB / f"models--{repo.replace('/', '--')}" / "snapshots").iterdir()))


def prep():
    WORK.mkdir(parents=True, exist_ok=True)
    from audioforge.data import load_wav
    from audioforge.datasets.ami import AMI, subset
    ds = AMI(subset({"dev": 4})["dev"], verbose=False)
    segs = ds.asr(1.0, 15.0)
    idx = sorted(random.Random(0).sample(range(len(segs)), min(200, len(segs))))
    np.savez(WORK / "ami.npz", **{f"a{i}": np.asarray(segs[j]["audio"], np.float32) for i, j in enumerate(idx)})
    (WORK / "ami_refs.json").write_text(json.dumps([segs[j]["text"] for j in idx]))
    lines = [json.loads(x) for x in (ROOT / "data/librispeech/test-clean-first200.jsonl").read_text().splitlines() if x.strip()][:200]
    np.savez(WORK / "libri.npz", **{f"a{i}": load_wav(x["audio_filepath"], 16000) for i, x in enumerate(lines)})
    (WORK / "libri_refs.json").write_text(json.dumps([x["text"] for x in lines]))
    print(f"prep: ami {len(idx)} of {len(segs)} segments, libri {len(lines)}; done", flush=True)


def load_set(name: str):
    z = np.load(WORK / f"{name}.npz")
    refs = json.loads((WORK / f"{name}_refs.json").read_text())
    return [z[f"a{i}"] for i in range(len(refs))], refs


def make_system(system: str):
    """-> (transcribe(list[np.ndarray]) -> list[str], batch size)."""
    if system in ("served", "parakeet"):
        import torch
        torch.set_num_threads(2)
        from audioforge.train import load_model
        ck = {"served": "runs/stage1_served.afm", "parakeet": "runs/nemo_parakeet_ctc_0.6b.afm"}[system]
        m = load_model(str(ROOT / ck), "cpu")
        m.eval()
        if system == "served":
            assert list(m.encoder.att_context_size) == [70, 1], m.encoder.att_context_size

            def fn(xs):
                with torch.inference_mode():
                    return m.transcribe(xs, head="rnnt")
        else:
            def fn(xs):
                with torch.inference_mode():
                    return m.transcribe(xs)
        return fn, 4
    from faster_whisper import WhisperModel
    repo = {"whisper_small": "Systran/faster-whisper-small", "whisper_turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}[system]
    wm = WhisperModel(_snap(repo), device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)

    def fn(xs):
        out = []
        for x in xs:
            segs, _ = wm.transcribe(x, language="en", beam_size=1, without_timestamps=True, vad_filter=False,
                                    condition_on_previous_text=False)
            out.append(" ".join(s.text.strip() for s in segs))
        return out
    return fn, 1


def run(system: str, set_name: str, budget: float):
    audios, refs = load_set(set_name)
    p = WORK / f"{system}_{set_name}.jsonl"
    done = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    if len(done) >= len(refs):
        print("done", flush=True)
        return
    t_load = time.time()
    fn, bs = make_system(system)
    print(f"{system} loaded in {time.time() - t_load:.1f}s", flush=True)
    if not done:  # warm-up on the first item (not timed, not stored)
        fn([audios[0][:16000 * 3]])
    t_start = time.time()
    with p.open("a") as f:
        i = len(done)
        while i < len(refs) and time.time() - t_start < budget:
            xs = audios[i:i + bs]
            t0 = time.perf_counter()
            hyps = fn(xs)
            dt = time.perf_counter() - t0
            for k, h in enumerate(hyps):
                f.write(json.dumps({"i": i + k, "hyp": h, "sec": dt / len(hyps), "audio_sec": len(xs[k]) / 16000}) + "\n")
            f.flush()
            i += len(xs)
    print(f"{system} {set_name}: {i}/{len(refs)}" + ("; done" if i >= len(refs) else " (re-run to continue)"), flush=True)


def _edits(r: str, h: str) -> tuple[int, int]:
    a, b = r.split(), h.split()
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)], len(a)


def score(refs, hyps, norm, n_boot=1000):
    e = np.array([_edits(norm(r), norm(h)) for r, h in zip(refs, hyps)], float)
    w = e[:, 0].sum() / max(1.0, e[:, 1].sum())
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(e), len(e))
        bs.append(e[k, 0].sum() / max(1.0, e[k, 1].sum()))
    return {"wer": round(float(w), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "errors": int(e[:, 0].sum()), "ref_words": int(e[:, 1].sum())}


def report():
    from audioforge.teachers import normalize_text as N
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    en = EnglishTextNormalizer(json.loads(Path(_snap("openai/whisper-tiny"), "normalizer.json").read_text()))

    def nofill(s):
        s = N(s)
        for f in ("mm hmm", "uh huh"):
            s = s.replace(f, " ")
        return " ".join(w for w in s.split() if w not in FILLERS)
    res = {"protocol": __doc__.split("Usage")[0].strip(), "threads": 2, "results": {}}
    for set_name in ("ami", "libri"):
        _, refs = load_set(set_name)
        for system in ("served", "parakeet", "whisper_small", "whisper_turbo"):
            p = WORK / f"{system}_{set_name}.jsonl"
            if not p.exists():
                continue
            rows = {r["i"]: r for r in map(json.loads, p.read_text().splitlines())}
            n = len(rows)
            if n < len(refs):
                res["results"][f"{system}/{set_name}"] = {"incomplete": f"{n}/{len(refs)}"}
                continue
            hyps = [rows[i]["hyp"] for i in range(len(refs))]
            sec, aud = sum(r["sec"] for r in rows.values()), sum(r["audio_sec"] for r in rows.values())
            res["results"][f"{system}/{set_name}"] = {
                "n": n, "audio_sec": round(aud, 1), "decode_sec": round(sec, 1), "rtf_cpu2": round(sec / aud, 4),
                "params": PARAMS[system], "wer_normalize_text": score(refs, hyps, N),
                "wer_nofill": score(refs, hyps, nofill), "wer_whisper_norm": score(refs, hyps, en),
                "examples": [{"ref": refs[i], "hyp": hyps[i]} for i in range(3)]}
    # paired utterance bootstrap: WER(served) - WER(other), same items, normalize_text
    res["paired_served_minus"] = {}
    for set_name in ("ami", "libri"):
        _, refs = load_set(set_name)
        hy = {}
        for system in ("served", "parakeet", "whisper_small", "whisper_turbo"):
            p = WORK / f"{system}_{set_name}.jsonl"
            if p.exists():
                rows = {r["i"]: r for r in map(json.loads, p.read_text().splitlines())}
                if len(rows) == len(refs):
                    hy[system] = np.array([_edits(N(refs[i]), N(rows[i]["hyp"])) for i in range(len(refs))], float)
        if "served" not in hy:
            continue
        rng = np.random.default_rng(0)
        ks = [rng.integers(0, len(refs), len(refs)) for _ in range(1000)]
        for o, e in hy.items():
            if o == "served":
                continue
            s_, d_ = hy["served"], e
            pt = s_[:, 0].sum() / s_[:, 1].sum() - d_[:, 0].sum() / d_[:, 1].sum()
            bs = [s_[k, 0].sum() / s_[k, 1].sum() - d_[k, 0].sum() / d_[k, 1].sum() for k in ks]
            res["paired_served_minus"][f"{o}/{set_name}"] = {"delta": round(float(pt), 4),
                "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    OUT.write_text(json.dumps(res, indent=1))
    for k, v in res["results"].items():
        print(k, {kk: v.get(kk) for kk in ("rtf_cpu2",)}, v.get("wer_normalize_text", v), flush=True)
    print(f"wrote {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["prep", "run", "report"])
    ap.add_argument("--system", choices=["served", "parakeet", "whisper_small", "whisper_turbo"])
    ap.add_argument("--set", default="ami", choices=["ami", "libri"])
    ap.add_argument("--budget", type=float, default=420.0)
    a = ap.parse_args()
    if a.stage == "prep":
        prep()
    elif a.stage == "run":
        run(a.system, a.set, a.budget)
    else:
        report()


if __name__ == "__main__":
    main()
