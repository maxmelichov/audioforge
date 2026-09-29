"""Hybrid front end, offline part (research/HYBRID_ASR.md): NVIDIA Parakeet-TDT 0.6B v3 as the per-turn final ASR.

Model: data/nemo/parakeet-tdt-0.6b-v3.nemo imported with audioforge.nemo_import (TDT head, greedy decoding), loaded
straight from the .nemo (memory-mapped; no .afm copy). Never loaded in the same process as parakeet-ctc-0.6b or the
nemotron streaming 0.6B (machine rules); the nemotron stage runs alone.

Sets (the scripts/research/final_asr.py protocol; its prepared audio and the other systems' hypotheses are reused)
  ami      AMI dev single-speaker segments, 200 (final_asr `prep`)
  libri    LibriSpeech test-clean, first 200 utterances
  fleurs   FLEURS test, the 17 languages x 150 utterances of research/LID.md (scripts/research/lid_data.py), full utterances;
           per language WER (Whisper BasicTextNormalizer; EnglishTextNormalizer for en) and CER (zh, ja: no spaces)
  fleurs2s the same utterances, 2 s from the Silero speech onset (lid.clip), for language ID at 2 s
Every item is transcribed alone (batch 1), CPU, 2 threads, fp32, greedy TDT. For fleurs / fleurs2s the mean-pooled
output of every encoder block is stored too (language-ID probe, `lidprobe`).

Stages (each process <= --budget s, resumable; re-run until it prints "done")
  run --set ami|libri|fleurs|fleurs2s [--lang xx] [--device cpu|mps]
  lidfeat --split train [--device mps]  pooled block features of FLEURS train 2 s / 5 s clips (for the probe)
  lidprobe                               per-block logistic-regression LID on the pooled features
  timing [--device cpu|mps]              per-turn latency (1/3/5/10 s turns, p50/p95), RTF, peak RSS, dtypes, compile
  nemotron --device mps|cpu              the nemotron streaming 0.6B at [70,1]: AMI-200 WER (masked offline = streaming)
                                         and StreamingSession RTF (separate process, never with TDT v3)
  report                                 -> runs/hybrid_asr.json (merges; the live section is written by hybrid_live.py)
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
from audioforge.metrics import pct  # noqa: E402
SCRATCH = Path(os.environ.get("AUDIOFORGE_SCRATCH", "/Volumes/ExternalSSD/nvidia-audio-models/scratch"))  # machine rule: scratch on the SSD
WORK = SCRATCH / "hybrid_asr"
FA_WORK = SCRATCH / "final_asr"
OUT = ROOT / "runs" / "hybrid_asr.json"
TDT_NEMO = ROOT / "data/nemo/parakeet-tdt-0.6b-v3.nemo"
NEMOTRON = ROOT / "runs/nemo_nemotron_speech_streaming_en_0.6b.afm"
SR = 16000
# the card's 25 languages (research/raw/cards/parakeet-tdt-0.6b-v3.md "Supported Languages")
V3_LANGS = ["bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt",
            "ro", "sk", "sl", "es", "sv", "ru", "uk"]
# the card's FLEURS WER (full test sets, PnC removed), for the languages of our FLEURS-17 subset
CARD_FLEURS = {"en": 4.85, "es": 3.45, "fr": 5.15, "de": 5.04, "it": 3.00, "nl": 7.48, "pl": 7.31, "pt": 4.76,
               "ru": 5.51, "uk": 6.79}
CARD = {"libri_test_clean": 1.93, "ami_test": 11.31}
CER_LANGS = ("zh", "ja")
SYSTEMS_FA = ("served", "parakeet", "whisper_small", "whisper_turbo")


def _json() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def _merge(key: str, rec):
    res = _json()
    res[key] = rec
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2 ** 20 if sys.platform == "darwin" else r / 1024, 1)


def load_tdt(device: str = "cpu"):
    import torch
    torch.set_num_threads(2)
    from audioforge.nemo_import import import_nemo
    m = import_nemo(TDT_NEMO)
    return m.to(device).eval()


TAG_LANGS = sorted(set(V3_LANGS) | {"en", "he", "ar", "ru", "es", "fr", "de", "pt", "it", "nl", "pl", "uk", "tr", "fa",
                                    "hi", "zh", "ja"})


def _tag_ids(m) -> tuple[list[int], list[int]]:
    """(ids of <|xx|> for TAG_LANGS, ids of every <|xx|> language tag) in the v3 vocabulary (Canary's tokenizer)."""
    import re
    sp = m.tokenizer.sp
    allp = [i for i in range(m.tokenizer.vocab_size) if re.fullmatch(r"<\|[a-z]{2,3}\|>", sp.id_to_piece(i))
            and sp.id_to_piece(i) not in ("<|pnc|>", "<|itn|>")]
    return [sp.piece_to_id(f"<|{c}|>") for c in TAG_LANGS], allp


def tdt_transcribe(m, x: np.ndarray, pooled: bool = False):
    """One item -> text (and, with ``pooled``, the per-block mean-pooled encoder output (L, D) fp16 plus the
    language-tag readout: the joint's logits for the <|xx|> tags of TAG_LANGS with the start (blank) prediction
    state, averaged over frames, and the largest per-frame probability mass on any language tag)."""
    import torch
    dev = next(m.parameters()).device
    with torch.inference_mode():
        a = torch.as_tensor(x, dtype=torch.float32, device=dev)[None]
        enc, elen, hidden = m.encode(a, torch.tensor([a.shape[1]], device=dev), return_hidden=True)
        ids = m.heads["rnnt"].decode(enc, elen)[0]
        text = m.tokenizer.decode(ids)
        if not pooled:
            return text
        n = int(elen[0])
        pool = torch.stack([h[0, :n].float().mean(0) for h in hidden]).cpu().numpy().astype(np.float16)
        h = m.heads["rnnt"]
        if not hasattr(m, "_tag_ids"):
            m._tag_ids = _tag_ids(m)
        cand, allp = m._tag_ids
        g, _ = h.pred(torch.tensor([[h.blank]], device=dev), None)
        z = h.joint.out(h.joint.enc(enc[0, :n]) + h.joint.pred(g)[0])[:, : h.vocab_size + 1].float()
        lp = z.log_softmax(-1)
        tag = {"tag_logit_mean": [round(float(v), 3) for v in lp[:, cand].mean(0)],
               "tag_pmax": float(lp[:, allp].exp().sum(-1).max())}
    return text, pool, tag


# --------------------------------------------------------------------------- data
def load_fa_set(name: str):
    d = WORK if (WORK / f"{name}.npz").exists() else FA_WORK  # icsi is prepared here, ami / libri by final_asr
    z = np.load(d / f"{name}.npz")
    refs = json.loads((d / f"{name}_refs.json").read_text())
    return [z[f"a{i}"] for i in range(len(refs))], refs


def stage_prep_icsi(a):
    """ICSI dev single-speaker segments, the AMI protocol: ICSI(all dev meetings).asr(1.0, 15.0) (untimed zones and
    overfull segments filtered), random.Random(0).sample(..., 200), sorted -> WORK/icsi.npz + icsi_refs.json."""
    import random

    from audioforge.datasets.icsi import ICSI, subset
    WORK.mkdir(parents=True, exist_ok=True)
    meetings = subset()["dev"]
    ds = ICSI(meetings, verbose=False)
    segs = ds.asr(1.0, 15.0)
    idx = sorted(random.Random(0).sample(range(len(segs)), min(200, len(segs))))
    np.savez(WORK / "icsi.npz", **{f"a{i}": np.asarray(segs[j]["audio"], np.float32) for i, j in enumerate(idx)})
    (WORK / "icsi_refs.json").write_text(json.dumps([segs[j]["text"] for j in idx]))
    print(f"prep icsi: {len(idx)} of {len(segs)} segments from {meetings}; done", flush=True)


def stage_lookahead(a):
    """The served 115M model's RNNT head at att_context [70,0] / [70,1] (served) / [70,13] on ami / libri / icsi:
    masked offline forward = cache-aware streaming at that context (tests/test_streaming.py, NEMO_IMPORT.md);
    batch 4 as final_asr's served row. Hypotheses -> WORK/served_la<R>_<set>.jsonl."""
    import torch
    torch.set_num_threads(2)
    from audioforge.train import load_model
    m = load_model(str(ROOT / a.model), a.device).eval()  # --model / --tag: another checkpoint of the same family
    t_start = time.time()
    left = []
    for set_name in a.sets.split(","):
        audios, refs = load_fa_set(set_name)
        for r in [int(x) for x in a.contexts.split(",")]:
            p = WORK / f"served{a.tag}_la{r}_{set_name}.jsonl"
            done = len(_rows(p))
            if done >= len(refs):
                continue
            with p.open("a") as f:
                i = done
                while i < len(refs) and time.time() - t_start < a.budget:
                    xs = audios[i:i + 4]
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        hyps = m.transcribe(xs, head="rnnt", att_context_size=[70, r])
                    dt = time.perf_counter() - t0
                    for k, h in enumerate(hyps):
                        f.write(json.dumps({"i": i + k, "hyp": h, "sec": dt / len(hyps),
                                            "audio_sec": len(xs[k]) / SR}) + "\n")
                    i += len(xs)
            if i < len(refs):
                left.append(f"{set_name}/la{r}")
    print("done" if not left else f"left {left}", flush=True)


def fleurs_refs() -> dict:
    """utterance id -> raw FLEURS transcription (the test TSVs cached by lid_data)."""
    out = {}
    for p in (ROOT / "data/lid/fleurs/_tsv/data").glob("*/test.tsv"):
        for line in p.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 3:
                out[f[1].removesuffix(".wav")] = f[2]
    return out


def fleurs_rows(lang: str, split: str = "test") -> list[dict]:
    from lid_data import read_manifest
    return [r for r in read_manifest(split, [lang])]


def stage_parakeet(a):
    """parakeet-ctc-0.6b (final_asr's `parakeet` system, batch 4) on the icsi set (ami / libri exist in final_asr).
    Its own process: never with TDT v3."""
    import torch
    torch.set_num_threads(2)
    from audioforge.train import load_model
    m = load_model(str(ROOT / "runs/nemo_parakeet_ctc_0.6b.afm"), "cpu").eval()
    audios, refs = load_fa_set("icsi")
    p = WORK / "parakeet_icsi.jsonl"
    i, t_start = len(_rows(p)), time.time()
    with p.open("a") as f:
        while i < len(refs) and time.time() - t_start < a.budget:
            xs = audios[i:i + 4]
            t0 = time.perf_counter()
            with torch.inference_mode():
                hyps = m.transcribe(xs)
            dt = time.perf_counter() - t0
            for k, h in enumerate(hyps):
                f.write(json.dumps({"i": i + k, "hyp": h, "sec": dt / len(hyps), "audio_sec": len(xs[k]) / SR}) + "\n")
            i += len(xs)
    print("done" if i >= len(refs) else f"{i}/{len(refs)}", flush=True)


# --------------------------------------------------------------------------- run
def stage_run(a):
    import torch
    WORK.mkdir(parents=True, exist_ok=True)
    if a.set in ("ami", "libri", "icsi"):
        jobs = [(a.set, None)]
    else:
        from lid_data import LANGS
        jobs = [(a.set, lg) for lg in ([a.lang] if a.lang else list(LANGS))]
    t_start, m = time.time(), None
    for set_name, lang in jobs:
        tag = set_name if lang is None else f"{set_name}_{lang}"
        p = WORK / f"tdt_{tag}.jsonl"
        done = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        if lang is None:
            audios, refs = load_fa_set(set_name)
            items = list(range(len(refs)))
        else:
            rows = fleurs_rows(lang)
            items = list(range(min(len(rows), a.n_lang) if a.n_lang else len(rows)))  # --n-lang: first N per language
        if len(done) >= len(items):
            continue
        if time.time() - t_start > a.budget:
            break
        if m is None:
            t0 = time.time()
            m = load_tdt(a.device)
            print(f"TDT v3 loaded in {time.time() - t0:.1f}s on {a.device}", flush=True)
            tdt_transcribe(m, np.zeros(SR * 2, np.float32))  # warm-up, not timed
        pooled = lang is not None
        feats = {}
        fp = WORK / f"pool_{tag}.npz"
        if pooled and fp.exists():
            z = np.load(fp)
            feats = {k: z[k] for k in z.files}
        if lang is not None:
            from lid import clip, load_onsets
            from lid_data import load_audio
            ons = load_onsets()
        with p.open("a") as f:
            for i in items[len(done):]:
                if time.time() - t_start > a.budget:
                    break
                if lang is None:
                    x, rid = audios[i], i
                else:
                    r = rows[i]
                    x = load_audio(r)
                    if set_name == "fleurs2s":
                        x = clip(x, ons[r["id"]][0], "2s")
                    rid = r["id"]
                t0 = time.perf_counter()
                out = tdt_transcribe(m, x, pooled=pooled)
                if a.device == "mps":
                    torch.mps.synchronize()
                dt = time.perf_counter() - t0
                extra = {}
                if pooled:
                    out, pool, extra = out
                    feats[str(rid)] = pool
                if pooled and len(feats) % 10 == 0:  # features first, so a killed run never has lines without them
                    np.savez(fp, **feats)
                f.write(json.dumps({"i": i, "id": rid, "hyp": out, "sec": dt, "audio_sec": len(x) / SR, **extra},
                                   ensure_ascii=False) + "\n")
                f.flush()
        if pooled:
            np.savez(fp, **feats)
        n = len(p.read_text().splitlines())
        print(f"{tag}: {n}/{len(items)}", flush=True)
    left = []
    for set_name, lang in jobs:
        tag = set_name if lang is None else f"{set_name}_{lang}"
        p = WORK / f"tdt_{tag}.jsonl"
        n = len(p.read_text().splitlines()) if p.exists() else 0
        want = 200 if lang is None else (min(len(fleurs_rows(lang)), a.n_lang) if a.n_lang else len(fleurs_rows(lang)))
        if n < want:
            left.append(tag)
    print("done" if not left else f"left: {left} (re-run to continue)", flush=True)


# --------------------------------------------------------------------------- scoring
def _edits(r: list, h: list) -> int:
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)]


def edits_words(r: str, h: str) -> tuple[int, int]:
    a, b = r.split(), h.split()
    return _edits(a, b), len(a)


def edits_chars(r: str, h: str) -> tuple[int, int]:
    a, b = list(r.replace(" ", "")), list(h.replace(" ", ""))
    return _edits(a, b), len(a)


def rate_ci(e: np.ndarray, n_boot=1000, seed=0) -> dict:
    w = e[:, 0].sum() / max(1.0, e[:, 1].sum())
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(e), len(e))
        bs.append(e[k, 0].sum() / max(1.0, e[k, 1].sum()))
    return {"wer": round(float(w), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                                 round(float(np.percentile(bs, 97.5)), 4)],
            "errors": int(e[:, 0].sum()), "ref_words": int(e[:, 1].sum()), "n": len(e)}


def paired(a: np.ndarray, b: np.ndarray, n_boot=1000, seed=0) -> dict:
    """WER(a) - WER(b), paired utterance bootstrap."""
    rng = np.random.default_rng(seed)
    ks = [rng.integers(0, len(a), len(a)) for _ in range(n_boot)]
    pt = a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()
    bs = [a[k, 0].sum() / a[k, 1].sum() - b[k, 0].sum() / b[k, 1].sum() for k in ks]
    return {"delta": round(float(pt), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                                    round(float(np.percentile(bs, 97.5)), 4)]}


def _normalizers():
    from final_asr import FILLERS, _snap
    from transformers.models.whisper.english_normalizer import (
        BasicTextNormalizer,
        EnglishTextNormalizer,
    )

    from audioforge.teachers import normalize_text
    en = EnglishTextNormalizer(json.loads(Path(_snap("openai/whisper-tiny"), "normalizer.json").read_text()))

    def nofill(s):
        s = normalize_text(s)
        for f in ("mm hmm", "uh huh"):
            s = s.replace(f, " ")
        return " ".join(w for w in s.split() if w not in FILLERS)
    return {"normalize_text": normalize_text, "nofill": nofill, "whisper_norm": en}, BasicTextNormalizer()


def _rows(p: Path) -> dict:
    return {r["i"]: r for r in map(json.loads, p.read_text().splitlines())} if p.exists() else {}


PAIRS = [("served_la13", "served_la1"), ("served_la0", "served_la1"), ("tdt_v3", "served_la1"),
         ("tdt_v3", "served_la13"), ("tdt_v3", "parakeet"), ("tdt_v3", "whisper_small"), ("tdt_v3", "whisper_turbo"),
         ("served_la13", "parakeet"), ("served_la13", "whisper_small"), ("served_la13", "whisper_turbo"),
         ("served_la1", "served"), ("nemotron_0.6b_160ms", "served_la1"), ("tdt_v3", "nemotron_0.6b_160ms")]


def report_en(norms) -> dict:
    """ami / libri / icsi: every system's WER (3 normalisations, bootstrap CI) and paired deltas (PAIRS)."""
    res = {"results": {}, "paired": {}, "card": CARD,
           "systems": {"served": "final_asr.json row (runs/stage1_served.afm, [70,1], batch 4)",
                       "served_la0/la1/la13": "same model, att_context [70,0] / [70,1] / [70,13] (this script)",
                       "tdt_v3": "parakeet-tdt-0.6b-v3, batch 1", "parakeet": "parakeet-ctc-0.6b (final_asr; icsi here)",
                       "whisper_small / whisper_turbo": "final_asr.json rows (ami, libri only)",
                       "nemotron_0.6b_160ms": "nemotron-speech-streaming-en-0.6b at [70,1] (masked offline = streaming)"}}
    for set_name in ("ami", "libri", "icsi"):
        try:
            _, refs = load_fa_set(set_name)
        except FileNotFoundError:
            continue
        srcs = {"tdt_v3": WORK / f"tdt_{set_name}.jsonl", "nemotron_0.6b_160ms": WORK / f"nemotron_{set_name}.jsonl",
                **{f"served_la{r}": WORK / f"served_la{r}_{set_name}.jsonl" for r in (0, 1, 13)},
                **{s: FA_WORK / f"{s}_{set_name}.jsonl" for s in SYSTEMS_FA}}
        if set_name == "icsi":
            srcs["parakeet"] = WORK / "parakeet_icsi.jsonl"
        hyps, cost = {}, {}
        for sys_name, pth in srcs.items():
            rows = _rows(pth)
            if len(rows) == len(refs):
                hyps[sys_name] = [rows[i]["hyp"] for i in range(len(refs))]
                sec, aud = sum(r["sec"] for r in rows.values()), sum(r["audio_sec"] for r in rows.values())
                cost[sys_name] = round(sec / aud, 4)
        E = {}
        for sys_name, hy in hyps.items():
            rec = res["results"].setdefault(f"{sys_name}/{set_name}", {"n": len(refs), "rtf_logged": cost[sys_name]})
            for nn, fn in norms.items():
                e = np.array([edits_words(fn(r), fn(h)) for r, h in zip(refs, hy)], float)
                E[(sys_name, nn)] = e
                rec[f"wer_{nn}"] = rate_ci(e)
        for x, y in PAIRS:
            for nn in ("normalize_text", "whisper_norm"):
                if (x, nn) in E and (y, nn) in E:
                    res["paired"][f"{x} - {y} / {set_name} / {nn}"] = paired(E[(x, nn)], E[(y, nn)])
        if "tdt_v3" in hyps:
            res["results"][f"tdt_v3/{set_name}"]["examples"] = [
                {"ref": refs[i], "tdt_v3": hyps["tdt_v3"][i], "served_la1": hyps.get("served_la1", [""] * 3)[i],
                 "served_la13": hyps.get("served_la13", [""] * 3)[i]} for i in range(3)]
    return res


def _cap(n_full: int, n_lang: int) -> int:
    return min(n_full, n_lang) if n_lang else n_full


def report_fleurs(norms, basic, n_lang: int = 0) -> dict:
    """Per-language rows; ``n_lang`` > 0 scores the first n_lang utterances of a language (a stated subset, ``n`` in
    the row) when the full 150 are not transcribed."""
    from lid_data import LANGS
    refs_all = fleurs_refs()
    out = {}
    for lang in LANGS:
        rows = _rows(WORK / f"tdt_fleurs_{lang}.jsonl")
        n_full = len(fleurs_rows(lang))
        n_want = n_full if len(rows) >= n_full else _cap(n_full, n_lang)
        if len(rows) < n_want:
            out[lang] = {"incomplete": f"{len(rows)}/{n_want}"}
            continue
        fn = norms["whisper_norm"] if lang == "en" else basic
        hy = [rows[i]["hyp"] for i in range(n_want)]
        rf = [refs_all[rows[i]["id"]] for i in range(n_want)]
        ew = np.array([edits_words(fn(r), fn(h)) for r, h in zip(rf, hy)], float)
        ec = np.array([edits_chars(fn(r), fn(h)) for r, h in zip(rf, hy)], float)
        sec, aud = sum(r["sec"] for r in rows.values()), sum(r["audio_sec"] for r in rows.values())
        rec = {"supported_by_v3": lang in V3_LANGS, "n": n_want, "audio_min": round(aud / 60, 1),
               "wer": rate_ci(ew), "cer": rate_ci(ec), "card_fleurs_wer": CARD_FLEURS.get(lang),
               "primary_metric": "cer" if lang in CER_LANGS else "wer",
               "empty_hyps": int(sum(not h.strip() for h in hy)),
               "rtf_batch1_under_load": round(sec / aud, 4),
               "examples": [{"ref": rf[i], "hyp": hy[i]} for i in range(2)]}
        if n_want < n_full:
            rec["subset"] = f"first {n_want} of the {n_full} FLEURS-17 test utterances of this language"
        out[lang] = rec
    return out


# --------------------------------------------------------------------------- language ID
def _script(s: str) -> str:
    import unicodedata
    counts = {}
    for ch in s:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            sc = name.split(" ")[0] if name else "?"
            counts[sc] = counts.get(sc, 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def report_lid(n_lang: int = 0) -> dict:
    """Language of TDT v3's own output. v3 has no language token (a single 8192-piece SentencePiece shared by 25
    languages, no prompt, no tag in the vocabulary: checked in `lid_vocab`), so its language decision is only
    visible as the language of the text it writes; that text is classified with langid.py (restricted to the union
    of v3's 25 languages and our 17)."""
    import langid
    from lid_data import LANGS
    cands = sorted(set(V3_LANGS) | set(LANGS))
    ident = langid.langid.LanguageIdentifier.from_modelstring(langid.langid.model, norm_probs=True)
    ident.set_languages([c if c != "he" else "he" for c in cands])
    sup = [lg for lg in LANGS if lg in V3_LANGS]
    res = {"method": "langid.py (BSD) on the TDT v3 transcript, candidate set = v3's 25 languages + the 17 test "
                     "languages; empty transcript = no decision (counted wrong)", "supported": sup,
           "unsupported": [lg for lg in LANGS if lg not in V3_LANGS]}
    for cond, tag in (("full", "fleurs"), ("2s", "fleurs2s")):
        per = {}
        tot_ok = tot_n = 0
        for lang in LANGS:
            rows = _rows(WORK / f"tdt_{tag}_{lang}.jsonl")
            n_full = len(fleurs_rows(lang))
            n_use = n_full if len(rows) >= n_full else _cap(n_full, n_lang)
            if len(rows) < n_use:
                continue
            rows = {i: rows[i] for i in range(n_use)}
            preds = []
            for i in range(len(rows)):
                h = rows[i]["hyp"].strip()
                preds.append(ident.classify(h)[0] if h else "empty")
            ok = sum(p == lang for p in preds)
            dist = {}
            for p in preds:
                dist[p] = dist.get(p, 0) + 1
            scripts = {}
            for i in range(len(rows)):
                s = _script(rows[i]["hyp"])
                scripts[s] = scripts.get(s, 0) + 1
            per[lang] = {"acc": round(ok / len(preds), 4), "n": len(preds),
                         "top_outputs": dict(sorted(dist.items(), key=lambda kv: -kv[1])[:4]),
                         "scripts": dict(sorted(scripts.items(), key=lambda kv: -kv[1])[:3])}
            if lang in sup:
                tot_ok += ok
                tot_n += len(preds)
        res[cond] = {"per_lang": per, "acc_supported": round(tot_ok / tot_n, 4) if tot_n else None,
                     "n_supported": tot_n}
        # the language-tag readout (Canary tokenizer's <|xx|> pieces; never emitted by v3's greedy decoding)
        tag_ok = tag_n = 0
        pm = []
        sup_idx = [TAG_LANGS.index(lg) for lg in sup]
        for lang in sup:
            rows = _rows(WORK / f"tdt_{tag}_{lang}.jsonl")
            for r in rows.values():
                if "tag_logit_mean" in r:
                    v = np.array(r["tag_logit_mean"])[sup_idx]
                    tag_ok += sup[int(v.argmax())] == lang
                    tag_n += 1
                    pm.append(r["tag_pmax"])
        if tag_n:
            res[cond]["tag_readout_acc_supported"] = round(tag_ok / tag_n, 4)
            res[cond]["tag_readout_n"] = tag_n
            res[cond]["tag_prob_mass_max_per_utt_median"] = float(np.median(pm))
            res[cond]["tag_prob_mass_max_overall"] = float(np.max(pm))
    # AmberNet (research/LID.md, data/lid/preds/ambernet) on the same utterances, restricted to the 10 languages
    codes = list(LANGS)
    idx = [codes.index(lg) for lg in sup]
    amb = {}
    for ci, cond in ((1, "2s"), (4, "full")):
        ok = n = 0
        for lang in sup:
            z = np.load(ROOT / "data/lid/preds/ambernet" / f"{lang}.npz")
            P = z["P"][:, ci][:, idx]
            ok += int((P.argmax(1) == sup.index(lang)).sum())
            n += len(P)
        amb[cond] = round(ok / n, 4)
    res["ambernet_same_utts_restricted_to_supported"] = amb
    return res


def stage_lid_vocab(a):
    """Does the v3 vocabulary / config carry a language token or prompt? Prints the evidence."""
    import tarfile

    import yaml

    from audioforge.nemo_import import _nemo_file, read_nemo
    nc, _, files = read_nemo(TDT_NEMO, mmap=True)
    from audioforge.tokenizer import SentencePieceTokenizer
    tok = SentencePieceTokenizer(_nemo_file(files, nc["tokenizer"]["model_path"]), specials=[])
    pieces = [tok.sp.id_to_piece(i) for i in range(tok.vocab_size)]
    special = [p for p in pieces if p.startswith("<") and p.endswith(">")]
    langish = [p for p in pieces if p.strip("▁") in ["en", "de", "fr", "es", "ru", "uk", "<en>", "<|en|>"]]
    ev = {"vocab_size": tok.vocab_size, "angle_bracket_pieces": special[:20], "n_angle_bracket_pieces": len(special),
          "lang_code_like_pieces": langish, "config_keys": sorted(nc), "tokenizer_cfg": nc.get("tokenizer"),
          "decoding": nc.get("decoding"), "prompt_or_lang_keys": [k for k in nc if "lang" in k or "prompt" in k]}
    print(json.dumps(ev, indent=1, ensure_ascii=False, default=str))
    _merge("lid_vocab", ev)
    del tarfile, yaml


def stage_lidfeat(a):
    """Pooled per-block TDT v3 encoder features for FLEURS train (2 s and 5 s clips from the onset) - the probe's
    training set. Test features come from the run stage (full utterances / 2 s clips)."""
    import torch
    from lid import clip, load_onsets
    from lid_data import LANGS, load_audio
    ons = load_onsets()
    t_start, m = time.time(), None
    left = []
    for lang in LANGS:
        fp = WORK / f"poolfeat_{a.split}_{lang}.npz"
        if fp.exists():
            continue
        if time.time() - t_start > a.budget:
            left.append(lang)
            continue
        if m is None:
            m = load_tdt(a.device)
        rows = fleurs_rows(lang, a.split)
        X2, X5 = [], []
        for r in rows:
            x = load_audio(r)
            for cond, acc in (("2s", X2), ("5s", X5)):
                xc = clip(x, ons[r["id"]][0], cond)
                with torch.inference_mode():
                    dev = next(m.parameters()).device
                    t = torch.as_tensor(xc, dtype=torch.float32, device=dev)[None]
                    _, elen, hidden = m.encode(t, torch.tensor([t.shape[1]], device=dev), return_hidden=True)
                    n = int(elen[0])
                    acc.append(torch.stack([h[0, :n].float().mean(0) for h in hidden]).cpu().numpy().astype(np.float16))
        np.savez(fp, x2=np.stack(X2), x5=np.stack(X5), ids=np.array([r["id"] for r in rows]))
        print(f"lidfeat {a.split} {lang}: {len(rows)} ({time.time() - t_start:.0f}s)", flush=True)
    print("done" if not left else f"left {left}", flush=True)


def stage_lidprobe(a):
    """Per-block standardise + multinomial logistic regression (C=0.1, as research/LID.md section 1) fitted on train
    5 s (and 2 s) clip features, scored on test 2 s clips and full utterances; 17 languages."""
    from lid_data import LANGS
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    codes = list(LANGS)
    Xtr = {"2s": [], "5s": []}
    ytr = []
    for lang in codes:
        z = np.load(WORK / f"poolfeat_train_{lang}.npz")
        Xtr["2s"].append(z["x2"])
        Xtr["5s"].append(z["x5"])
        ytr += [codes.index(lang)] * len(z["x2"])
    Xtr = {k: np.concatenate(v).astype(np.float32) for k, v in Xtr.items()}
    ytr = np.array(ytr)
    Xte, yte = {"2s": [], "full": []}, []
    for lang in codes:
        rows = fleurs_rows(lang)
        z2, zf = np.load(WORK / f"pool_fleurs2s_{lang}.npz"), np.load(WORK / f"pool_fleurs_{lang}.npz")
        Xte["2s"].append(np.stack([z2[r["id"]] for r in rows]))
        Xte["full"].append(np.stack([zf[r["id"]] for r in rows]))
        yte += [codes.index(lang)] * len(rows)
    Xte = {k: np.concatenate(v).astype(np.float32) for k, v in Xte.items()}
    yte = np.array(yte)
    L = Xtr["5s"].shape[1]
    res = {"blocks": L, "fit": "train 2 s + 5 s clips pooled together (4250 x 2), C=0.1", "per_block": {}}
    blocks = list(range(L)) if not a.blocks else [int(b) for b in a.blocks.split(",")]
    for b in blocks:
        X = np.concatenate([Xtr["2s"][:, b], Xtr["5s"][:, b]])
        y = np.concatenate([ytr, ytr])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(X), y)
        rec = {}
        for cond in ("2s", "full"):
            p = clf.predict(sc.transform(Xte[cond][:, b]))
            rec[cond] = round(float((p == yte).mean()), 4)
        res["per_block"][b] = rec
        print(b, rec, flush=True)
    best = max(res["per_block"], key=lambda k: res["per_block"][k]["2s"] + res["per_block"][k]["full"])
    res["best_block"] = best
    res["best"] = res["per_block"][best]
    # bootstrap CI for the best block
    b = best
    X = np.concatenate([Xtr["2s"][:, b], Xtr["5s"][:, b]])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(X), np.concatenate([ytr, ytr]))
    rng = np.random.default_rng(0)
    for cond in ("2s", "full"):
        ok = (clf.predict(sc.transform(Xte[cond][:, b])) == yte).astype(float)
        bs = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(1000)]
        res[f"best_{cond}_ci95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    _merge("lid_probe", res)


# --------------------------------------------------------------------------- cost
def _turns(seconds: float, n: int) -> list[np.ndarray]:
    """n AMI dev audio crops of exactly ``seconds`` (consecutive segments concatenated, deterministic)."""
    audios, _ = load_fa_set("ami")
    cat = np.concatenate(audios)
    L = int(seconds * SR)
    step = max(L, (len(cat) - L) // n)
    return [cat[i * step: i * step + L] for i in range(n)]


def _time_turns(fn, dev_sync, lens=(1, 3, 5, 10), n=20, warm=2):
    out = {}
    for s in lens:
        xs = _turns(s, n + warm)
        ts = []
        for k, x in enumerate(xs):
            t0 = time.perf_counter()
            fn(x)
            dev_sync()
            if k >= warm:
                ts.append((time.perf_counter() - t0) * 1000)
        out[f"{s}s"] = {"p50_ms": round(float(np.percentile(ts, 50)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1),
                        "mean_ms": round(float(np.mean(ts)), 1), "rtf": round(float(np.mean(ts)) / (1000 * s), 4), "n": n}
        print(s, out[f"{s}s"], flush=True)
    return out


def stage_timing(a):
    import os

    import torch
    torch.set_num_threads(2)
    dev = a.device
    sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
    t0 = time.time()
    m = load_tdt(dev)
    load_s = time.time() - t0
    rss_load = peak_rss_mb()
    res = _json().get("timing", {})
    key = dev + (f"+{a.variants}" if a.variants else "")
    rec = {"device": dev, "threads": torch.get_num_threads() if dev == "cpu" else None, "load_s": round(load_s, 1),
           "peak_rss_mb_after_load": rss_load, "load1": round(os.getloadavg()[0], 2)}
    if dev == "mps":
        rec["note"] = "Apple MPS (M-series GPU) as a proxy for a CUDA GPU; not a CUDA measurement"

    def fp32(x):
        return tdt_transcribe(m, x)
    rec["fp32"] = _time_turns(fp32, sync, n=a.n)
    rec["peak_rss_mb_fp32"] = peak_rss_mb()
    variants = a.variants.split(",") if a.variants else []
    if "bf16" in variants:  # autocast (weights stay fp32)
        dt = torch.bfloat16

        def bf16(x):
            with torch.autocast(device_type=dev, dtype=dt):
                return tdt_transcribe(m, x)
        try:
            rec["bf16_autocast"] = _time_turns(bf16, sync, lens=(1, 5), n=max(5, a.n // 2))
            rec["bf16_autocast_text_equal_5s"] = _same_text(m, fp32, bf16)
        except Exception as e:  # noqa: BLE001
            rec["bf16_autocast"] = f"failed: {type(e).__name__}: {e}"[:300]
    if "fp16" in variants:
        try:
            mh = load_tdt(dev).half()

            def fp16(x):
                return tdt_transcribe(mh, x.astype(np.float32))
            import types as _t
            # the log-mel front end stays fp32: run it in fp32 and feed half features into the encoder
            pp = mh.preprocessor.float()
            orig = mh.encode

            def enc_half(audio, alen, att_context_size=None, spk_act=None, return_hidden=False):
                feats, flen = pp(audio.float(), alen)
                return mh.encoder(feats.half(), flen, att_context_size, return_hidden=return_hidden)
            mh.encode = enc_half
            rec["fp16"] = _time_turns(fp16, sync, lens=(1, 5), n=max(5, a.n // 2))
            rec["fp16_text_equal_5s"] = _same_text(m, fp32, fp16)
            mh.encode = orig
            del _t, mh
        except Exception as e:  # noqa: BLE001
            rec["fp16"] = f"failed: {type(e).__name__}: {e}"[:300]
    if "compile" in variants:
        try:
            enc0 = m.encoder.forward
            m.encoder.forward = torch.compile(m.encoder.forward, dynamic=True)
            tc = time.time()
            fp32(_turns(3, 1)[0])
            sync()
            rec["compile_first_call_s"] = round(time.time() - tc, 1)
            rec["compiled"] = _time_turns(fp32, sync, lens=(1, 5), n=max(5, a.n // 2))
            m.encoder.forward = enc0
        except Exception as e:  # noqa: BLE001
            rec["compiled"] = f"failed: {type(e).__name__}: {e}"[:300]
    rec["peak_rss_mb_end"] = peak_rss_mb()
    res[key] = rec
    _merge("timing", res)
    print(json.dumps(rec, indent=1), flush=True)


def _same_text(m, f1, f2) -> float:
    xs = _turns(5, 10)
    return round(float(np.mean([f1(x) == f2(x) for x in xs])), 2)


# --------------------------------------------------------------------------- streaming core cost (streams per device)
def stage_core(a):
    """Per-stream compute of the served streaming core (audioforge.serve Session: ASR [70,1] + VAD + turn head +
    Sortformer 0.32 s, timeout policy), optionally with the --asr-lookahead 13 second pass, fed 60 s of AMI audio in
    20 ms blocks as fast as possible (no pacing): RTF = compute / audio. The server serializes all sessions' compute
    on one worker thread, so one such worker sustains ~1/RTF real-time streams. --device mps moves both models to the
    Apple GPU (not a supported server mode; a proxy for a CUDA GPU)."""
    import torch
    torch.set_num_threads(2)
    from audioforge.serve import Engine, Session, SessionConfig
    from audioforge.train import load_model
    dev = a.device
    sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
    asr = load_model(str(ROOT / "runs/stage1_served.afm"), "cpu")
    diar = load_model(str(ROOT / "runs/nemo_sortformer_v2.afm"), "cpu")
    res = _json().get("core", {})
    audios, _ = load_fa_set("ami")
    x = np.concatenate(audios)[: int(a.seconds * SR)]
    for la in ([None, 13] if not a.lookahead_only else [13]):
        eng = Engine(asr.to(dev), diar.to(dev), name="core", threads=2, fast=dev == "cpu", asr_lookahead=la)
        eng.warmup()
        s = Session(eng, SessionConfig())
        t0 = time.perf_counter()
        for i in range(0, len(x), 320):
            s.process(x[i:i + 320])
        s.finish()
        sync()
        wall = time.perf_counter() - t0
        rec = {"device": dev, "lookahead": la, "audio_s": round(len(x) / SR, 1), "compute_s": round(wall, 2),
               "rtf": round(wall / (len(x) / SR), 4), "streams_per_worker": round((len(x) / SR) / wall, 2),
               "chunk_ms_p50": _pct(s.chunk_ms, 50), "chunk_ms_p95": _pct(s.chunk_ms, 95),
               "asr_ms_mean_per_block": round(float(np.mean(s.asr_ms)), 3),
               "diar_ms_mean_per_block": round(float(np.mean(s.diar_ms)), 3),
               "lookahead_ms_per_160ms": round(s.la.ms / max(1, len(s.chunk_ms)), 3) if s.la is not None else None,
               "peak_rss_mb": peak_rss_mb(), "load1": round(__import__("os").getloadavg()[0], 2)}
        if dev == "mps":
            rec["note"] = "MPS as a proxy for a CUDA GPU; the server itself supports --device cpu only"
        res[f"{dev}_la{la or 1}"] = rec
        print(json.dumps(rec), flush=True)
    _merge("core", res)


def _pct(xs, q):
    return pct(xs, q, nd=2, empty=None)


# --------------------------------------------------------------------------- nemotron streaming 0.6B (alone)
def stage_nemotron(a):
    import torch
    torch.set_num_threads(2)
    from audioforge.model import StreamingSession
    from audioforge.train import load_model
    dev = a.device
    sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
    m = load_model(str(NEMOTRON), dev).eval()
    att = [70, 1]
    audios, refs = load_fa_set("ami")
    p = WORK / "nemotron_ami.jsonl"
    done = _rows(p)
    t_start = time.time()
    with p.open("a") as f:
        for i in range(len(done), len(refs)):
            if time.time() - t_start > a.budget:
                break
            t0 = time.perf_counter()
            with torch.inference_mode():
                h = m.transcribe([audios[i]], att_context_size=att)[0]
            sync()
            f.write(json.dumps({"i": i, "hyp": h, "sec": time.perf_counter() - t0, "audio_sec": len(audios[i]) / SR}) + "\n")
            f.flush()
    n = len(_rows(p))
    print(f"nemotron ami {n}/{len(refs)}", flush=True)
    if n < len(refs) or a.skip_stream:
        return
    # streaming cost: StreamingSession fed 160 ms pieces over 60 s of AMI audio
    x = np.concatenate(audios)[: 60 * SR]
    s = StreamingSession(m, att_context_size=att)
    piece = SR * 160 // 1000
    ts = []
    with torch.inference_mode():
        s.feed(np.zeros(piece * 4, np.float32))
        s = StreamingSession(m, att_context_size=att)
        for k in range(0, len(x), piece):
            t0 = time.perf_counter()
            s.feed(x[k:k + piece], final=k + piece >= len(x))
            sync()
            ts.append((time.perf_counter() - t0) * 1000)
    rec = {"device": dev, "att_context": att, "chunk_ms": 160, "stream_ms_per_chunk_p50": round(float(np.percentile(ts, 50)), 1),
           "stream_ms_per_chunk_p95": round(float(np.percentile(ts, 95)), 1), "stream_rtf": round(sum(ts) / 1000 / 60, 3),
           "peak_rss_mb": peak_rss_mb(), "note": "MPS as a proxy for a CUDA GPU" if dev == "mps" else "CPU 2 threads",
           "wer_protocol": "masked offline forward at [70,1] = cache-aware streaming (research/ENC_0P6B.md)"}
    res = _json().get("nemotron_stream", {})
    res[dev] = rec
    _merge("nemotron_stream", res)
    print(json.dumps(rec, indent=1), flush=True)


# --------------------------------------------------------------------------- report
def stage_adapt_report(a):
    """IMPROVE_115M Part B: the adapted decoder (--tag) vs the served model at [70,1] on ami / libri / icsi, paired
    -> runs/hybrid_asr.json["adapt"][tag]."""
    norms, _ = _normalizers()
    out = {"model": a.model, "sets": {}}
    for set_name in a.sets.split(","):
        try:
            _, refs = load_fa_set(set_name)
        except FileNotFoundError:
            continue
        base, new = _rows(WORK / f"served_la1_{set_name}.jsonl"), _rows(WORK / f"served{a.tag}_la1_{set_name}.jsonl")
        if len(base) < len(refs) or len(new) < len(refs):
            out["sets"][set_name] = {"incomplete": f"served {len(base)}, adapted {len(new)} of {len(refs)}"}
            continue
        rec = {"n": len(refs)}
        for nn in ("normalize_text", "whisper_norm"):
            fn = norms[nn]
            eb = np.array([edits_words(fn(r), fn(base[i]["hyp"])) for i, r in enumerate(refs)], float)
            en = np.array([edits_words(fn(r), fn(new[i]["hyp"])) for i, r in enumerate(refs)], float)
            rec[nn] = {"served": rate_ci(eb), "adapted": rate_ci(en), "adapted - served": paired(en, eb)}
        out["sets"][set_name] = rec
    res = _json()
    res.setdefault("adapt", {})[a.tag or "served"] = out
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1))


def stage_report(a):
    norms, basic = _normalizers()
    res = _json()
    res["protocol"] = __doc__.split("Stages")[0].strip()
    res["model"] = {"id": "nvidia/parakeet-tdt-0.6b-v3", "license": "CC-BY-4.0",
                    "license_verbatim": "GOVERNING TERMS: Use of this model is governed by the [CC-BY-4.0]"
                                        "(https://creativecommons.org/licenses/by/4.0/legalcode.en) license.",
                    "supported_languages": V3_LANGS}
    if not a.fleurs_only:
        res["english"] = report_en(norms)
    prev_f, prev_l = res.get("fleurs", {}), res.get("lid_text", {})
    new_f = report_fleurs(norms, basic, a.n_lang)
    # per-language merge: a row measured on more utterances earlier (another scratch dir) is kept
    res["fleurs"] = {lg: (prev_f[lg] if isinstance(prev_f.get(lg), dict) and "n" in prev_f[lg]
                          and prev_f[lg]["n"] > v.get("n", 0) else v) for lg, v in new_f.items()}
    try:
        new_l = report_lid(a.n_lang)
        for cond in ("full", "2s"):
            pl, nl = prev_l.get(cond, {}).get("per_lang", {}), new_l[cond]["per_lang"]
            for lg, v in pl.items():
                if lg not in nl or nl[lg]["n"] < v["n"]:
                    nl[lg] = v
            sup = [lg for lg in new_l["supported"] if lg in nl]
            new_l[cond]["n_supported"] = int(sum(nl[lg]["n"] for lg in sup))
            new_l[cond]["acc_supported"] = round(sum(nl[lg]["acc"] * nl[lg]["n"] for lg in sup)
                                                 / new_l[cond]["n_supported"], 4) if sup else None
            new_l[cond]["n_per_lang_min"] = int(min(v["n"] for v in nl.values())) if nl else 0
        res["lid_text"] = new_l
    except FileNotFoundError as e:
        res["lid_text"] = {"incomplete": str(e)}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for k, v in res["english"]["results"].items():
        print(k, {n: v[n]["wer"] for n in v if n.startswith("wer_")})
    for k, v in res["english"]["paired"].items():
        print(k, v)
    for lg, v in res["fleurs"].items():
        print(lg, v.get("incomplete") or (v["supported_by_v3"], v["wer"]["wer"], v["cer"]["wer"], v["card_fleurs_wer"]))
    print(f"wrote {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["core", "prep_icsi", "lookahead", "adapt_report", "parakeet", "run", "lid_vocab", "lidfeat", "lidprobe", "timing", "nemotron",
                                      "report"])
    ap.add_argument("--set", default="ami", choices=["ami", "libri", "icsi", "fleurs", "fleurs2s"])
    ap.add_argument("--sets", default="ami,libri,icsi")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=float, default=480.0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--variants", default="")
    ap.add_argument("--blocks", default="")
    ap.add_argument("--skip-stream", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--lookahead-only", action="store_true")
    ap.add_argument("--model", default="runs/stage1_served.afm", help="lookahead: checkpoint (an adapted served model)")
    ap.add_argument("--tag", default="", help="lookahead / adapt_report: hypothesis-file tag of --model ('' = served)")
    ap.add_argument("--contexts", default="0,1,13", help="lookahead: right contexts R to run")
    ap.add_argument("--n-lang", type=int, default=0, help="run/report: first N FLEURS utterances per language (0 = all 150)")
    ap.add_argument("--fleurs-only", action="store_true", help="report: keep the stored english block (its scratch is gone)")
    a = ap.parse_args()
    {"core": stage_core, "prep_icsi": stage_prep_icsi, "lookahead": stage_lookahead, "adapt_report": stage_adapt_report, "parakeet": stage_parakeet, "run": stage_run, "lid_vocab": stage_lid_vocab, "lidfeat": stage_lidfeat, "lidprobe": stage_lidprobe,
     "timing": stage_timing, "nemotron": stage_nemotron, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
