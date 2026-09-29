"""nvidia/nemotron-speech-streaming-en-0.6b as a voice-agent front-end encoder (research/ENC_0P6B.md).

Every stage is a separate CPU process (2 threads, < 10 min, resumable) that builds the model straight from the
.nemo through audioforge.nemo_import (memory-mapped weights, one fp32 copy ~2.5 GB); results merge into --json.

  import                      import, save runs/nemo_nemotron_speech_streaming_en_0.6b.afm, tensor counts, quick timing
  wer --att 70,13 | 70,1      LibriSpeech test-clean first-200 WER (offline chunked-limited forward == streaming), resumable
  equality                    stream_step vs offline forward and StreamingSession transcript == offline transcript
  rtf                         StreamingSession on 160 ms chunks (att [70,1]): per-chunk time, RTF, peak RSS (own process)
  features --set <name>       per-layer fp16 frame features (24 x T x 1024) at [70,1] -> data/cache/enc0p6b/<set>/<id>.npz
                              sets: spk_ami_200 spk_ami_64 spk_icsi_200 spk_icsi_64 vad_ami_dev vad_icsi_dev vad_ami_train
                              asr_ami_train
  spk_probe                   untrained per-layer mean-pool speaker EER (all pairs / within meeting) from the cache
  vad_probe [--layers ...]    per-layer VAD probes (vad_layers.fit_logreg / fit_mlp, BASELINES protocol) from the cache
  report                      markdown tables (runs/enc_0p6b.json -> stdout)
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

NEMO = ROOT / "data/nemo/nemotron-speech-streaming-en-0.6b.nemo"
AFM = ROOT / "runs/nemo_nemotron_speech_streaming_en_0.6b.afm"
CACHE = ROOT / "data/cache/enc0p6b"
JSON = ROOT / "runs/enc_0p6b.json"
ATT_160 = [70, 1]
ATT_DEFAULT = [70, 13]
SETS = ("spk_ami_200", "spk_ami_64", "spk_icsi_200", "spk_icsi_64", "vad_ami_dev", "vad_icsi_dev", "vad_ami_train",
        "asr_ami_train")


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _merge(path: Path, key: str, rec, sub: str | None = None):
    res = _json(path)
    if sub is None:
        res[key] = rec
    else:
        res.setdefault(key, {})[sub] = rec
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30  # macOS: bytes


def load(threads: int = 2):
    from audioforge.nemo_import import import_nemo
    torch.set_num_threads(threads)
    t0 = time.time()
    m = import_nemo(NEMO)
    print(f"[load] {m.num_params() / 1e6:.1f}M params in {time.time() - t0:.1f}s, rss {rss_gb():.2f} GB", flush=True)
    return m


def att_of(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


# --------------------------------------------------------------------------- import
def stage_import(a):
    from audioforge.train import save_model
    m = load(a.threads)
    info = dict(m.import_info)
    rec = {"nemo": str(NEMO.relative_to(ROOT)), "afm": str(AFM.relative_to(ROOT)), "params_m": round(m.num_params() / 1e6, 2),
           "import_info": info, "encoder": {k: m.cfg["encoder"][k] for k in ("d_model", "n_layers", "n_heads", "feat_in",
                                                                              "causal", "att_context_sizes", "xscaling",
                                                                              "conv_norm")},
           "heads": m.cfg["heads"], "preprocessor": m.cfg["preprocessor"], "nemo_source": m.cfg["nemo_source"],
           "vocab": m.tokenizer.vocab_size, "rss_after_load_gb": round(rss_gb(), 2)}
    # quick timing: one 20 s window, offline forward with the [70,1] mask (what streaming computes), batch 1
    x = torch.randn(1, 16000 * 20) * 0.05
    with torch.no_grad():
        m.encode(x, torch.tensor([x.shape[1]]), ATT_160)
        t0 = time.time()
        m.encode(x, torch.tensor([x.shape[1]]), ATT_160)
    rec["offline_rtf_20s_b1_70_1"] = round((time.time() - t0) / 20, 4)
    print("[import]", json.dumps(rec, indent=1), flush=True)
    if not a.no_save:
        t0 = time.time()
        save_model(m, AFM)
        rec["save_sec"] = round(time.time() - t0, 1)
        rec["afm_bytes"] = AFM.stat().st_size
        print(f"[import] wrote {AFM} ({rec['afm_bytes'] / 2 ** 20:.0f} MB) in {rec['save_sec']}s", flush=True)
    rec["peak_rss_gb"] = round(rss_gb(), 2)
    _merge(JSON, "import", rec)


# --------------------------------------------------------------------------- WER
def _ls_rows():
    from benchmark_teachers import utterances
    return utterances(200)


def stage_wer(a):
    from audioforge.metrics import wer
    from audioforge.teachers import load_audio, normalize_text
    att = att_of(a.att)
    tag = f"{att[0]}-{att[1]}"
    rows = _ls_rows()
    out = ROOT / "runs/enc0p6b" / f"hyps_ls200_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"]: json.loads(l) for l in out.read_text().splitlines()} if out.exists() else {}
    todo = [r for r in rows if r["id"] not in done]
    print(f"[wer {tag}] {len(done)} done, {len(todo)} to do", flush=True)
    if todo:
        m = load(a.threads)
        todo.sort(key=lambda r: r["duration"])
        t0 = time.time()
        with out.open("a") as f:
            for i in range(0, len(todo), a.batch_size):
                b = todo[i: i + a.batch_size]
                audios = [load_audio(ROOT / r["audio_filepath"]) for r in b]
                t1 = time.time()
                hyps = m.transcribe(audios, head="rnnt", att_context_size=att)
                sec = time.time() - t1
                for r, h in zip(b, hyps):
                    rec = {"id": r["id"], "hyp": h, "ref": r["text"], "duration": r["duration"],
                           "sec": round(sec / len(b), 3)}
                    f.write(json.dumps(rec) + "\n")
                    done[r["id"]] = rec
                f.flush()
                print(f"[wer {tag}] {len(done)}/{len(rows)} ({time.time() - t0:.0f}s, batch rtf "
                      f"{sec / sum(len(x) for x in audios) * 16000:.3f})", flush=True)
                if time.time() - t0 > a.budget:
                    print("[wer] budget reached, rerun to resume", flush=True)
                    break
    if len(done) < len(rows):
        return
    refs = [normalize_text(r["text"]) for r in rows]
    hyps = [normalize_text(done[r["id"]]["hyp"]) for r in rows]
    n_digit = sum(any(c.isdigit() for c in h) for h in hyps)
    errs = [(r["id"], x, y) for r, x, y in zip(rows, refs, hyps) if x != y]
    rec = {"att_context_size": att, "lookahead_sec": round(att[1] * 0.08, 2), "n": len(rows), "wer": round(wer(refs, hyps), 4),
           "ref_words": sum(len(r.split()) for r in refs), "hyps_with_digits": n_digit, "n_utt_with_errors": len(errs),
           "audio_sec": round(sum(r["duration"] for r in rows), 1),
           "transcribe_sec": round(sum(done[r["id"]]["sec"] for r in rows), 1),
           "normalization": "audioforge.teachers.normalize_text (lowercase, punctuation stripped; no number expansion)",
           "examples": [{"id": i, "ref": x, "hyp": y} for i, x, y in errs[:8]]}
    rec["offline_rtf_batched"] = round(rec["transcribe_sec"] / rec["audio_sec"], 4)
    print(f"[wer {tag}] WER {100 * rec['wer']:.2f}% on {rec['n']} utts; {n_digit} hyps with digits", flush=True)
    _merge(JSON, "wer", rec, tag)


# --------------------------------------------------------------------------- streaming == offline
def stage_equality(a):
    from audioforge.model import StreamingSession
    from audioforge.teachers import load_audio
    rows = _ls_rows()[:3]
    m = load(a.threads)
    rec = {"utterances": [r["id"] for r in rows], "contexts": {}}
    for att in (ATT_DEFAULT, ATT_160):
        tag = f"{att[0]}-{att[1]}"
        r_att = {"max_abs_diff": [], "act_scale": [], "frames_equal": [], "session_equals_offline": [], "text": []}
        for r in rows:
            audio = load_audio(ROOT / r["audio_filepath"])
            x, lens = torch.tensor(audio)[None], torch.tensor([len(audio)])
            feats, fl = m.preprocessor(x, lens)
            with torch.no_grad():
                off, olen = m.encoder(feats, fl, att)
            cs, T, st, outs = m.encoder.stream_chunk_frames(att), int(fl), None, []
            for s in range(0, T, cs):
                y, st = m.encoder.stream_step(feats[..., s:s + cs], st, att, final=s + cs >= T)
                outs.append(y)
            on = torch.cat(outs, 1)
            r_att["frames_equal"].append(on.shape[1] == int(olen))
            r_att["max_abs_diff"].append(float((on - off[:, :on.shape[1]]).abs().max()) if on.shape[1] == int(olen) else None)
            r_att["act_scale"].append(round(float(off.abs().mean()), 3))
            text = m.transcribe([audio], head="rnnt", att_context_size=att)[0]
            sess = StreamingSession(m, "rnnt", att)
            piece = 2560 if att == ATT_160 else 1600
            for s in range(0, len(audio), piece):
                sess.feed(audio[s:s + piece])
            st_text = sess.feed(audio[:0], final=True)
            r_att["session_equals_offline"].append(st_text == text)
            r_att["text"].append({"offline": text, "session": st_text, "ref": r["text"]})
            print(f"[equality {tag}] {r['id']}: frames {r_att['frames_equal'][-1]} max|d| {r_att['max_abs_diff'][-1]} "
                  f"session==offline {st_text == text}", flush=True)
        rec["contexts"][tag] = r_att
    _merge(JSON, "equality", rec)


# --------------------------------------------------------------------------- RTF / RSS
def stage_rtf(a):
    from audioforge.model import StreamingSession
    from audioforge.teachers import load_audio
    rows = _ls_rows()
    audio = np.concatenate([load_audio(ROOT / r["audio_filepath"]) for r in rows[:40]])[: 16000 * a.seconds]
    rss0 = rss_gb()
    t0 = time.time()
    m = load(a.threads)
    load_sec = time.time() - t0
    rss_load = rss_gb()
    att = ATT_160
    piece = 2560  # 160 ms
    sess = StreamingSession(m, "rnnt", att)
    times = []
    # warm-up: first 5 chunks are not timed (allocations)
    for i, s in enumerate(range(0, len(audio), piece)):
        t1 = time.perf_counter()
        sess.feed(audio[s:s + piece])
        dt = time.perf_counter() - t1
        if i >= 5:
            times.append(dt)
    text = sess.feed(audio[:0], final=True)
    t = np.array(times)
    rec = {"att_context_size": att, "chunk_ms": 160, "audio_sec": round(len(audio) / 16000, 1), "n_chunks_timed": len(t),
           "chunk_ms_mean": round(1000 * t.mean(), 1), "chunk_ms_p50": round(1000 * np.median(t), 1),
           "chunk_ms_p95": round(1000 * np.percentile(t, 95), 1), "chunk_ms_max": round(1000 * t.max(), 1),
           "rtf": round(t.sum() / (len(t) * 0.16), 3), "threads": a.threads, "load_sec": round(load_sec, 1),
           "rss_before_load_gb": round(rss0, 2), "rss_after_load_gb": round(rss_load, 2), "peak_rss_gb": round(rss_gb(), 2),
           "text_head": text[:200], "note": "StreamingSession (Python path: mel + stream_step + greedy RNNT per chunk)"}
    print("[rtf]", json.dumps(rec, indent=1), flush=True)
    _merge(JSON, "rtf", rec)


# --------------------------------------------------------------------------- features
def load_items(name: str) -> list[dict]:
    """Items with audio (+ speaker / meeting / vad); ``id`` is stable across processes."""
    from audioforge.data import segment_id
    if name.startswith("spk_"):
        from spk_head import dev_segments
        _, corpus, n = name.split("_")
        val = dev_segments(corpus, int(n))
        for v in val:
            v["id"] = segment_id(v)
        return val
    if name.startswith("vad_"):
        from vad_layers import load_set
        # train: the 300-window seeded cap that the 115M rows in runs/vad_layers.json were fitted on (100 min of audio)
        val = load_set({"vad_ami_dev": "ami_dev", "vad_icsi_dev": "icsi_dev", "vad_ami_train": "train"}[name], n_train=300)
        for i, v in enumerate(val):
            v["id"] = f"{v.get('meeting', 'w')}:{float(v.get('start', i)):.3f}:{len(v['audio'])}"
        return val
    if name == "asr_ami_train":
        from audioforge.datasets.ami import recipe_data
        val = recipe_data({"data": {"ami": {"mode": "asr"}}}, "train")
        for v in val:
            v["id"] = segment_id(v)
        return val
    raise ValueError(name)


def item_path(name: str, id_: str) -> Path:
    return CACHE / name / (id_.replace("/", "_") + ".npz")


@torch.no_grad()
def stage_features(a):
    from audioforge.data import Collate
    val = load_items(a.set)
    d = CACHE / a.set
    d.mkdir(parents=True, exist_ok=True)
    meta_p = d / "meta.json"
    if not meta_p.exists():
        meta_p.write_text(json.dumps({"set": a.set, "att_context_size": ATT_160, "n": len(val), "dtype": "float16",
                                      "layout": "h: (n_layers, T, d_model) per-layer block outputs, full frame sequence; "
                                                "vad: (T,) any-speaker label when present",
                                      "items": [{"id": v["id"], "meeting": v.get("meeting"), "speaker": v.get("speaker"),
                                                 "duration": round(len(v["audio"]) / 16000, 3)} for v in val]}, indent=1))
    todo = [v for v in val if not item_path(a.set, v["id"]).exists()]
    print(f"[features {a.set}] {len(val) - len(todo)} cached, {len(todo)} to do", flush=True)
    if not todo:
        return
    m = load(a.threads)
    col = Collate(None)
    bs = a.batch_size
    t0 = time.time()
    n_done = 0
    for i in range(0, len(todo), bs):
        b = todo[i: i + bs]
        batch = col([{"audio": v["audio"]} for v in b])
        enc, elen, hidden = m.encode(batch["audio"], batch["audio_len"], ATT_160, return_hidden=True)
        for j, v in enumerate(b):
            T = int(elen[j])
            h = torch.stack([x[j, :T] for x in hidden]).to(torch.float16).numpy()
            extra = {}
            if "vad" in v:
                lab = np.asarray(v["vad"], np.float32)
                extra["vad"] = lab[:T] if len(lab) >= T else np.pad(lab, (0, T - len(lab)))
            tmp = item_path(a.set, v["id"]).with_suffix(".tmp.npz")
            np.savez(tmp, h=h, **extra)
            tmp.rename(item_path(a.set, v["id"]))
        n_done += len(b)
        el = time.time() - t0
        print(f"[features {a.set}] {n_done}/{len(todo)} in {el:.0f}s (rss {rss_gb():.2f} GB)", flush=True)
        if el > a.budget:
            print("[features] budget reached, rerun to resume", flush=True)
            return
    print(f"[features {a.set}] complete", flush=True)


def load_cached(name: str, ids: list[str]):
    """-> list of (h fp16 (L, T, D), extras dict) in the order of ids; raises if any is missing."""
    out = []
    for i in ids:
        p = item_path(name, i)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run features --set {name}")
        z = np.load(p)
        out.append((z["h"], {k: z[k] for k in z.files if k != "h"}))
    return out


# --------------------------------------------------------------------------- speaker probe
def stage_spk_probe(a):
    from spk_head import eer_block
    rec = {"method": "mean-pooled per-layer block output (fp16 cache, att [70,1]), cosine, scripts/spk_head.eer_block trials"}
    for corpus in ("ami", "icsi"):
        for n in (64, 200):
            name = f"spk_{corpus}_{n}"
            val = load_items(name)
            items = load_cached(name, [v["id"] for v in val])
            L = items[0][0].shape[0]
            E = torch.stack([torch.tensor(h.astype(np.float32).mean(1)) for h, _ in items], 1)  # L, N, D
            rows = [eer_block(E[i], val) for i in range(L)]
            best = min(range(L), key=lambda i: rows[i]["eer_within_meeting"])
            rec[f"{corpus}_n{n}"] = {"n_segments": len(val), "n_speakers": rows[0]["n_speakers"],
                                     "per_layer": [{"layer": i + 1, "eer": r["eer"], "eer_within_meeting": r["eer_within_meeting"]}
                                                   for i, r in enumerate(rows)],
                                     "best_layer": best + 1, "best_within": rows[best]["eer_within_meeting"],
                                     "top_within": rows[-1]["eer_within_meeting"]}
            print(f"[spk_probe] {corpus} n={n}: best block {best + 1} within {100 * rows[best]['eer_within_meeting']:.1f}% "
                  f"(all {100 * rows[best]['eer']:.1f}%), top block within {100 * rows[-1]['eer_within_meeting']:.1f}%", flush=True)
    _merge(JSON, "spk_probe", rec)


# --------------------------------------------------------------------------- VAD probe
def _vad_matrix(name: str, layer: int):
    val = load_items(name)
    Xs, ys, ws = [], [], []
    for w, v in enumerate(val):
        z = np.load(item_path(name, v["id"]))
        h, lab = z["h"][layer], z["vad"]
        T = min(len(h), len(lab))
        Xs.append(h[:T].astype(np.float32))
        ys.append(lab[:T])
        ws.append(np.full(T, w, np.int32))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def stage_vad_probe(a):
    from vad_layers import FPR_POINT, fit_logreg, fit_mlp, per_window, vad_report
    torch.set_num_threads(a.threads)
    res = _json(JSON)
    q = res.setdefault("vad_probe", {})
    q["setup"] = {"train": "vad_ami_train (AMI train diar windows, vad_layers.load_set('train'))",
                  "dev": ["vad_ami_dev (BASELINES VAD set, 64 x 20 s)", "vad_icsi_dev (64 x 20 s, held-out corpus)"],
                  "features": "fp16 per-layer block outputs at [70,1] (= streaming), standardized per block on train frames",
                  "logreg": "sklearn LogisticRegression C=1", "mlp": "Linear(1024,64)-SiLU-Linear(64,1), vad_layers.fit_mlp",
                  "metrics": f"sd.score_vad @0.5, AUC, miss @FPR {FPR_POINT}"}
    n_layers = np.load(item_path("vad_ami_dev", load_items("vad_ami_dev")[0]["id"]))["h"].shape[0]
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(1, n_layers + 1))
    rows = q.setdefault("rows", {})
    t_all = time.time()
    for lay in layers:
        key = f"block{lay:02d}"
        if key in rows and not a.force:
            continue
        t0 = time.time()
        X, y, _ = _vad_matrix("vad_ami_train", lay - 1)
        if a.max_train_frames and len(y) > a.max_train_frames:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(len(y), a.max_train_frames, replace=False))
            X, y = X[idx], y[idx]
        mu, sdv = X.mean(0), X.std(0) + 1e-5
        X = (X - mu) / sdv
        yb = y > 0.5
        row = {"block": lay, "train_frames": int(len(yb))}
        dev = {n: _vad_matrix(n, lay - 1) for n in ("vad_ami_dev", "vad_icsi_dev")}
        for name, fit in (("logreg", fit_logreg), ("mlp64", fit_mlp)):
            t1 = time.time()
            pred = fit(X, yb)
            row[name] = {"fit_sec": round(time.time() - t1, 1)}
            for n, (Xd, yd, wd) in dev.items():
                p = pred((Xd - mu) / sdv)
                row[name][n] = vad_report(per_window(p, wd), per_window(yd > 0.5, wd))
        row["sec"] = round(time.time() - t0, 1)
        rows[key] = row
        _merge(JSON, "vad_probe", q)
        print(f"[vad_probe] {key}: logreg AMI F1 {row['logreg']['vad_ami_dev']['f1']:.4f} AUC {row['logreg']['vad_ami_dev']['auc']:.4f}"
              f" | mlp64 AMI F1 {row['mlp64']['vad_ami_dev']['f1']:.4f} AUC {row['mlp64']['vad_ami_dev']['auc']:.4f} "
              f"miss@fpr {row['mlp64']['vad_ami_dev'][f'miss_at_fpr{FPR_POINT}']:.4f} | ICSI mlp F1 "
              f"{row['mlp64']['vad_icsi_dev']['f1']:.4f} AUC {row['mlp64']['vad_icsi_dev']['auc']:.4f} ({row['sec']:.0f}s)", flush=True)
        if time.time() - t_all > a.budget:
            print("[vad_probe] budget reached, rerun to resume", flush=True)
            break


# --------------------------------------------------------------------------- report
def stage_report(a):
    r = _json(JSON)
    ref = _json(ROOT / "runs/spk_head.json").get("variants", {}).get("untrained_meanpool", {}).get("layers", {})
    vref = _json(ROOT / "runs/vad_layers.json").get("q1_probes", {})
    if "wer" in r:
        print("| att context | lookahead | 0.6B WER % (LS test-clean-200) | card (full test-clean) | 115M hybrid RNNT |")
        print("|---|---|---:|---:|---:|")
        card = {"70-13": 2.32, "70-6": 2.46, "70-1": 2.56, "70-0": 2.80}
        ours115 = {"70-13": 1.92, "70-6": 1.94, "70-1": 2.29, "70-0": 2.48}
        for tag, w in r["wer"].items():
            print(f"| [{tag.replace('-', ', ')}] | {w['lookahead_sec']} s | {100 * w['wer']:.2f} | {card.get(tag, '-')} | {ours115.get(tag, '-')} |")
    if "rtf" in r:
        x = r["rtf"]
        print(f"\nRTF (2 threads, 160 ms chunks, StreamingSession): {x['rtf']} ; chunk mean/p50/p95 {x['chunk_ms_mean']}/"
              f"{x['chunk_ms_p50']}/{x['chunk_ms_p95']} ms ; peak RSS {x['peak_rss_gb']} GB (after load {x['rss_after_load_gb']})")
    if "spk_probe" in r:
        print("\nuntrained mean-pooled layer, within-meeting EER % (0.6B block / 115M block):")
        for key in ("ami_n200", "icsi_n200", "ami_n64", "icsi_n64"):
            if key not in r["spk_probe"]:
                continue
            ours = r["spk_probe"][key]["per_layer"]
            base = ref.get(key, {}).get("per_layer", [])
            print(f"  {key}: 0.6B " + " ".join(f"L{p['layer']}:{100 * p['eer_within_meeting']:.0f}" for p in ours))
            if base:
                print(f"  {key}: 115M " + " ".join(f"L{p['layer'] + 1}:{100 * p['eer_within_meeting']:.0f}" for p in base))
    if "vad_probe" in r:
        print("\n| block | 0.6B logreg AMI F1 / AUC | 0.6B mlp64 AMI F1 / AUC / miss@0.075 | 0.6B mlp64 ICSI F1 / AUC | 115M mlp64 AMI F1 / AUC | 115M mlp64 ICSI F1 / AUC |")
        print("|---|---|---|---|---|---|")
        for key, row in sorted(r["vad_probe"].get("rows", {}).items()):
            b = vref.get("rows", {}).get(key, {}).get("mlp64", {})
            lr, ml = row["logreg"], row["mlp64"]
            print(f"| {row['block']} | {lr['vad_ami_dev']['f1']:.3f} / {lr['vad_ami_dev']['auc']:.3f} | "
                  f"{ml['vad_ami_dev']['f1']:.3f} / {ml['vad_ami_dev']['auc']:.3f} / {ml['vad_ami_dev']['miss_at_fpr0.075']:.3f} | "
                  f"{ml['vad_icsi_dev']['f1']:.3f} / {ml['vad_icsi_dev']['auc']:.3f} | "
                  f"{b.get('ami_dev', {}).get('f1', float('nan')):.3f} / {b.get('ami_dev', {}).get('auc', float('nan')):.3f} | "
                  f"{b.get('icsi_dev', {}).get('f1', float('nan')):.3f} / {b.get('icsi_dev', {}).get('auc', float('nan')):.3f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("import", "wer", "equality", "rtf", "features", "spk_probe", "vad_probe", "report"))
    ap.add_argument("--att", default="70,13")
    ap.add_argument("--set", default=None, choices=SETS)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--budget", type=float, default=450.0, help="seconds of work per process before exiting (resume by rerun)")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--max-train-frames", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    {"import": stage_import, "wer": stage_wer, "equality": stage_equality, "rtf": stage_rtf, "features": stage_features,
     "spk_probe": stage_spk_probe, "vad_probe": stage_vad_probe, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
