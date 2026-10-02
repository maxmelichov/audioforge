"""research/CORE_0P6B.md: nemotron-speech-streaming-en-0.6b as a second served core (``--core 0.6b``) with every head
retrained on it, measured on the same benchmarks as the 115M -> runs/core_0p6b.json.

Stages (each one process, < 10 min, resumable; run through scripts/dev/gate.sh, one torch/MPS job at a time):
  verify            the imported 0.6B .afm (AFM) reads back, sha256 of the .nemo = the hub pin, streaming equality at
                    [70,1]: encoder.stream_step (16 mel frames per step) vs the masked offline forward, and
                    StreamingSession (raw audio in 160 ms pieces) text == offline text, 3 LibriSpeech utterances
  core --device D   the bare ASR core as it streams words (model.StreamingSession: mel + cache-aware encoder at [70,1]
                    + greedy RNNT, batch 1, 160 ms chunks, 2 threads): per-chunk compute p50 / p95 and RTF on the first
                    20 LibriSpeech-200 utterances (scripts/research/mps_115m.py time_stream, best of 3), memory
  core_streams --device D
                    K bare-core StreamingSessions fed interleaved 160 ms block by block on one thread (the
                    streams_cpu.py criterion: real time while p95 of the summed per-block compute < 160 ms)

Paths: the 0.6B .afm lives on the SSD (AUDIOFORGE_0P6B_AFM overrides); caches under W.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

# --core 0.6b (English nemotron-speech-streaming-en-0.6b, research/CORE_0P6B.md) | 115m (the default core,
# research/LAYER_SWEEP_115M.md) | 3.5 (nemotron-3.5-asr-streaming-0.6b, research/CORE_3P5.md: the multilingual
# extension of the English 0.6B, same 24 x 1024 cache-aware FastConformer, trained at left context 56; the language
# prompt only enters the RNNT joint, the encoder blocks every head reads do not depend on it)
CORE = sys.argv[sys.argv.index("--core") + 1] if "--core" in sys.argv else "0.6b"
C115 = CORE == "115m"
C35 = CORE == "3.5"
TAG = {"0.6b": "0p6b", "115m": "115m", "3.5": "3p5"}[CORE]  # file / key tag of the core
NEMO = ROOT / ("data/nemo/nemotron-3.5-asr-streaming-0.6b.nemo" if C35 else
               "data/nemo/nemotron-speech-streaming-en-0.6b.nemo")
AFM = Path(os.environ.get("AUDIOFORGE_3P5_AFM",
                          "/Volumes/ExternalSSD/nvidia-audio-models/runs/nemo_nemotron_3p5_asr_streaming_0.6b.afm")) \
    if C35 else Path(os.environ.get("AUDIOFORGE_0P6B_AFM",
                                    "/Volumes/ExternalSSD/nvidia-audio-models/runs/nemo_nemotron_speech_streaming_en_0.6b.afm"))
# --core 115m: the same stages on the DEFAULT 115M core (research/LAYER_SWEEP_115M.md; the shipped
# stage1_served_v3.afm's frozen encoder + its served heads; caches under scratch/core_115m, results runs/core_115m.json)
AFM115 = Path(os.environ.get("CORE115_AFM", str(ROOT / "runs" / "stage1_served_v3.afm")))  # a candidate build overrides
W = Path(os.environ.get("CORE_0P6B_W", "/Volumes/ExternalSSD/nvidia-audio-models/scratch/core_" + TAG))
OUT = ROOT / "runs" / f"core_{TAG}.json"
D = 512 if C115 else 1024  # encoder width
NB = 17 if C115 else 24  # encoder blocks
SR, CHUNK = 16000, 2560
ATT = [56, 1] if C35 else [70, 1]  # 160 ms chunks at the core's trained left context


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(key, val, sub=None):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    if sub is None:
        d[key] = val
    else:
        d.setdefault(key, {})[sub] = val
    OUT.write_text(json.dumps(d, indent=1, default=float))


def load_json(key, default=None):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    return d.get(key, default)


# --------------------------------------------------------------------------- verify
def stage_verify(a):
    import torch
    from audioforge.hub import sha256_file
    from audioforge.model import StreamingSession
    from audioforge.teachers import load_audio
    from audioforge.train import load_model
    import enc0p6b as E0
    torch.set_num_threads(2)
    rec = {"afm": str(AFM), "afm_bytes": AFM.stat().st_size}
    if a.sha:
        t0 = time.time()
        rec["nemo_sha256"] = sha256_file(NEMO)
        rec["nemo_bytes"] = NEMO.stat().st_size
        log(f"sha256 {rec['nemo_sha256']} ({time.time() - t0:.0f} s)")
    t0 = time.time()
    m = load_model(str(AFM), "cpu").eval()
    rec["load_s"] = round(time.time() - t0, 1)
    sd = m.state_dict()
    rec["n_tensors"] = len(sd)
    rec["params_m"] = round(sum(v.numel() for v in sd.values()) / 1e6, 2)
    big = max(sd, key=lambda k: sd[k].numel())
    rec["largest_tensor_nonzero"] = bool(float(sd[big].abs().sum()) > 0)
    rec["encoder"] = {k: m.cfg["encoder"].get(k) for k in ("d_model", "n_layers", "n_heads", "feat_in", "att_context_sizes")}
    rows = E0._ls_rows()[:3]
    eq = {"utterances": [r["id"] for r in rows], "max_abs_diff": [], "act_scale": [], "frames_equal": [],
          "session_equals_offline": [], "text": []}
    for r in rows:
        audio = load_audio(ROOT / r["audio_filepath"])
        x, lens = torch.tensor(audio)[None], torch.tensor([len(audio)])
        with torch.no_grad():
            feats, fl = m.preprocessor(x, lens)
            off, olen = m.encoder(feats, fl, ATT)
            cs, T, st, outs = m.encoder.stream_chunk_frames(ATT), int(fl), None, []
            for s in range(0, T, cs):
                y, st = m.encoder.stream_step(feats[..., s:s + cs], st, ATT, final=s + cs >= T)
                outs.append(y)
        on = torch.cat(outs, 1)
        ok = on.shape[1] == int(olen)
        eq["frames_equal"].append(ok)
        eq["max_abs_diff"].append(float((on - off[:, :on.shape[1]]).abs().max()) if ok else None)
        eq["act_scale"].append(round(float(off.abs().mean()), 4))
        text = m.transcribe([audio], head="rnnt", att_context_size=ATT)[0]
        sess = StreamingSession(m, "rnnt", ATT)
        for s in range(0, len(audio), CHUNK):
            sess.feed(audio[s:s + CHUNK])
        st_text = sess.feed(audio[:0], final=True)
        eq["session_equals_offline"].append(st_text == text)
        eq["text"].append({"offline": text, "session": st_text, "ref": r["text"]})
        log(f"{r['id']}: frames {ok} max|d| {eq['max_abs_diff'][-1]:.2e} session==offline {st_text == text}")
    rec["equality_70_1"] = eq
    save("verify", rec)
    log("verify done")


# --------------------------------------------------------------------------- import + bare WER (--core 3.5)
LANG_TAG = __import__("re").compile(r"\s*<[a-z]{2,3}(?:-[A-Za-z]{2,4})?>")  # the auto-detect language tag token


def stage_import(a):
    """The .nemo -> AFM (audioforge.nemo_import: the prompt kernel goes into heads.rnnt.joint.enc), read back
    tensor-identical, sha256 of the .nemo."""
    import torch
    from audioforge.hub import sha256_file
    from audioforge.nemo_import import import_nemo
    from audioforge.train import load_model, save_model
    torch.set_num_threads(2)
    t0 = time.time()
    m = import_nemo(NEMO)
    info = {k: v for k, v in m.import_info.items() if isinstance(v, (int, float, str, bool))}
    AFM.parent.mkdir(parents=True, exist_ok=True)
    save_model(m, AFM)
    back = load_model(str(AFM), "cpu").state_dict()
    src = m.state_dict()
    assert set(back) == set(src) and all(torch.equal(back[k], src[k]) for k in src), "read-back mismatch"
    rec = {"params_m": round(m.num_params() / 1e6, 2), "n_tensors": len(src), "import": info,
           "att_context_sizes": m.encoder.att_context_sizes, "prompt": {k: v for k, v in
                                                                          m.cfg["heads"]["rnnt"]["prompt"].items()
                                                                          if k != "dictionary"},
           "n_prompt_keys": len(m.cfg["heads"]["rnnt"]["prompt"]["dictionary"]), "afm": str(AFM),
           "afm_bytes": AFM.stat().st_size, "sec": round(time.time() - t0, 1), "nemo_sha256": sha256_file(NEMO),
           "nemo_bytes": NEMO.stat().st_size}
    log(rec)
    save("import", rec)


WER_SETS = ("libri", "ami", "icsi")


def stage_wer(a):
    """Bare ASR WER at 160 ms chunks (masked [L,1] forward == cache-aware streaming, batch 4, greedy RNNT) on the
    hybrid_asr LibriSpeech-200 / AMI-200 / ICSI-200 sets, language prompt --which en-US | auto (auto: the detected
    language tag is stripped from the text and kept in 'tag'). -> W/wer/<prompt>_<set>.jsonl (resumable)."""
    import torch
    import hybrid_asr as H
    from audioforge.train import load_model
    torch.set_num_threads(2)
    m = load_model(str(AFM), a.device).eval()
    m.heads["rnnt"].set_prompt(a.which)
    m.tokenizer.specials = []  # keep the language-tag pieces in the text here (stripped below, kept in 'tag')
    od = W / "wer"
    od.mkdir(parents=True, exist_ok=True)
    t0, left = time.time(), []
    att = [ATT[0], a.att_right] if a.att_right is not None else list(ATT)
    sfx = "" if a.att_right is None else f"_r{a.att_right}"
    for set_name in WER_SETS:
        audios, refs = H.load_fa_set(set_name)
        p = od / f"{a.which}{sfx}_{set_name}.jsonl"
        done = len(p.read_text().splitlines()) if p.exists() else 0
        with p.open("a") as f:
            i = done
            while i < len(refs) and time.time() - t0 < a.budget:
                xs = audios[i:i + 4]
                t1 = time.perf_counter()
                with torch.inference_mode():
                    hyps = m.transcribe(xs, head="rnnt", att_context_size=att)
                dt = time.perf_counter() - t1
                for k, h in enumerate(hyps):
                    tags = LANG_TAG.findall(h)
                    f.write(json.dumps({"i": i + k, "hyp": LANG_TAG.sub("", h).strip(), "raw": h,
                                        "tag": [t.strip() for t in tags], "sec": dt / len(hyps),
                                        "audio_sec": len(xs[k]) / SR}) + "\n")
                i += len(xs)
        if i < len(refs):
            left.append(set_name)
    log("wer", a.which, "done" if not left else f"left {left}")


def stage_wer_report(a):
    """WER (normalize_text and whisper_norm, utterance bootstrap 95 % CI) of the 3.5 (en-US prompt / auto) beside
    the English 0.6B and the 115M at 160 ms (their hybrid_asr [70,1] hypotheses), paired deltas. -> OUT wer."""
    import hybrid_asr as H
    norms, _ = H._normalizers()
    od = W / "wer"
    res = {"protocol": f"masked [{ATT[0]},{ATT[1]}] forward (== cache-aware streaming at 160 ms), greedy RNNT, batch 4; "
                       "115M / English 0.6B: hybrid_asr served_la1 / served_0p6b_la1 ([70,1])", "sets": {}}
    for set_name in WER_SETS:
        _, refs = H.load_fa_set(set_name)
        srcs = {"115m": H.WORK / f"served_la1_{set_name}.jsonl", "0p6b_en": H.WORK / f"served_0p6b_la1_{set_name}.jsonl",
                "3p5_enUS": od / f"en-US_{set_name}.jsonl", "3p5_auto": od / f"auto_{set_name}.jsonl",
                "3p5_enUS_80ms": od / f"en-US_r0_{set_name}.jsonl", "3p5_enUS_320ms": od / f"en-US_r3_{set_name}.jsonl"}
        E, out = {}, {}
        for name, pth in srcs.items():
            rows = H._rows(pth)
            if len(rows) < len(refs):
                continue
            hy = [rows[i]["hyp"] for i in range(len(refs))]
            out[name] = {}
            for nn in ("normalize_text", "nofill", "whisper_norm"):
                e = np.array([H.edits_words(norms[nn](r), norms[nn](h)) for r, h in zip(refs, hy)], float)
                E[(name, nn)] = e
                out[name][nn] = H.rate_ci(e)
            if name == "3p5_auto":
                tags = [t for r in rows.values() for t in r.get("tag", [])]
                out[name]["tags"] = {t: tags.count(t) for t in sorted(set(tags))}
                out[name]["utts_without_tag"] = sum(1 for r in rows.values() if not r.get("tag"))
        out["paired"] = {f"{x} - {y} / {nn}": H.paired(E[(x, nn)], E[(y, nn)])
                         for x, y in (("3p5_enUS", "0p6b_en"), ("3p5_enUS", "115m"), ("3p5_auto", "3p5_enUS"),
                                      ("3p5_enUS_80ms", "0p6b_en"), ("3p5_enUS_320ms", "0p6b_en"))
                         for nn in ("normalize_text", "nofill") if (x, nn) in E and (y, nn) in E}
        res["sets"][set_name] = out
        log(set_name, {k: v["normalize_text"]["wer"] for k, v in out.items() if k != "paired"})
    save("wer", res)


# --------------------------------------------------------------------------- bare core cost
def _load_core(device, perf_on=True):
    import torch
    from audioforge import perf
    from audioforge.train import load_model
    torch.set_num_threads(2)
    m = load_model(str(AFM), device).eval()
    if perf_on:
        if device == "cpu":
            from audioforge.server.streams import fast_conv
            fast_conv(m)
        perf.apply(m, None, perf.DEFAULT)
    return m


def stage_core(a):
    import mps_115m as M
    dev = a.device
    t0 = time.time()
    m = _load_core(dev, not a.no_perf)
    load_s = time.time() - t0
    audios, refs = M.load_libri()
    sub = audios[:a.n_utt]
    M.time_stream(m, sub[0], dev)  # warm-up
    runs = []
    for _ in range(a.repeats):
        chunks, tot = [], 0.0
        for x in sub:
            c, t, _ = M.time_stream(m, x, dev)
            chunks += c
            tot += t
        runs.append({"rtf": tot / (sum(len(x) for x in sub) / SR), "chunks": chunks})
        log(f"run rtf {runs[-1]['rtf']:.3f}")
    best = min(runs, key=lambda r: r["rtf"])
    c = np.array(best["chunks"])
    rec = {"utterances": len(sub), "audio_s": round(sum(len(x) for x in sub) / SR, 1), "chunks": len(c),
           "chunk_ms_p50": round(float(np.percentile(c, 50)), 2), "chunk_ms_p95": round(float(np.percentile(c, 95)), 2),
           "chunk_ms_mean": round(float(c.mean()), 2), "rtf": round(best["rtf"], 4),
           "rtf_all": [round(r["rtf"], 4) for r in runs], "load_s": round(load_s, 1), "threads": 2,
           "perf": not a.no_perf, **M.mem(dev)}
    log(rec)
    save("core_cost", rec, sub=dev)


def stage_core_streams(a):
    import torch
    import mps_115m as M
    from audioforge.data import load_wav
    from audioforge.model import StreamingSession
    dev = a.device
    m = _load_core(dev, True)
    ks = [int(k) for k in a.ks.split(",")]
    clips = []
    for mtg, st in M.AMI_WINDOWS[: max(ks)]:
        x = load_wav(str(ROOT / "data" / "ami" / "audio" / f"{mtg}.Mix-Headset.wav"), SR).astype(np.float32)
        clips.append(x[st * SR:(st + int(a.seconds)) * SR])
    out = load_json("core_streams", {}).get(dev, {})
    s = StreamingSession(m, "rnnt", ATT)
    with torch.inference_mode():
        s.feed(clips[0][: 5 * SR])
    for K in ks:
        if str(K) in out:
            continue
        ss = [StreamingSession(m, "rnnt", ATT) for _ in range(K)]
        agg = []
        n = min(len(c) for c in clips[:K]) // CHUNK
        with torch.inference_mode():
            for i in range(n):
                M.sync(dev)
                t0 = time.perf_counter()
                for s, c in zip(ss, clips[:K]):
                    s.feed(c[i * CHUNK:(i + 1) * CHUNK])
                M.sync(dev)
                agg.append((time.perf_counter() - t0) * 1000)
        x = np.array(agg[5:])
        out[str(K)] = {"agg_block_ms_p50": round(float(np.median(x)), 1),
                       "agg_block_ms_p95": round(float(np.percentile(x, 95)), 1),
                       "per_stream_block_ms_p50": round(float(np.median(x)) / K, 1),
                       "real_time": bool(np.percentile(x, 95) < 160), **M.mem(dev)}
        log(K, out[str(K)])
        save("core_streams", out, sub=dev)
        if not out[str(K)]["real_time"]:
            break
    log("done")


# --------------------------------------------------------------------------- block probes
PROBE_BLOCKS = (4, 8, 12, 16, 20, 24)  # 1-based; every 4th block of the 24
VADW_OLD = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/core_0p6b/vad/" + ("115m" if C115 else "0p6b"))  # IMPROVEMENTS item 6 cache


def _vad_old(set_name, blocks):
    meta = json.loads((VADW_OLD / f"{set_name}.json").read_text())
    X = np.memmap(VADW_OLD / f"{set_name}.f16", np.float16, "r", shape=(meta["N"], meta["L"], meta["D"]))
    out = np.empty((meta["N"], len(blocks), meta["D"]), np.float16)
    for i in range(0, meta["N"], 4096):
        out[i:i + 4096] = X[i:i + 4096][:, [b - 1 for b in blocks]]
    return out, np.load(VADW_OLD / f"{set_name}_labels.npy"), np.load(VADW_OLD / f"{set_name}_win.npy")


def stage_vad_probe(a):
    """Which block carries VAD: the served VAD head's shape (Linear(1024,64)-SiLU-Linear(64,1), input dropout 0.2,
    Adam 1e-3 wd 1e-3, cosine, 2048-frame batches, BCE) on each of blocks 4/8/12/16/20/24 alone and on a softmax mix of
    the six, fitted on the 300 seeded AMI train diar windows, scored on AMI dev 64 x 20 s and ICSI dev 64 (held-out
    corpus) at 0.5 (vad_layers.vad_report)."""
    import torch
    import vad_layers as V
    torch.set_num_threads(2)
    Xtr, ytr, _ = _vad_old("train", PROBE_BLOCKS)
    evs = {s: _vad_old(s, PROBE_BLOCKS) for s in ("ami_dev", "icsi_dev")}
    res = load_json("vad_probe", {})
    for name in [f"b{b}" for b in PROBE_BLOCKS] + ["mix6"]:
        if name in res:
            continue
        sel = list(range(len(PROBE_BLOCKS))) if name == "mix6" else [PROBE_BLOCKS.index(int(name[1:]))]
        torch.manual_seed(0)
        mix = torch.nn.Parameter(torch.zeros(len(sel)))
        net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(1024, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam([mix, *net.parameters()], lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = np.random.default_rng(0)
        for step in range(a.steps):
            idx = np.sort(g.integers(0, len(Xtr), 2048))
            x = torch.from_numpy(Xtr[idx][:, sel].astype(np.float32))
            h = (x * mix.softmax(0)[None, :, None]).sum(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(h).squeeze(-1), torch.from_numpy(ytr[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        rec = {"mix": [round(float(v), 3) for v in mix.detach().softmax(0)]}
        for sn, (X, y, win) in evs.items():
            with torch.no_grad():
                p = np.concatenate([net((torch.from_numpy(X[i:i + 8192][:, sel].astype(np.float32))
                                         * mix.softmax(0)[None, :, None]).sum(1)).squeeze(-1).sigmoid().numpy()
                                    for i in range(0, len(X), 8192)])
            r = V.vad_report(V.per_window(p, win), V.per_window(y, win))
            rec[sn] = {k: r[k] for k in ("f1", "auc", "miss", "fpr") if k in r}
            rec[sn].update({k: v for k, v in r.items() if k.startswith("at_fpr")})
        res[name] = rec
        log(name, {sn: (rec[sn]["f1"], round(rec[sn]["auc"], 4)) for sn in evs})
        save("vad_probe", res)


# --------------------------------------------------------------------------- the 0.6B core + heads (in memory)
HEADS_DIR = W / "heads"
# the speaker head (spk_frame.py crop-level TitaNet distillation) and the TS-VAD head read block 5: the layer sweep
# (research/LAYER_SWEEP_0P6B.md; held-out meetings) put the speaker and TS-VAD probes' best blocks at 4-6, and the
# block-5 head beats the block-11 one (c0p6b_fresh) on AMI within-meeting EER 13.6 vs 16.2 % (ICSI 3.6 vs 2.9 %)
SPK_HEAD = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/spk_frame/head_c0p6b5_fresh.pt")  # spk_frame.py
SPK_TAP = 3 if C115 else 4  # 0-based: block 5 (115M: block 4, the shipped speaker head's tap)
VAD_BLOCKS = PROBE_BLOCKS  # the VAD head reads a softmax mix of blocks 4/8/12/16/20/24 (vad_probe: mix6 best on ICSI)
HEAD_CFGS = {
    # VAD v2 (user request, stage vad2 tag gru64_sa): causal GRU-64 frame head on the block mix, trained with
    # SpecAugment re-encoded views + feature dropout / noise / time masks + 25 % room-tone crops
    "vad": {"type": "frame_gru", "key": "vad", "hidden": 64, "from_layers": [b - 1 for b in VAD_BLOCKS], "weight": 0.5},
    "spk": {"type": "speaker", "num_speakers": 190, "from_layers": [SPK_TAP], "weight": 0.3, "aam_weight": 1.0},
}


def head_file(name) -> Path:
    return SPK_HEAD if name == "spk" else HEADS_DIR / f"{name}.pt"


def load_core(device="cpu", heads=("vad", "spk"), cfgs=None, fresh=()):
    """The imported 0.6B (.afm) + the trained heads ``heads`` (their HEAD_CFGS / ``cfgs`` entries and files; the ones
    in ``fresh`` are left at their initialisation)."""
    import copy
    import torch
    import audioforge.train as AT
    from audioforge.model import SpeechModel
    if C115:  # the shipped 115M (every served head as shipped; new heads are attached by the 115M stages)
        return getattr(AT, "_orig_load_model", AT.load_model)(str(AFM115), "cpu").to(device).eval()
    base = getattr(AT, "_orig_load_model", AT.load_model)(str(AFM), "cpu")
    cfgs = {**HEAD_CFGS, **(cfgs or {})}
    cfg = copy.deepcopy(base.cfg)
    cfg["encoder"]["att_context_size"] = list(ATT)  # served at 160 ms chunks (the import's default is [70, 13])
    cfg["heads"] = {**cfg["heads"], **{h: cfgs[h] for h in heads}}
    m = SpeechModel(cfg, base.tokenizer)
    sd = base.state_dict()
    for h in heads:
        if h in fresh:
            sd.update({f"heads.{h}.{k}": v for k, v in m.heads[h].state_dict().items()})
            continue
        ck = torch.load(head_file(h), map_location="cpu", weights_only=False)
        sd.update({f"heads.{h}.{k}": v for k, v in ck["state_dict"].items()})
        if "mix" in ck:
            sd[f"layer_mix.{h}"] = ck["mix"]
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


@__import__("functools").lru_cache(maxsize=None)
def _mask_fn():
    from audioforge.modules.fastconformer import chunked_attention_mask
    return chunked_attention_mask


def encode_blocks(m, audios, blocks, top=False, sa=None):
    """Masked [70,1] forward (= cache-aware streaming) of a batch; -> (list of {block: (T, D) fp16}, lengths, top
    (B, T, D) tensor or None). Only as many blocks as needed are run unless ``top``."""
    import torch
    enc = m.encoder
    dev = next(m.parameters()).device
    x, lens = m._pad(audios)
    with torch.inference_mode():
        feats, flen = m.preprocessor(x.to(dev), lens.to(dev))
        if sa is not None:  # SpecAugment on the mel input (the 115M head's training-time augmentation)
            feats = sa(feats, flen)
        h, hl = enc.pre_encode(feats, flen)
        if enc.xscale is not None:
            h = h * enc.xscale
        T = h.shape[1]
        am = _mask_fn()(T, hl, list(ATT))
        pm = torch.arange(T, device=h.device)[None] >= hl[:, None]
        need = len(enc.layers) if top else max(blocks)
        keep = {}
        for li in range(need):
            h, _, _ = enc.layers[li](h, am, pm)
            if li + 1 in blocks:
                keep[li + 1] = h.half().cpu().numpy()
        L = [int(v) for v in hl.cpu()]
    out = [{b: keep[b][j, : L[j]] for b in blocks} for j in range(len(audios))]
    return out, L, (h if top else None)


# --------------------------------------------------------------------------- VAD
VADW = W / "vad"
N_VAD_TRAIN = 1200
N_ROOM = {"train": 360, "held": 120}


def room_tone_items(split: str) -> list[dict]:
    """Noise-only negatives (HEAD_ARCHITECTURES.md VAD v3, the cheap part): 8 s clips of (a) the quietest 1.5 s runs
    of AMI train meetings with nobody annotated speaking, tiled, (b) white / pink / brown noise, (c) near-digital
    silence, each at a level drawn from -70..-35 dBFS (-90 for c); label 0 everywhere. ``held`` uses other
    meetings / seeds (the room-tone check)."""
    from audioforge.datasets.ami import AMI, split_lists
    rng = np.random.default_rng(0 if split == "train" else 1)
    root = ROOT / "data" / "ami" / "audio"
    ms = sorted(m for m in split_lists()["train"] if (root / f"{m}.Mix-Headset.wav").exists())
    ms = ms[: len(ms) // 2] if split == "train" else ms[len(ms) // 2:]
    ds = AMI(ms, verbose=False)
    gaps = []
    for mt in ms:
        iv = sorted(i for s in ds.acts[mt].values() for i in s)
        end = 0.0
        for a0, b0 in iv:
            if a0 - end >= 2.0:
                gaps.append((mt, end + 0.25, a0 - 0.25))
            end = max(end, b0)
    rng.shuffle(gaps)
    out, n = [], N_ROOM[split]
    L = 8 * SR
    for i in range(n):
        kind = i % 3
        db = float(rng.uniform(-70, -35))
        if kind == 0 and gaps:
            mt, a0, b0 = gaps[i % len(gaps)]
            x = np.asarray(ds._clip(mt, a0, min(b0, a0 + 4.0)), np.float32)
            x = np.tile(x, L // max(1, len(x)) + 1)[:L]
            src = f"ami_gap:{mt}:{a0:.1f}"
        elif kind == 1 or not gaps:
            w = rng.standard_normal(L).astype(np.float32)
            col = ["white", "pink", "brown"][int(rng.integers(3))]
            if col != "white":
                f = np.fft.rfft(w)
                k = np.arange(len(f)) + 1.0
                f /= np.sqrt(k) if col == "pink" else k
                w = np.fft.irfft(f, L).astype(np.float32)
            x, src = w, col
        else:
            x, src, db = rng.standard_normal(L).astype(np.float32), "near_silence", -90.0
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
        x = (x / rms * 10 ** (db / 20)).astype(np.float32)
        out.append({"id": f"room_{split}_{i:04d}", "audio": x, "vad": np.zeros(L // 1280 + 2, np.float32),
                    "src": src, "dbfs": round(db, 1)})
    return out


def centre_labels(v: dict, T: int):
    """Frame-centre VAD labels (HEAD_ARCHITECTURES.md: label by the frame centre +-20 ms, not "any part of the 80 ms
    frame") from the window's speaker intervals, when the window carries them; else None."""
    acts = v.get("acts") or v.get("spk_intervals")
    if acts is None:
        return None
    off = float(v.get("start", v.get("offset", 0.0)))
    y = np.zeros(T, np.float32)
    c = (np.arange(T) + 0.5) * 0.08 + off
    for ivs in (acts.values() if isinstance(acts, dict) else acts):
        for a0, b0 in ivs:
            y[(c + 0.02 > a0) & (c - 0.02 < b0)] = 1
    return y


def stage_vad_feats(a):
    """Blocks 4/8/12/16/20/24 of the VAD training windows (N_VAD_TRAIN seeded AMI train diar windows,
    vad_layers.load_set; labels "any part of the frame" as the 115M head, plus frame-centre labels when available)
    and of the room-tone negatives -> VADW/<set>.f16 memmap (N, 6, 1024) + labels + window ids. Resumable."""
    import torch
    import vad_layers as V
    torch.set_num_threads(2)
    VADW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = None
    blocks = [int(b) for b in a.vad_blocks.split(",") if b.strip()] if a.vad_blocks else list(VAD_BLOCKS)
    for spec in a.vad_sets.split(","):
        base_name, _, btag = spec.partition(":")  # "train1200:b" -> set train1200, files train1200_b.*
        set_name = f"{base_name}_{btag}" if btag else base_name
        meta_p = VADW / f"{set_name}.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        if meta and meta.get("done") == meta.get("n"):
            continue
        if base_name.startswith("train"):
            from audioforge.datasets.ami import AMI
            val = V.load_set("train", int(base_name[5:] or N_VAD_TRAIN))
            ds = AMI(sorted({v["meeting"] for v in val}), verbose=False)
            for v in val:  # every annotated speaker of the meeting (dropped columns included) for centre labels
                v["acts"] = ds.acts[v["meeting"]]
        elif base_name.startswith("icsival"):  # ICSI TRAIN meetings (the ICSI test set is the dev meetings)
            from audioforge.datasets.icsi import ICSI, recipe_data as icsi_rd
            from audioforge.train import derive_labels
            val = [derive_labels(dict(v), ["vad"]) for v in icsi_rd(
                {"data": {"icsi": {"mode": "diar", "n_train": int(base_name[7:] or 150), "seed": 0}}}, "train")]
            ds = ICSI(sorted({v["meeting"] for v in val}), verbose=False)
            for v in val:
                v["acts"] = ds.acts[v["meeting"]]
        else:
            val = room_tone_items(base_name.split("_")[1])
        if m is None:
            m = load_core(a.device, heads=())
        Ts = [min(int(m.encoder.pre_encode.out_lengths(m.preprocessor.num_frames(torch.tensor(len(v["audio"]))))),
                  len(v["vad"])) for v in val]
        off = np.r_[0, np.cumsum(Ts)]
        N = int(off[-1])
        X = np.memmap(VADW / f"{set_name}.f16", np.float16, "r+" if meta else "w+", shape=(N, len(blocks), D))
        if not meta:
            lab = np.zeros(N, np.float32)
            cen = np.full(N, np.nan, np.float32)
            win = np.zeros(N, np.int32)
            for w, v in enumerate(val):
                lab[off[w]: off[w + 1]] = v["vad"][: Ts[w]]
                c = centre_labels(v, Ts[w])
                if c is not None:
                    cen[off[w]: off[w + 1]] = c
                win[off[w]: off[w + 1]] = w
            np.save(VADW / f"{set_name}_labels.npy", lab)
            np.save(VADW / f"{set_name}_centre.npy", cen)
            np.save(VADW / f"{set_name}_win.npy", win)
            if set_name.startswith("room"):
                (VADW / f"{set_name}_items.json").write_text(json.dumps([{k: v[k] for k in ("id", "src", "dbfs")}
                                                                         for v in val]))
            else:
                (VADW / f"{set_name}_meetings.json").write_text(json.dumps([v["meeting"] for v in val]))
        done = int(meta.get("done", 0))
        sa = None
        if btag.startswith("sa"):  # "train:sa1": SpecAugment view, seed 1 (2 x 27 freq, 10 x 5 % time masks)
            from audioforge.features import SpecAugment
            torch.manual_seed(int(btag[2:]) * 1000 + done)
            sa = SpecAugment(2, 27, 10, 0.05).train()
        for i in range(done, len(val), a.batch):
            outs, L, _ = encode_blocks(m, [np.asarray(v["audio"], np.float32) for v in val[i: i + a.batch]], blocks,
                                       sa=sa)
            for j, o in enumerate(outs):
                w = i + j
                X[off[w]: off[w + 1]] = np.stack([o[b][: Ts[w]] for b in blocks], 1)
            done = min(i + a.batch, len(val))
            X.flush()
            meta_p.write_text(json.dumps({"done": done, "n": len(val), "N": N, "blocks": list(blocks)}))
            if time.time() - t0 > a.budget:
                log(f"vad_feats {set_name}: {done}/{len(val)} (budget)")
                return
        Xr = np.memmap(VADW / f"{set_name}.f16", np.float16, "r", shape=(N, len(blocks), D))
        assert np.any(Xr[-20:].astype(np.float32)), "read-back zero-filled"
        log(f"vad_feats {set_name}: {len(val)} windows, {N} frames ({time.time() - t0:.0f} s)")


def _vad_set(set_name):
    meta = json.loads((VADW / f"{set_name}.json").read_text())
    assert meta["done"] == meta["n"], (set_name, meta)
    X = np.memmap(VADW / f"{set_name}.f16", np.float16, "r", shape=(meta["N"], len(meta.get("blocks", VAD_BLOCKS)), D))
    return X, np.load(VADW / f"{set_name}_labels.npy"), np.load(VADW / f"{set_name}_centre.npy"), \
        np.load(VADW / f"{set_name}_win.npy")


def stage_vad_train(a):
    """The served VAD head on the 0.6B: FrameHead(1024, hidden 64) on a softmax mix of blocks 4/8/12/16/20/24
    (HEAD_CFGS["vad"]), BCE, AdamW 1e-3 wd 1e-3, cosine, --steps x 2048 frames, input dropout 0.1; training frames =
    the AMI train windows (frame-centre labels where available, else the 115M head's any-part labels) + room-tone
    negatives at share --room-share. -> HEADS_DIR/vad.pt; scored on AMI dev / ICSI dev (vad_probe protocol) and on the
    held-out room tone (median P(speech), share of frames > 0.5)."""
    import torch
    import vad_layers as V
    from audioforge.model import build_head
    torch.set_num_threads(2)
    dev = a.device
    Xtr, ytr, ctr, _ = _vad_set(a.vad_train)
    Xr, yr, _, _ = _vad_set("room_train")
    y_use = np.where(np.isnan(ctr), ytr, ctr) if a.centre else ytr
    log(f"train frames {len(Xtr)} (centre labels on {int((~np.isnan(ctr)).sum())}), room {len(Xr)}")
    Xtr_m = np.asarray(Xtr)  # into RAM (N x 6 x 1024 fp16)
    Xr_m = np.asarray(Xr)
    torch.manual_seed(a.seed)
    head = build_head(HEAD_CFGS["vad"], 1024).to(dev)
    mix = torch.nn.Parameter(torch.zeros(len(VAD_BLOCKS), device=dev))
    drop = torch.nn.Dropout(a.drop)
    opt = torch.optim.AdamW([mix, *head.parameters()], lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = np.random.default_rng(a.seed)
    nb = int(2048 * a.room_share)
    t0 = time.time()
    for step in range(a.steps):
        i1 = np.sort(g.integers(0, len(Xtr_m), 2048 - nb))
        i2 = np.sort(g.integers(0, len(Xr_m), nb))
        x = torch.from_numpy(np.concatenate([Xtr_m[i1], Xr_m[i2]]).astype(np.float32)).to(dev)
        y = torch.from_numpy(np.concatenate([y_use[i1], yr[i2]]).astype(np.float32)).to(dev)
        h = drop((x * mix.softmax(0)[None, :, None]).sum(1))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(h), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 1000 == 0:
            log(f"step {step + 1} loss {float(loss):.4f} ({time.time() - t0:.0f} s)")
    head.eval()
    HEADS_DIR.mkdir(parents=True, exist_ok=True)
    w = mix.detach().cpu()

    def predict(X):
        with torch.no_grad():
            return np.concatenate([head((torch.from_numpy(np.asarray(X[i:i + 8192], np.float32)).to(dev)
                                         * w.to(dev).softmax(0)[None, :, None]).sum(1)).sigmoid().cpu().numpy()
                                   for i in range(0, len(X), 8192)])
    rec = {"args": {k: v for k, v in vars(a).items() if k in ("steps", "room_share", "centre", "seed", "drop")},
           "mix": [round(float(v), 3) for v in w.softmax(0)], "train_frames": int(len(Xtr)), "room_frames": int(len(Xr))}
    for sn in ("ami_dev", "icsi_dev"):
        X, y, win = _vad_old(sn, VAD_BLOCKS)
        p = predict(X)
        r = V.vad_report(V.per_window(p, win), V.per_window(y, win))
        rec[sn] = {k: r[k] for k in ("f1", "auc", "miss", "fpr") if k in r}
        rec[sn].update({k: v for k, v in r.items() if k.startswith("at_fpr")})
        np.save(VADW / f"pred_{a.tag}_{sn}.npy", p)
    Xh, _, _, _ = _vad_set("room_held")
    ph = predict(Xh)
    rec["room_held"] = {"median_p": round(float(np.median(ph)), 4), "p95_p": round(float(np.percentile(ph, 95)), 4),
                        "share_gt_0.5": round(float((ph > 0.5).mean()), 4), "frames": int(len(ph))}
    log(rec)
    torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "mix": w, "cfg": HEAD_CFGS["vad"],
                "eval": rec}, HEADS_DIR / f"vad_{a.tag}.pt")
    save("vad", rec, sub=a.tag)


def stage_vad_room115(a):
    """The shipped 115M VAD head on the same held-out room-tone clips (the comparison row)."""
    import torch
    from audioforge.train import load_model
    torch.set_num_threads(2)
    m = load_model(str(ROOT / "runs" / "stage1_served_v2.afm"), a.device).eval()
    items = room_tone_items("held")
    P = []
    with torch.inference_mode():
        for i in range(0, len(items), 8):
            x, xl = m._pad([v["audio"] for v in items[i:i + 8]])
            enc, elen, hid = m.encode(x.to(a.device), xl.to(a.device), ATT, return_hidden=True)
            p = m.heads["vad"](m.head_input("vad", enc, hid)).sigmoid().cpu().numpy()
            P += [p[j, : int(elen[j])] for j in range(len(elen))]
    ph = np.concatenate(P)
    rec = {"median_p": round(float(np.median(ph)), 4), "p95_p": round(float(np.percentile(ph, 95)), 4),
           "share_gt_0.5": round(float((ph > 0.5).mean()), 4), "frames": int(len(ph))}
    log(rec)
    save("vad_room_115m", rec)

# --------------------------------------------------------------------------- TS-VAD (scripts/research/tsvad.py recipe)
TSV = W / "tsvad_b5"  # tsvad.CACHE layout (block 5; W / "tsvad" = the first block-11 pass): feats/<corpus>_<meeting>.npz, enroll/<...>.npz (+ sim_*)
TSV_WORK = W / "tsvad_work_b5"  # tsvad.WORK layout: <corpus>/feat/<key>.npy, <corpus>/vprints.npz
if C115:  # block SPK_TAP + 1; the shipped block-4 caches (tsvad.CACHE / tsvad.WORK) are linked in, never written
    TSV, TSV_WORK = W / f"tsvad_b{SPK_TAP + 1}", W / f"tsvad_work_b{SPK_TAP + 1}"


def _tsvad_mod(device="cpu"):
    """scripts/research/tsvad.py pointed at the 0.6B: block 11 (SPK_TAP) features, the 0.6B speaker head for the
    enrollment clips / voice prints, caches under W."""
    import functools
    import tsvad as T
    if C115 and not getattr(T, "_0p6b", False):
        if SPK_TAP == 3 and not (TSV / "feats").exists():  # link the shipped block-4 train / enroll caches
            for sub in ("feats", "enroll"):
                (TSV / sub).mkdir(parents=True, exist_ok=True)
                for f in sorted((T.CACHE / sub).glob("*.npz")):
                    (TSV / sub / f.name).symlink_to(f)
            for c in ("ami", "icsi"):
                (TSV_WORK / c).mkdir(parents=True, exist_ok=True)
                for f in ("feat", "vprints.npz"):
                    if (T.WORK / c / f).exists() and not (TSV_WORK / c / f).exists():
                        (TSV_WORK / c / f).symlink_to(T.WORK / c / f)
    if not getattr(T, "_0p6b", False):
        orig = T.block_feats
        T.block_feats = functools.partial(orig, layer=SPK_TAP)
        T.TAP, T.CACHE, T.WORK, T._0p6b = SPK_TAP, TSV, TSV_WORK, True
        T._model = None

        def load_served(device=device, path=None):
            if T._model is None:
                T._model = load_core(device, heads=("spk",))
            return T._model
        T.load_served = load_served
    return T


def stage_tsvad_feats(a):
    """tsvad.py trainfeats (16 s crops, hop 12 s, 12 AMI + 12 ICSI train meetings) and enroll (32 clips of 1-5 s per
    (meeting, speaker), embedded by the 0.6B speaker head) on block 11 of the 0.6B."""
    T = _tsvad_mod(a.device)
    a.titanet = False
    T.stage_trainfeats(a)
    T.stage_enroll(a)


N_SIM_MEET, SIM_CROPS, SIM_SPK = 30, 40, 6


def _libri_utts():
    root = ROOT / "data" / "librispeech" / "LibriSpeech" / "train-clean-100"
    spk = sorted(p.name for p in root.iterdir() if p.is_dir())
    return root, spk


def _trim(x, db=40.0):
    n = len(x) // 160
    if n < 3:
        return x
    e = 10 * np.log10(np.mean(x[: n * 160].reshape(n, 160) ** 2, 1) + 1e-10)
    on = np.nonzero(e > e.max() - db)[0]
    return x[on[0] * 160: (on[-1] + 1) * 160]


def stage_tsvad_sim(a):
    """Simulated conversations for TS-VAD (HEAD_ARCHITECTURES.md TS-VAD v2 stage 1, the cheap part): N_SIM_MEET
    "meetings" of SIM_SPK LibriSpeech train-clean-100 speakers; each has SIM_CROPS 16 s crops in which 2-3 of them
    talk in turns (one utterance each, trimmed, <= 6 s, gain +-6 dB), gaps U(-0.8, 1.2) s (negative = overlap),
    room-tone noise at -60..-40 dBFS on 70 % of crops, level ~N(-28, 5) dBFS; enrollment = 32 clips of 1-5 s per
    speaker from utterances the crops do not use. Written in the tsvad.py cache format (feats / enroll) as sim_XX."""
    import soundfile as sf
    from audioforge.datasets.ami import frames
    T = _tsvad_mod(a.device)
    model = T.load_served()
    root, spks = _libri_utts()
    t0 = time.time()
    for k in range(N_SIM_MEET):
        name = f"sim_{k:02d}"
        if (TSV / "feats" / f"{name}.npz").exists() and (TSV / "enroll" / f"{name}.npz").exists():
            continue
        if time.time() - t0 > a.budget:
            log(f"tsvad_sim: budget at {name}")
            return
        rng = np.random.default_rng(100 + k)
        pick = [spks[i] for i in rng.choice(len(spks), SIM_SPK, replace=False)]
        files = {s_: sorted((root / s_).glob("*/*.flac")) for s_ in pick}
        for s_ in pick:
            rng.shuffle(files[s_])
        cut = {s_: len(files[s_]) // 3 for s_ in pick}  # the first third is the enrollment pool
        ptr = {s_: cut[s_] for s_ in pick}

        def utt(s_):
            f = files[s_][ptr[s_]]
            ptr[s_] = ptr[s_] + 1 if ptr[s_] + 1 < len(files[s_]) else cut[s_]
            x, sr = sf.read(str(f), dtype="float32")
            return _trim(x)[: 6 * SR]
        L = 16 * SR
        auds, acts = [], []
        for c in range(SIM_CROPS):
            act_spk = [pick[i] for i in rng.choice(SIM_SPK, int(rng.integers(2, 4)), replace=False)]
            y = np.zeros(L, np.float32)
            ivs = {s_: [] for s_ in pick}
            t = float(rng.uniform(0.0, 1.0))
            j = 0
            while t < 15.5:
                s_ = act_spk[j % len(act_spk)] if rng.random() < 0.8 else act_spk[int(rng.integers(len(act_spk)))]
                x = utt(s_) * 10 ** (rng.uniform(-6, 6) / 20)
                a0 = int(t * SR)
                b0 = min(L, a0 + len(x))
                y[a0:b0] += x[: b0 - a0]
                ivs[s_].append((a0 / SR, b0 / SR))
                t = b0 / SR + float(rng.uniform(-0.8, 1.2))
                j += 1
            if rng.random() < 0.7:
                nz = rng.standard_normal(L).astype(np.float32)
                y += nz / np.sqrt(np.mean(nz ** 2)) * 10 ** (rng.uniform(-60, -40) / 20)
            y *= 10 ** (rng.normal(-28, 5) / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9)
            auds.append(np.clip(y, -1, 1).astype(np.float32))
            acts.append(ivs)
        fs = T.block_feats(model, auds)
        Tm = min(len(f) for f in fs)
        F = np.stack([f[:Tm].astype(np.float16) for f in fs])
        A = np.stack([np.stack([frames(ivs[s_], Tm, 0.0) for s_ in pick], 1) for ivs in acts]).astype(np.uint8)
        T.save_npz(TSV / "feats" / f"{name}.npz", feats=F, act=A, starts=np.zeros(len(F)) - 100.0,
                   speakers=np.array(pick))
        espk, eauds = [], []
        for s_ in pick:
            pool = [_trim(sf.read(str(f), dtype="float32")[0]) for f in files[s_][: min(cut[s_], 12)]]
            for _ in range(T.N_ENROLL):
                x = pool[int(rng.integers(len(pool)))]
                n = int(rng.uniform(*T.ENROLL_SEC) * SR)
                o = int(rng.integers(0, max(1, len(x) - n)))
                espk.append(s_)
                eauds.append(x[o: o + n])
        E = T.spk_embed(model, T.block_feats(model, eauds))
        T.save_npz(TSV / "enroll" / f"{name}.npz", spk=np.array(espk), emb=E.astype(np.float32),
                   ivs=np.full((len(espk), 8, 2), -1.0), dur=np.array([len(x) / SR for x in eauds]))
        log(f"  {name}: {len(F)} crops x {Tm} frames, speakers {pick} ({time.time() - t0:.0f} s)")
    log("tsvad_sim done")


def stage_tsvad_train(a):
    """tsvad.py train on the 0.6B (TSVADHead(1024, emb 192, hidden 128, prenet), AdamW 2e-3 one-cycle, batch 64,
    201 frames, p_drop 0.15) + the cheap HEAD_ARCHITECTURES.md improvements: simulated conversations (sim_*, share
    --sim-share of the batches; AMI and ICSI split the rest 50/50) and an overlap-weighted loss (frames where the
    target and someone else both speak weigh --ov-w). -> HEADS_DIR/tsvad_<tag>.pt (runs/tsvad_*.pt format)."""
    import torch
    import torch.nn.functional as F
    from audioforge.heads.tsvad import TSVADHead
    T = _tsvad_mod(a.device)
    torch.set_num_threads(2)
    dev = a.device
    names = sorted(p.stem for p in (TSV / "feats").glob("*.npz"))
    if a.sim_share <= 0:
        names = [n for n in names if not n.startswith("sim_")]
    val = [n for n in names if n in T.VAL_MEETINGS] if a.holdout else []
    tr = [n for n in names if n not in val]
    torch.manual_seed(a.seed)
    data = T.TrainData(tr, "emb", p_drop=0.15, seed=a.seed)
    corpus = np.array([tr[k].split("_")[0] for k in data.src])
    share = {"ami": (1 - a.sim_share) / 2, "icsi": (1 - a.sim_share) / 2, "sim": a.sim_share}
    w = np.array([share[c] / max(1, (corpus == c).sum()) for c in corpus])
    data.w = w / w.sum()
    vdata = T.TrainData(val, "emb", p_drop=0.0, seed=123) if val else None
    head = TSVADHead(D, emb_dim=192, hidden=128, prenet=True).to(dev)
    nparam = sum(p_.numel() for p_ in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.steps, pct_start=0.05)
    hist, t0 = [], time.time()
    vb = [vdata.batch(64) for _ in range(8)] if vdata else []
    for step in range(1, a.steps + 1):
        head.train()
        x, y, e, has = data.batch(64)
        x, y, e, has = x.to(dev), y.to(dev), e.to(dev), has.to(dev)
        Tn = min(x.shape[1], y.shape[1], 201)
        x, y = x[:, :Tn], y[:, :Tn]
        logits = head(x, None, e, has)
        wt = 1.0 + (a.ov_w - 1.0) * ((y[..., 0] > 0.5) & (y[..., 1] > 0.5)).float()
        l = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none") * wt[..., None]).mean()
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps:
            rec = {"step": step, "loss": round(float(l), 4), "sec": round(time.time() - t0, 1)}
            if vb:
                head.eval()
                P, Y = [], []
                with torch.no_grad():
                    for xv, yv, ev, hv in vb:
                        P.append(head.decode(xv.to(dev), None, ev.to(dev), hv.to(dev)).cpu().numpy())
                        Y.append(yv.numpy())
                P, Y = np.concatenate(P), np.concatenate(Y)
                rec["val_f1_target"] = round(T.frame_prf(P[..., 0], Y[..., 0])[2], 4)
                ov = (Y[..., 0] > 0.5) & (Y[..., 1] > 0.5)
                rec["val_recall_target_overlap"] = round(float((P[..., 0][ov] > 0.5).mean()), 4) if ov.any() else None
            hist.append(rec)
            log(json.dumps(rec))
    cfg = {"type": "tsvad", "from_layers": [SPK_TAP], "emb_dim": 192, "hidden": 128, "prenet": True,
           "enroll_embedder": "spk", "weight": 0.0}
    HEADS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "cfg": cfg, "params": nparam,
                "args": vars(a), "train": tr, "val": val, "history": hist}, HEADS_DIR / f"tsvad_{a.tag}.pt")
    log(f"tsvad_train {a.tag}: {nparam} params, {time.time() - t0:.0f} s")


def stage_tsvad_win(a):
    """eot-bench v2 extended windows (AMI dev 974, ICSI held-out 1312): block-11 features (tsvad.py winfeats) and
    voice prints (vprints: 5 s / 1.5 s clips elsewhere in the meeting, embedded by the 0.6B speaker head)."""
    T = _tsvad_mod(a.device)
    a.corpora = a.corpora or "ami,icsi"
    T.stage_winfeats(a)
    left = sum(1 for c in a.corpora.split(",") for v in T.bench_windows(c)[0]
               if not (TSV_WORK / c / "feat" / f"{T.wkey(v)}.npy").exists())
    if left:
        log(f"winfeats: {left} left")
        return
    T.stage_vprints(a)


def stage_tsvad_eval(a):
    """Target-speaker frame metrics (tsvad.py frame protocol: every labelled column speaker with a voice print as the
    target; 'primary' = column 0, the turn's speaker; p > 0.5; tracking F1 and DER = miss + FA over target frames) of
    the 0.6B TS-VAD head(s) --heads tag,... -> runs/core_0p6b.json tsvad_frame."""
    import torch
    from audioforge.heads.tsvad import TSVADHead
    T = _tsvad_mod("cpu")
    torch.set_num_threads(2)
    heads = {}
    for tag in a.heads.split(","):
        ck = torch.load(HEADS_DIR / f"tsvad_{tag}.pt", map_location="cpu", weights_only=False)
        c = {k: v for k, v in ck["cfg"].items() if k not in ("type", "from_layers", "weight", "enroll_embedder")}
        h = TSVADHead(D, **c)
        h.load_state_dict(ck["state_dict"])
        heads[tag] = h.eval()
    res = load_json("tsvad_frame", {})
    for corpus in (a.corpora or "ami,icsi").split(","):
        ext, meta, ds = T.bench_windows(corpus)
        kidx, VP = T.load_vprints(corpus)
        st = {"primary": T.FrameStats(), "all_speakers": T.FrameStats()}
        n_t = 0
        for v in ext:
            k = kidx[T.wkey(v)]
            f = T.win_feats(corpus, v)
            y = np.asarray(v["spk_targets"], np.float32)
            cls = T.frame_classes(y)
            for c in range(len(T.window_speakers(ds, v))):
                yt = y[:, c]
                if not yt.any() or not VP["has_5p0"][k, c]:
                    continue
                groups = ["primary", "all_speakers"] if c == 0 else ["all_speakers"]
                for tag, h in heads.items():
                    for L in ("5p0", "1p5"):
                        if VP[f"has_{L}"][k, c]:
                            pt = T.tsvad_probs(h, f, VP[f"spk_{L}"][k, c])[:, 0]
                            for g in groups:
                                st[g].add(f"tsvad_{tag}_vp{L}", pt, yt, cls)
                n_t += 1
        out = {g: s_.summary() for g, s_ in st.items()}
        for g in out.values():
            for sysn in g.values():
                for cl in sysn.values():
                    if cl.get("miss") is not None and cl["target_frames"]:
                        fa_frames = cl["fa"] * cl["nontarget_frames"]
                        cl["der"] = round(cl["miss"] + fa_frames / cl["target_frames"], 4)
        out["n_windows"], out["n_targets"] = len(ext), n_t
        res[corpus] = out
        log(corpus, {k_: v_["all"] for k_, v_ in out["primary"].items()})
    save("tsvad_frame", res)


# --------------------------------------------------------------------------- turn-head inputs (turn_v4 / turn_v5 sets)
TURN = W / "turn"
TURN_INP, TURN_BLK = TURN / "inp", TURN / "blk"  # served-input npz (vad, pu, po, y, n) / blocks npz (b<k>)
TURN_BLOCKS = (8, 12, 24)
PROBE_MOD = 6  # clips whose stable hash % PROBE_MOD == 0 also keep blocks 4 / 16 / 20 (the block probe)


def _probe_clip(cid: str) -> bool:
    import zlib
    return zlib.crc32(cid.encode()) % PROBE_MOD == 0


def turn_items(which: str):
    """(id, audio fn, print-audio fn or None) of the turn_v5 training sets: 'man' = the turn_v4 manifest (oto, AMI,
    ICSI clips with a 5 s print of the user from elsewhere; smart-turn human_5_all clips printed by themselves),
    'cuts' (smart-turn style negatives, the source clip's print), 'st3' (smart-turn clips + 3 s silence, no print)."""
    import turn_v4 as V4
    import turn_v5 as V5
    man = V5.manifest()
    au = V4.ClipAudio()
    if which == "man":
        return [(c["id"], (lambda c=c: au(c, man)[0]), (lambda c=c: au(c, man)[1])) for c in man]
    if which == "stt":  # smart-turn human_5_all eval-split clips (the stest set), printed by themselves
        from audioforge.datasets import smartturn as ST
        meta = json.loads(ST.build_cache(verbose=False).read_text())
        return [(f"stt_{i:05d}", (lambda i=i: au({"src": "st", "idx": i}, [])[0]),
                 (lambda i=i: au({"src": "st", "idx": i}, [])[1])) for i in ST.split_indices(meta)["eval"]]
    idx = {c["id"]: c for c in man}
    out = []
    for cid, fn, pk, _ in V5.extract_items(which, man, au):
        out.append((cid, fn, None if pk is None else (lambda pk=pk: au(idx[pk], man)[1]), pk))
    return [(i, f, p) for i, f, p, _ in out]


class TurnExtractor:
    """The served turn inputs of a clip on the 0.6B (turn_v4.Extractor's pass 1): VAD head, block-11 TS-VAD track
    (tsvad_stream.track_probs with the stored print set before frame 0, anchored adaptation on), greedy RNNT tokens
    and their frames (top block), plus the kept blocks."""

    def __init__(self, device, vad="vad", tsvad="tsvad"):
        import torch
        from audioforge.tsvad_stream import load_tsvad
        self.torch = torch
        self.m = load_core(device, heads=("vad", "spk"))
        self.dev = device
        self.tsvad = load_tsvad(str(HEADS_DIR / f"{tsvad}.pt"), 1024)
        self.spk_cpu = __import__("copy").deepcopy(self.m.heads["spk"]).cpu().eval()
        self.rnnt_cpu = __import__("copy").deepcopy(self.m.heads["rnnt"]).cpu().eval()

    def prints(self, audios):
        from audioforge.tsvad_stream import embed_frames
        ok = [i for i, x in enumerate(audios) if x is not None and len(x) >= 8000]
        out = [None] * len(audios)
        if ok:
            fs, L, _ = encode_blocks(self.m, [audios[i] for i in ok], (SPK_TAP + 1,))
            for i, f in zip(ok, fs):
                out[i] = embed_frames(self.spk_cpu, f[SPK_TAP + 1].astype(np.float32))
        return out

    def __call__(self, audios, prints, blocks):
        torch = self.torch
        from audioforge.heads.turn import greedy_decode_frames, token_counts
        from audioforge.tsvad_stream import track_probs
        need = sorted(set(blocks) | set(VAD_BLOCKS) | {SPK_TAP + 1})
        outs, L, top = encode_blocks(self.m, audios, need, top=True)
        res = []
        w = self.m.layer_mix["vad"].detach().softmax(0).cpu().numpy()
        for i, o in enumerate(outs):
            n = L[i]
            hv = sum(w[j] * o[b].astype(np.float32) for j, b in enumerate(VAD_BLOCKS))
            vad = self._vad(hv)
            if prints[i] is not None:
                P = track_probs(self.tsvad, self.spk_cpu, o[SPK_TAP + 1].astype(np.float32), prints[i])
            else:
                P = np.zeros((n, 2), np.float32)
            fa = top[i, :n].float().cpu()
            hyp, fr = greedy_decode_frames(self.rnnt_cpu, fa)
            nn_ = token_counts(torch.tensor([fr], dtype=torch.long), torch.tensor([len(hyp)]), n)[0].numpy() \
                if hyp else np.zeros(n, np.int64)
            inp = {"vad": vad.astype(np.float16), "pu": P[:, 0].astype(np.float16), "po": P[:, 1].astype(np.float16),
                   "y": np.asarray(hyp, np.int32), "n": nn_.astype(np.int32)}
            res.append((inp, {f"b{b}": o[b] for b in blocks}))
        return res

    def _vad(self, hv):
        torch = self.torch
        head = self.m.heads["vad"]
        with torch.no_grad():
            x = torch.from_numpy(hv).to(next(head.parameters()).device)
            return torch.sigmoid(head(x[None]))[0].float().cpu().numpy()


def stage_turn_cache(a):
    """Turn-head inputs of the turn_v5 sets (--set man|cuts|st3) on the 0.6B -> TURN_INP/<id>.npz (vad, pu, po, y,
    n: turn_v4's served-input format) + TURN_BLK/<id>.npz (blocks 8 / 12 / 24; + 4 / 16 / 20 on the probe subset).
    Needs HEADS_DIR/vad.pt and tsvad.pt. Resumable, --budget s per call."""
    import torch
    torch.set_num_threads(2)
    TURN_INP.mkdir(parents=True, exist_ok=True)
    TURN_BLK.mkdir(parents=True, exist_ok=True)
    items = turn_items(a.set)
    todo = [it for it in items if not (TURN_INP / f"{it[0]}.npz").exists()]
    log(f"turn_cache {a.set}: {len(todo)} of {len(items)} to do")
    if not todo:
        return
    t0 = time.time()
    ex = TurnExtractor(a.device)
    prc = {}
    done = 0
    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > a.budget:
            break
        cs = todo[i: i + a.batch]
        xs = [fn() for _, fn, _ in cs]
        pr_audio = [None if pf is None else pf() for _, _, pf in cs]
        prints = ex.prints(pr_audio)
        # group by the probe flag so a batch keeps one block list
        for probe in (False, True):
            sel = [j for j, (cid, _, _) in enumerate(cs) if _probe_clip(cid) == probe]
            if not sel:
                continue
            blocks = tuple(sorted(set(TURN_BLOCKS) | ({4, 16, 20} if probe else set())))
            outs = ex([xs[j] for j in sel], [prints[j] for j in sel], blocks)
            for j, (inp, blk) in zip(sel, outs):
                cid = cs[j][0]
                np.savez(TURN_BLK / f"{cid}.tmp.npz", **blk)
                (TURN_BLK / f"{cid}.tmp.npz").rename(TURN_BLK / f"{cid}.npz")
                np.savez(TURN_INP / f"{cid}.tmp.npz", **inp)
                (TURN_INP / f"{cid}.tmp.npz").rename(TURN_INP / f"{cid}.npz")
        done += len(cs)
    el = time.time() - t0
    log(f"turn_cache {a.set}: {done} in {el:.0f} s ({el / max(done, 1):.2f} s each); {len(todo) - done} left")

# --------------------------------------------------------------------------- turn head v5 (segment classifier)
def _v5_mod(probe_only=False):
    """scripts/research/turn_v5.py reading the 0.6B turn inputs (TURN_INP / TURN_BLK) and writing under TURN; the
    smart-turn distillation targets (audio-only) and the LM completeness scores are turn_v5's own. The LM scores are
    per word of the 115M transcript, so they are mapped to frames through the 115M token counts of the same clip
    (time-aligned, whatever the encoder)."""
    import turn_v5 as V5
    if not getattr(V5, "_0p6b", False):
        W5, FEATS4 = V5.W, V5.FEATS4
        TURN.mkdir(parents=True, exist_ok=True)
        for name in ("cuts.json", "texts.jsonl"):
            if not (TURN / name).exists():
                (TURN / name).symlink_to(W5 / name)
        (TURN / "st3").mkdir(exist_ok=True)
        V5.W, V5.BLK = TURN, TURN_BLK
        V5.feat_path = lambda c: TURN_INP / f"{c['id']}.npz"
        orig_lm = V5.lm_frames

        def lm_frames(cid, n, w_of):
            f = FEATS4 / f"{cid}.npz"
            if not f.exists():
                f = W5 / "cuts" / f"{cid}.npz"
            if not f.exists():
                return np.full(len(n), -20.0, np.float32)
            n115 = np.load(f)["n"].astype(np.int64)
            n115 = np.concatenate([n115, np.repeat(n115[-1:], max(0, len(n) - len(n115)))])[: len(n)] \
                if len(n115) else np.zeros(len(n), np.int64)
            return orig_lm(cid, n115, w_of)
        V5.lm_frames = lm_frames
        V5._0p6b = True
        V5._all_clips = V5.all_clips
    if probe_only:
        V5.all_clips = lambda with_cuts=True: [c for c in V5._all_clips(with_cuts) if _probe_clip(c["id"])]
    else:
        V5.all_clips = V5._all_clips
    return V5


def stage_seg_train(a):
    """turn_v5.py train on the 0.6B inputs (the shipped c5 recipe: 2 x 256 transformer over <= 8 s of one encoder
    block + VAD / P(user) / P(other) + the RNNT text, 3000 steps, lr 1e-4, dropout 0.2, wd 0.05, smart-turn and LM
    completeness auxiliaries, no smart-turn v3.2 clips) with --block; --probe restricts to the probe subset (blocks
    4 / 8 / 12 / 16 / 20 / 24 cached) for the block probe. -> TURN/<tag>/model.pt."""
    V5 = _v5_mod(a.probe)
    ns = argparse.Namespace(tag=a.tag, block=str(a.block), pros=False, no_text=False, max_clips=None, max_val=30000,
                            steps=a.steps, batch=256, lr=1e-4, d=256, layers=2, w_pause=1.5, w_cut=0.7, w_st=0.5,
                            w_st_aux=0.5, w_compl=0.3, eval_every=250, pu_drop=0.25, calibrate_now=False,
                            dropout=0.2, no_st32=True, wd=0.05, budget=a.budget, seed=a.seed, device=a.device)
    V5.cmd_train(ns)
    h = TURN / a.tag / "history.json"
    if h.exists():
        d = json.loads(h.read_text())
        if d["hist"] and d["hist"][-1]["step"] >= a.steps:
            best = max(d["hist"], key=lambda r: r["auc"])
            save("seg", {"block": a.block, "probe": a.probe, "steps": a.steps, "best": best,
                         "last": d["hist"][-1], "params": d["params"]}, sub=a.tag)


# --------------------------------------------------------------------------- per-frame turn head (served objective)
TURN_CFG = {"type": "turn", "mode": "none", "use_text": True, "condition_on_speaker": False, "k_tokens": 4,
            "text_dim": 64, "align": "greedy", "hidden": 96, "history": 8, "pos_weight": 2.0, "weight": 1.0,
            "act_columns": 4, "duration_feats": True, "dur_max": 64}


def _turn_train_data(val_frac=0.08, max_clips=None):
    import random
    import turn_v4 as V4
    man = json.loads((V4.WORK / "manifest.json").read_text())
    groups = sorted({(c["src"], c.get("meeting", c["id"])) for c in man})
    val_g = set(random.Random(0).sample(groups, int(val_frac * len(groups))))
    tr, va = [], []
    for c in man[: max_clips] if max_clips else man:
        f, b = TURN_INP / f"{c['id']}.npz", TURN_BLK / f"{c['id']}.npz"
        if not f.exists() or not b.exists():
            continue
        z = np.load(f)
        d = {k: z[k] for k in z.files}
        T = len(d["pu"])
        e = np.load(b)["b24"]
        d["e"] = e[:T] if len(e) >= T else np.concatenate([e, np.repeat(e[-1:], T - len(e), 0)])
        d["tgt"], d["wt"], d["ends"], d["pauses"] = V4.targets(c, d["vad"].astype(np.float32), T)
        d["tgt0"], d["wt0"] = V4.targets.orig
        d["src"], d["id"] = c["src"], c["id"]
        (va if (c["src"], c.get("meeting", c["id"])) in val_g else tr).append(d)
    return tr, va


def stage_turn_train(a):
    """The per-frame turn head of the balanced preset on the 0.6B: the served TurnHead (text-aware GRU 96, 4 activity
    columns + duration counters, k 4 tokens), without the speaker-kernel pass (the 0.6B has no trained kernels; it
    reads pass-1 top-layer frames and the TS-VAD columns [P(user), P(other), 0, 0]), trained heads-only on the cached
    turn inputs with the served head's objective (eot = 1 from the audible end to the next onset, BCE pos_weight 2)
    + turn_v4's early-confidence term at --aux (0 = off); turn_v4 mix (oto 0.45 / meetings 0.35 / smart-turn 0.2),
    batch 16, AdamW --lr, cosine. -> HEADS_DIR/turn_<tag>.pt (best val end-vs-pause AUC)."""
    import random
    import torch
    import torch.nn.functional as F
    import turn_v4 as V4
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    dev = a.device
    od = TURN / f"frame_{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    m = load_core(dev, heads=("turn",), cfgs={"turn": TURN_CFG}, fresh=("turn",))
    head = m.heads["turn"]
    for p_ in m.parameters():
        p_.requires_grad_(False)
    params = list(head.parameters())
    for p_ in params:
        p_.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    step, hist, best = 0, [], 1e9
    last = od / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        step, hist, best = ck["step"], ck["hist"], ck["best"]
    if step >= a.steps:
        return
    tr, va = _turn_train_data()
    by = {}
    for d in tr:
        by.setdefault("meet" if d["src"] in ("ami", "icsi") else d["src"], []).append(d)
    mix = {"oto": 0.45, "meet": 0.35, "st": 0.2}
    log(f"train {len(tr)} ({ {k: len(v) for k, v in by.items()} }), val {len(va)}; step {step}")
    rng = random.Random(a.seed + step)
    head.train()
    t0 = time.time()
    while step < a.steps and time.time() - t0 < a.budget:
        lr = a.lr * min(1.0, (step + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(step / a.steps, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        src = rng.choices(list(mix), weights=list(mix.values()))[0]
        b = V4.collate(rng.sample(by[src], min(16, len(by[src]))), dev)
        z = V4.head_logits(head, b)
        l0 = F.binary_cross_entropy_with_logits(z, b[7], reduction="none", pos_weight=torch.tensor(2.0, device=dev))
        loss = (l0 * b[8]).sum() / b[8].sum().clamp(min=1)
        if a.aux > 0:
            l1 = F.binary_cross_entropy_with_logits(z, b[5], reduction="none")
            loss = loss + a.aux * (l1 * b[6]).sum() / b[6].sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 250 == 0 or step == a.steps:
            r = {"step": step, "loss": round(float(loss), 4), **V4.val_metrics(head, va, dev)}
            hist.append(r)
            log(r)
            if -r["auc"] < best:
                best = -r["auc"]
                torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "cfg": TURN_CFG,
                            "step": step, "val": r}, HEADS_DIR / f"turn_{a.tag}.pt")
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step, "hist": hist, "best": best},
                       last)
    log(f"stopped at {step} ({time.time() - t0:.0f} s)")
    if step >= a.steps:
        save("turn_frame", {"hist": hist, "best": max(hist, key=lambda r: r["auc"])}, sub=a.tag)

# --------------------------------------------------------------------------- build the served 0.6B model
SERVED_0P6B = W / "served_0p6b_v0.1.afm"


def stage_build(a):
    """served_0p6b_v0.1.afm = the imported 0.6B (every tensor unchanged) + heads vad (mix of blocks 4..24), spk
    (block 11), turn (per-frame, TS-VAD columns, no kernels), turn_seg (v5 classifier --seg-tag); then
    assets/served_heads_0p6b_v0.1.pt = its non-NVIDIA tensors (hub.export_heads against the .nemo) and a rebuild
    check (hub.build_served == the recorded hash); TS-VAD / LID heads copied to assets/tsvad_0p6b.pt / lid_0p6b.pt."""
    import shutil
    import torch
    from audioforge import hub
    from audioforge.train import load_model, save_model
    torch.set_num_threads(2)
    ck = torch.load(TURN / a.seg_tag / "model.pt", map_location="cpu", weights_only=False)
    seg_cfg = {"type": "turn_seg", "weight": 0.0, **ck["cfg"]}
    tsd = HEADS_DIR / f"turn_{a.turn_tag}.pt"
    heads = ("vad", "spk", "turn", "turn_seg")
    shutil.copy(tsd, HEADS_DIR / "turn.pt")
    torch.save({"state_dict": ck["state_dict"]}, HEADS_DIR / "turn_seg.pt")
    m = load_core("cpu", heads=heads, cfgs={"turn": TURN_CFG, "turn_seg": seg_cfg})
    m.cfg["name"] = "served_0p6b_v0.1"
    # the assistant preset re-tuned for the 0.6B's heads (stage preset_scan: smart-turn test 91.2 % at 374 ms with the
    # 0.6B's v5 classifier, vs 78.2 % with the 115M constants); balanced / fast keep the shared constants (no setting
    # of the scan met the 115M's bars on calls and AMI)
    m.cfg["turn_presets"] = {"assistant": {"vad_wait_ms": [320, 2960], "turn_model": {"p": 0.99}}}
    save_model(m.eval(), SERVED_0P6B)
    back = load_model(str(SERVED_0P6B), "cpu")
    assert hub.state_hash(back.state_dict()) == hub.state_hash(m.state_dict())
    out = ROOT / "assets" / "served_heads_0p6b_v0.1.pt"
    info = hub.export_heads(SERVED_0P6B, NEMO, out, base_key="asr_0p6b")
    got = hub.build_served(NEMO, out, W / "rebuilt_0p6b.afm")
    assert got == info["state_hash"], (got, info)
    (W / "rebuilt_0p6b.afm").unlink()
    shutil.copy(HEADS_DIR / f"tsvad_{a.heads}.pt", ROOT / "assets" / "tsvad_0p6b.pt")
    rec = {**info, "size": out.stat().st_size, "sha256": hub.sha256_file(out), "afm": str(SERVED_0P6B),
           "seg_tag": a.seg_tag, "turn_tag": a.turn_tag, "tsvad_tag": a.heads,
           "tsvad": {"size": (ROOT / "assets/tsvad_0p6b.pt").stat().st_size,
                     "sha256": hub.sha256_file(ROOT / "assets/tsvad_0p6b.pt")}}
    lid = HEADS_DIR / "lid.pt"
    if lid.exists():
        shutil.copy(lid, ROOT / "assets" / "lid_0p6b.pt")
        rec["lid"] = {"size": lid.stat().st_size, "sha256": hub.sha256_file(ROOT / "assets/lid_0p6b.pt")}
    log(rec)
    save("build", rec)

# --------------------------------------------------------------------------- end-of-turn benchmarks (same harnesses)
EOT_DUMP, ASST_DUMP = W / "eot_dump", W / "asst_dump"
PRINTS = Path(os.environ.get("CORE_PRINTS", str(W / "prints_0p6b.json")))


def stage_prints(a):
    """The stored 5 s voice prints of the end-of-turn sessions re-made for the 0.6B (a print belongs to one core's
    speaker head): the same audio intervals as the 115M prints (scratch/e2e_tsvad/prints.json '5.0' for the 32
    two-party user channels; eot_latency AMI clips' print_ivs), embedded by tsvad_stream.voiceprint on the 0.6B."""
    import torch
    import eot_latency as E
    import tsvad as T0
    from audioforge.datasets import dyadic as D
    from audioforge.tsvad_stream import voiceprint
    torch.set_num_threads(2)
    m = load_core(a.device, heads=("spk",))
    out = json.loads(PRINTS.read_text()) if PRINTS.exists() else {}
    refs = {r["name"]: r for r in json.loads((E.E2E / "clips.json").read_text())}
    pr = json.loads((E.E2E / "prints.json").read_text())
    dy = {}
    for name, r in refs.items():
        if r["set"] == "ami" or f"{name}.user" in out or not (pr.get(name) or {}).get("5.0"):
            continue
        ivs = pr[name]["5.0"]["ivs"]
        cid, h = r["conversation"], r["human_channel"]
        if r["set"] not in dy:
            cids = [x["conversation"] for x in refs.values() if x["set"] == r["set"]]
            dy[r["set"]] = D.Dyadic(cids, r["set"], verbose=False,
                                    **({"cache_dtype": "float16"} if r["set"] == "oto" else {}))
        ds = dy[r["set"]]
        if r["set"] == "turnbench":
            chan = np.asarray(ds.channels16k(cid)[h], np.float32)
        else:
            xx, sr, _ = D.read_stereo(r["set"], ds.root, cid)
            ch = D.resample16k(xx, sr)
            chan = np.asarray(ch.T if ch.shape[0] != 2 else ch, np.float32)[h]
        x = np.concatenate([chan[int(p_ * SR): int(q * SR)] for p_, q in ivs])
        out[f"{name}.user"] = [round(float(z), 6) for z in voiceprint(m, x)]
    ami = json.loads((E.AMI_DIR / "clips.json").read_text())
    import eval_stage1 as ES
    ds = None
    for r in ami:
        k = f"{r['name']}.mono"
        if k in out:
            continue
        if ds is None:
            _, _, _, ds, _ = ES.v2_data() if E.AMI_SPLIT == "dev" else ES.v2_data_split(E.AMI_SPLIT)
        out[k] = [round(float(z), 6) for z in voiceprint(m, T0.clip_audio(ds, r["meeting"], r["print_ivs"]))]
    PRINTS.write_text(json.dumps(out))
    log(f"prints: {len(out)}")


def stage_build115(a):
    """A 115M candidate: stage1_served_v3.afm with heads replaced (every NVIDIA tensor and every other head unchanged):
    --vad-tag T -> heads.vad = the vad2_T.pt head (frame_gru for gru64 / frame for mlp, its blocks; a learned mix
    when > 1 block); --spk-tag T -> heads.spk = spk_frame head_T.pt (same 0.5 M SpeakerHead shape, block 4);
    --seg-tag T -> heads.turn_seg = W/turn/T/model.pt (v5 classifier). -> W/cand_<--tag>.afm (tensor check)."""
    import copy
    import torch
    from audioforge import hub
    from audioforge.model import SpeechModel
    from audioforge.train import load_model, save_model
    torch.set_num_threads(2)
    base = load_model(str(ROOT / "runs" / "stage1_served_v3.afm"), "cpu")
    cfg = copy.deepcopy(base.cfg)
    sd = {k: v for k, v in base.state_dict().items()}
    changed = []
    if a.vad_tag:
        ck = torch.load(HEADS_DIR / f"vad2_{a.vad_tag}.pt", map_location="cpu", weights_only=False)
        bl = [int(b) for b in ck["blocks"]]
        typ = "frame_gru" if ck["arch"].startswith("gru") else "frame"
        assert ck["arch"] in ("gru64", "mlp"), ck["arch"]
        cfg["heads"]["vad"] = {"type": typ, "key": "vad", "hidden": 64, "from_layers": [b - 1 for b in bl], "weight": 0.5}
        sd = {k: v for k, v in sd.items() if not k.startswith("heads.vad.") and k != "layer_mix.vad"}
        st = ck["state_dict"]
        if typ == "frame":  # _vad_net mlp (inp, out) -> FrameHead (net.0, net.2)
            st = {"net.0.weight": st["inp.weight"], "net.0.bias": st["inp.bias"], "net.2.weight": st["out.weight"],
                  "net.2.bias": st["out.bias"]}
        sd.update({f"heads.vad.{k}": v for k, v in st.items()})
        if len(bl) > 1:
            sd["layer_mix.vad"] = ck["mix"]
        changed.append(f"vad<-{a.vad_tag}")
    if a.spk_tag:
        ck = torch.load(Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/spk_frame") / f"head_{a.spk_tag}.pt",
                        map_location="cpu", weights_only=False)
        st = ck.get("state_dict", ck)
        sd.update({f"heads.spk.{k}": v for k, v in st.items()})
        changed.append(f"spk<-{a.spk_tag}")
    if a.seg_tag and a.seg_tag != "none" and (W / "turn" / a.seg_tag / "model.pt").exists():
        ck = torch.load(W / "turn" / a.seg_tag / "model.pt", map_location="cpu", weights_only=False)
        cfg["heads"]["turn_seg"] = {"type": "turn_seg", "weight": 0.0, **ck["cfg"]}
        sd = {k: v for k, v in sd.items() if not k.startswith("heads.turn_seg.")}
        sd.update({f"heads.turn_seg.{k}": v for k, v in ck["state_dict"].items()})
        changed.append(f"turn_seg<-{a.seg_tag}")
    m = SpeechModel(cfg, base.tokenizer)
    m.load_state_dict(sd, strict=True)
    if getattr(a, "presets_json", None):
        m.cfg["turn_presets"] = json.loads(a.presets_json)
    m.cfg["name"] = f"stage1_served_cand_{a.tag}"
    out = W / f"cand_{a.tag}.afm"
    save_model(m.eval(), out)
    back = load_model(str(out), "cpu")
    assert hub.state_hash(back.state_dict()) == hub.state_hash(m.state_dict())
    same = [k for k, v in base.state_dict().items() if not k.startswith(("heads.vad.", "heads.spk.", "heads.turn_seg.",
                                                                          "layer_mix.vad"))]
    bsd = back.state_dict()
    assert all(torch.equal(base.state_dict()[k], bsd[k]) for k in same), "a frozen tensor changed"
    log(f"build115 {out}: {changed}; {len(same)} other tensors identical")
    save("build115", {"afm": str(out), "changed": changed}, sub=a.tag)


def engine_115m(device, afm=None, tsvad=None, **kw):
    """The 115M --mode single engine (single_model.single_engine's options): ``afm`` (default CORE115_AFM),
    ``tsvad`` (default the shipped assets/tsvad_spk.pt or $CORE115_TSVAD)."""
    from audioforge.serve import Engine
    from audioforge.server.cli import MODES
    opts = {**MODES["single"], "enroll": "explicit",
            "tsvad": str(tsvad or os.environ.get("CORE115_TSVAD", ROOT / "assets" / "tsvad_spk.pt")),
            "lid": str(ROOT / "assets" / "lid_distill.pt"),
            "silero": str(ROOT / "data" / "silero" / "silero_vad_v5.onnx"), "preload_silero": True, **kw}
    return Engine.load(str(afm or AFM115), None, device, threads=2, **opts)


def engine_0p6b(device, **kw):
    """The --mode single engine on the 0.6B (served_0p6b_v0.1.afm + tsvad_0p6b.pt + its LID head), as
    single_model.single_engine builds the 115M one (explicit enrolment, Silero preloaded for the dump's track)."""
    from audioforge.serve import Engine
    from audioforge.server.cli import MODES
    lid = ROOT / "assets" / "lid_0p6b.pt"
    opts = {**MODES["single"], "enroll": "explicit", "tsvad": str(ROOT / "assets" / "tsvad_0p6b.pt"),
            "lid": str(lid) if lid.exists() else None,
            "silero": str(ROOT / "data" / "silero" / "silero_vad_v5.onnx"), "preload_silero": True, **kw}
    return Engine.load(str(SERVED_0P6B), None, device, threads=2, **opts)


def _record(s, rec, enrolled_only=False):
    """Wrap a session's turn pass: per frame (v, ready t, turn p, vad, P(user), P(other), v5 p)."""
    orig = s.asr.run_turn_on_diar

    def rtod(avail, act_fn, flush=False, _o=orig, _s=s):
        out = _o(avail, act_fn, flush)
        tp = _s.asr.tsvad_p if (_s.tsvad is not None and (_s.tsvad.enrolled or not enrolled_only)) else []
        for v, p_, vad in out:
            pu, po = (float(tp[v][0]), float(tp[v][1])) if v < len(tp) else (0.0, 0.0)
            t0 = time.perf_counter()
            p5 = float(_s.asr.seg_prob(v)) if _s.asr.seg is not None else None
            _s._rec_extra = getattr(_s, "_rec_extra", 0.0) + time.perf_counter() - t0
            rec.append((int(v), round(_s._asr_ready_t(v), 4), float(p_), round(float(vad), 5), pu, po, p5))
        return out
    s.asr.run_turn_on_diar = rtod
    orig_proc = s.process

    def process(x, *args, _o=orig_proc, _s=s, **kw):
        """the recording's own v5 calls are not part of the served compute: taken out of this call's chunk_ms"""
        _s._rec_extra, n0 = 0.0, len(_s.chunk_ms)
        out = _o(x, *args, **kw)
        new = len(_s.chunk_ms) - n0
        if new > 0 and _s._rec_extra:
            for i in range(new):
                _s.chunk_ms[n0 + i] = max(0.0, _s.chunk_ms[n0 + i] - 1000 * _s._rec_extra / new)
        return out
    s.process = process


def stage_eot_dump(a):
    """The served 0.6B single-mode session once per clip, every per-frame signal a turn rule reads (eot_latency.py
    dump / eot_assistant.py dump protocols: calls = 32 two-party user channels + 200 AMI eot-bench windows with the
    stored 5 s print, hybrid_dyn session config, 160 ms blocks; asst = the 399 smart-turn v3.2 test clips, vad_head, no
    print, 20 ms blocks), plus the v5 classifier's p on every frame. --which calls|asst. Resumable."""
    import torch
    import audioforge.serve as S
    import eot_latency as E
    import eot_assistant as EA
    torch.set_num_threads(2)
    prints = json.loads(PRINTS.read_text()) if a.which == "calls" else {}
    od = EOT_DUMP if a.which == "calls" else ASST_DUMP
    if C115:  # one dump directory per candidate (--tag)
        od = W / f"{'eot' if a.which == 'calls' else 'asst'}_dump_{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    items = E.sessions() if a.which == "calls" else EA.clips()
    key = (lambda x: x["key"])
    todo = [x for x in items if not (od / f"{key(x)}.json").exists()]
    log(f"eot_dump {a.which}: {len(todo)} of {len(items)} to do")
    if not todo:
        return
    eng = engine_115m(a.device) if C115 else engine_0p6b(a.device)
    eng.warmup()
    t0, n = time.time(), 0
    for it in todo:
        if time.time() - t0 > a.budget:
            break
        k = key(it)
        if a.which == "calls":
            x = E.read_audio(it)
            s = S.Session(eng, S.SessionConfig(turn_policy="hybrid_dyn", timeout_ms=1000))
            if it["embedding"] is not None:
                s.arm_enrollment("enroll", 0, embedding=prints[k])
            blk = SR * 160 // 1000
        else:
            x = EA.audio(it)
            s = S.Session(eng, S.SessionConfig(turn_policy="vad_head"))
            blk = 320
        if s.asr.seg is None and eng.seg_name is not None:  # record the v5 classifier's p on every frame too
            sh = eng.asr.heads[eng.seg_name]
            s.asr.attach_seg(sh, next(sh.parameters()).device)
        rec = []
        _record(s, rec, enrolled_only=a.which == "asst")
        msgs = []
        for i in range(0, len(x), blk):
            msgs += s.process(x[i:i + blk])
        msgs += s.finish()
        cm = np.asarray(list(s.chunk_ms), float)
        d = {"key": k, "audio_s": round(len(x) / SR, 3),
             "head": {kk: [r[j] for r in rec] for j, kk in enumerate(("v", "t", "p", "vad", "pu", "po", "p5"))},
             "tok_at": [int(q) for q in s.asr.tok_at], "text": s.asr.text if hasattr(s.asr, "text") else None,
             "chunk_ms": {"p50": round(float(np.median(cm)), 2), "p95": round(float(np.percentile(cm, 95)), 2),
                          "mean": round(float(cm.mean()), 2), "n": int(len(cm))},
             "turn_ends": [{kk: m_.get(kk) for kk in ("t", "policy", "p", "silence_ms", "path")} for m_ in msgs
                           if m_["type"] == "turn_end"], "device": a.device}
        if a.which == "calls":
            d.update({"set": it["set"], "cond": it["cond"], "has_print": it["embedding"] is not None})
        else:
            d.update({"complete": it["complete"], "clip_s": round((it["b"] - it["a"]) / SR, 4)})
        (od / f"{k}.json").write_text(json.dumps(d))
        n += 1
    log(f"eot_dump {a.which}: {n} this call ({time.time() - t0:.0f} s), {len(todo) - n} left")


def _preset_rules():
    import turn_v5 as V5
    return {"balanced": V5.BAL, "fast": V5.F5, "assistant": dict(V5.PRESET_RULES["assistant"])}


def _score_sets(dump, ad, p_key, comp_scale=1.0, rules=None):
    """Every preset on (calls/AMI dump, assistant dump) with the turn_v5 run_policy twin of the served VadHeadPolicy
    and the eot_latency / eot_assistant scorers; 'p' of the head presets = the per-frame turn head, of the model
    presets (fast / assistant) = the v5 classifier (``p_key``)."""
    import eot_latency as E
    import eot_assistant as EA
    import turn_v5 as V5
    sess = [s_ for s_ in E.sessions() if s_["key"] in dump]
    asess = EA.sessions(ad)
    comp = {k: comp_scale * d["chunk_ms"]["p50"] / 1000 for k, d in dump.items()}
    acomp = {k: comp_scale * d["chunk_ms"]["p50"] / 1000 for k, d in ad.items()}
    dbs = {k: np.load(V5.ENERGY / f"{k}.npy") for k in dump}
    adbs = {k: np.load(V5.ENERGY / f"asst_{k}.npy") for k in ad}
    for dd, dbd in ((dump, dbs), (ad, adbs)):
        for k, d in dd.items():
            v = np.asarray(d["head"]["v"])
            dbd[k] = dbd[k][np.clip(v, 0, len(dbd[k]) - 1)]
    res = {}
    for name, r in (rules or _preset_rules()).items():
        pk = "p" if r["mode"] == "head" else p_key

        def view(d):
            h = dict(d["head"])
            h["p"] = h[pk]
            return {"head": h}
        per = {sc: [] for sc in E.SCOPES}
        for ss in sess:
            t = [x + comp[ss["key"]] for x, _ in V5.run_policy(view(dump[ss["key"]]), dbs[ss["key"]], r, {})]
            for sc, sets in E.SCOPES.items():
                if ss["set"] in sets:
                    per[sc].append(E.score_session(t, ss, comp[ss["key"]]))
        at = {s_["key"]: [x + acomp[s_["key"]] for x, _ in V5.run_policy(view(ad[s_["key"]]), adbs[s_["key"]], r, {})]
              for s_ in asess}
        asc = EA.score_system(asess, at, acomp)
        res[name] = {"calls": E.pool(per["two_party_user"]), "ami": E.pool(per["ami"]),
                     "asst": {"accuracy_pct": asc["accuracy_pct"], "p50": asc["complete"]["eot_total_ms_p50"],
                              "p95": asc["complete"]["eot_total_ms_p95"],
                              "false_fire_pct": asc["incomplete"]["false_fire_pct"],
                              "missed_pct": asc["complete"]["missed_pct"]},
                     "compute_ms_p50": round(1000 * float(np.median(list(comp.values()))), 1)}
    return res


def stage_eot_score(a):
    """Balanced / fast / assistant on both cores with the same scorer: 115M = the shipped dumps (turn_v4
    dump_served = the served head on stage1_served_v2 + the v5 c5 classifier's p, turn_v5.load_sets('c5')); 0.6B =
    EOT_DUMP / ASST_DUMP (the served 0.6B engine's own frames, turn p and v5 p). -> runs/core_0p6b.json eot."""
    import turn_v5 as V5
    import eot_assistant as EA
    dump, ad, _ = V5.load_sets("c5")
    import turn_v4 as V4
    served = {d_["key"]: d_ for d_ in (json.loads(p_.read_text()) for p_ in sorted((V5.W4 / "dump_served").glob("*.json")))}
    for k, d in dump.items():  # 'p' = the served per-frame head, 'p5' = the v5 classifier
        d["head"]["p5"] = d["head"]["p"]
        d["head"]["p"] = served[k]["head"]["p"]
    sd = EA.load_dump()
    for k, d in ad.items():
        d["head"]["p5"] = d["head"]["p"]
        d["head"]["p"] = sd[k]["head"]["p"]
    res = {"115m": _score_sets(dump, ad, "p5")}
    d6 = {d_["key"]: d_ for d_ in (json.loads(p_.read_text()) for p_ in sorted(EOT_DUMP.glob("*.json")))}
    a6 = {d_["key"]: d_ for d_ in (json.loads(p_.read_text()) for p_ in sorted(ASST_DUMP.glob("*.json")))}
    for k, d in a6.items():  # the reference speech end comes from Silero on the audio (core-independent)
        d["conf"] = sd[k]["conf"]
    res["0p6b_shared_presets"] = _score_sets(d6, a6, "p5")
    rules6 = _preset_rules()
    rules6["assistant"] = dict(rules6["assistant"], k=4, mp=0.99)  # the served 0.6B's assistant (cfg turn_presets)
    res["0p6b"] = _score_sets(d6, a6, "p5", rules=rules6)
    res["n"] = {"115m": [len(dump), len(ad)], "0p6b": [len(d6), len(a6)]}
    for c in ("115m", "0p6b_shared_presets", "0p6b"):
        for pn, r in res[c].items():
            log(c, pn, "calls", r["calls"].get("eot_total_ms_p50"), r["calls"].get("false_interruption_pct"),
                r["calls"].get("missed_pct"), "| AMI", r["ami"].get("eot_total_ms_p50"),
                r["ami"].get("false_interruption_pct"), r["ami"].get("missed_pct"), "| asst", r["asst"])
    save("eot", res)

# --------------------------------------------------------------------------- target-speaker WER (tswer.py protocols)
def _tsvad_head(tag):
    import torch
    from audioforge.heads.tsvad import TSVADHead
    ck = torch.load(HEADS_DIR / f"tsvad_{tag}.pt", map_location="cpu", weights_only=False)
    c = {k: v for k, v in ck["cfg"].items() if k not in ("type", "from_layers", "weight", "enroll_embedder")}
    h = TSVADHead(D, **c)
    h.load_state_dict(ck["state_dict"])
    return h.eval()


def stage_tswer(a):
    """tswer.py meetings on the 0.6B system: the 0.6B streaming RNNT words (tswer's asr_0p6b.jsonl, [70,1]) kept where
    the 0.6B's own TS-VAD track (block 11, 0.6B voice print of the same 5 s clip, served track with adaptation) is on,
    tswer's keep rule (mask dilated +-2 frames at the word's emission frame minus the 0.6B's measured lag); units =
    tswer's (window, target) units. Side by side with the 115M 'tsvad_d2' arm of the same units; bootstrap by meeting.
    -> runs/core_0p6b.json tswer."""
    import tswer as TW
    T = _tsvad_mod("cpu")
    head = _tsvad_head(a.heads)
    spk = T.load_served().heads["spk"].cpu()
    res = load_json("tswer", {})
    for corpus in (a.corpora or "ami,icsi").split(","):
        ext, meta, ds = T.bench_windows(corpus)
        asr6 = TW.load_asr(corpus, "0p6b")
        L6 = json.loads((TW.WORK / corpus / "lag.json").read_text())["lag_0p6b_frames"]
        kidx, VP = T.load_vprints(corpus)
        rows = [json.loads(x) for x in (TW.WORK / corpus / "units.jsonl").read_text().splitlines() if x.strip()]
        byk = {T.wkey(v): v for v in ext}
        units = []
        for r in rows:
            v = byk[r["key"]]
            f = T.win_feats(corpus, v)
            spks = T.window_speakers(ds, v)
            for u in r["units"]:
                c = u["col"]
                k = kidx[r["key"]]
                if not VP["has_5p0"][k, c]:
                    continue
                ref = [x[0] for x in TW.ref_words(ds, v, spks[c])]
                mk = TW.served_tsvad(head, spk, f, VP["spk_5p0"][k, c])[:, 0] > 0.5
                hyp6 = asr6[r["key"]]["words"]
                kp = TW.keep_mask(hyp6, mk, L6, TW.DIL)
                h = [x[0] for x, q in zip(hyp6, kp) if q]
                s_, d_, i_, _, _ = TW.align_counts(ref, h)
                u2 = {**u, "arms": {"core115_tsvad_d2": u["arms"]["tsvad_d2"], "core0p6b_tsvad_d2":
                                    [int(s_), int(d_), int(i_), int(len(h))],
                                    "asr0p6b_mask115_d2": u["arms"].get("0p6b_tsvad_d2"),
                                    "core115_none": u["arms"]["none"], "core0p6b_none": u["arms"].get("0p6b_none")}}
                units.append(u2)
        (W / "tswer").mkdir(parents=True, exist_ok=True)
        (W / "tswer" / f"{corpus}_units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
        prim = [u for u in units if u["col"] == 0]
        out = {}
        for gname, us in (("primary", prim), ("all_speakers", units)):
            arms = ["core115_tsvad_d2", "core0p6b_tsvad_d2", "asr0p6b_mask115_d2"]
            draws, ncl = TW.boot(us, arms, "meeting")
            out[gname] = {arm: {"twer": round(100 * TW.rate(TW.agg(us, arm)), 2), "ci_meeting": TW.ci(draws[arm])}
                          for arm in arms}
            out[gname]["0p6b_minus_115m"] = {
                "delta": round(out[gname]["core0p6b_tsvad_d2"]["twer"] - out[gname]["core115_tsvad_d2"]["twer"], 2),
                "ci_meeting": TW.ci(draws["core0p6b_tsvad_d2"] - draws["core115_tsvad_d2"])}
            out[gname]["n_units"], out[gname]["n_meetings"] = len(us), ncl
        res[corpus] = out
        log(corpus, json.dumps(out["primary"]))
    save("tswer", res)


def stage_live_wer(a):
    """--core 3.5: the bare-ASR row of tswer_live (arm none_full: every word of the session against its full
    reference) on the 32 live TurnBench sessions (16 clips x {mono mix, user channel}), the session wav through the
    masked [L,1] forward + greedy timed RNNT words, prompt --prompt; beside the English 0.6B's and the 115M's stored
    none_full rows (core_0p6b live_sessions.json, tswer_live sessions.json). -> OUT live_wer.<prompt>."""
    import copy
    import torch
    import tswer as TW
    from audioforge.teachers import normalize_text
    torch.set_num_threads(2)
    m = load_core(a.device, heads=())
    m.heads["rnnt"].set_prompt(a.prompt)
    rnnt = copy.deepcopy(m.heads["rnnt"]).cpu()
    clips = json.loads((TW.E2E / "clips.json").read_text())
    sessions = []
    for c in clips:
        if c["set"] != "turnbench" or not c.get("text_user"):
            continue
        for cond in ("mono", "user"):
            x = TW._wav(TW.E2E / "clips" / f"{c['name']}.{cond}.wav")
            _, L, top = encode_blocks(m, [x], (1,), top=True)
            hw = TW.words_of(m.tokenizer, TW.greedy_timed(rnnt, top[0, :L[0]].float().cpu()))
            full = normalize_text(c["text_mono"] if cond == "mono" else c["text_user"]).split()
            sessions.append({"key": f"{c['name']}|{cond}", "clip": c["name"], "cond": cond, "N_full": len(full),
                             "N": len(full), "arms": {"none_full": TW._counts(full, [w[0] for w in hw])}})
    p6 = json.loads(Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/core_0p6b/tswer/live_sessions.json")
                    .read_text())["sessions"]
    old = json.loads((TW.LIVE_WORK / "sessions.json").read_text())["sessions"]
    res = {}
    for g, sel in (("all", lambda u: True), ("user_channel", lambda u: u["cond"] == "user")):
        res[g] = {name: TW._live_group([u for u in S_ if sel(u)], ["none_full"])
                  for name, S_ in ((TAG, sessions), ("0p6b", p6), ("115m", old))}
    save("live_wer", res, sub=a.prompt)
    log(a.prompt, {g: {k: v["none_full"]["wer"] for k, v in r.items()} for g, r in res.items()})


def _spk_head_115(spk_tag):
    """The 115M speaker head: the shipped one (None) or a spk_frame head_<tag>.pt (same SpeakerHead shape)."""
    import copy
    import torch
    m = load_core("cpu", heads=("spk",))
    spk = copy.deepcopy(m.heads["spk"]).cpu().eval()
    if spk_tag:
        ck = torch.load(Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/spk_frame") / f"head_{spk_tag}.pt",
                        map_location="cpu", weights_only=False)
        spk.load_state_dict(ck.get("state_dict", ck))
    return m, spk


def _vprints_115(corpus, spk_tag, m, spk):
    """tsvad.py vprints of a corpus re-embedded by another speaker head (the same audio intervals, 5 s prints)."""
    import tsvad as T0
    from audioforge.tsvad_stream import embed_frames
    T = _tsvad_mod("cpu")
    kidx, VP = T.load_vprints(corpus)
    if not spk_tag:
        return kidx, VP["spk_5p0"], VP["has_5p0"]
    f = W / "vprints" / f"{corpus}_{spk_tag}.npy"
    if f.exists():
        return kidx, np.load(f), VP["has_5p0"]
    _, _, ds = T.bench_windows(corpus)
    keys = {i: k for k, i in kidx.items()}
    meet = {T.wkey(v): v["meeting"] for v in T.bench_windows(corpus)[0]}
    E = np.zeros_like(VP["spk_5p0"])
    cache = {}
    for w in range(len(keys)):
        for c in range(E.shape[1]):
            if not VP["has_5p0"][w, c]:
                continue
            iv = tuple(tuple(x) for x in VP["ivs_5p0"][w, c] if x[0] >= 0)
            ck = (meet[keys[w]], iv)
            if ck not in cache:
                fs, _, _ = encode_blocks(m, [T0.clip_audio(ds, ck[0], list(iv))], (SPK_TAP + 1,))
                cache[ck] = embed_frames(spk, fs[0][SPK_TAP + 1].astype(np.float32))
            E[w, c] = cache[ck]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, E)
    return kidx, E, VP["has_5p0"]


def stage_tswer115(a):
    """Target-speaker WER on the 115M (tswer.py protocols, served 115M streaming RNNT words, keep rule tsvad_d2 =
    served TS-VAD track P(target) > 0.5 dilated +-2 frames at the word's emission frame minus the measured lag): the
    shipped head (recomputed; must equal runs/tswer.json's tsvad_d2 arm) and every head --heads tag,... (HEADS_DIR
    tsvad_<tag>.pt; prints of --spk-tag's speaker head when given, from the same 5 s intervals). AMI / ICSI eot-bench
    units (bootstrap by meeting) and the 32 live TurnBench sessions (mono + user channel; bootstrap by clip).
    -> runs/core_115m.json tswer115.<label>."""
    import copy
    import torch
    import tswer as TW
    from audioforge.teachers import normalize_text
    T = _tsvad_mod("cpu")
    torch.set_num_threads(2)
    m, spk = _spk_head_115(a.spk_tag)
    spk0 = copy.deepcopy(load_core("cpu", heads=("spk",)).heads["spk"]).cpu().eval()
    tags = [t for t in a.heads.split(",") if t]
    heads = {t: _tsvad_head(t) for t in tags}
    heads["shipped"] = _tsvad_head("shipped")
    label = a.tag
    res = {}
    for corpus in (a.corpora or "ami,icsi").split(","):
        ext, meta, ds = T.bench_windows(corpus)
        asr = TW.load_asr(corpus, "served")
        L = json.loads((TW.WORK / corpus / "lag.json").read_text())["lag_frames"]
        kidx, VP0, has = _vprints_115(corpus, None, m, spk0)
        _, VPn, _ = _vprints_115(corpus, a.spk_tag, m, spk)
        rows = [json.loads(x) for x in (TW.WORK / corpus / "units.jsonl").read_text().splitlines() if x.strip()]
        byk = {T.wkey(v): v for v in ext}
        units = []
        for r in rows:
            v = byk[r["key"]]
            f = T.win_feats(corpus, v)
            spks = T.window_speakers(ds, v)
            k = kidx[r["key"]]
            for u in r["units"]:
                c = u["col"]
                if not has[k, c]:
                    continue
                ref = [x[0] for x in TW.ref_words(ds, v, spks[c])]
                hyp = asr[r["key"]]["words"]
                arms = {"stored_tsvad_d2": u["arms"]["tsvad_d2"], "none": u["arms"]["none"],
                        "oracle_d2": u["arms"]["oracle_d2"]}
                for t, h_ in heads.items():
                    e, sp = (VP0[k, c], spk0) if t == "shipped" else (VPn[k, c], spk)
                    mk = TW.served_tsvad(h_, sp, f, e)[:, 0] > 0.5
                    kp = TW.keep_mask(hyp, mk, L, TW.DIL)
                    hh = [x[0] for x, q in zip(hyp, kp) if q]
                    s_, d_, i_, _, _ = TW.align_counts(ref, hh)
                    arms[f"{t}_d2"] = [int(s_), int(d_), int(i_), int(len(hh))]
                units.append({**u, "arms": arms})
        (W / "tswer").mkdir(parents=True, exist_ok=True)
        (W / "tswer" / f"{corpus}_{label}_units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
        out = {}
        for gname, us in (("primary", [u for u in units if u["col"] == 0]), ("all_speakers", units)):
            arms = ["none", "oracle_d2", "stored_tsvad_d2"] + [f"{t}_d2" for t in heads]
            draws, ncl = TW.boot(us, arms, "meeting")
            out[gname] = {arm: {"twer": round(100 * TW.rate(TW.agg(us, arm)), 2), "ci_meeting": TW.ci(draws[arm])}
                          for arm in arms}
            out[gname]["paired_vs_shipped"] = {t: {"delta": round(out[gname][f"{t}_d2"]["twer"]
                                                                  - out[gname]["shipped_d2"]["twer"], 2),
                                                   "ci_meeting": TW.ci(draws[f"{t}_d2"] - draws["shipped_d2"])}
                                               for t in tags}
            out[gname]["n_units"], out[gname]["n_meetings"] = len(us), ncl
        res[corpus] = out
        log(corpus, json.dumps(out["primary"]))
    # live: the 16 TurnBench clips x {mono, user channel}
    lagf = json.loads((TW.WORK / "ami" / "lag.json").read_text())["lag_frames"]
    clips = json.loads((TW.E2E / "clips.json").read_text())
    prints = json.loads((TW.E2E / "prints.json").read_text())
    sessions = []
    for c in clips:
        if c["set"] != "turnbench" or not c.get("text_user"):
            continue
        _, segs = TW.tb_user_words(c["conversation"], c["human_channel"], c["start"], c["start"] + c["dur"])
        ref = normalize_text(c["text_user"]).split()
        e0 = np.asarray(prints[c["name"]]["5.0"]["embedding"], np.float32)
        if a.spk_tag:  # the same 5 s interval embedded by the new head (stage prints with CORE115_AFM = its build)
            en = np.asarray(json.loads(PRINTS.read_text())[f"{c['name']}.user"], np.float32)
        else:
            en = e0
        for cond in ("mono", "user"):
            x = TW._wav(TW.E2E / "clips" / f"{c['name']}.{cond}.wav")
            outs, Ls, top = encode_blocks(m, [x], (SPK_TAP + 1,), top=True)
            n = Ls[0]
            hw = TW.words_of(m.tokenizer, TW.greedy_timed(copy.deepcopy(m.heads["rnnt"]).cpu(), top[0, :n].float().cpu()))
            f = outs[0][SPK_TAP + 1].astype(np.float32)
            orc = np.zeros(n, bool)
            for s0, e1 in segs:
                orc[int(s0 / TW.FRAME_S): int(np.ceil(e1 / TW.FRAME_S))] = True
            full = normalize_text(c["text_mono"] if cond == "mono" else c["text_user"]).split()
            allw = [w[0] for w in hw]
            arms = {"none_full": TW._counts(full, allw), "none": TW._counts(ref, allw)}
            masks = {"oracle": orc}
            for t, h_ in heads.items():
                masks[t] = TW.served_tsvad(h_, spk0 if t == "shipped" else spk, f, e0 if t == "shipped" else en)[:, 0] > 0.5
            for mn, mk in masks.items():
                kp = TW.keep_mask(hw, mk, lagf, TW.DIL)
                arms[f"{mn}_d{TW.DIL}"] = TW._counts(ref, [w[0] for w, q in zip(hw, kp) if q])
            sessions.append({"key": f"{c['name']}|{cond}", "clip": c["name"], "cond": cond, "N": len(ref),
                             "N_full": len(full), "arms": arms})
    old = {u["key"]: u for u in json.loads((TW.LIVE_WORK / "sessions.json").read_text())["sessions"]}
    for u in sessions:
        u["arms"]["stored_tsvad_d2"] = old[u["key"]]["arms"]["tsvad_d2"]
    live = {}
    arms = ["none_full", "none", "oracle_d2", "stored_tsvad_d2"] + [f"{t}_d2" for t in heads]
    for g, sel in (("all", lambda u: True), ("user_channel", lambda u: u["cond"] == "user"),
                   ("mono", lambda u: u["cond"] == "mono")):
        ss = [u for u in sessions if sel(u)]
        live[g] = TW._live_group(ss, arms)
        base, _ = TW._live_boot(ss, "shipped_d2")
        live[g]["paired_vs_shipped"] = {}
        for t in tags:
            d_, _ = TW._live_boot(ss, f"{t}_d2")
            live[g]["paired_vs_shipped"][t] = {"delta": round(live[g][f"{t}_d2"]["wer"] - live[g]["shipped_d2"]["wer"], 2),
                                               "ci_clip": TW.ci(d_ - base)}
    res["live"] = live
    log("live", {g: {arm: r[arm]["wer"] for arm in arms} for g, r in live.items()})
    save("tswer115", {"heads": tags, "spk_tag": a.spk_tag, **res}, sub=label)


def stage_tswer_live(a):
    """tswer.py live on the 0.6B: the 16 TurnBench clips x {mono mix, user channel} with a per-speaker reference; 0.6B
    streaming words on the session wav, the 0.6B TS-VAD track (0.6B print of the clip's stored 5 s interval), the
    same keep rule and scorer; rows none_full / none / tsvad_d2 / oracle_d2 next to runs/tswer_live.json's 115M rows.
    -> runs/core_0p6b.json tswer_live."""
    import torch
    import tswer as TW
    from audioforge.teachers import normalize_text
    T = _tsvad_mod("cpu")
    torch.set_num_threads(2)
    m = load_core(a.device, heads=("spk",))
    head = _tsvad_head(a.heads)
    spk = __import__("copy").deepcopy(m.heads["spk"]).cpu().eval()
    prints = json.loads(PRINTS.read_text())
    L6 = json.loads((TW.WORK / "ami" / "lag.json").read_text())["lag_0p6b_frames"]
    clips = json.loads((TW.E2E / "clips.json").read_text())
    sessions = []
    for c in clips:
        if c["set"] != "turnbench" or not c.get("text_user"):
            continue
        ref, segs = TW.tb_user_words(c["conversation"], c["human_channel"], c["start"], c["start"] + c["dur"])
        ref = normalize_text(c["text_user"]).split()
        for cond in ("mono", "user"):
            x = TW._wav(TW.E2E / "clips" / f"{c['name']}.{cond}.wav")
            outs, L, top = encode_blocks(m, [x], (SPK_TAP + 1,), top=True)
            n = L[0]
            hw = TW.words_of(m.tokenizer, TW.greedy_timed(__import__("copy").deepcopy(m.heads["rnnt"]).cpu(),
                                                          top[0, :n].float().cpu()))
            f = outs[0][SPK_TAP + 1].astype(np.float32)
            ptgt = TW.served_tsvad(head, spk, f, prints[f"{c['name']}.user"])[:, 0]
            orc = np.zeros(n, bool)
            for s0, e0 in segs:
                orc[int(s0 / TW.FRAME_S): int(np.ceil(e0 / TW.FRAME_S))] = True
            full = normalize_text(c["text_mono"] if cond == "mono" else c["text_user"]).split()
            allw = [w[0] for w in hw]
            arms = {"none_full": TW._counts(full, allw), "none": TW._counts(ref, allw)}
            for mn, mk in (("tsvad", ptgt > 0.5), ("oracle", orc)):
                kp = TW.keep_mask(hw, mk, L6, TW.DIL)
                arms[f"{mn}_d{TW.DIL}"] = TW._counts(ref, [w[0] for w, q in zip(hw, kp) if q])
            sessions.append({"key": f"{c['name']}|{cond}", "clip": c["name"], "cond": cond, "N": len(ref),
                             "N_full": len(full), "arms": arms})
            log(sessions[-1]["key"], arms)
    old = json.loads((TW.LIVE_WORK / "sessions.json").read_text())["sessions"]
    res = {}
    for g, sel in (("all", lambda u: True), ("user_channel", lambda u: u["cond"] == "user"),
                   ("mono", lambda u: u["cond"] == "mono")):
        arms = ["none_full", "none", "tsvad_d2", "oracle_d2"]
        res[g] = {"0p6b": TW._live_group([u for u in sessions if sel(u)], arms),
                  "115m": TW._live_group([u for u in old if sel(u)], arms)}
    (W / "tswer").mkdir(parents=True, exist_ok=True)
    (W / "tswer" / "live_sessions.json").write_text(json.dumps({"lag_frames": L6, "sessions": sessions}))
    save("tswer_live", res)
    log({g: {c: r[c]["tsvad_d2"]["wer"] for c in r} for g, r in res.items()})

# --------------------------------------------------------------------------- LID (lid_fix.py recipe)
LIDC = W / "lid"  # lid_fix.FIX_CACHE layout
LID_BLOCKS = "8,12,16,20"  # frame caches (the probe on pooled means of all 24 blocks picks the head's mix)


def _lid_mod(device="mps"):
    """scripts/research/lid_fix.py pointed at the 0.6B: its load_model returns the 0.6B + our VAD head (the onset of
    the 'full' view and the eval VAD come from the core's own head), 1024-d heads and head files, caches under W,
    results into runs/core_0p6b.json (not runs/lid.json)."""
    import audioforge.model as AM
    import audioforge.train as AT
    import lid_fix as L
    if not getattr(L, "_0p6b", False):
        L.FIX_CACHE, L.FIX_RUNS, L._0p6b = LIDC, LIDC / "runs", True
        AT._orig_load_model = AT.load_model
        AT.load_model = lambda path, device="cpu": load_core("cpu", heads=("vad",))
        orig_bh = AM.build_head
        AM.build_head = lambda cfg, d_model, tokenizer=None: orig_bh(cfg, 1024 if d_model == 512 else d_model, tokenizer)
        orig_save = L.save_head_file

        def save_head_file(head, mix, cfg, layers, path, meta):
            orig_save(head, mix, cfg, layers, path, meta)
            import torch
            b = torch.load(path, map_location="cpu", weights_only=False)
            b["encoder"] = {"d_model": 1024, "n_layers": 24, "att_context_size": [70, 1]}
            torch.save(b, path)
        L.save_head_file = save_head_file
        L.merge_fix = lambda key, rec: save("lid_fix", rec, sub=key)
    return L


def stage_lid_feats(a):
    """lid_fix.py feats on the 0.6B for --set / --view (frames of blocks 8/12/16/20, pooled means of all 24)."""
    L = _lid_mod(a.device)
    ns = argparse.Namespace(set=a.set, view=a.view, blocks=LID_BLOCKS, device=a.device, batch_sec=a.batch_sec,
                            cap=None, stride=1, budget=a.budget)
    L.stage_feats(ns)


def stage_lid_teacher(a):
    """The AmberNet teacher logits of lid_fix.py are keyed by (shard, row, frame window) of the 'on' view; the 0.6B
    has the same 80 ms grid, so the 115M cache's t####.npz files apply unchanged once every shard's frame counts
    match (checked row by row); linked into the 0.6B cache."""
    import lid_fix as L0
    src = Path("/Volumes/ExternalSSD/nvidia-audio-models/cache/lid_fix")
    rec = {}
    for set_ in ("train", "extra_en"):
        d0, d1 = src / set_ / "on", LIDC / set_ / "on"
        ok = bad = 0
        for f in sorted(d1.glob("s[0-9]*.json")):
            m1, m0 = json.loads(f.read_text()), json.loads((d0 / f.name).read_text())
            t0 = d0 / f"t{f.stem[1:]}.npz"
            if m1["ids"] == m0["ids"] and m1["n"] == m0["n"] and m1["onset"] == m0["onset"] and t0.exists():
                dst = d1 / t0.name
                if not dst.exists():
                    dst.symlink_to(t0)
                ok += 1
            else:
                bad += 1
        rec[set_] = {"shards_linked": ok, "shards_mismatch": bad}
    lab = src / "ambernet_labels.json"
    if not (LIDC / lab.name).exists():
        (LIDC / lab.name).symlink_to(lab)
    log(rec)
    save("lid_teacher", rec)


def stage_lid_probe(a):
    """lid_fix.py probe on the 0.6B pooled means: every 4th block (4..24)."""
    L = _lid_mod()
    rec = {}
    L.merge_fix = lambda key, r: rec.update(r)
    ns = argparse.Namespace(rest=[",".join(str(b) for b in PROBE_BLOCKS)])
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L.stage_probe(ns)
    print(buf.getvalue())
    save("lid_probe", buf.getvalue().strip().splitlines())


def stage_lid_train(a):
    """lid_fix.py train on the 0.6B: the shipped kd_h512 recipe (LanguageHead hidden 512 + softmax block mix, CE +
    0.8 x T^2 KL to AmberNet at T 2, class-balanced batches of 128, AdamW 2e-3 one-cycle, 8000 steps, dev picks the
    step) on FLEURS train (250 / language) + extra English, without trainx (47 k more FLEURS rows, not cached for the
    0.6B) and without the 'full' training view; --blocks. The same-data control on the 115M: --core115 (its cache)."""
    if a.core115:
        import lid_fix as L
        L.merge_fix = lambda key, rec: save("lid_fix_115m", rec, sub=key)
        L.FIX_RUNS = LIDC / "runs115"
    else:
        L = _lid_mod(a.device)
    ns = argparse.Namespace(tag=a.tag, blocks=a.blocks_lid, steps=8000, alpha=0.8, temp=2.0, trainx=False,
                            extra_en=True, hidden=512, context=0, rnn=0, feat_noise=0.0, dropout=0.2, lr=2e-3,
                            wd=1e-2, batch_size=128, k_rand=1, n_per_lang=0, aug=False, aug_p=0.25, full_p=0.0,
                            crop=False, frame_drop=0.1, extra_frac=0.5, eval_every=250, seed=0,
                            segment=a.budget, device=a.device)
    L.stage_train(ns)

# --------------------------------------------------------------------------- VAD v2: recipe work on cached blocks
VAD_HOLD = ("TS3011b", "ES2015c")  # AMI train meetings held out of VAD training (early stopping / selection)


def _vad_net(arch, d_in=D, hidden=64):
    """mlp = the served FrameHead shape (Linear-SiLU-Linear, per frame); conv = Linear(1024,64)-SiLU-causal
    Conv1d(64,64,k=3)-SiLU-Linear(64,1) (2 past frames of state, ~78 K params); gru = Linear(1024,48)-SiLU-GRU(48)-
    Linear(48,1) (~64 K)."""
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            h = (48 if arch == "gru" else hidden) if arch != "gru64" else 64
            self.inp = nn.Linear(d_in, h)
            if arch == "conv":
                self.conv = nn.Conv1d(h, h, 3)
            elif arch.startswith("gru"):
                self.rnn = nn.GRU(h, h, batch_first=True)
            self.out = nn.Linear(h, 1)

        def forward(self, x):  # (B, T, D) -> logits (B, T)
            h = F.silu(self.inp(x))
            if self.arch == "conv":
                h = F.silu(self.conv(F.pad(h.transpose(1, 2), (2, 0))).transpose(1, 2))
            elif self.arch.startswith("gru"):
                h, _ = self.rnn(h)
            return self.out(h).squeeze(-1)
    return Net()


def _old300_meetings():
    p = VADW / "old300_meetings.json"
    if not p.exists():
        from audioforge.datasets.ami import recipe_data
        val = recipe_data({"data": {"ami": {"mode": "diar", "n_train": 300, "seed": 0}}}, "train")
        p.write_text(json.dumps([v["meeting"] for v in val]))
    return json.loads(p.read_text())


def _vad_src(name, blocks):
    """(X (N, nb, D) array of the requested 1-based blocks, labels, win, meetings per window) of a VAD set."""
    if name == "old300":
        X, y, win = _vad_old("train", blocks)
        return X, y, win, _old300_meetings()
    if name in ("ami_dev", "icsi_dev"):
        X, y, win = _vad_old(name, blocks)
        return X, y, win, None
    meta = json.loads((VADW / f"{name}.json").read_text())
    Xm, y, cen, win = _vad_set(name)
    have = meta.get("blocks", list(VAD_BLOCKS))
    if list(have) == list(blocks):
        X = Xm  # memory-mapped, read crop by crop
    else:
        X = np.empty((len(Xm), len(blocks), D), np.float16)
        for i in range(0, len(Xm), 8192):
            X[i:i + 8192] = Xm[i:i + 8192][:, [have.index(b) for b in blocks]]
    mp = VADW / f"{name}_meetings.json"
    if not mp.exists() and name.split("_")[0].startswith("train"):
        base = name.split("_")[0]
        mp = VADW / f"{'train' if base == 'train1200' else base}_meetings.json"
    if not np.isnan(cen).all():
        y = np.where(np.isnan(cen), y, cen)
    return X, y, win, (json.loads(mp.read_text()) if mp.exists() else None)


def stage_vad2(a):
    """VAD recipe variants on cached 0.6B blocks (user request: match the shipped 115M head on AMI, keep ICSI and the
    room-tone fix). Train = --vad-train source minus the held-out AMI meetings VAD_HOLD; crops of 32 frames; 15 %
    room-tone crops; options: --arch mlp|conv|gru, --blocks-v (1-based, >1 = learned softmax mix), --aug (feature
    dropout --fdrop, Gaussian noise --noise x feature std, 2 time masks <= 4 frames (masked frames carry no loss for
    the per-frame mlp), mixup --mixup between crops), --ls label smoothing, --bw weight of frames within 1 of a label
    change; AdamW 1e-3 wd 1e-3, cosine over --steps, early stopping on the held-out set (AMI held-out meetings +
    ICSI train windows, mean AUC) every 250 steps; --seeds s1,s2,..: data seeds from one init, the final heads
    weight-averaged (model soup) as well. Scored on AMI dev / ICSI dev (64 x 20 s, the published protocol), held-out
    room tone, miss at FPR 0.075. -> runs/core_0p6b.json vad2.<tag>, HEADS_DIR/vad2_<tag>.pt."""
    import copy
    import torch
    import torch.nn.functional as F
    import vad_layers as V
    torch.set_num_threads(2)
    dev = a.device
    blocks = [int(b) for b in a.blocks_v.split(",")]
    L = 32
    srcs = []  # training sources (clean + SpecAugment views): (X, y, window starts, ends), held-out meetings dropped
    for k, name in enumerate(a.vad_train.split(",")):
        X, y, win, meets = _vad_src(name, blocks)
        idx = np.nonzero(np.r_[True, win[1:] != win[:-1]])[0]
        ends = np.r_[idx[1:], len(win)]
        keep = np.array([meets[win[i]] not in VAD_HOLD for i in idx])
        srcs.append((X, y, idx[keep], ends[keep]))
        if k == 0:  # the held-out AMI meetings of the clean source
            m_ = np.array([meets[w] not in VAD_HOLD for w in win])
            Xv1 = np.asarray(X[np.nonzero(~m_)[0]]) if isinstance(X, np.memmap) else X[~m_]
            yv1, winv1 = y[~m_], win[~m_]
    Xv2, yv2, winv2, _ = _vad_src("icsival150_all", blocks)
    Xr, yr, winr, _ = _vad_src("room_train_all", blocks)
    ridx = np.nonzero(np.r_[True, winr[1:] != winr[:-1]])[0]
    rend = np.r_[ridx[1:], len(winr)]
    nrv = len(ridx) // 5  # the last fifth of the room-tone windows: validation negatives (never trained on)
    room = (Xr, yr, ridx[:-nrv], rend[:-nrv])
    rv0 = ridx[-nrv]
    Xrv, yrv, winrv = np.asarray(Xr[rv0:]), yr[rv0:], winr[rv0:]
    sw = np.array([len(s_[2]) for s_ in srcs], float)

    def crops(name, n, g):
        out_x, out_y = [], []
        which = g.choice(len(srcs), n, p=sw / sw.sum()) if name == "tr" else np.zeros(n, int)
        for k in np.unique(which):
            Xs, ys, idx, ends = srcs[k] if name == "tr" else room
            m = int((which == k).sum())
            w = g.integers(0, len(idx), m)
            s0 = idx[w] + (g.random(m) * np.maximum(1, ends[w] - idx[w] - L)).astype(int)
            s0 = np.minimum(s0, ends[w] - L)
            for q in s0:
                out_x.append(np.asarray(Xs[q:q + L]))
                out_y.append(ys[q:q + L])
        return np.stack(out_x), np.stack(out_y)

    X0 = srcs[0][0]
    fstd = float(np.asarray(X0[: 20000], np.float32).std())

    def mixw(mix):
        return mix.softmax(0) if mix is not None else None

    def feed(net, mix, x):
        h = (x * mixw(mix)[None, None, :, None]).sum(2) if mix is not None else x[:, :, 0]
        return net(h)

    def predict(net, mix, Xs, W_):
        net.eval()
        out = np.zeros(len(Xs), np.float32)
        idx = np.nonzero(np.r_[True, W_[1:] != W_[:-1]])[0]
        ends = np.r_[idx[1:], len(W_)]
        with torch.no_grad():
            for i0, i1 in zip(idx, ends):
                x = torch.from_numpy(np.asarray(Xs[i0:i1], np.float32))[None].to(dev)
                out[i0:i1] = torch.sigmoid(feed(net, mix, x))[0].cpu().numpy()
        net.train()
        return out

    def val_score(net, mix):
        """mean AUC over (AMI held-out + room-tone val) and (ICSI train windows + room-tone val): the room-tone
        frames are negatives, so a head that fires on room tone loses AUC."""
        from sklearn.metrics import roc_auc_score
        p1, p2 = predict(net, mix, Xv1, winv1), predict(net, mix, Xv2, winv2)
        pr = predict(net, mix, Xrv, winrv)
        return 0.5 * (roc_auc_score(np.r_[yv1, yrv] > 0.5, np.r_[p1, pr])
                      + roc_auc_score(np.r_[yv2, yrv] > 0.5, np.r_[p2, pr]))

    torch.manual_seed(a.seed)
    net0 = _vad_net(a.arch).to(dev)
    mix0 = torch.nn.Parameter(torch.zeros(len(blocks), device=dev)) if len(blocks) > 1 else None
    finals = []
    hist = {}
    for ds in [int(x) for x in a.seeds.split(",")]:
        net, mix = copy.deepcopy(net0), (torch.nn.Parameter(mix0.detach().clone()) if mix0 is not None else None)
        params = list(net.parameters()) + ([mix] if mix is not None else [])
        opt = torch.optim.AdamW(params, lr=a.lr_v, weight_decay=a.wd_v)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = np.random.default_rng(1000 + ds)
        best, best_state, bad, h = -1.0, None, 0, []
        B = 64
        nr = int(round(B * a.room_share))
        t0 = time.time()
        for step in range(1, a.steps + 1):
            x1, y1 = crops("tr", B - nr, g)
            x2, y2 = crops("room", nr, g)
            x = torch.from_numpy(np.concatenate([x1, x2]).astype(np.float32)).to(dev)
            yy = torch.from_numpy(np.concatenate([y1, y2]).astype(np.float32)).to(dev)
            w = torch.ones_like(yy)
            if a.bw != 1.0:
                ch = torch.zeros_like(yy)
                d = (yy[:, 1:] != yy[:, :-1]).float()
                ch[:, 1:] += d
                ch[:, :-1] += d
                w = torch.where(ch > 0, torch.full_like(w, a.bw), w)
            if a.aug:
                x = F.dropout(x, a.fdrop) if a.fdrop > 0 else x
                x = x + a.noise * fstd * torch.randn_like(x)
                for _ in range(2):
                    ln = torch.randint(0, 5, (len(x),), device=dev)
                    st = (torch.rand(len(x), device=dev) * (L - 4)).long()
                    tt = torch.arange(L, device=dev)[None]
                    msk = (tt >= st[:, None]) & (tt < (st + ln)[:, None])
                    x = x.masked_fill(msk[..., None, None], 0.0)
                    if a.arch == "mlp":
                        w = w.masked_fill(msk, 0.0)
                if a.mixup > 0:
                    lam = float(np.random.default_rng(step + 7 * ds).beta(a.mixup, a.mixup))
                    perm = torch.randperm(len(x), device=dev)
                    x = lam * x + (1 - lam) * x[perm]
                    yy = lam * yy + (1 - lam) * yy[perm]
                    w = torch.minimum(w, w[perm])
            tgt = yy * (1 - a.ls) + 0.5 * a.ls
            z = feed(net, mix, x)
            loss = (F.binary_cross_entropy_with_logits(z, tgt, reduction="none") * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 250 == 0 or step == a.steps:
                v = val_score(net, mix)
                h.append({"step": step, "loss": round(float(loss), 4), "val_auc": round(v, 4)})
                if v > best:
                    best, bad = v, 0
                    best_state = (copy.deepcopy(net.state_dict()), mix.detach().clone() if mix is not None else None)
                else:
                    bad += 1
                if bad >= a.patience:
                    break
        hist[ds] = {"hist": h, "best_val": round(best, 4), "sec": round(time.time() - t0, 1)}
        log(a.tag, "seed", ds, hist[ds]["best_val"], "at", max(h, key=lambda r: r["val_auc"])["step"], f"{time.time() - t0:.0f} s")
        finals.append(best_state)

    def evaluate(state):
        net = _vad_net(a.arch).to(dev)
        net.load_state_dict(state[0])
        mix = torch.nn.Parameter(state[1].to(dev)) if state[1] is not None else None
        r = {"val_auc": round(val_score(net, mix), 4)}
        for sn in ("ami_dev", "icsi_dev"):
            Xs, ys, ws, _ = _vad_src(sn, blocks)
            p_ = predict(net, mix, Xs, ws)
            q = V.vad_report(V.per_window(p_, ws), V.per_window(ys, ws))
            r[sn] = {"f1": q["f1"], "auc": round(q["auc"], 4)}
            r[sn].update({k: v for k, v in q.items() if k.startswith("at_fpr")})
            np.save(VADW / f"pred2_{a.tag}_{sn}.npy", p_)
        Xh, yh, wh, _ = _vad_src("room_held_all", blocks)
        ph = predict(net, mix, Xh, wh)
        r["room_held"] = {"median_p": round(float(np.median(ph)), 4), "p95_p": round(float(np.percentile(ph, 95)), 4),
                          "share_gt_0.5": round(float((ph > 0.5).mean()), 4)}
        return r, net, mix
    res = {"args": {k: getattr(a, k) for k in ("arch", "blocks_v", "aug", "fdrop", "noise", "mixup", "ls", "bw", "steps",
                                                "seeds", "vad_train", "room_share", "patience", "lr_v", "wd_v")},
           "params": sum(p_.numel() for p_ in net0.parameters()), "seeds": hist}
    per = []
    for st in finals:
        r, _, _ = evaluate(st)
        per.append(r)
    res["per_seed"] = per
    best_i = int(np.argmax([r["val_auc"] for r in per]))
    res["best_seed"] = res["per_seed"][best_i]
    ship = finals[best_i]
    if len(finals) > 1:
        soup = ({k: sum(f[0][k] for f in finals) / len(finals) for k in finals[0][0]},
                (sum(f[1] for f in finals) / len(finals)) if finals[0][1] is not None else None)
        rs, _, _ = evaluate(soup)
        res["soup"] = rs
        if rs["val_auc"] > res["best_seed"]["val_auc"]:
            ship = soup
    res["shipped_from"] = "soup" if ship is not finals[best_i] else f"seed {best_i}"
    rf = res.get("soup") if res["shipped_from"] == "soup" else res["best_seed"]
    log(a.tag, json.dumps({k: rf[k] for k in ("val_auc", "ami_dev", "icsi_dev", "room_held")}))
    torch.save({"arch": a.arch, "blocks": blocks, "state_dict": {k: v.cpu() for k, v in ship[0].items()},
                "mix": ship[1].cpu() if ship[1] is not None else None, "eval": rf}, HEADS_DIR / f"vad2_{a.tag}.pt")
    save("vad2", res, sub=a.tag)

def stage_vad_cmp(a):
    """The shipped 115M VAD head (stage1_served_v2, block 4) on AMI dev / ICSI dev (the same 64 x 20 s windows and
    labels) and the paired window bootstrap (1000) of F1 and AUC: 0.6B head --tag (vad2) minus 115M."""
    import torch
    import vad_layers as V
    from sklearn.metrics import roc_auc_score
    from audioforge.train import load_model
    torch.set_num_threads(2)
    m = load_model(str(ROOT / "runs" / "stage1_served_v2.afm"), a.device).eval()
    res = {}
    for sn in ("ami_dev", "icsi_dev"):
        f = VADW / f"pred115_{sn}.npy"
        _, y, win = _vad_old(sn, (4,))
        if not f.exists():
            val = V.load_set(sn)
            P = []
            with torch.inference_mode():
                for i in range(0, len(val), 8):
                    x, xl = m._pad([v["audio"] for v in val[i:i + 8]])
                    enc, elen, hid = m.encode(x.to(a.device), xl.to(a.device), ATT, return_hidden=True)
                    p_ = m.heads["vad"](m.head_input("vad", enc, hid)).sigmoid().cpu().numpy()
                    P += [p_[j, : int(elen[j])] for j in range(len(elen))]
            pw = [P[w][: int((win == w).sum())] for w in range(len(P))]
            np.save(f, np.concatenate(pw))
        pb = np.load(f)
        pa = np.load(VADW / f"pred2_{a.tag}_{sn}.npy")
        assert len(pa) == len(pb) == len(y)
        W_ = int(win.max()) + 1
        rows = [np.nonzero(win == w)[0] for w in range(W_)]

        def f1(p_, t):
            d, tt = p_ > 0.5, t > 0.5
            return 2 * float((d & tt).sum()) / max(float(d.sum() + tt.sum()), 1)
        base = {"f1_0p6b": f1(pa, y), "f1_115m": f1(pb, y), "auc_0p6b": roc_auc_score(y > 0.5, pa),
                "auc_115m": roc_auc_score(y > 0.5, pb)}
        rng = np.random.default_rng(0)
        df, da = [], []
        for _ in range(1000):
            ii = np.concatenate([rows[k] for k in rng.integers(0, W_, W_)])
            df.append(f1(pa[ii], y[ii]) - f1(pb[ii], y[ii]))
            da.append(roc_auc_score(y[ii] > 0.5, pa[ii]) - roc_auc_score(y[ii] > 0.5, pb[ii]))
        rep = V.vad_report(V.per_window(pb, win), V.per_window(y, win))
        res[sn] = {**{k: round(v, 4) for k, v in base.items()},
                   "f1_delta_ci95": [round(float(np.percentile(df, q)), 4) for q in (2.5, 97.5)],
                   "auc_delta_ci95": [round(float(np.percentile(da, q)), 4) for q in (2.5, 97.5)],
                   "115m_at_fpr": {k: v for k, v in rep.items() if k.startswith("at_fpr")}}
        log(sn, res[sn])
    save("vad_cmp", {"tag": a.tag, **res})

# --------------------------------------------------------------------------- layer sweeps (research/LAYER_SWEEP_0P6B.md)
ALL_BLOCKS = tuple(range(1, NB + 1))


def _probe_configs(scores: dict, n_blocks=NB):
    """(a) every single block, (b) the learned softmax mix over all blocks, (c) concatenation of the top-3 singles."""
    return ([("b%d" % b, [b], "single") for b in range(1, n_blocks + 1)] + [("mix_all", list(range(1, n_blocks + 1)), "mix")])


def _top3(res, key):
    singles = sorted([(v[key], int(k[1:])) for k, v in res.items() if k.startswith("b") and k[1:].isdigit()], reverse=True)
    return sorted(b for _, b in singles[:3])


def stage_sweep_vad(a):
    """VAD layer sweep: the served FrameHead probe (Linear(D,64)-SiLU-Linear) on every block 1..24, on a learned
    softmax mix over all 24 and on the concatenation of the top-3 singles; train = the 300 AMI train windows minus
    the held-out meetings VAD_HOLD, + 15 % room-tone frames; selection metric = held-out AUC (AMI held-out meetings,
    ICSI train windows, each with the room-tone validation negatives), reported per corpus (domain shift). 1500
    steps x 2048 frames, AdamW 1e-3, input dropout 0.2. -> runs/core_0p6b.json sweep_vad."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(2)
    dev = a.device
    X, y, win = _vad_old("train", ALL_BLOCKS)
    meets = np.array(_old300_meetings())[win]
    tr = ~np.isin(meets, VAD_HOLD)
    Xi, yi, _, _ = _vad_src("icsival150_all", list(ALL_BLOCKS))
    Xi = np.asarray(Xi)
    Xr, yr, winr, _ = _vad_src("room_train_all", list(ALL_BLOCKS))
    Xr = np.asarray(Xr)
    cut = int(np.nonzero(winr >= winr.max() - winr.max() // 5)[0][0])
    Xrt, Xrv = Xr[:cut], Xr[cut:]
    Xt, yt, Xa, ya = X[tr], y[tr], X[~tr], y[~tr]
    res = load_json("sweep_vad", {})
    cfgs = _probe_configs(res)
    if all(k in res for k, _, _ in cfgs) and "concat_top3" not in res:
        t3 = _top3(res, "val_auc")
        cfgs.append(("concat_top3", t3, "concat"))
    for name, blocks, kind in cfgs + ([("concat_top3", _top3(res, "val_auc"), "concat")] if "concat_top3" not in res and
                                       all(k in res for k, _, _ in cfgs) else []):
        if name in res:
            continue
        idx = [b - 1 for b in blocks]
        d_in = D * (len(idx) if kind == "concat" else 1)
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(d_in, 64), torch.nn.SiLU(),
                                  torch.nn.Linear(64, 1)).to(dev)
        mix = torch.nn.Parameter(torch.zeros(len(idx), device=dev)) if kind == "mix" else None
        opt = torch.optim.AdamW(list(net.parameters()) + ([mix] if mix is not None else []), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = np.random.default_rng(0)

        def inp(Xs):
            x = torch.from_numpy(np.asarray(Xs[:, idx], np.float32)).to(dev)
            if kind == "mix":
                return (x * mix.softmax(0)[None, :, None]).sum(1)
            return x.reshape(len(x), -1)
        t0 = time.time()
        for step in range(a.steps):
            i1 = g.integers(0, len(Xt), 1741)
            i2 = g.integers(0, len(Xrt), 307)
            x = torch.cat([inp(Xt[i1]), inp(Xrt[i2])])
            yy = torch.from_numpy(np.r_[yt[i1], np.zeros(len(i2))].astype(np.float32)).to(dev)
            loss = F.binary_cross_entropy_with_logits(net(x).squeeze(-1), yy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            pr = lambda Xs: np.concatenate([torch.sigmoid(net(inp(Xs[i:i + 8192]))).squeeze(-1).cpu().numpy()  # noqa
                                            for i in range(0, len(Xs), 8192)])
            pa, pi, prv = pr(Xa), pr(Xi), pr(Xrv)

        def f1(p_, t):
            d_, tt = p_ > 0.5, t > 0.5
            return 2 * float((d_ & tt).sum()) / max(float(d_.sum() + tt.sum()), 1)
        r = {"blocks": blocks, "kind": kind,
             "ami_heldout_auc": round(roc_auc_score(np.r_[ya, np.zeros(len(prv))] > 0.5, np.r_[pa, prv]), 4),
             "icsi_heldout_auc": round(roc_auc_score(np.r_[yi, np.zeros(len(prv))] > 0.5, np.r_[pi, prv]), 4),
             "ami_heldout_f1": round(f1(pa, ya), 4), "icsi_heldout_f1": round(f1(pi, yi), 4),
             "room_val_p95": round(float(np.percentile(prv, 95)), 4)}
        r["val_auc"] = round(0.5 * (r["ami_heldout_auc"] + r["icsi_heldout_auc"]), 4)
        if mix is not None:
            r["mix"] = [round(float(v), 3) for v in mix.detach().softmax(0).cpu()]
        res[name] = r
        save("sweep_vad", res)
        log(name, r["val_auc"], r["ami_heldout_auc"], r["icsi_heldout_auc"], f"{time.time() - t0:.0f} s")
        if time.time() - T_START > a.budget:
            log("budget")
            return
    log("sweep_vad done")


T_START = time.time()

SWEEP = W / "sweep"


def _pool_blocks(m, audios):
    """(B, 24, 2048) fp16: per block, mean and std over the item's frames (masked [70,1] forward)."""
    outs, L, _ = encode_blocks(m, audios, ALL_BLOCKS)
    res = []
    for o in outs:
        res.append(np.stack([np.concatenate([o[b].astype(np.float32).mean(0), o[b].astype(np.float32).std(0)])
                             for b in ALL_BLOCKS]).astype(np.float16))
    return res


def stage_sweep_spk_feats(a):
    """Pooled statistics of all 24 blocks for the speaker sweep: (1) LibriSpeech train-clean-100 items of the
    speaker recipe (spk_frame 'librispeech', <= 20 per speaker, 200 speakers) with their TitaNet-L teacher
    embeddings; (2) held-out meeting clips: the TS-VAD enrollment clips (1-5 s of one speaker alone) of the 12 AMI + 12
    ICSI train meetings, 8 per (meeting, speaker). Resumable in parts."""
    import torch
    import spk_frame as SF
    import tsvad as T0
    torch.set_num_threads(2)
    SWEEP.mkdir(parents=True, exist_ok=True)
    m = load_core(a.device, heads=())
    t0 = time.time()
    out = SWEEP / "spk_libri.npz"
    if not out.exists():
        data = SF.load_source("librispeech")
        z = np.load(SF.feat_path("librispeech", "115m" if C115 else "0p6b"), mmap_mode="r")
        tch = np.load(SF.CACHE / "librispeech.teacher.npz")["whole"]
        spk = np.array([int(d.get("speaker", -1)) for d in data])
        rng = np.random.default_rng(0)
        spks = sorted(set(spk.tolist()))
        pick_s = set(rng.choice(spks, min(200, len(spks)), replace=False).tolist())
        idx = [i for s_ in sorted(pick_s) for i in np.nonzero(spk == s_)[0][:20]]
        part = SWEEP / "spk_libri.part.npz"
        P = dict(np.load(part)) if part.exists() else {"X": np.zeros((0, NB, 2 * D), np.float16), "n": np.array(0)}
        done = int(P["n"])
        X = [P["X"]]
        for i in range(done, len(idx), 8):
            if time.time() - t0 > a.budget:
                break
            X += _pool_blocks(m, [np.asarray(data[j]["audio"], np.float32) for j in idx[i:i + 8]])
            done = min(i + 8, len(idx))
        Xc = np.concatenate([x if x.ndim == 3 else x[None] for x in X])
        if done < len(idx):
            np.savez(part, X=Xc, n=np.array(done))
            log(f"spk_libri {done}/{len(idx)}")
            return
        assert [str(z["ids"][j]) for j in idx[:5]] and len(tch) == len(data)
        np.savez(out, X=Xc, teacher=tch[idx], speaker=spk[idx])
        part.unlink(missing_ok=True)
        log(f"spk_libri: {len(idx)} items")
    out = SWEEP / "spk_meet.npz"
    if not out.exists():
        T = _tsvad_mod(a.device)
        rows, auds = [], []
        for corpus, ds in T.train_sets():
            for mt in ds.meetings:
                e = np.load(TSV / "enroll" / f"{corpus}_{mt}.npz")
                for sp in sorted(set(e["spk"].tolist())):
                    for k in np.nonzero(e["spk"] == sp)[0][:8]:
                        iv = [tuple(x) for x in e["ivs"][k] if x[0] >= 0]
                        rows.append((corpus, mt, sp))
                        auds.append(T0.clip_audio(ds, mt, iv))
        X = []
        for i in range(0, len(auds), 16):
            X += _pool_blocks(m, auds[i:i + 16])
        np.savez(out, X=np.stack(X), corpus=np.array([r[0] for r in rows]), meeting=np.array([r[1] for r in rows]),
                 speaker=np.array([r[2] for r in rows]))
        log(f"spk_meet: {len(rows)} clips ({time.time() - t0:.0f} s)")


def _eer_within(E, meet, spk):
    from audioforge.metrics import eer
    import torch
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    iu = np.triu_indices(len(E), 1)
    same_m = meet[iu[0]] == meet[iu[1]]
    s_ = (E[iu[0]] * E[iu[1]]).sum(1)[same_m]
    lab = (spk[iu[0]] == spk[iu[1]])[same_m]
    return float(eer(torch.tensor(s_), torch.tensor(lab.astype(np.int64))))


def stage_sweep_spk(a):
    """Speaker layer sweep: the same small probe on every block / the mix of all 24 / the concat of the top-3 --
    pooled mean + std of the block (2048-d) -> Linear(., 192), trained on the 200-speaker LibriSpeech subset to the
    TitaNet-L embedding (0.5 x (1 - cos) + 1.0 x relational MSE, the speaker recipe's distillation terms; AdamW 1e-3,
    1500 steps x 128); scored by within-meeting EER on held-out meeting clips (AMI and ICSI train meetings, never
    seen by the probe), per corpus. -> runs/core_0p6b.json sweep_spk."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    dev = a.device
    zl, zm = np.load(SWEEP / "spk_libri.npz"), np.load(SWEEP / "spk_meet.npz")
    Xl, Tl = zl["X"].astype(np.float32), torch.tensor(zl["teacher"], dtype=torch.float32)
    Xm, cm, mm, sm = zm["X"].astype(np.float32), zm["corpus"], zm["meeting"], zm["speaker"]
    res = load_json("sweep_spk", {})
    cfgs = _probe_configs(res)
    if all(k in res for k, _, _ in cfgs):
        cfgs.append(("concat_top3", _top3(res, "score"), "concat"))
    for name, blocks, kind in cfgs:
        if name in res:
            continue
        idx = [b - 1 for b in blocks]
        torch.manual_seed(0)
        d_in = 2 * D * (len(idx) if kind == "concat" else 1)
        net = torch.nn.Linear(d_in, 192).to(dev)
        mix = torch.nn.Parameter(torch.zeros(len(idx), device=dev)) if kind == "mix" else None
        mu = Xl[:, idx].mean(0, keepdims=True)
        sd = Xl[:, idx].std(0, keepdims=True) + 1e-4

        def inp(Xs):
            x = torch.from_numpy((Xs[:, idx] - mu) / sd).to(dev)
            if kind == "mix":
                return (x * mix.softmax(0)[None, :, None]).sum(1)
            return x.reshape(len(x), -1)
        opt = torch.optim.AdamW(list(net.parameters()) + ([mix] if mix is not None else []), lr=1e-3, weight_decay=1e-2)
        g = np.random.default_rng(0)
        for step in range(a.steps):
            ii = g.integers(0, len(Xl), 128)
            e = F.normalize(net(inp(Xl[ii])), dim=-1)
            t = F.normalize(Tl[ii].to(dev), dim=-1)
            loss = 0.5 * (1 - (e * t).sum(-1)).mean() + F.mse_loss(e @ e.T, t @ t.T)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            E = net(inp(Xm)).cpu().numpy()
        r = {"blocks": blocks, "kind": kind}
        for c in ("ami", "icsi"):
            k = cm == c
            r[f"{c}_eer_within"] = round(_eer_within(E[k], mm[k], sm[k]), 4)
        r["score"] = round(-0.5 * (r["ami_eer_within"] + r["icsi_eer_within"]), 4)
        if mix is not None:
            r["mix"] = [round(float(v), 3) for v in mix.detach().softmax(0).cpu()]
        res[name] = r
        save("sweep_spk", res)
        log(name, r["ami_eer_within"], r["icsi_eer_within"])
    log("sweep_spk done")

ICSI_HOLD = ("Bro026", "Bmr022")  # ICSI train meetings held out of the TS-VAD sweep's probe training


def _tsvad_sweep_data():
    """Windows of the VAD sweep caches (old300: AMI train, icsival150_all: ICSI train; 24 blocks) with their
    per-speaker activity and the TS-VAD enrollment clips (0.6B speaker-head embeddings) of each meeting speaker."""
    from audioforge.datasets.ami import AMI, recipe_data as ami_rd
    from audioforge.datasets.icsi import ICSI, recipe_data as icsi_rd
    out = []
    for corpus, rd, DS, src in (("ami", ami_rd, AMI, "old300"), ("icsi", icsi_rd, ICSI, "icsival150_all")):
        n = 300 if corpus == "ami" else 150
        wins = rd({"data": {corpus: {"mode": "diar", "n_train": n, "seed": 0}}}, "train")
        ds = DS(sorted({v["meeting"] for v in wins}), verbose=False)
        X, y, win, _ = _vad_src(src, list(ALL_BLOCKS))
        starts = np.nonzero(np.r_[True, win[1:] != win[:-1]])[0]
        ends = np.r_[starts[1:], len(win)]
        enr = {}
        for w, v in enumerate(wins):
            mt = v["meeting"]
            if mt not in enr:
                e = np.load(TSV / "enroll" / f"{corpus}_{mt}.npz")
                enr[mt] = {sp: (e["emb"][e["spk"] == sp], e["ivs"][e["spk"] == sp]) for sp in set(e["spk"].tolist())}
            off = getattr(ds, "speaker_offset", 0)
            names = [ds.speaker_ids[int(s_) - off] if int(s_) >= 0 else None for s_ in v["speakers"]]
            T = ends[w] - starts[w]
            act = np.asarray(v["spk_targets"], np.float32)[:T]
            out.append({"corpus": corpus, "meeting": mt, "a": float(v["start"]), "X": (X, starts[w], ends[w]),
                        "act": act, "names": names, "enr": enr[mt]})
    return out


def stage_sweep_tsvad(a):
    """TS-VAD layer sweep: TSVADHead (hidden 128, prenet; the served recipe's head) on every block / the mix of all 24 /
    the concat of the top-3, the enrollment = a 1-5 s clip of the target from elsewhere in the meeting embedded by the
    0.6B speaker head (block 11, fixed); train = AMI + ICSI train windows minus held-out meetings (VAD_HOLD, ICSI_HOLD),
    1000 steps x 32 windows, AdamW 2e-3 one-cycle, overlap weight 2.5; held-out target F1 and overlap recall per
    corpus. -> runs/core_0p6b.json sweep_tsvad."""
    import torch
    import torch.nn.functional as F
    from audioforge.heads.tsvad import TSVADHead
    torch.set_num_threads(2)
    dev = a.device
    data = _tsvad_sweep_data()
    hold = set(VAD_HOLD) | set(ICSI_HOLD)
    tr = [d for d in data if d["meeting"] not in hold]
    va = [d for d in data if d["meeting"] in hold]
    rng = np.random.default_rng(0)

    def enroll(d, j, rng, first=False):
        name = d["names"][j]
        if name not in d["enr"]:
            return None
        E, IV = d["enr"][name]
        a0, b0 = d["a"] - 1.0, d["a"] + len(d["act"]) * 0.08 + 1.0
        ok = [c for c in range(len(E)) if not any(iv[0] >= 0 and iv[0] < b0 and iv[1] > a0 for iv in IV[c])]
        if not ok:
            return None
        return E[ok[0] if first else int(rng.choice(ok))]

    def sample(d, rng, first=False, j=None):
        act = d["act"]
        cols = [c for c in range(min(act.shape[1], len(d["names"]))) if d["names"][c] in d["enr"]]
        if j is None:
            active = [c for c in cols if act[:, c].any()]
            if not cols:
                return None
            j = int(rng.choice(active if active and rng.random() < 0.8 else cols))
        e = enroll(d, j, rng, first)
        if e is None:
            return None
        oth = np.delete(act, j, 1).max(1) if act.shape[1] > 1 else np.zeros(len(act))
        return np.stack([act[:, j], oth], 1), e

    res = load_json("sweep_tsvad", {})
    cfgs = _probe_configs(res)
    if all(k in res for k, _, _ in cfgs):
        cfgs.append(("concat_top3", _top3(res, "score"), "concat"))
    for name, blocks, kind in cfgs:
        if name in res:
            continue
        idx = [b - 1 for b in blocks]
        torch.manual_seed(0)
        head = TSVADHead(D * (len(idx) if kind == "concat" else 1), emb_dim=192, hidden=128, prenet=True).to(dev)
        mix = torch.nn.Parameter(torch.zeros(len(idx), device=dev)) if kind == "mix" else None

        def feats(d):
            X, s0, s1 = d["X"]
            x = torch.from_numpy(np.asarray(X[s0:s1][:, idx], np.float32)).to(dev)
            if kind == "mix":
                return (x * mix.softmax(0)[None, :, None]).sum(1)
            return x.reshape(len(x), -1)
        params = list(head.parameters()) + ([mix] if mix is not None else [])
        steps = a.steps
        opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
        g = np.random.default_rng(0)
        t0 = time.time()
        for step in range(steps):
            xs, ys, es = [], [], []
            while len(xs) < 32:
                d = tr[int(g.integers(len(tr)))]
                smp = sample(d, g)
                if smp is None:
                    continue
                T = min(len(smp[0]), 200)
                xs.append(feats(d)[:T])
                ys.append(torch.from_numpy(smp[0][:T]))
                es.append(torch.from_numpy(np.asarray(smp[1], np.float32)))
            T = min(len(x) for x in xs)
            x = torch.stack([x_[:T] for x_ in xs])
            yy = torch.stack([y_[:T] for y_ in ys]).to(dev)
            e = torch.stack(es).to(dev)
            z = head(x, None, e, None)
            wt = 1.0 + 1.5 * ((yy[..., 0] > 0.5) & (yy[..., 1] > 0.5)).float()
            loss = (F.binary_cross_entropy_with_logits(z.float(), yy, reduction="none") * wt[..., None]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        head.eval()
        r = {"blocks": blocks, "kind": kind}
        with torch.no_grad():
            for c in ("ami", "icsi"):
                P, Y = [], []
                for d in va:
                    if d["corpus"] != c:
                        continue
                    x = feats(d)[None]
                    for j in range(min(d["act"].shape[1], len(d["names"]))):
                        if not d["act"][:, j].any():
                            continue
                        smp = sample(d, None, first=True, j=j)
                        if smp is None:
                            continue
                        p_ = head.decode(x, None, torch.from_numpy(np.asarray(smp[1], np.float32))[None].to(dev))
                        P.append(p_[0].cpu().numpy())
                        Y.append(smp[0][: P[-1].shape[0]])
                P, Y = np.concatenate(P), np.concatenate(Y)
                d_, t_ = P[:, 0] > 0.5, Y[:, 0] > 0.5
                tp = float((d_ & t_).sum())
                r[f"{c}_f1_target"] = round(2 * tp / max(float(d_.sum() + t_.sum()), 1), 4)
                ov = (Y[:, 0] > 0.5) & (Y[:, 1] > 0.5)
                r[f"{c}_recall_overlap"] = round(float((P[:, 0][ov] > 0.5).mean()), 4) if ov.any() else None
        r["score"] = round(0.5 * (r["ami_f1_target"] + r["icsi_f1_target"]), 4)
        if mix is not None:
            r["mix"] = [round(float(v), 3) for v in mix.detach().softmax(0).cpu()]
        res[name] = r
        save("sweep_tsvad", res)
        log(name, r, f"{time.time() - t0:.0f} s")
        if time.time() - T_START > a.budget:
            log("budget")
            return
    log("sweep_tsvad done")

SEG_SWEEP_N = 1000


def _seg_sweep_ids():
    """The v5 sweep's training subset: SEG_SWEEP_N clips of the turn sets (stable hash order; 50 % oto / AMI / ICSI,
    30 % smart-turn, 20 % cuts), all with cached turn inputs."""
    import zlib
    import turn_v5 as V5
    man = V5.manifest()
    cuts = json.loads((V5.W / "cuts.json").read_text()) if not getattr(V5, "_0p6b", False) else \
        json.loads((TURN / "cuts.json").read_text())
    key = lambda cid: zlib.crc32(("sweep" + cid).encode())  # noqa: E731
    conv = sorted([c["id"] for c in man if c["src"] != "st"], key=key)
    st = sorted([c["id"] for c in man if c["src"] == "st"], key=key)
    cu = sorted([c["id"] for c in cuts], key=key)
    ids = conv[: SEG_SWEEP_N // 2] + st[: SEG_SWEEP_N * 3 // 10] + cu[: SEG_SWEEP_N // 5]
    if C115:  # the 115M's own served turn inputs (turn_v4 / turn_v5 caches)
        allc = {c["id"]: c for c in V5.all_clips()}
        return [i for i in ids if i in allc and V5.feat_path(allc[i]).exists()]
    return [i for i in ids if (TURN_INP / f"{i}.npz").exists()]


def _stt_eval_ids():
    """smart-turn human_5_all eval-split clip ids (stt_<i>) with cached served inputs on this core."""
    if not C115:
        return [p_.stem for p_ in sorted(TURN_INP.glob("stt_*.npz"))]
    import turn_v5 as V5
    from audioforge.datasets import smartturn as ST
    meta = json.loads(ST.build_cache(verbose=False).read_text())
    return [f"stt_{i:05d}" for i in ST.split_indices(meta)["eval"] if (V5.STEST4 / f"st_{i:05d}.npz").exists()]


def _served_inp(cid):
    """The served turn inputs (vad, pu, po, y, n) file of a turn clip id on this core."""
    if not C115:
        return TURN_INP / f"{cid}.npz"
    import turn_v5 as V5
    if cid.startswith("stt_"):
        return V5.STEST4 / f"st_{cid[4:]}.npz"
    if not hasattr(_served_inp, "idx"):
        _served_inp.idx = {x["id"]: x for x in V5.all_clips()}
    c = _served_inp.idx.get(cid)
    return V5.feat_path(c) if c else V5.FEATS4 / f"{cid}.npz"


def stage_sweep_seg_feats(a):
    """All 24 blocks of the v5 sweep subset + the smart-turn eval clips (stt_*) -> SWEEP/blk24/<id>.npz."""
    import torch
    torch.set_num_threads(2)
    od = SWEEP / f"blk{NB}"
    od.mkdir(parents=True, exist_ok=True)
    ids = _seg_sweep_ids() + _stt_eval_ids()
    todo = [i for i in ids if not (od / f"{i}.npz").exists()]
    log(f"sweep_seg_feats: {len(todo)} of {len(ids)}")
    if not todo:
        return
    items = {}
    for which in ("man", "cuts", "stt"):
        for cid, fn, _ in turn_items(which):
            items[cid] = fn
    m = load_core(a.device, heads=())
    t0 = time.time()
    for i in range(0, len(todo), 8):
        if time.time() - t0 > a.budget:
            break
        cs = todo[i:i + 8]
        outs, L, _ = encode_blocks(m, [items[c]() for c in cs], ALL_BLOCKS)
        for c, o in zip(cs, outs):
            if C115:  # the served inputs' frame count (turn_v5 blocks: |T - L| <= 1)
                T_ = len(np.load(_served_inp(c))["pu"])
                assert abs(T_ - len(o[1])) <= 1, (c, T_, len(o[1]))
                o = {b: np.concatenate([o[b], o[b][-1:]])[:T_] for b in ALL_BLOCKS}
            np.savez(od / f"{c}.tmp.npz", **{f"b{b}": o[b] for b in ALL_BLOCKS})
            (od / f"{c}.tmp.npz").rename(od / f"{c}.npz")
    log(f"sweep_seg_feats: {len(todo) - min(len(todo), i + 8)} left ({time.time() - t0:.0f} s)")


def stage_sweep_seg(a):
    """v5 turn-classifier layer sweep on the sweep subset: SegTurn (2 x 256, the c5 architecture) on every block /
    the learned softmax mix of all 24 / the concat of the top-3 singles; main BCE + 0.5 x smart-turn distillation,
    --steps x 256 (128 for the mix), AdamW 1e-4 wd 0.05, dropout 0.2; selection = held-out end-vs-pause AUC
    (turn_v5's val groups inside the subset) and its early-quiet-frame AUC; reported with the smart-turn eval clips'
    accuracy (p 160 ms after the last VAD frame >= 0.5; the stest protocol; not used for selection).
    -> runs/core_0p6b.json sweep_seg."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import turn_v4 as V4
    from audioforge.heads.turn_seg import SegTurn
    if C115:
        import turn_v5 as V5
    else:
        V5 = _v5_mod()
    torch.set_num_threads(2)
    dev = a.device
    V5.BLK = SWEEP / f"blk{NB}"  # SegData keeps only the clips with a block file: the sweep subset
    res = load_json("sweep_seg", {})
    # the mix reads the 12 even blocks (2, 4, .., 24): 24 blocks of 1000 clips do not fit in RAM next to the model;
    # 115M: the 9 odd blocks (1, 3, .., 17)
    cfgs = [c for c in _probe_configs(res) if c[0] != "mix_all"] + \
        ([("mix_odd", list(range(1, 18, 2)), "mix")] if C115 else [("mix_even", list(range(2, 25, 2)), "mix")])
    if a.rerun:  # re-run named configs with the held-out smart-turn-clip accuracy recorded (key <name>_r)
        cfgs = [(f"{n}_r", b, k) for n, b, k in cfgs if n in a.rerun.split(",")] + \
               ([("concat_top3_r", _top3(res, "auc"), "concat")] if "concat_top3" in a.rerun.split(",") else [])
    if all(k in res for k, _, _ in cfgs):
        cfgs.append(("concat_top3", _top3(res, "auc"), "concat"))

    class MixSeg(nn.Module):
        def __init__(self, nb):
            super().__init__()
            self.nb = nb
            self.mix = nn.Parameter(torch.zeros(nb))
            self.seg = SegTurn(d_in=D, n_extra=3, d=256, n_layers=2, heads=4, ff=1024, win=V5.WIN, use_text=True,
                               dropout=0.2, block="mix")

        def forward(self, x, *rest):
            B, T_, D = x.shape
            x = (x.reshape(B, T_, self.nb, D) * self.mix.softmax(0)[None, None, :, None]).sum(2)
            return self.seg(x, *rest)
    for name, blocks, kind in cfgs:
        if name in res:
            continue
        bstr = "+".join(str(b) for b in blocks)
        data = V5.SegData(bstr)
        S = data.S
        trn, va = S[S[:, 7] == 0], S[S[:, 7] == 1]
        torch.manual_seed(0)
        model = (MixSeg(len(blocks)) if kind == "mix" else
                 SegTurn(d_in=D * len(blocks), n_extra=3, d=256, n_layers=2, heads=4, ff=1024, win=V5.WIN,
                         use_text=True, dropout=0.2, block=bstr)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4 if kind != "mix" else 3e-4, weight_decay=0.05)
        kinds = trn[:, 4].astype(int)
        kw = {0: 1.0, 1: 1.5, 2: 0.5, 3: 0.7, 4: 0.5}
        wts = np.array([kw[k] for k in kinds]) / np.bincount(kinds, minlength=5)[kinds]
        wts /= wts.sum()
        g = np.random.default_rng(0)
        B = 128 if kind == "mix" else 256
        t0 = time.time()
        for step in range(a.steps):
            lr = (1e-4 if kind != "mix" else 3e-4) * min(1.0, (step + 1) / 100) * 0.5 * (1 + np.cos(np.pi * step / a.steps))
            for gr in opt.param_groups:
                gr["lr"] = lr
            rows = trn[g.choice(len(trn), B, p=wts)]
            b = data.batch(rows, dev)
            drop = torch.rand(len(rows), device=dev) < 0.25
            b[2][drop, :, 1:3] = 0.0
            out = model(*b)
            lab = torch.tensor(rows[:, 2], dtype=torch.float32, device=dev)
            npos = lab.sum().clamp(min=1)
            loss = F.binary_cross_entropy_with_logits(out["main"], lab, pos_weight=((len(lab) - npos) / npos).clamp(0.1, 10))
            stt = torch.tensor(rows[:, 5], dtype=torch.float32, device=dev)
            if (stt >= 0).any():
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(out["st"][stt >= 0], stt[stt >= 0])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        r_, _ = V5.seg_eval(model, data, va[: 20000], dev)
        # smart-turn eval clips (stest protocol)
        model.eval()
        from audioforge.datasets import smartturn as ST
        meta = json.loads(ST.build_cache(verbose=False).read_text())
        lab_st = np.asarray(meta["complete"], bool)
        pt, yt = [], []
        for i in ST.split_indices(meta)["eval"]:
            cid = f"stt_{i:05d}"
            fp, bp = _served_inp(cid), SWEEP / f"blk{NB}" / f"{cid}.npz"
            if not (fp.exists() and bp.exists()):
                continue
            d = V5.npz(fp)
            T_ = len(d["pu"])
            x = V5.load_block(d, bp, bstr)[:T_].astype(np.float32)
            ex = np.stack([d["vad"], d["pu"], d["po"]], 1).astype(np.float32)
            c = {"x": x, "ex": ex, "n": d["n"].astype(np.int32), "y": d["y"].astype(np.int32),
                 "since": V5.since_token(d["n"].astype(np.int32))}
            fake = type("D", (), {})()
            fake.clips = [c]
            v = np.nonzero(np.asarray(d["vad"], float) > V4.VAD_ON)[0]
            fr = min(T_ - 1, (v[-1] if len(v) else 0) + 2)
            with torch.no_grad():
                bb = V5.SegData.batch(fake, np.array([[0, fr]], np.float64), dev)
                pt.append(float(torch.sigmoid(model(*bb)["main"])[0]))
            yt.append(lab_st[i])
        pt, yt = np.array(pt), np.array(yt)
        r = {"blocks": blocks, "kind": kind, "auc": round(r_["auc"], 4), "auc_early": round(r_["auc_early"], 4),
             "acc_st_heldout": round(100 * r_["acc_st"], 1) if r_.get("acc_st") is not None else None,
             "auc_end_vs_cut": round(r_["auc_end_vs_cut"], 4) if r_["auc_end_vs_cut"] else None,
             "stest_acc": round(100 * float(((pt >= 0.5) == yt).mean()), 1) if len(pt) else None, "stest_n": int(len(pt)),
             "sec": round(time.time() - t0, 1)}
        if kind == "mix":
            r["mix"] = [round(float(v_), 3) for v_ in model.mix.detach().softmax(0).cpu()]
        res[name] = r
        save("sweep_seg", res)
        log(name, r)
        del data
        if time.time() - T_START > a.budget:
            log("budget")
            return
    log("sweep_seg done")

def stage_sweep_lid(a):
    """LID layer sweep (lid_fix.py probe protocol, selected on FLEURS dev, never test): standardise + logistic
    regression (C 0.1) on the 2 s onset-anchored mean of each block (1..24), on the concatenation of the top-3 singles,
    and a learned softmax mix of all 24 (torch linear probe, same data); fitted on FLEURS train (250 / language),
    dev accuracy at 2 s (+ full utterance). -> runs/core_0p6b.json sweep_lid."""
    import torch
    L = _lid_mod()
    Ptr, ytr = L.load_pool("train", "on")
    Pdv, ydv = L.load_pool("dev", "on")
    Pdf, _ = L.load_pool("dev", "full")
    res = load_json("sweep_lid", {})
    w2, wa = L.POOL_WIN.index("2s"), L.POOL_WIN.index("all")
    for b in ALL_BLOCKS:
        k = f"b{b}"
        if k in res:
            continue
        p2 = L._logreg(Ptr[:, w2, b - 1].astype(np.float32), ytr, Pdv[:, w2, b - 1].astype(np.float32))
        pf = L._logreg(Ptr[:, wa, b - 1].astype(np.float32), ytr, Pdf[:, wa, b - 1].astype(np.float32))
        res[k] = {"blocks": [b], "kind": "single", "dev_acc_2s": round(float((p2 == ydv).mean()), 4),
                  "dev_acc_full": round(float((pf == ydv).mean()), 4)}
        log(k, res[k])
        save("sweep_lid", res)
    t3 = _top3(res, "dev_acc_2s")
    if "concat_top3" not in res:
        X = lambda P, w: np.concatenate([P[:, w, b - 1].astype(np.float32) for b in t3], 1)  # noqa: E731
        p2 = L._logreg(X(Ptr, w2), ytr, X(Pdv, w2))
        pf = L._logreg(X(Ptr, wa), ytr, X(Pdf, wa))
        res["concat_top3"] = {"blocks": t3, "kind": "concat", "dev_acc_2s": round(float((p2 == ydv).mean()), 4),
                              "dev_acc_full": round(float((pf == ydv).mean()), 4)}
        save("sweep_lid", res)
    if "mix_all" not in res:
        torch.manual_seed(0)
        out = {}
        for wn, w, Pd in (("2s", w2, Pdv), ("full", wa, Pdf)):
            xt = torch.from_numpy(Ptr[:, w].astype(np.float32))
            mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True) + 1e-4
            xt = (xt - mu) / sd
            xd = (torch.from_numpy(Pd[:, w].astype(np.float32)) - mu) / sd
            mix = torch.nn.Parameter(torch.zeros(24))
            lin = torch.nn.Linear(1024, int(ytr.max()) + 1)
            opt = torch.optim.AdamW([mix, *lin.parameters()], lr=3e-3, weight_decay=1e-2)
            yt = torch.from_numpy(ytr).long()
            for step in range(1500):
                ii = torch.randint(0, len(xt), (256,))
                z = lin((xt[ii] * mix.softmax(0)[None, :, None]).sum(1))
                loss = torch.nn.functional.cross_entropy(z, yt[ii])
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                pd = lin((xd * mix.softmax(0)[None, :, None]).sum(1)).argmax(-1).numpy()
            out[wn] = round(float((pd == ydv).mean()), 4)
            if wn == "2s":
                mw = [round(float(v), 3) for v in mix.detach().softmax(0)]
        res["mix_all"] = {"blocks": list(ALL_BLOCKS), "kind": "mix", "dev_acc_2s": out["2s"], "dev_acc_full": out["full"],
                          "mix": mw}
        save("sweep_lid", res)
    log("sweep_lid", {k: v["dev_acc_2s"] for k, v in res.items()})

def stage_spk_frame(a):
    """spk_frame.py (the crop-level TitaNet distillation recipe of the shipped 0.6B speaker head c0p6b_fresh, block 11)
    on another 0.6B block (--block, 1-based) as core '0p6b<block>': feats / train (fresh head, 4000 steps) / evalcore
    (--sub feats|train|evalcore)."""
    import spk_frame as SF
    core = f"0p6b{a.block}"
    SF.CORES[core] = (AFM, int(a.block) - 1, 1024)
    ns = argparse.Namespace(core=core, device=a.device, budget=a.budget, tag=f"c0p6b{a.block}_fresh", steps=4000,
                            batch=128, lr=5e-4, cos_w=0.5, rel_w=1.0, aam_w=1.0, p_crop=0.7, seed=0, init="fresh")
    SF.OUT = W / "spk_frame.json"
    getattr(SF, f"stage_{a.sub}")(ns)

def stage_turn_retrack(a):
    """After moving the speaker / TS-VAD heads to block 5: recompute P(user) / P(other) of every cached turn clip with a
    print (--set man|cuts|stt) from blocks 1-5 of the clip and of its print audio (the served TS-VAD track, stored
    print, anchored adaptation), in place (marker key 'tsv5'). Resumable."""
    import torch
    from audioforge.tsvad_stream import embed_frames, load_tsvad, track_probs
    torch.set_num_threads(2)
    items = turn_items(a.set)
    todo = [it for it in items if it[2] is not None and "tsv5" not in np.load(TURN_INP / f"{it[0]}.npz").files]
    log(f"turn_retrack {a.set}: {len(todo)} of {len(items)} to do")
    if not todo:
        return
    m = load_core(a.device, heads=("spk",))
    spk = __import__("copy").deepcopy(m.heads["spk"]).cpu().eval()
    head = load_tsvad(str(HEADS_DIR / "tsvad.pt"), 1024)
    t0, done = time.time(), 0
    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > a.budget:
            break
        cs = todo[i:i + a.batch]
        xs = [fn() for _, fn, _ in cs]
        prs = [pf() for _, _, pf in cs]
        ok = [j for j, p_ in enumerate(prs) if len(p_) >= 8000]
        fx, _, _ = encode_blocks(m, xs, (SPK_TAP + 1,))
        fp, _, _ = encode_blocks(m, [prs[j] for j in ok], (SPK_TAP + 1,)) if ok else ([], None, None)
        pmap = {j: embed_frames(spk, f[SPK_TAP + 1].astype(np.float32)) for j, f in zip(ok, fp)}
        for j, (cid, _, _) in enumerate(cs):
            f = TURN_INP / f"{cid}.npz"
            d = dict(np.load(f))
            T = len(d["pu"])
            if j in pmap:
                P = track_probs(head, spk, fx[j][SPK_TAP + 1].astype(np.float32)[:T], pmap[j])
                P = np.concatenate([P, np.repeat(P[-1:], max(0, T - len(P)), 0)])[:T]
            else:
                P = np.zeros((T, 2), np.float32)
            d["pu"], d["po"], d["tsv5"] = P[:, 0].astype(np.float16), P[:, 1].astype(np.float16), np.array(1)
            np.savez(f.with_name(f.stem + ".tmp.npz"), **d)
            f.with_name(f.stem + ".tmp.npz").rename(f)
        done += len(cs)
    log(f"turn_retrack {a.set}: {done} in {time.time() - t0:.0f} s; {len(todo) - done} left")

def stage_sweep_report(a):
    """Markdown tables of the layer sweeps (runs/core_0p6b.json sweep_*) for research/LAYER_SWEEP_0P6B.md."""
    d = json.loads(OUT.read_text())
    spec = {"sweep_vad": ("VAD", [("ami_heldout_auc", "AMI held-out AUC"), ("icsi_heldout_auc", "ICSI held-out AUC"),
                                   ("ami_heldout_f1", "AMI F1"), ("icsi_heldout_f1", "ICSI F1"),
                                   ("room_val_p95", "room-tone p95"), ("val_auc", "selection (mean AUC)")], "val_auc", 1),
            "sweep_spk": ("speaker", [("ami_eer_within", "AMI within-meeting EER"), ("icsi_eer_within", "ICSI within-meeting EER")],
                          "score", 1),
            "sweep_tsvad": ("TS-VAD", [("ami_f1_target", "AMI target F1"), ("ami_recall_overlap", "AMI overlap recall"),
                                       ("icsi_f1_target", "ICSI target F1"), ("icsi_recall_overlap", "ICSI overlap recall"),
                                       ("score", "selection (mean F1)")], "score", 1),
            "sweep_seg": ("v5 turn classifier", [("auc", "end-vs-pause AUC"), ("auc_early", "AUC, first quiet frames"),
                                                 ("acc_st_heldout", "held-out smart-turn clip acc %"),
                                                 ("auc_end_vs_cut", "end vs cut AUC"), ("stest_acc", "smart-turn test acc %")],
                          "auc", 1),
            "sweep_lid": ("LID", [("dev_acc_2s", "FLEURS dev acc @2 s"), ("dev_acc_full", "dev acc, full")], "dev_acc_2s", 1)}
    for key, (title, cols, sel, _) in spec.items():
        r = d.get(key)
        if not r:
            continue
        best = max(r, key=lambda k: r[k][sel] if r[k].get(sel) is not None else -9)
        print(f"\n### {title}\n")
        print("| config | " + " | ".join(c[1] for c in cols) + " |")
        print("|---|" + "---:|" * len(cols))
        for k, v in r.items():
            name = k if k != "concat_top3" else f"concat {'+'.join(str(b) for b in v['blocks'])}"
            cells = []
            for c, _ in cols:
                x = v.get(c)
                cells.append("-" if x is None else (f"{100 * x:.1f}" if "eer" in c else f"{x}"))
            print(f"| {'**' + name + '**' if k == best else name} | " + " | ".join(cells) + " |")
        print(f"\nselected by `{sel}`: **{best}**")

def stage_lid_teacher_run(a):
    """lid_fix.py teacher (AmberNet logits on the training windows of the 0.6B 'on' shards) for --set; used where the
    115M cache's teacher files do not apply (extra_en: rows without a Silero onset take the core's own VAD onset)."""
    L = _lid_mod(a.device)
    d = LIDC / a.set / "on"
    for f in d.glob("t*.npz"):
        if f.is_symlink():
            f.unlink()
    L.stage_teacher(argparse.Namespace(set=a.set, device=a.device, budget=a.budget, k_rand=1, batch_sec=a.batch_sec))

def stage_cost(a):
    """Served-engine cost of the 0.6B (mps_115m.py protocols, the same code paths): --sub engine = full single-mode
    engine per-160-ms-chunk compute on the bundled clip (warm-up, best of 3), --sub streams = K single-mode sessions
    interleaved on one thread (real time while p95 of the summed per-block compute < 160 ms). The clip's voice print
    is re-made for the 0.6B from the clip's user intervals. -> runs/core_0p6b.json cost_engine / cost_streams."""
    import functools
    import audioforge
    import mps_115m as M
    from audioforge.data import load_wav
    pf = W / "two_party_call_16s.voiceprint_0p6b.json"
    if not pf.exists():
        meta = json.loads((ROOT / "examples/audio/two_party_call_16s.json").read_text())
        x = load_wav(str(M.CLIP), SR).astype(np.float32)
        seg = np.concatenate([x[int(s0 * SR): int(e0 * SR)] for s0, e0 in meta["user_intervals"]])
        pf.write_text(json.dumps(audioforge.voiceprint(seg, core="0.6b")))
    M.PRINT = pf
    M.SCRATCH = W / "cost"
    orig = audioforge.load
    audioforge.load = functools.partial(orig, core="0.6b")
    if a.sub == "engine":
        M.stage_engine(a)
        save("cost_engine", json.loads((M.SCRATCH / "engine.json").read_text()))
    else:
        a.ks = [int(k) for k in a.ks.split(",")]
        M.stage_streams(a)
        save("cost_streams", json.loads((M.SCRATCH / f"streams_{a.device}.json").read_text()), sub=a.device)

def _preset_grid():
    import itertools
    G = {"balanced": [], "fast": [], "assistant": []}
    for k, fb, vt in itertools.product((1, 2, 3, 4), (6, 8, 10, 12), (0.4, 0.5, 0.6)):
        G["balanced"].append({"mode": "head", "gate": True, "quiet_db": None, "k": k, "th": 0.99, "fb": fb,
                              "vad_thr": vt, "others": (12, 8)})
    for k, mvt, mp, fb in itertools.product((1, 2, 3), (0.4, 0.5, 0.6),
                                            (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99), (8, 9, 12)):
        G["fast"].append({"mode": "model", "gate": True, "quiet_db": None, "k": k, "mvt": mvt, "mp": mp, "fb": fb,
                          "vad_thr": 0.4, "reask": True})
    for k, mvt, mp in itertools.product((1, 2, 3, 4), (0.4, 0.5), (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99)):
        G["assistant"].append({"mode": "model", "gate": True, "quiet_db": 6.0, "mqo": True, "k": k, "mvt": mvt, "mp": mp,
                               "fb": 37, "vad_thr": 0.4, "reask": True})
    return G


def stage_preset_scan(a):
    """Re-tune the three presets' constants for the 0.6B's own heads, the way the 115M presets were chosen on the same
    sets (turn_v5 scan4 / EOT_LATENCY: run_policy twin of the served VadHeadPolicy on the dumped frames, eot_latency /
    eot_assistant scorers): balanced = the fastest calls p50 that is at least as good as the 115M's balanced on calls
    and AMI false interruptions and misses; fast = the fastest calls p50 with calls FI <= 25 % and calls missed <= 8.3 %
    (the 115M fast rule's bar) and the bundled clip not cut; assistant = the best smart-turn-test accuracy with p50 <=
    400 ms. Tuned on the evaluation sets themselves, as the 115M presets were. -> runs/core_0p6b.json presets_0p6b."""
    import eot_latency as E
    import eot_assistant as EA
    import turn_v5 as V5
    sd = EA.load_dump()
    d6 = {d_["key"]: d_ for d_ in (json.loads(p_.read_text()) for p_ in sorted(EOT_DUMP.glob("*.json")))}
    a6 = {d_["key"]: d_ for d_ in (json.loads(p_.read_text()) for p_ in sorted(ASST_DUMP.glob("*.json")))}
    for k, d in a6.items():
        d["conf"] = sd[k]["conf"]
    sess = [s_ for s_ in E.sessions() if s_["key"] in d6]
    asess = EA.sessions(a6)
    comp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in d6.items()}
    acomp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in a6.items()}
    dbs = {k: np.load(V5.ENERGY / f"{k}.npy") for k in d6}
    adbs = {k: np.load(V5.ENERGY / f"asst_{k}.npy") for k in a6}
    for dd, dbd in ((d6, dbs), (a6, adbs)):
        for k, d in dd.items():
            v = np.asarray(d["head"]["v"])
            dbd[k] = dbd[k][np.clip(v, 0, len(dbd[k]) - 1)]
    ref = load_json("eot")["115m"]
    rows = {}
    gc = {}
    for fam, rules in _preset_grid().items():
        out = []
        for r in rules:
            pk = "p" if r["mode"] == "head" else "p5"
            view = lambda d: {"head": {**d["head"], "p": d["head"][pk]}}  # noqa: E731
            per = {sc: [] for sc in E.SCOPES}
            for ss in sess:
                t = [x + comp[ss["key"]] for x, _ in V5.run_policy(view(d6[ss["key"]]), dbs[ss["key"]], r,
                                                                    gc.setdefault(ss["key"], {}))]
                for sc, sets in E.SCOPES.items():
                    if ss["set"] in sets:
                        per[sc].append(E.score_session(t, ss, comp[ss["key"]]))
            row = {"rule": r, "calls": E.pool(per["two_party_user"]), "ami": E.pool(per["ami"])}
            if fam == "assistant" or fam == "fast":
                at = {s_["key"]: [x + acomp[s_["key"]] for x, _ in V5.run_policy(view(a6[s_["key"]]), adbs[s_["key"]], r,
                                                                                 gc.setdefault("a:" + s_["key"], {}))]
                      for s_ in asess}
                asc = EA.score_system(asess, at, acomp)
                row["asst"] = {"accuracy_pct": asc["accuracy_pct"], "p50": asc["complete"]["eot_total_ms_p50"],
                               "false_fire_pct": asc["incomplete"]["false_fire_pct"]}
            out.append(row)
        c, m = (lambda x: x["calls"]), (lambda x: x["ami"])
        if fam == "balanced":
            rb = ref["balanced"]
            ok = [x for x in out if c(x)["eot_total_ms_p50"] is not None
                  and c(x)["false_interruption_pct"] <= rb["calls"]["false_interruption_pct"]
                  and c(x)["missed_pct"] <= rb["calls"]["missed_pct"]
                  and m(x)["false_interruption_pct"] <= rb["ami"]["false_interruption_pct"]
                  and m(x)["missed_pct"] <= rb["ami"]["missed_pct"]]
            pick = min(ok, key=lambda x: c(x)["eot_total_ms_p50"]) if ok else \
                min(out, key=lambda x: c(x)["missed_pct"] + c(x)["false_interruption_pct"]
                    + m(x)["missed_pct"] + m(x)["false_interruption_pct"])
        elif fam == "fast":
            ok = [x for x in out if c(x)["eot_total_ms_p50"] is not None and c(x)["false_interruption_pct"] <= 25.0
                  and c(x)["missed_pct"] <= 8.3]
            pick = min(ok, key=lambda x: c(x)["eot_total_ms_p50"]) if ok else None
        else:
            ok = [x for x in out if (x["asst"]["p50"] or 1e9) <= 400]
            pick = max(ok, key=lambda x: (x["asst"]["accuracy_pct"] or 0, -(x["asst"]["p50"] or 0))) if ok else None
        rows[fam] = {"n_rules": len(out), "n_ok": len(ok), "pick": pick}
        log(fam, len(out), len(ok), json.dumps(pick)[:400])
    save("presets_0p6b", rows)

def stage_vad_swap(a):
    """What-if for the VAD head on the end-of-turn benchmarks: the session audio of the calls / AMI dumps re-encoded
    (masked [70,1] forward = the streamed frames), the VAD of head file(s) --heads (vad_<tag>.pt / vad2_<tag>.pt in
    HEADS_DIR) put in place of the dumped VAD, every preset re-scored (turn head p and v5 p as dumped).
    -> runs/core_0p6b.json vad_swap."""
    import torch
    import eot_latency as E
    import turn_v5 as V5
    torch.set_num_threads(2)
    m = load_core(a.device, heads=())
    d6 = {d_["key"]: d_ for d_ in (json.loads(p_.read_text()) for p_ in sorted(EOT_DUMP.glob("*.json")))}
    sess = [s_ for s_ in E.sessions() if s_["key"] in d6]
    heads = {}
    for tag in a.heads.split(","):
        f = HEADS_DIR / (f"{tag}.pt" if tag.startswith("vad") else f"vad_{tag}.pt")
        ck = torch.load(f, map_location="cpu", weights_only=False)
        if "arch" in ck:
            net = _vad_net(ck["arch"]).eval()
        else:
            net = torch.nn.Sequential(torch.nn.Linear(1024, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).eval()
            net.load_state_dict({k.replace("net.", ""): v for k, v in ck["state_dict"].items()})
            net = (lambda n_: (lambda x: n_(x).squeeze(-1)))(net)
        if "arch" in ck:
            net.load_state_dict(ck["state_dict"])
        heads[tag] = (net, ck["mix"].softmax(0), ck.get("blocks", list(VAD_BLOCKS)))
    vads = {t: {} for t in heads}
    blocks = sorted({b for _, _, bl in heads.values() for b in bl})
    for ss in sess:
        x = E.read_audio(ss)
        o, L, _ = encode_blocks(m, [x], blocks)
        for t, (net, w, bl) in heads.items():
            h = sum(float(w[j]) * torch.from_numpy(o[0][b].astype(np.float32)) for j, b in enumerate(bl))
            with torch.no_grad():
                vads[t][ss["key"]] = torch.sigmoid(net(h[None]))[0].numpy()
    res = {}
    for t in heads:
        d2 = {}
        for k, d in d6.items():
            h = dict(d["head"])
            v = np.asarray(h["v"])
            vv = vads[t][k]
            h["vad"] = vv[np.clip(v, 0, len(vv) - 1)].tolist()
            d2[k] = {**d, "head": h}
        dbs = {k: np.load(V5.ENERGY / f"{k}.npy") for k in d2}
        comp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in d2.items()}
        out = {}
        for name, r in _preset_rules().items():
            pk = "p" if r["mode"] == "head" else "p5"
            per = {sc: [] for sc in E.SCOPES}
            for ss in sess:
                d = d2[ss["key"]]
                db = dbs[ss["key"]][np.clip(np.asarray(d["head"]["v"]), 0, len(dbs[ss["key"]]) - 1)]
                tt = [x + comp[ss["key"]] for x, _ in V5.run_policy({"head": {**d["head"], "p": d["head"][pk]}}, db, r, {})]
                for sc, sets in E.SCOPES.items():
                    if ss["set"] in sets:
                        per[sc].append(E.score_session(tt, ss, comp[ss["key"]]))
            out[name] = {"calls": E.pool(per["two_party_user"]), "ami": E.pool(per["ami"])}
            log(t, name, {c: [out[name][c][q] for q in ("eot_total_ms_p50", "false_interruption_pct", "missed_pct")]
                          for c in ("calls", "ami")})
        res[t] = out
    save("vad_swap", res)

def stage_served_check(a):
    """The served 0.6B session with --turn-preset assistant / fast / balanced (vad_head) on N assistant clips and N
    calls sessions: its turn_end times vs the offline twin (run_policy) of the preset as the engine resolved it, on the
    session's own frames (the per-model preset override included). -> runs/core_0p6b.json served_check."""
    import torch
    import audioforge.serve as S
    import eot_latency as E
    import eot_assistant as EA
    import turn_v5 as V5
    torch.set_num_threads(2)
    eng = engine_0p6b(a.device)
    eng.warmup()
    prints = json.loads(PRINTS.read_text())
    res = {}
    rules = {"assistant": dict(V5.PRESET_RULES["assistant"], k=4, mp=0.99), "fast": V5.F5, "balanced": V5.BAL}
    for preset in ("assistant", "fast", "balanced"):
        same, n = 0, 0
        items = [("a", c) for c in EA.clips()[: a.n]] + [("c", s_) for s_ in E.sessions()[: a.n]]
        for kind, it in items:
            x = EA.audio(it) if kind == "a" else E.read_audio(it)
            s = S.Session(eng, S.SessionConfig(turn_policy="vad_head", turn_preset=preset))
            if kind == "c" and it["embedding"] is not None:
                s.arm_enrollment("enroll", 0, embedding=prints[it["key"]])
            if s.asr.seg is None and eng.seg_name is not None:
                sh = eng.asr.heads[eng.seg_name]
                s.asr.attach_seg(sh, next(sh.parameters()).device)
            rec = []
            _record(s, rec, enrolled_only=True)
            msgs = []
            for i in range(0, len(x), 320):
                msgs += s.process(x[i:i + 320])
            msgs += s.finish()
            h = {k: [r[j] for r in rec] for j, k in enumerate(("v", "t", "p", "vad", "pu", "po", "p5"))}
            if rules[preset]["mode"] == "model":
                h["p"] = h["p5"]
            db = V5.ready_db(x, max(h["v"]) + 1, h["v"], h["t"])
            off = [round(t_, 3) for t_, _ in V5.run_policy({"head": h}, db[np.asarray(h["v"])], rules[preset], {})]
            served = [m_["t"] for m_ in msgs if m_["type"] == "turn_end"]
            same += served == off
            n += 1
        res[preset] = f"{same}/{n}"
        log(preset, res[preset])
    save("served_check", res)


STAGES = {"import": stage_import, "wer": stage_wer, "wer_report": stage_wer_report, "live_wer": stage_live_wer,
          "vad_probe": stage_vad_probe, "vad_feats": stage_vad_feats, "vad_train": stage_vad_train,
          "vad_room115": stage_vad_room115, "vad2": stage_vad2, "sweep_vad": stage_sweep_vad, "sweep_spk_feats": stage_sweep_spk_feats,
          "sweep_spk": stage_sweep_spk, "sweep_tsvad": stage_sweep_tsvad,
          "sweep_seg_feats": stage_sweep_seg_feats, "sweep_seg": stage_sweep_seg, "sweep_lid": stage_sweep_lid, "spk_frame": stage_spk_frame, "turn_retrack": stage_turn_retrack,
          "sweep_report": stage_sweep_report, "cost": stage_cost, "preset_scan": stage_preset_scan, "vad_swap": stage_vad_swap, "served_check": stage_served_check, "vad_cmp": stage_vad_cmp, "tsvad_feats": stage_tsvad_feats, "tsvad_sim": stage_tsvad_sim,
          "tsvad_train": stage_tsvad_train, "tsvad_win": stage_tsvad_win, "tsvad_eval": stage_tsvad_eval,
          "turn_cache": stage_turn_cache, "seg_train": stage_seg_train, "turn_train": stage_turn_train,
          "build": stage_build, "build115": stage_build115, "prints": stage_prints, "eot_dump": stage_eot_dump, "eot_score": stage_eot_score,
          "tswer": stage_tswer, "tswer115": stage_tswer115, "tswer_live": stage_tswer_live, "lid_feats": stage_lid_feats,
          "lid_teacher": stage_lid_teacher, "lid_teacher_run": stage_lid_teacher_run, "lid_probe": stage_lid_probe, "lid_train": stage_lid_train, "verify": stage_verify, "core": stage_core, "core_streams": stage_core_streams}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sha", action="store_true")
    ap.add_argument("--no-perf", action="store_true")
    ap.add_argument("--ks", default="1,2,3,4,5,6")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--budget", type=float, default=540)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--room-share", type=float, default=0.15)
    ap.add_argument("--centre", action="store_true")
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--vad-sets", default="train,room_train,room_held")
    ap.add_argument("--vad-blocks", default=None)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--blocks-v", default="4,8,12,16,20,24")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--fdrop", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--ls", type=float, default=0.0)
    ap.add_argument("--bw", type=float, default=1.0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr-v", type=float, default=1e-3)
    ap.add_argument("--wd-v", type=float, default=1e-3)
    ap.add_argument("--vad-train", default="train")
    ap.add_argument("--sim-share", type=float, default=0.3)
    ap.add_argument("--ov-w", type=float, default=2.5)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--corpora", default=None)
    ap.add_argument("--heads", default="a")
    ap.add_argument("--set", default="man")
    ap.add_argument("--block", default="12")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux", type=float, default=0.0)
    ap.add_argument("--seg-tag", default="s12")
    ap.add_argument("--which", default="calls")
    ap.add_argument("--sub", default="feats")
    ap.add_argument("--rerun", default="")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--view", default="on")
    ap.add_argument("--batch-sec", type=float, default=240.0)
    ap.add_argument("--blocks-lid", default="12,16")
    ap.add_argument("--core115", action="store_true")
    ap.add_argument("--turn-tag", default="f1")
    ap.add_argument("--n-utt", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--core", default="0.6b", choices=("0.6b", "115m", "3.5"))
    ap.add_argument("--vad-tag", default=None)
    ap.add_argument("--spk-tag", default=None)
    ap.add_argument("--presets-json", default=None)
    ap.add_argument("--prompt", default="en-US")
    ap.add_argument("--att-right", type=int, default=None)
    a = ap.parse_args()
    if C35:  # mps_115m.time_stream (stage core) streams at the core's 160 ms context
        import mps_115m as M
        M.ATT = list(ATT)
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
