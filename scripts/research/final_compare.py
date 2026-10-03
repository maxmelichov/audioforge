"""research/FINAL_COMPARE.md: one consolidated measurement pass of both audioforge cores against the strongest open
models that run locally, per job, on the same audio and labels -> runs/final_compare.json.

Every stage is one process, < 10 min per call (``--budget``), resumable, and runs through scripts/dev/gate.sh (one
heavy torch / MPS job at a time on this Mac). Intermediates live on the SSD under WORK. ``report`` reads the stage
outputs (and the earlier run files that are reused, each named in the json) and writes runs/final_compare.json.

  asr     --system S --set X   S: whisper_small (faster-whisper int8 CPU, the LiveKit default), whisper_turbo /
                               whisper_large_v3 (transformers fp16 on MPS), tdt_v3 (NeMo Parakeet-TDT 0.6B v3, CPU),
                               core_115m / core_0p6b (the served cores' streaming RNNT, masked forward == 160 ms
                               streaming, CPU); X: libri | ami | icsi (the 200-item sets of final_asr / hybrid_asr) |
                               live (the 32 two-party sessions with references: 16 TurnBench clips x mono / user) |
                               fleurs_en (the 150 FLEURS en_us test clips cached for the LID test, raw transcription);
                               also core_115m_f1120 / core_0p6b_f1120 (the 1.12 s final, att [70,13], DUAL_RATE.md)
  sttlat  --sys 115m|0p6b      streaming word latency (stt_latency.py protocol) through the served engine on --device
  vad     --system S           per-frame speech scores on AMI dev 64 x 20 s and ICSI dev 64 x 20 s (vad_layers sets)
  lid     --system S           FLEURS-17 test (2550 clips), 2 s from the speech onset and the full clip
  eou                          NVIDIA Parakeet-Realtime-EOU streamed in 160 ms blocks over the 232 calls / AMI sessions
                               and the 399 smart-turn clips: a turn end wherever it emits <EOU>
  eotdump --sys 115m|0p6b --which calls|asst
                               the served single-mode engine on --device, every per-frame signal a preset reads
  spkeer  --system S           within-meeting speaker EER on the AMI / ICSI dev segments of spk_head.py (n = 200)
  pyatracks / pyatn / pyaframe / pyatwer
                               pyannote speaker-diarization-3.1 + the same 5 s voice print on the tswer windows
  cost    --sys 115m|0p6b --sub engine|streams   ms per 160 ms chunk / real-time streams (mps_115m.py protocols)
  params                       parameter counts / file sizes of every system
  report                       -> runs/final_compare.json

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/final_compare.py <stage> [...]
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

SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
WORK = Path(os.environ.get("FINAL_COMPARE_W", SSD / "scratch" / "final_compare"))
OUT = ROOT / "runs" / "final_compare.json"
SR = 16000
CHUNK = 2560  # 160 ms
HUB = Path.home() / ".cache/huggingface/hub"
# The shipped builds are the defaults (no environment variable is needed to reproduce the published numbers): the
# 115M with served heads v0.4, the English 0.6B with heads v0.4 (hub.HEADS / HEADS_0P6B) and the LID heads v2. The
# environment variables only point a stage at a candidate build.
AFM_115M = Path(os.environ.get("FINAL_115M_AFM", ROOT / "runs" / "stage1_served_v4.afm"))
AFM_0P6B = SSD / "scratch" / "core_0p6b" / "served_0p6b_v0.4.afm"
LID_HEAD = {"115m": ROOT / "assets" / "lid_115m_v2.pt", "0p6b": ROOT / "assets" / "lid_0p6b_v2.pt"}
E2E = SSD / "scratch" / "e2e_tsvad"
T_START = time.time()
# the AMI turn rows come from the AMI test (eval) meetings (eot_latency reads this when it is imported)
os.environ.setdefault("EOT_AMI_SPLIT", "eval")
# served-engine turn dumps per core: the 0.6B's were re-made with heads v0.4 (research/TURN_DATA.md stage fc_v04)
EOT_DIR = {"115m": WORK / "eot", "0p6b": SSD / "scratch" / "turndata" / "fc_v04" / "eot"}
# Files that produced each published section (labels only: the per-item outputs do not record the build). The 0.6B
# v0.3 and v0.4 builds have bit-identical tensors except v0.4's two extra turn heads (turn_vad, turn_seg_a) and its
# assistant preset; the 115M v0.3 and v0.4 builds are identical except v0.4's extra speech head (plans/fixwave).
FILES_USED = {
    "words": {"115m": "runs/stage1_served_v4.afm", "0p6b": "served_0p6b_v0.3.afm (tensors = v0.4 minus the turn heads)"},
    "vad": {"115m": "runs/stage1_served_v4.afm (heads.speech)", "0p6b": "served_0p6b_v0.3.afm (heads.speech, = v0.4)"},
    "lid": {"115m": "assets/lid_115m_v2.pt on stage1_served_v4.afm", "0p6b": "assets/lid_0p6b_v2.pt on served_0p6b_v0.3.afm"},
    "speaker_eer": {"115m": "runs/stage1_served_v4.afm (heads.spk)", "0p6b": "served_0p6b_v0.3.afm (heads.spk, = v0.4)"},
    "turn": {"115m": "dumps: calls + assistant clips with stage1_served_v3.afm, AMI test with stage1_served_v4.afm (the "
                     "turn path reads heads.vad / turn / turn_seg, bit-identical in v3 and v4)",
             "0p6b": "dumps: served_0p6b_v0.4.afm (scratch/turndata/fc_v04/eot)"},
    "cost": {"115m": "runs/stage1_served_v4.afm + lid_115m_v2.pt", "0p6b": "served_0p6b_v0.3.afm + lid_0p6b_v2.pt"},
}


def afm_0p6b() -> Path:
    return Path(os.environ.get("FINAL_0P6B_AFM", AFM_0P6B))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def over(a) -> bool:
    return time.time() - T_START > a.budget


def snap(repo: str) -> str:
    return str(next((HUB / f"models--{repo.replace('/', '--')}" / "snapshots").iterdir()))


def peak_rss_mb() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2 ** 20 if sys.platform == "darwin" else r / 1024, 1)


def jl_rows(p: Path) -> dict:
    return {r["i"]: r for r in map(json.loads, p.read_text().splitlines()) if r} if p.exists() else {}


# =========================================================================== words (ASR)
def live_sessions() -> list[dict]:
    """The 32 scored live sessions of research/TSWER.md "live suite" (tswer.py stage_live): the 16 TurnBench clips with
    per-speaker transcripts x {mono mix, user channel}; reference = clips.json text_mono (mono, both parties) or
    text_user (user channel). Order: clips.json order, mono then user."""
    out = []
    for c in json.loads((E2E / "clips.json").read_text()):
        if c["set"] != "turnbench" or not c.get("text_user"):
            continue
        for cond in ("mono", "user"):
            out.append({"key": f"{c['name']}|{cond}", "clip": c["name"], "cond": cond,
                        "wav": str(E2E / "clips" / f"{c['name']}.{cond}.wav"),
                        "ref": c["text_mono"] if cond == "mono" else c["text_user"]})
    return out


def asr_set(name: str):
    """-> (list of audio arrays, list of reference strings, list of group ids for the bootstrap)."""
    if name == "live":
        import soundfile as sf
        S = live_sessions()
        xs = []
        for s in S:
            x, sr = sf.read(s["wav"], dtype="float32")
            assert sr == SR
            xs.append(x if x.ndim == 1 else x.mean(1))
        return xs, [s["ref"] for s in S], [s["clip"] for s in S]
    if name == "fleurs_en":  # FLEURS en_us test: the 150 clips of data/lid/fleurs/manifest.jsonl (the LID test subset,
        import soundfile as sf  # fixed before any WER); references = raw transcription column of the en_us test.tsv
        tsv = ROOT / "data/lid/fleurs/_tsv/data/en_us/test.tsv"
        ref = {r[1].rsplit(".", 1)[0]: r[2] for r in (ln.split("\t") for ln in tsv.read_text().splitlines() if ln)}
        rows = [r for r in map(json.loads, (ROOT / "data/lid/fleurs/manifest.jsonl").read_text().splitlines())
                if r.get("lang") == "en" and r.get("split") == "test"]
        assert len(rows) == 150, len(rows)
        xs = []
        for r in rows:
            x, sr = sf.read(str(ROOT / r["path"]), dtype="float32")
            assert sr == SR
            xs.append(x if x.ndim == 1 else x.mean(1))
        return xs, [ref[r["id"]] for r in rows], list(range(len(rows)))
    if name in ("ls_clean", "ls_other"):  # LibriSpeech test-clean / test-other: 300 utterances, seeded random over the
        import random                        # whole split (every speaker can be drawn; FIXALL test audit)
        import soundfile as sf
        split = {"ls_clean": "test-clean", "ls_other": "test-other"}[name]
        root = ROOT / "data/librispeech/LibriSpeech" / split
        items = []
        for tr in sorted(root.glob("*/*/*.trans.txt")):
            for ln in tr.read_text().splitlines():
                uid, txt = ln.split(" ", 1)
                items.append((uid, tr.parent / f"{uid}.flac", txt.lower()))
        pick = sorted(random.Random(0).sample(range(len(items)), 300))
        xs = []
        for i in pick:
            x, sr = sf.read(str(items[i][1]), dtype="float32")
            assert sr == SR
            xs.append(x)
        return xs, [items[i][2] for i in pick], list(range(len(pick)))
    if name.endswith("_eval"):  # the corpora's eval meetings, the final_asr protocol (FIXALL test audit)
        f = WORK / "asr_sets" / f"{name}.npz"
        if not f.exists():
            import random
            if name == "ami_eval":
                from audioforge.datasets.ami import AMI, subset
                ds = AMI(subset({"eval": 4})["eval"], verbose=False)
            else:
                from audioforge.datasets.icsi import ICSI, subset
                ds = ICSI(subset()["eval"], verbose=False)
            segs = ds.asr(1.0, 15.0)
            idx = sorted(random.Random(0).sample(range(len(segs)), min(200, len(segs))))
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, **{f"a{i}": np.asarray(segs[j]["audio"], np.float32) for i, j in enumerate(idx)})
            (WORK / "asr_sets" / f"{name}_refs.json").write_text(json.dumps([segs[j]["text"] for j in idx]))
        z = np.load(f)
        refs = json.loads((WORK / "asr_sets" / f"{name}_refs.json").read_text())
        import final_scoring as FS  # bootstrap groups = meetings (WORK/meta, final_scoring.py meta)
        groups = FS.asr_groups(name)
        assert groups is None or len(groups) == len(refs), name
        return [z[f"a{i}"] for i in range(len(refs))], refs, groups if groups else list(range(len(refs)))
    import hybrid_asr as H
    xs, refs = H.load_fa_set(name)
    return xs, refs, list(range(len(refs)))


ASR_SYSTEMS = {
    "whisper_small_b5": "OpenAI Whisper small (244M) via faster-whisper 1.2.1 / CTranslate2 (Systran/faster-whisper-small), "
                        "int8, CPU 2 threads, faster-whisper's own transcribe() defaults (beam 5, best_of 5, temperature "
                        "fallback, previous-text conditioning, timestamps) with language en: what Pipecat 1.12's local "
                        "WhisperSTTService calls (it passes only language / hotwords / prompt)",
    "whisper_small": "OpenAI Whisper small (244M) via faster-whisper 1.2.1 / CTranslate2 (Systran/faster-whisper-small), "
                     "int8, CPU 2 threads, beam 1 (greedy), language en, no timestamps, no VAD filter, no previous-text "
                     "conditioning (final_asr.py settings; second row, the headline row is whisper_small_b5)",
    "whisper_turbo": "OpenAI Whisper large-v3-turbo (809M) via transformers 5.17 (openai/whisper-large-v3-turbo), fp16 "
                     "on MPS, greedy, language en; clips > 30 s: Whisper's sequential long-form decoding",
    "whisper_large_v3": "OpenAI Whisper large-v3 (1.55B) via transformers 5.17 (openai/whisper-large-v3), fp16 on MPS, "
                        "greedy, language en; clips > 30 s: sequential long-form decoding",
    "tdt_v3": "NVIDIA Parakeet-TDT 0.6B v3 (data/nemo/parakeet-tdt-0.6b-v3.nemo imported with audioforge.nemo_import), "
              "offline full context, greedy TDT, batch 1, CPU 2 threads (hybrid_asr.py load_tdt / tdt_transcribe)",
    "core_115m": "audioforge 115M core (runs/stage1_served_v4.afm), streaming RNNT at att [70,1] = 160 ms chunks "
                 "(masked offline forward == cache-aware streaming, tests/test_streaming.py), greedy, CPU 2 threads",
    "core_115m_beam8": "audioforge 115M core as core_115m, RNNT beam search width 8 (<= 3 tokens per frame; serve "
                       "--beam 8, research/FIXALL.md step 5), CPU 2 threads",
    "core_0p6b": "audioforge 0.6B core (nemotron-speech-streaming-en-0.6b inside served_0p6b_v0.3.afm; base tensors "
                 "unchanged by the heads and identical in v0.4), streaming RNNT at [70,1], greedy, CPU 2 threads",
    "core_115m_f1120": "audioforge 115M core, the 1.12 s final (--final-chunk-ms 1120): masked offline forward at att "
                       "[70,13] (== cache-aware streaming in 1120 ms chunks, research/DUAL_RATE.md), greedy RNNT, CPU 2 threads",
    "core_0p6b_f1120": "audioforge 0.6B core, the 1.12 s final (--final-chunk-ms 1120): masked offline forward at att "
                       "[70,13], greedy RNNT, CPU 2 threads",
}
OPTIONAL_ASR = ("core_115m_f1120", "core_0p6b_f1120", "whisper_small_b5")  # scored only on the sets where they were run


def make_asr(system: str, device: str):
    """-> transcribe(np.ndarray) -> str."""
    import torch
    torch.set_num_threads(2)
    if system in ("whisper_small", "whisper_small_b5"):
        from faster_whisper import WhisperModel
        wm = WhisperModel(snap("Systran/faster-whisper-small"), device="cpu", compute_type="int8", cpu_threads=2,
                          num_workers=1)
        if system == "whisper_small_b5":
            def fn5(x):  # faster-whisper's transcribe() defaults, English (Pipecat's WhisperSTTService call)
                segs, _ = wm.transcribe(np.asarray(x, np.float32), language="en")
                return " ".join(s.text.strip() for s in segs)
            return fn5

        def fn(x):
            segs, _ = wm.transcribe(x, language="en", beam_size=1, without_timestamps=True, vad_filter=False,
                                    condition_on_previous_text=False)
            return " ".join(s.text.strip() for s in segs)
        return fn
    if system in ("whisper_turbo", "whisper_large_v3", "whisper_small_hf_b5"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        repo = {"whisper_turbo": "openai/whisper-large-v3-turbo", "whisper_large_v3": "openai/whisper-large-v3",
                "whisper_small_hf_b5": "openai/whisper-small"}[system]
        nb = 5 if system == "whisper_small_hf_b5" else 1  # Whisper small on MPS: transformers, beam 5
        proc = WhisperProcessor.from_pretrained(snap(repo))
        dt = torch.float16 if device != "cpu" else torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(snap(repo), dtype=dt).to(device).eval()

        def fn(x):
            x = np.asarray(x, np.float32)
            with torch.inference_mode():
                if len(x) <= 30 * SR:
                    f = proc(x, sampling_rate=SR, return_tensors="pt").input_features.to(device, dt)
                    ids = model.generate(f, language="en", task="transcribe", num_beams=nb, do_sample=False)
                else:
                    inp = proc(x, sampling_rate=SR, return_tensors="pt", truncation=False, padding="longest",
                               return_attention_mask=True)
                    ids = model.generate(inp.input_features.to(device, dt), attention_mask=inp.attention_mask.to(device),
                                         language="en", task="transcribe", num_beams=nb, do_sample=False,
                                         return_timestamps=True, condition_on_prev_tokens=False)
            return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return fn
    if system == "tdt_v3":
        import hybrid_asr as H
        m = H.load_tdt(device)
        return lambda x: H.tdt_transcribe(m, np.asarray(x, np.float32))
    from audioforge.train import load_model
    path = AFM_115M if system.startswith("core_115m") else afm_0p6b()
    m = load_model(str(path), "cpu").eval()
    assert list(m.encoder.att_context_size) == [70, 1], m.encoder.att_context_size
    if system.endswith("_f1120"):
        def fns(x):
            with torch.inference_mode():
                return m.transcribe([np.asarray(x, np.float32)], head="rnnt", att_context_size=[70, 13])[0]
        return fns
    if system.endswith("_beam8"):
        def fnb(x):
            with torch.inference_mode():
                xx, ll = m._pad([np.asarray(x, np.float32)])
                enc, elen, hid = m.encode(xx, ll, [70, 1], return_hidden=True)
                e = m.head_input("rnnt", enc, hid)[0, : int(elen[0])]
                return m.tokenizer.decode(m.heads["rnnt"].beam_search(e, beam=8, max_sym=3))
        return fnb

    def fn(x):
        with torch.inference_mode():
            return m.transcribe([np.asarray(x, np.float32)], head="rnnt")[0]
    return fn


def stage_asr(a):
    xs, refs, _ = asr_set(a.set)
    d = WORK / "asr"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{a.system}_{a.set}.jsonl"
    done = jl_rows(p)
    todo = [i for i in range(len(refs)) if i not in done]
    log(f"asr {a.system} {a.set}: {len(todo)} of {len(refs)} to do")
    if not todo:
        log("STAGE_COMPLETE")
        return
    fn = make_asr(a.system, "cpu" if a.system == "tdt_v3" else a.device)  # the words table runs Parakeet-TDT on CPU
    fn(xs[todo[0]][: 3 * SR])  # warm-up, not timed
    with p.open("a") as f:
        n = 0
        for i in todo:
            if over(a):
                break
            t0 = time.perf_counter()
            h = fn(xs[i])
            dt = time.perf_counter() - t0
            f.write(json.dumps({"i": i, "hyp": h, "sec": round(dt, 4), "audio_sec": round(len(xs[i]) / SR, 3)}) + "\n")
            f.flush()
            n += 1
    log(f"asr {a.system} {a.set}: {n} this call, {len(todo) - n} left")


def _boot_rate(e: np.ndarray, groups, n_boot=1000, seed=0):
    """e (N, 2) = [edits, ref words] per item; bootstrap over groups (items, or clips for live)."""
    g = np.asarray(groups)
    ug = sorted(set(g.tolist()), key=str)
    per = np.array([e[g == k].sum(0) for k in ug], np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        s = per[rng.integers(0, len(ug), len(ug))].sum(0)
        draws.append(s[0] / max(s[1], 1))
    return per, np.array(draws)


def score_asr() -> dict:
    """Every system on every set with Whisper's English normalizer (Open ASR Leaderboard; primary) and
    audioforge.teachers.normalize_text (the repo's earlier tables); 1000-resample bootstrap CIs (items; clips for the
    live sessions) and paired deltas against each core."""
    import final_scoring as FS
    import hybrid_asr as H
    norms, _ = H._normalizers()
    norms = {**norms, "whisper_norm_disfl": FS.disfl_norm(norms["whisper_norm"])}
    res = {}
    for set_name in ("libri", "ami", "icsi", "live", "ami_eval", "icsi_eval", "ls_clean", "ls_other", "fleurs_en"):
        if set_name.endswith("_eval") and not (WORK / "asr_sets" / f"{set_name}_refs.json").exists():
            continue
        if (set_name.startswith("ls_") or set_name == "fleurs_en") and not any((WORK / "asr").glob(f"*_{set_name}.jsonl")):
            continue
        hyp = {s: jl_rows(WORK / "asr" / f"{s}_{set_name}.jsonl") for s in ASR_SYSTEMS}
        _, refs, groups = asr_set(set_name)
        n = len(refs)
        full = [s for s in ASR_SYSTEMS if len(hyp[s]) >= n]
        out = {"n": n, "systems_complete": full, "missing": {s: n - len(hyp[s]) for s in ASR_SYSTEMS if s not in full
                                                             and not (s in OPTIONAL_ASR and not hyp[s])},
               "bootstrap": ("1000 resamples of meetings" if set_name.endswith("_eval") and isinstance(groups[0], str)
                             else "1000 resamples of sessions (clips; mono and user channel together)" if set_name == "live"
                             else "1000 resamples of items"),
               "bootstrap_groups": len(set(groups))}
        for nn in ("whisper_norm", "normalize_text") + (("whisper_norm_disfl",) if set_name == "live" else ()):
            fn = norms[nn]
            rr = [fn(r) for r in refs]
            E = {}
            for s in full:
                E[s] = np.array([H.edits_words(rr[i], fn(hyp[s][i]["hyp"])) for i in range(n)], np.float64)
            sub_sets = {"all": np.ones(n, bool)}
            if set_name == "live":
                sub_sets["user_channel"] = np.array([k.endswith("|user") for k in [s["key"] for s in live_sessions()]])
                sub_sets["mono"] = ~sub_sets["user_channel"]
            for sub, msk in sub_sets.items():
                o = {}
                draws = {}
                gg = [g for g, m_ in zip(groups, msk) if m_]
                for s in full:
                    e = E[s][msk]
                    _, dr = _boot_rate(e, gg)
                    draws[s] = dr
                    o[s] = {"wer_pct": round(100 * e[:, 0].sum() / e[:, 1].sum(), 2),
                            "ci95": [round(100 * float(np.percentile(dr, q)), 2) for q in (2.5, 97.5)],
                            "errors": int(e[:, 0].sum()), "ref_words": int(e[:, 1].sum())}
                for base in ("core_0p6b", "core_115m"):
                    if base not in draws:
                        continue
                    for s in full:
                        if s != base:
                            o[f"{s} - {base}"] = {"delta_pp": round(o[s]["wer_pct"] - o[base]["wer_pct"], 2),
                                                  "ci95": [round(100 * float(np.percentile(draws[s] - draws[base], q)), 2)
                                                           for q in (2.5, 97.5)]}
                out[f"{nn}|{sub}"] = o
        rtf = {}
        for s in full:
            sec = sum(r["sec"] for r in hyp[s].values())
            au = sum(r["audio_sec"] for r in hyp[s].values())
            rtf[s] = {"rtfx": round(au / sec, 1), "decode_s": round(sec, 1), "audio_s": round(au, 1)}
        out["rtfx_this_run"] = rtf
        res[set_name] = out
    return res


# =========================================================================== streaming word latency (ours)
def stage_sttlat(a):
    import stt_latency as SL
    import audioforge
    from audioforge.data import load_wav
    d = WORK / "sttlat" / a.sys
    d.mkdir(parents=True, exist_ok=True)
    todo = [(m, st) for m in SL.MEETINGS for st in SL.STARTS if not (d / f"{m}_{int(st)}.json").exists()]
    if not todo:
        log("sttlat: all windows done STAGE_COMPLETE")
        return
    kw = {"core": "0.6b", "asr": str(afm_0p6b())} if a.sys == "0p6b" else {"asr": str(AFM_115M)}
    fe = audioforge.load(threads=2, device=a.device, **kw)
    cache = {}
    for m, st in todo:
        if over(a):
            break
        if m not in cache:
            cache = {m: load_wav(str(SL.AMI / "audio" / f"{m}.Mix-Headset.wav"), SR).astype(np.float32)}
        x = cache[m][int(st * SR):int((st + SL.WIN_S) * SR)]
        ref = SL.ref_words(m, st, st + SL.WIN_S)
        r = SL.stream(fe, x)
        sc = SL.score(ref, r["hyp"])
        rec = {"meeting": m, "start": st, **sc, "block_ms_p50": float(np.median(r["block_ms"])),
               "block_ms_p95": float(np.percentile(r["block_ms"], 95)), "device": a.device}
        (d / f"{m}_{int(st)}.json").write_text(json.dumps(rec))
        L = np.array(sc["latency_s"]) * 1000
        log(f"{a.sys} {m} {st:.0f}: WER {sc['wer']:.3f} hits {sc['hits']} p50 {np.median(L):.0f} ms, block p50 "
            f"{rec['block_ms_p50']:.1f} ms")


def score_sttlat() -> dict:
    out = {}
    for c in ("115m", "0p6b"):
        recs = [json.loads(p.read_text()) for p in sorted((WORK / "sttlat" / c).glob("*.json"))]
        if len(recs) < 12:
            out[c] = {"n_windows": len(recs), "complete": False}
            continue
        per = [np.array(r["latency_s"]) * 1000 for r in recs]
        L = np.concatenate(per)
        rng = np.random.default_rng(0)
        b = np.array([[np.median(s), np.percentile(s, 95)] for s in
                      (np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))]) for _ in range(1000))])
        n_ref = sum(r["n_ref"] for r in recs)
        out[c] = {"n_windows": len(recs), "n_hits": int(len(L)), "p50_ms": round(float(np.median(L))),
                  "p50_ci95": [round(float(x)) for x in np.percentile(b[:, 0], [2.5, 97.5])],
                  "p95_ms": round(float(np.percentile(L, 95))),
                  "p95_ci95": [round(float(x)) for x in np.percentile(b[:, 1], [2.5, 97.5])],
                  "block_ms_p50": round(float(np.median([r["block_ms_p50"] for r in recs])), 1),
                  "wer_all_speech_pct": round(100 * sum(r["sub"] + r["del"] + r["ins"] for r in recs) / n_ref, 2),
                  "device": recs[0].get("device"), "complete": True}
    return out


# =========================================================================== speech detection (VAD)
VAD_SETS = ("ami_dev", "icsi_dev", "ami_eval", "icsi_eval")  # eval: the corpora's eval meetings (FIXALL test audit)
VAD_SYSTEMS = {
    "silero_v5": "Silero VAD v5.1.2 (TorchScript, the model Pipecat and LiveKit ship), 32 ms chunks with its "
                 "recurrent state from the window start; max-pooled onto the 80 ms label grid. Causal",
    "marblenet_v2": "NVIDIA Frame_VAD_Multilingual_MarbleNet_v2.0 (data/nemo .nemo, audioforge.baselines.sd.FrameVAD), "
                    "20 ms frames, max-pooled onto 80 ms. Offline convolution over the whole 20 s window (non-causal)",
    "pyannote_seg3": "pyannote/segmentation-3.0 (pyannote.audio 4.0.7), speech = 1 - P(no speaker) of the powerset "
                     "output, 10 s windows with pyannote's default 1 s step, overlap-averaged, max-pooled onto 80 ms. "
                     "Offline (sees up to 10 s of future audio)",
    "ten_vad": "TEN VAD 1.0.6.8 (pip ten-vad, Agora; Apache-2.0 with conditions), 16 ms hop on int16 PCM, max-pooled "
               "onto 80 ms. Causal",
    "core_115m": "audioforge 115M served speech head (heads.speech of stage1_served_v4.afm), native 80 ms frames, "
                 "att [70,1] (80 ms right context)",
    "core_0p6b": "audioforge 0.6B served speech head (heads.speech of served_0p6b_v0.3.afm, identical in v0.4), native "
                 "80 ms frames, att [70,1]",
}


def vad_sets():
    import vad_layers as V
    return {sn: V.load_set(sn) for sn in VAD_SETS}


def stage_vad(a):
    import torch
    from audioforge.baselines import sd
    torch.set_num_threads(2)
    d = WORK / "vad"
    d.mkdir(parents=True, exist_ok=True)
    if all((d / f"{a.system}_{sn}.npz").exists() for sn in VAD_SETS):
        log(f"vad {a.system}: done STAGE_COMPLETE")
        return
    import vad_layers as V
    sets = {sn: V.load_set(sn) for sn in VAD_SETS if not (d / f"{a.system}_{sn}.npz").exists()}
    if a.system == "silero_v5":
        m = torch.jit.load(str(SSD / "scratch/metrics/vad/silero_vad_v5.1.2.jit")).eval()

        def fn(x, T):
            return sd.pool_probs(*sd.silero_probs(m, x), T)
    elif a.system == "marblenet_v2":
        mb = sd.load_nemo_conv(str(ROOT / "data/nemo/frame_vad_multilingual_marblenet_v2.0.nemo"))

        def fn(x, T):
            return sd.pool_probs(mb.probs(x), mb.hop, T, offset=-mb.hop / 2)
    elif a.system == "pyannote_seg3":
        os.environ["HF_HUB_OFFLINE"] = "1"
        from pyannote.audio import Inference, Model
        pm = Model.from_pretrained("pyannote/segmentation-3.0")
        inf = Inference(pm, skip_conversion=True, device=torch.device(a.device),
                        pre_aggregation_hook=lambda s: 1.0 - np.exp(s[..., :1]))

        def fn(x, T):
            swf = inf({"waveform": torch.from_numpy(np.asarray(x, np.float32))[None], "sample_rate": SR})
            sw = swf.sliding_window
            return sd.pool_probs(np.asarray(swf.data)[:, 0], sw.step, T, offset=sw.start)
    elif a.system == "ten_vad":
        from ten_vad import TenVad

        def fn(x, T):
            v = TenVad(256, 0.5)
            pcm = np.clip(np.asarray(x) * 32768.0, -32768, 32767).astype(np.int16)
            p = [v.process(pcm[i:i + 256])[0] for i in range(0, len(pcm) - 255, 256)]
            return sd.pool_probs(np.asarray(p), 256 / SR, T)
    else:
        from audioforge.train import load_model
        path = AFM_115M if a.system == "core_115m" else afm_0p6b()
        m = load_model(str(path), a.device).eval()

        def fn(x, T):
            with torch.inference_mode():
                xx = torch.from_numpy(np.asarray(x, np.float32))[None].to(a.device)
                enc, elen, hid = m.encode(xx, torch.tensor([xx.shape[1]], device=a.device), [70, 1], return_hidden=True)
                vh = "speech" if "speech" in m.heads else "vad"  # the client's speech probability (FIXALL.md step 1)
                p = m.heads[vh](m.head_input(vh, enc, hid)).sigmoid()[0, : int(elen[0])].float().cpu().numpy()
            p = p.reshape(-1)
            return np.concatenate([p, np.zeros(max(0, T - len(p)))])[:T]
    for sn, val in sets.items():
        f = d / f"{a.system}_{sn}.npz"
        if f.exists():
            continue
        P, L = [], []
        t0 = time.perf_counter()
        for v in val:
            y = np.asarray(v["vad"], np.float32)
            P.append(np.asarray(fn(np.asarray(v["audio"], np.float32), len(y)), np.float32))
            L.append(y)
        np.savez(f, p=np.concatenate(P), y=np.concatenate(L), lens=np.array([len(y) for y in L]),
                 sec=time.perf_counter() - t0)
        log(f"vad {a.system} {sn}: {len(val)} windows, {sum(len(y) for y in L)} frames, {time.perf_counter() - t0:.1f} s")


def score_vad() -> dict:
    """final_scoring.score_vad: frames inside the audio only (every system), meeting bootstrap."""
    import final_scoring as FS
    return FS.score_vad(WORK, VAD_SYSTEMS, VAD_SETS)


# =========================================================================== language ID
LID_SYSTEMS = {
    "whisper_small": "openai/whisper-small", "whisper_turbo": "openai/whisper-large-v3-turbo",
    "whisper_large_v3": "openai/whisper-large-v3",
}
LID_COND = ("2s", "full")


def stage_lid(a):
    import torch
    import lid as L
    from audioforge.baselines.lid import restrict
    torch.set_num_threads(2)
    d = WORK / "lid"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{a.system}.npz"
    part = d / f"{a.system}.part.npz"
    if f.exists():
        log(f"lid {a.system}: done STAGE_COMPLETE")
        return
    rows = L.rows_for("fleurs")
    ons = L.load_onsets()
    P = dict(np.load(part)) if part.exists() else {"P": np.zeros((len(rows), len(LID_COND), len(L.CODES)), np.float32),
                                                    "n": np.array(0), "sec": np.array(0.0)}
    start = int(P["n"])
    dev = a.device
    if a.system in LID_SYSTEMS:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        proc = WhisperProcessor.from_pretrained(snap(LID_SYSTEMS[a.system]))
        dt = torch.float16 if dev != "cpu" else torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(snap(LID_SYSTEMS[a.system]), dtype=dt).to(dev).eval()
        tok = proc.tokenizer
        sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        codes = list(LANGUAGES)
        ids = [tok.convert_tokens_to_ids(f"<|{c}|>") for c in codes]

        def probs(clips):
            feats = proc.feature_extractor([np.asarray(c, np.float32) for c in clips], sampling_rate=SR,
                                           return_tensors="pt").input_features.to(dev, dt)
            with torch.inference_mode():
                lg = model(input_features=feats, decoder_input_ids=torch.full((len(clips), 1), sot, device=dev)).logits[:, -1]
                pr = lg[:, ids].float().softmax(-1).cpu().numpy()
            return [restrict(dict(zip(codes, row.tolist())), L.CODES) for row in pr]
        bs = 16
    else:  # a core's LID head through the real clip path (lid.py eval_head, VAD-gated as served)
        from audioforge.lid import attach_head
        from audioforge.train import load_model
        if a.system == "core_115m":  # FIXALL: the 115M through this path too (FINAL_LID_HEAD: a candidate head file)
            model = load_model(str(AFM_115M), dev).eval()
            name = attach_head(model, Path(os.environ.get("FINAL_LID_HEAD", LID_HEAD["115m"])))
        else:
            model = load_model(str(afm_0p6b()), dev).eval()
            name = attach_head(model, Path(os.environ.get("FINAL_LID_HEAD", LID_HEAD["0p6b"])))
        head = model.heads[name]

        def probs(clips):
            lens = torch.tensor([len(c) for c in clips])
            x = torch.zeros(len(clips), int(lens.max()))
            for i, c in enumerate(clips):
                x[i, : len(c)] = torch.from_numpy(np.asarray(c, np.float32))
            with torch.inference_mode():
                enc, elen, hid = model.encode(x.to(dev), lens.to(dev), [70, 1], return_hidden=True)
                e = model.head_input(name, enc, hid)
                vad = model.heads["vad"](model.head_input("vad", enc, hid)).sigmoid()
                if vad.dim() == 2:
                    vad = vad[..., None]
                w, wx, wxx = head._terms(e)
                valid = (torch.arange(e.shape[1], device=dev)[None] < elen[:, None]).float()[..., None]
                m = valid * (vad > 0.5).float()
                m = torch.where(m.sum(1, keepdim=True) > 0, m, valid)
                g = head._classify((w * m).sum(1), (wx * m).sum(1), (wxx * m).sum(1)).softmax(-1).float().cpu().numpy()
            return list(g)
        bs = 8
    t0 = time.time()
    order = list(range(len(rows)))
    i = start
    sec = float(P["sec"])
    while i < len(rows) and not over(a):
        idx = order[i:i + bs]
        aud = {j: L.load_audio(rows[j]) for j in idx}
        for ci, cond in enumerate(LID_COND):
            clips = [L.clip(aud[j], ons[rows[j]["id"]][0], cond) for j in idx]
            t1 = time.perf_counter()
            out = probs(clips)
            sec += time.perf_counter() - t1
            for j, pr in zip(idx, out):
                P["P"][j, ci] = pr
        i += len(idx)
    if i < len(rows):
        np.savez(part, P=P["P"], n=np.array(i), sec=np.array(sec))
        log(f"lid {a.system}: {i}/{len(rows)} ({time.time() - t0:.0f} s); rerun")
        return
    y = np.array([L.CODES.index(r["lang"]) for r in rows])
    np.savez(f, P=P["P"], y=y, ids=np.array([r["id"] for r in rows]), sec=np.array(sec))
    part.unlink(missing_ok=True)
    log(f"lid {a.system}: done; acc 2s {np.mean(P['P'][:, 0].argmax(-1) == y):.4f} full {np.mean(P['P'][:, 1].argmax(-1) == y):.4f}")


def score_lid() -> dict:
    import lid as L
    rows = L.rows_for("fleurs")
    y = np.array([L.CODES.index(r["lang"]) for r in rows])
    ids = [r["id"] for r in rows]
    src = {}
    for s in list(LID_SYSTEMS) + ["core_0p6b", "core_115m"]:
        f = WORK / "lid" / f"{s}.npz"
        if f.exists():
            z = np.load(f)
            assert list(z["ids"]) == ids
            src[s] = (z["P"][:, 0], z["P"][:, 1], f"{f}")
    # reused, identical clips and clip path (lid.py: same rows, onsets, PRE_ROLL, peak normalisation)
    for s, tag in (("core_115m", "lid_distill_vadgated"), ("ambernet", "ambernet")):
        if s in src:  # measured in this pass (FIXALL: the 115M through stage_lid)
            continue
        r = L._load_preds(tag)
        if r is not None:
            P, yy, _ = r
            assert np.array_equal(yy, y)
            src[s] = (P[:, L.COND.index("2s")], P[:, L.COND.index("full")], f"data/lid/preds/{tag}")
    out = {"n": len(y), "languages": L.CODES}
    rng = np.random.default_rng(0)
    bi = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    for s, (p2, pf, where) in src.items():
        o = {"source": where}
        for nm, P in (("2s", p2), ("full", pf)):
            c = (P.argmax(-1) == y).astype(float)
            o[f"acc_{nm}_pct"] = round(100 * c.mean(), 2)
            o[f"acc_{nm}_ci95"] = [round(100 * float(np.percentile([c[k].mean() for k in bi], q)), 2) for q in (2.5, 97.5)]
        out[s] = o
    return out


# =========================================================================== end of turn
EOU_NEMO = ROOT / "data" / "nemo" / "parakeet_realtime_eou_120m-v1.nemo"


def stage_eou(a):
    """Parakeet-Realtime-EOU (NVIDIA, 120M, cache-aware streaming FastConformer-RNNT, att [70,1]) through
    model.StreamingSession in 160 ms blocks on --device: a turn end at the end of every block whose decode emits <EOU>,
    then a fresh streaming state (the utterance is closed); decision time = the audio fed so far, total = + that
    block's measured compute."""
    import torch
    import eot_assistant as EA
    import eot_latency as E
    from audioforge.model import StreamingSession
    from audioforge.nemo_import import import_eou
    torch.set_num_threads(2)
    d = WORK / "eou" / a.which
    d.mkdir(parents=True, exist_ok=True)
    items = ([(s["key"], (lambda s=s: E.read_audio(s))) for s in E.sessions()] if a.which == "calls"
             else [(c["key"], (lambda c=c: EA.audio(c))) for c in EA.clips()])
    todo = [(k, f) for k, f in items if not (d / f"{k}.json").exists()]
    log(f"eou {a.which}: {len(todo)} of {len(items)} to do")
    if not todo:
        log("STAGE_COMPLETE")
        return
    m = import_eou(EOU_NEMO).eval().to(a.device)
    eou = m.eou_ids["<EOU>"]
    att = list(m.encoder.att_context_size)
    with torch.inference_mode():
        s = StreamingSession(m, "rnnt", att)
        s.feed(np.zeros(SR, np.float32))
    n = 0
    for k, fx in todo:
        if over(a):
            break
        x = fx()
        s = StreamingSession(m, "rnnt", att)
        ends, ms, ntok, texts = [], [], 0, []
        with torch.inference_mode():
            for i in range(0, len(x), CHUNK):
                t0 = time.perf_counter()
                s.feed(x[i:i + CHUNK], final=i + CHUNK >= len(x))
                if a.device == "mps":
                    torch.mps.synchronize()
                dt = time.perf_counter() - t0
                ms.append(dt * 1000)
                new = list(s.tokens[ntok:])
                ntok = len(s.tokens)
                if eou in new:
                    ends.append({"t_dec": round(min(i + CHUNK, len(x)) / SR, 4), "compute_s": round(dt, 5)})
                    # the model is trained on single utterances: after one <EOU> its decoder never emits another
                    # (checked on the calls), so the utterance is closed and a fresh streaming state starts
                    texts.append(m.tokenizer.decode([t for t in s.tokens if t != eou]))
                    s = StreamingSession(m, "rnnt", att)
                    ntok = 0
        (d / f"{k}.json").write_text(json.dumps({"key": k, "ends": ends, "att": att, "device": a.device,
                                                 "chunk_ms_p50": round(float(np.median(ms)), 2),
                                                 "reset_after_eou": True,
                                                 "text": " | ".join(texts + [m.tokenizer.decode([t for t in s.tokens if t != eou])])}))
        n += 1
    log(f"eou {a.which}: {n} this call, {len(todo) - n} left")


def eot_dir(sys_: str, which: str, noprint: bool = False) -> Path:
    """Where a core's served-engine turn dumps live: EOT_DIR[sys]/<sys>_<which>; no-print runs: WORK/eot/..._noprint."""
    return (WORK / "eot" if noprint else EOT_DIR[sys_]) / (f"{sys_}_{which}" + ("_noprint" if noprint else ""))


def stage_eotdump(a):
    """core_0p6b_turn.stage_evdump (the served single-mode session, every per-frame signal a preset reads, its own
    chunk compute) for either core on --device -> eot_dir(sys, which). ``--noprint``: the AMI test sessions only, with
    no voice print armed (no TS-VAD "others" path), to measure what knowing the user's voice is worth."""
    import torch
    import audioforge.serve as S
    import core_0p6b_heads as C
    import eot_assistant as EA
    import eot_latency as E
    from audioforge.server.cli import MODES
    torch.set_num_threads(2)
    od = eot_dir(a.sys, a.which, a.noprint)
    od.mkdir(parents=True, exist_ok=True)
    items = E.sessions() if a.which == "calls" else EA.clips()
    if a.noprint:
        assert a.which == "calls" and E.AMI_SPLIT == "eval"
        items = [x for x in items if x["set"] == "ami"]
    todo = [x for x in items if not (od / f"{x['key']}.json").exists()]
    log(f"eotdump {a.sys} {a.which}: {len(todo)} of {len(items)} to do")
    if not todo:
        log("STAGE_COMPLETE")
        return
    if a.sys == "0p6b":
        prints = json.loads(C.PRINTS.read_text())  # the 5 s prints re-made for the 0.6B's speaker head
        afm, ts, lid = afm_0p6b(), ROOT / "assets" / "tsvad_0p6b.pt", LID_HEAD["0p6b"]
    else:
        prints = None
        afm, ts, lid = AFM_115M, ROOT / "assets" / "tsvad_spk.pt", LID_HEAD["115m"]
    opts = {**MODES["single"], "enroll": "explicit", "tsvad": str(ts), "lid": str(lid) if lid.exists() else None,
            "silero": str(ROOT / "data" / "silero" / "silero_vad_v5.onnx"), "preload_silero": True}
    eng = S.Engine.load(str(afm), None, a.device, threads=2, **opts)
    eng.warmup()
    n = 0
    for it in todo:
        if over(a):
            break
        k = it["key"]
        if a.which == "calls":
            x = E.read_audio(it)
            s = S.Session(eng, S.SessionConfig(turn_policy="hybrid_dyn", timeout_ms=1000))
            emb = (prints.get(k) if prints is not None else it["embedding"]) if it["embedding"] is not None else None
            if emb is not None and not a.noprint:
                s.arm_enrollment("enroll", 0, embedding=emb)
            blk = SR * 160 // 1000
        else:
            x = EA.audio(it)
            s = S.Session(eng, S.SessionConfig(turn_policy="vad_head"))
            blk = 320
        if s.asr.seg is None and eng.seg_name is not None:
            sh = eng.asr.heads[eng.seg_name]
            s.asr.attach_seg(sh, next(sh.parameters()).device)
        if "turn_seg_a" in s.asr.m.heads and s.asr.seg2 is None:  # research/TURN_DATA.md: the second classifier
            s.asr.attach_seg2("turn_seg_a", next(s.asr.m.heads["turn_seg_a"].parameters()).device)
        rec = []
        C._record(s, rec, enrolled_only=a.which == "asst" or a.noprint)
        msgs = []
        for i in range(0, len(x), blk):
            msgs += s.process(x[i:i + blk])
        msgs += s.finish()
        cm = np.asarray(list(s.chunk_ms), float)
        dd = {"key": k, "audio_s": round(len(x) / SR, 3),
              "head": {kk: [r[j] for r in rec] for j, kk in enumerate(("v", "t", "p", "vad", "pu", "po", "p5", "vad_m", "p5b"))},
              "tok_at": [int(q) for q in s.asr.tok_at],
              "chunk_ms": {"p50": round(float(np.median(cm)), 2), "p95": round(float(np.percentile(cm, 95)), 2),
                           "mean": round(float(cm.mean()), 2), "n": int(len(cm))}, "device": a.device,
              "afm": str(afm),
              "turn_ends": [{kk: m_.get(kk) for kk in ("t", "policy", "p", "path")} for m_ in msgs
                            if m_["type"] == "turn_end"]}
        if a.which == "calls":
            dd.update({"set": it["set"], "cond": it["cond"], "has_print": it["embedding"] is not None and not a.noprint})
        else:
            dd.update({"complete": it["complete"], "clip_s": round((it["b"] - it["a"]) / SR, 4)})
        (od / f"{k}.json").write_text(json.dumps(dd))
        n += 1
    log(f"eotdump {a.sys} {a.which}: {n} this call, {len(todo) - n} left")


def _turn_boot(per_calls, per_ami, asess, times, comps, boot=1000, seed=0):
    """Point values and session-bootstrap CIs, the core_0p6b_turn.ev_score protocol, for any system's turn-end times."""
    import eot_assistant as EA
    import eot_latency as E

    def asst(ss):
        sc = EA.score_system(ss, times, comps)
        return {"accuracy_pct": sc["accuracy_pct"], "p50": sc["complete"]["eot_total_ms_p50"],
                "p95": sc["complete"]["eot_total_ms_p95"], "false_fire_pct": sc["incomplete"]["false_fire_pct"],
                "missed_pct": sc["complete"]["missed_pct"], "early_pct": sc["complete"]["early_fire_pct"]}
    res = {}
    if per_calls is not None:
        res["calls"] = E.pool(per_calls)
        res["ami"] = E.pool(per_ami)
    if asess is not None:
        res["asst"] = asst(asess)
    rng = np.random.default_rng(seed)
    bc = {"calls": [], "ami": [], "asst": []}
    for _ in range(boot):
        if per_calls is not None:
            for sc, P in (("calls", per_calls), ("ami", per_ami)):
                bc[sc].append(E.pool([P[i] for i in rng.integers(0, len(P), len(P))]))
        if asess is not None:
            bc["asst"].append(asst([asess[i] for i in rng.integers(0, len(asess), len(asess))]))

    def ci(rows, q):
        x = np.array([r_[q] for r_ in rows if r_[q] is not None], float)
        return [round(float(np.percentile(x, 2.5)), 1), round(float(np.percentile(x, 97.5)), 1)] if len(x) else None
    for sc in ("calls", "ami"):
        if sc in res:
            res[sc]["ci95"] = {q: ci(bc[sc], q) for q in ("eot_total_ms_p50", "eot_total_ms_p95",
                                                          "false_interruption_pct", "missed_pct")}
    if "asst" in res:
        res["asst"]["ci95"] = {q: ci(bc["asst"], q) for q in ("accuracy_pct", "p50", "p95", "false_fire_pct",
                                                              "missed_pct")}
    return res


def _score_times(tcalls, ccalls, tasst, casst, boot=1000):
    """tcalls[key] = turn-end totals (s) on the calls / AMI sessions, ccalls[key] = compute per time (s); the same for
    the assistant clips. Missing sides are skipped."""
    import eot_assistant as EA
    import eot_latency as E
    pc = pa = asess = None
    if tcalls is not None:
        pc, pa = [], []
        for ss in E.sessions():
            sc = E.score_session(tcalls[ss["key"]], ss, ccalls[ss["key"]])
            (pc if ss["set"] in E.SCOPES["two_party_user"] else pa).append(sc)
    if tasst is not None:
        asess = EA.sessions(EA.load_dump())
    return _turn_boot(pc, pa, asess, tasst or {}, casst or {}, boot)


def score_turn(boot=1000) -> dict:
    """Every system on the identical sessions; assistant clips also at false-fire windows 2.0 / 2.5 / 3.0 / 3.5 s
    (final_scoring.asst_curve) with the headline window chosen by final_scoring.headline_window; AMI test also with
    our cores un-enrolled (no voice print, eotdump --noprint)."""
    import core_0p6b_turn as CT
    import eot_assistant as EA
    import eot_latency as E
    import final_scoring as FS
    assert E.AMI_SPLIT == "eval", "the AMI turn rows are scored on the AMI test meetings"
    out = {}
    asess = EA.sessions(EA.load_dump())
    # ---- baselines on the identical sessions: the stored per-session decisions of eot_latency.cmd_baselines
    # (baselines4) and eot_assistant.cmd_baselines, rescored here with CIs
    bl_c = {d["key"]: d for d in (json.loads(p.read_text()) for p in sorted(E.BASE.glob("*.json")))}
    bl_a = EA.load_baselines()
    sess_c = E.sessions()
    clips = EA.clips()
    for bk, name in (("pipecat_smartturn", "pipecat_smartturn_v3.2_silero"), ("livekit_eou", "livekit_en_turn_detector_silero")):
        if all(s["key"] in bl_c for s in sess_c) and all(c["key"] in bl_a for c in clips):
            def tc(bl, k):
                ev = bl[k][bk]
                cc = [e["compute_ms"] / 1000 if bk == "pipecat_smartturn" else 0.0 for e in ev]
                return [e["t"] + c for e, c in zip(ev, cc)], cc
            T1 = {s["key"]: tc(bl_c, s["key"]) for s in sess_c}
            T2 = {c["key"]: tc(bl_a, c["key"]) for c in clips}
            out[name] = _score_times({k: v[0] for k, v in T1.items()}, {k: v[1] for k, v in T1.items()},
                                     {k: v[0] for k, v in T2.items()}, {k: v[1] for k, v in T2.items()}, boot)
            out[name]["asst_windows"] = FS.asst_curve(asess, {k: v[0] for k, v in T2.items()},
                                                      {k: v[1] for k, v in T2.items()}, boot)
            out[name]["source"] = f"{E.BASE} + {EA.BASE} (per-session decisions, rescored with CIs)"
    # ---- Parakeet-Realtime-EOU
    ec = {p.stem: json.loads(p.read_text()) for p in (WORK / "eou" / "calls").glob("*.json")}
    ea = {p.stem: json.loads(p.read_text()) for p in (WORK / "eou" / "asst").glob("*.json")}
    if all(s["key"] in ec for s in sess_c) and all(c["key"] in ea for c in clips):
        ec = {s["key"]: ec[s["key"]] for s in sess_c}
        ea = {c["key"]: ea[c["key"]] for c in clips}
        f = lambda D: ({k: [e["t_dec"] + e["compute_s"] for e in d["ends"]] for k, d in D.items()},  # noqa: E731
                       {k: [e["compute_s"] for e in d["ends"]] for k, d in D.items()})
        t1, c1 = f(ec)
        t2, c2 = f(ea)
        out["parakeet_realtime_eou"] = _score_times(t1, c1, t2, c2, boot)
        out["parakeet_realtime_eou"]["asst_windows"] = FS.asst_curve(asess, t2, c2, boot)
        out["parakeet_realtime_eou"]["chunk_ms_p50"] = round(float(np.median([d["chunk_ms_p50"] for d in ec.values()])), 1)
    else:
        out["parakeet_realtime_eou"] = {"complete": False, "calls": len(ec), "asst": len(ea)}
    # ---- the smart-turn v3.2 classifier alone (upper bound: one call on each whole clip, no timing)
    au = json.loads((ROOT / "runs" / "smartturn_audit.json").read_text())["eval_human_5_all"]["eval"]
    k = int(round(au["accuracy"] / 100 * au["n"]))
    rng = np.random.default_rng(0)
    c = np.r_[np.ones(k), np.zeros(au["n"] - k)]
    out["smartturn_classifier_alone"] = {"asst": {"accuracy_pct": au["accuracy"],
                                                  "ci95": {"accuracy_pct": [round(100 * float(np.percentile(
                                                      [c[rng.integers(0, len(c), len(c))].mean() for _ in range(1000)], q)), 1)
                                                      for q in (2.5, 97.5)]},
                                                  "false_fire_pct": round(100 - au["recall_incomplete"], 1),
                                                  "missed_pct": round(100 - au["recall_complete"], 1)},
                                         "source": "runs/smartturn_audit.json eval_human_5_all > eval (reused: same 399 clips, "
                                                   "same model file)"}
    # ---- ours: the served engines' dumps, each preset as the served model resolves it (cfg turn_presets merged)
    sd = EA.load_dump()
    fallbacks = {}
    for sysn, afm in (("115m", AFM_115M), ("0p6b", afm_0p6b())):
        dc, da = eot_dir(sysn, "calls"), eot_dir(sysn, "asst")
        D = {x["key"]: x for x in (json.loads(p_.read_text()) for p_ in sorted(dc.glob("*.json")))}
        A = {x["key"]: x for x in (json.loads(p_.read_text()) for p_ in sorted(da.glob("*.json")))}
        if not all(s["key"] in D for s in sess_c) or not all(c["key"] in A for c in clips):
            out[f"ours_{sysn}"] = {"complete": False, "calls": len(D), "asst": len(A)}
            continue
        D = {s["key"]: D[s["key"]] for s in sess_c}
        for kk, d in A.items():
            d["conf"] = sd[kk]["conf"]
        rules = CT.served_rules(afm)
        afms = sorted({d["afm"] for d in list(D.values()) + list(A.values())})
        o = {"afm": str(afm), "dump_dirs": [str(dc), str(da)], "dump_afms": afms, "rules": rules,
             "chunk_ms_p50_median": round(float(np.median([d["chunk_ms"]["p50"] for d in D.values()])), 2),
             "device": next(iter(D.values()))["device"]}
        Dn = {x["key"]: x for x in (json.loads(p_.read_text()) for p_ in sorted(eot_dir(sysn, "calls", True).glob("*.json")))}
        n_ami = sum(s["set"] == "ami" for s in sess_c)
        for pn, r in rules.items():
            o[pn] = CT.ev_score(D, A, r, boot=boot)
            tt, cc = FS.ours_asst_times(A, r, asess)
            o[pn]["asst_windows"] = FS.asst_curve(asess, tt, cc, boot)
            assert o[pn]["asst_windows"]["3"]["accuracy_pct"] == o[pn]["asst"]["accuracy_pct"], "window replay drifted"
            if len(Dn) >= n_ami:  # AMI test, no voice print armed (no "others" path): the value of knowing the voice
                o[pn]["ami_noprint"] = FS.ours_ami_score(Dn, r, boot=200)
            fallbacks[f"audioforge {sysn} {pn} (fallback {r['fb']} frames)"] = round(r["fb"] * 0.08, 2)
        o["ami_noprint_dumps"] = {"dir": str(eot_dir(sysn, "calls", True)), "n": len(Dn), "of": n_ami}
        out[f"ours_{sysn}"] = o
    out["asst_headline_window"] = FS.headline_window(fallbacks)
    return out


# =========================================================================== speaker: EER
def spk_segments(c, n=200):
    """spk_head.dev_segments (AMI / ICSI dev, n = 200), or with c = ami_eval / icsi_eval the same recipe path on the
    corpora's eval meetings (FIXALL test audit)."""
    import spk_head as SH
    if not c.endswith("_eval"):
        return SH.dev_segments(c, n)
    corpus = c.split("_")[0]
    if corpus == "ami":
        from audioforge.datasets.ami import recipe_data
    else:
        from audioforge.datasets.icsi import recipe_data
    return recipe_data({"data": {corpus: {"mode": "asr", "val_split": "eval", "n_val": n, "seed": 0}}}, "val")


def stage_spkeer(a):
    """Within-meeting EER on spk_head.dev_segments(corpus, 200) (AMI dev / ICSI dev) for one system -> WORK/spk/."""
    import torch
    import spk_head as SH
    torch.set_num_threads(2)
    d = WORK / "spk"
    d.mkdir(parents=True, exist_ok=True)
    if all((d / f"{a.system}_{c}.npy").exists() for c in ("ami", "icsi", "ami_eval", "icsi_eval")):
        log("spkeer: done STAGE_COMPLETE")
        return
    if a.system == "titanet_l":
        from audioforge.nemo_import import import_titanet
        tn = import_titanet(str(ROOT / "data/nemo/speakerverification_en_titanet_large.nemo")).eval()

        def emb(val):
            with torch.inference_mode():
                return np.stack([tn.embed([np.asarray(v["audio"], np.float32)])[0].float().numpy() for v in val])
    elif a.system == "wespeaker_pyannote":
        os.environ["HF_HUB_OFFLINE"] = "1"
        from pyannote.audio import Inference, Model
        pm = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        inf = Inference(pm, window="whole", device=torch.device(a.device))

        def emb(val):
            return np.stack([np.asarray(inf({"waveform": torch.from_numpy(np.asarray(v["audio"], np.float32))[None],
                                             "sample_rate": SR})).reshape(-1) for v in val])
    else:
        from audioforge.train import load_model
        m = load_model(str(AFM_115M if a.system == "core_115m" else afm_0p6b()), "cpu").eval()

        def emb(val):
            return SH.head_embeddings(m, val).float().numpy()
    for c in ("ami", "icsi", "ami_eval", "icsi_eval"):
        f = d / f"{a.system}_{c}.npy"
        if f.exists():
            continue
        val = spk_segments(c)
        np.save(f, emb(val))
        log(f"spkeer {a.system} {c}: {len(val)} segments")


def score_spkeer() -> dict:
    """Within-meeting EER; CI = 1000 resamples of meetings (each draw pools the within-meeting pairs of the drawn
    meetings; final_scoring.eer_meeting_boot)."""
    import final_scoring as FS
    out = {}
    for c in ("ami", "icsi", "ami_eval", "icsi_eval"):
        if c.endswith("_eval") and not any((WORK / "spk").glob(f"*_{c}.npy")):
            continue
        val = spk_segments(c)
        spk = np.array([v["speaker"] for v in val])
        meet = np.array([v["meeting"] for v in val])
        o = {"n_segments": len(val), "meetings": sorted(set(meet.tolist())), "bootstrap": "1000 resamples of meetings"}
        for s in ("core_115m", "core_0p6b", "titanet_l", "wespeaker_pyannote"):
            f = WORK / "spk" / f"{s}_{c}.npy"
            if not f.exists():
                continue
            pairs = FS.eer_pairs(np.load(f), spk, meet)
            pt, b = FS.eer_meeting_boot(pairs)
            o[s] = {"eer_within_meeting_pct": round(100 * pt, 2), "ci95": FS.ci95(b, 2, 100),
                    "n_pairs": int(sum(len(v[1]) for v in pairs.values()))}
        out[c] = o
    return out


# =========================================================================== speaker: pyannote 3.1 + print
def _tv():
    import tsvad as TV
    return TV


def pya_dirs(corpus):
    return WORK / "pya" / corpus / "tracks", WORK / "pya" / corpus


def stage_pyatracks(a):
    """pyannote/speaker-diarization-3.1 (pyannote.audio 4.0.7, offline on the whole window, its default
    hyper-parameters) on every tswer / A.1 extended window -> (T, 8) 0/1 speaker activity on the 80 ms grid (columns =
    pyannote speakers by first appearance), the Nemotron-3 track format (<key>.stream_rc.npy)."""
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    from pyannote.audio import Pipeline
    TV = _tv()
    torch.set_num_threads(2)
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipe.to(torch.device(a.device))
    for corpus in a.corpora.split(","):
        ext, meta, ds = TV.bench_windows(corpus)
        d = pya_dirs(corpus)[0]
        d.mkdir(parents=True, exist_ok=True)
        todo = [v for v in ext if not (d / f"{TV.wkey(v)}.stream_rc.npy").exists()]
        n = 0
        for v in todo:
            if over(a):
                break
            T = len(v["spk_act"])
            x = torch.from_numpy(np.asarray(v["audio"], np.float32))[None]
            out = pipe({"waveform": x, "sample_rate": SR})
            ann = getattr(out, "speaker_diarization", out)
            order, P = [], np.zeros((T, 8), np.float32)
            for seg, _, lab in ann.itertracks(yield_label=True):
                if lab not in order:
                    order.append(lab)
                c = order.index(lab)
                if c >= 8:
                    continue
                P[int(np.floor(seg.start / 0.08)): int(np.ceil(seg.end / 0.08)), c] = 1.0
            np.save(d / f"{TV.wkey(v)}.stream_rc.npy", P)
            n += 1
        log(f"pyatracks {corpus}: {n} done, {len(todo) - n} left")
        if len(todo) - n:
            return
    log("STAGE_COMPLETE")


def stage_pyatn(a):
    """TitaNet-L look-back column embeddings on the pyannote tracks (eval_stage1.v2_embed_titanet, the same binder
    backend the Nemotron-3 column used)."""
    import eval_stage1 as E
    from audioforge.enrollment import TitaNetEmbedder
    TV = _tv()
    TV.set_threads(2)
    tn = TitaNetEmbedder()
    for corpus in a.corpora.split(","):
        ext, meta, ds = TV.bench_windows(corpus)
        tdir, twork = pya_dirs(corpus)
        r = E.v2_embed_titanet(tn, ext, twork, a.budget - (time.time() - T_START), tracks_dir=tdir)
        log(f"pyatn {corpus}: {r}")
        if r["todo"]:
            return
    log("STAGE_COMPLETE")


def stage_pyaframe(a):
    """tsvad.stage_framearm for the pyannote arm (counts in WORK/pya/<corpus>/framecounts/)."""
    TV = _tv()
    TV.DIAR_ARMS["pyannote31"] = pya_dirs
    TV.framecount_path = lambda corpus, arm, ex: WORK / "pya" / corpus / "framecounts" / f"{TV.wkey(ex)}.npz"
    a.arm = "pyannote31"
    for c in a.corpora.split(","):
        (WORK / "pya" / c / "framecounts").mkdir(parents=True, exist_ok=True)
    TV.stage_framearm(a)
    if all((WORK / "pya" / c / "framecounts" / f"{TV.wkey(v)}.npz").exists()
           for c in a.corpora.split(",") for v in TV.bench_windows(c)[0]):
        log("STAGE_COMPLETE")


def score_pyaframe() -> dict:
    """Pool the pyannote arm's frame counts as tsvad.stage_framearmreport pools the Nemotron-3 arm's: target F1 and
    target-speaker DER (miss + FA over target frames), primary group, all frames."""
    TV = _tv()
    out = {}
    systems = [s.format(arm="pyannote31") for s in TV.FRAME_ARM_SYSTEMS]
    for corpus in ("ami", "icsi"):
        ext, meta, ds = TV.bench_windows(corpus)
        fs = [WORK / "pya" / corpus / "framecounts" / f"{TV.wkey(v)}.npz" for v in ext]
        if not all(f.exists() for f in fs):
            out[corpus] = {"complete": False, "n_done": sum(f.exists() for f in fs), "n": len(fs)}
            continue
        cnt = sum(np.load(f)["counts"] for f in fs)
        o = {}
        for i, s in enumerate(systems):
            tp, fp, fn, neg = (float(x) for x in cnt[i, 0, 0])
            o[s] = {"f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4), "der_pct": round(100 * (fn + fp) / max(tp + fn, 1), 2),
                    "target_frames": int(tp + fn)}
        out[corpus] = o
    return out


def stage_pyatwer(a):
    """Target-speaker WER of the pyannote arm (tswer.stage_score's keep rule, words, lag and references; the pyannote
    column bound by the same 5 s print with tsvad.vp_follow through the served speaker head / TitaNet-L) -> per-unit
    S/D/I/N in WORK/pya/<corpus>/units.jsonl."""
    import eval_stage1 as E
    import tswer as TW
    TV = _tv()
    TV.set_threads(2)
    model = TV.load_served()
    for corpus in a.corpora.split(","):
        ext, meta, ds = TV.bench_windows(corpus)
        asr = TW.load_asr(corpus, "served")
        asr6 = TW.load_asr(corpus, "0p6b")
        L = json.loads((TW.WORK / corpus / "lag.json").read_text())["lag_frames"]
        L6 = json.loads((TW.WORK / corpus / "lag.json").read_text()).get("lag_0p6b_frames")
        kidx, VP = TV.load_vprints(corpus)
        tdir, twork = pya_dirs(corpus)
        n3dir, n3work = TV.DIAR_ARMS["nemotron3"](corpus)
        up = WORK / "pya" / corpus / "units.jsonl"
        done = {json.loads(x)["key"] for x in up.read_text().splitlines() if x.strip()} if up.exists() else set()
        todo = [v for v in ext if TV.wkey(v) not in done]
        n = 0
        with up.open("a") as fh:
            for v in todo:
                if over(a):
                    break
                key = TV.wkey(v)
                k = kidx[key]
                f = TV.win_feats(corpus, v)
                y = np.asarray(v["spk_targets"], np.float32)
                T = len(y)
                p = np.load(tdir / f"{key}.stream_rc.npy")
                p = np.stack([E._fit(p[:, j], T) for j in range(p.shape[1])], 1)
                S = p.shape[1]
                emb_ok = TV.spk_column_embeddings(model, f, p)
                tn_ok = E.v2_load_titanet(twork, v, T)
                p3 = np.load(n3dir / f"{key}.stream_rc.npy")  # the cached Nemotron-3 column, for the 0.6B words
                p3 = np.stack([E._fit(p3[:, j], T) for j in range(p3.shape[1])], 1)
                emb3 = TV.spk_column_embeddings(model, f, p3)
                tn3 = E.v2_load_titanet(n3work, v, T)
                units = []
                for c, spk in enumerate(TV.window_speakers(ds, v)):
                    if not y[:, c].any() or not VP["has_5p0"][k, c]:
                        continue
                    ref = [x[0] for x in TW.ref_words(ds, v, spk)]
                    arms = {}
                    for nm, col, pp in (("pya_spk", TV.vp_follow(emb_ok[0], emb_ok[1], VP["spk_5p0"][k, c]), p),
                                        ("pya_tn", TV.vp_follow(tn_ok[0], tn_ok[1], VP["tn_5p0"][k, c]), p),
                                        ("n3_spk", TV.vp_follow(emb3[0], emb3[1], VP["spk_5p0"][k, c]), p3),
                                        ("n3_tn", TV.vp_follow(tn3[0], tn3[1], VP["tn_5p0"][k, c]), p3)):
                        mk = np.where(col >= 0, pp[np.arange(T), np.clip(col, 0, pp.shape[1] - 1)], 0.0) > 0.5
                        for tag, hyp, lag in (("", asr[key]["words"], L), ("0p6b_", asr6.get(key, {}).get("words"), L6)):
                            if hyp is None or lag is None:
                                continue
                            kp = TW.keep_mask(hyp, mk, lag, TW.DIL)
                            h = [x[0] for x, q in zip(hyp, kp) if q]
                            s_, d_, i_, _, _ = TW.align_counts(ref, h)
                            arms[f"{tag}{nm}_d{TW.DIL}"] = [int(s_), int(d_), int(i_), len(h)]
                    units.append({"key": key, "meeting": v["meeting"], "col": c, "N": len(ref), "arms": arms})
                fh.write(json.dumps({"key": key, "units": units}) + "\n")
                fh.flush()
                n += 1
        log(f"pyatwer {corpus}: {n} done, {len(todo) - n} left")
        if len(todo) - n:
            return
    log("STAGE_COMPLETE")


def score_pyatwer() -> dict:
    """tWER of the pyannote arms, primary group (column 0), 1000-resample meeting bootstrap (tswer.report's protocol)."""
    TV = _tv()
    out = {}
    for corpus in ("ami", "icsi"):
        ext, meta, ds = TV.bench_windows(corpus)
        up = WORK / "pya" / corpus / "units.jsonl"
        rows = [json.loads(x) for x in up.read_text().splitlines() if x.strip()] if up.exists() else []
        if len(rows) < len(ext):
            out[corpus] = {"complete": False, "n_done": len(rows), "n": len(ext)}
            continue
        U = [u for r in rows for u in r["units"] if u["col"] == 0]
        meets = sorted({u["meeting"] for u in U})
        o = {"n_units": len(U), "n_meetings": len(meets)}
        rng = np.random.default_rng(0)
        bi = [rng.integers(0, len(meets), len(meets)) for _ in range(1000)]
        for arm in sorted({k for u in U for k in u["arms"]}):
            per = np.array([[sum(sum(u["arms"][arm][:3]) for u in U if u["meeting"] == m), sum(u["N"] for u in U if u["meeting"] == m)]
                            for m in meets], float)
            w = per[:, 0].sum() / per[:, 1].sum()
            b = [per[k, 0].sum() / per[k, 1].sum() for k in bi]
            o[arm] = {"twer_pct": round(100 * w, 2), "ci_meeting": [round(100 * float(np.percentile(b, q)), 2) for q in (2.5, 97.5)]}
        out[corpus] = o
    return out


# =========================================================================== cost and sizes
def stage_cost(a):
    """mps_115m.py's engine / streams protocols on either core, --device (ms per 160 ms chunk of the full single-mode
    engine on the bundled clip, best of 3; K interleaved sessions, real time while p95 of the summed block compute <
    160 ms)."""
    import functools
    import audioforge
    import core_0p6b_heads as C
    import mps_115m as M
    M.SCRATCH = WORK / "cost" / a.sys
    lidkw = {"lid": os.environ["FINAL_LID_HEAD"]} if os.environ.get("FINAL_LID_HEAD") else {}  # a candidate LID head
    if a.sys == "0p6b":
        M.PRINT = C.W / "two_party_call_16s.voiceprint_0p6b.json"
        assert M.PRINT.exists()
        audioforge.load = functools.partial(audioforge.load, core="0.6b", asr=str(afm_0p6b()), **lidkw)
    else:
        audioforge.load = functools.partial(audioforge.load, asr=str(AFM_115M), **lidkw)
    if a.sub == "engine":
        M.stage_engine(a)
    else:
        a.ks = [int(k) for k in a.ks.split(",")]
        M.stage_streams(a)
    log("STAGE_COMPLETE")


def stage_params(a):
    """Parameter counts of every system, counted from the weights on disk (no numbers from model cards)."""
    import torch
    out = {}

    def st_count(repo):
        from safetensors import safe_open
        n = 0
        p = Path(snap(repo)) / "model.safetensors"
        with safe_open(str(p), "pt") as f:
            for k in f.keys():
                n += int(np.prod(f.get_slice(k).get_shape()))
        return n, p.stat().st_size
    for name, repo in (("whisper_small", "openai/whisper-small"), ("whisper_turbo", "openai/whisper-large-v3-turbo"),
                       ("whisper_large_v3", "openai/whisper-large-v3")):
        n, sz = st_count(repo)
        out[name] = {"params": n, "file_mb": round(sz / 2 ** 20, 1), "file": repo}
    from audioforge.nemo_import import read_nemo
    for name, f in (("tdt_v3", "parakeet-tdt-0.6b-v3.nemo"), ("parakeet_realtime_eou", "parakeet_realtime_eou_120m-v1.nemo"),
                    ("marblenet_v2", "frame_vad_multilingual_marblenet_v2.0.nemo"), ("ambernet", "langid_ambernet.nemo"),
                    ("titanet_l", "speakerverification_en_titanet_large.nemo"), ("nemotron3_diar", "Nemotron-3-Diarization.nemo"),
                    ("nemotron_streaming_0p6b_base", "nemotron-speech-streaming-en-0.6b.nemo")):
        p = ROOT / "data" / "nemo" / f
        _, sd, _ = read_nemo(p)
        n = int(sum(v.numel() for k, v in sd.items() if hasattr(v, "numel") and "preprocessor" not in k
                    and not k.endswith(("num_batches_tracked", "running_mean", "running_var"))))
        out[name] = {"params": n, "file_mb": round(p.stat().st_size / 2 ** 20, 1), "file": f"data/nemo/{f}"}
    m = torch.jit.load(str(SSD / "scratch/metrics/vad/silero_vad_v5.1.2.jit"))
    out["silero_v5"] = {"params": int(sum(p.numel() for p in m.parameters())), "file": "silero_vad_v5.1.2.jit"}
    import onnx
    from onnx import numpy_helper

    def onnx_count(p):
        mm = onnx.load(str(p))
        return int(sum(numpy_helper.to_array(t).size for t in mm.graph.initializer)), round(Path(p).stat().st_size / 2 ** 20, 1)
    import pipecat
    st = next(Path(pipecat.__file__).parent.rglob("smart-turn-v3*.onnx"))
    n, sz = onnx_count(st)
    out["smartturn_v3.2"] = {"params": n, "file_mb": sz, "file": str(st.relative_to(Path(pipecat.__file__).parent.parent))}
    lk = [p for p in (HUB / "models--livekit--turn-detector").rglob("*.onnx")]
    for p in lk:
        n, sz = onnx_count(p)
        out[f"livekit_turn_detector/{p.parent.name}/{p.name}"] = {"params": n, "file_mb": sz}
    os.environ["HF_HUB_OFFLINE"] = "1"
    from pyannote.audio import Model
    for name, repo in (("pyannote_seg3", "pyannote/segmentation-3.0"), ("wespeaker_resnet34", "pyannote/wespeaker-voxceleb-resnet34-LM")):
        pm = Model.from_pretrained(repo)
        out[name] = {"params": int(sum(p.numel() for p in pm.parameters())), "file": repo}
    import ten_vad
    lib = next(Path(ten_vad.__file__).parent.rglob("ten_vad.framework/Versions/A/ten_vad"))
    out["ten_vad"] = {"params": None, "file_mb": round(lib.stat().st_size / 2 ** 20, 2),
                      "note": "compiled library (weights embedded); parameter count not exposed"}
    from audioforge.train import load_model
    for name, p in (("core_115m", AFM_115M), ("core_0p6b", afm_0p6b())):
        m = load_model(str(p), "cpu")
        out[name] = {"params": int(sum(x.numel() for x in m.parameters())), "file": str(p),
                     "file_mb": round(p.stat().st_size / 2 ** 20, 1)}
    (WORK / "params.json").write_text(json.dumps(out, indent=1))
    log(json.dumps(out, indent=1))
    log("STAGE_COMPLETE")


# =========================================================================== report
def load_json(rel):
    return json.loads((ROOT / rel).read_text())


def compile_speaker() -> dict:
    """Target-speaker rows on the identical eot-bench windows: reused (runs/tswer.json, runs/core_0p6b.json tswer /
    tsvad_frame, runs/improve_115m.json frame) plus this pass's pyannote 3.1 arm and the Nemotron-3 column on the
    0.6B words."""
    T = load_json("runs/tswer.json")["results"]
    C6 = load_json("runs/core_0p6b.json")
    FR = load_json("runs/improve_115m.json")["frame"]
    py, pf = score_pyatwer(), score_pyaframe()
    out = {"twer": {}, "frame": {},
           "sources": {"twer": "runs/tswer.json results > <corpus> > primary > arms (115M words; reused, identical "
                               "windows / words / keep rule), runs/core_0p6b.json tswer (0.6B), this pass: pyannote 3.1 "
                               "arm and Nemotron-3 on the 0.6B words (WORK/pya/<corpus>/units.jsonl)",
                       "frame": "runs/improve_115m.json frame (115M, Nemotron-3), runs/core_0p6b.json tsvad_frame "
                                "(0.6B), this pass: pyannote 3.1 (WORK/pya/<corpus>/framecounts)"}}
    for c in ("ami", "icsi"):
        A = T[c]["primary"]["arms"]
        g = lambda arm: {"twer": A[arm]["twer"], "ci_meeting": A[arm].get("ci_meeting")}  # noqa: E731
        o = {"core115_tsvad_d2": g("tsvad_d2"), "none": g("none"), "oracle_d2": g("oracle_d2"),
             "0p6b_none": g("0p6b_none"), "0p6b_oracle_d2": g("0p6b_oracle_d2"),
             "nemotron3_spk": g("n3_spk_d2"), "nemotron3_tn": g("n3_tn_d2"),
             "core0p6b_tsvad_d2": C6["tswer"][c]["primary"]["core0p6b_tsvad_d2"]}
        o["nemotron3_best"] = min((o["nemotron3_spk"], o["nemotron3_tn"]), key=lambda r: r["twer"])
        P = py.get(c, {})
        if P.get("complete", True) and P:
            for arm in ("pya_spk_d2", "pya_tn_d2", "0p6b_pya_spk_d2", "0p6b_pya_tn_d2", "n3_spk_d2", "n3_tn_d2",
                        "0p6b_n3_spk_d2", "0p6b_n3_tn_d2"):
                if arm in P:
                    o[f"this_pass/{arm}"] = {"twer": P[arm]["twer_pct"], "ci_meeting": P[arm]["ci_meeting"]}
            best = lambda a1, a2: min((o[f"this_pass/{a1}"], o[f"this_pass/{a2}"]), key=lambda r: r["twer"])  # noqa: E731
            o["pyannote31_best"] = best("pya_spk_d2", "pya_tn_d2")
            o["0p6b_pyannote31_best"] = best("0p6b_pya_spk_d2", "0p6b_pya_tn_d2")
            o["0p6b_nemotron3_best"] = best("0p6b_n3_spk_d2", "0p6b_n3_tn_d2")
        out["twer"][c] = o

        def fr_row(r):
            tf, nf = r["target_frames"], r["nontarget_frames"]
            return {"f1": r["f1"], "der_pct": round(100 * (r["miss"] * tf + r["fa"] * nf) / tf, 2), "f1_ci": r.get("f1_ci")}
        F = FR[c]["primary"]
        f = {"ours_115m": fr_row(F["tsvad_spk_vp5p0"]["all"]),
             "ours_0p6b": fr_row(C6["tsvad_frame"][c]["primary"]["tsvad_a_vp5p0"]["all"])}
        f["ours_115m"]["f1_ci"] = FR[c]["tsvad_spk_vp5p0_ci"]["primary"]["all"]["f1_ci"]
        n3 = [fr_row(F[k]["all"]) for k in ("nemotron3_vp_spk_vp5p0", "nemotron3_vp_titanet_vp5p0")]
        f["nemotron3"] = max(n3, key=lambda r: r["f1"])
        if pf.get(c, {}).get("complete", True) and c in pf:
            rows = pf[c]
            cand = [rows[k] for k in ("pyannote31_vp_spk_vp5p0", "pyannote31_vp_titanet_vp5p0") if k in rows]
            if cand:
                f["pyannote31"] = max(cand, key=lambda r: r["f1"])
            f["this_pass_check_tsvad_115m"] = rows.get("tsvad_spk_vp5p0")
        out["frame"][c] = f
    return out


def cost_summary(cost: dict) -> dict:
    out = {}
    for c, files in cost.items():
        o = {}
        for dev in ("mps", "cpu"):
            st = files.get(f"streams_{dev}")
            if st:
                ok = [int(k) for k, v in st.items() if v.get("real_time")]
                o[dev] = {"realtime_streams": max(ok) if ok else 0,
                          "per_stream_block_ms_p50_at_1": st.get("1", {}).get("per_stream_block_ms_p50")}
        out[c] = o
    return out


def stage_report(a):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["generated"] = time.strftime("%Y-%m-%d %H:%M")
    res["machine"] = "Apple M5 laptop (macOS), MPS for the served cores' served metrics and the large Whisper models, CPU 2 threads elsewhere"
    res["afm"] = {"115m": str(AFM_115M), "0p6b": str(afm_0p6b()), "0p6b_heads": Path(afm_0p6b()).stem,
                  "lid_heads": {k: str(v.relative_to(ROOT)) for k, v in LID_HEAD.items()},
                  "note": "the builds the report resolves turn presets from; the files that produced each section: "
                          "files_used"}
    res["files_used"] = FILES_USED
    for k, fn in (("words", score_asr), ("stt_latency", score_sttlat), ("vad", score_vad), ("lid", score_lid),
                  ("turn", score_turn), ("speaker_eer", score_spkeer), ("pyannote_frame", score_pyaframe),
                  ("pyannote_twer", score_pyatwer)):
        if a.only and k not in a.only.split(","):
            continue
        try:
            res[k] = fn()
            res.get("errors", {}).pop(k, None)
            log(f"report: {k} ok")
        except Exception as ex:  # noqa: BLE001
            log(f"report: {k} failed: {type(ex).__name__}: {ex}")
            res.setdefault("errors", {})[k] = f"{type(ex).__name__}: {ex}"
    try:  # extra words row: nemotron-3.5-asr-streaming-0.6b (en-US prompt), reused from runs/core_3p5.json (same sets)
        c35 = load_json("runs/core_3p5.json")
        row = {"label": "Nemotron 3.5 ASR 0.6B (multilingual), en-US prompt", "source": "runs/core_3p5.json wer / live_wer "
               "(reused: identical 200-item sets and 32 live sessions; masked [56,1] forward = 160 ms streaming)"}
        for st in ("libri", "ami", "icsi"):
            r = c35["wer"]["sets"][st]["3p5_enUS"]
            row[st] = {nn: {"wer_pct": round(100 * r[nn]["wer"], 2), "ci95": [round(100 * x, 2) for x in r[nn]["ci95"]]}
                       for nn in ("whisper_norm", "normalize_text")}
        for sub in ("all", "user_channel"):
            r = c35["live_wer"]["en-US"][sub]["3p5"]["none_full"]
            row[f"live|{sub}"] = {"normalize_text": {"wer_pct": r["wer"], "ci95": r["ci_clip"]}}
        res["words_nemotron35"] = row
    except Exception as ex:  # noqa: BLE001
        res.setdefault("errors", {})["words_nemotron35"] = f"{type(ex).__name__}: {ex}"
    if (WORK / "params.json").exists():
        res["params"] = json.loads((WORK / "params.json").read_text())
    import torch
    for c, f in LID_HEAD.items():  # the LID heads v2 (attached at load, not inside the afm)
        t = torch.load(f, map_location="cpu", weights_only=False)
        sd = t.get("tensors") or t.get("state_dict") or t
        res.setdefault("params", {})[f"lid_head_v2_{c}"] = {
            "params": int(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))), "file": str(f.relative_to(ROOT))}
    res["cost"] = {c: {p.stem: json.loads(p.read_text()) for p in (WORK / "cost" / c).glob("*.json")}
                   for c in ("115m", "0p6b") if (WORK / "cost" / c).exists()}
    res["cost_summary"] = cost_summary(res["cost"])
    if not a.only or "speaker" in a.only.split(","):
        try:
            res["speaker"] = compile_speaker()
            res.get("errors", {}).pop("speaker", None)
        except Exception as ex:  # noqa: BLE001
            res.setdefault("errors", {})["speaker"] = f"{type(ex).__name__}: {ex}"
    fx = load_json("runs/fixall.json") if (ROOT / "runs/fixall.json").exists() else {}

    def frame_best(fr):
        """der_pct here is the target-speaker miss + false-alarm rate ((missed + false target frames) / target
        frames, 80 ms, no collar), not the standard DER (that one: stdder)."""
        if not fr:
            return {}
        o = {"ours_115m": fr["pyannote31"]["tsvad_spk_vp5p0"], "ours_0p6b": fr["ours_0p6b"],
             "nemotron3": max((fr["nemotron3"]["nemotron3_vp_spk_vp5p0"], fr["nemotron3"]["nemotron3_vp_titanet_vp5p0"]),
                              key=lambda r: r["f1"]),
             "pyannote31": max((fr["pyannote31"]["pyannote31_vp_spk_vp5p0"], fr["pyannote31"]["pyannote31_vp_titanet_vp5p0"]),
                               key=lambda r: r["f1"]),
             # upper bound of the diarizer arms: the column that best matches the target, chosen with the labels
             "nemotron3_oracle_binding": fr["nemotron3"]["nemotron3_oracle_column"],
             "pyannote31_oracle_binding": fr["pyannote31"]["pyannote31_oracle_column"]}
        for r in o.values():
            r["target_miss_fa_pct"] = r["der_pct"]
        return o
    if fx.get("twsub"):  # target-speaker rows on the ICSI / AMI TEST meetings (research/FIXALL.md test audit)
        fr = fx.get("frsub") or {}
        res["speaker_test"] = {"icsi": {"twer": fx["twsub"], "frame": fr, "frame_best": frame_best(fr)},
                               "source": "runs/fixall.json twsub / frsub: the stored per-unit / per-window counts of the "
                                         "eot-bench v2 ICSI windows re-pooled over the test meetings Bmr013, Bmr018, "
                                         "Bro021 (the 0.6B TS-VAD frame metrics re-run on those windows); ami_eval: the "
                                         "same pipeline run on the AMI test meetings' windows"}
        if fx.get("twsub_ami_eval"):
            fa = fx.get("frsub_ami_eval") or {}
            res["speaker_test"]["ami_eval"] = {"twer": fx["twsub_ami_eval"], "frame": fa, "frame_best": frame_best(fa)}
    import eot_latency as E
    import final_scoring as FS
    res["ami_turn_split"] = E.AMI_SPLIT
    td = load_json("runs/turn_data.json")["evalv04"]  # the 0.6B heads v0.3 turn row (before v0.4), from its run file
    res["superseded"] = {"turn.ours_0p6b (heads v0.3)": {**td["v0.3"], "source": "runs/turn_data.json evalv04 > v0.3 "
                                                         "(the same scorer, dumps of served_0p6b_v0.3.afm)"}}
    if (ROOT / "runs/turn_clean.json").exists():  # held-out re-pick of the turn presets (plans/fixwave/turn_clean.md)
        tc = load_json("runs/turn_clean.json")
        res["turn_clean"] = {"source": "runs/turn_clean.json (the turn-clean agent's run; keys: plans/fixwave/turn_clean.md "
                                       "section 7)", **tc}
    # speech heads retrained without any ICSI training meeting holding an ICSI test speaker (turn_clean.json speech):
    # their per-frame outputs scored here with the same scorer and the same baselines (frames inside the audio, meeting
    # CIs). The ICSI headline speech row of our cores; the shipped heads' ICSI row is seen-speaker (leakage item 3).
    sc = SSD / "scratch" / "speech_clean"
    if all((sc / f"fc_{c}_new" / "vad" / f"core_{c}_icsi_eval.npz").exists() for c in ("115m", "0p6b")):
        mix = WORK / "vad_speakers_unseen"
        (mix / "vad").mkdir(parents=True, exist_ok=True)
        for sn in ("ami_eval", "icsi_eval"):
            for s_ in VAD_SYSTEMS:
                c = s_.replace("core_", "")
                src = (sc / f"fc_{c}_new" / "vad" / f"{s_}_{sn}.npz") if s_.startswith("core_") else (WORK / "vad" / f"{s_}_{sn}.npz")
                dst = mix / "vad" / f"{s_}_{sn}.npz"
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src)
        res["vad_speakers_unseen"] = {
            "label": "speakers unseen in training: our speech heads retrained without every ICSI training meeting that "
                     "holds any of the 13 ICSI test speakers (runs/turn_clean.json speech; not shipped); baselines as in vad",
            "heads": {c: str(sc / f"fc_{c}_new") for c in ("115m", "0p6b")},
            **FS.score_vad(mix, VAD_SYSTEMS, ("ami_eval", "icsi_eval"))}
    old = SSD / "scratch" / "fixall" / "fc_old"  # the previous heads (115M v0.3 / 0.6B v0.2 VAD; lid_distill / lid_0p6b)
    if not a.only or "vad" in a.only.split(","):
        res["vad_previous_heads"] = {"source": str(old / "vad"), **FS.score_vad(old, VAD_SYSTEMS, VAD_SETS)}
    import lid as L
    y = np.array([L.CODES.index(r["lang"]) for r in L.rows_for("fleurs")])
    res["lid_previous_heads"] = {"source": str(old / "lid")}
    for s_ in ("core_115m", "core_0p6b"):
        z = np.load(old / "lid" / f"{s_}.npz")
        res["lid_previous_heads"][s_] = {f"acc_{c}_pct": round(100 * float(np.mean(z["P"][:, i].argmax(-1) == y)), 2)
                                         for i, c in enumerate(LID_COND)}
    fsd = WORK / "meta" / "stdder.json"
    if fsd.exists() and "speaker_test" in res:  # standard DER of the open diarizers' tracks (final_scoring.py stdder)
        res["speaker_test"]["stdder"] = json.loads(fsd.read_text())
    res["systems"] = {"asr": ASR_SYSTEMS, "vad": VAD_SYSTEMS}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    log(f"-> {OUT}")


STAGES = {"asr": stage_asr, "sttlat": stage_sttlat, "vad": stage_vad, "lid": stage_lid, "eou": stage_eou,
          "eotdump": stage_eotdump, "spkeer": stage_spkeer, "pyatracks": stage_pyatracks, "pyatn": stage_pyatn,
          "pyaframe": stage_pyaframe, "pyatwer": stage_pyatwer, "cost": stage_cost, "params": stage_params,
          "report": stage_report}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--system", default="")
    p.add_argument("--set", default="ami")
    p.add_argument("--sys", default="115m", choices=("115m", "0p6b"))
    p.add_argument("--which", default="calls", choices=("calls", "asst"))
    p.add_argument("--device", default="mps")
    p.add_argument("--budget", type=float, default=540)
    p.add_argument("--corpora", default="ami,icsi")
    p.add_argument("--sub", default="engine")
    p.add_argument("--ks", default="1,2,3,4,5,6")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--only", default="")
    p.add_argument("--noprint", action="store_true")
    a = p.parse_args()
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
