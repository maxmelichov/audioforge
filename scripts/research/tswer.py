"""Target-speaker WER (tWER) of single-model mode: how well do we transcribe ONLY the enrolled user's words in a
multi-speaker recording (research/TSWER.md, runs/tswer.json).

Units = (window, target) of the IMPROVE_115M A.1 / A.1b frame evaluation: the eot-bench v2 extended windows (AMI dev
974, ICSI held-out 1312), every labelled column speaker that speaks in the window and has a 5 s voice print (the
cached tsvad.py ``vprints``: that speaker's single-speaker speech elsewhere in the meeting, served block-4 speaker
head; column 0 = the turn's primary = the A.1 "primary" group).

  reference   the target's own words (forced-aligned word times of the corpus annotations) whose midpoint lies in
              the window, teachers.normalize_text
  hypothesis  the served model's streaming RNNT words (pass 1 of ``--mode single``: runs/stage1_served.afm, rnnt head,
              att [70, 1] = masked offline forward == cache-aware streaming) on the window audio, each word timed by
              the encoder frames at which its tokens were emitted; a word is kept when the arm's target mask is on at
              its timing frame (emission frame shifted back by the measured emission lag, mask dilated by +-DIL
              frames; the same rule for every arm)
  arms        tsvad   P(target) > 0.5 of the served TS-VAD head (assets/tsvad_spk.pt == runs/tsvad_spk.pt) with the
                      5 s print: the served track (tsvad_stream.track_probs = TSVADTrack with the print set at frame
                      0, anchored adaptation on; research/TSWER.md "Fix"; before the fix tsvad.tsvad_probs, as A.1)
              oracle  the target's label activity (word intervals, 80 ms frames): isolates ASR error
              none    every word (no target speaker: other people's words are insertions)
              n3_spk / n3_tn  the Nemotron-3-Diarization column (served settings, cached A.1b tracks) bound by the same
                      5 s print (tsvad.vp_follow, speaker head / TitaNet-L), p > 0.5
              0p6b_tsvad  words of nemotron-speech-streaming-en-0.6b at [70, 1] (if decoded) filtered by the TS-VAD mask

Stages (each <= --budget s, resumable, CPU 2 threads; run through scripts/dev/gate.sh):
  asr    [--asr served|0p6b]   decode the windows -> <scratch>/tswer/<corpus>/asr_<name>.jsonl
  score                        per-unit S/D/I/N for every arm -> <scratch>/tswer/<corpus>/units.jsonl (+ lag.json)
  turnbench                    the 16 TurnBench dev clips of the live single-model study (user = the human party)
  tsvad_arms                   recompute only the tsvad_* arms of <corpus>/units.jsonl with the served track (the rest of
                               the score stage, Nemotron-3 arms included, does not depend on it)
  report                       aggregate, 1000-resample bootstraps (by meeting and by window) -> runs/tswer.json
  live / live_report           the 69-session live suite of runs/single_model.json table.live_69 (every session with a
                               per-speaker reference: 16 TurnBench clips x mono / user channel) -> runs/tswer_live.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import tsvad as TV  # noqa: E402

WORK = TV.SCRATCH / "tswer"
SR, FRAME_S = 16000, 0.08
ATT = [70, 1]
ASR_MODELS = {"served": ROOT / "runs" / "stage1_served.afm",
              "0p6b": ROOT / "data" / "nemo" / "nemotron-speech-streaming-en-0.6b.nemo"}
DIL = 2  # mask dilation in frames (+-160 ms) around a word's timing frame, every arm
N_BOOT = 1000


def log(*a):
    print(*a, flush=True)


def served_tsvad(head, spk, f, e) -> np.ndarray:
    """(T, 2) the served TS-VAD track of block-4 frames ``f`` with the print ``e`` (audioforge.tsvad_stream.TSVADTrack:
    what `--mode single` runs, anchored adaptation included)."""
    from audioforge.tsvad_stream import track_probs
    return track_probs(head, spk, f, np.asarray(e, np.float32))


def asr_path(corpus, name) -> Path:
    return WORK / corpus / f"asr_{name}.jsonl"


def load_asr(corpus, name) -> dict:
    p = asr_path(corpus, name)
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


# --------------------------------------------------------------------------- decoding with emission frames
@torch.no_grad()
def greedy_timed(head, f):
    """RNNTHead._greedy with the encoder frame of every emitted token: [(token id, frame)]."""
    V1, dev = head.vocab_size + 1, f.device
    hyp, state = [], None
    g, state = head.pred(torch.tensor([[head.blank]], device=dev), state)
    fe = head.joint.enc(f)
    t, T, emitted = 0, f.shape[0], 0
    while t < T:
        z = head.joint.out(fe[t][None, None] + head.joint.pred(g))[0, 0]
        k = int(z[:V1].argmax())
        if head.is_tdt:
            d = head.durations[int(z[V1:].argmax())]
            if k == head.blank and d == 0:
                d = 1
        else:
            d = 1 if k == head.blank else 0
        if k != head.blank:
            hyp.append((k, t))
            g, state = head.pred(torch.tensor([[k]], device=dev), state)
            emitted += 1
        if d == 0 and emitted >= head.max_symbols:
            d = 1
        if d > 0:
            emitted = 0
        t += d
    return hyp


def words_of(tok, hyp):
    """[(id, frame)] -> [[normalized word, first frame, last frame]] (a piece starting with U+2581 starts a word)."""
    from audioforge.teachers import normalize_text
    groups = []
    for k, t in hyp:
        piece = tok.sp.id_to_piece(k)
        if not groups or piece.startswith("▁"):
            groups.append([[k], t, t])
        else:
            groups[-1][0].append(k)
            if any(ch.isalnum() for ch in piece):  # a trailing punctuation piece (the 0.6B emits PnC) keeps the time
                groups[-1][2] = t
    out = []
    for ids, f0, f1 in groups:
        for w in normalize_text(tok.sp.decode(ids)).split():
            out.append([w, int(f0), int(f1)])
    return out


def load_asr_model(name):
    if name == "served":
        from audioforge.train import load_model
        return load_model(str(ASR_MODELS[name]), "cpu").eval()
    from audioforge.nemo_import import import_nemo
    return import_nemo(ASR_MODELS[name]).eval()


def stage_asr(a):
    TV.set_threads(2)
    t0 = time.time()
    m = None
    for corpus in a.corpora.split(","):
        ext, meta, ds = TV.bench_windows(corpus)
        done = load_asr(corpus, a.asr)
        todo = [v for v in ext if TV.wkey(v) not in done]
        if not todo:
            log(f"  {corpus} asr {a.asr}: all {len(ext)} done")
            continue
        if m is None:
            m = load_asr_model(a.asr)
            hname = "rnnt" if "rnnt" in m.heads else next(k for k, v in m.head_cfg.items() if v["type"] in ("rnnt", "tdt"))
            log(f"  model {a.asr}: head {hname} ({type(m.heads[hname]).__name__}), frame {m.frame_sec}s")
        p = asr_path(corpus, a.asr)
        p.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with p.open("a") as fh:
            for i in range(0, len(todo), a.batch):
                if time.time() - t0 > a.budget:
                    break
                cc = todo[i: i + a.batch]
                xs = [np.asarray(v["audio"], np.float32) for v in cc]
                x, lens = m._pad(xs)
                with torch.inference_mode():
                    enc, elen, hidden = m.encode(x, lens, ATT, return_hidden=True)
                    e = m.head_input(hname, enc, hidden)
                    for j, v in enumerate(cc):
                        hyp = greedy_timed(m.heads[hname], e[j, : int(elen[j])])
                        fh.write(json.dumps({"key": TV.wkey(v), "T_enc": int(elen[j]),
                                             "words": words_of(m.tokenizer, hyp)}) + "\n")
                fh.flush()
                n += len(cc)
        left = len(todo) - n
        log(f"  {corpus} asr {a.asr}: {n} done, {left} left ({time.time() - t0:.0f}s)")
        if left:
            break


# --------------------------------------------------------------------------- scoring
def align_counts(ref: list[str], hyp: list[str]):
    """(S, D, I, hits, alignment) by jiwer (Levenshtein, words)."""
    import jiwer
    if not ref:
        return 0, 0, len(hyp), 0, []
    if not hyp:
        return 0, len(ref), 0, 0, []
    o = jiwer.process_words(" ".join(ref), " ".join(hyp))
    return o.substitutions, o.deletions, o.insertions, o.hits, o.alignments[0]


def ref_words(ds, v, spk):
    a = float(v["start"])
    b = a + len(v["audio"]) / SR
    from audioforge.teachers import normalize_text
    out = []
    for s, e, w in ds.words[v["meeting"]].get(spk, []):
        if a <= (s + e) / 2 < b:
            for t in normalize_text(w).split():
                out.append((t, s - a, e - a))
    return out


def word_frame(w, lag):
    return int(round((w[1] + w[2]) / 2 - lag))


def keep_mask(words, mask, lag, dil):
    T = len(mask)
    md = mask.copy()
    for d in range(1, dil + 1):
        md[d:] |= mask[:-d]
        md[:-d] |= mask[d:]
    return [bool(md[min(max(word_frame(w, lag), 0), T - 1)]) for w in words]


def measure_lag(corpus, ext, ds, asr, sample=300):
    """Emission lag (frames) of hyp words vs their aligned reference word (all speakers' words in the window, hits
    only): median of (word timing frame at lag 0) - (reference word mid frame). Target-agnostic ASR property."""
    lags = []
    for v in ext[::max(1, len(ext) // sample)]:
        r = asr[TV.wkey(v)]
        refs = sorted([w for s in TV.window_speakers(ds, v) for w in ref_words(ds, v, s)], key=lambda x: x[1])
        hyp = r["words"]
        _, _, _, _, al = align_counts([w[0] for w in refs], [w[0] for w in hyp])
        for ch in al:
            if ch.type == "equal":
                for q in range(ch.ref_end_idx - ch.ref_start_idx):
                    rw, hw = refs[ch.ref_start_idx + q], hyp[ch.hyp_start_idx + q]
                    lags.append((hw[1] + hw[2]) / 2 - (rw[1] + rw[2]) / 2 / FRAME_S)
    lags = np.array(lags)
    return {"median": float(np.median(lags)), "p10": float(np.percentile(lags, 10)),
            "p90": float(np.percentile(lags, 90)), "n_hits": int(len(lags))}


def stage_score(a):
    import eval_stage1 as E
    TV.set_threads(2)
    model = TV.load_served()
    head = TV.load_head(ROOT / "assets" / "tsvad_spk.pt")[0]
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = TV.bench_windows(corpus)
        asr = load_asr(corpus, "served")
        asr6 = load_asr(corpus, "0p6b")
        assert all(TV.wkey(v) in asr for v in ext), "run the asr stage first"
        use6 = all(TV.wkey(v) in asr6 for v in ext)
        lag = measure_lag(corpus, ext, ds, asr)
        lag6 = measure_lag(corpus, ext, ds, asr6) if use6 else None
        L, L6 = round(lag["median"]), (round(lag6["median"]) if use6 else 0)
        log(f"  {corpus}: emission lag {lag} -> {L} frames" + (f"; 0.6B {lag6} -> {L6}" if use6 else ""))
        kidx, VP = TV.load_vprints(corpus)
        tdir, twork = TV.DIAR_ARMS["nemotron3"](corpus)
        up = WORK / corpus / "units.jsonl"
        done = {json.loads(x)["key"] for x in up.read_text().splitlines() if x.strip()} if up.exists() else set()
        todo = [v for v in ext if TV.wkey(v) not in done]
        fh = up.open("a")
        n = 0
        for v in todo:
            if time.time() - t0 > a.budget:
                break
            key = TV.wkey(v)
            units = []
            k = kidx[key]
            f = TV.win_feats(corpus, v)
            y = np.asarray(v["spk_targets"], np.float32)
            T = len(y)
            p = np.load(tdir / f"{key}.stream_rc.npy")
            p = np.stack([E._fit(p[:, j], T) for j in range(p.shape[1])], 1)
            S = p.shape[1]
            emb_ok = TV.spk_column_embeddings(model, f, p)
            tn_ok = E.v2_load_titanet(twork, v, T)
            hyp = asr[key]["words"]
            hyp6 = asr6[key]["words"] if use6 else None
            spks = TV.window_speakers(ds, v)
            for c, spk in enumerate(spks):
                yt = y[:, c]
                if not yt.any() or not VP["has_5p0"][k, c]:
                    continue
                ref = [x[0] for x in ref_words(ds, v, spk)]
                masks = {"tsvad": served_tsvad(head, model.heads["spk"], f, VP["spk_5p0"][k, c])[:, 0] > 0.5,
                         "oracle": yt > 0.5}
                col = TV.vp_follow(emb_ok[0], emb_ok[1], VP["spk_5p0"][k, c])
                masks["n3_spk"] = np.where(col >= 0, p[np.arange(T), np.clip(col, 0, S - 1)], 0.0) > 0.5
                col = TV.vp_follow(tn_ok[0], tn_ok[1], VP["tn_5p0"][k, c])
                masks["n3_tn"] = np.where(col >= 0, p[np.arange(T), np.clip(col, 0, S - 1)], 0.0) > 0.5
                arms = {"none": [x[0] for x in hyp]}
                for name, mk in masks.items():
                    for dil in (0, DIL, 4):
                        kp = keep_mask(hyp, mk, L, dil)
                        arms[f"{name}_d{dil}"] = [x[0] for x, q in zip(hyp, kp) if q]
                if use6:
                    arms["0p6b_none"] = [x[0] for x in hyp6]
                    for name in ("tsvad", "oracle"):
                        kp = keep_mask(hyp6, masks[name], L6, DIL)
                        arms[f"0p6b_{name}_d{DIL}"] = [x[0] for x, q in zip(hyp6, kp) if q]
                u = {"key": key, "meeting": v["meeting"], "col": c, "N": len(ref), "arms": {}}
                for name, h in arms.items():
                    s_, d_, i_, hit, _ = align_counts(ref, h)
                    u["arms"][name] = [int(s_), int(d_), int(i_), int(len(h))]
                units.append(u)
            fh.write(json.dumps({"key": key, "units": units}) + "\n")
            fh.flush()
            n += 1
        fh.close()
        (WORK / corpus / "lag.json").write_text(json.dumps({"lag": lag, "lag_frames": L, "lag_0p6b": lag6,
                                                            "lag_0p6b_frames": L6, "n_windows": len(ext)}))
        left = len(todo) - n
        log(f"  {corpus} score: {n} windows done, {left} left ({time.time() - t0:.0f}s)")
        if left:
            break



def stage_tsvad_arms(a):
    """Rewrite the tsvad_* / 0p6b_tsvad_* arms of <corpus>/units.jsonl with the served track (every other arm, the
    Nemotron-3 ones included, is independent of the TS-VAD track and kept). The old file is kept as units_v1.jsonl."""
    TV.set_threads(2)
    model = TV.load_served()
    head = TV.load_head(ROOT / "assets" / "tsvad_spk.pt")[0]
    for corpus in a.corpora.split(","):
        ext, meta, ds = TV.bench_windows(corpus)
        asr, asr6 = load_asr(corpus, "served"), load_asr(corpus, "0p6b")
        z = json.loads((WORK / corpus / "lag.json").read_text())
        L, L6 = z["lag_frames"], z.get("lag_0p6b_frames") or 0
        kidx, VP = TV.load_vprints(corpus)
        up = WORK / corpus / "units.jsonl"
        old = up.with_name("units_v1.jsonl")
        if not old.exists():
            old.write_text(up.read_text())
        rows = [json.loads(x) for x in old.read_text().splitlines() if x.strip()]
        byk = {TV.wkey(v): v for v in ext}
        out = []
        for r in rows:
            v = byk[r["key"]]
            f = TV.win_feats(corpus, v)
            spks = TV.window_speakers(ds, v)
            for u in r["units"]:
                c = u["col"]
                ref = [x[0] for x in ref_words(ds, v, spks[c])]
                mk = served_tsvad(head, model.heads["spk"], f, VP["spk_5p0"][kidx[r["key"]], c])[:, 0] > 0.5
                hyp = asr[r["key"]]["words"]
                for dil in (0, DIL, 4):
                    kp = keep_mask(hyp, mk, L, dil)
                    h = [x[0] for x, q in zip(hyp, kp) if q]
                    s_, d_, i_, _, _ = align_counts(ref, h)
                    u["arms"][f"tsvad_d{dil}"] = [int(s_), int(d_), int(i_), int(len(h))]
                if f"0p6b_tsvad_d{DIL}" in u["arms"]:
                    hyp6 = asr6[r["key"]]["words"]
                    kp = keep_mask(hyp6, mk, L6, DIL)
                    h = [x[0] for x, q in zip(hyp6, kp) if q]
                    s_, d_, i_, _, _ = align_counts(ref, h)
                    u["arms"][f"0p6b_tsvad_d{DIL}"] = [int(s_), int(d_), int(i_), int(len(h))]
            out.append(r)
        up.write_text("".join(json.dumps(r) + "\n" for r in out))
        log(f"  {corpus}: tsvad arms of {len(out)} windows rewritten ({up})")


# --------------------------------------------------------------------------- TurnBench two-party clips
E2E = TV.SCRATCH / "e2e_tsvad"  # IMPROVEMENTS.md section 1 live clips: clips.json + prints.json (e2e_tsvad.py)


def tb_user_words(cid, human, a, b):
    """The user's reference words in [a, b) (clip-relative seconds): annotator a's segments of the user's channel whose
    midpoint lies inside (e2e_final.tb_ref_text's rule, bracketed tags removed), word times interpolated inside each
    segment by character length (dyadic.interpolate_words); also the segments as the oracle activity."""
    import re

    from audioforge.datasets import dyadic as D
    from audioforge.teachers import normalize_text
    row = D.tb_row(D.TB_ROOT, cid, ["conversation_id", f"speaker_{human + 1}_annotation_a"])
    words, segs = [], []
    for e in sorted(row[f"speaker_{human + 1}_annotation_a"], key=lambda e: e["start_s"]):
        if not a <= (e["start_s"] + e["end_s"]) / 2 < b:
            continue
        segs.append((e["start_s"] - a, e["end_s"] - a))
        txt = re.sub(r"\[[^\]]*\]", " ", e["text"])
        for s0, e0, w in D.interpolate_words(txt, e["start_s"] - a, e["end_s"] - a):
            for t in normalize_text(w).split():
                words.append((t, s0, e0))
    return words, segs


def stage_turnbench(a):
    """The 16 TurnBench dev clips of the live single-model study (35-53 s, mono mix of the two channels), target = the
    human party, the stored 5 s print (user's own channel, other party silent, outside the clip +-2 s; served speaker
    head). Served and 0.6B words on the mono clip at [70, 1]; TS-VAD on the clip's block-4 features; oracle = the
    user's annotated segments. No Nemotron-3 tracks exist for these clips (not run)."""
    from audioforge.datasets import dyadic as D
    TV.set_threads(2)
    clips = [c for c in json.loads((E2E / "clips.json").read_text()) if c["set"] == "turnbench"]
    prints = json.loads((E2E / "prints.json").read_text())
    ds = D.Dyadic(sorted({c["conversation"] for c in clips}), "turnbench", verbose=False)
    auds = []
    for c in clips:
        ch = np.asarray(ds.channels16k(c["conversation"])[:, int(c["start"] * SR): int((c["start"] + c["dur"]) * SR)],
                        np.float32)
        auds.append(D.mix_mono(ch))
    served = load_asr_model("served")
    head = TV.load_head(ROOT / "assets" / "tsvad_spk.pt")[0]
    hyps = {}
    for name in ("served", "0p6b"):
        m = served if name == "served" else load_asr_model(name)
        out = []
        for x in auds:
            xx, lens = m._pad([x])
            with torch.inference_mode():
                enc, elen, hidden = m.encode(xx, lens, ATT, return_hidden=True)
                e = m.head_input("rnnt", enc, hidden)
                out.append(words_of(m.tokenizer, greedy_timed(m.heads["rnnt"], e[0, : int(elen[0])])))
        hyps[name] = out
        if name != "served":
            del m
    lagf = {"served": json.loads((WORK / "ami" / "lag.json").read_text())["lag_frames"],
            "0p6b": json.loads((WORK / "ami" / "lag.json").read_text())["lag_0p6b_frames"]}
    units = []
    for i, c in enumerate(clips):
        f = TV.block_feats(served, [auds[i]])[0]
        T = len(f)
        pr = prints[c["name"]]["5.0"]
        ptgt = served_tsvad(head, served.heads["spk"], f, pr["embedding"])[:, 0]
        ref, segs = tb_user_words(c["conversation"], c["human_channel"], c["start"], c["start"] + c["dur"])
        orc = np.zeros(T, bool)
        for s0, e0 in segs:
            orc[int(s0 / FRAME_S): int(np.ceil(e0 / FRAME_S))] = True
        masks = {"tsvad": ptgt > 0.5, "oracle": orc}
        arms = {}
        for hn, pre in (("served", ""), ("0p6b", "0p6b_")):
            hw = hyps[hn][i]
            arms[f"{pre}none"] = [w[0] for w in hw]
            for mn, mk in masks.items():
                for dil in ((0, DIL, 4) if hn == "served" else (DIL,)):
                    kp = keep_mask(hw, mk, lagf[hn], dil)
                    arms[f"{pre}{mn}_d{dil}"] = [w[0] for w, q in zip(hw, kp) if q]
        u = {"key": c["name"], "meeting": c["conversation"], "col": 0, "N": len(ref), "arms": {}}
        for name, h in arms.items():
            s_, d_, i_, _, _ = align_counts([w[0] for w in ref], h)
            u["arms"][name] = [int(s_), int(d_), int(i_), int(len(h))]
        units.append(u)
        log(f"  {c['name']}: N {len(ref)} none {u['arms']['none']} tsvad {u['arms']['tsvad_d2']} "
            f"oracle {u['arms']['oracle_d2']}")
    (WORK / "turnbench").mkdir(parents=True, exist_ok=True)
    (WORK / "turnbench" / "lag.json").write_text(json.dumps({"lag": None, "lag_frames": lagf["served"],
                                                             "lag_0p6b_frames": lagf["0p6b"], "n_windows": len(clips),
                                                             "note": "lag taken from AMI (interpolated TurnBench word times)"}))
    (WORK / "turnbench" / "units.jsonl").write_text("".join(json.dumps({"key": u["key"], "units": [u]}) + "\n"
                                                            for u in units))
    log(f"  turnbench: {len(units)} clips -> {WORK / 'turnbench'}")


# --------------------------------------------------------------------------- the live suite (69 sessions)
LIVE_WORK = TV.SCRATCH / "tswer_live"


def _wav(p: Path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(p), dtype="float32")
    assert sr == SR, (p, sr)
    return x if x.ndim == 1 else x.mean(1)


def _counts(ref, hyp):
    """[S, D, I, hyp words] by jiwer; S + D + I is checked against audioforge.metrics.edit_distance (the scorer of
    e2e_final.score_record)."""
    from audioforge.metrics import edit_distance
    s_, d_, i_, _, _ = align_counts(ref, hyp)
    assert s_ + d_ + i_ == edit_distance(ref, hyp), "jiwer / edit_distance disagree"
    return [int(s_), int(d_), int(i_), int(len(hyp))]


def stage_live(a):
    """The live suite of runs/single_model.json table.live_69 (37 clips / 69 sessions, runs/e2e_final.json clip list,
    prepared under scratch/e2e_tsvad/clips): every session with a per-speaker reference, i.e. the 16 TurnBench clips x
    {mono mix, user channel}; the 5 AMI windows (rebuilt clips carry no transcript) and the 16 oto clips (no human
    transcripts) have none. Per session: served streaming words on the session's audio (the wav the live session
    played), TS-VAD on its block-4 features with the clip's stored 5 s print, oracle = the user's annotated segments;
    the same keep rule as the TurnBench stage. Also the live Pipecat finals of system S (the 23.2 % records)."""
    from audioforge.teachers import normalize_text
    TV.set_threads(2)
    clips = json.loads((E2E / "clips.json").read_text())
    prints = json.loads((E2E / "prints.json").read_text())
    live = {}
    for p in sorted((E2E / "runs").glob("pipecat_S_*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            live[(r["clip"], r["cond"])] = " ".join(f["text"] for f in r["raw"].get("finals", []))
    lagf = json.loads((WORK / "ami" / "lag.json").read_text())["lag_frames"]
    served = load_asr_model("served")
    head = TV.load_head(ROOT / "assets" / "tsvad_spk.pt")[0]
    sessions, excluded = [], []
    for c in clips:
        conds = ("mono", "user") if c["set"] != "ami" else ("mono",)
        if c["set"] != "turnbench" or not c.get("text_user"):
            excluded += [{"clip": c["name"], "set": c["set"], "cond": k,
                          "why": "no human transcript (otoSpeech)" if c["set"] == "oto" else
                          "rebuilt AMI window: no per-speaker (or any) reference in the clip list; not in the 23.2 %"}
                         for k in conds]
            continue
        ref, segs = tb_user_words(c["conversation"], c["human_channel"], c["start"], c["start"] + c["dur"])
        # the scored reference is clips.json text_user (the 23.2 % scorer's text); the interpolated words of the
        # TurnBench stage split two tokens differently ("2,000" in tb_42, "p.m." in tb_157: 1 word each)
        ref_interp = [w[0] for w in ref]
        ref = normalize_text(c["text_user"]).split()
        for cond in conds:
            x = _wav(E2E / "clips" / f"{c['name']}.{cond}.wav")
            xx, lens = served._pad([x])
            with torch.inference_mode():
                enc, elen, hidden = served.encode(xx, lens, ATT, return_hidden=True)
                e = served.head_input("rnnt", enc, hidden)
                hw = words_of(served.tokenizer, greedy_timed(served.heads["rnnt"], e[0, : int(elen[0])]))
            f = TV.block_feats(served, [x])[0]
            T = len(f)
            ptgt = served_tsvad(head, served.heads["spk"], f, prints[c["name"]]["5.0"]["embedding"])[:, 0]
            orc = np.zeros(T, bool)
            for s0, e0 in segs:
                orc[int(s0 / FRAME_S): int(np.ceil(e0 / FRAME_S))] = True
            full = normalize_text(c["text_mono"] if cond == "mono" else c["text_user"]).split()
            allw = [w[0] for w in hw]
            arms = {"none_full": _counts(full, allw), "none": _counts(ref, allw)}
            for mn, mk in (("tsvad", ptgt > 0.5), ("oracle", orc)):
                for dil in (0, DIL, 4):
                    kp = keep_mask(hw, mk, lagf, dil)
                    arms[f"{mn}_d{dil}"] = _counts(ref, [w[0] for w, q in zip(hw, kp) if q])
                    if dil == DIL:
                        arms[f"{mn}_d{dil}_interp_ref"] = _counts(ref_interp, [w[0] for w, q in zip(hw, kp) if q])
            lw = normalize_text(live.get((c["name"], cond), "")).split()
            arms["live_none_full"] = _counts(full, lw)
            arms["live_none"] = _counts(ref, lw)
            arms["none_interp_ref"] = _counts(ref_interp, allw)  # consistency with runs/tswer.json
            u = {"key": f"{c['name']}|{cond}", "clip": c["name"], "cond": cond, "N": len(ref), "N_full": len(full),
                 "N_interp": len(ref_interp),
                 "T": T, "tsvad_on_frac": round(float((ptgt > 0.5).mean()), 3), "oracle_on_frac": round(float(orc.mean()), 3),
                 "arms": arms}
            sessions.append(u)
            log(f"  {u['key']}: N {len(ref)} (full {len(full)}) none_full {arms['none_full']} none {arms['none']} "
                f"tsvad {arms['tsvad_d2']} oracle {arms['oracle_d2']} live {arms['live_none_full']}")
    LIVE_WORK.mkdir(parents=True, exist_ok=True)
    (LIVE_WORK / "sessions.json").write_text(json.dumps({"lag_frames": lagf, "sessions": sessions,
                                                         "excluded": excluded}, indent=1))
    log(f"  live: {len(sessions)} sessions scored, {len(excluded)} excluded -> {LIVE_WORK / 'sessions.json'}")


LIVE_ROWS = {"a": ("none_full", "all words, no filter, vs the full reference (both parties on mono; the user on the "
                                "user channel)"),
             "b": ("none", "all words, no filter, vs the user's words only"),
             "c": ("tsvad_d2", "our TS-VAD-filtered words (5 s stored print, P > 0.5, +-160 ms) vs the user's words"),
             "d": ("oracle_d2", "oracle filter (the user's annotated segments) vs the user's words")}


def _nkey(arm):
    return "N_full" if arm.endswith("_full") else "N_interp" if arm.endswith("_interp_ref") else "N"


def _live_boot(sess, arm, seed=0):
    """By-clip bootstrap (a clip's mono and user sessions move together)."""
    nk = _nkey(arm)
    cl = sorted({u["clip"] for u in sess})
    per = np.array([[sum(sum(u["arms"][arm][:3]) for u in sess if u["clip"] == k),
                     sum(u[nk] for u in sess if u["clip"] == k)] for k in cl], np.float64)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        s = per[rng.integers(0, len(cl), len(cl))].sum(0)
        d.append(s[0] / max(s[1], 1))
    return np.array(d), len(cl)


def _live_group(sess, arms):
    out = {"n_sessions": len(sess), "n_clips": len({u["clip"] for u in sess})}
    draws = {}
    for arm in arms:
        nk = _nkey(arm)
        c = np.array([u["arms"][arm] for u in sess], np.float64).sum(0)
        n = sum(u[nk] for u in sess)
        draws[arm], _ = _live_boot(sess, arm)
        out[arm] = {"wer": round(100 * c[:3].sum() / n, 2), "ci_clip": ci(draws[arm]),
                    "sub": round(100 * c[0] / n, 2), "del": round(100 * c[1] / n, 2), "ins": round(100 * c[2] / n, 2),
                    "S": int(c[0]), "D": int(c[1]), "I": int(c[2]), "ref_words": int(n), "hyp_words": int(c[3])}
    out["paired"] = {f"{x} - {y}": {"delta": round(out[x]["wer"] - out[y]["wer"], 2), "ci_clip": ci(draws[x] - draws[y])}
                     for x, y in (("tsvad_d2", "none"), ("tsvad_d2", "oracle_d2"), ("none", "none_full"))
                     if x in arms and y in arms}
    return out


def stage_live_report(a):
    z = json.loads((LIVE_WORK / "sessions.json").read_text())
    S = z["sessions"]
    arms = list(S[0]["arms"])
    res = {}
    for g, sel in (("all", lambda u: True), ("user_channel", lambda u: u["cond"] == "user"),
                   ("mono", lambda u: u["cond"] == "mono")):
        res[g] = _live_group([u for u in S if sel(u)], arms)
    # sensitivity: tb_160, the one clip where the track drops nearly all of the user's words (both sessions)
    for g, sel in (("all_without_tb_160", lambda u: True), ("user_channel_without_tb_160", lambda u: u["cond"] == "user"),
                   ("mono_without_tb_160", lambda u: u["cond"] == "mono")):
        res[g] = _live_group([u for u in S if sel(u) and u["clip"] != "tb_160"], ["none_full", "none", "tsvad_d2", "oracle_d2"])
    # consistency: the 16 TurnBench (mono) units of runs/tswer.json, same protocol
    tsw = json.loads((ROOT / "runs" / "tswer.json").read_text())["results"]["turnbench"]["primary"]["arms"]
    # (same interpolated-word reference as the TurnBench stage; the prepared mono wav vs the stage's re-mixed audio)
    cons = {arm: {"tswer_json": tsw[arm]["twer"], "here_mono": res["mono"][arm + "_interp_ref"]["wer"],
                  "tswer_json_SDIN": [tsw[arm][k] for k in ("S", "D", "I", "ref_words")],
                  "here_SDIN": [res["mono"][arm + "_interp_ref"][k] for k in ("S", "D", "I", "ref_words")]}
            for arm in ("none", "tsvad_d2", "oracle_d2")}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"),
           "question": "WER of the user's transcript on the live two-party sessions with the target-speaker filter on, "
                       "next to the published 23.2 % (runs/single_model.json table.live_69 single_S, all words, no filter)",
           "rows": {k: {"arm": v[0], "label": v[1]} for k, v in LIVE_ROWS.items()},
           "sessions_scored": {"n": len(S), "clips": sorted({u["clip"] for u in S})},
           "sessions_excluded": {"n": len(z["excluded"]), "list": z["excluded"]},
           "note_23p2": "the published 23.2 % = 818 / 3525 over the same 32 TurnBench sessions (16 mono vs both parties' "
                        "text, 16 user channel vs the user's text): the other 37 of the 69 sessions carry no reference "
                        "and add nothing to it. Row (a) here re-decodes the same audio offline (masked [70,1] forward); "
                        "'live_none_full' re-scores the stored live finals (reproduces 23.2 exactly).",
           "protocol": {"asr": "runs/stage1_served.afm rnnt, att [70,1], masked forward on the session wav (== cache-aware "
                               "streaming pass 1), greedy, emission frames recorded",
                        "filter": f"word kept if the mask (dilated +-{DIL} frames) is on at mean emission frame - "
                                  f"{z['lag_frames']} frames (the AMI median emission lag, 400 ms); d0 / d4 in the json",
                        "tsvad": "assets/tsvad_spk.pt on the session's block-4 features, stored 5 s print of the clip "
                                 "(scratch/e2e_tsvad/prints.json '5.0', the print the live sessions enrolled), P > 0.5",
                        "oracle": "annotator a's segments of the user's channel (80 ms frames)",
                        "reference": "user = annotator a's segments of the user's channel with midpoint in the clip, "
                                     "bracketed tags removed (== clips.json text_user), teachers.normalize_text; full = "
                                     "text_mono (both parties) on mono sessions, text_user on user-channel sessions",
                        "scoring": "(S + D + I) / N pooled over sessions; S / D / I by jiwer, total checked equal to "
                                   "audioforge.metrics.edit_distance (e2e_final.score_record); 95 % CI: "
                                   f"{N_BOOT} bootstrap resamples of clips (seed 0)",
                        "script": "scripts/research/tswer.py live / live_report"},
           "results": res, "consistency_turnbench_16_vs_runs_tswer_json": cons}
    p = ROOT / "runs" / "tswer_live.json"
    p.write_text(json.dumps(out, indent=1))
    log(f"[live_report] -> {p}")
    for g, r in res.items():
        log(f"\n{g}: {r['n_sessions']} sessions / {r['n_clips']} clips")
        for arm in [x for x in arms if x in r]:
            x = r[arm]
            log(f"  {arm:15s} WER {x['wer']:6.2f} {x['ci_clip']}  S {x['S']} D {x['D']} I {x['I']} N {x['ref_words']} "
                f"hyp {x['hyp_words']}")
        log(f"  paired {r['paired']}")
    log(f"\nconsistency: {cons}")


# --------------------------------------------------------------------------- report
ARM_LABELS = {
    "tsvad_d2": "ours: served RNNT words filtered by our TS-VAD track (5 s print, P > 0.5)",
    "oracle_d2": "oracle filter: the target's reference activity",
    "none": "no filter: every word",
    "n3_spk_d2": "Nemotron-3 column bound by the 5 s print (served speaker head)",
    "n3_tn_d2": "Nemotron-3 column bound by the 5 s print (TitaNet-L)",
    "0p6b_tsvad_d2": "nemotron-speech-streaming-en-0.6b [70,1] words filtered by our TS-VAD track",
    "0p6b_oracle_d2": "0.6B words, oracle filter",
    "0p6b_none": "0.6B words, no filter",
}


def agg(units, arm):
    c = np.array([u["arms"][arm][:3] + [u["N"], u["arms"][arm][3]] for u in units], np.float64)
    return c  # (n, 5) S, D, I, N, hyp words


def rate(c):
    s = c.sum(0)
    return (s[0] + s[1] + s[2]) / max(s[3], 1)


def boot(units, arms, by, seed=0):
    """Percentile 95 % CIs of tWER per arm and of paired differences (same resamples), clusters = meeting or window."""
    rng = np.random.default_rng(seed)
    cl = sorted({u[by] for u in units})
    idx = {k: [i for i, u in enumerate(units) if u[by] == k] for k in cl}
    C = {a: agg(units, a) for a in arms}
    per = {a: np.stack([C[a][idx[k]].sum(0) for k in cl]) for a in arms}  # cluster sums
    draws = {a: [] for a in arms}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(cl), len(cl))
        for a in arms:
            s = per[a][pick].sum(0)
            draws[a].append((s[0] + s[1] + s[2]) / max(s[3], 1))
    return {a: np.array(d) for a, d in draws.items()}, len(cl)


def ci(x):
    return [round(100 * float(np.percentile(x, 2.5)), 2), round(100 * float(np.percentile(x, 97.5)), 2)]


def context_wer(corpus):
    """ASR ceiling on the same windows: every speaker's words (time order) vs every decoded word, per ASR model."""
    ext, meta, ds = TV.bench_windows(corpus)
    out = {}
    for name in ("served", "0p6b"):
        asr = load_asr(corpus, name)
        if not all(TV.wkey(v) in asr for v in ext):
            continue
        tot = np.zeros(4)
        for v in ext:
            refs = sorted([w for s_ in TV.window_speakers(ds, v) for w in ref_words(ds, v, s_)], key=lambda x: x[1])
            s_, d_, i_, _, _ = align_counts([w[0] for w in refs], [w[0] for w in asr[TV.wkey(v)]["words"]])
            tot += [s_, d_, i_, len(refs)]
        out[name] = {"wer": round(100 * tot[:3].sum() / tot[3], 2), "sub": round(100 * tot[0] / tot[3], 2),
                     "del": round(100 * tot[1] / tot[3], 2), "ins": round(100 * tot[2] / tot[3], 2),
                     "ref_words": int(tot[3])}
    return out


def stage_report(a):
    out = {"definition": {
        "tWER": "(S + D + I) / N summed over (window, target) units; reference = the target's words (annotation word "
                "times, midpoint in the window), hypothesis = the arm's kept streaming words; teachers.normalize_text",
        "units": "eot-bench v2 extended windows (AMI dev 974, ICSI held-out 1312) x column speakers who speak in the "
                 "window and have a 5 s voice print (tsvad.py vprints, as IMPROVE_115M A.1 / A.1b); primary = column 0",
        "asr": "runs/stage1_served.afm rnnt head, att [70,1], masked offline forward on the window audio (== cache-aware "
               "streaming pass 1 of --mode single), greedy, token emission frames recorded",
        "word_time": "mean emission frame of the word's tokens minus the corpus' median emission lag (measured target-"
                     "agnostically on hits vs all speakers' reference words); kept if the arm's mask, dilated by +-DIL "
                     f"frames, is on there (headline DIL = {DIL} = 160 ms; 0 / 4 in 'dilation')",
        "tsvad": "assets/tsvad_spk.pt (== runs/tsvad_spk.pt), the served track (tsvad_stream.track_probs: TSVADTrack, "
                 "print set at frame 0, anchored adaptation) on the cached block-4 window features, "
                 "P(target) > 0.5",
        "nemotron3": "cached A.1b Nemotron-3 tracks (served settings low_latency_032), tsvad.vp_follow on the same 5 s print "
                     "(served speaker head ColumnEmbedder / cached TitaNet-L column embeddings), p > 0.5",
        "bootstrap": f"{N_BOOT} resamples (seed 0) of meetings (AMI dev 4, ICSI 5: coarse) and of windows; paired deltas",
        "turnbench": "the 16 TurnBench dev clips of IMPROVEMENTS.md section 1 (scratch/e2e_tsvad clips.json / prints.json): "
                     "mono mix, target = the human party, stored 5 s print from the user's own channel; reference = "
                     "annotator a's segments of the user's channel (word times interpolated inside segments); oracle = "
                     "those segments; emission lag from AMI; bootstrap by clip; no Nemotron-3 tracks for these clips",
        "oto": "not scored: otoSpeech has no human transcripts (DYADIC.md), only our own decodes of each channel",
        "script": "scripts/research/tswer.py asr / score / turnbench / report"}, "results": {}}
    for corpus in a.corpora.split(","):
        z = json.loads((WORK / corpus / "lag.json").read_text())
        rows_ = [json.loads(x) for x in (WORK / corpus / "units.jsonl").read_text().splitlines() if x.strip()]
        assert len({r["key"] for r in rows_}) == z["n_windows"], "score stage not finished"
        U = [u for r in rows_ for u in r["units"]]
        ctx = context_wer(corpus) if corpus != "turnbench" else None
        res = {"context_all_speaker_window_wer": ctx, "lag": z["lag"], "lag_frames": z["lag_frames"], "lag_0p6b": z.get("lag_0p6b"), "n_windows": z["n_windows"]}
        groups = (("primary", lambda u: u["col"] == 0), ("all_speakers", lambda u: True))
        for g, sel in groups[: 1 if corpus == "turnbench" else 2]:
            units = [u for u in U if sel(u)]
            arms = [x for x in units[0]["arms"]]
            head_arms = [x for x in ARM_LABELS if x in arms]
            bm, nm = boot(units, head_arms, "meeting")
            bw, nw = boot(units, head_arms, "key")
            rows = {}
            for arm in arms:
                c = agg(units, arm).sum(0)
                r = {"twer": round(100 * (c[0] + c[1] + c[2]) / max(c[3], 1), 2),
                     "sub": round(100 * c[0] / c[3], 2), "del": round(100 * c[1] / c[3], 2),
                     "ins": round(100 * c[2] / c[3], 2), "S": int(c[0]), "D": int(c[1]), "I": int(c[2]),
                     "ref_words": int(c[3]), "hyp_words": int(c[4])}
                if arm in head_arms:
                    r["ci_meeting"], r["ci_window"] = ci(bm[arm]), ci(bw[arm])
                    r["label"] = ARM_LABELS[arm]
                rows[arm] = r
            pairs = {}
            for x, y in (("tsvad_d2", "none"), ("tsvad_d2", "n3_spk_d2"), ("tsvad_d2", "n3_tn_d2"),
                         ("tsvad_d2", "oracle_d2"), ("0p6b_tsvad_d2", "tsvad_d2")):
                if x in head_arms and y in head_arms:
                    pairs[f"{x} - {y}"] = {"delta": round(rows[x]["twer"] - rows[y]["twer"], 2),
                                           "ci_meeting": ci(bm[x] - bm[y]), "ci_window": ci(bw[x] - bw[y])}
            res[g] = {"n_units": len(units), "n_meetings": nm, "n_windows": nw, "arms": rows, "paired": pairs}
        out["results"][corpus] = res
    p = ROOT / "runs" / "tswer.json"
    p.write_text(json.dumps(out, indent=1))
    log(f"[report] -> {p}")
    for corpus, res in out["results"].items():
        for g in [x for x in ("primary", "all_speakers") if x in res]:
            log(f"\n{corpus} {g}: units {res[g]['n_units']}")
            for arm, r in res[g]["arms"].items():
                log(f"  {arm:16s} tWER {r['twer']:6.2f}  S {r['sub']:5.2f} D {r['del']:5.2f} I {r['ins']:6.2f}  "
                    f"N {r['ref_words']} hyp {r['hyp_words']}  {r.get('ci_meeting', '')} {r.get('ci_window', '')}")
            for k, d in res[g]["paired"].items():
                log(f"  {k}: {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("asr", "score", "turnbench", "report", "live", "live_report", "tsvad_arms"))
    ap.add_argument("--corpora", default="ami,icsi", help="report: ami,icsi,turnbench")
    ap.add_argument("--asr", default="served", choices=tuple(ASR_MODELS))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--budget", type=float, default=540)
    a = ap.parse_args()
    {"asr": stage_asr, "score": stage_score, "turnbench": stage_turnbench, "report": stage_report,
     "live": stage_live, "live_report": stage_live_report, "tsvad_arms": stage_tsvad_arms}[a.stage](a)


if __name__ == "__main__":
    main()
