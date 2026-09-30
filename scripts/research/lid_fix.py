"""LID fix pass (research/archive/LID.md, "Fix pass, 2026-09-29"): diagnose the head, scale the data, distil AmberNet into a
head on the shared encoder, streaming decision rule, fusion fallback.

Data (all on the SSD through the data/ symlinks):
  fetch_more   the rest of every FLEURS-17 train tarball beyond the 250 utterances research/archive/LID.md used
               (split "trainx", data/lid/fleurs/<lang>/trainx, manifest data/lid/fleurs/manifest_trainx.jsonl).
               FLEURS train / dev / test are speaker-disjoint, so dev / test stay untouched. Resumable per language.
  extra_en     accented / conversational English for the English class: LibriSpeech train-clean-100 utterances,
               AMI and ICSI single-speaker segments (manifest data/lid/extra_en.jsonl).

Stages (every model-loading call goes through scripts/dev/gate.sh; each call stops at --budget and is resumable):
  onsets_more  Silero onsets for the new rows (CPU).
  feats        frozen served encoder (MPS or CPU) over a fixed crop set: pooled per-block statistics for the probes and
               frame features of the tapped blocks for head training -> SSD cache.
  teacher      AmberNet posteriors (107 languages, logits) on the same crops -> SSD cache.
  probe        per-block linear probes (2 s / 5 s), data-scaling curve.
  train        head training on cached features: CE to the label + KL to AmberNet (temperature T), feature-level
               augmentation; checkpoints on the SSD, read back after writing.
  eval         the head on the research/archive/LID.md clips (FLEURS-17 test 1/2/3/5 s/full, EdAcc), CIs.
  rule         streaming announcement rule (first-announced language accuracy, time to announce).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import lid_data as D  # noqa: E402
from lid_data import LANGS, PEAK  # noqa: E402

SR = 16000
CODES = list(LANGS)
TRAINX_MANIFEST = D.OUT / "manifest_trainx.jsonl"
EXTRA_EN = ROOT / "data/lid/extra_en.jsonl"
SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
FIX_CACHE = SSD / "cache/lid_fix"
FIX_RUNS = SSD / "runs/lid_fix"
JSON = ROOT / "runs/lid.json"


def _json() -> dict:
    return json.loads(JSON.read_text()) if JSON.exists() else {}


def merge_fix(key: str, rec):
    """runs/lid.json["fix_2026_09_29"][key] = rec."""
    res = _json()
    res.setdefault("fix_2026_09_29", {})[key] = rec
    JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False))


def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


# --------------------------------------------------------------------------- data: the rest of FLEURS train
def fetch_more_one(lang: str) -> int:
    import requests
    import soundfile as sf
    from huggingface_hub import get_token, hf_hub_url
    fl = LANGS[lang]
    done = D.OUT / lang / ".trainx.done"
    if done.exists():
        return 0
    have = {r["id"] for r in D.read_manifest("train", [lang])}
    meta = D._tsv(fl, "train")
    url = hf_hub_url(D.REPO, f"data/{fl}/audio/train.tar.gz", repo_type="dataset")
    tok = get_token()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    d = D.OUT / lang / "trainx"
    d.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile() or not m.name.endswith(".wav"):
                    continue
                name = Path(m.name).name
                uid = name.removesuffix(".wav")
                if uid in have:
                    continue
                a, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
                assert sr == 16000, (name, sr)
                if a.ndim > 1:
                    a = a[:, 0]
                path = d / f"{uid}.flac"
                pk = float(np.abs(a).max()) if len(a) else 0.0
                sf.write(path, a * (PEAK / pk) if pk > 0 else a, 16000, format="FLAC", subtype="PCM_16")
                info = meta.get(name, {})
                rows.append({"lang": lang, "fleurs": fl, "split": "trainx", "id": uid,
                             "path": str(path.relative_to(ROOT)), "duration": round(len(a) / 16000, 3),
                             "gender": info.get("gender"), "sent_id": info.get("sent_id")})
    with TRAINX_MANIFEST.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    done.write_text(str(len(rows)))
    print(f"[fetch_more] {lang}: {len(rows)} more utts, {sum(r['duration'] for r in rows) / 3600:.2f} h, "
          f"{time.time() - t0:.0f}s", flush=True)
    return len(rows)


def stage_fetch_more(a):
    t0 = time.time()
    for lang in a.langs or CODES:
        if time.time() - t0 > a.budget:
            print("[fetch_more] budget reached; re-run", flush=True)
            return
        fetch_more_one(lang)
    print("[fetch_more] all languages done", flush=True)



# --------------------------------------------------------------------------- data: extra (accented / conversational) English
def stage_extra_en(a):
    """Single-speaker English segments of the SPK_HEAD teacher sets (data/cache/titanet/*.npz ids): AMI train (930,
    many non-native speakers), ICSI train (3249) and ``--n-libri`` LibriSpeech train-clean-100 utterances; cut,
    capped at 15 s, peak-normalised FLAC under data/lid/extra_en; manifest data/lid/extra_en.jsonl (lang en)."""
    import soundfile as sf
    if EXTRA_EN.exists():
        print("[extra_en] done already", flush=True)
        return
    out = ROOT / "data/lid/extra_en"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for corpus in ("ami", "icsi"):
        ids = np.load(ROOT / f"data/cache/titanet/{corpus}_train.npz")["ids"].tolist()
        by_m: dict[str, list] = {}
        for i in ids:
            m, st, spk, n = str(i).split(":")
            by_m.setdefault(m, []).append((float(st), spk, int(n)))
        for m, segs in sorted(by_m.items()):
            wav = ROOT / f"data/{corpus}/audio/{m}.Mix-Headset.wav"
            info = sf.info(wav)
            for st, spk, n in segs:
                n = min(n, 15 * SR)
                x, sr = sf.read(wav, start=int(round(st * SR)), frames=n, dtype="float32")
                assert sr == SR
                if x.ndim > 1:
                    x = x.mean(1)
                if len(x) < SR:
                    continue
                pk = float(np.abs(x).max())
                uid = f"{corpus}_{m}_{st:.2f}_{spk}"
                path = out / f"{uid}.flac"
                sf.write(path, x * (PEAK / pk) if pk > 0 else x, SR, format="FLAC", subtype="PCM_16")
                rows.append({"lang": "en", "split": "extra_en", "corpus": corpus, "id": uid, "speaker": f"{corpus}_{spk}",
                             "path": str(path.relative_to(ROOT)), "duration": round(len(x) / SR, 3)})
            del info
        print(f"[extra_en] {corpus}: {sum(r['corpus'] == corpus for r in rows)} segments", flush=True)
    lib = np.load(ROOT / "data/cache/titanet/librispeech_train.npz")["ids"].tolist()[: a.n_libri]
    for uid in lib:
        spk, ch, _ = uid.split("-")
        src = ROOT / f"data/librispeech/LibriSpeech/train-clean-100/{spk}/{ch}/{uid}.flac"
        x, sr = sf.read(src, dtype="float32")
        x = x[: 15 * SR]
        pk = float(np.abs(x).max())
        path = out / f"libri_{uid}.flac"
        sf.write(path, x * (PEAK / pk) if pk > 0 else x, SR, format="FLAC", subtype="PCM_16")
        rows.append({"lang": "en", "split": "extra_en", "corpus": "librispeech", "id": f"libri_{uid}",
                     "speaker": f"libri_{spk}", "path": str(path.relative_to(ROOT)), "duration": round(len(x) / SR, 3)})
    EXTRA_EN.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for c in ("ami", "icsi", "librispeech"):
        rs = [r for r in rows if r["corpus"] == c]
        print(f"[extra_en] {c}: {len(rs)} segments, {sum(r['duration'] for r in rs) / 3600:.2f} h, "
              f"{len({r['speaker'] for r in rs})} speakers", flush=True)


# --------------------------------------------------------------------------- encoder feature cache (SSD)
SETS = ("train", "trainx", "extra_en", "dev", "test", "edacc")
POOL_WIN = ("1s", "2s", "3s", "5s", "all")  # pooled-mean windows from the onset frame (probes)
FRAME = 1280  # samples per encoder frame (80 ms)
SHARD = 500


def set_rows(set_: str) -> list[dict]:
    if set_ in ("train", "dev", "test"):
        return D.read_manifest(set_)
    if set_ == "trainx":
        return read_rows(TRAINX_MANIFEST)
    if set_ == "extra_en":
        return read_rows(EXTRA_EN)
    if set_ == "edacc":
        return D.read_edacc()
    raise SystemExit(f"unknown set {set_}")


def parse_blocks(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def shard_dir(set_: str, view: str) -> Path:
    return FIX_CACHE / set_ / view


def _aug_audio(x: np.ndarray, rng) -> tuple[np.ndarray, float]:
    """Speed perturbation 0.9 / 0.95 / 1 / 1.05 / 1.1 (resampling; shards 0-5 of the train set used a continuous
    0.9-1.1 factor, 12x slower), coloured noise at 5-30 dB SNR (white / pink / brown), random
    first-order tilt (telephone-ish or bass-heavy), gain -12..0 dB. -> (audio, speed factor)."""
    import torch
    import torchaudio
    num, den = [(9, 10), (19, 20), (1, 1), (21, 20), (11, 10)][int(rng.integers(0, 5))]  # small ratios: fast polyphase
    s = num / den
    y = (torchaudio.functional.resample(torch.from_numpy(x), num, den).numpy() if num != den else x.copy())
    if rng.random() < 0.3:  # spectral tilt
        k = float(rng.uniform(-0.9, 0.9))
        y = np.concatenate([y[:1], y[1:] - k * y[:-1]]).astype(np.float32)
    kind = rng.integers(0, 3)
    n = rng.standard_normal(len(y)).astype(np.float32)
    if kind > 0:  # pink / brown via spectral shaping
        f = np.fft.rfft(n)
        fr = np.arange(len(f)) + 1.0
        f = f / (fr ** (0.5 * kind))
        n = np.fft.irfft(f, len(n)).astype(np.float32)
    snr = float(rng.uniform(5, 30))
    ps = float(np.mean(y ** 2)) + 1e-9
    n *= np.sqrt(ps / (float(np.mean(n ** 2)) + 1e-12) / 10 ** (snr / 10))
    y = y + n
    y = y / (np.abs(y).max() + 1e-9) * PEAK * float(10 ** (rng.uniform(-12, 0) / 20))
    return y.astype(np.float32), s


def _view_audio(r: dict, view: str, onsets: dict, cap_s: float, rng):
    """(audio, offset_samples, speed) of row ``r`` for ``view``: 'on' = from the Silero onset - 0.1 s (the
    research/archive/LID.md clips), 'full' = from the file start, 'aug' = 'full' augmented (_aug_audio)."""
    x = D.load_audio(r)
    off, s = 0, 1.0
    if view in ("on", "aug"):  # the augmented view is the 'on' view augmented (frames map by the speed factor)
        off = int(max(0.0, onsets[r["id"]][0] - 0.1) * SR)
        x = x[off:]
    x = x[: int(cap_s * SR)]
    if view == "aug":
        x, s = _aug_audio(x, rng)
    return x, off, s


def _onset_frame(vad: np.ndarray) -> int:
    """First of 3 consecutive frames with VAD > 0.5, minus 1 (~0.1 s pre-roll); 0 if none."""
    on = vad > 0.5
    run = np.nonzero(on[:-2] & on[1:-1] & on[2:])[0] if len(on) >= 3 else np.nonzero(on)[0]
    return max(0, int(run[0]) - 1) if len(run) else 0


def stage_feats(a):
    """Frozen served encoder over one (set, view): per shard of 500 rows -> <SSD>/cache/lid_fix/<set>/<view>/
    s####.npy (frames x blocks x 512, float16, the --blocks taps), s####_pool.npy (rows x 5 windows x 17 blocks x
    512, float16: mean over the first 1/2/3/5 s from the onset frame and over all frames), s####.json (ids, offsets,
    n_frames, onset frame, speed, VAD per frame). Every shard is read back after writing (exFAT)."""
    import torch

    from audioforge.train import load_model
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    blocks = parse_blocks(a.blocks) if a.blocks != "none" else []
    rows = set_rows(a.set)
    ons = {}
    if a.view in ("on", "aug"):
        ons = json.loads((ROOT / "data/lid/fleurs/onsets.json").read_text())
        full = shard_dir(a.set, "full")  # rows without a Silero onset: the served VAD head's onset of the full pass
        for f in sorted(full.glob("s[0-9]*.json")):
            m = json.loads(f.read_text())
            for rid, o in zip(m["ids"], m["onset"]):
                ons.setdefault(rid, [o * 0.08 + 0.1])
    cap = a.cap or (5.3 if a.set in ("dev", "test", "edacc") and a.view == "on" else 20.0)
    d = shard_dir(a.set, a.view)
    d.mkdir(parents=True, exist_ok=True)
    model = load_model(ROOT / "runs/stage1_served.afm", "cpu").eval()
    if a.view == "aug":
        from audioforge.features import SpecAugment
        model.spec_augment = SpecAugment(freq_masks=2, freq_width=15, time_masks=4, time_width=0.05)
    model = model.to(dev)
    t0 = time.time()
    n_sh = (len(rows) + SHARD - 1) // SHARD
    for si in range(0, n_sh, a.stride):
        f_meta = d / f"s{si:04d}.json"
        if f_meta.exists():
            continue
        if time.time() - t0 > a.budget:
            print(f"[feats] {a.set}/{a.view}: budget reached at shard {si}/{n_sh}; re-run", flush=True)
            return
        rs = rows[si * SHARD: (si + 1) * SHARD]
        rng = np.random.default_rng(12345 + si)
        auds = [_view_audio(r, a.view, ons, cap, rng) for r in rs]
        order = sorted(range(len(rs)), key=lambda i: len(auds[i][0]))
        per = [None] * len(rs)
        i = 0
        while i < len(order):  # batches up to ~a.batch_sec seconds of padded audio
            j = i
            while j < len(order) and (j - i + 1) * len(auds[order[j]][0]) <= a.batch_sec * SR:
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            lens = torch.tensor([len(auds[k][0]) for k in idx])
            x = torch.zeros(len(idx), int(lens.max()))
            for q, k in enumerate(idx):
                x[q, : lens[q]] = torch.from_numpy(auds[k][0])
            with torch.no_grad():
                model.train(a.view == "aug")
                enc, elen, hid = model.encode(x.to(dev), lens.to(dev), return_hidden=True)
                model.eval()
                vad = model.heads["vad"](model.head_input("vad", enc, hid)).sigmoid().float().cpu().numpy()
                H = torch.stack(hid, 2)  # (B, T, 17, D)
                Hc = H.half().cpu().numpy()
            for q, k in enumerate(idx):
                n = int(elen[q])
                v = vad[q, :n]
                o = 0 if a.view in ("on", "aug") else _onset_frame(v)
                pools = []
                for w in POOL_WIN:
                    e = n if w == "all" else min(n, o + int(round(float(w[:-1]) / 0.08)))
                    seg = H[q, o:max(e, o + 1)].float().mean(0)
                    pools.append(seg)
                per[k] = {"frames": Hc[q, :n][:, [b - 1 for b in blocks]] if blocks else np.zeros((n, 0, 512), np.float16), "pool": torch.stack(pools).half().cpu().numpy(),
                          "vad": v.astype(np.float16), "onset": o, "n": n}
            i = j
        offs = np.cumsum([0] + [p["n"] for p in per])
        F_ = np.concatenate([p["frames"] for p in per])
        P_ = np.stack([p["pool"] for p in per])
        np.save(d / f"s{si:04d}.npy", F_)
        np.save(d / f"s{si:04d}_pool.npy", P_)
        chk = np.load(d / f"s{si:04d}.npy", mmap_mode="r")
        assert chk.shape == F_.shape and np.array_equal(chk[:: max(1, len(F_) // 997)], F_[:: max(1, len(F_) // 997)])
        assert np.array_equal(np.load(d / f"s{si:04d}_pool.npy"), P_)
        meta = {"ids": [r["id"] for r in rs], "lang": [r["lang"] for r in rs], "offsets": offs.tolist(),
                "n": [p["n"] for p in per], "onset": [p["onset"] for p in per], "speed": [auds[k][2] for k in range(len(rs))],
                "audio_offset": [auds[k][1] for k in range(len(rs))], "blocks": blocks, "view": a.view,
                "vad": [p["vad"].tolist() for p in per]}
        f_meta.write_text(json.dumps(meta))
        assert json.loads(f_meta.read_text())["offsets"] == meta["offsets"]
        print(f"[feats] {a.set}/{a.view} shard {si + 1}/{n_sh}: {len(rs)} rows, {offs[-1]} frames, "
              f"{time.time() - t0:.0f}s", flush=True)
    print(f"[feats] {a.set}/{a.view} all shards done", flush=True)


def load_pool(set_: str, view: str):
    """(pooled means (N, 5, 17, D) float16, labels (N,)) of a cached (set, view)."""
    d = shard_dir(set_, view)
    P, y = [], []
    for f in sorted(d.glob("s*.json")):
        m = json.loads(f.read_text())
        P.append(np.load(d / f"{f.stem}_pool.npy"))
        y += [CODES.index(l) for l in m["lang"]]
    return np.concatenate(P), np.array(y)


# --------------------------------------------------------------------------- AmberNet teacher on training windows
AMBER_CODES = None


def amber_index(model) -> list[int]:
    """Indices of the 17 CODES in AmberNet's 107 labels (Hebrew is 'iw')."""
    from audioforge.baselines.lid import AMBERNET_ALIASES
    return [model.labels.index(AMBERNET_ALIASES.get(c, c)) for c in CODES]


def windows_for(rid: str, o: int, n: int, k_rand: int, with_all: bool = True) -> list[tuple[int, int]]:
    """Training windows (frame ranges [f0, f1)) of one cached row: onset-anchored 1 / 2 / 3 / 5 s, everything from the
    onset (``with_all``), and ``k_rand`` random windows of 1-5 s (seeded by the row id)."""
    import zlib
    rng = np.random.default_rng(zlib.crc32(rid.encode()))
    out = []
    for L in (1.0, 2.0, 3.0, 5.0):
        f1 = min(n, o + int(round(L / 0.08)))
        if f1 - o >= 6:
            out.append((o, f1))
    if with_all and n - o >= 6:
        out.append((o, n))
    for _ in range(k_rand):
        Lf = int(round(float(rng.choice(np.arange(1.0, 5.01, 0.5))) / 0.08))
        if n - o < Lf:
            Lf = n - o
        if Lf < 6:
            continue
        s0 = int(rng.integers(o, n - Lf + 1))
        out.append((s0, s0 + Lf))
    return list(dict.fromkeys(out))


def stage_teacher(a):
    """AmberNet logits (107 VoxLingua107 classes, float16) on the training windows of a cached (set, 'full') view ->
    <SSD>/cache/lid_fix/<set>/full/t####.npz (row index in shard, f0, f1, logits); read back after writing.
    Window audio = samples [f0 * 1280, f1 * 1280) of the peak-normalised file (the frames' own span)."""
    import torch

    from audioforge.baselines.lid import load_ambernet
    from audioforge.nemo_import import AMBERNET_NEMO, import_ambernet
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    if not AMBERNET_NEMO.exists():
        import_ambernet()
    amb = load_ambernet(AMBERNET_NEMO, fast=dev.type == "cpu").to(dev).eval()  # MPS: native grouped conv is faster
    lab_file = FIX_CACHE / "ambernet_labels.json"
    if not lab_file.exists():
        lab_file.write_text(json.dumps(amb.labels))
    rows = {r["id"]: r for r in set_rows(a.set)}
    d = shard_dir(a.set, "on")
    t0 = time.time()
    metas = sorted(d.glob("s[0-9]*.json"))
    for f in metas:
        out = d / f"t{f.stem[1:]}.npz"
        if out.exists():
            continue
        if time.time() - t0 > a.budget:
            print(f"[teacher] {a.set}: budget reached at {f.stem}; re-run", flush=True)
            return
        m = json.loads(f.read_text())
        W = []  # (row index, f0, f1)
        auds = {}
        for i, rid in enumerate(m["ids"]):
            for f0, f1 in windows_for(rid, m["onset"][i], m["n"][i], a.k_rand, with_all=False):
                W.append((i, f0, f1))
        order = sorted(range(len(W)), key=lambda k: W[k][2] - W[k][1])
        logits = np.zeros((len(W), 107), np.float16)
        q = 0
        while q < len(order):
            L = (W[order[q]][2] - W[order[q]][1]) * FRAME
            nb = max(1, int(a.batch_sec * SR // max(L, 1)))
            idx = order[q: q + nb]
            clips = []
            for k in idx:
                i, f0, f1 = W[k]
                if i not in auds:
                    ao = m["audio_offset"][i]
                    auds[i] = D.load_audio(rows[m["ids"][i]])[ao: ao + int(20.0 * SR)]
                clips.append(auds[i][f0 * FRAME: f1 * FRAME])
            lens = torch.tensor([len(c) for c in clips])
            x = torch.zeros(len(clips), int(lens.max()))
            for j, c in enumerate(clips):
                x[j, : len(c)] = torch.from_numpy(c)
            with torch.no_grad():
                z = amb(x.to(dev), lens.to(dev)).float().cpu().numpy()
            logits[idx] = z.astype(np.float16)
            q += nb
        Wa = np.array(W, np.int32)
        np.savez(out, w=Wa, logits=logits)
        chk = np.load(out)
        assert np.array_equal(chk["w"], Wa) and np.array_equal(chk["logits"], logits)
        ai = amber_index(amb)
        y = np.array([CODES.index(m["lang"][i]) for i, _, _ in W])
        acc = float((logits[:, ai].astype(np.float32).argmax(1) == y).mean())
        print(f"[teacher] {a.set} {f.stem}: {len(W)} windows, AmberNet acc (17-way) {acc:.3f}, {time.time() - t0:.0f}s",
              flush=True)
    print(f"[teacher] {a.set} all shards done", flush=True)


# --------------------------------------------------------------------------- diagnosis: probes, confusions, data scaling
def _logreg(Xtr, ytr, Xte, C=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def boot_ci(ok: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[float]:
    ok = np.asarray(ok, np.float64)
    bs = np.random.default_rng(seed).integers(0, len(ok), (n_boot, len(ok)))
    m = ok[bs].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def stage_probe(a):
    """Linear probe per encoder block (1-17) and window (1 / 2 / 3 / 5 s from the onset, all frames): mean-pooled
    block output, standardise + logistic regression (C = 0.1), fitted on FLEURS train (the 250 per language of
    research/archive/LID.md, 'on' view: encoder started at the onset, as the test clips), scored on the FLEURS-17 test clips ('on' view,
    = the research/archive/LID.md clips) and EdAcc (fraction called en). --scale: the same probe on the best blocks with
    50 / 125 / 250 / 1000 / all train utterances per language (train + trainx)."""
    Ptr, ytr = load_pool("train", "on")
    Pte, yte = load_pool("test", "on")
    Pte_full, _ = load_pool("test", "full")
    Ped, _ = load_pool("edacc", "on")
    rec = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "C": 0.1, "per_block": []}
    blocks = [int(b) for b in (a.rest[0].split(",") if a.rest else range(1, 18))]
    for b in blocks:
        row = {"block": b}
        for wi, w in enumerate(POOL_WIN):
            Xtr = Ptr[:, wi, b - 1].astype(np.float32)
            te = (Pte_full if w == "all" else Pte)[:, wi, b - 1].astype(np.float32)
            pred = _logreg(Xtr, ytr, np.concatenate([te, Ped[:, wi, b - 1].astype(np.float32)]))
            pt, pe = pred[: len(yte)], pred[len(yte):]
            row[w] = {"acc": round(float((pt == yte).mean()), 4), "ci95": boot_ci(pt == yte),
                      "edacc_en": round(float((pe == 0).mean()), 4)}
        rec["per_block"].append(row)
        print(f"block {b:2d}: " + " ".join(f"{w} {100 * row[w]['acc']:.1f}" for w in POOL_WIN)
              + f" | edacc 2s {100 * row['2s']['edacc_en']:.1f} all {100 * row['all']['edacc_en']:.1f}", flush=True)
    if not a.rest:
        merge_fix("probe", rec)


def stage_probe_scale(a):
    """Data scaling: linear probe (block(s) ``--blocks``, concatenated means) fitted on the first n train utterances
    per language, n = 250 / 1000 / all (train then trainx), scored on the test clips (2 s, all) and
    EdAcc."""
    blocks = parse_blocks(a.blocks)
    P1, y1 = load_pool("train", "on")
    P2, y2 = load_pool("trainx", "on")
    Ptr, ytr = np.concatenate([P1, P2]), np.concatenate([y1, y2])
    Pte, yte = load_pool("test", "on")
    Pte_full, _ = load_pool("test", "full")
    Ped, _ = load_pool("edacc", "on")
    bi = [b - 1 for b in blocks]
    rec = {"blocks": blocks, "rows": []}
    for n in (250, 1000, 100000):
        sel = np.concatenate([np.nonzero(ytr == c)[0][:n] for c in range(len(CODES))])
        row = {"n_per_lang": int(min(n, np.bincount(ytr).max())), "n_train": int(len(sel))}
        for w in ("2s", "all"):
            wi = POOL_WIN.index(w)
            X = Ptr[sel][:, wi][:, bi].reshape(len(sel), -1).astype(np.float32)
            te = (Pte_full if w == "all" else Pte)[:, wi][:, bi].reshape(len(yte), -1).astype(np.float32)
            ed = Ped[:, wi][:, bi].reshape(len(Ped), -1).astype(np.float32)
            pred = _logreg(X, ytr[sel], np.concatenate([te, ed]))
            pt, pe = pred[: len(yte)], pred[len(yte):]
            row[w] = {"acc": round(float((pt == yte).mean()), 4), "ci95": boot_ci(pt == yte),
                      "edacc_en": round(float((pe == 0).mean()), 4)}
        rec["rows"].append(row)
        print(row, flush=True)
    merge_fix(f"probe_scale_b{'_'.join(map(str, blocks))}", rec)


# --------------------------------------------------------------------------- head training on cached features (MPS)
class Shards:
    """Memory-mapped frame caches of one (set, view): frames(shard, row, f0, f1, blocks) -> (n, len(blocks), D)."""

    def __init__(self, set_: str, view: str):
        self.d = shard_dir(set_, view)
        self.metas = {int(f.stem[1:]): json.loads(f.read_text()) for f in sorted(self.d.glob("s[0-9]*.json"))}
        self.mm = {}
        if set_ in ("train", "trainx", "extra_en"):  # per-frame VAD is only needed for the evaluation sets
            for m in self.metas.values():
                m.pop("vad", None)
        self.blocks = next(iter(self.metas.values()))["blocks"] if self.metas else []

    def arr(self, si):
        if si not in self.mm:
            self.mm[si] = np.load(self.d / f"s{si:04d}.npy", mmap_mode="r")
        return self.mm[si]

    def frames(self, si, i, f0, f1, bidx):
        o = self.metas[si]["offsets"][i]
        return self.arr(si)[o + f0: o + f1][:, bidx]


def build_windows(sets, k_rand, need_teacher: bool):
    """Window table over training sets: (set, shard, row, f0, f1, label, teacher logits (17, float32) or None)."""
    import zlib  # noqa: F401

    from audioforge.baselines.lid import AMBERNET_ALIASES
    amb_labels = None
    lab_file = FIX_CACHE / "ambernet_labels.json"
    if lab_file.exists():
        amb_labels = json.loads(lab_file.read_text())
    ai = [amb_labels.index(AMBERNET_ALIASES.get(c, c)) for c in CODES] if amb_labels else None
    W, T = [], []
    for set_ in sets:
        sh = Shards(set_, "on")
        for si, m in sh.metas.items():
            tf = sh.d / f"t{si:04d}.npz"
            if tf.exists() and ai is not None:
                z = np.load(tf)
                have = set()
                for (i, f0, f1), lg in zip(z["w"], z["logits"]):
                    W.append((set_, si, int(i), int(f0), int(f1), CODES.index(m["lang"][int(i)])))
                    T.append(lg[ai].astype(np.float32))
                    have.add((int(i), int(f0), int(f1)))
                for i in range(len(m["ids"])):  # the utterance from the onset (<= 12 s): label only (CE)
                    if m["n"][i] >= 6 and (i, 0, m["n"][i]) not in have:
                        W.append((set_, si, i, 0, min(m["n"][i], 150), CODES.index(m["lang"][i])))
                        T.append(None)
            elif not need_teacher:
                for i, rid in enumerate(m["ids"]):
                    for f0, f1 in windows_for(rid, m["onset"][i], m["n"][i], k_rand):
                        W.append((set_, si, i, f0, f1, CODES.index(m["lang"][i])))
                        T.append(None)
    return W, T


def head_cfg(hidden: int, dropout: float, layers: list[int], context: int = 0, rnn: int = 0) -> dict:
    cfg = {"type": "language", "num_languages": len(CODES), "hidden": hidden, "att_hidden": 128,
           "cls_hidden": 256, "dropout": dropout, "min_frames": 6, "labels": CODES, "weight": 1.0,
           "from_layers": [b - 1 for b in layers]}
    if context:
        cfg["context"] = int(context)
    if rnn:
        cfg["rnn"] = int(rnn)
    return cfg


def save_head_file(head, mix, cfg, layers, path: Path, meta: dict):
    """The audioforge.lid head-file format (attach_head), written without a SpeechModel; read back and compared."""
    import torch
    blob = {"format": "audioforge-lid-head-v1", "name": "lid", "cfg": cfg,
            "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "labels": list(CODES),
            "from_layers": [b - 1 for b in layers], "layer_mix": (mix.detach().cpu() if mix is not None else None),
            "encoder": {"d_model": 512, "n_layers": 17, "att_context_size": [70, 1]}, "meta": meta}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    back = torch.load(path, map_location="cpu", weights_only=False)
    assert all(torch.equal(back["state_dict"][k], v) for k, v in blob["state_dict"].items())
    if mix is not None:
        assert torch.equal(back["layer_mix"], blob["layer_mix"])


def _pad_batch(chunks, dev):
    import torch
    L = max(len(c) for c in chunks)
    x = np.zeros((len(chunks), L) + chunks[0].shape[1:], np.float16)
    for j, c in enumerate(chunks):
        x[j, : len(c)] = c
    return torch.from_numpy(x).to(dev).float(), torch.tensor([len(c) for c in chunks], device=dev)


def packed_input(chunks, mix, dev):
    """Chunks (n_i, nb, D) float16 -> the block-mixed, padded head input (B, T_max, D) and lengths: the frames travel
    packed (no padding) as float16, the mix runs on the device, then they are scattered into the padded batch."""
    import torch
    ln = np.array([len(c) for c in chunks])
    flat = torch.from_numpy(np.concatenate(chunks)).to(dev).float()  # (sum n, nb, D)
    e = mixed_input(flat[None], mix)[0] if flat.shape[1] > 1 else flat[:, 0]
    bi = torch.from_numpy(np.repeat(np.arange(len(ln)), ln)).to(dev)
    ti = torch.from_numpy(np.concatenate([np.arange(n) for n in ln])).to(dev)
    out = torch.zeros(len(ln), int(ln.max()), e.shape[-1], device=dev)
    out[bi, ti] = e
    return out, torch.from_numpy(ln).to(dev)


def mixed_input(x, mix):
    """x (B, T, nb, D) -> (B, T, D): softmax mix over the tapped blocks (one block: itself)."""
    if x.shape[2] == 1:
        return x[:, :, 0]
    w = mix.softmax(0)
    return (x * w[None, None, :, None]).sum(2)


ELEN = {1.0: 14, 2.0: 26, 3.0: 39, 5.0: 64}  # encoder frames of a standalone L-s clip (stage_elen asserts these)


def stage_elen(a):
    import torch

    from audioforge.train import load_model
    m = load_model(ROOT / "runs/stage1_served.afm", "cpu").eval()
    with torch.no_grad():
        got = {L: int(m.encode(torch.zeros(1, int(L * SR)), torch.tensor([int(L * SR)]))[1][0]) for L in ELEN}
    print(got, flush=True)
    assert got == ELEN, got


class EvalCache:
    """Cached features of an evaluation set: 'on' view (first 5.3 s from the Silero onset) and 'full' view."""

    def __init__(self, set_: str, layers):
        self.on, self.full = Shards(set_, "on"), Shards(set_, "full")
        self.bidx = [self.on.blocks.index(b) for b in layers]
        self.items = [(si, i) for si, m in sorted(self.on.metas.items()) for i in range(len(m["ids"]))]
        self.y = np.array([CODES.index(self.on.metas[si]["lang"][i]) for si, i in self.items])
        self.vad = {"on": [np.array(self.on.metas[si]["vad"][i], np.float32) for si, i in self.items],
                    "full": [np.array(self.full.metas[si]["vad"][i], np.float32) for si, i in self.items]}

    def chunks(self, view):
        sh = self.on if view == "on" else self.full
        return [sh.frames(si, i, 0, sh.metas[si]["n"][i], self.bidx) for si, i in self.items]


def eval_cached(head, mix, ev: EvalCache, dev, bs: int = 64, gated: bool = False) -> dict:
    """Posteriors (N, 5 conditions, 17) on the cached eval clips: 1 / 2 / 3 / 5 s from the running posterior of the
    'on' view at the clip's frame count, full from the 'full' view."""
    import torch
    head.eval()
    N = len(ev.items)
    P = np.zeros((N, 5, len(CODES)), np.float32)
    with torch.no_grad():
        for view in ("on", "full"):
            ch = ev.chunks(view)
            for s0 in range(0, N, bs):
                e, ln = packed_input(ch[s0: s0 + bs], mix, dev)
                w, wx, wxx = head._terms(e)
                m = (torch.arange(e.shape[1], device=dev)[None] < ln[:, None]).float()[..., None]
                if gated:
                    g = np.zeros(m.shape[:2], np.float32)
                    for j in range(len(ln)):
                        v = ev.vad[view][s0 + j]
                        g[j, : len(v)] = v > 0.5
                    g = torch.from_numpy(g).to(dev)[..., None] * m
                    m = torch.where(g.sum(1, keepdim=True) > 0, g, m)
                z = head._classify((w * m).cumsum(1), (wx * m).cumsum(1), (wxx * m).cumsum(1)).softmax(-1)
                lnc = ln.cpu().numpy()
                if view == "on":
                    idx = torch.from_numpy(np.stack([np.minimum(lnc, ELEN[L]) - 1 for L in (1.0, 2.0, 3.0, 5.0)], 1))
                    P[s0: s0 + len(lnc), :4] = z.gather(1, idx.to(dev)[..., None].expand(-1, -1, z.shape[-1])).cpu().numpy()
                else:
                    idx = torch.from_numpy(lnc - 1)
                    P[s0: s0 + len(lnc), 4] = z.gather(1, idx.to(dev)[:, None, None].expand(-1, 1, z.shape[-1]))[:, 0].cpu().numpy()
    head.train()
    return P


def summarize(P, y, edP=None) -> dict:
    out = {}
    for ci, c in enumerate(("1s", "2s", "3s", "5s", "full")):
        ok = P[:, ci].argmax(-1) == y
        out[c] = {"acc": round(float(ok.mean()), 4), "ci95": boot_ci(ok)}
        if edP is not None:
            e = edP[:, ci].argmax(-1) == 0
            out[c]["edacc_en"] = round(float(e.mean()), 4)
            out[c]["edacc_ci95"] = boot_ci(e)
    return out


def stage_train(a):
    """Head on cached encoder features: loss = (1 - alpha) * CE (anytime running posterior + last frame, as
    heads.audio.LanguageHead.loss) + alpha * T^2 * KL(AmberNet_T || head_T) at the window's last frame (teacher
    restricted to the 17 languages, renormalised). Class-balanced batches (English: half FLEURS, half extra English
    when --extra-en); views: clean 'full' and, with --aug, the augmented 'aug' view of the same row (frames mapped by
    the speed factor); frame dropout on the pooling. Resumable in --segment-second pieces; state + best head on the
    SSD (read back); dev (2 s + full) picks the step."""
    import torch
    import torch.nn.functional as F

    from audioforge.model import build_head
    dev = torch.device(a.device)
    torch.manual_seed(0)
    layers = parse_blocks(a.blocks)
    sets = ["train"] + (["trainx"] if a.trainx else []) + (["extra_en"] if a.extra_en else [])
    W, T = build_windows(sets, a.k_rand, need_teacher=a.alpha > 0)
    if a.n_per_lang:  # data-scaling runs: keep the first n rows per language of train (+ trainx)
        keep, seen = set(), {}
        for set_ in sets:
            sh = Shards(set_, "on")
            for si, m in sorted(sh.metas.items()):
                for i, l in enumerate(m["lang"]):
                    k = (l, set_ == "extra_en")
                    if seen.get(k, 0) < a.n_per_lang:
                        keep.add((set_, si, i))
                        seen[k] = seen.get(k, 0) + 1
        sel = [j for j, w in enumerate(W) if (w[0], w[1], w[2]) in keep]
        W, T = [W[j] for j in sel], [T[j] for j in sel]
    shards = {s_: Shards(s_, "on") for s_ in sets}
    fullsh = {s_: Shards(s_, "full") for s_ in sets}
    augsh = {s_: Shards(s_, "aug") for s_ in sets} if a.aug else {}
    bidx = {(s_, v): [sh[s_].blocks.index(b) for b in layers]
            for v, sh in (("on", shards), ("full", fullsh), ("aug", augsh)) for s_ in sets if s_ in sh and sh[s_].metas}
    lab = np.array([w[5] for w in W])
    is_extra = np.array([w[0] == "extra_en" for w in W])
    by_class = {c: np.nonzero((lab == c) & ~is_extra)[0] for c in range(len(CODES))}
    extra_idx = np.nonzero(is_extra)[0]
    print(f"[train] {a.tag}: {len(W)} windows, sets {sets}, blocks {layers}, teacher windows "
          f"{sum(t is not None for t in T)}", flush=True)
    cfg = head_cfg(a.hidden, a.dropout, layers, a.context, a.rnn)
    head = build_head({k: v for k, v in cfg.items() if k != "from_layers"}, 512).to(dev)
    mix = torch.nn.Parameter(torch.zeros(len(layers), device=dev)) if len(layers) > 1 else None
    params = list(head.parameters()) + ([mix] if mix is not None else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    FIX_RUNS.mkdir(parents=True, exist_ok=True)
    state_path = FIX_RUNS / f"state_{a.tag}.pt"
    st = {"step": 0, "log": [], "best": -1.0, "best_state": None, "sec": 0.0}
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        head.load_state_dict(st["head"])
        if mix is not None:
            mix.data.copy_(st["mix"].to(dev))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
    evd = EvalCache("dev", layers)
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    io_pool = ThreadPoolExecutor(8)  # random mmap reads from the SSD are latency-bound: read a batch in parallel

    def read(spec):
        kind, s_, si, i, g0, g1 = spec
        sh = {"on": shards, "full": fullsh, "aug": augsh}[kind][s_]
        return np.ascontiguousarray(sh.frames(si, i, g0, g1, bidx[(s_, kind)]))

    def make_batch(step):
        rng = np.random.default_rng(1_000_003 * (a.seed + 1) + step)
        cls = rng.integers(0, len(CODES), a.batch_size)
        pick = []
        for c in cls:
            if c == 0 and len(extra_idx) and rng.random() < a.extra_frac:
                pick.append(int(extra_idx[rng.integers(0, len(extra_idx))]))
            else:
                pick.append(int(by_class[c][rng.integers(0, len(by_class[c]))]))
        specs, ys, tt, has_t, crops = [], [], [], [], []
        for j in pick:
            s_, si, i, f0, f1, y = W[j]
            u = rng.random()
            n_on = shards[s_].metas[si]["n"][i]
            if u < a.full_p and si in fullsh[s_].metas:  # the same audio span read from the file-start pass
                nf = fullsh[s_].metas[si]["n"][i]
                sh_ = int(round(shards[s_].metas[si]["audio_offset"][i] / FRAME))
                g0, g1 = (0, nf) if (f0 == 0 and f1 == n_on) else (min(nf - 1, f0 + sh_), min(nf, f1 + sh_))
                specs.append(("full", s_, si, i, g0, max(g1, g0 + 1)))
            elif a.aug and u < a.full_p + a.aug_p and si in augsh[s_].metas:
                sp = augsh[s_].metas[si]["speed"][i]
                na = augsh[s_].metas[si]["n"][i]
                g0, g1 = min(na - 1, int(round(f0 / sp))), min(na, int(round(f1 / sp)))
                specs.append(("aug", s_, si, i, g0, max(g1, g0 + 1)))
            else:
                specs.append(("on", s_, si, i, f0, f1))
            crops.append(a.crop and rng.random() < 0.3)
            ys.append(y)
            has_t.append(T[j] is not None)
            tt.append(T[j] if T[j] is not None else np.zeros(len(CODES), np.float32))
        chunks = list(io_pool.map(read, specs))
        for k, c in enumerate(chunks):
            if crops[k] and len(c) > 12:  # extra prefix crops (anytime)
                chunks[k] = c[: int(rng.integers(6, len(c)))]
        return chunks, ys, tt, has_t

    prefetch = ThreadPoolExecutor(1)
    nxt = prefetch.submit(make_batch, st["step"])
    t_data = 0.0
    while st["step"] < a.steps and time.time() - t0 < a.segment:
        td = time.time()
        chunks, ys, tt, has_t = nxt.result()
        nxt = prefetch.submit(make_batch, st["step"] + 1)
        e, ln = packed_input(chunks, mix, dev)
        y = torch.tensor(ys, device=dev)
        t_data += time.time() - td
        if a.feat_noise > 0:
            e = e + a.feat_noise * torch.randn_like(e) * e.std(-1, keepdim=True)
        w, wx, wxx = head._terms(e)
        Tn = e.shape[1]
        tpos = torch.arange(Tn, device=dev)[None]
        m = (tpos < ln[:, None]).float()
        if a.frame_drop > 0:
            m = m * (torch.rand_like(m) > a.frame_drop).float()
            m[:, 0] = (ln > 0).float()
        m3 = m[..., None]
        z = head._classify((w * m3).cumsum(1), (wx * m3).cumsum(1), (wxx * m3).cumsum(1))
        B, _, L = z.shape
        use = (tpos < ln[:, None]) & (tpos >= torch.clamp(ln[:, None] - 1, max=head.min_frames - 1))
        ce = F.cross_entropy(z.reshape(-1, L), y[:, None].expand(B, Tn).reshape(-1), reduction="none").view(B, Tn)
        anytime = (ce * use).sum(1) / use.sum(1).clamp(min=1)
        zl = z.gather(1, (ln - 1)[:, None, None].expand(B, 1, L))[:, 0]
        last = F.cross_entropy(zl, y, reduction="none")
        ce_loss = 0.5 * (anytime.mean() + last.mean())
        loss = ce_loss
        kl_v = torch.tensor(0.0)
        ht = torch.tensor(has_t, device=dev)
        if a.alpha > 0 and bool(ht.any()):
            tl = torch.from_numpy(np.stack(tt)).to(dev)
            pt = (tl / a.temp).log_softmax(-1)
            kl = F.kl_div((zl / a.temp).log_softmax(-1), pt, log_target=True, reduction="none").sum(-1)
            kl_v = (kl * ht).sum() / ht.sum()
            loss = (1 - a.alpha) * ce_loss + a.alpha * a.temp ** 2 * kl_v
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()
        st["step"] += 1
        if st["step"] % a.eval_every == 0 or st["step"] == a.steps:
            P = eval_cached(head, mix, evd, dev)
            acc = summarize(P, evd.y)
            rec = {"step": st["step"], "loss": round(float(loss), 4), "ce": round(float(ce_loss), 4),
                   "kl": round(float(kl_v), 4), "dev_2s": acc["2s"]["acc"], "dev_full": acc["full"]["acc"],
                   "sec": round(st["sec"] + time.time() - t0, 1), "data_sec_segment": round(t_data, 1)}
            st["log"].append(rec)
            print(rec, flush=True)
            score = acc["2s"]["acc"] + acc["full"]["acc"]
            if score > st["best"]:
                st["best"], st["best_step"] = score, st["step"]
                st["best_state"] = {"head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                                    "mix": mix.detach().cpu().clone() if mix is not None else None}
    st["sec"] += time.time() - t0
    st["head"] = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    st["mix"] = mix.detach().cpu() if mix is not None else None
    st["opt"], st["sched"] = opt.state_dict(), sched.state_dict()
    torch.save(st, state_path)
    back = torch.load(state_path, map_location="cpu", weights_only=False)
    assert back["step"] == st["step"] and all(torch.equal(back["head"][k], v) for k, v in st["head"].items())
    if st["step"] < a.steps:
        print(f"[train] segment done at step {st['step']}/{a.steps}; re-run to resume", flush=True)
        return
    head.load_state_dict(st["best_state"]["head"])
    if mix is not None:
        mix.data.copy_(st["best_state"]["mix"].to(dev))
    head = head.cpu().eval()
    mixc = mix.detach().cpu() if mix is not None else None
    out = FIX_RUNS / f"{a.tag}.pt"
    meta = {"tag": a.tag, "layers_1based": layers, "sets": sets, "alpha": a.alpha, "temp": a.temp,
            "steps": a.steps, "best_step": st["best_step"], "aug": bool(a.aug), "n_per_lang": a.n_per_lang,
            "context": a.context, "rnn": a.rnn, "hidden": a.hidden, "feat_noise": a.feat_noise,
            "dropout": a.dropout, "wd": a.wd}
    save_head_file(head, mixc, cfg, layers, out, meta)
    import shutil
    shutil.copyfile(out, ROOT / f"runs/lid_{a.tag}.pt")
    # held-out numbers on the cached test clips + EdAcc (the official check re-runs the real clip path)
    evt, eve = EvalCache("test", layers), EvalCache("edacc", layers)
    Pt, Pe = eval_cached(head, mixc, evt, "cpu"), eval_cached(head, mixc, eve, "cpu")
    np.savez(FIX_RUNS / f"{a.tag}_cached_preds.npz", test=Pt, y=evt.y, edacc=Pe)
    res = summarize(Pt, evt.y, Pe)
    rec = {**meta, "sec": round(st["sec"], 1), "device": a.device, "hidden": a.hidden, "lr": a.lr,
           "batch_size": a.batch_size, "frame_drop": a.frame_drop, "extra_frac": a.extra_frac,
           "head_params": int(sum(p.numel() for p in head.parameters())),
           "mix": (mixc.softmax(0).numpy().round(3).tolist() if mixc is not None else None),
           "log": st["log"], "cached_test": res, "out": str(out)}
    merge_fix(f"train_{a.tag}", rec)
    print("[train] " + a.tag + " cached test: " + " ".join(f"{c} {100 * res[c]['acc']:.1f}" for c in res)
          + " | EdAcc en " + " ".join(f"{c} {100 * res[c]['edacc_en']:.1f}" for c in res), flush=True)
    print("[train] finished", flush=True)


def stage_confusion(a):
    """Diagnosis of the current served-candidate head (research/archive/LID.md lid_aug) from its saved predictions
    (data/lid/preds/lid_aug.npz): per-language accuracy, the largest off-diagonal confusions at 2 s and on full
    utterances, accuracy per window length with CIs; the same for AmberNet for reference."""
    import sys as _s
    _s.path.insert(0, str(ROOT / "scripts/research"))
    from lid import _load_preds
    rec = {}
    for tag in (a.rest or ["lid_aug", "ambernet"]):
        P, y, _ = _load_preds(tag)
        r = {"per_window": {}, "top_confusions": {}, "per_lang": {}}
        for ci, c in enumerate(("1s", "2s", "3s", "5s", "full")):
            pred = P[:, ci].argmax(-1)
            ok = pred == y
            r["per_window"][c] = {"acc": round(float(ok.mean()), 4), "ci95": boot_ci(ok)}
            if c in ("2s", "full"):
                cm = np.zeros((len(CODES), len(CODES)), int)
                np.add.at(cm, (y, pred), 1)
                off = [(CODES[i], CODES[j], int(cm[i, j])) for i in range(len(CODES)) for j in range(len(CODES))
                       if i != j and cm[i, j]]
                off.sort(key=lambda t: -t[2])
                r["top_confusions"][c] = [f"{i}->{j}: {n}/150" for i, j, n in off[:12]]
                r["per_lang"][c] = {CODES[i]: round(float(cm[i, i] / cm[i].sum()), 3) for i in range(len(CODES))}
                r.setdefault("errors_in_top10_pairs", {})[c] = round(sum(t[2] for t in off[:10]) / max(1, sum(t[2] for t in off)), 3)
        rec[tag] = r
        print(tag, json.dumps(r["per_window"]), "\n  2s:", r["top_confusions"]["2s"][:10], "\n  full:",
              r["top_confusions"]["full"][:8], "\n  top-10 pairs share of errors:", r["errors_in_top10_pairs"], flush=True)
    merge_fix("confusion_after" if "lid_distill" in rec else "confusion_before", rec)


def load_head_blob(path):
    """(head, mix or None, layers 1-based) from an audioforge.lid head file."""
    import torch

    from audioforge.lid import load_head
    from audioforge.model import build_head
    blob = load_head(path)
    head = build_head(dict(blob["cfg"]), 512)
    head.load_state_dict(blob["state_dict"])
    layers = [k + 1 for k in blob["from_layers"]]
    mix = blob["layer_mix"]
    return head.eval(), (torch.as_tensor(mix) if mix is not None and len(layers) > 1 else None), layers


def stage_rule(a):
    """Streaming announcement rule on whole utterances streamed from the file start (the server's view): the head's
    VAD-gated running posterior (30 s half-life of pooled speech, audioforge.lid.LIDStream's numbers, computed on the
    cached file-start encoder pass) -> audioforge.lid.LangDecider with ``threshold`` and optional ``max_ms`` (announce
    the top language after max_ms of pooled speech if nothing was confident). Per rule: announced fraction, accuracy of
    the FIRST announced language [CI], time to announce (s from the file start; p50 / p90) and flips per utterance;
    FLEURS-17 test (all 2550) and EdAcc (257, correct = en)."""
    import torch

    from audioforge.lid import LangDecider
    head, mix, layers = load_head_blob(a.head)
    decay = 0.5 ** (0.08 / 30.0)
    rules = [(0.9, None), (0.9, 3000.0), (0.8, 3000.0), (0.95, 3000.0), (0.9, 2000.0)]
    rec = {"head": a.head, "rules": {}}
    for set_ in ("test", "edacc"):
        ev = EvalCache(set_, layers)
        ch = ev.chunks("full")
        posts, keeps = [], []
        with torch.no_grad():
            for j, c in enumerate(ch):
                x = torch.from_numpy(np.ascontiguousarray(c)).float()[None]
                e = mixed_input(x, mix)
                keep = torch.from_numpy(ev.vad["full"][j] > 0.5)[None]
                st = head.init_stream(1)
                posts.append(head.step(e, st, keep=keep, decay=decay).softmax(-1)[0].numpy())
                keeps.append(keep[0].numpy())
        for th, mx in rules:
            first_ok, first_t, flips, ann = [], [], [], 0
            for P, K, y in zip(posts, keeps, ev.y):
                d = LangDecider(CODES, threshold=th, min_ms=1000.0, frame_ms=80.0, max_ms=mx)
                evs = [e for t in range(len(P)) if K[t] for e in [d.update(P[t], (t + 1) * 0.08)] if e]
                if evs:
                    ann += 1
                    first_ok.append(evs[0]["language"] == CODES[y])
                    first_t.append(evs[0]["t"])
                    flips.append(len(evs) - 1)
            key = f"th{th}_max{int(mx) if mx else 'none'}"
            r = {"announced": round(ann / len(posts), 4),
                 "first_acc_of_all": round(float(np.sum(first_ok) / len(posts)), 4),
                 "first_acc": round(float(np.mean(first_ok)), 4) if first_ok else None,
                 "first_acc_ci95": boot_ci(np.array(first_ok)) if first_ok else None,
                 "t_p50": round(float(np.median(first_t)), 2) if first_t else None,
                 "t_p90": round(float(np.percentile(first_t, 90)), 2) if first_t else None,
                 "flips_per_utt": round(float(np.mean(flips)), 3) if flips else None, "n": len(posts)}
            rec["rules"].setdefault(set_, {})[key] = r
            print(set_, key, r, flush=True)
    merge_fix(f"rule_{Path(a.head).stem}", rec)


# --------------------------------------------------------------------------- fusion fallback: head + transcript-text LID
V3_LANGS = ["bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt",
            "ro", "sk", "sl", "es", "sv", "ru", "uk"]  # Parakeet-TDT v3's 25 (research/archive/HYBRID_ASR.md section 1)
TDT_WORK = SSD / "scratch/hybrid_asr"


def stage_fusion(a):
    """Head early call + text LID (langid.py over the Parakeet-TDT v3 transcript, research/archive/HYBRID_ASR.md section 5) to
    confirm or override, on the utterances that have a v3 transcript: the 2 s clips (30 per language, 510) and the
    full utterances (150 per language for 10 languages, 30 for the others). Rules: head alone; text alone (empty =
    wrong); 'override' = the text's language when it is one of v3's supported FLEURS-17 languages with langid
    probability >= p_min, else the head; 'guarded' = the same, but only when the head's own top-1 is also a supported
    language (v3 cannot write he / ar / tr / fa / hi / zh / ja, so a head call for those is kept)."""
    import langid
    sup = [c for c in CODES if c in V3_LANGS]
    ident = langid.langid.LanguageIdentifier.from_modelstring(langid.langid.model, norm_probs=True)
    ident.set_languages(sorted(set(V3_LANGS) | set(CODES)))
    z = np.load(FIX_RUNS / f"{a.tag}_cached_preds.npz")
    P, y = z["test"], z["y"]
    ids = [r["id"] for r in D.read_manifest("test")]
    pos = {rid: k for k, rid in enumerate(ids)}
    rec = {"head": a.tag, "supported": sup}
    for cond, ci, tag in (("2s", 1, "fleurs2s"), ("full", 4, "fleurs")):
        idx, txt_lang, txt_p, secs = [], [], [], []
        for lang in CODES:
            f = TDT_WORK / f"tdt_{tag}_{lang}.jsonl"
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get("id") not in pos:
                    continue
                idx.append(pos[r["id"]])
                h = r["hyp"].strip()
                lg, pr = ident.classify(h) if h else ("empty", 0.0)
                txt_lang.append(lg)
                txt_p.append(float(pr))
                secs.append(float(r.get("sec", np.nan)))
        idx = np.array(idx)
        yy, hp = y[idx], P[idx, ci].argmax(-1)
        head_top = np.array([CODES[k] for k in hp])
        truth = np.array([CODES[k] for k in yy])
        is_sup = np.isin(truth, sup)
        out = {"n": int(len(idx)), "n_supported": int(is_sup.sum()), "tdt_sec_p50": round(float(np.nanmedian(secs)), 3)}
        preds = {"head": head_top, "text": np.array(txt_lang)}
        for pmin in (0.5, 0.9):
            tl = np.array(txt_lang)
            ok_txt = np.isin(tl, sup) & (np.array(txt_p) >= pmin)
            preds[f"override_p{pmin}"] = np.where(ok_txt, tl, head_top)
            preds[f"guarded_p{pmin}"] = np.where(ok_txt & np.isin(head_top, sup), tl, head_top)
        for k, pr in preds.items():
            ok = pr == truth
            out[k] = {"acc": round(float(ok.mean()), 4), "ci95": boot_ci(ok),
                      "acc_supported": round(float(ok[is_sup].mean()), 4),
                      "acc_unsupported": round(float(ok[~is_sup].mean()), 4) if (~is_sup).any() else None}
        # balanced view: every language weighted equally (the full-utterance set has 150 vs 30 per language)
        for k, pr in preds.items():
            out[k]["acc_lang_balanced"] = round(float(np.mean([np.mean(pr[truth == c] == c) for c in CODES
                                                                if (truth == c).any()])), 4)
        rec[cond] = out
        print(cond, json.dumps({k: v for k, v in out.items()}, indent=None)[:1500], flush=True)
    merge_fix(f"fusion_{a.tag}", rec)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage")
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--budget", type=float, default=480.0)
    ap.add_argument("--n-libri", type=int, default=3000)
    ap.add_argument("--set", default="train")
    ap.add_argument("--view", default="full", choices=("on", "full", "aug"))
    ap.add_argument("--blocks", default="6-13")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-sec", type=float, default=240.0)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--k-rand", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1, help="feats: every n-th shard only (e.g. a partial aug view)")
    ap.add_argument("--tag", default="fix")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--context", type=int, default=0, help="train: causal left-context frames of the head's conv")
    ap.add_argument("--rnn", type=int, default=0, help="train: units of the head's causal GRU (0 = none)")
    ap.add_argument("--feat-noise", type=float, default=0.0, help="train: Gaussian noise on the head input (x std)")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.0, help="weight of the AmberNet KL term")
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--trainx", action="store_true")
    ap.add_argument("--extra-en", action="store_true")
    ap.add_argument("--extra-frac", type=float, default=0.5)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--aug-p", type=float, default=0.25)
    ap.add_argument("--full-p", type=float, default=0.35)
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--frame-drop", type=float, default=0.1)
    ap.add_argument("--n-per-lang", type=int, default=0)
    ap.add_argument("--head", default=None)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segment", type=float, default=540.0)
    a, rest = ap.parse_known_args()
    a.rest = rest
    fn = globals().get(f"stage_{a.stage}")
    if fn is None:
        raise SystemExit(f"unknown stage {a.stage}")
    fn(a)


if __name__ == "__main__":
    main()
