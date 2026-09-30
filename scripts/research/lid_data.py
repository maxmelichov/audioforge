"""Spoken-LID data (research/archive/LID.md): a small FLEURS subset, streamed (no full tarball on disk).

  fetch   for each language / split, stream data/<fleurs>/audio/<split>.tar.gz from the google/fleurs HF dataset
          repo and keep the first N utterances (FLAC, 16 kHz mono) + a manifest; the rest of the tarball is never
          downloaded. Resumable: a finished (language, split) is skipped.
  stats   hours / utterances / genders per language and split.
  edacc   accented-English probe: 5 x 100-row groups of the EdAcc test split (CC-BY-SA-4.0) -> data/lid/edacc.

FLEURS (google/fleurs, CC-BY-4.0): n-way parallel read speech, 102 languages, speaker-disjoint train / dev / test.
Manifest: data/lid/fleurs/manifest.jsonl, one line per utterance {lang, fleurs, split, id, path, duration, gender}.

  .venv/bin/python scripts/research/lid_data.py fetch --langs en he ar          # a few languages per process (< 10 min)
  .venv/bin/python scripts/research/lid_data.py stats
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
OUT = ROOT / "data/lid/fleurs"
MANIFEST = OUT / "manifest.jsonl"
REPO = "google/fleurs"
LICENSE = "CC-BY-4.0"

# ISO 639-1 -> FLEURS config. The 7 required languages first, then 10 more common ones (chosen for coverage of
# scripts / families and for hard pairs: ru/uk, es/pt/it, ar/fa, hi, zh/ja, de/nl).
LANGS = {"en": "en_us", "he": "he_il", "ar": "ar_eg", "ru": "ru_ru", "es": "es_419", "fr": "fr_fr", "de": "de_de",
         "pt": "pt_br", "it": "it_it", "nl": "nl_nl", "pl": "pl_pl", "uk": "uk_ua", "tr": "tr_tr", "fa": "fa_ir",
         "hi": "hi_in", "zh": "cmn_hans_cn", "ja": "ja_jp"}
NAMES = {"en": "English", "he": "Hebrew", "ar": "Arabic (Egyptian)", "ru": "Russian", "es": "Spanish (Latin Am.)",
         "fr": "French", "de": "German", "pt": "Portuguese (Brazil)", "it": "Italian", "nl": "Dutch", "pl": "Polish",
         "uk": "Ukrainian", "tr": "Turkish", "fa": "Persian", "hi": "Hindi", "zh": "Mandarin", "ja": "Japanese"}
N_PER_SPLIT = {"train": 250, "dev": 60, "test": 150}  # ~50 min / ~12 min / ~30 min per language


def read_manifest(split: str | None = None, langs=None) -> list[dict]:
    if not MANIFEST.exists():
        return []
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    return [r for r in rows if (split is None or r["split"] == split) and (langs is None or r["lang"] in langs)]


PEAK = 0.89  # -1 dBFS


def load_audio(row: dict, normalize: bool = True) -> np.ndarray:
    """16 kHz float32. FLEURS levels vary by 55 dB between recordings (peak 0.005 .. 1.0), so every system gets the
    same peak-normalised (-1 dBFS) waveform, as a client-side AGC would deliver."""
    import soundfile as sf
    a, sr = sf.read(ROOT / row["path"], dtype="float32")
    assert sr == 16000
    if normalize:
        pk = float(np.abs(a).max()) if len(a) else 0.0
        if pk > 0:
            a = a * (PEAK / pk)
    return a.astype(np.float32)


def _tsv(fl: str, split: str) -> dict:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, f"data/{fl}/{split}.tsv", repo_type="dataset", local_dir=str(OUT / "_tsv"))
    meta = {}
    for line in Path(p).read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 7:
            meta[f[1]] = {"sent_id": int(f[0]), "gender": f[6].strip()}
    return meta


def fetch_one(lang: str, split: str, n: int) -> list[dict]:
    import soundfile as sf
    import requests
    from huggingface_hub import get_token, hf_hub_url
    fl = LANGS[lang]
    done = OUT / lang / f".{split}.done"
    if done.exists():
        return []
    meta = _tsv(fl, split)
    url = hf_hub_url(REPO, f"data/{fl}/audio/{split}.tar.gz", repo_type="dataset")
    tok = get_token()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    d = OUT / lang / split
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
                a, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")  # 32-bit float WAVs
                assert sr == 16000, (name, sr)
                if a.ndim > 1:
                    a = a[:, 0]
                uid = name.removesuffix(".wav")
                path = d / f"{uid}.flac"
                pk = float(np.abs(a).max()) if len(a) else 0.0  # levels vary by 55 dB: store peak-normalised 16 bit
                sf.write(path, a * (PEAK / pk) if pk > 0 else a, 16000, format="FLAC", subtype="PCM_16")
                info = meta.get(name, {})
                rows.append({"lang": lang, "fleurs": fl, "split": split, "id": uid, "path": str(path.relative_to(ROOT)),
                             "duration": round(len(a) / 16000, 3), "gender": info.get("gender"),
                             "sent_id": info.get("sent_id")})
                if len(rows) >= n:
                    break
    with MANIFEST.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    done.write_text(str(len(rows)))
    print(f"[fetch] {lang} {split}: {len(rows)} utts, {sum(r['duration'] for r in rows) / 60:.1f} min, "
          f"{time.time() - t0:.0f}s", flush=True)
    return rows


def stage_fetch(a):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for lang in a.langs or list(LANGS):
        for split in a.splits:
            if time.time() - t0 > a.budget:
                print("[fetch] time budget reached; rerun to resume", flush=True)
                return
            fetch_one(lang, split, N_PER_SPLIT[split])


# --------------------------------------------------------------------------- accented-English probe (EdAcc)
EDACC_REPO = "edinburghcstr/edacc"
EDACC_LICENSE = "CC-BY-SA-4.0"
EDACC_OUT = ROOT / "data/lid/edacc"
EDACC_MANIFEST = EDACC_OUT / "manifest.jsonl"


def read_edacc() -> list[dict]:
    if not EDACC_MANIFEST.exists():
        return []
    return [json.loads(l) for l in EDACC_MANIFEST.read_text().splitlines() if l.strip()]


def stage_edacc(a):
    """A few 100-row parquet row groups of the EdAcc test split (conversational English, speakers with ~40 L1s),
    read by HTTP range (no full shard download); segments >= 2 s, peak-normalised 16 kHz FLAC, lang = en."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfFileSystem, list_repo_files
    if EDACC_MANIFEST.exists():
        print("[edacc] done already", flush=True)
        return
    files = sorted(f for f in list_repo_files(EDACC_REPO, repo_type="dataset") if f.startswith("data/test-"))
    fs = HfFileSystem()
    EDACC_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for fi in files[:: max(1, len(files) // a.edacc_files)][: a.edacc_files]:
        pf = pq.ParquetFile(fs.open(f"datasets/{EDACC_REPO}/{fi}"))
        g = pf.metadata.num_row_groups // 2  # the middle row group of the shard
        t = pf.read_row_group(g, columns=["speaker", "accent", "l1", "gender", "audio"]).to_pylist()
        for j, r in enumerate(t):
            x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                import torch
                import torchaudio
                x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
            if len(x) < 2 * 16000:
                continue
            pk = float(np.abs(x).max())
            uid = f"{Path(fi).stem.split('-')[1]}_{g}_{j}"
            path = EDACC_OUT / f"{uid}.flac"
            sf.write(path, x * (PEAK / pk) if pk > 0 else x, 16000, format="FLAC", subtype="PCM_16")
            rows.append({"lang": "en", "split": "edacc", "id": f"edacc_{uid}", "path": str(path.relative_to(ROOT)),
                         "duration": round(len(x) / 16000, 3), "speaker": r["speaker"], "accent": r["accent"],
                         "l1": r["l1"], "gender": r["gender"]})
        print(f"[edacc] {fi}: row group {g}, {len(rows)} segments so far", flush=True)
    EDACC_MANIFEST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"[edacc] {len(rows)} segments, {sum(r['duration'] for r in rows) / 60:.1f} min, "
          f"{len({r['speaker'] for r in rows})} speakers, {len({r['l1'] for r in rows})} L1s", flush=True)


def stage_stats(a):
    rows = read_manifest()
    print("| lang | FLEURS | train utts / min | dev utts / min | test utts / min | test F/M |")
    print("|---|---|---|---|---|---|")
    for lang in LANGS:
        cells = []
        for split in ("train", "dev", "test"):
            rs = [r for r in rows if r["lang"] == lang and r["split"] == split]
            cells.append(f"{len(rs)} / {sum(r['duration'] for r in rs) / 60:.0f}")
        te = [r for r in rows if r["lang"] == lang and r["split"] == "test"]
        g = f"{sum(r['gender'] == 'FEMALE' for r in te)}/{sum(r['gender'] == 'MALE' for r in te)}"
        print(f"| {lang} ({NAMES[lang]}) | {LANGS[lang]} | " + " | ".join(cells) + f" | {g} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("fetch", "stats", "edacc"))
    ap.add_argument("--edacc-files", type=int, default=5)
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--splits", nargs="*", default=["test", "dev", "train"])
    ap.add_argument("--budget", type=float, default=480.0, help="seconds; stop starting new downloads after this")
    a = ap.parse_args()
    {"fetch": stage_fetch, "stats": stage_stats, "edacc": stage_edacc}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
