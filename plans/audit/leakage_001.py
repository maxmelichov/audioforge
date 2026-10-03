# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Leakage audit 001: id-level checks behind plans/audit/leakage_001.md. Read-only, light CPU (one pass over the
1.1 GB smart-turn cache for audio hashes). It does not import audioforge (no torch) and never reads the HF token.

    uv run plans/audit/leakage_001.py            # from the repo root
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
TEST_AMI = {"IS1009b", "ES2004b", "TS3003b", "EN2002a"}
TEST_ICSI = {"Bmr013", "Bmr018", "Bro021"}
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def uid(s: str) -> str:  # audioforge.datasets.smartturn._uuid
    m = UUID.search(str(s).lower())
    if m:
        return m.group(0)
    b = str(s).replace("\\", "/").split("/")[-1]
    return b.rsplit(".", 1)[0] if "." in b else b


def smart_turn():
    print("== smart-turn v3.2 test (399 human_5_all clips)")
    R = ROOT / "data/smartturn"
    meta = json.loads((R / "cache/human_5_all.json").read_text())
    j = json.loads((R / "v32_human5_ids.json").read_text())
    te = {uid(x["id"]) for x in j["test"]}
    ids = meta["ids"]
    ev = [i for i, s in enumerate(ids) if uid(s) in te]
    tr = [i for i in range(len(ids)) if i not in set(ev)]
    print(f"  split: eval {len(ev)} / train {len(tr)}; v3.2 train&test uuid overlap {len({uid(x['id']) for x in j['train']} & te)}")
    w = np.load(R / "cache/human_5_all.npy", mmap_mode="r")
    off = meta["offsets"]
    h, fp = {}, {}
    for i in range(len(ids)):
        x = np.asarray(w[off[i]: off[i + 1]])
        h.setdefault(hashlib.md5(np.round(x * 32767).astype(np.int16).tobytes()).hexdigest(), []).append(i)
        nz = np.nonzero(np.abs(x) > 1e-3)[0]
        if len(nz):
            y = x[nz[0]: nz[-1] + 1]
            fp.setdefault((len(y) // 160, hashlib.md5(np.round(y[::16] * 200).astype(np.int8).tobytes()).hexdigest()), []).append(i)
    evs = set(ev)
    cross = lambda d: sum(1 for v in d.values() if any(i in evs for i in v) and any(i not in evs for i in v))  # noqa: E731
    print(f"  exact-audio groups across train/eval: {cross(h)}; trimmed coarse fingerprint groups across: {cross(fp)}")
    tfp = {k for k, v in fp.items() if any(i in evs for i in v)}
    te_u = {uid(ids[i]) for i in ev}
    n = idh = fph = 0
    for jf in sorted((SSD / "data/smart_turn_v3_2").glob("eng_*.json")):
        m = json.loads(jf.read_text())
        a = np.load(jf.with_suffix(".npy"), mmap_mode="r")
        for k, u in enumerate(m["ids"]):
            n += 1
            idh += uid(u) in te_u
            x = np.asarray(a[m["offsets"][k]: m["offsets"][k + 1]], np.float32) / 32768
            nz = np.nonzero(np.abs(x) > 1e-3)[0]
            if len(nz):
                y = x[nz[0]: nz[-1] + 1]
                fph += (len(y) // 160, hashlib.md5(np.round(y[::16] * 200).astype(np.int8).tobytes()).hexdigest()) in tfp
    print(f"  smart-turn-data-v3.2-train English rows fetched: {n}; uuid hits on the 399: {idh}; fingerprint hits: {fph}")


def turn_splits():
    print("== TURN_DATA disjointness (scratch/turndata/splits.json, written by turn_data.py splits)")
    s = {k: set(v) for k, v in json.loads((SSD / "scratch/turndata/splits.json").read_text()).items()}
    for k, v in s.items():
        print(f"  {k}: {len(v)} {dict(Counter(x.split(':')[0] for x in v))}")
    for a, b in [("heldout_old", "eval"), ("train_shipped", "heldout_old"), ("train_new", "eval"),
                 ("heldout_new", "eval"), ("train_shipped", "eval"), ("never_new", "eval")]:
        i = sorted(s[a] & s[b])
        print(f"  {a} & {b}: {len(i)} {i[:6]}")
    print("  (turn_data.stage_splits does not test heldout_old & eval or train_shipped & heldout_old)")
    o280 = json.loads((SSD / "data/oto280/cache/oto_index.json").read_text())
    o141 = json.loads((SSD / "data/oto141/cache/oto_index.json").read_text())
    clips = json.loads((SSD / "scratch/e2e_tsvad/clips.json").read_text())
    ev = {c["conversation"] for c in clips if c["set"] == "oto"}
    print(f"  oto id namespace: 280h raw {len(o280)} ids, 141h {len(o141)} ids, shared {len(set(o280) & set(o141))} "
          f"(non-zero: same id scheme, so the id checks are meaningful); 16 eval convs in 141h {len(ev & set(o141))}, "
          f"in 280h slice {len(ev & set(o280))}")
    for d in ("ami", "ami_eval"):
        c = json.loads((SSD / "scratch/eot_latency" / d / "clips.json").read_text())
        print(f"  eot_latency {d} turn windows: {dict(Counter(x['meeting'] for x in c))}")


def librispeech():
    print("== LibriSpeech")
    R = ROOT / "data/librispeech"
    first = [Path(json.loads(ln)["audio_filepath"]).stem for ln in (R / "test-clean-first200.jsonl").read_text().splitlines() if ln.strip()]
    trs = {p.name for p in (R / "LibriSpeech/train-clean-100").iterdir()}
    for split in ("test-clean", "test-other"):
        items = [ln.split(" ", 1)[0] for tr in sorted((R / "LibriSpeech" / split).glob("*/*/*.trans.txt"))
                 for ln in tr.read_text().splitlines()]
        pick = {items[i] for i in random.Random(0).sample(range(len(items)), 300)}
        spk = {p.name for p in (R / "LibriSpeech" / split).iterdir() if p.is_dir()}
        print(f"  {split}: speakers {len(spk)}, in train-clean-100 {len(spk & trs)}; final_compare picks {len(pick)}, "
              f"in test-clean-first200 (training WER gate) {len(pick & set(first))}")


def fleurs():
    print("== FLEURS")
    R = ROOT / "data/lid/fleurs"
    rows = [json.loads(ln) for f in ("manifest.jsonl", "manifest_trainx.jsonl") for ln in (R / f).read_text().splitlines() if ln.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["split"], []).append(r)
    sent = lambda sp: {r["sent_id"] for r in by[sp]}  # noqa: E731
    uidx = lambda sp: {r["id"] for r in by[sp]}  # noqa: E731
    print(f"  rows {({k: len(v) for k, v in by.items()})}")
    print(f"  test utt ids in train/trainx/dev: {len(uidx('test') & (uidx('train') | uidx('trainx') | uidx('dev')))}")
    print(f"  test sentence ids {len(sent('test'))}: in train {len(sent('test') & sent('train'))}, trainx "
          f"{len(sent('test') & sent('trainx'))}, dev {len(sent('test') & sent('dev'))} (pooled over 17 languages)")


def ami_icsi_refs():
    print("== AMI / ICSI test meetings named in research scripts (selection candidates)")
    for p in sorted((ROOT / "scripts/research").glob("*.py")):
        t = p.read_text()
        hit = sorted(m for m in TEST_AMI | TEST_ICSI if m in t)
        if hit:
            print(f"  {p.name}: {hit}")


if __name__ == "__main__":
    for f in (smart_turn, turn_splits, librispeech, fleurs, ami_icsi_refs):
        try:
            f()
        except Exception as e:  # a missing cache or volume: report and go on
            print(f"  {f.__name__}: skipped ({type(e).__name__}: {e})")
