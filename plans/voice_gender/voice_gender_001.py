# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "scikit-learn>=1.3"]
# ///
"""Voice-gender block sweep (linear probe), both cores, from the LID heads' cached FLEURS pool features
(lid_fix / core_0p6b lid caches: per row the mean of every block over the first 1/2/3/5 s from the speech onset and
over the whole 'on' view). Train = FLEURS train rows, held-out = FLEURS dev rows (selection split). Probe = standardised
logistic regression, class-balanced. Prints dev balanced accuracy and log loss per block at 2 s and the whole view."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
CACHE = {"115m": SSD / "cache/lid_fix", "0p6b": SSD / "scratch/core_0p6b/lid"}
G = {}
for l in (ROOT / "data/lid/fleurs/manifest.jsonl").read_text().splitlines():
    r = json.loads(l)
    G[r["id"]] = r["gender"]


def load(core, split):
    X, y = [], []
    for f in sorted((CACHE[core] / split / "on").glob("s[0-9]*.json")):
        m = json.loads(f.read_text())
        p = np.load(f.with_name(f.stem + "_pool.npy"), mmap_mode="r")
        keep = [i for i, rid in enumerate(m["ids"]) if G.get(rid) in ("FEMALE", "MALE")]
        X.append(np.asarray(p[keep], np.float32))
        y += [int(G[m["ids"][i]] == "MALE") for i in keep]
    return np.concatenate(X), np.array(y)


for core in sys.argv[1:] or ["115m", "0p6b"]:
    Xt, yt = load(core, "train")
    Xd, yd = load(core, "dev")
    print(f"{core}: train {len(yt)} rows ({yt.mean():.2f} male), dev {len(yd)} ({yd.mean():.2f} male), "
          f"{Xt.shape[2]} blocks", flush=True)
    print("block | dev bal.acc 2 s | dev log loss 2 s | dev bal.acc whole | dev log loss whole")
    for b in range(Xt.shape[2]):
        row = []
        for w in (1, 4):
            sc = StandardScaler().fit(Xt[:, w, b])
            lr = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced").fit(sc.transform(Xt[:, w, b]), yt)
            pd_ = lr.predict_proba(sc.transform(Xd[:, w, b]))[:, 1]
            row += [balanced_accuracy_score(yd, pd_ > 0.5), log_loss(yd, pd_)]
        print(f"{b + 1} | {row[0]:.4f} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f}", flush=True)
