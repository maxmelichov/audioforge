"""Drop the rendered frames (and their 2 fps DOM samples) of the named shots so the next `film.py --frames` run
re-renders only those shots.   $V demo/archive/v5/invalidate.py --cut full --size 1920x1080 today_call end"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import film  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cut", default="full"); ap.add_argument("--size", default="1920x1080"); ap.add_argument("shots", nargs="+")
a = ap.parse_args()
W, H = map(int, a.size.split("x"))
shots, total = film.timeline(a.cut, W)
d = film.frame_dir(a.cut, W, H)
drop = set()
for sh in shots:
    if sh["name"] in a.shots:
        drop |= set(range(int(round(sh["start"] * film.FPS)), int(round(sh["end"] * film.FPS))))
for f in drop:
    (d / f"{f:05d}.png").unlink(missing_ok=True)
qa = d / "dom_samples.jsonl"
if qa.exists():
    keep = [line for line in qa.open() if json.loads(line)["f"] not in drop]
    qa.write_text("".join(keep))
print(f"dropped {len(drop)} frames of {a.shots} in {d}")
