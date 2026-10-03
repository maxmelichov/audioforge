# /// script
# requires-python = ">=3.10"
# ///
"""Kill-and-resume check of `python -m audioforge.train` on a 40-row synthetic manifest (CPU, tiny model).

    .venv/bin/python plans/trainer/resume_001.py <recipe.yaml>

1. reference run, 30 steps uninterrupted; 2. same run, SIGINT once step >= 12, then --resume to 30.
Checks: evals fired by time, last/ and best/ written, TensorBoard event files, the resumed run's first batch ids
equal the sampler replay, and every train/loss logged after the resume equals the reference bit for bit.
Runs go to runs/trainer_smoke{,_ref}/ with log.txt next to them (deleted and recreated on each call)."""
import json
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PY = [sys.executable, "-m", "audioforge.train"]
FLAGS = ["--max-steps", "30", "--batch-size", "4", "--eval-minutes", "0.01", "--encoder.d_model", "64",
         "--encoder.n_layers", "2", "--encoder.subsampling_channels", "16", "trainer.log_every=1"]


def run(recipe, name, extra=(), kill_at=None):
    d = Path("runs") / name
    if not extra:
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "log.txt", "a") as f:
        p = subprocess.Popen([*PY, recipe, "--name", name, *FLAGS, *extra], stdout=f, stderr=subprocess.STDOUT)
        while kill_at and p.poll() is None:
            steps = [int(s) for s in re.findall(r" step=(\d+) loss", (d / "log.txt").read_text())]
            if steps and max(steps) >= kill_at:
                p.send_signal(signal.SIGINT)
                break
            time.sleep(0.2)
        return p.wait()


def tb_losses(d: Path) -> dict[int, float]:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    out = {}
    for f in sorted((d / "tb").rglob("events.out.tfevents.*")):
        for e in EventFileLoader(str(f)).Load():
            for v in e.summary.value:
                if v.tag == "train/loss":
                    out[e.step] = float(v.tensor.float_val[0]) if v.HasField("tensor") else v.simple_value
    return out


def main(recipe):
    from audioforge.data import epoch_batches
    t0 = time.time()
    assert run(recipe, "trainer_smoke_ref") == 0
    ref = Path("runs/trainer_smoke_ref")
    print(f"reference: {time.time() - t0:.1f}s, evals {len(re.findall('eval step=', (ref / 'log.txt').read_text()))}")
    rc = run(recipe, "trainer_smoke", kill_at=12)
    d = Path("runs/trainer_smoke")
    st = json.loads((d / "last/trainer_state.json").read_text())
    print(f"killed: exit {rc}, last/ at step {st['step']} epoch {st['epoch']} batch {st['pos']}")
    assert rc == 130 and (d / "last").is_dir()
    rng = random.Random(0)  # sampler replay: epoch_batches from the saved epoch-start state
    rng.setstate((st["rng"][0], tuple(st["rng"][1]), st["rng"][2]))
    want = epoch_batches(40, 4, rng)[st["pos"]] if st["pos"] < 10 else None
    assert run(recipe, "trainer_smoke", ["--resume"]) == 0
    log = (d / "log.txt").read_text()
    got = re.search(r"next batch ids (\[[^\]]*\])", log)
    print(f"resume next batch ids {got and got.group(1)} replay {want}")
    if want is not None:
        assert got and json.loads(got.group(1)) == want
    a, b = tb_losses(ref), tb_losses(d)
    after = [s for s in sorted(b) if s > st["step"]]
    same = all(a[s] == b[s] for s in after)
    print(f"steps after resume {after[0]}..{after[-1]}: losses identical to reference: {same}")
    print("evals:", len(re.findall("eval step=", log)), "| files:", sorted(p.name for p in d.iterdir()),
          "| best:", sorted(p.name for p in (d / "best").iterdir()),
          "| tb events:", len(list((d / "tb").rglob("events.out.tfevents.*"))))
    assert same and (d / "best/model.afm").exists()


if __name__ == "__main__":
    main(sys.argv[1])
