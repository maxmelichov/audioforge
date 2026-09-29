"""final_asr.FinalASRWorker (process mode): a child that dies while loading its model (a missing file here; CUDA out
of memory on a shared GPU in research/GPU_RUN_2026-09-29.md) raises in the parent instead of blocking it forever on
the ready message."""
import time

import pytest

from audioforge.final_asr import FinalASRWorker


def test_process_worker_that_dies_while_loading_raises(tmp_path):
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="exited"):
        FinalASRWorker(str(tmp_path / "missing.nemo"), mode="process", threads=1)
    assert time.perf_counter() - t0 < 120
