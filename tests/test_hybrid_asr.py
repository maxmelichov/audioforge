"""Hybrid front end: the Parakeet-TDT import (strict tensor count, duration outputs), the
incremental greedy TDT decoder (any split of the encoder frames == offline greedy), and serve --final-asr.

Uses tiny synthetic .nemo archives; the real-checkpoint tests need data/nemo/parakeet-tdt-0.6b-v3.nemo and
AUDIOFORGE_BIG_TESTS=1 (they load ~2.5 GB; machine rules).
"""
import io
import itertools
import os
import re
import tarfile
from pathlib import Path

import pytest
import torch
import yaml

from audioforge.nemo_import import import_nemo, tdt_durations, translate_config
from audioforge.tokenizer import SentencePieceTokenizer
from tests.test_nemo_import import _INV, nemo_config

ROOT = Path(__file__).resolve().parent.parent
REAL_TDT = ROOT / "data" / "nemo" / "parakeet-tdt-0.6b-v3.nemo"
DURS = [0, 1, 2, 3, 4]
_NO_BIAS = re.compile(r"encoder\.layers\.\d+\.(ff[12]\.[14]|att\.linear_(q|k|v|out)|conv\.(pw1|dw|pw2))\.bias")

SLACK = 1.0 if os.environ.get("BULLETPROOF_STRICT_TIMING") == "1" else 3.0


def tdt_config(vocab=40, d=32):
    """parakeet-tdt-0.6b-v3's config shape (non-causal, batch-norm conv, per_feature, no biases, 2-layer LSTM,
    joint with 5 duration outputs, an aux_ctc stub without weights), scaled down."""
    nc = nemo_config(vocab=vocab, d=d, causal=False, conv_norm="batch_norm")
    nc["preprocessor"].update(features=128, normalize="per_feature")
    nc["encoder"].update(feat_in=128, use_bias=False, xscaling=False)
    nc["decoder"]["prednet"]["pred_rnn_layers"] = 2
    nc["joint"]["num_extra_outputs"] = len(DURS)
    nc["decoding"] = {"strategy": "greedy_batch", "model_type": "tdt", "durations": list(DURS),
                      "greedy": {"max_symbols": 10}}
    nc["loss"] = {"loss_name": "tdt", "tdt_kwargs": {"durations": list(DURS), "sigma": 0.02, "omega": 0.1}}
    nc["model_defaults"] = {"tdt_durations": list(DURS), "num_tdt_durations": len(DURS)}
    nc["aux_ctc"] = {"ctc_loss_weight": 0.3, "decoder": {"_target_": "nemo.collections.asr.modules.ConvASRDecoder",
                                                          "feat_in": None, "num_classes": -1, "vocabulary": []}}
    nc["target"] = "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
    return nc


def _write_nemo(path, nc, tok, sd):
    with tarfile.open(path, "w") as tar:
        for name, data in [("model_config.yaml", yaml.safe_dump(nc).encode()), ("abc123_tokenizer.model", tok.model_bytes)]:
            info = tarfile.TarInfo("./" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf = io.BytesIO()
        torch.save(sd, buf)
        info = tarfile.TarInfo("./model_weights.ckpt")
        info.size = buf.tell()
        buf.seek(0)
        tar.addfile(info, buf)


@pytest.fixture(scope="module")
def fake_tdt(tmp_path_factory):
    from audioforge.model import SpeechModel
    words = ["the", "cat", "sat", "on", "a", "mat", "with", "hat", "and", "bat", "while", "rat", "ran", "far"]
    tok = SentencePieceTokenizer.train([" ".join(words[i:] + words[:i]) for i in range(len(words))] * 5,
                                       vocab_size=40, specials=[])
    nc = tdt_config(vocab=tok.vocab_size)
    torch.manual_seed(2)
    src = SpeechModel(translate_config(nc, nemo_keys=[]), SentencePieceTokenizer(tok.model_bytes, [])).eval()
    with torch.no_grad():
        for n, p in src.named_parameters():
            if "pos_bias" in n or "norm" in n:
                p.add_(0.1 * torch.randn_like(p))
            if _NO_BIAS.fullmatch(n):
                p.zero_()
        for n, b in src.named_buffers():  # non-trivial BatchNorm statistics
            if n.endswith("running_mean"):
                b.normal_(0, 0.1)
            elif n.endswith("running_var"):
                b.uniform_(0.5, 1.5)
    sd = {}
    for k, v in src.state_dict().items():
        if _NO_BIAS.fullmatch(k):
            continue
        for pat, rep in _INV:
            k = re.sub(pat, rep, k)
        sd[k] = v
    sd["preprocessor.featurizer.window"] = src.preprocessor.window.clone()
    sd["preprocessor.featurizer.fb"] = src.preprocessor.fb.clone()[None]
    path = tmp_path_factory.mktemp("nemo") / "tiny_tdt.nemo"
    _write_nemo(path, nc, tok, sd)
    return path, src, sd, nc, tok


def test_tdt_import_strict_tensor_count(fake_tdt):
    path, src, nemo_sd, _, tok = fake_tdt
    m = import_nemo(path)
    h = m.heads["rnnt"]
    assert set(m.heads) == {"rnnt"} and m.primary == "rnnt" and m.cfg["heads"]["rnnt"]["type"] == "tdt"
    assert h.is_tdt and h.durations == DURS and h.sigma == 0.02 and h.pred.lstm.num_layers == 2
    assert h.joint.out[2].out_features == tok.vocab_size + 1 + len(DURS)  # [vocab..., blank, durations...]
    info = m.import_info
    n_layers = len(m.encoder.layers)
    assert info["n_tensors"] == len(nemo_sd) and info["n_loaded"] == len(nemo_sd) - 2
    assert info["zero_biases"] == 11 * n_layers
    assert info["n_loaded"] + info["zero_biases"] == len(m.state_dict())
    for k, v in src.state_dict().items():
        assert torch.equal(m.state_dict()[k], v), k
    audio = torch.randn(1, 12000) * 0.1
    with torch.no_grad():
        a, _ = src.encode(audio, torch.tensor([12000]))
        b, _ = m.encode(audio, torch.tensor([12000]))
    assert torch.equal(a, b)
    assert m.transcribe([audio[0].numpy()]) == src.transcribe([audio[0].numpy()])


def test_tdt_import_fails_loudly(fake_tdt, tmp_path):
    _, _, nemo_sd, nc, tok = fake_tdt
    # an unmapped tensor
    bad = dict(nemo_sd, **{"joint.extra_duration_proj.weight": torch.zeros(3, 3)})
    _write_nemo(tmp_path / "a.nemo", nc, tok, bad)
    with pytest.raises(KeyError, match="unmapped"):
        import_nemo(tmp_path / "a.nemo")
    # a missing tensor (the joint output layer)
    miss = {k: v for k, v in nemo_sd.items() if k != "joint.joint_net.2.weight"}
    _write_nemo(tmp_path / "b.nemo", nc, tok, miss)
    with pytest.raises(KeyError, match="missing"):
        import_nemo(tmp_path / "b.nemo")
    # duration outputs that do not match the stated durations
    nc2 = yaml.safe_load(yaml.safe_dump(nc))
    nc2["joint"]["num_extra_outputs"] = 4
    with pytest.raises(NotImplementedError, match="num_extra_outputs"):
        translate_config(nc2)
    nc3 = yaml.safe_load(yaml.safe_dump(nc))
    nc3["decoding"]["durations"] = [0, 1, 2, 4, 8]
    with pytest.raises(NotImplementedError, match="inconsistent"):
        translate_config(nc3)
    # a plain RNNT config is not TDT
    assert tdt_durations(nemo_config()) is None


@pytest.mark.parametrize("pieces", [[1], [3, 5, 7], [2] * 20, [11, 1, 1, 9]])
def test_tdt_incremental_decoder_equals_offline(fake_tdt, pieces):
    """The greedy TDT decoder fed encoder frames in pieces (durations jumping across piece boundaries, prediction
    state carried) emits exactly the offline greedy tokens."""
    path, *_ = fake_tdt
    m = import_nemo(path)
    h = m.heads["rnnt"]
    torch.manual_seed(5)
    V1 = h.vocab_size + 1
    with torch.no_grad():  # sharper random joint: tokens and blanks with durations 0 / 2 / 4 all occur (checked)
        h.joint.out[2].weight[V1:].mul_(10.0)
        h.joint.out[2].weight[:V1].mul_(4.0)
        h.joint.out[2].bias[h.blank] += 2.0
    f = torch.randn(40, m.encoder.d_model) * 3
    off = h._greedy(f)
    s, i, k = h.greedy_stream(), 0, 0
    while i < f.shape[0]:
        n = pieces[k % len(pieces)]
        s.feed(f[i:i + n])
        i, k = i + n, k + 1
    assert s.tokens == off
    assert 10 <= len(off) < 40


@pytest.mark.skipif(not (REAL_TDT.exists() and os.environ.get("AUDIOFORGE_BIG_TESTS")),
                    reason="needs the 2.5 GB checkpoint and AUDIOFORGE_BIG_TESTS=1 (loads ~2.5 GB; machine rules)")
def test_real_tdt_v3_counts_and_decoder_consistency():
    torch.set_num_threads(2)
    import json

    from audioforge.data import load_wav
    m = import_nemo(REAL_TDT)
    info = m.import_info
    assert (info["n_tensors"], info["n_loaded"], info["zero_biases"]) == (725, 723, 264)
    assert info["n_loaded"] + info["zero_biases"] == len(m.state_dict())
    assert m.tokenizer.vocab_size == 8192 and m.heads["rnnt"].durations == DURS
    assert m.cfg["nemo_source"]["license"] == "CC-BY-4.0"
    line = json.loads((ROOT / "data/librispeech/test-clean-first200.jsonl").read_text().splitlines()[1])
    x = torch.as_tensor(load_wav(line["audio_filepath"], 16000))[None]
    with torch.no_grad():
        enc, elen = m.encode(x, torch.tensor([x.shape[1]]))
    h = m.heads["rnnt"]
    off = h._greedy(enc[0, :int(elen[0])])
    s = h.greedy_stream()
    for i in range(0, int(elen[0]), 3):
        s.feed(enc[0, i:min(i + 3, int(elen[0]))])
    assert s.tokens == off
    assert m.tokenizer.decode(off) == "Stuff it into you, his belly counseled him."


# --------------------------------------------------------------------------- serve --final-asr
import asyncio
import json
import time

import numpy as np

from audioforge.final_asr import FinalASRWorker
from audioforge.serve import (
    FRAME_SAMPLES,
    SR,
    Session,
    SessionConfig,
    validate,
)
from tests.test_serve import (
    EnergyEngine,
    _diar_model,
    _speech_silence,
    _talky_asr_model,
    _with_server,
)

_FA = {}


def _fa_engine(spec=None, mode="thread", debug=True):
    key = (spec, mode, debug)
    if key not in _FA:
        fa = FinalASRWorker(spec, mode=mode, threads=1) if spec else None
        _FA[key] = EnergyEngine(_talky_asr_model(), _diar_model(), name="tiny", threads=1, final_asr=fa, debug=debug)
    return _FA[key]


def _run_session(eng, x, fs=320, cfg=None):
    s = Session(eng, cfg or SessionConfig(timeout_ms=400))
    msgs = []
    for i in range(0, len(x), fs):
        msgs += s.process(x[i:i + fs])
    msgs += s.finish()
    return s, msgs


def test_final_asr_off_keeps_protocol_and_on_only_adds():
    """Flag off: exactly the protocol's keys, no source anywhere. Flag on: the same streaming messages (decisions,
    texts, times) plus source="stream" on the streaming finals; the offline finals are extra messages."""
    x = _speech_silence()
    _, off = _run_session(_fa_engine(None, debug=False), x)
    _, on = _run_session(_fa_engine("fake:0", debug=False), x)
    for m in off:
        validate(m)
        assert "source" not in m and "final_asr" not in m
    for m in on:
        validate(m)
    strip = []
    for m in on:
        if m["type"] == "final" and m["source"] != "stream":
            continue
        m = {k: v for k, v in m.items() if k not in ("source", "final_asr", "final_latency_ms", "final_asr_rss_mb")}
        strip.append(m)
    drop = ("rtf", "chunk_ms_p50", "chunk_ms_p95", "peak_rss_mb", "first_partial_ms")  # timing-dependent stats
    norm = [json.dumps({k: v for k, v in m.items() if not (m["type"] == "stats" and k in drop)}) for m in strip]
    assert norm == [json.dumps({k: v for k, v in m.items() if not (m["type"] == "stats" and k in drop)}) for m in off]


def test_final_asr_span_selection_onset_preroll_postroll():
    """The span: first served-VAD speech frame of the segment - 0.3 s, never before the previous span's end, to the
    last speech frame + 0.5 s, never after the decision time; no speech -> an immediate empty final."""
    eng = _fa_engine("fake:0")
    s = Session(eng, SessionConfig())
    n = 120 * FRAME_SAMPLES
    s._fbuf = np.arange(n, dtype=np.float32)  # sample value = its index
    s._fbuf0 = 0
    vad = np.zeros(120)
    vad[10:30] = 0.9  # segment 1: speech frames 10..29
    vad[52:57] = 0.9  # segment 2: speech right after the first span's end (ends 1 frame before the decision)
    s._fvad = vad.tolist()
    out = []
    s._final_job(50 * FRAME_SAMPLES, 50, 50 * 0.08, 0, out)
    (j1,) = s.take_final_jobs()
    assert out == []
    a1 = j1["audio"]
    assert a1[0] == 10 * FRAME_SAMPLES - int(0.3 * SR) and a1[-1] == 30 * FRAME_SAMPLES + int(0.5 * SR) - 1
    assert j1["msg"]["start"] == round(a1[0] / SR, 3) and j1["msg"]["end"] == round((a1[-1] + 1) / SR, 3)
    # segment 2: pre-roll would reach back into segment 1's span (ends at 46400 = frame 36.25): clamped to it
    s._final_job(58 * FRAME_SAMPLES, 58, 58 * 0.08, 1, out)
    (j2,) = s.take_final_jobs()
    assert j2["audio"][0] == max(52 * FRAME_SAMPLES - int(0.3 * SR), 30 * FRAME_SAMPLES + int(0.5 * SR))
    assert j2["audio"][-1] == 58 * FRAME_SAMPLES - 1  # speech continues past the decision: post-roll capped at the decision time
    # segment 3 has no speech: an empty final at once, no job
    s._final_job(100 * FRAME_SAMPLES, 100, 8.0, 0, out)
    assert s.take_final_jobs() == [] and len(out) == 1
    assert out[0]["text"] == "" and out[0]["start"] is None and out[0]["source"] == "fake"
    validate(out[0])


def test_final_asr_session_jobs_cover_each_turn():
    """Speech 0-1 s / 2.6-3.6 s with timeout turn_ends: one offline final per streaming final, same t, each span
    inside its segment and starting at the speech onset (VAD always on in this tiny model: the segment start)."""
    eng = _fa_engine("fake:0")
    s, msgs = _run_session(eng, _speech_silence())
    fin = [m for m in msgs if m["type"] == "final"]
    jobs_done = s.run_final_jobs_sync()
    stream = [m for m in fin if m["source"] == "stream"]
    offline = [m for m in fin if m["source"] != "stream"]
    assert len(stream) == 3 and not offline  # all offline finals are jobs (VAD on everywhere)
    assert [m["t"] for m in jobs_done] == [m["t"] for m in stream]
    assert jobs_done[0]["start"] == 0.0 and jobs_done[0]["end"] <= stream[0]["t"]
    for a, b in itertools.pairwise(jobs_done):
        assert b["start"] >= a["end"]
    for m in jobs_done:
        validate(m)
        assert m["text"] == f"<{m['end'] - m['start']:.2f}s>"  # the fake returns the span's duration


def _socket_run(eng, x, cfg):
    async def body(url):
        from websockets.asyncio.client import connect
        out = []
        async with connect(url) as ws:
            out.append((time.perf_counter(), json.loads(await ws.recv())))
            await ws.send(json.dumps({"type": "config", **cfg}))
            pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()

            async def rx():
                async for m in ws:
                    out.append((time.perf_counter(), json.loads(m)))
            task = asyncio.create_task(rx())
            for i in range(0, len(pcm), 640 * 4):  # 80 ms blocks at ~4x real time
                await ws.send(pcm[i:i + 640 * 4])
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "end"}))
            await task
        return out
    return asyncio.run(_with_server(eng, body))


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_final_asr_socket_protocol_and_non_blocking(mode):
    """Over the socket: every offline final arrives after its streaming final, carries latency_ms >= the pass time,
    the stats (last message) wait for all of them; while a 1.2 s pass runs, frame messages keep flowing (the
    streaming loop is not blocked). process mode uses a pass that spins the CPU in Python (holds a GIL)."""
    spec = "fake:1200:busy" if mode == "process" else "fake:1200"
    eng = _fa_engine(spec, mode=mode)
    x = _speech_silence(((1.0, 0.2), (2.0, 0.0), (1.0, 0.2), (2.0, 0.0)))
    out = _socket_run(eng, x, {"turn_policy": "timeout", "timeout_ms": 400})
    msgs = [m for _, m in out]
    for m in msgs:
        validate(m, debug=True)
    assert msgs[0]["type"] == "ready" and msgs[0]["final_asr"] == "fake"
    assert msgs[-1]["type"] == "stats"
    st = msgs[-1]
    off = [(t, m) for t, m in out if m["type"] == "final" and m["source"] == "fake"]
    stream = [(t, m) for t, m in out if m["type"] == "final" and m["source"] == "stream"]
    assert len(off) == len(stream) == st["final_latency_ms"]["fake"]["n"] == 3
    for (ts, ms), (to, mo) in zip(stream, off):
        # latency_ms >= the 1.2 s pass; serve takes t_sub just after submitting, so a descheduled event loop (a loaded
        # laptop) shortens it (978 ms seen under swap): the strict 1150 only with BULLETPROOF_STRICT_TIMING=1
        assert ms["t"] == mo["t"] and to > ts and mo["latency_ms"] >= 1150 / SLACK
    # the first pass runs from the first turn_end for 1.2 s: frames of later audio arrive meanwhile
    t0, t1 = stream[0][0], off[0][0]
    during = [t for t, m in out if m["type"] in ("frame", "frames") and t0 < t < t1]
    assert len(during) >= 5
    gaps = np.diff([t0] + during + [t1])
    # 0.6 s on a quiet machine (BULLETPROOF_STRICT_TIMING=1); x SLACK by default: the process variant spins a CPU on
    # purpose, and the suite shares the laptop with other jobs
    assert gaps.max() < 0.6 * SLACK, gaps.max()
    if mode == "process":
        assert st["final_asr_rss_mb"] is not None and st["final_asr_rss_mb"] > 0


def test_validate_final_asr_keys():
    validate({"type": "final", "t": 1.0, "text": "a", "speaker": 0, "source": "stream"})
    validate({"type": "final", "t": 1.0, "text": "a", "speaker": 0, "source": "tdt_v3", "start": 0.2, "end": 0.9,
              "latency_ms": 120.0})
    with pytest.raises(ValueError):
        validate({"type": "final", "t": 1.0, "text": "a", "speaker": 0, "source": "tdt_v3"})
    with pytest.raises(ValueError):
        validate({"type": "final", "t": 1.0, "text": "a", "speaker": 0, "latency_ms": 3.0})
    with pytest.raises(ValueError):
        validate({"type": "stats", "rtf": 0.1, "chunk_ms_p50": 1, "chunk_ms_p95": 1, "first_partial_ms": None,
                  "peak_rss_mb": 1.0, "final_asr": "tdt_v3"})


def test_cli_final_asr_flags():
    from audioforge.serve import main
    with pytest.raises(SystemExit):
        main(["--asr", "x", "--diar", "y", "--final-asr-worker", "bogus"])


# --------------------------------------------------------------------------- serve --asr-lookahead (dual lookahead)
def _la_engine(la=3, spec=None, talky=True):
    key = ("la", la, spec, talky)
    if key not in _FA:
        from tests.test_serve import _asr_model
        fa = FinalASRWorker(spec, mode="thread", threads=1) if spec else None
        _FA[key] = EnergyEngine(_talky_asr_model() if talky else _asr_model(), _diar_model(), name="tiny", threads=1,
                                final_asr=fa, asr_lookahead=la, debug=True)
    return _FA[key]


@pytest.mark.parametrize("talky", [True, False])
def test_lookahead_heads_bit_identical_and_text_partition(talky):
    """--asr-lookahead: every frame / turn_end / partial / streaming final is identical to the single-pass server
    (heads untouched); the lookahead finals partition exactly the tokens of a standalone streaming pass at [8, 3]."""
    from audioforge.model import StreamingSession
    x = _speech_silence()
    eng1 = EnergyEngine(_la_engine(talky=talky).asr, _diar_model(), name="tiny", threads=1, debug=True)
    _, one = _run_session(eng1, x)
    s2, two = _run_session(_la_engine(talky=talky), x)
    for m in two:
        validate(m, debug=True)
    la = [m for m in two if m["type"] == "final" and m["source"] == "lookahead"]
    rest = [{k: v for k, v in m.items() if k != "source"} for m in two
            if m["type"] not in ("stats",) and not (m["type"] == "final" and m["source"] == "lookahead")]
    assert rest == [m for m in one if m["type"] != "stats"]
    stream = [m for m in two if m["type"] == "final" and m["source"] == "stream"]
    assert len(la) == len(stream) and [m["t"] for m in la] == [m["t"] for m in stream]
    ref = StreamingSession(eng1.asr, att_context_size=[8, 3])
    for i in range(0, len(x), 320):
        ref.feed(x[i:i + 320])
    ref.feed(np.zeros(0, np.float32), final=True)
    assert "".join(m["text"] for m in la) == ref.text
    assert s2.la.tokens == ref.tokens
    for a, b in itertools.pairwise(la):
        assert a["end"] == b["start"]
    st = two[-1]
    assert st["final_asr"] == "lookahead" and st["final_latency_ms"]["lookahead"]["n"] == len(la)


def test_lookahead_due_after_margin_and_latency_is_audio_time():
    """The lookahead final of a segment is emitted once its pass has decoded 3 frames past the last VAD speech frame
    (VAD always on here: the streaming cut frame); latency_ms = max(0, that frame's ready time - decision time)."""
    eng = _la_engine()
    x = _speech_silence()
    s, msgs = _run_session(eng, x)
    la = [m for m in msgs if m["type"] == "final" and m["source"] == "lookahead"]
    first = la[0]
    cut = round(first["end"] / 0.08)
    ready = s.la.ready_t(cut - 1)
    assert first["latency_ms"] == round(max(0.0, ready - first["t"]) * 1000, 1)
    # emitted in the first process() call after that audio time, i.e. not before it
    idx = msgs.index(first)
    later_frames = [m["t"] for m in msgs[:idx] if m["type"] == "frame"]
    assert max(later_frames) + 0.2 >= ready


def test_lookahead_and_final_asr_together():
    eng = _la_engine(spec="fake:0")
    s, msgs = _run_session(eng, _speech_silence())
    done = s.run_final_jobs_sync()
    for m in msgs + done:
        validate(m, debug=True)
    srcs = [m["source"] for m in msgs + done if m["type"] == "final"]
    assert srcs.count("stream") == srcs.count("lookahead") == srcs.count("fake") == 3
    assert msgs[0]["type"] == "frame" and s.e.ready_msg()["final_asr"] == "lookahead,fake"
