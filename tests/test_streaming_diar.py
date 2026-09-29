"""Streaming Sortformer with an arrival-order speaker cache (audioforge/streaming_diar.py)."""
import numpy as np
import pytest
import torch

from audioforge.data import ToneLanguage
from audioforge.model import SpeechModel
from audioforge.streaming_diar import (
    AOSCConfig,
    SpeakerCache,
    StreamingDiarizer,
    StreamingSpeakerASR,
    compose_conversation,
    select_cache_frames,
    slot_flips,
)
from audioforge.tokenizer import CharTokenizer


def _model(causal: bool, seed: int = 0):
    torch.manual_seed(seed)
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "fixed" if causal else "per_feature", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": causal,
                       "att_context_size": [8, 1] if causal else [-1, -1], "dropout": 0.0,
                       "speaker_kernel_layers": [0]},
           "heads": {"diar": {"type": "sortformer", "num_spks": 4, "d_hidden": 16, "n_layers": 1, "dropout": 0.0,
                              "pos_emb": True},
                     "asr": {"type": "rnnt", "pred_hidden": 16, "joint_hidden": 16, "condition_on_speaker": True}},
           "streaming": {"chunk_len": 4, "chunk_right_context": 2, "fifo_len": 6, "spkcache_len": 8,
                         "spkcache_update_period": 4, "sil_frames": 2}}
    return SpeechModel(cfg, CharTokenizer(list("abc "))).eval()


def _audio(n=16000 * 2 + 333, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 0.1).astype(np.float32)


def _offline_probs(m, x):
    enc, el = m.encode(torch.from_numpy(x)[None], torch.tensor([len(x)]))
    return m.heads["diar"](enc, el).sigmoid()[0]


def _stream(d, x, push):
    outs = [d.feed(x[i:i + push]) for i in range(0, len(x), push)]
    outs.append(d.feed([], final=True))
    return torch.cat(outs)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("n", [16000 * 2 + 333, 12800 * 2, 4000])
def test_streaming_frame_count_equals_offline(causal, n):
    m = _model(causal)
    x = _audio(n)
    off = _offline_probs(m, x)
    p = _stream(StreamingDiarizer(m), x, 777)  # odd push size, cache + fifo active
    assert p.shape == off.shape


@pytest.mark.parametrize("causal", [True, False])
def test_single_chunk_without_cache_equals_offline(causal):
    m = _model(causal)
    x = _audio()
    off = _offline_probs(m, x)
    d = StreamingDiarizer(m, chunk_len=10 ** 6, chunk_right_context=0, spkcache_len=0, fifo_len=0)
    p = _stream(d, x, 1000)
    assert d.steps == 1
    assert torch.allclose(p, off, atol=1e-5), (p - off).abs().max()


def test_cache_selection_rule():
    # frames 0-5 speaker 0 (confidence rising), 6-9 speaker 1, 10-11 overlap, 12-13 silence
    p = torch.full((14, 4), 0.01)
    p[0:6, 0] = torch.linspace(0.6, 0.99, 6)
    p[6:10, 1] = 0.9
    p[10:12, 0] = p[10:12, 1] = 0.9
    idx = select_cache_frames(p, torch.arange(14), cap=8, sil_frames=2, strong_boost_rate=0.75)
    assert len(idx) == 8
    # slot-major layout: speaker-0 frames (time order), then speaker-1 frames, then silence
    spk = [0 if p[i, 0] > 0.5 and p[i, 1] <= 0.5 else 1 if p[i, 1] > 0.5 and p[i, 0] <= 0.5 else -1
           for i in idx.tolist()]
    first1 = spk.index(1)
    assert all(s == 0 for s in spk[:first1]) and all(s in (1, -1) for s in spk[first1:])
    assert idx.tolist()[-2:] == [12, 13]  # silence at the end
    assert not set(idx.tolist()) & {10, 11}  # overlapped frames are ambiguous: never kept
    assert idx[:first1].tolist() == sorted(idx[:first1].tolist())
    assert 5 in idx.tolist()  # the most confident speaker-0 frame survives


def _ideal_arrival_head(c, f, x, r):
    """An ideal arrival-order Sortformer on one-hot 'speaker embeddings' (0 vector = silence):
    the k-th distinct speaker seen in the *input sequence* gets output slot k."""
    seq = torch.cat([c, f, x, r])
    probs = torch.full((len(seq), 4), 0.02)
    order = []
    for i, e in enumerate(seq):
        if e.abs().sum() == 0:
            continue
        spk = int(e.argmax())
        if spk not in order:
            order.append(spk)
        probs[i, order.index(spk)] = 0.97
    return probs


def _run_cache(emb, cfg):
    cache, out = SpeakerCache(cfg, 4), []
    C, R = cfg.chunk_len, cfg.chunk_right_context
    for t in range(0, len(emb), C):
        cp, _ = cache.step(emb[t:t + C], emb[t + C:t + C + R], t, _ideal_arrival_head)
        out.append(cp)
    return torch.cat(out), cache


def test_speaker_slot_stability_with_cache():
    # hand-made A B A B A B conversation; every turn (30 frames) is longer than the FIFO, so without a
    # speaker cache the window forgets A while B talks and re-numbers speakers by what it can see.
    turns, sil = [0, 1, 0, 1, 0, 1], torch.zeros(5, 4)
    emb, ref = [], []
    for s in turns:
        emb += [torch.eye(4)[s].repeat(30, 1), sil]
        ref += [torch.eye(4)[s].repeat(30, 1), sil]
    emb, ref = torch.cat(emb), torch.cat(ref)
    cfg = AOSCConfig(chunk_len=4, chunk_right_context=2, fifo_len=8, spkcache_len=16, spkcache_update_period=4,
                     sil_frames=2)
    p, cache = _run_cache(emb, cfg)
    assert len(p) == len(emb)
    b = (p > 0.5).float()
    flips, trans = slot_flips(b, ref, cfg.chunk_len)
    assert trans > 40 and flips == 0
    assert torch.equal(b[:, :2], ref[:, :2])  # A is always slot 0, B always slot 1
    assert cache.n_compress > 0 and len(cache.c_emb) <= cfg.spkcache_len
    # the cache holds both speakers, A's block first (arrival order)
    spk = cache.c_emb[:, :2].argmax(1)[cache.c_emb.abs().sum(1) > 0]
    assert spk[0] == 0 and set(spk.tolist()) == {0, 1} and (spk.diff() >= 0).all()
    # ablation: no cache -> slots flip whenever the FIFO has forgotten the earlier speaker
    p0, _ = _run_cache(emb, AOSCConfig(**{**cfg.__dict__, "spkcache_len": 0}))
    f0, _ = slot_flips((p0 > 0.5).float(), ref, cfg.chunk_len)
    assert f0 >= 4


@pytest.mark.parametrize("causal", [True, False])
def test_streaming_speaker_asr_smoke(causal):
    m = _model(causal)
    lang = ToneLanguage(seed=0)
    import random
    conv = compose_conversation(lang, random.Random(0), [0, 1], 2)
    x = conv["audio"]
    sa = StreamingSpeakerASR(m, threshold=0.3, min_active_frames=1)
    for i in range(0, len(x), 2000):
        texts = sa.feed(x[i:i + 2000])
        assert isinstance(texts, dict)
    texts = sa.feed([], final=True)
    off = _offline_probs(m, x)
    assert len(sa.diar.all_probs) == len(off)
    assert sa.slots, "no slot became active"
    assert all(isinstance(t, str) for t in texts.values())
    assert sa.asr_frames_computed >= len(off)  # >= one extra encoder pass per active slot
    assert isinstance(sa.transcript(), list)
