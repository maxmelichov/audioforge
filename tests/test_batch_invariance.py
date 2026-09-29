"""Variable-length batch invariance: an utterance encoded alone must give the same result as the
same utterance zero-padded inside a batch with longer ones (valid frames only)."""
import numpy as np
import pytest
import torch

from audioforge.data import Collate, ToneLanguage
from audioforge.features import LogMel
from audioforge.model import SpeechModel
from audioforge.modules.fastconformer import FastConformerEncoder
from audioforge.tokenizer import CharTokenizer

CONTEXTS = [[-1, -1], [4, 1], [5, 1], [-1, 1], [3, 0]]  # full, chunked, L not a multiple of chunk, ...


def _randomize(m):
    """Speaker kernels start as identity; give every parameter a non-trivial value."""
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)


@pytest.mark.parametrize("att", CONTEXTS)
@pytest.mark.parametrize("causal", [False, True])
def test_encoder_batch_invariance(causal, att):
    torch.manual_seed(0)
    enc = FastConformerEncoder(feat_in=40, d_model=32, n_layers=2, n_heads=2, subsampling_channels=16,
                               dropout=0.0, causal=causal, att_context_size=att,
                               speaker_kernel_layers=[1]).eval()
    _randomize(enc)
    lens = [203, 331, 97, 256]
    feats = torch.randn(len(lens), 40, max(lens))  # padding holds garbage on purpose
    act = torch.rand(len(lens), max(lens) // 8 + 1)
    with torch.no_grad():
        full, flen = enc(feats, torch.tensor(lens), att, spk_act=act)
        for i, n in enumerate(lens):
            one, olen = enc(feats[i:i + 1, :, :n], torch.tensor([n]), att, spk_act=act[i:i + 1])
            k = int(olen[0])
            assert k == int(flen[i]) == one.shape[1]
            assert torch.allclose(one[0], full[i, :k], atol=1e-5), (i, (one[0] - full[i, :k]).abs().max())
            assert full[i, k:].abs().max() == 0 if k < full.shape[1] else True


@pytest.mark.parametrize("normalize", ["per_feature", "fixed"])
def test_preprocessor_batch_invariance(normalize):
    pp = LogMel(n_mels=40, normalize=normalize, dither=0.0).eval()
    rng = np.random.default_rng(0)
    lens = [12345, 16000, 8007]
    audio = torch.zeros(len(lens), max(lens))
    for i, n in enumerate(lens):
        audio[i, :n] = torch.from_numpy(rng.standard_normal(n).astype(np.float32) * 0.1)
    full, flen = pp(audio, torch.tensor(lens))
    for i, n in enumerate(lens):
        one, olen = pp(audio[i:i + 1, :n], torch.tensor([n]))
        k = int(olen[0])
        assert k == int(flen[i])
        assert torch.allclose(one[0], full[i, :, :k], atol=1e-4), (i, (one[0] - full[i, :, :k]).abs().max())


def _model(causal, att):
    tok = CharTokenizer(sorted(set("".join(ToneLanguage.LEXICON))) + [" "], specials=["<|en|>", "<|transcribe|>"])
    enc = {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "dropout": 0.0,
           "causal": causal, "att_context_size": att}
    small = {"pred_hidden": 16, "joint_hidden": 16}
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "per_feature", "dither": 0.0}, "encoder": enc,
           "heads": {"ctc": {"type": "ctc"}, "tdt": {"type": "tdt", **small}, "rnnt": {"type": "rnnt", **small},
                     "aed": {"type": "aed", "d_dec": 16, "n_layers": 1, "n_heads": 2, "dropout": 0.0},
                     "diar": {"type": "sortformer", "num_spks": 2, "d_hidden": 16, "n_layers": 1, "dropout": 0.0},
                     "spk": {"type": "speaker", "num_speakers": 3, "emb_dim": 8},
                     "vad": {"type": "frame", "key": "vad"}}}
    torch.manual_seed(0)
    return SpeechModel(cfg, tok).eval(), tok


@pytest.mark.parametrize("causal,att", [(False, [-1, -1]), (False, [4, 1]), (True, [-1, -1]), (True, [5, 1])])
def test_heads_batch_invariance(causal, att):
    """Losses, decodes and frame outputs of every head: batch == per-utterance."""
    m, tok = _model(causal, att)
    rng = np.random.default_rng(1)
    exs = []
    for n, text in [(14007, "go left"), (21000, "stop red one"), (9123, "up")]:
        T = ToneLanguage.n_frames(n)
        exs.append(dict(audio=rng.standard_normal(n).astype(np.float32) * 0.1, text=text,
                        prompt="<|en|><|transcribe|>", speaker=int(rng.integers(3)),
                        vad=(rng.random(T) > 0.5).astype(np.float32),
                        spk_targets=(rng.random((T, 2)) > 0.5).astype(np.float32)))
    col = Collate(tok)
    with torch.no_grad():
        full = m(col(exs))
        solo = [m(col([ex])) for ex in exs]
        # per-utterance-normalized losses: the batch value is the mean of the solo values
        for name in ("loss_ctc", "loss_tdt", "loss_rnnt", "loss_spk"):
            want = sum(float(s[name]) for s in solo) / len(solo)
            assert abs(float(full[name]) - want) < 1e-4 * max(1.0, abs(want)), (name, float(full[name]), want)
        # token/frame-weighted losses
        ntok = [len(tok.encode(ex["text"])) + 1 for ex in exs]
        want = sum(float(s["loss_aed"]) * k for s, k in zip(solo, ntok)) / sum(ntok)
        assert abs(float(full["loss_aed"]) - want) < 1e-4 * max(1.0, want)
        nfr = [len(ex["vad"]) for ex in exs]
        want = sum(float(s["loss_vad"]) * k for s, k in zip(solo, nfr)) / sum(nfr)
        assert abs(float(full["loss_vad"]) - want) < 1e-4 * max(1.0, want)

        audios = [ex["audio"] for ex in exs]
        x, lens = m._pad(audios)
        enc, elen = m.encode(x, lens)
        for i, a in enumerate(audios):
            e1, l1 = m.encode(*m._pad([a]))
            k = int(l1[0])
            assert k == int(elen[i])
            assert torch.allclose(e1[0], enc[i, :k], atol=1e-5)
            h = m.heads
            assert torch.allclose(h["diar"](e1, l1)[0], h["diar"](enc, elen)[i, :k], atol=1e-4)
            assert torch.allclose(h["spk"].embed(e1, l1)[0], h["spk"].embed(enc, elen)[i], atol=1e-4)
            assert torch.allclose(h["vad"].decode(e1, l1)[0], h["vad"].decode(enc, elen)[i, :k], atol=1e-5)
            for name in ("ctc", "tdt", "rnnt"):
                assert h[name].decode(e1, l1) == [h[name].decode(enc, elen)[i]]
            p = [tok.encode(exs[i]["prompt"])]
            assert h["aed"].decode(e1, l1, prompt=p, max_new=8) == \
                [h["aed"].decode(enc, elen, prompt=p * len(audios), max_new=8)[i]]
