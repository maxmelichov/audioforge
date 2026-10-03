"""The optional speech-detector head (heads.speech) in the streaming session, and the RNNT beam
search (split invariance of the streaming decoder). Tiny random models, CPU, no checkpoints."""
import numpy as np
import torch

from audioforge.heads.asr import RNNTHead
from audioforge.model import SpeechModel
from audioforge.server.streams import ASRStream
from audioforge.tokenizer import CharTokenizer


def _model(seed=0, speech=True):
    torch.manual_seed(seed)
    heads = {"rnnt": {"type": "rnnt", "pred_hidden": 16, "joint_hidden": 16},
             "vad": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": [0]}}
    if speech:
        heads["speech"] = {"type": "frame", "key": "speech", "hidden": 8, "from_layers": [0, 1], "weight": 0.0}
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, 1], "dropout": 0.0},
           "heads": heads}
    return SpeechModel(cfg, CharTokenizer(list("abc "))).eval()


def _frames(m, x, step=555):
    s = ASRStream(m, turn=None)
    out = []
    for i in range(0, len(x), step):
        out += s.feed_frames(x[i:i + step])
    out += s.feed_frames(np.zeros(0, np.float32), final=True)
    return out


def test_speech_head_streams_and_vad_unchanged():
    x = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.1).astype(np.float32)
    with_sp, without = _frames(_model(speech=True), x), _frames(_model(speech=False), x)
    assert all("speech" in f for f in with_sp) and not any("speech" in f for f in without)
    # the turn / TS-VAD input (heads.vad) is the same with or without the speech head
    assert [round(f["vad"], 6) for f in with_sp] == [round(f["vad"], 6) for f in without]
    # streamed speech probabilities equal the whole-utterance masked forward
    m = _model(speech=True)
    with torch.no_grad():
        xx = torch.from_numpy(x)[None]
        enc, elen, hid = m.encode(xx, torch.tensor([xx.shape[1]]), [8, 1], return_hidden=True)
        p = m.heads["speech"](m.head_input("speech", enc, hid)).sigmoid()[0, : int(elen[0])].numpy()
    got = np.array([f["speech"] for f in with_sp])[: len(p)]
    assert np.allclose(got, p[: len(got)], atol=1e-4)


def test_beam_stream_split_invariance():
    torch.manual_seed(0)
    h = RNNTHead(16, 12, pred_hidden=8, joint_hidden=8).eval()
    for _s in range(5):
        f = torch.randn(30, 16) * 2
        whole = h.beam_search(f, beam=4, max_sym=3)
        st = h.beam_stream(4, 3)
        for i in range(0, 30, 7):
            st.feed(f[i:i + 7])
        assert st.tokens == whole
        assert all(0 <= t < 12 for t in whole)
        assert st.tokens[: len(st.stable)] == st.stable


def test_vad_head_policy_reset_thr():
    from audioforge.server.policies import VadHeadPolicy
    vad = [0.9] * 5 + [0.3, 0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    def fire(rt):
        pol = VadHeadPolicy(0.99, 2, 4, 0.4, others=None, reset_thr=rt)
        return [v for v, x in enumerate(vad) if pol.update(0.0, x) is not None]
    assert fire(None) == [8]  # fallback: 4 frames since the last frame >= 0.4 (frame 4)
    assert fire(0.25) == [11]  # the 0.3 frames (unsure) restart the silence clock: 4 frames after frame 7


def test_asr_stream_beam_cut_segments():
    m = _model(speech=False)
    x = (np.random.default_rng(1).standard_normal(16000 * 3) * 0.1).astype(np.float32)
    s = ASRStream(m, turn=None, beam=4)
    for i in range(0, len(x), 700):
        s.feed_frames(x[i:i + 700])
    s.feed_frames(np.zeros(0, np.float32), final=True)
    frames = torch.cat(s.beam_frames, 0)
    n = s.n_frames
    assert frames.shape[0] == n
    cut = n // 2
    first = s.beam_cut(cut)
    assert first == m.heads["rnnt"].beam_search(frames[:cut], beam=4, max_sym=3)
    second = s.beam_cut(n)  # the frames after the first cut were re-fed into a fresh beam
    assert second == m.heads["rnnt"].beam_search(frames[cut:], beam=4, max_sym=3)
    assert s.beam_frames == [] and s.beam_b0 == n
