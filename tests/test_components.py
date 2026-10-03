
import pytest
import torch

from audioforge import catalog
from audioforge.codec import FSQ, AudioCodec
from audioforge.heads.audio import sort_by_arrival
from audioforge.metrics import eer, frame_der, wer
from audioforge.speech_llm import SpeechLLM, TinyCausalLM
from audioforge.tokenizer import CharTokenizer, SentencePieceTokenizer


@pytest.fixture(autouse=True)
def _seeded():
    """Fixed torch / numpy / random seeds per test: random inputs do not depend on which tests ran before."""
    import random

    import numpy as np
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def test_fsq_roundtrip_and_codebook():
    fsq = FSQ([8, 5, 5, 4], num_codebooks=3)
    assert fsq.codebook_size == 800
    z = torch.randn(2, 12, 7) * 3
    zq, codes = fsq(z)
    assert codes.shape == (2, 7, 3) and codes.min() >= 0 and codes.max() < 800
    assert torch.allclose(fsq.decode_codes(codes), zq, atol=1e-6)


def test_codec_shapes_12_5fps():
    c = AudioCodec(channels=8, strides=(4, 4, 8, 10), levels=(8, 5, 5, 4), num_codebooks=2)
    assert c.frame_rate == 12.5
    x = torch.randn(2, 16000) * 0.1
    y, codes = c(x)
    assert y.shape == x.shape and codes.shape == (2, 13, 2)
    assert c.decode(codes).shape[-1] == 13 * 1280


def test_sort_by_arrival():
    t = torch.zeros(1, 6, 3)
    t[0, 4:, 0] = 1  # speaker 0 arrives last
    t[0, 1:3, 2] = 1  # speaker 2 arrives first
    s = sort_by_arrival(t)
    assert s[0, 1, 0] == 1 and s[0, 4, 1] == 1 and s[0, :, 2].sum() == 0


def test_metrics():
    assert wer(["a b c"], ["a x c"]) == 1 / 3
    ref = torch.tensor([[1, 0], [1, 1], [0, 1]]).float()
    assert frame_der(ref[:, [1, 0]], ref) == 0.0
    assert eer(torch.tensor([0.9, 0.8, 0.2, 0.1]), torch.tensor([1, 1, 0, 0])) == 0.0


def test_tokenizers_with_specials():
    specials = ["<|en|>", "<|transcribe|>"]
    ct = CharTokenizer.train(["hello world"], specials=specials)
    ids = ct.encode("<|en|><|transcribe|>hello")
    assert ids[:2] == [ct.token_id("<|en|>"), ct.token_id("<|transcribe|>")]
    assert ct.decode(ids) == "hello"
    sp = SentencePieceTokenizer.train(["hello world", "yes no stop go"] * 50, vocab_size=40, specials=specials)
    ids = sp.encode("<|en|> hello")
    assert sp.token_id("<|en|>") in ids and sp.decode(ids).strip() == "hello"


def test_speech_llm_forward_generate():
    tok = CharTokenizer.train(["abc"], specials=["<|audio|>"])
    llm = TinyCausalLM(tok.vocab_size, 32, 1, 2)
    m = SpeechLLM(dict(d_model=32, n_layers=1, n_heads=2, subsampling_channels=8), llm, tok)
    x = torch.randn(2, 8000) * 0.1
    prompt = tok.encode("a<|audio|>b")
    loss = m(x, torch.tensor([8000, 6000]), [prompt] * 2, [tok.encode("abc"), tok.encode("ab")])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in llm.parameters())  # LLM frozen
    assert isinstance(m.generate(x, torch.tensor([8000, 8000]), prompt, max_new=3), list)


def test_catalog_search(tmp_path):
    models = [
        {"id": "nvidia/parakeet-tdt-0.6b-v2", "pipeline_tag": "automatic-speech-recognition", "downloads": 10,
         "tags": ["license:cc-by-4.0", "en"]},
        {"id": "nvidia/diar_streaming_sortformer_4spk-v2", "downloads": 5, "tags": []},
        {"id": "nvidia/Cosmos-Tokenizer-CI8x8-Lidar", "downloads": 1, "tags": []},
    ]
    aud = [m for m in models if catalog.is_audio(m)]
    assert len(aud) == 2
    entries = [catalog.classify(m, "Params-600M 16kHz cache-aware") for m in aud]
    assert entries[0].decoder == "TDT" and entries[0].encoder == "FastConformer" and entries[0].params == "600M"
    assert catalog.search(entries, family="diar")[0].id.endswith("v2")
    catalog.save(entries, tmp_path / "c.json")
    assert catalog.load(tmp_path / "c.json")[0].id == entries[0].id


def test_onnx_export(tmp_path):
    from audioforge.model import SpeechModel
    from audioforge.train import export_onnx
    cfg = {"encoder": {"d_model": 32, "n_layers": 1, "n_heads": 2, "subsampling_channels": 8},
           "heads": {"ctc": {"type": "ctc"}}}
    m = SpeechModel(cfg, CharTokenizer(list("ab")))
    p = export_onnx(m, tmp_path / "enc.onnx", seconds=1.0)
    import onnx
    onnx.checker.check_model(str(p))


def test_fsq_levels_reachable_straight_through_and_roundtrip():
    for L in (2, 3, 4, 5, 7, 8):
        fsq = FSQ([L], num_codebooks=1)
        z = torch.linspace(-8, 8, 4001).view(1, 1, -1).requires_grad_(True)
        zq, codes = fsq(z)
        assert sorted(codes.unique().tolist()) == list(range(L)), L  # every level used, indices 0..L-1
        assert torch.equal(fsq.decode_codes(codes), zq.detach()), L  # code <-> value round trip
        assert zq.min() >= -1 and zq.max() <= 1
        assert L == 2 or zq.detach()[0, 0, 2000] == 0  # z=0 -> centre level (L=2 has none)
        zq.sum().backward()  # straight-through: gradient of the bounded value, positive for |z| small
        assert torch.isfinite(z.grad).all() and (z.grad[0, 0, 1500:2500] > 0).all(), L


def test_frame_der_counts_extra_predicted_speakers():
    ref = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]]).float()
    pred = torch.cat([ref, torch.ones(4, 1)], 1)  # correct + one spurious always-on speaker
    assert frame_der(pred, ref) == 1.0  # 4 false-alarm frames / 4 speech frames
    assert frame_der(ref, pred) == 0.5  # and a missed third speaker counts as miss


def test_codec_odd_strides_keep_length():
    c = AudioCodec(channels=4, strides=(3, 5, 7), levels=(8, 5, 5, 4), num_codebooks=1)
    x = torch.randn(2, 105 * 4 + 17)
    y, codes = c(x)
    assert y.shape == x.shape and codes.shape == (2, 5, 1)
    assert c.decode(codes).shape[-1] == 5 * 105


def test_aed_supervises_text_and_eos_only():
    """Decoder input = <bos> prompt text; loss only on text + <eos> (prompt masked); padding ignored."""
    import torch.nn.functional as F

    from audioforge.heads.asr import AEDHead
    torch.manual_seed(0)
    h = AEDHead(8, vocab_size=12, bos_id=1, eos_id=2, pad_id=0, d_dec=8, n_layers=1, n_heads=2, dropout=0.0).eval()
    enc, el = torch.randn(2, 5, 8), torch.tensor([5, 3])
    batch = {"text": torch.tensor([[5, 6, 7], [8, 0, 0]]), "text_len": torch.tensor([3, 1]),
             "prompt": torch.tensor([[9, 10], [11, 0]]), "prompt_len": torch.tensor([2, 1])}
    got = h.loss(enc, el, batch)
    nll, n = 0.0, 0
    for b in range(2):
        p = [1] + batch["prompt"][b, : batch["prompt_len"][b]].tolist()
        t = batch["text"][b, : batch["text_len"][b]].tolist() + [2]
        mem, pad = h._mem(enc[b:b + 1, : el[b]], el[b:b + 1])
        logits = h._forward(torch.tensor([p + t[:-1]]), mem, pad)[0, len(p) - 1:]
        nll, n = nll + F.cross_entropy(logits, torch.tensor(t), reduction="sum"), n + len(t)
    assert torch.allclose(got, nll / n, atol=1e-5)
