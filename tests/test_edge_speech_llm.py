import torch

from audioforge.speech_llm import SpeechLLM, TinyCausalLM
from audioforge.tokenizer import CharTokenizer


def test_fp32_encoder_with_bf16_llm_gives_finite_loss_and_projector_grads():
    torch.manual_seed(0)
    tok = CharTokenizer.train(["abc"], specials=["<|audio|>"])
    llm = TinyCausalLM(tok.vocab_size, 32, 1, 2).to(torch.bfloat16)  # HF LLMs are usually loaded in bf16
    m = SpeechLLM(dict(d_model=32, n_layers=1, n_heads=2, subsampling_channels=8), llm, tok)
    x = torch.randn(1, 8000) * 0.1
    loss = m(x, torch.tensor([8000]), [tok.encode("a<|audio|>b")], [tok.encode("abc")])
    assert torch.isfinite(loss)
    loss.backward()
    g = m.projector.net[0].weight.grad
    assert g is not None and g.dtype == torch.float32 and torch.isfinite(g).all()
