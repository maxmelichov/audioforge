"""Speech-augmented LM (SALM) bridge, the Canary-Qwen / VoiceChat / Audio-Flamingo pattern.

    FastConformer encoder -> frame stacking + MLP projector -> LLM input embeddings

The prompt contains an ``<|audio|>`` placeholder which is replaced by the
projected audio frames. Loss is computed on answer tokens only. Any causal LM
exposing ``get_input_embeddings()`` and ``forward(inputs_embeds=..., attention_mask=...)``
with ``.logits`` works (Hugging Face models do). ``TinyCausalLM`` is included
for tests and small experiments.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import LogMel
from .modules.fastconformer import FastConformerEncoder


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4, n_heads: int = 4, max_len: int = 1024):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, 0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.config = SimpleNamespace(hidden_size=d_model)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds, attention_mask=None):
        T = inputs_embeds.shape[1]
        h = inputs_embeds + self.pos(torch.arange(T, device=inputs_embeds.device))[None]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
        pad = None if attention_mask is None else ~attention_mask.bool()
        h = self.tf(h, mask=causal, src_key_padding_mask=pad, is_causal=True)
        return SimpleNamespace(logits=self.head(self.norm(h)))


class Projector(nn.Module):
    def __init__(self, d_in: int, d_out: int, stack: int = 2):
        super().__init__()
        self.stack = stack
        self.net = nn.Sequential(nn.Linear(d_in * stack, d_out), nn.GELU(), nn.Linear(d_out, d_out))

    def forward(self, x, lengths):
        B, T, D = x.shape
        pad = (-T) % self.stack
        x = F.pad(x, (0, 0, 0, pad)).reshape(B, (T + pad) // self.stack, D * self.stack)
        return self.net(x), torch.div(lengths + self.stack - 1, self.stack, rounding_mode="floor")


class SpeechLLM(nn.Module):
    def __init__(self, encoder_cfg: dict, llm: nn.Module, tokenizer, audio_token: str = "<|audio|>",
                 stack: int = 2, freeze_llm: bool = True, preprocessor: dict | None = None):
        super().__init__()
        self.preprocessor = LogMel(**(preprocessor or {}))
        enc = dict(encoder_cfg)
        enc.setdefault("feat_in", self.preprocessor.n_mels)
        self.encoder = FastConformerEncoder(**enc)
        self.llm = llm
        self.tok = tokenizer
        self.audio_id = tokenizer.token_id(audio_token)
        self.projector = Projector(self.encoder.d_model, llm.config.hidden_size, stack)
        if freeze_llm:  # Canary-Qwen trains encoder + projector + LoRA; here: encoder + projector
            for p in self.llm.parameters():
                p.requires_grad_(False)

    def _embed(self, audio, audio_len, prompts: list[list[int]], answers: list[list[int]] | None):
        feats, flen = self.preprocessor(audio, audio_len)
        enc, elen = self.encoder(feats, flen)
        a, alen = self.projector(enc, elen)
        E = self.llm.get_input_embeddings()
        a = a.to(E.weight.dtype)  # fp32 encoder/projector feeding a bf16/fp16 LLM (Canary-Qwen setup)
        seqs, labels = [], []
        for b, p in enumerate(prompts):
            i = p.index(self.audio_id)
            ids_pre = torch.tensor(p[:i], dtype=torch.long, device=a.device)
            ids_post = torch.tensor(p[i + 1:] + (answers[b] if answers else []), dtype=torch.long, device=a.device)
            seqs.append(torch.cat([E(ids_pre), a[b, : alen[b]], E(ids_post)]))
            if answers:
                n_ctx = i + int(alen[b]) + len(p) - i - 1
                labels.append([-100] * n_ctx + answers[b])
        L = max(s.shape[0] for s in seqs)
        x = torch.zeros(len(seqs), L, seqs[0].shape[-1], device=a.device, dtype=seqs[0].dtype)
        mask = torch.zeros(len(seqs), L, dtype=torch.bool, device=a.device)
        lab = torch.full((len(seqs), L), -100, device=a.device)
        for b, s in enumerate(seqs):
            x[b, : len(s)], mask[b, : len(s)] = s, True
            if answers:
                lab[b, : len(labels[b])] = torch.tensor(labels[b], device=a.device)
        return x, mask, lab

    def forward(self, audio, audio_len, prompts, answers):
        x, mask, lab = self._embed(audio, audio_len, prompts, answers)
        logits = self.llm(inputs_embeds=x, attention_mask=mask).logits
        # predict token t+1 from position t
        return F.cross_entropy(logits[:, :-1].float().transpose(1, 2), lab[:, 1:], ignore_index=-100)

    @torch.no_grad()
    def generate(self, audio, audio_len, prompt: list[int], max_new: int = 100) -> list[int]:
        eos = self.tok.token_id("<eos>")
        out: list[int] = []
        for _ in range(max_new):
            x, mask, _ = self._embed(audio[:1], audio_len[:1], [prompt + out], None)
            nxt = int(self.llm(inputs_embeds=x, attention_mask=mask).logits[0, -1].argmax())
            if nxt == eos:
                break
            out.append(nxt)
        return out
