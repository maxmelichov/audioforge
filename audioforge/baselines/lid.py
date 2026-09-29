"""Dedicated spoken-language-ID baselines (research/LID.md), CPU, each exposing
``probs(audio) -> {language code: probability}`` over its own label set (the caller restricts to the evaluated set):

* ``AmberNet``       NVIDIA langid_ambernet (NeMo EncDecSpeakerLabelModel, VoxLingua107, 107 languages; NGC
                     terms of use), ported without NeMo: log-mel (per-feature norm) -> ConvASREncoder (SE separable
                     blocks, reused from baselines.sd) -> x-vector stats pooling (mean, unbiased std) -> Linear + BN
                     (no affine) -> ReLU -> Linear(107). ``nemo_import.import_ambernet``.
* ``SpeechBrainLID`` speechbrain/lang-id-voxlingua107-ecapa (ECAPA-TDNN, VoxLingua107, Apache-2.0).
* ``WhisperLID``     OpenAI Whisper (tiny / base, Apache-2.0) language detection through transformers: the decoder's
                     first-step logits after <|startoftranscript|> over the language tokens (the model's own
                     ``detect_language`` rule, restricted to the requested codes).
* ``WhisperCT2LID``  the same rule through faster-whisper (CTranslate2 conversion of the OpenAI weights).
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sd import ConvEncoder, _logmel

log = logging.getLogger(__name__)

SR = 16000
# codes that differ between label sets and ISO 639-1
AMBERNET_ALIASES = {"he": "iw"}  # VoxLingua107 (AmberNet, SpeechBrain) uses the old Hebrew code


class XVectorDecoder(nn.Module):
    """NeMo 1.x SpeakerDecoder, pool_mode xvector: [mean, std] over time -> Linear -> BN(affine=False) -> ReLU ->
    final Linear (the class logits). Keys: emb_layers.0.{0,1}, final."""

    def __init__(self, feat_in: int, emb: int, n_classes: int):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Sequential(nn.Linear(2 * feat_in, emb), nn.BatchNorm1d(emb, affine=False),
                                                       nn.ReLU())])
        self.final = nn.Linear(emb, n_classes)

    def forward(self, x, lens):
        valid = (torch.arange(x.shape[-1], device=x.device)[None] < lens[:, None])[:, None].float()
        n = valid.sum(-1)
        mean = (x * valid).sum(-1) / n
        std = (((x - mean[..., None]) ** 2 * valid).sum(-1) / (n - 1).clamp(min=1)).sqrt()
        return self.final(self.emb_layers[0](torch.cat([mean, std], -1)))


class AmberNet(nn.Module):
    def __init__(self, nc: dict):
        super().__init__()
        self.preprocessor = _logmel(nc["preprocessor"])
        self.encoder = ConvEncoder(nc["encoder"]["feat_in"], nc["encoder"]["jasper"])
        d = nc["decoder"]
        assert d.get("pool_mode") == "xvector" and not d.get("angular", False), d
        self.decoder = XVectorDecoder(d["feat_in"], int(d["emb_sizes"]), d["num_classes"])
        self.labels = list(nc["train_ds"]["labels"])

    def forward(self, audio, lengths):
        mel, ml = self.preprocessor(audio, lengths)
        x, xl = self.encoder(mel, ml)
        return self.decoder(x, xl)

    @torch.no_grad()
    def probs(self, audio: np.ndarray) -> dict:
        x = torch.as_tensor(np.asarray(audio, np.float32))[None]
        p = self(x, torch.tensor([x.shape[1]])).softmax(-1)[0]
        inv = {v: k for k, v in AMBERNET_ALIASES.items()}
        return {inv.get(l, l): float(v) for l, v in zip(self.labels, p)}


def _depthwise_forward(self, x, lens):
    """sd.MaskedConv1d.forward for a depthwise conv (groups == channels, stride 1, dilation 1) as unfold * w: PyTorch's
    CPU path runs a grouped Conv1d as one small conv per channel (16k calls per AmberNet forward, ~200 ms on 2
    threads); this is the same math in a few ops (outputs equal to float rounding, tests/test_lid.py)."""
    from .sd import _pad_mask
    x = x.masked_fill(_pad_mask(lens, x.shape[-1])[:, None], 0.0)
    c = self.conv
    k, p = c.kernel_size[0], c.padding[0]
    y = (F.pad(x, (p, p)).unfold(2, k, 1) * c.weight[:, 0][None, :, None, :]).sum(-1)
    if c.bias is not None:
        y = y + c.bias[None, :, None]
    lens = lens + 2 * p - (k - 1)
    return y, lens


def fast_depthwise(model) -> int:
    """Re-bind every depthwise MaskedConv1d of ``model`` to ``_depthwise_forward``; -> number of convs re-bound."""
    import types

    from .sd import MaskedConv1d
    n = 0
    for mod in model.modules():
        c = getattr(mod, "conv", None)
        if (isinstance(mod, MaskedConv1d) and c.groups == c.in_channels == c.out_channels and c.groups > 1
                and c.stride[0] == 1 and c.dilation[0] == 1):
            mod.forward = types.MethodType(_depthwise_forward, mod)
            n += 1
    return n


def load_ambernet(path, verbose: bool = False, fast: bool = True) -> AmberNet:
    """.nemo -> AmberNet, strict load (loss.weight and the two preprocessor buffers are not part of the model)."""
    from pathlib import Path

    from ..nemo_import import read_nemo
    nc, sd, _ = read_nemo(Path(path))
    model = AmberNet(nc)
    skip = ("loss.weight", "preprocessor.featurizer.window", "preprocessor.featurizer.fb")
    model.load_state_dict({k: v for k, v in sd.items() if k not in skip}, strict=True)
    model.import_info = {"source": str(path), "target": nc.get("target"), "nemo_version": nc.get("nemo_version"),
                         "params": sum(p.numel() for p in model.parameters()), "n_classes": len(model.labels),
                         "fast_depthwise": fast_depthwise(model) if fast else 0}
    if verbose:
        log.info("%s %s", "loaded", model.import_info)
    return model.eval()


class SpeechBrainLID:
    """speechbrain/lang-id-voxlingua107-ecapa via speechbrain's EncoderClassifier (labels like 'he: Hebrew')."""

    def __init__(self, source: str = "speechbrain/lang-id-voxlingua107-ecapa", savedir: str | None = None):
        from speechbrain.inference.classifiers import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(source=source, savedir=savedir, run_opts={"device": "cpu"})
        enc = self.model.hparams.label_encoder
        inv = {v: k for k, v in AMBERNET_ALIASES.items()}  # VoxLingua107 codes (iw = Hebrew) -> ISO 639-1
        self.labels = [inv.get(c, c) for c in (enc.ind2lab[i].split(":")[0].strip() for i in range(len(enc.ind2lab)))]

    @torch.no_grad()
    def probs(self, audio: np.ndarray) -> dict:
        x = torch.as_tensor(np.asarray(audio, np.float32))[None]
        out = self.model.classify_batch(x)[0][0]  # log posteriors
        p = out.exp() if float(out.max()) <= 0 else out.softmax(-1)
        return {l: float(v) for l, v in zip(self.labels, p)}


class WhisperLID:
    def __init__(self, name: str = "openai/whisper-tiny"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.proc = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name).eval()
        tok = self.proc.tokenizer
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        self.codes = [c for c in LANGUAGES]
        self.ids = [tok.convert_tokens_to_ids(f"<|{c}|>") for c in self.codes]

    @torch.no_grad()
    def probs(self, audio: np.ndarray) -> dict:
        f = self.proc.feature_extractor(np.asarray(audio, np.float32), sampling_rate=SR, return_tensors="pt")
        logits = self.model(input_features=f.input_features, decoder_input_ids=torch.tensor([[self.sot]])).logits[0, -1]
        p = logits[self.ids].softmax(-1)
        return {c: float(v) for c, v in zip(self.codes, p)}


class WhisperCT2LID:
    """Whisper through faster-whisper / CTranslate2 (e.g. Systran/faster-whisper-base, the converted OpenAI weights):
    ``WhisperModel.detect_language`` on the clip (first 30 s window), all language probabilities."""

    def __init__(self, name: str = "base", threads: int = 2):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(name, device="cpu", compute_type="float32", cpu_threads=threads)

    def probs(self, audio: np.ndarray) -> dict:
        _, _, allp = self.model.detect_language(np.asarray(audio, np.float32), language_detection_segments=1)
        return {c: float(v) for c, v in allp}


def restrict(probs: dict, codes: list[str]) -> np.ndarray:
    """Posterior over ``codes`` (renormalised); codes missing from the model's label set get 0."""
    v = np.array([probs.get(c, 0.0) for c in codes], np.float64)
    s = v.sum()
    return (v / s if s > 0 else np.full(len(codes), 1.0 / len(codes))).astype(np.float32)
