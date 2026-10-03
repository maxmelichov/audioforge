"""Dedicated VAD / diarization / speaker-verification baselines, scored like our heads.

What is here:
  * frame conversions to the AMI label grid (80 ms frames, "any overlap": ``int(s/.08) .. ceil(e/.08)-1``,
    datasets/ami.py) - fine-grained outputs are max-pooled onto it, segments are rasterised with the same rule;
  * VAD scoring (acc / recall / precision / F1 / FPR) and a few wrappers (Silero, WebRTC);
  * a NeMo ``ConvASREncoder`` (Jasper/QuartzNet blocks: masked separable 1-D convs, optional squeeze-excite)
    with two heads, loaded strictly from .nemo archives without NeMo:
      - ``TitaNet``  (nvidia/speakerverification_en_titanet_large, EncDecSpeakerLabelModel,
                      attentive statistics pooling -> 192-d embedding);
      - ``FrameVAD`` (nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0, EncDecFrameClassificationModel,
                      20 ms speech probabilities).
    The module / parameter names mirror NeMo's, so the "mapping" is the identity minus the two
    preprocessor buffers (recomputed by our LogMel and compared, see ``load_nemo_conv``).
  * diarization helpers: segments -> (T, 4) arrival-order matrix (top-4 by activity) and frame matrices ->
    pyannote ``Annotation`` for pyannote.metrics.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import LogMel, mel_filterbank

log = logging.getLogger(__name__)

FRAME_SEC = 0.08
SR = 16000


# --------------------------------------------------------------------------- frame grid
def intervals_to_frames(intervals, T: int, frame: float = FRAME_SEC) -> np.ndarray:
    """(start, end) seconds -> (T,) bool on the label grid, the AMI rule int(s/f) .. ceil(e/f)-1."""
    out = np.zeros(T, bool)
    for s, e in intervals:
        if e <= s:
            continue
        a, b = max(0, int(s / frame)), min(T, int(math.ceil(e / frame - 1e-9)))
        if b > a:
            out[a:b] = True
    return out


def pool_probs(p: np.ndarray, hop: float, T: int, offset: float = 0.0, frame: float = FRAME_SEC) -> np.ndarray:
    """Max-pool fine scores onto the label grid. Fine frame j spans [offset + j*hop, offset + (j+1)*hop);
    label frame i gets the max over every fine frame overlapping [i*frame, (i+1)*frame) (0 if none)."""
    p = np.asarray(p, np.float64)
    out = np.zeros(T)
    for i in range(T):
        lo, hi = i * frame, (i + 1) * frame
        j0 = max(0, int(math.floor((lo - offset) / hop + 1e-9)))
        j1 = min(len(p), int(math.ceil((hi - offset) / hop - 1e-9)))
        if j1 > j0:
            out[i] = p[j0:j1].max()
    return out


def runs(active: np.ndarray, hop: float, offset: float = 0.0) -> list[tuple[float, float]]:
    """Binary fine frames -> merged (start, end) intervals in seconds."""
    a = np.concatenate([[0], np.asarray(active, np.int8), [0]])
    d = np.diff(a)
    return [(offset + s * hop, offset + e * hop) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def segments_to_matrix(segments, T: int, max_spks: int = 4, frame: float = FRAME_SEC) -> np.ndarray:
    """[(start, end, label)] -> (T, max_spks) float: the max_spks most active labels (frames), columns in
    arrival order (first active frame), each rasterised with the label rule."""
    cols = {}
    for s, e, lab in segments:
        cols.setdefault(lab, []).append((s, e))
    mats = {lab: intervals_to_frames(iv, T, frame) for lab, iv in cols.items()}
    mats = {k: v for k, v in mats.items() if v.any()}
    keep = sorted(mats, key=lambda k: -int(mats[k].sum()))[:max_spks]
    keep.sort(key=lambda k: int(np.argmax(mats[k])))
    out = np.zeros((T, max_spks), np.float32)
    for c, k in enumerate(keep):
        out[:, c] = mats[k]
    return out


def matrix_to_annotation(M: np.ndarray, frame: float = FRAME_SEC, prefix: str = "s"):
    """(T, S) binary frame matrix -> pyannote.core.Annotation (frame i = [i*frame, (i+1)*frame))."""
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for c in range(M.shape[1]):
        for s, e in runs(M[:, c] > 0.5, frame):
            ann[Segment(s, e), f"{prefix}{c}"] = f"{prefix}{c}"
    return ann


# --------------------------------------------------------------------------- VAD scoring
def vad_counts(pred: np.ndarray, lab: np.ndarray) -> dict:
    pred, lab = np.asarray(pred, bool), np.asarray(lab, bool)
    return dict(tp=int((pred & lab).sum()), fp=int((pred & ~lab).sum()), fn=int((~pred & lab).sum()),
                tn=int((~pred & ~lab).sum()))


def vad_metrics(c: dict) -> dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    rec, prec = tp / max(1, tp + fn), tp / max(1, tp + fp)
    return dict(acc=round((tp + tn) / max(1, tp + fp + fn + tn), 4), recall=round(rec, 4),
                precision=round(prec, 4), f1=round(2 * prec * rec / max(1e-12, prec + rec), 4),
                fpr=round(fp / max(1, fp + tn), 4), miss=round(fn / max(1, tp + fn), 4))


def score_vad(scores: list[np.ndarray], labels: list[np.ndarray], thresholds) -> dict:
    """Pooled VAD metrics over windows at each threshold (scores already on the label grid)."""
    out = {}
    for th in thresholds:
        c = dict(tp=0, fp=0, fn=0, tn=0)
        for s, l in zip(scores, labels):
            for k, v in vad_counts(np.asarray(s) > th, l).items():
                c[k] += v
        out[str(th)] = vad_metrics(c)
    return out


# --------------------------------------------------------------------------- external VADs
def silero_probs(model, audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Silero VAD (jit), 512-sample (32 ms) chunks with its recurrent state -> (probs, hop)."""
    with torch.no_grad():
        model.reset_states()
        x = torch.from_numpy(np.asarray(audio, np.float32))[None]
        n = 512
        if x.shape[1] % n:
            x = F.pad(x, (0, n - x.shape[1] % n))
        out = [model(x[:, i:i + n], SR) for i in range(0, x.shape[1], n)]
    return torch.cat(out, 1)[0].numpy(), n / SR


def webrtc_decisions(audio: np.ndarray, mode: int, frame_ms: int = 30) -> tuple[np.ndarray, float]:
    """WebRTC VAD (GMM), one decision per frame_ms frame -> (0/1 array, hop)."""
    import webrtcvad
    vad = webrtcvad.Vad(mode)
    pcm = (np.clip(np.asarray(audio), -1, 1) * 32767).astype("<i2")
    n = SR * frame_ms // 1000
    k = len(pcm) // n
    return np.array([vad.is_speech(pcm[i * n:(i + 1) * n].tobytes(), SR) for i in range(k)], np.float32), n / SR


# --------------------------------------------------------------------------- NeMo ConvASREncoder
def _same_pad(k: int, stride: int, dilation: int) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("only stride 1 with dilation > 1")
    return dilation * (k - 1) // 2


def _pad_mask(lens: torch.Tensor, T: int) -> torch.Tensor:
    """(B, T) True on padding."""
    return torch.arange(T, device=lens.device)[None] >= lens[:, None]


class MaskedConv1d(nn.Module):
    """NeMo MaskedConv1d: zero the padding of the input, convolve, update the lengths."""

    def __init__(self, cin, cout, k, stride=1, dilation=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, stride=stride, dilation=dilation, padding=padding, groups=groups,
                              bias=bias)

    def forward(self, x, lens):
        x = x.masked_fill(_pad_mask(lens, x.shape[-1])[:, None], 0.0)
        c = self.conv
        lens = torch.div(lens + 2 * c.padding[0] - c.dilation[0] * (c.kernel_size[0] - 1) - 1, c.stride[0],
                         rounding_mode="floor") + 1
        return c(x), lens


class SqueezeExcite(nn.Module):
    """NeMo SqueezeExcite, global context (se_context_size -1): masked time-mean -> fc -> sigmoid gate."""

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch // reduction, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(ch // reduction, ch, bias=False))

    def forward(self, x, lens):
        valid = ~_pad_mask(lens, x.shape[-1])[:, None]
        y = (x * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        return x * torch.sigmoid(self.fc(y))[..., None], lens


class JasperBlock(nn.Module):
    """NeMo JasperBlock (residual_mode add, dense residual off): ``mconv`` = repeat x [dw conv, pw conv, BN,
    (act, dropout) except after the last], + SE; ``res`` = 1x1 conv + BN on the block input; ``mout`` = act + dropout.
    Indices of ``mconv`` match NeMo's so state_dict keys are identical."""

    def __init__(self, cin, cout, repeat, kernel, stride, dilation, residual, separable, se, bn_eps=1e-3):
        super().__init__()
        pad = _same_pad(kernel, stride, dilation)
        layers, c = [], cin
        for r in range(repeat):
            if separable:
                layers += [MaskedConv1d(c, c, kernel, stride, dilation, pad, groups=c),
                           MaskedConv1d(c, cout, 1)]
            else:
                layers += [MaskedConv1d(c, cout, kernel, stride, dilation, pad)]
            layers.append(nn.BatchNorm1d(cout, eps=bn_eps, momentum=0.1))
            if r < repeat - 1:
                layers += [nn.ReLU(), nn.Dropout(0.0)]
            c = cout
        if se:
            layers.append(SqueezeExcite(cout))
        self.mconv = nn.ModuleList(layers)
        self.res = (nn.ModuleList([nn.ModuleList([MaskedConv1d(cin, cout, 1), nn.BatchNorm1d(cout, eps=bn_eps)])])
                    if residual else None)
        self.mout = nn.Sequential(nn.ReLU(), nn.Dropout(0.0))

    def forward(self, x, lens):
        out, l0 = x, lens
        for layer in self.mconv:
            if isinstance(layer, (MaskedConv1d, SqueezeExcite)):
                out, lens = layer(out, lens)
            else:
                out = layer(out)
        if self.res is not None:
            r = x
            for layer in self.res[0]:
                r = layer(r, l0)[0] if isinstance(layer, MaskedConv1d) else layer(r)
            out = out + r
        return self.mout(out), lens


class ConvEncoder(nn.Module):
    """NeMo ConvASREncoder from its ``jasper`` block list (``encoder.encoder.N.*`` keys)."""

    def __init__(self, feat_in: int, jasper: list[dict]):
        super().__init__()
        blocks, c = [], feat_in
        for b in jasper:
            assert b.get("kernel_size_factor", 1.0) == 1.0 and not b.get("dense_residual", False), b
            assert b.get("se_context_size", -1) == -1, "only global squeeze-excite"
            blocks.append(JasperBlock(c, b["filters"], b["repeat"], b["kernel"][0], b["stride"][0], b["dilation"][0],
                                      b["residual"], b.get("separable", False), b.get("se", False)))
            c = b["filters"]
        self.encoder = nn.ModuleList(blocks)
        self.out_channels = c

    def forward(self, x, lens):
        for blk in self.encoder:
            x, lens = blk(x, lens)
        return x, lens


class _TDNN(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv_layer = nn.Conv1d(cin, cout, 1)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return self.bn(self.activation(self.conv_layer(x)))


def _masked_stats(x, w, eps=1e-10):
    mean = (w * x).sum(2)
    std = torch.sqrt((w * (x - mean[..., None]).pow(2)).sum(2).clamp(eps))
    return mean, std


class AttentivePool(nn.Module):
    """NeMo AttentivePoolLayer (ECAPA channel- and context-dependent attentive statistics pooling)."""

    def __init__(self, cin: int, att: int = 128):
        super().__init__()
        self.attention_layer = nn.Sequential(_TDNN(cin * 3, att), nn.Tanh(), nn.Conv1d(att, cin, 1))

    def forward(self, x, lens):
        T = x.shape[2]
        mask = (~_pad_mask(lens, T))[:, None].float()          # (B, 1, T)
        mean, std = _masked_stats(x, mask / mask.sum(2, keepdim=True))
        a = self.attention_layer(torch.cat([x, mean[..., None].expand(-1, -1, T), std[..., None].expand(-1, -1, T)], 1))
        alpha = torch.softmax(a.masked_fill(mask == 0, float("-inf")), dim=2)
        mu, sg = _masked_stats(x, alpha)
        return torch.cat([mu, sg], 1)[..., None]               # (B, 2C, 1)


class SpeakerDecoder(nn.Module):
    """NeMo SpeakerDecoder, pool_mode attention, one embedding layer (BN + 1x1 conv); ``final`` = AAM classes."""

    def __init__(self, feat_in: int, emb: int, n_classes: int):
        super().__init__()
        self._pooling = AttentivePool(feat_in)
        self.emb_layers = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(2 * feat_in), nn.Conv1d(2 * feat_in, emb, 1))])
        self.final = nn.Linear(emb, n_classes, bias=False)

    def forward(self, x, lens):
        return self.emb_layers[0](self._pooling(x, lens))[..., 0]


def _logmel(pc: dict) -> LogMel:
    norm = pc.get("normalize", "per_feature")
    assert pc.get("sample_rate", SR) == SR and pc.get("features", 80) == 80 and pc.get("n_fft", 512) == 512
    assert pc.get("window_size", 0.025) == 0.025 and pc.get("window_stride", 0.01) == 0.01
    assert pc.get("window", "hann") == "hann" and pc.get("frame_splicing", 1) == 1
    return LogMel(normalize=None if norm in (None, "None", "none") else norm, dither=pc.get("dither", 1e-5))


class TitaNet(nn.Module):
    """NVIDIA TitaNet (EncDecSpeakerLabelModel): log-mel -> ConvASREncoder (SE separable blocks) ->
    attentive statistics pooling -> 192-d embedding. ``embed`` returns L2-normalised embeddings."""

    def __init__(self, nc: dict):
        super().__init__()
        self.preprocessor = _logmel(nc["preprocessor"])
        self.encoder = ConvEncoder(nc["encoder"]["feat_in"], nc["encoder"]["jasper"])
        d = nc["decoder"]
        assert d.get("pool_mode", "attention") == "attention"
        emb = d["emb_sizes"] if isinstance(d["emb_sizes"], int) else d["emb_sizes"][0]
        self.decoder = SpeakerDecoder(d["feat_in"], int(emb), d["num_classes"])

    def forward(self, audio: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mel, ml = self.preprocessor(audio, lengths)
        x, xl = self.encoder(mel, ml)
        return self.decoder(x, xl)

    @torch.no_grad()
    def embed(self, audios, batch_size: int = 1) -> torch.Tensor:
        """list of 16 kHz float arrays -> (N, 192) unit-norm embeddings (masked batching is exact)."""
        out = []
        for i in range(0, len(audios), batch_size):
            chunk = [torch.as_tensor(np.asarray(a, np.float32)) for a in audios[i:i + batch_size]]
            lens = torch.tensor([len(a) for a in chunk])
            x = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True)
            out.append(F.normalize(self(x, lens), dim=-1))
        return torch.cat(out)


class FrameVAD(nn.Module):
    """NVIDIA Frame_VAD_Multilingual_MarbleNet (EncDecFrameClassificationModel): log-mel (no normalisation) ->
    ConvASREncoder (stride 2) -> Linear(128, 2) per 20 ms frame. ``probs`` -> speech probability per 20 ms."""

    hop = 0.02

    def __init__(self, nc: dict):
        super().__init__()
        self.preprocessor = _logmel(nc["preprocessor"])
        self.encoder = ConvEncoder(nc["encoder"]["feat_in"], nc["encoder"]["jasper"])
        d = nc["decoder"]
        assert d.get("num_layers", 1) == 1
        self.decoder = nn.Module()
        self.decoder.layer0 = nn.Linear(d["hidden_size"], d["num_classes"])

    def forward(self, audio, lengths):
        mel, ml = self.preprocessor(audio, lengths)
        x, xl = self.encoder(mel, ml)
        return self.decoder.layer0(x.transpose(1, 2)), xl

    @torch.no_grad()
    def probs(self, audio: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(audio, np.float32))[None]
        logits, xl = self(x, torch.tensor([x.shape[1]]))
        return torch.softmax(logits, -1)[0, : int(xl[0]), 1].numpy()


PRE_SKIP = ("preprocessor.featurizer.window", "preprocessor.featurizer.fb")
CLASSES = {"EncDecSpeakerLabelModel": TitaNet, "EncDecFrameClassificationModel": FrameVAD}


def load_nemo_conv(path_or_hf_id, verbose: bool = False):
    """.nemo (path or HF id) of a TitaNet / frame-VAD MarbleNet -> our module, ``strict=True``.
    The two preprocessor buffers are not loaded: they are compared with our LogMel's (window exact,
    mel filterbank max |diff| recorded in ``import_info``)."""
    from ..nemo_import import read_nemo, resolve
    t0 = time.time()
    path = resolve(path_or_hf_id)
    nc, sd, _ = read_nemo(path)
    cls = CLASSES[str(nc.get("target", "")).rsplit(".", 1)[-1]]
    model = cls(nc)
    ours = {k: v for k, v in sd.items() if k not in PRE_SKIP}
    model.load_state_dict(ours, strict=True)
    info = dict(source=str(path_or_hf_id), target=nc.get("target"), nemo_version=nc.get("nemo_version"),
                n_tensors=len(sd), n_loaded=len(ours), params=sum(p.numel() for p in model.parameters()),
                load_sec=round(time.time() - t0, 2))
    if "preprocessor.featurizer.fb" in sd:
        info["fb_max_abs_diff"] = float((sd["preprocessor.featurizer.fb"][0].float() - mel_filterbank(SR, 512, 80)).abs().max())
    if "preprocessor.featurizer.window" in sd:
        info["window_max_abs_diff"] = float((sd["preprocessor.featurizer.window"].float()
                                             - torch.hann_window(400, periodic=False)).abs().max())
    model.import_info = info
    if verbose:
        log.info(f"loaded {Path(path).name}: {cls.__name__} {info}")
    return model.eval()
