"""Load NVIDIA's pretrained NeMo FastConformer checkpoints (.nemo) into audioforge, without NeMo.

    from audioforge.nemo_import import import_nemo
    model = import_nemo("nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")  # or a .nemo path
    model.transcribe([audio])                       # RNNT (primary) ; head="ctc" for the aux CTC head
    save_model(model, "runs/nemo_hybrid_streaming_multi.afm")   # fine-tune it like any .afm

    .venv/bin/python -m audioforge.nemo_import nvidia/stt_en_fastconformer_hybrid_large_streaming_multi \
        runs/nemo_hybrid_streaming_multi.afm

A .nemo file is a tar of model_config.yaml + model_weights.ckpt (a torch state_dict) + the
SentencePiece tokenizer. Supported: ``ConformerEncoder`` with rel_pos attention and dw_striding 8x
subsampling (causal or not, chunked_limited or full context, layer/batch-norm conv), plus an RNNT
decoder+joint, a CTC ``ConvASRDecoder``, or both (hybrid RNNT-CTC), and TDT joints with duration outputs
(parakeet-tdt-*, ``tdt_durations``; e.g. nvidia/parakeet-tdt-0.6b-v3, research/archive/HYBRID_ASR.md). Every checkpoint
tensor is mapped; unmapped or missing weights raise. See research/archive/NEMO_IMPORT.md.

nvidia/nemotron-speech-streaming-en-0.6b (research/archive/ENC_0P6B.md) goes through the same path: 24 x d1024 / 8 heads,
128 mels, ``use_bias: false`` (the 264 missing encoder biases are filled with zeros, which is exact), a 2-layer
LSTM prediction net, ``xscaling: false``, and an ``aux_ctc`` config stub without CTC weights (no CTC head is built).
Checkpoints above 1 GB are extracted once to data/nemo/.extracted/ and memory-mapped, so the import peaks at one
copy of the weights.

Also Sortformer diarizers (``SortformerEncLabelModel``, e.g. nvidia/diar_streaming_sortformer_4spk-v2):
the NEST encoder + a single ``diar`` SortformerHead (post-LN, no positions, ReLU-first output MLP).
See research/archive/SORTFORMER_IMPORT.md.
"""
from __future__ import annotations

import logging
import re
import sys
import tarfile
import time
from pathlib import Path

import torch
import yaml

from .model import SpeechModel
from .paths import DATA_ROOT
from .tokenizer import SentencePieceTokenizer

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
NEMO_DIR = DATA_ROOT / "nemo"


# --------------------------------------------------------------------------- reading .nemo
def resolve(path_or_hf_id: str | Path) -> Path:
    """Local .nemo path, or an HF repo id whose ``<name>.nemo`` is downloaded into data/nemo/."""
    p = Path(path_or_hf_id)
    if p.exists():
        return p
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = str(path_or_hf_id)
    files = [f for f in list_repo_files(repo) if f.endswith(".nemo")]
    if not files:
        raise FileNotFoundError(f"{repo}: no .nemo file in the repo")
    return Path(hf_hub_download(repo, files[0], local_dir=str(NEMO_DIR)))


MMAP_MIN_BYTES = 1 << 30  # weights larger than this are extracted once and memory-mapped (0.6B: 2.4 GB fp32)


def _extracted_ckpt(path: Path, tar: tarfile.TarFile, member: tarfile.TarInfo) -> Path:
    """Extract ``model_weights.ckpt`` next to the .nemo (data/nemo/.extracted/<stem>.ckpt) once; return its path."""
    out = path.parent / ".extracted" / f"{path.stem}.ckpt"
    if out.exists() and out.stat().st_size == member.size:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with tar.extractfile(member) as f, open(tmp, "wb") as g:
        while chunk := f.read(1 << 24):
            g.write(chunk)
    tmp.replace(out)
    return out


def read_nemo(path: Path, mmap: bool | None = None) -> tuple[dict, dict, dict[str, bytes]]:
    """-> (config, state_dict, {member name: bytes} of the small non-weight files).

    ``mmap`` (default: only for weights > MMAP_MIN_BYTES) extracts the checkpoint once and loads it memory-mapped,
    so importing a 0.6B model peaks at one copy of the weights (the model) instead of two."""
    cfg, sd, files = None, None, {}
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = Path(m.name).name
            if name == "model_config.yaml":
                cfg = yaml.safe_load(tar.extractfile(m).read())
            elif name == "model_weights.ckpt":
                if mmap or (mmap is None and m.size > MMAP_MIN_BYTES):
                    sd = torch.load(_extracted_ckpt(path, tar, m), map_location="cpu", weights_only=False, mmap=True)
                else:
                    sd = torch.load(tar.extractfile(m), map_location="cpu", weights_only=False)
            else:
                files[name] = tar.extractfile(m).read()
    if cfg is None or sd is None:
        raise ValueError(f"{path}: not a .nemo archive (model_config.yaml / model_weights.ckpt missing)")
    return cfg, sd, files


def _nemo_file(files: dict[str, bytes], ref: str) -> bytes:
    """Resolve a config reference like 'nemo:<hash>_tokenizer.model'."""
    name = ref.split("nemo:", 1)[-1]
    name = Path(name).name
    if name in files:
        return files[name]
    hits = [k for k in files if k.endswith(name)]
    if len(hits) != 1:
        raise KeyError(f"tokenizer file {ref!r} not found in the archive ({sorted(files)})")
    return files[hits[0]]


# --------------------------------------------------------------------------- config translation
def _translate_preprocessor(nc: dict) -> tuple[int, dict]:
    """NeMo AudioToMelSpectrogramPreprocessor config -> (sample_rate, audioforge LogMel cfg)."""
    sr = int(nc.get("sample_rate", 16000))
    pp = nc["preprocessor"]
    unsupported = {k: pp[k] for k in ("frame_splicing", "pad_to") if pp.get(k) not in (None, 0, 1)}
    if pp.get("window", "hann") != "hann" or unsupported or pp.get("log", True) is not True:
        raise NotImplementedError(f"preprocessor options not supported: {pp}")
    norm = pp.get("normalize", "per_feature")
    if norm not in ("per_feature", "NA", None, "none"):
        raise NotImplementedError(f"preprocessor normalize={norm!r}")
    pre = dict(sample_rate=sr, n_fft=int(pp.get("n_fft", 512)), win_length=int(round(pp["window_size"] * sr)),
               hop_length=int(round(pp["window_stride"] * sr)), n_mels=int(pp["features"]),
               preemph=float(pp.get("preemph", 0.97)), dither=float(pp.get("dither", 1e-5)),
               normalize="per_feature" if norm == "per_feature" else "NA")
    return sr, pre


def _translate_frontend(nc: dict) -> tuple[int, dict, dict]:
    """NeMo preprocessor + ConformerEncoder config -> (sample_rate, preprocessor cfg, encoder cfg)."""
    sr, pre = _translate_preprocessor(nc)
    e = nc["encoder"]
    if not e["_target_"].endswith("ConformerEncoder"):
        raise NotImplementedError(e["_target_"])
    checks = {"self_attention_model": "rel_pos", "subsampling": "dw_striding", "subsampling_factor": 8,
              "reduction": None, "untie_biases": True}
    for k, want in checks.items():
        if e.get(k, want) != want:
            raise NotImplementedError(f"encoder.{k}={e.get(k)!r} (only {want!r} is supported)")
    if e.get("feat_out", -1) not in (-1, None):
        raise NotImplementedError("encoder.feat_out")
    causal = bool(e.get("causal_downsampling", False))
    ccs = e.get("conv_context_size")
    conv_causal = ccs == "causal" or (isinstance(ccs, (list, tuple)) and list(ccs)[1] == 0)
    if conv_causal != causal:
        raise NotImplementedError(f"causal_downsampling={causal} with conv_context_size={ccs!r}")
    att = e.get("att_context_size", [-1, -1])
    atts = [list(a) for a in att] if isinstance(att[0], (list, tuple)) else [list(att)]
    style = e.get("att_context_style", "regular")
    if style == "regular" and any(a != [-1, -1] for a in atts):
        raise NotImplementedError("att_context_style=regular with limited context (only chunked_limited)")
    conv_norm = {"layer_norm": "layer", "batch_norm": "batch"}.get(e.get("conv_norm_type", "batch_norm"))
    if conv_norm is None:
        raise NotImplementedError(f"conv_norm_type={e.get('conv_norm_type')!r}")
    enc = dict(feat_in=int(e["feat_in"]), d_model=int(e["d_model"]), n_layers=int(e["n_layers"]),
               n_heads=int(e["n_heads"]), conv_kernel=int(e.get("conv_kernel_size", 9)),
               ff_expansion=int(e.get("ff_expansion_factor", 4)),
               subsampling_channels=int(e.get("subsampling_conv_channels", 256)),
               dropout=float(e.get("dropout", 0.1)), causal=causal, att_context_sizes=atts,
               conv_norm=conv_norm, pos_emb="rel_pos", xscaling=bool(e.get("xscaling", True)),
               subsampling_activation="relu", subsampling_padding="nemo")
    return sr, pre, enc


def translate_config(nc: dict, nemo_keys=None) -> dict:
    """NeMo model_config.yaml -> audioforge SpeechModel config.

    ``nemo_keys`` (the checkpoint's tensor names, optional): an ``aux_ctc`` section only becomes a CTC head when the
    checkpoint has ``ctc_decoder.*`` weights (nemotron-speech-streaming-en-0.6b is a plain EncDecRNNTBPEModel that
    keeps an empty aux_ctc stub in its config)."""
    if is_sortformer(nc):
        return translate_sortformer_config(nc)
    sr, pre, enc = _translate_frontend(nc)
    heads = {}
    dec = nc.get("decoder", {})
    has_ctc = "aux_ctc" in nc and (nemo_keys is None or any(k.startswith("ctc_decoder.") for k in nemo_keys))
    ctc_w = float(nc.get("aux_ctc", {}).get("ctc_loss_weight", 0.3)) if has_ctc else 0.0
    if dec.get("_target_", "").endswith("RNNTDecoder"):
        pn, jn = dec["prednet"], nc["joint"]["jointnet"]
        if jn.get("activation", "relu") != "relu":
            raise NotImplementedError(f"prednet/joint options: {pn} {jn}")
        if dec.get("blank_as_pad", True) is not True:
            raise NotImplementedError("decoder.blank_as_pad=false")
        # pred_rnn_layers > 1: NeMo stacks them in one nn.LSTM(num_layers=n), as our PredictionNet does
        heads["rnnt"] = dict(type="rnnt", pred_hidden=int(pn["pred_hidden"]), pred_layers=int(pn.get("pred_rnn_layers", 1)),
                             joint_hidden=int(jn["joint_hidden"]),
                             max_symbols=int(nc.get("decoding", {}).get("greedy", {}).get("max_symbols", 10)),
                             weight=1.0 - ctc_w)
        if str(nc.get("target", "")).endswith("WithPrompt"):  # nemotron-3.5-asr-streaming-0.6b: language prompt
            md = nc.get("model_defaults") or {}
            pd = {str(k): int(v) for k, v in (md.get("prompt_dictionary") or {}).items()}
            heads["rnnt"]["prompt"] = dict(num_prompts=int(nc.get("num_prompts", md.get("num_prompts", 128))),
                                           hidden=2 * int(md.get("enc_hidden", nc["encoder"]["d_model"])),
                                           default="en-US" if "en-US" in pd else None, dictionary=pd)
        durations = tdt_durations(nc)
        if durations is not None:  # TDT (parakeet-tdt-*): the joint's last len(durations) outputs are durations
            heads["rnnt"].update(type="tdt", durations=durations,
                                 sigma=float(((nc.get("loss") or {}).get("tdt_kwargs") or {}).get("sigma", 0.0)))
        if has_ctc:
            heads["ctc"] = dict(type="ctc", weight=ctc_w)
    elif dec.get("_target_", "").endswith("ConvASRDecoder"):
        heads["ctc"] = dict(type="ctc")
    else:
        raise NotImplementedError(f"decoder {dec.get('_target_')!r}")

    sa = nc.get("spec_augment") or {}
    cfg = dict(sample_rate=sr, preprocessor=pre, encoder=enc, heads=heads,
               decoding=dict(primary="rnnt" if "rnnt" in heads else "ctc"))
    if sa:
        cfg["spec_augment"] = dict(freq_masks=int(sa.get("freq_masks", 2)), freq_width=int(sa.get("freq_width", 27)),
                                   time_masks=int(sa.get("time_masks", 10)),
                                   time_width=sa.get("time_width", 0.05))
    return cfg


def tdt_durations(nc: dict) -> list[int] | None:
    """TDT duration set of an RNNT-family config, or None for a plain RNNT. NeMo's RNNTJoint for TDT has
    ``num_extra_outputs = len(durations)`` extra logits after [vocab..., blank]; the durations are stated in
    ``model_defaults.tdt_durations``, ``loss.tdt_kwargs.durations`` and ``decoding.durations``. Every place that
    states them must agree with each other and with the joint's extra outputs, else NotImplementedError."""
    j = nc.get("joint") or {}
    extra = int(j.get("num_extra_outputs", 0) or 0)
    cands = {"model_defaults.tdt_durations": (nc.get("model_defaults") or {}).get("tdt_durations"),
             "loss.tdt_kwargs.durations": ((nc.get("loss") or {}).get("tdt_kwargs") or {}).get("durations"),
             "decoding.durations": (nc.get("decoding") or {}).get("durations")}
    cands = {k: [int(d) for d in v] for k, v in cands.items() if v}
    loss_name = str((nc.get("loss") or {}).get("loss_name", "")).lower()
    model_type = str((nc.get("decoding") or {}).get("model_type", "")).lower()
    is_tdt = extra > 0 or bool(cands) or loss_name == "tdt" or model_type == "tdt"
    if not is_tdt:
        return None
    vals = {tuple(v) for v in cands.values()}
    if len(vals) != 1:
        raise NotImplementedError(f"TDT durations missing or inconsistent: {cands}")
    durations = list(vals.pop())
    if extra != len(durations):
        raise NotImplementedError(f"joint.num_extra_outputs={extra} but {len(durations)} TDT durations {durations}")
    if 1 not in durations or durations != sorted(set(durations)) or durations[0] < 0:
        raise NotImplementedError(f"TDT durations {durations} (need sorted, unique, >= 0, containing 1)")
    return durations


def is_sortformer(nc: dict) -> bool:
    return str(nc.get("target", "")).endswith("SortformerEncLabelModel")


def translate_sortformer_config(nc: dict) -> dict:
    """SortformerEncLabelModel (NEST FastConformer -> encoder_proj -> NeMo TransformerEncoder ->
    SortformerModules speaker sigmoids) -> SpeechModel with one ``diar`` SortformerHead.

    NeMo's forward (sortformer_diar_models.py / sortformer_modules.py / transformer_encoders.py):
    emb = encoder_proj(ConformerEncoder(mel)); h = TransformerEncoder(emb, mask) with no positional
    encoding; logits = single_hidden_to_spks(relu(first_hidden_to_hidden(relu(h)))).
    """
    sr, pre, enc = _translate_frontend(nc)
    sm, te = nc["sortformer_modules"], nc.get("transformer_encoder")
    if te is None or not te["_target_"].endswith("TransformerEncoder"):
        raise NotImplementedError(f"transformer_encoder {te!r}")
    if te.get("hidden_act", "relu") != "relu" or te.get("mask_future", False):
        raise NotImplementedError(f"transformer_encoder options: {te}")
    d_hidden = int(te["hidden_size"])
    if int(te["inner_size"]) != 4 * d_hidden:
        raise NotImplementedError(f"transformer inner_size {te['inner_size']} != 4 x hidden_size")
    if int(sm["tf_d_model"]) != d_hidden or int(sm["fc_d_model"]) != enc["d_model"]:
        raise NotImplementedError(f"sortformer_modules dims {sm}")
    if int(nc.get("upsample_factor", 1) or 1) != 1 or nc.get("activity_weight", 0) or nc.get("high_resolution"):
        raise NotImplementedError("sortformer upsample / activity head / high_resolution")
    head = dict(type="sortformer", num_spks=int(sm["num_spks"]), d_hidden=d_hidden,
                n_layers=int(te["num_layers"]), n_heads=int(te["num_attention_heads"]),
                dropout=float(te.get("ffn_dropout", 0.1)), pil_weight=float(nc.get("pil_weight", 0.5)),
                pos_emb=False, norm_first=bool(te.get("pre_ln", False)), out_pre_relu=True)
    if head["norm_first"]:
        raise NotImplementedError("pre_ln transformer (would need the final LayerNorm)")
    # NeMo's streaming settings (80 ms frames), kept for audioforge/streaming_diar.py
    streaming = {k: int(sm[k]) for k in ("chunk_len", "chunk_right_context", "fifo_len", "spkcache_len",
                                         "spkcache_update_period") if k in sm}
    if not nc.get("streaming_mode", False):
        # offline-mode NeMo Sortformers peak-normalize the waveform in process_signal(); we have no such
        # step, and streaming_mode models (v2, v2.1) skip it too
        raise NotImplementedError("non-streaming Sortformer (waveform peak normalization not implemented)")
    return dict(sample_rate=sr, preprocessor=pre, encoder=enc, heads={"diar": head}, streaming=streaming)


# --------------------------------------------------------------------------- weight mapping
_LAYER = [  # (NeMo suffix regex, ours) inside encoder.layers.N.
    (r"norm_feed_forward([12])\.(weight|bias)", r"ff\1.0.\2"),
    (r"feed_forward([12])\.linear1\.(weight|bias)", r"ff\1.1.\2"),
    (r"feed_forward([12])\.linear2\.(weight|bias)", r"ff\1.4.\2"),
    (r"norm_self_att\.(weight|bias)", r"norm_att.\1"),
    (r"self_attn\.(linear_[qkv]|linear_out|linear_pos)\.(weight|bias)", r"att.\1.\2"),
    (r"self_attn\.(pos_bias_[uv])", r"att.\1"),
    (r"norm_conv\.(weight|bias)", r"norm_conv.\1"),
    (r"conv\.pointwise_conv1\.(weight|bias)", r"conv.pw1.\1"),
    (r"conv\.depthwise_conv\.(weight|bias)", r"conv.dw.\1"),
    (r"conv\.batch_norm\.(weight|bias|running_mean|running_var|num_batches_tracked)", r"conv.norm.\1"),
    (r"conv\.pointwise_conv2\.(weight|bias)", r"conv.pw2.\1"),
    (r"norm_out\.(weight|bias)", r"norm_out.\1"),
]
_TOP = [
    (r"encoder\.pre_encode\.conv\.0\.(weight|bias)", r"encoder.pre_encode.convs.0.\1"),
    (r"encoder\.pre_encode\.conv\.2\.(weight|bias)", r"encoder.pre_encode.convs.1.0.\1"),
    (r"encoder\.pre_encode\.conv\.3\.(weight|bias)", r"encoder.pre_encode.convs.1.1.\1"),
    (r"encoder\.pre_encode\.conv\.5\.(weight|bias)", r"encoder.pre_encode.convs.2.0.\1"),
    (r"encoder\.pre_encode\.conv\.6\.(weight|bias)", r"encoder.pre_encode.convs.2.1.\1"),
    (r"encoder\.pre_encode\.out\.(weight|bias)", r"encoder.pre_encode.out.\1"),
    (r"decoder\.prediction\.embed\.weight", r"heads.rnnt.pred.embed.weight"),
    (r"decoder\.prediction\.dec_rnn\.lstm\.(\w+_l\d+)", r"heads.rnnt.pred.lstm.\1"),
    (r"joint\.pred\.(weight|bias)", r"heads.rnnt.joint.pred.\1"),
    (r"joint\.enc\.(weight|bias)", r"heads.rnnt.joint.enc.\1"),
    (r"joint\.joint_net\.2\.(weight|bias)", r"heads.rnnt.joint.out.2.\1"),
    (r"prompt_kernel\.(0|2)\.(weight|bias)", r"heads.rnnt.joint.enc.kernel.\1.\2"),  # EncDecRNNTBPEModelWithPrompt
    (r"(?:ctc_decoder|decoder)\.decoder_layers\.0\.(weight|bias)", r"heads.ctc.proj.\1"),
]
SKIP = ("preprocessor.featurizer.window", "preprocessor.featurizer.fb")  # recomputed by our LogMel (checked)
# SortformerModules.hidden_to_spks (Linear(2*hidden, S), requires_grad False) is never called in NeMo's forward
SKIP_SORTFORMER = ("sortformer_modules.hidden_to_spks.weight", "sortformer_modules.hidden_to_spks.bias")
_SORTFORMER = [  # (NeMo regex, ours); q/k/v are fused into in_proj separately
    (r"sortformer_modules\.encoder_proj\.(weight|bias)", r"heads.diar.proj.\1"),
    (r"sortformer_modules\.first_hidden_to_hidden\.(weight|bias)", r"heads.diar.out.1.\1"),
    (r"sortformer_modules\.single_hidden_to_spks\.(weight|bias)", r"heads.diar.out.3.\1"),
    (r"transformer_encoder\.layers\.(\d+)\.first_sub_layer\.out_projection\.(weight|bias)",
     r"heads.diar.tf.layers.\1.self_attn.out_proj.\2"),
    (r"transformer_encoder\.layers\.(\d+)\.layer_norm_1\.(weight|bias)", r"heads.diar.tf.layers.\1.norm1.\2"),
    (r"transformer_encoder\.layers\.(\d+)\.layer_norm_2\.(weight|bias)", r"heads.diar.tf.layers.\1.norm2.\2"),
    (r"transformer_encoder\.layers\.(\d+)\.second_sub_layer\.dense_in\.(weight|bias)",
     r"heads.diar.tf.layers.\1.linear1.\2"),
    (r"transformer_encoder\.layers\.(\d+)\.second_sub_layer\.dense_out\.(weight|bias)",
     r"heads.diar.tf.layers.\1.linear2.\2"),
]
_QKV = r"transformer_encoder\.layers\.(\d+)\.first_sub_layer\.(query|key|value)_net\.(weight|bias)"


def _map_sortformer(nemo_sd: dict) -> tuple[dict, dict]:
    """Map the Sortformer-specific tensors; returns (mapped, rest-for-the-encoder-mapper)."""
    out, rest, qkv = {}, {}, {}
    for k, v in nemo_sd.items():
        if k in SKIP_SORTFORMER:
            continue
        m = re.fullmatch(_QKV, k)
        if m:
            qkv[(int(m.group(1)), m.group(3), m.group(2))] = v
            continue
        for pat, rep in _SORTFORMER:
            if re.fullmatch(pat, k):
                out[re.sub(pat, rep, k)] = v
                break
        else:
            rest[k] = v
    for n in sorted({i for i, _, _ in qkv}):
        for wb in ("weight", "bias"):
            parts = [qkv.pop((n, wb, x)) for x in ("query", "key", "value")]  # KeyError if one is missing
            out[f"heads.diar.tf.layers.{n}.self_attn.in_proj_{wb}"] = torch.cat(parts, 0)
    if qkv:
        raise KeyError(f"unpaired q/k/v tensors {sorted(qkv)}")
    return out, rest


def map_state_dict(nemo_sd: dict) -> dict:
    out = {}
    if any(k.startswith("sortformer_modules.") for k in nemo_sd):
        out, nemo_sd = _map_sortformer(nemo_sd)
    for k, v in nemo_sd.items():
        if k in SKIP:
            continue
        new = None
        m = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", k)
        if m:
            for pat, rep in _LAYER:
                if re.fullmatch(pat, m.group(2)):
                    new = f"encoder.layers.{m.group(1)}." + re.sub(pat, rep, m.group(2))
                    break
        else:
            for pat, rep in _TOP:
                if re.fullmatch(pat, k):
                    new = re.sub(pat, rep, k)
                    break
        if new is None:
            raise KeyError(f"unmapped NeMo weight {k} {tuple(v.shape)}")
        if new.startswith("heads.ctc.proj.weight"):
            v = v.squeeze(-1)  # Conv1d(k=1) -> Linear
        out[new] = v
    return out


def check_frontend(model: SpeechModel, nemo_sd: dict) -> dict:
    """Max abs difference between our recomputed window/mel filterbank and the checkpoint's."""
    pp, res = model.preprocessor, {}
    if "preprocessor.featurizer.fb" in nemo_sd:
        res["fb_max_abs_diff"] = float((pp.fb.cpu() - nemo_sd["preprocessor.featurizer.fb"][0]).abs().max())
    if "preprocessor.featurizer.window" in nemo_sd:
        res["window_max_abs_diff"] = float((pp.window.cpu() - nemo_sd["preprocessor.featurizer.window"]).abs().max())
    return res


# --------------------------------------------------------------------------- Nemotron-3-Diarization
# nvidia/Nemotron-3-Diarization (OpenMDW-1.1) is a SortformerEncLabelModel with a different network: no
# FastConformer and no Sortformer transformer. NeMo's forward (checked against HF transformers'
# modeling_nemotron3_diarization.py + convert_nemotron3_diarization_to_hf.py):
#   mel (10 ms, 128) -> stack 8 frames -> Linear(1024->512, no bias)        = encoder.pre_encode.proj
#   -> LayerNorm (embed_norm) -> 31 pre-LN blocks [x + MHA_rope(LN1 x); x + W2 gelu(W1 LN2 x)] -> final LN
#   -> encoder_proj 512->192 -> Conv1d(192 -> 8*192, k3) sub-pixel upsample to 10 ms frames
#   -> relu -> first_hidden_to_hidden -> relu -> single_hidden_to_spks (8 speakers) -> sigmoid.
# RoPE: rotate-half, theta 10000, full head_dim 64, positions restart at 0 on every forward (every streaming
# chunk). The fused w_qkv rows are [q | k | v], no bias; out_proj has a bias. Exact GELU.
# The streaming speaker cache / FIFO store the stacked projections (pre-embed_norm), so here the "encoder" is
# only the frame stacking and the whole transformer lives in the diar head: audioforge/streaming_diar.py's
# StreamingDiarizer then re-runs the full network over [cache | fifo | chunk | rc] exactly like NeMo.
# SpeechModel (model.py) always builds a FastConformer, so this is a separate duck-typed model class exposing
# the SpeechModel surface that scripts/research/eval_stage1.py and StreamingDiarizer use.

def is_nemotron_diar(nc: dict) -> bool:
    e = nc.get("encoder") or {}
    return (str(nc.get("target", "")).endswith("SortformerEncLabelModel")
            and str(e.get("_target_", "")).endswith("TransformerEncoder") and e.get("subsampling") == "feature_stacking")


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], -1)


class _RoPEAttention(torch.nn.Module):
    def __init__(self, d: int, n_heads: int, theta: float = 10000.0):
        super().__init__()
        self.h, self.hd = n_heads, d // n_heads
        self.w_qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.out_proj = torch.nn.Linear(d, d)
        self.register_buffer("inv_freq", 1.0 / theta ** (torch.arange(0, self.hd, 2).float() / self.hd),
                             persistent=False)

    def forward(self, x, valid):
        B, T, D = x.shape
        q, k, v = (t.view(B, T, self.h, self.hd).transpose(1, 2) for t in self.w_qkv(x).chunk(3, -1))
        f = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * self.inv_freq.float()[None]
        emb = torch.cat([f, f], -1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        return self.out_proj(a.transpose(1, 2).reshape(B, T, D))


class _PreLNBlock(torch.nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, theta: float):
        super().__init__()
        self.norm1, self.norm2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
        self.attn = _RoPEAttention(d, n_heads, theta)
        self.ffn = torch.nn.Module()
        self.ffn.net = torch.nn.Sequential(torch.nn.Linear(d, d_ff), torch.nn.GELU(), torch.nn.Identity(),
                                           torch.nn.Linear(d_ff, d))  # NeMo: net.0 / gelu / dropout / net.3

    def forward(self, x, valid):
        x = x + self.attn(self.norm1(x), valid)
        return x + self.ffn.net(self.norm2(x))


class FeatureStackingEncoder(torch.nn.Module):
    """NeMo feature_stacking subsampling: zero-pad to a multiple of 8 mel frames, stack, project (no bias)."""

    causal = False
    att_context_size = [-1, -1]
    att_context_sizes = [[-1, -1]]

    def __init__(self, feat_in: int = 128, d_model: int = 512, subsampling_factor: int = 8):
        super().__init__()
        self.d_model, self.subsampling_factor = d_model, subsampling_factor
        self.pre_encode = torch.nn.Module()
        self.pre_encode.proj = torch.nn.Linear(subsampling_factor * feat_in, d_model, bias=False)
        self.layers = torch.nn.ModuleList()  # the transformer is in the diar head (see above)

    def forward(self, feats, flen, att_context_size=None, spk_act=None, return_hidden=False):
        f = self.subsampling_factor
        x = feats.transpose(1, 2)  # (B, T, n_mels); LogMel already zeroes frames >= flen, as NeMo
        x = torch.nn.functional.pad(x, (0, 0, 0, -x.shape[1] % f))
        B, T, M = x.shape
        enc = self.pre_encode.proj(x.reshape(B, T // f, M * f))
        elen = torch.div(flen + f - 1, f, rounding_mode="floor")
        return (enc, elen, [enc]) if return_hidden else (enc, elen)


class Nemotron3DiarHead(torch.nn.Module):
    """embed_norm -> pre-LN RoPE transformer -> final LN -> proj -> sub-pixel upsample -> output MLP.

    ``forward_hr`` gives NeMo's 10 ms logits (B, 8T, S). ``forward``/``forward_emb`` give 80 ms logits
    logit(mean of the 8 sub-frame probabilities) - the pooling NeMo/HF use to score frames for the speaker
    cache - so it has the 80 ms frame interface of SortformerHead (eval, StreamingDiarizer).
    ``pool="max"``: max over the sub-frames instead (an 80 ms frame is active if any 10 ms sub-frame is,
    the "any overlap" convention of data.rttm_to_frames / the AMI labels).
    ``decode_top_k``: keep only the k most active columns (frames > threshold, ties by summed probability),
    zero the rest (8-speaker model scored against <= 4-speaker references)."""

    key = "spk_targets"

    def __init__(self, d_model: int = 512, num_spks: int = 8, d_hidden: int = 192, n_layers: int = 31,
                 n_heads: int = 8, d_ff: int = 2048, upsample: int = 8, rope_theta: float = 10000.0,
                 decode_top_k: int | None = None, pool: str = "mean"):
        super().__init__()
        if pool not in ("mean", "max"):
            raise ValueError(f"pool={pool!r}")
        self.num_spks, self.up, self.decode_top_k, self.pool = num_spks, upsample, decode_top_k, pool
        self.embed_norm = torch.nn.LayerNorm(d_model)
        self.layers = torch.nn.ModuleList([_PreLNBlock(d_model, n_heads, d_ff, rope_theta) for _ in range(n_layers)])
        self.final_norm = torch.nn.LayerNorm(d_model)
        self.proj = torch.nn.Linear(d_model, d_hidden)
        self.upsample = torch.nn.Conv1d(d_hidden, d_hidden * upsample, 3, padding=1)
        self.out = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(d_hidden, d_hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(d_hidden, num_spks))
        self.sil_emb = torch.nn.Parameter(torch.zeros(d_model))  # NeMo learnable_sil_emb (NeMo's cache rule only)

    def embed_frames(self, enc):
        return enc  # the cache stores encoder-input frames: the head re-runs the whole transformer

    def forward_hr(self, x, valid=None):
        if valid is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        h = self.embed_norm(x)
        for layer in self.layers:
            h = layer(h, valid)
        h = self.proj(self.final_norm(h))
        B, T, D = h.shape
        h = self.upsample(h.transpose(1, 2)).transpose(1, 2).reshape(B, T * self.up, D)
        return self.out(h)

    def forward_emb(self, x, valid=None):
        p = self.forward_hr(x, valid).float().sigmoid()
        B, T8, S = p.shape
        p = p.view(B, T8 // self.up, self.up, S)
        p = (p.mean(2) if self.pool == "mean" else p.amax(2)).clamp(1e-7, 1 - 1e-7)
        return torch.logit(p).to(x.dtype)

    def forward_chunk(self, spkcache, fifo, chunk, right_context=None):
        parts = [p for p in (spkcache, fifo, chunk, right_context) if p is not None and p.shape[1] > 0]
        return self.forward_emb(torch.cat(parts, 1))

    def forward(self, enc, enc_len):
        valid = torch.arange(enc.shape[1], device=enc.device)[None] < enc_len[:, None]
        return self.forward_emb(enc, valid)

    @torch.no_grad()
    def decode(self, enc, enc_len, threshold: float = 0.5, top_k: int | None = None, **_):
        p = self(enc, enc_len).sigmoid()
        return top_k_columns(p, top_k or self.decode_top_k, threshold, enc_len)


def top_k_columns(p: torch.Tensor, k: int | None, threshold: float = 0.5, lengths=None) -> torch.Tensor:
    """(B,T,S) probabilities -> hard (B,T,S) decisions keeping only each item's k most active columns
    (count of frames > threshold within its length; ties: summed probability). Column order is kept."""
    hard = (p > threshold).float()
    if not k or k >= p.shape[2]:
        return hard
    valid = torch.ones(p.shape[:2], dtype=torch.bool, device=p.device) if lengths is None else (
        torch.arange(p.shape[1], device=p.device)[None] < lengths[:, None])
    score = (hard * valid[..., None]).sum(1) + 1e-3 * (p * valid[..., None]).sum(1) / p.shape[1]
    keep = torch.zeros_like(score, dtype=torch.bool).scatter_(1, score.topk(k, dim=1).indices, True)
    return hard * keep[:, None, :]


class Nemotron3Diarizer(torch.nn.Module):
    """LogMel + FeatureStackingEncoder + one ``diar`` Nemotron3DiarHead, with the SpeechModel surface used by
    scripts/research/eval_stage1.py (encode/heads/head_cfg/head_input/tokenizer) and StreamingDiarizer."""

    def __init__(self, cfg: dict, tokenizer=None):
        super().__init__()
        from .features import LogMel
        self.cfg = dict(cfg)
        self.tokenizer = None
        pre = dict(cfg["preprocessor"])
        pre.setdefault("sample_rate", cfg.get("sample_rate", 16000))
        self.preprocessor = LogMel(**pre)
        self.encoder = FeatureStackingEncoder(**cfg["encoder"])
        hc = {k: dict(v) for k, v in cfg["heads"].items()}
        self.head_cfg = hc
        self.heads = torch.nn.ModuleDict({k: Nemotron3DiarHead(**{a: b for a, b in v.items() if a != "type"})
                                          for k, v in hc.items()})
        self.layer_mix = torch.nn.ParameterDict()
        self.primary = None

    def features(self, audio, audio_len):
        return self.preprocessor(audio, audio_len)

    def encode(self, audio, audio_len, att_context_size=None, spk_act=None, return_hidden=False):
        feats, flen = self.features(audio, audio_len)
        return self.encoder(feats, flen, return_hidden=return_hidden)

    def head_input(self, name, enc, hidden):
        return enc

    @torch.no_grad()
    def diarize(self, audio, high_res: bool = False) -> torch.Tensor:
        """One offline pass over a 1-D waveform -> (T, S) probabilities (80 ms, or 10 ms with high_res)."""
        x = torch.as_tensor(audio, dtype=torch.float32).flatten()[None]
        enc, elen = self.encode(x, torch.tensor([x.shape[1]]))
        h = self.heads["diar"]
        return (h.forward_hr(enc) if high_res else h(enc, elen))[0].sigmoid()

    @property
    def frame_sec(self) -> float:
        return self.preprocessor.hop * self.encoder.subsampling_factor / self.preprocessor.sample_rate

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def translate_nemotron_diar_config(nc: dict) -> dict:
    sr, pre = _translate_preprocessor(nc)
    e, sm = nc["encoder"], nc["sortformer_modules"]
    want = {"self_attention_model": "rope", "qkv_bias": False, "qk_norm": False, "pre_block_norm": True,
            "xscaling": False, "attn_mode": "full", "feat_out": -1}
    bad = {k: e.get(k) for k, v in want.items() if e.get(k, v) != v}
    if bad or pre["normalize"] != "NA":
        raise NotImplementedError(f"Nemotron diar encoder options {bad} normalize={pre['normalize']}")
    if not nc.get("high_resolution") or int(nc.get("output_subsampling_factor", 1)) != 1:
        raise NotImplementedError("only high_resolution, output_subsampling_factor 1 (10 ms sub-pixel output)")
    if not nc.get("streaming_mode", False):
        raise NotImplementedError("non-streaming Sortformer (waveform peak normalization not implemented)")
    d = int(e["d_model"])
    if int(sm["fc_d_model"]) != d:
        raise NotImplementedError(f"sortformer_modules dims {sm}")
    f = int(e["subsampling_factor"])
    # type "sortformer": scripts/research/eval_stage1.py / heads.turn._diar_name find the diar head by it; the class is
    # chosen by cfg arch (Nemotron3Diarizer builds Nemotron3DiarHead, never SortformerHead)
    head = dict(type="sortformer", d_model=d, num_spks=int(sm["num_spks"]), d_hidden=int(sm["tf_d_model"]),
                n_layers=int(e["n_layers"]), n_heads=int(e["n_heads"]), d_ff=int(float(e.get("ff_expansion", 4)) * d),
                upsample=f, rope_theta=float(e.get("rope_base", 10000.0)))
    # the .nemo holds training-time streaming values (fifo 0, chunk 264); these are the card's "low latency"
    # (1.04 s) profile, also the HF streaming_config default
    streaming = dict(chunk_len=9, chunk_right_context=4, fifo_len=264, spkcache_len=264,
                     spkcache_update_period=222, sil_frames=int(sm.get("spkcache_sil_frames_per_spk", 1)),
                     strong_boost_rate=float(sm.get("strong_boost_rate", 0.75)))
    return dict(arch="nemotron3_diar", sample_rate=sr, preprocessor=pre,
                encoder=dict(feat_in=int(e["feat_in"]), d_model=d, subsampling_factor=f),
                heads={"diar": head}, streaming=streaming)


# dropped: frontend buffers (recomputed, checked), hidden_to_spks (never called), activity_head (aux training loss)
SKIP_NEMOTRON = (r"preprocessor\.featurizer\.(window|fb)", r"sortformer_modules\.hidden_to_spks\.(weight|bias)",
                 r"sortformer_modules\.activity_head\..+")
_NEMOTRON = [
    (r"encoder\.pre_encode\.proj\.weight", r"encoder.pre_encode.proj.weight"),
    (r"encoder\.embed_norm\.(weight|bias)", r"heads.diar.embed_norm.\1"),
    (r"encoder\.final_norm\.(weight|bias)", r"heads.diar.final_norm.\1"),
    (r"encoder\.layers\.(\d+)\.(norm[12]\.(?:weight|bias)|attn\.w_qkv\.weight|attn\.out_proj\.(?:weight|bias)|"
     r"ffn\.net\.[03]\.(?:weight|bias))", r"heads.diar.layers.\1.\2"),
    (r"sortformer_modules\.encoder_proj\.(weight|bias)", r"heads.diar.proj.\1"),
    (r"sortformer_modules\.subpixel_upsample\.(weight|bias)", r"heads.diar.upsample.\1"),
    (r"sortformer_modules\.first_hidden_to_hidden\.(weight|bias)", r"heads.diar.out.1.\1"),
    (r"sortformer_modules\.single_hidden_to_spks\.(weight|bias)", r"heads.diar.out.3.\1"),
    (r"sortformer_modules\.learnable_sil_emb", r"heads.diar.sil_emb"),
]


def map_nemotron_state_dict(nemo_sd: dict) -> dict:
    out = {}
    for k, v in nemo_sd.items():
        if any(re.fullmatch(p, k) for p in SKIP_NEMOTRON):
            continue
        for pat, rep in _NEMOTRON:
            if re.fullmatch(pat, k):
                out[re.sub(pat, rep, k)] = v.float()  # the checkpoint is bf16
                break
        else:
            raise KeyError(f"unmapped NeMo weight {k} {tuple(v.shape)}")
    return out


def _import_nemotron_diar(src, path, nc, nemo_sd, t0, verbose):
    cfg = translate_nemotron_diar_config(nc)
    cfg["nemo_source"] = dict(checkpoint=str(src), target=nc.get("target"), nemo_version=nc.get("nemo_version"),
                              license=license_of(src))
    model = Nemotron3Diarizer(cfg)
    sd = map_nemotron_state_dict(nemo_sd)
    ours = model.state_dict()
    missing, unexpected = sorted(set(ours) - set(sd)), sorted(set(sd) - set(ours))
    shape = [k for k in sd if k in ours and ours[k].shape != sd[k].shape]
    if missing or unexpected or shape:
        raise KeyError(f"missing={missing[:10]} unexpected={unexpected[:10]} "
                       f"shape_mismatch={[(k, tuple(ours[k].shape), tuple(sd[k].shape)) for k in shape[:10]]}")
    model.load_state_dict(sd, strict=True)
    model.eval()
    fe = {k: v.float() for k, v in nemo_sd.items() if k.startswith("preprocessor.")}
    model.import_info = dict(load_sec=time.time() - t0, n_tensors=len(nemo_sd), n_loaded=len(sd),
                             **check_frontend(model, fe))
    if verbose:
        log.info(f"imported {path.name}: {model.num_params() / 1e6:.1f}M params, {len(sd)} tensors, "
              f"{model.import_info}")
    return model


def load_any(path: str | Path, device="cpu"):
    """train.load_model, plus .afm archives of the arch-tagged imports here (cfg arch: nemotron3_diar)."""
    import tempfile

    from .train import load_model
    with tarfile.open(path) as tar:
        cfg = yaml.safe_load(tar.extractfile("config.yaml").read())
    if cfg.get("arch") != "nemotron3_diar":
        return load_model(path, device)
    with tempfile.TemporaryDirectory() as d, tarfile.open(path) as tar:
        tar.extract("model_weights.pt", d, filter="data")
        sd = torch.load(Path(d) / "model_weights.pt", map_location="cpu")
    model = Nemotron3Diarizer(cfg)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


# --------------------------------------------------------------------------- TitaNet (speaker embeddings)
def import_titanet(path_or_hf_id: str | Path = "nvidia/speakerverification_en_titanet_large", verbose: bool = False):
    """NVIDIA TitaNet-Large (EncDecSpeakerLabelModel, CC-BY-4.0) -> ``baselines.sd.TitaNet``, strict load.
    Not a SpeechModel: a ConvASREncoder (SE separable conv blocks) + attentive-stats pooling, 192-d embeddings
    via ``model.embed([audio, ...])``. See research/archive/BASELINES.md, "Speaker verification"."""
    from .baselines.sd import TitaNet, load_nemo_conv
    model = load_nemo_conv(path_or_hf_id, verbose=verbose)
    if not isinstance(model, TitaNet):
        raise ValueError(f"{path_or_hf_id}: not an EncDecSpeakerLabelModel")
    return model


# --------------------------------------------------------------------------- AmberNet (spoken language ID)
AMBERNET_NGC = ("https://api.ngc.nvidia.com/v2/models/nvidia/nemo/langid_ambernet/versions/1.12.0/files/ambernet.nemo")
AMBERNET_NEMO = NEMO_DIR / "langid_ambernet.nemo"


def import_ambernet(path: str | Path | None = None, verbose: bool = False):
    """NVIDIA langid_ambernet (EncDecSpeakerLabelModel, 107 VoxLingua107 languages) -> ``baselines.lid.AmberNet``,
    strict load. Not on Hugging Face: fetched from NGC (public, guest download; "NGC Terms of Use", see
    research/archive/LID.md) into data/nemo/langid_ambernet.nemo when ``path`` is None and the file is missing."""
    from .baselines.lid import load_ambernet
    p = Path(path) if path else AMBERNET_NEMO
    if not p.exists():
        import requests
        p.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(AMBERNET_NGC, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(p.with_suffix(".part"), "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        p.with_suffix(".part").replace(p)
    return load_ambernet(p, verbose=verbose)


# --------------------------------------------------------------------------- entry point
# licenses from the model cards (research/archive/raw/cards/); anything else is assumed CC-BY-4.0 like the ASR cards
LICENSES = {"Nemotron-3-Diarization": "OpenMDW-1.1",
            "langid_ambernet": "NGC Terms of Use",  # NGC card; not on Hugging Face
            "nemotron-speech-streaming-en-0.6b": "NVIDIA Open Model License",  # the card; not CC-BY-4.0
            "nemotron-3.5-asr-streaming-0.6b": "NVIDIA Open Model License",
            "multitalker-parakeet-streaming-0.6b-v1": "NVIDIA Open Model License",
            "diar_streaming_sortformer_4spk-v2.1": "NVIDIA Open Model License",
            "diar_streaming_sortformer_4spk-v2": "CC-BY-4.0",
            "diar_sortformer_4spk-v1": "CC-BY-NC-4.0"}


def license_of(path_or_hf_id) -> str:
    name = Path(str(path_or_hf_id)).name.removesuffix(".nemo")
    return LICENSES.get(name, "CC-BY-4.0")


def import_nemo(path_or_hf_id: str | Path, verbose: bool = False) -> SpeechModel:
    """Build an audioforge SpeechModel carrying the .nemo checkpoint's weights and tokenizer."""
    t0 = time.time()
    path = resolve(path_or_hf_id)
    nc, nemo_sd, files = read_nemo(path)
    if is_nemotron_diar(nc):
        return _import_nemotron_diar(path_or_hf_id, path, nc, nemo_sd, t0, verbose)
    tok = (SentencePieceTokenizer(_nemo_file(files, nc["tokenizer"]["model_path"]), specials=[])
           if "tokenizer" in nc else None)  # diarization models have no tokenizer
    cfg = translate_config(nc, nemo_sd.keys())
    if tok is not None and "prompt" in cfg["heads"].get("rnnt", {}):
        # a language-prompted model emits its detected language as a tag piece ("<en-US>") after the terminal
        # punctuation: kept in the token ids (language ID), dropped from the decoded text
        tok.specials = [p for p in (tok.sp.id_to_piece(i) for i in range(tok.vocab_size))
                        if re.fullmatch(r"<[a-z]{2,3}-[A-Za-z]{2,4}>", p)]
        cfg["heads"]["rnnt"]["prompt"]["tag_pieces"] = len(tok.specials)
    cfg["nemo_source"] = dict(checkpoint=str(path_or_hf_id), target=nc.get("target"),
                              nemo_version=nc.get("nemo_version"), license=license_of(path_or_hf_id))
    n_cls = nc.get("decoder", {}).get("vocab_size") or nc.get("decoder", {}).get("num_classes")
    if n_cls is not None and tok is not None and int(n_cls) != tok.vocab_size:
        raise ValueError(f"tokenizer has {tok.vocab_size} pieces but the decoder expects {n_cls}")
    model = SpeechModel(cfg, tok)
    sd = map_state_dict(nemo_sd)
    ours = model.state_dict()
    # encoder.use_bias: false (nemotron-speech-streaming-en-0.6b, parakeet_realtime_eou): the conformer layers'
    # linear / conv layers have no bias tensors; a zero bias is exact. Only those may be missing.
    zeros = _zero_missing_encoder_biases(nc, sd, ours)
    missing = sorted(set(ours) - set(sd))
    unexpected = sorted(set(sd) - set(ours))
    shape = [k for k in sd if k in ours and ours[k].shape != sd[k].shape]
    if missing or unexpected or shape:
        raise KeyError(f"missing={missing[:10]} unexpected={unexpected[:10]} "
                       f"shape_mismatch={[(k, tuple(ours[k].shape), tuple(sd[k].shape)) for k in shape[:10]]}")
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.import_info = dict(load_sec=time.time() - t0, n_tensors=len(nemo_sd), n_loaded=len(sd) - len(zeros),
                             zero_biases=len(zeros), **check_frontend(model, nemo_sd))
    if verbose:
        log.info(f"imported {path.name}: {model.num_params() / 1e6:.1f}M params, {len(sd)} tensors, "
              f"{model.import_info}")
    return model


import_hybrid = import_nemo  # the hybrid RNNT+CTC case is just the most complete one


# --------------------------------------------------------------------------- Parakeet-Realtime-EOU (RNNT + <EOU> token)
# nvidia/parakeet_realtime_eou_120m-v1 (NVIDIA Open Model License): a cache-aware streaming FastConformer-RNNT
# (17 layers, att_context [70, 1] chunked_limited, causal subsampling, layer-norm conv, xscaling off, normalize NA)
# whose SentencePiece vocabulary ends with two control pieces, <EOU> (id 1024: end of utterance) and <EOB> (1025:
# end of backchannel); the RNNT blank is 1026. The only thing the generic importer does not cover is
# ``encoder.use_bias: false``: NeMo builds the conformer layers' linear / conv layers without biases, while our
# FastConformer always has them. A zero bias is the identity, so the importer fills exactly the missing encoder
# bias tensors with zeros (anything else missing still raises); import_nemo does the same since the 0.6B import.
EOU_PIECES = ("<EOU>", "<EOB>")
LICENSES_EOU = {"parakeet_realtime_eou_120m-v1": "NVIDIA Open Model License"}


def _zero_missing_encoder_biases(nc: dict, sd: dict, ours: dict) -> list[str]:
    """Add zero tensors for the encoder-layer biases a ``use_bias: false`` checkpoint lacks; returns their names."""
    if nc.get("encoder", {}).get("use_bias", True):
        return []
    added = []
    for k in sorted(set(ours) - set(sd)):
        if re.fullmatch(r"encoder\.layers\.\d+\..+\.bias", k):
            sd[k] = torch.zeros_like(ours[k])
            added.append(k)
    return added


def import_eou(path_or_hf_id: str | Path = "nvidia/parakeet_realtime_eou_120m-v1", verbose: bool = False) -> SpeechModel:
    """Import a Parakeet-Realtime-EOU checkpoint (RNNT path of import_nemo + zero biases for use_bias: false).
    ``model.eou_ids`` = {piece: token id} of the control pieces; ``model.import_info['zero_biases']`` counts the
    filled tensors."""
    t0 = time.time()
    path = resolve(path_or_hf_id)
    nc, nemo_sd, files = read_nemo(path)
    tok = SentencePieceTokenizer(_nemo_file(files, nc["tokenizer"]["model_path"]), specials=[])
    cfg = translate_config(nc)
    if "rnnt" not in cfg["heads"]:
        raise NotImplementedError("import_eou expects an RNNT checkpoint")
    name = Path(str(path_or_hf_id)).name.removesuffix(".nemo")
    cfg["nemo_source"] = dict(checkpoint=str(path_or_hf_id), target=nc.get("target"),
                              nemo_version=nc.get("nemo_version"),
                              license=LICENSES_EOU.get(name, license_of(path_or_hf_id)))
    n_cls = nc["decoder"].get("vocab_size")
    if n_cls is not None and int(n_cls) != tok.vocab_size:
        raise ValueError(f"tokenizer has {tok.vocab_size} pieces but the decoder expects {n_cls}")
    vocab = nc.get("joint", {}).get("vocabulary") or []
    eou_ids = {p: vocab.index(p) for p in EOU_PIECES if p in vocab}
    if "<EOU>" not in eou_ids:
        raise ValueError("no <EOU> piece in the vocabulary")
    model = SpeechModel(cfg, tok)
    sd = map_state_dict(nemo_sd)
    ours = model.state_dict()
    zeros = _zero_missing_encoder_biases(nc, sd, ours)
    missing, unexpected = sorted(set(ours) - set(sd)), sorted(set(sd) - set(ours))
    shape = [k for k in sd if k in ours and ours[k].shape != sd[k].shape]
    if missing or unexpected or shape:
        raise KeyError(f"missing={missing[:10]} unexpected={unexpected[:10]} "
                       f"shape_mismatch={[(k, tuple(ours[k].shape), tuple(sd[k].shape)) for k in shape[:10]]}")
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.eou_ids = eou_ids
    model.import_info = dict(load_sec=time.time() - t0, n_tensors=len(nemo_sd), n_loaded=len(sd) - len(zeros),
                             zero_biases=len(zeros), eou_ids=eou_ids, **check_frontend(model, nemo_sd))
    if verbose:
        log.info(f"imported {path.name}: {model.num_params() / 1e6:.1f}M params, {model.import_info}")
    return model


def main(argv=None):
    import argparse

    from .train import save_model
    p = argparse.ArgumentParser(description="convert a NeMo .nemo FastConformer checkpoint to an audioforge .afm")
    p.add_argument("src", help=".nemo path or HF repo id")
    p.add_argument("out", help="output .afm path")
    p.add_argument("--att-context-size", default=None, help="default context, e.g. 70,13 (first listed if unset)")
    a = p.parse_args(argv)
    m = import_nemo(a.src, verbose=True)
    if isinstance(m, Nemotron3Diarizer):
        save_model(m, a.out)  # same .afm layout; load it with nemo_import.load_any (cfg arch: nemotron3_diar)
        print(f"wrote {a.out}")
        return
    if a.att_context_size:
        want = [int(x) for x in a.att_context_size.split(",")]
        sizes = m.cfg["encoder"]["att_context_sizes"]
        m.cfg["encoder"]["att_context_sizes"] = [want] + [s for s in sizes if s != want]
    save_model(m, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
