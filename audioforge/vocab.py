"""Add tokens to a trained model's SentencePiece vocabulary without disturbing the existing ids.

Used for the conversational action tokens <HOLD> / <YIELD>: the pretrained RNNT / CTC heads
of runs/stage1_heads_pretrained.afm use NVIDIA's 1024-piece BPE model, in which the blank is id 1024 (= vocab_size,
heads/asr.py). New pieces are APPENDED to the SentencePiece proto as user-defined symbols (ids 1024, 1025, ...), so
every existing piece keeps its id and its rows; the blank moves to the new vocab_size and its rows move with it.

    extend_spm(model_bytes, pieces)          -> new model proto bytes
    extend_model_vocab(model, pieces, init)  -> a new SpeechModel (same config) with the extended tokenizer, whose
                                                RNNT / CTC / AED output rows and PredictionNet embeddings are the old
                                                ones re-indexed; the new tokens' rows start at the mean of the token
                                                rows with the lowest token bias (``init="mean"``), or zero.

    python -m audioforge.vocab runs/stage1_heads_pretrained.afm runs/stage1_rnnt_yieldhold_init.afm '<HOLD>' '<YIELD>'
"""
from __future__ import annotations

import sys

import torch

from .model import SpeechModel
from .tokenizer import SentencePieceTokenizer


def extend_spm(model_bytes: bytes, pieces: list[str]) -> bytes:
    from sentencepiece import sentencepiece_model_pb2 as pb
    m = pb.ModelProto()
    m.ParseFromString(model_bytes)
    have = {p.piece for p in m.pieces}
    for s in pieces:
        if s in have:
            raise ValueError(f"piece {s!r} already in the vocabulary")
        pc = m.pieces.add()
        pc.piece, pc.score, pc.type = s, 0.0, pb.ModelProto.SentencePiece.USER_DEFINED
    return m.SerializeToString()


def _remap_rows(old: torch.Tensor, V: int, n_new: int, init: str) -> torch.Tensor:
    """Rows [0, V) stay, the old row V (blank / SOS) moves to V + n_new, rows [V, V + n_new) are new."""
    new = old.new_zeros((old.shape[0] + n_new, *old.shape[1:]))
    new[:V] = old[:V]
    new[V + n_new:] = old[V:]
    if init == "mean":
        new[V:V + n_new] = old[:V].mean(0, keepdim=True)
    return new


def extend_model_vocab(model: SpeechModel, pieces: list[str], init: str = "mean") -> SpeechModel:
    tok = model.tokenizer
    assert isinstance(tok, SentencePieceTokenizer), "only SentencePiece tokenizers can be extended"
    V, n = tok.vocab_size, len(pieces)
    new_tok = SentencePieceTokenizer(extend_spm(tok.model_bytes, pieces), list(tok.specials) + list(pieces))
    assert new_tok.vocab_size == V + n and [new_tok.token_id(p) for p in pieces] == list(range(V, V + n))
    new = SpeechModel(model.cfg, new_tok)
    sd, nsd = model.state_dict(), new.state_dict()
    out = {}
    for k, v in sd.items():
        if k in nsd and nsd[k].shape == v.shape:
            out[k] = v
            continue
        assert k in nsd and nsd[k].shape[0] == v.shape[0] + n and v.shape[0] == V + 1, (k, v.shape, nsd[k].shape)
        r = _remap_rows(v, V, n, init)
        if k.endswith("bias") and init == "mean":  # new tokens start rare: the lowest token bias
            r[V:V + n] = v[:V].min()
        out[k] = r
    new.load_state_dict(out, strict=True)
    new.vocab_extension = {"pieces": list(pieces), "ids": list(range(V, V + n)), "old_blank": V, "blank": V + n,
                           "resized": sorted(k for k in sd if sd[k].shape != nsd[k].shape)}
    return new


def main(argv=None):
    from .train import load_model, save_model
    src, dst, *pieces = argv or sys.argv[1:]
    m = load_model(src)
    new = extend_model_vocab(m, pieces)
    new.cfg["vocab_extension"] = new.vocab_extension
    save_model(new, dst)
    print(f"{src} -> {dst}: {new.vocab_extension}")


if __name__ == "__main__":
    main()
