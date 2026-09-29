"""Tokenizers. NVIDIA uses one unified SentencePiece BPE vocabulary per model
(1k-16k tokens) plus special tokens for task prompts (<|en|>, <|transcribe|>,
<|pnc|>, <EOU>, ...). ``CharTokenizer`` is a dependency-free fallback.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

DEFAULT_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class CharTokenizer:
    kind = "char"

    def __init__(self, chars: list[str], specials: list[str] | None = None):
        specials = list(dict.fromkeys(DEFAULT_SPECIALS + (specials or [])))
        self.itos = specials + [c for c in chars if c not in specials]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.specials = specials

    @classmethod
    def train(cls, texts, specials=None, **_):
        return cls(sorted(set("".join(texts))), specials)

    @property
    def vocab_size(self):
        return len(self.itos)

    def token_id(self, s: str) -> int:
        return self.stoi[s]

    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for sp in self.specials:  # allow inline special tokens
                if text.startswith(sp, i):
                    out.append(self.stoi[sp])
                    i += len(sp)
                    break
            else:
                out.append(self.stoi.get(text[i], self.stoi["<unk>"]))
                i += 1
        return out

    def decode(self, ids) -> str:
        n = len(DEFAULT_SPECIALS)
        return "".join(self.itos[i] for i in ids if i >= n and self.itos[i] not in self.specials)

    def save(self, d: Path):
        (Path(d) / "tokenizer.json").write_text(json.dumps({"kind": "char", "itos": self.itos,
                                                            "specials": self.specials}))

    @classmethod
    def load(cls, d: Path):
        cfg = json.loads((Path(d) / "tokenizer.json").read_text())
        n = len(cfg["specials"])
        return cls(cfg["itos"][n:], cfg["specials"])


class SentencePieceTokenizer:
    kind = "spm"

    def __init__(self, model_bytes: bytes, specials: list[str]):
        import sentencepiece as spm
        self.model_bytes, self.specials = model_bytes, specials
        self.sp = spm.SentencePieceProcessor(model_proto=model_bytes)

    @classmethod
    def train(cls, texts, vocab_size: int = 1024, specials=None, model_type="bpe"):
        import sentencepiece as spm
        specials = list(dict.fromkeys(DEFAULT_SPECIALS + (specials or [])))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "text.txt"
            p.write_text("\n".join(texts))
            spm.SentencePieceTrainer.train(
                input=str(p), model_prefix=str(Path(d) / "m"), vocab_size=vocab_size, model_type=model_type,
                pad_id=0, bos_id=1, eos_id=2, unk_id=3, user_defined_symbols=specials[4:],
                character_coverage=1.0, hard_vocab_limit=False, minloglevel=2)
            return cls((Path(d) / "m.model").read_bytes(), specials)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def token_id(self, s):
        return self.sp.piece_to_id(s)

    def encode(self, text):
        """Inline specials (e.g. "<HOLD>", "<YIELD>": audioforge.vocab.extend_model_vocab) map to their own id and
        never change the pieces around them: the text is split at the specials and each plain segment is encoded
        on its own (SentencePiece itself would insert a lone "\u2581" piece before a user-defined symbol)."""
        if not self.specials or not any(sp in text for sp in self.specials):
            return self.sp.encode(text)
        pat = "(" + "|".join(re.escape(sp) for sp in sorted(self.specials, key=len, reverse=True)) + ")"
        out = []
        for part in re.split(pat, text):
            if part in self.specials:
                out.append(self.sp.piece_to_id(part))
            elif part.strip():
                out += self.sp.encode(part.strip())
        return out

    def decode(self, ids):
        special = {self.token_id(s) for s in self.specials}
        return self.sp.decode([i for i in ids if i not in special])

    def save(self, d: Path):
        (Path(d) / "tokenizer.model").write_bytes(self.model_bytes)
        (Path(d) / "tokenizer.json").write_text(json.dumps({"kind": "spm", "specials": self.specials}))

    @classmethod
    def load(cls, d: Path):
        cfg = json.loads((Path(d) / "tokenizer.json").read_text())
        return cls((Path(d) / "tokenizer.model").read_bytes(), cfg["specials"])


def train_tokenizer(kind: str, texts, **kw):
    return {"char": CharTokenizer, "spm": SentencePieceTokenizer}[kind].train(texts, **kw)


def load_tokenizer(d: Path):
    kind = json.loads((Path(d) / "tokenizer.json").read_text())["kind"]
    return {"char": CharTokenizer, "spm": SentencePieceTokenizer}[kind].load(d)
