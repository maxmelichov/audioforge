"""Catalog of NVIDIA audio models on Hugging Face: fetch, classify, search.

No torch dependency. Data comes from the public HF API
(https://huggingface.co/api/models?author=nvidia) plus each model's README.
"""
from __future__ import annotations

import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

HF_API = "https://huggingface.co/api/models?author=nvidia&limit=5000&full=true"
HF_README = "https://huggingface.co/{id}/raw/main/README.md"

AUDIO_PIPELINES = {
    "automatic-speech-recognition", "text-to-speech", "audio-classification",
    "audio-to-audio", "voice-activity-detection", "audio-text-to-text", "text-to-audio",
}
AUDIO_NAME = re.compile(
    r"(asr|stt_|tts|speech|audio|parakeet|canary|conformer|citrinet|titanet|diar|sortformer|"
    r"vad|marblenet|fastpitch|hifigan|codec|magpie|bigvgan|voice|speaker|flamingo|se_d|sr_ssl|"
    r"slu_|ssl_en|re-use|personaplex|omnivinci|omni-embed|audio2)", re.I)
NOT_AUDIO = re.compile(r"(cosmos|drivaerml|mask_rcnn|privasis|riva-translate|climb|rnapro|diffusion_renderer)", re.I)

# (family, regex on id) -- first match wins
FAMILIES = [
    ("Speech LLM (full-duplex)", r"voicechat|personaplex"),
    ("Speech LLM (SALM)", r"canary-qwen"),
    ("Audio LLM (Flamingo)", r"flamingo|omnivinci"),
    ("Multimodal embedding", r"omni-embed"),
    ("Canary AED (ASR+AST)", r"canary"),
    ("Diarization (Sortformer)", r"sortformer|diarization"),
    ("Multitalker ASR", r"multitalker"),
    ("Parakeet / Nemotron ASR", r"parakeet|nemotron.*(asr|speech)"),
    ("FastConformer ASR (stt_*)", r"stt_.*fastconformer"),
    ("Conformer ASR (stt_*)", r"stt_.*conformer"),
    ("Citrinet ASR", r"citrinet"),
    ("SSL encoder (NEST)", r"ssl_en_nest"),
    ("Spoken language understanding", r"slu_"),
    ("Speaker embedding (TitaNet)", r"titanet|speakerverification"),
    ("Voice activity detection", r"vad|marblenet"),
    ("Neural audio codec (FSQ)", r"codec"),
    ("Vocoder (GAN)", r"bigvgan|hifigan"),
    ("Text-to-speech", r"tts|fastpitch|magpie"),
    ("Speech enhancement / restoration", r"re-use|se_d|sr_ssl|schrodinger|bandwidth"),
    ("Audio-driven animation", r"audio2"),
]

ENCODERS = [
    ("FastConformer", r"fastconformer|parakeet|canary|nemotron.*(asr|speech|diar)|sortformer|nest|multitalker|voicechat|slu_"),
    ("Conformer", r"conformer"),
    ("Citrinet (1D conv)", r"citrinet"),
    ("MarbleNet (1D conv)", r"marblenet"),
    ("TitaNet (1D conv + SE)", r"titanet"),
    ("AF-Whisper", r"flamingo"),
    ("Conv encoder + FSQ", r"codec"),
    ("Mamba (bi-dir SSM)", r"re-use"),
    ("Causal Transformer (text -> codec tokens)", r"magpie"),
    ("FastPitch (Transformer, mel)", r"fastpitch"),
    ("U-Net (Schrodinger bridge)", r"se_d|schrodinger"),
    ("Transformer (flow matching)", r"sr_ssl"),
    ("Audio feature net", r"audio2|bandwidth"),
    ("LLM embedder", r"omni-embed"),
    ("None (mel input)", r"bigvgan|hifigan"),
    ("Mimi codec + Transformer (Moshi)", r"personaplex"),
]

DECODERS = [
    ("TDT + CTC (hybrid)", r"tdt_ctc"),
    ("TDT", r"\btdt|[-_]tdt"),
    ("RNNT + CTC (hybrid)", r"hybrid"),
    ("RNNT", r"rnnt|transducer|nemotron.*(asr|speech)|realtime_eou|multitalker|parakeet-unified"),
    ("CTC", r"ctc|citrinet"),
    ("Transformer AED", r"canary(?!-qwen)|slu_"),
    ("LLM (decoder-only)", r"canary-qwen|flamingo|voicechat|personaplex|omnivinci"),
    ("None (SSL: masked token prediction)", r"ssl_en_nest"),
    ("Sigmoid per-speaker (sort loss)", r"sortformer|diarization"),
    ("Per-frame sigmoid", r"vad"),
    ("Pooling + AAM-softmax", r"titanet|speakerverification"),
    ("HiFi-GAN upsampler", r"codec|hifigan|bigvgan"),
    ("Codec tokens + local transformer", r"magpie"),
    ("Mel spectrogram", r"fastpitch"),
    ("Generative (waveform/spectrogram)", r"se_d|schrodinger|sr_ssl|re-use"),
    ("Blendshapes / emotion", r"audio2"),
    ("Classifier / embedding", r"bandwidth|omni-embed"),
]


@dataclass
class ModelEntry:
    id: str
    family: str
    encoder: str
    decoder: str
    task: str | None
    library: str | None
    downloads: int
    likes: int
    license: str
    created: str
    params: str = ""
    sample_rate: str = ""
    streaming: bool = False
    languages: list[str] = field(default_factory=list)
    summary: str = ""

    def text(self) -> str:
        return " ".join(str(v) for v in asdict(self).values()).lower()


def _first(rules, name, default="other"):
    for label, rx in rules:
        if re.search(rx, name, re.I):
            return label
    return default


def is_audio(model: dict) -> bool:
    mid = model["id"]
    if NOT_AUDIO.search(mid):
        return False
    return model.get("pipeline_tag") in AUDIO_PIPELINES or bool(AUDIO_NAME.search(mid))


def _card_facts(card: str) -> dict:
    facts = {}
    m = re.search(r"Params-([0-9.]+[KMB])", card) or re.search(
        r"(\d[\d.,]*\s?(?:[KMB]|million|billion))[ -]param", card, re.I)
    if m:
        facts["params"] = m.group(1).replace(" ", "")
    m = re.search(r"(16|22\.05|22|24|44\.1|44|48)\s?k[hH]z", card)
    if m:
        facts["sample_rate"] = m.group(1) + "kHz"
    facts["streaming"] = bool(re.search(r"cache[- ]aware|streaming", card, re.I))
    body = re.sub(r"^---.*?---", "", card, flags=re.S)
    for line in body.splitlines():
        s = re.sub(r"<[^>]+>|\[!\[.*?\)\]\(.*?\)|[*`#|]", "", line).strip()
        if len(s) > 80 and not s.startswith(("http", "[", "!")):
            facts["summary"] = s[:300]
            break
    return facts


def classify(model: dict, card: str = "") -> ModelEntry:
    mid = model["id"]
    tags = model.get("tags", [])
    lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "")
    langs = [t for t in tags if re.fullmatch(r"[a-z]{2,3}", t)]
    facts = _card_facts(card) if card else {}
    return ModelEntry(
        id=mid,
        family=_first(FAMILIES, mid),
        encoder=_first(ENCODERS, mid, "n/a"),
        decoder=_first(DECODERS, mid, "n/a"),
        task=model.get("pipeline_tag"),
        library=model.get("library_name"),
        downloads=model.get("downloads", 0) or 0,
        likes=model.get("likes", 0) or 0,
        license=lic,
        created=(model.get("createdAt") or "")[:10],
        languages=langs,
        **facts,
    )


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def fetch(out_dir: str | Path, with_cards: bool = True) -> list[ModelEntry]:
    """Pull the live model list from HF, keep audio models, save raw + catalog."""
    out = Path(out_dir)
    (out / "cards").mkdir(parents=True, exist_ok=True)
    models = [m for m in json.loads(_get(HF_API)) if is_audio(m)]
    (out / "audio_models.json").write_text(json.dumps(models, indent=1))

    def card(m):
        p = out / "cards" / (m["id"].split("/")[1] + ".md")
        if with_cards and not p.exists():
            try:
                p.write_bytes(_get(HF_README.format(id=m["id"])))
            except Exception:
                return ""  # gated or missing README
        return p.read_text(errors="ignore") if p.exists() else ""

    with ThreadPoolExecutor(16) as ex:
        cards = list(ex.map(card, models))
    entries = [classify(m, c) for m, c in zip(models, cards)]
    save(entries, out / "catalog.json")
    return entries


def build_from_raw(raw_dir: str | Path) -> list[ModelEntry]:
    raw = Path(raw_dir)
    models = [m for m in json.loads((raw / "audio_models.json").read_text()) if is_audio(m)]
    out = []
    for m in models:
        p = raw / "cards" / (m["id"].split("/")[1] + ".md")
        out.append(classify(m, p.read_text(errors="ignore") if p.exists() else ""))
    return out


def save(entries: list[ModelEntry], path: str | Path):
    Path(path).write_text(json.dumps([asdict(e) for e in entries], indent=1))


def load(path: str | Path) -> list[ModelEntry]:
    return [ModelEntry(**d) for d in json.loads(Path(path).read_text())]


def search(entries, query: str = "", family: str = "", encoder: str = "", decoder: str = "",
           streaming: bool | None = None, sort: str = "downloads") -> list[ModelEntry]:
    terms = query.lower().split()
    res = [
        e for e in entries
        if all(t in e.text() for t in terms)
        and family.lower() in e.family.lower()
        and encoder.lower() in e.encoder.lower()
        and decoder.lower() in e.decoder.lower()
        and (streaming is None or e.streaming == streaming)
    ]
    return sorted(res, key=lambda e: getattr(e, sort), reverse=sort in ("downloads", "likes", "created"))


def summarize(entries) -> dict:
    from collections import Counter
    return {
        "total": len(entries),
        "by_family": Counter(e.family for e in entries).most_common(),
        "by_encoder": Counter(e.encoder for e in entries).most_common(),
        "by_decoder": Counter(e.decoder for e in entries).most_common(),
        "streaming": sum(e.streaming for e in entries),
        "by_license": Counter(e.license for e in entries).most_common(),
        "total_downloads": sum(e.downloads for e in entries),
    }
