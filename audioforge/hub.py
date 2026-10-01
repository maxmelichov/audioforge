"""Fetch, verify and convert the checkpoints the server needs (``audioforge-download``).

    audioforge-download                         # single-model mode (the default): served ASR + TS-VAD head (~0.46 GB)
    audioforge-download --diarizer nemotron3    # + NVIDIA Nemotron-3-Diarization for room mode (OpenMDW-1.1)
    audioforge-download --diarizer sortformer   # + NVIDIA Streaming Sortformer v2 for room mode (CC-BY-4.0)
    audioforge-download --with tdt_v3 titanet ambernet silero
    audioforge-download --from-local data/nemo  # reuse .nemo files you already have (sha256-checked, not copied)
    audioforge-download --list                  # what exists, sizes, licences

Every component is pinned to a Hugging Face revision (or a fixed URL) and a sha256; a file whose hash does not match
is rejected. Model weights are NVIDIA's (or Silero's) and keep their own licences: the script shows each licence and
asks you to accept it before downloading (``--yes`` or ``AUDIOFORGE_ACCEPT_LICENSES=1`` to accept non-interactively).

The served ASR model is NVIDIA's ``stt_en_fastconformer_hybrid_large_streaming_multi`` (every one of its 654 tensors
unchanged) plus this project's trained heads (VAD, EOU, turn, speaker, diar, and the turn head v5 segment classifier
``turn_seg``; 190 tensors, 29.4 MB), shipped as ``assets/served_heads_v0.3.pt``. ``build_served`` merges the two and
checks the result's tensor hash against the hash recorded when the heads were exported from
``runs/stage1_served_v3.afm`` (= ``stage1_served_v2.afm`` + heads.turn_seg, research/TURN_V5.md; the VAD head reads
block 4 only, research/VAD_SINGLE.md), so the rebuilt model is bit-identical to the shipped one.
``--heads-version 0.2`` rebuilds ``stage1_served_v2.afm`` (no v5 classifier: ``--turn-preset fast`` / ``assistant``
need v0.3) and ``--heads-version 0.1`` ``stage1_served.afm``, the 2026-09-27 checkpoint most numbers in research/ were
measured with.

Second core (``--core 0.6b``, research/CORE_0P6B.md): NVIDIA's ``nemotron-speech-streaming-en-0.6b`` (NVIDIA Open
Model License; 618 M parameters, every tensor unchanged) plus heads retrained on it (``assets/served_heads_0p6b_v0.1.pt``
-> ``served_0p6b_v0.1.afm``) and its own TS-VAD and LID heads (``tsvad_0p6b.pt``, ``lid_0p6b.pt``). The 115M stays the
default. ``audioforge-download --core 0.6b`` fetches that set (2.5 GB download, ~2.3 GB on disk).

Output directory: ``--dir``, else ``$AUDIOFORGE_HOME``, else ``<repo>/models`` in a source checkout, else
``~/.cache/audioforge``. ``audioforge-serve`` looks there.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import DATA_ROOT, ROOT

__all__ = [
    "accept_licenses", "build_served", "CC_BY", "Component", "diarizer_defaults", "DIARIZERS", "export_heads",
    "fetch", "find_model", "HEADS_FILE", "heads_path", "HEADS_URL", "install", "list_components", "main",
    "models_dir", "OPTIONAL", "RELEASE_URL", "SERVED", "SINGLE", "sha256_file", "state_hash", "HEADS", "HEADS_VERSION",
    "CORES", "CORE_DEFAULT", "HEADS_0P6B", "SERVED_0P6B", "core_keys",
]

# the heads asset: v0.3 ships (v0.2 + heads.turn_seg, the turn head v5 segment classifier of --turn-preset fast /
# assistant, research/TURN_V5.md; every v0.2 tensor unchanged); v0.2 (block-4 VAD head) and v0.1 stay for reproducing
# the research measurements
HEADS = {"0.3": ("served_heads_v0.3.pt", 29481643, "ea1e8331fa9b9efdee76f1b44d4352f9e1660f34e3da6491b5655ce56ab18848",
                 "stage1_served_v3.afm"),
         "0.2": ("served_heads_v0.2.pt", 19629563, "cb5aa06974f27576c0f66b9453868701106969dea5d5ad100b06b2b779f121d2",
                 "stage1_served_v2.afm"),
         "0.1": ("served_heads_v0.1.pt", 19629955, "834f3e94467bc4110555d8d4cbdbe0ce75254a80d8ca203ecd975d4f007286f5",
                 "stage1_served.afm")}
HEADS_VERSION = "0.3"
HEADS_FILE = HEADS[HEADS_VERSION][0]
RELEASE_URL = "https://github.com/maxmelichov/audioforge/releases/download/v0.1.0"
HEADS_URL = f"{RELEASE_URL}/{HEADS_FILE}"
SERVED = HEADS[HEADS_VERSION][3]  # what ships: stage1_served_v3.afm (= v2 + heads.turn_seg; runs/stage1_served.afm = v1)

# the second core (--core 0.6b): nemotron-speech-streaming-en-0.6b + heads retrained on it (research/CORE_0P6B.md)
HEADS_0P6B = {"0.1": ("served_heads_0p6b_v0.1.pt", 16045143,
                       "664be5a0e498b9268d088ccf2fa079d909cd70311c4096f8d28f94258adc4e6e", "served_0p6b_v0.1.afm")}
HEADS_0P6B_VERSION = "0.1"
SERVED_0P6B = HEADS_0P6B[HEADS_0P6B_VERSION][3]
# what each core needs: (served ASR + heads, TS-VAD head, LID head) component keys
CORES = {"115m": ("asr", "tsvad", "lid"), "0.6b": ("asr_0p6b", "tsvad_0p6b", "lid_0p6b")}
CORE_DEFAULT = "115m"


def core_keys(core: str) -> tuple[str, str, str]:
    """(asr, tsvad, lid) component keys of ``core`` (115m | 0.6b)."""
    if core not in CORES:
        raise ValueError(f"unknown core {core!r}: one of {', '.join(CORES)}")
    return CORES[core]


@dataclass(frozen=True)
class Component:
    key: str
    title: str
    filename: str          # the downloaded file
    size: int              # bytes of the downloaded file
    sha256: str
    license: str
    license_url: str
    used_for: str
    output: str            # what ends up in the models directory
    output_mb: int         # approximate size of ``output``
    convert: str           # served | afm | keep (``keep``: used in place; the small head files ship in assets/)
    repo: str | None = None
    revision: str | None = None
    url: str | None = None

    @property
    def source(self) -> str:
        return f"https://huggingface.co/{self.repo}" if self.repo else str(self.url)


CC_BY = "https://creativecommons.org/licenses/by/4.0/"
COMPONENTS: dict[str, Component] = {c.key: c for c in [
    Component("asr", "NVIDIA FastConformer hybrid streaming (114M) + audioforge heads",
              "stt_en_fastconformer_hybrid_large_streaming_multi.nemo", 459673600,
              "8db5289d5238aca839b84f5afdd66eb1aa64413feb9c2a915940b291012aff8b", "CC-BY-4.0", CC_BY,
              "streaming ASR, VAD, turn and speaker heads (required)", SERVED, 443, "served",
              repo="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
              revision="ae98143333690bd7ced4bc8ec16769bcb8918374"),
    Component("asr_0p6b", "NVIDIA Nemotron Speech Streaming EN 0.6B + audioforge heads (--core 0.6b)",
              "nemotron-speech-streaming-en-0.6b.nemo", 2473041920,
              "283638054c44f6794e74fe9af9048d78a6d9d6c058c12131856c7859a62ac9cd", "NVIDIA Open Model License",
              "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
              "second core (serve --core 0.6b): streaming ASR, VAD, turn and speaker heads; one real-time stream on "
              "2 CPU threads, 3 on an Apple GPU", SERVED_0P6B, 2300, "served",
              repo="nvidia/nemotron-speech-streaming-en-0.6b", revision="ebe59e5a817142986528bbbee5dba8db7b38ed50"),
    Component("sortformer", "NVIDIA Streaming Sortformer 4spk v2 (117M)", "diar_streaming_sortformer_4spk-v2.nemo",
              471367680, "b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329", "CC-BY-4.0", CC_BY,
              "speaker activity / diarizer (default --diar)", "nemo_sortformer_v2.afm", 436, "afm",
              repo="nvidia/diar_streaming_sortformer_4spk-v2", revision="84edd514b8ef68004c10086918cd62f2148cbd59"),
    Component("nemotron3", "NVIDIA Nemotron-3-Diarization", "Nemotron-3-Diarization.nemo", 198676480,
              "867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d", "OpenMDW-1.1",
              "https://huggingface.co/nvidia/Nemotron-3-Diarization", "alternative diarizer (--diarizer nemotron3)",
              "nemo_nemotron3_diar.afm", 177, "afm",
              repo="nvidia/Nemotron-3-Diarization", revision="f667ed73aee57d40cc39428eb768b4fd87a0a29e"),
    Component("tdt_v3", "NVIDIA Parakeet-TDT 0.6B v3", "parakeet-tdt-0.6b-v3.nemo", 2509332480,
              "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d", "CC-BY-4.0", CC_BY,
              "offline per-turn final ASR (serve --final-asr tdt_v3)", "nemo/parakeet-tdt-0.6b-v3.nemo", 2393,
              "keep", repo="nvidia/parakeet-tdt-0.6b-v3", revision="541d1f99c6b0c3cd0b11a95167540bb8edefd82b"),
    Component("titanet", "NVIDIA TitaNet-Large speaker embeddings (23M)", "speakerverification_en_titanet_large.nemo",
              101621760, "e838520693f269e7984f55bc8eb3c2d60ccf246bf4b896d4be9bcabe3e4b0fe3", "CC-BY-4.0", CC_BY,
              "voice enrollment (serve --enroll after_agent / explicit)",
              "nemo/speakerverification_en_titanet_large.nemo", 97, "keep",
              repo="nvidia/speakerverification_en_titanet_large", revision="0dc382f40121a5fbd34db10a2bb04d826c2be6a8"),
    Component("ambernet", "NVIDIA AmberNet spoken language ID (NGC)", "langid_ambernet.nemo", 116049920,
              "2f92d645b9ea5824d7663584fecb9ecc52557d0d700e24266747f38a61ba1681", "NGC Terms of Use",
              "https://ngc.nvidia.com/legal/terms",
              "language ID (serve --lid ambernet)", "nemo/langid_ambernet.nemo", 111, "keep",
              url="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/langid_ambernet/versions/1.12.0/files/ambernet.nemo"),
    # this project's small heads for single-model mode (research/SINGLE_MODEL.md): shipped in assets/ and attached to
    # the v0.1.0 release (docs/RELEASE_CHECKLIST.md), like served_heads_v0.1.pt
    Component("tsvad", "audioforge TS-VAD head (target speaker, 0.26 M)", "tsvad_spk.pt", 1048542,
              "dbc6230d8d722bad65aaf598dce69569995bd96bc002da40a2069d664c427683", "Apache-2.0 (audioforge heads)",
              "https://www.apache.org/licenses/LICENSE-2.0", "single-model mode: the user's track (serve --tsvad)",
              "tsvad_spk.pt", 1, "keep", url=f"{RELEASE_URL}/tsvad_spk.pt"),
    Component("lid", "audioforge LID head (distilled from AmberNet, 0.92 M)", "lid_distill.pt", 3704886,
              "07de4e4da5444ecae250b4c5ef0172372451ba2da9ea7ae492e42b1759f1d487",
              "Apache-2.0 (audioforge heads; trained on AmberNet outputs, NGC Terms of Use)",
              "https://ngc.nvidia.com/legal/terms", "single-model mode: language ID (serve --lid head)",
              "lid_distill.pt", 4, "keep", url=f"{RELEASE_URL}/lid_distill.pt"),
    # the 0.6B core's own TS-VAD and LID heads (research/CORE_0P6B.md; a 115M head does not fit the 1024-d encoder)
    Component("tsvad_0p6b", "audioforge TS-VAD head for the 0.6B core", "tsvad_0p6b.pt", 1441712,
              "06fc6e3a421ac344216a593bcae0922099435aef8b8834e62fa401eaafa8a9d8",
              "Apache-2.0 (audioforge heads)", "https://www.apache.org/licenses/LICENSE-2.0",
              "--core 0.6b: the user's track", "tsvad_0p6b.pt", 2, "keep", url=f"{RELEASE_URL}/tsvad_0p6b.pt"),
    Component("lid_0p6b", "audioforge LID head for the 0.6B core (distilled from AmberNet)", "lid_0p6b.pt", 4757196,
              "22c8667afbb5be64cbd3fe41b7b941f9a0b07b3453139da062524af7d0307fed",
              "Apache-2.0 (audioforge heads; trained on AmberNet outputs, NGC Terms of Use)",
              "https://ngc.nvidia.com/legal/terms", "--core 0.6b: language ID", "lid_0p6b.pt", 5, "keep",
              url=f"{RELEASE_URL}/lid_0p6b.pt"),
    Component("silero", "Silero VAD v5 (v5.1.2 ONNX)", "silero_vad.onnx", 2327524,
              "2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f", "MIT",
              "https://github.com/snakers4/silero-vad/blob/master/LICENSE",
              "turn policies hybrid_silero / hybrid_dyn (not single mode's default)", "silero_vad_v5.onnx", 2, "keep",
              url="https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"),
]}
DIARIZERS = {"sortformer": "sortformer", "nemotron3": "nemotron3"}
OPTIONAL = ("tdt_v3", "titanet", "ambernet", "lid", "silero")
# what single-model mode (the default) needs; + "lid" for language ID when available. No Silero: single mode's turn rule
# (vad_head, research/EOT_LATENCY.md) reads the model's own heads; --with silero for hybrid_silero / hybrid_dyn
SINGLE = ("asr", "tsvad")


def diarizer_defaults(diarizer: str) -> dict[str, Any]:
    """Engine options the product applies for a downloaded diarizer (docs/CONFIGURATION.md section 7.3 / 7.4):
    Nemotron-3 with max pooling and a frame-local encoder, and hold-on-shed for either diarizer."""
    extra: dict[str, Any] = {"diar_pool": "max", "diar_left": 1} if diarizer == "nemotron3" else {}
    return {**extra, "shed_diar": "hold"}


# --------------------------------------------------------------------------- locations
def models_dir(explicit: str | Path | None = None) -> Path:
    """--dir, else $AUDIOFORGE_HOME, else <repo>/models in a source checkout, else ~/.cache/audioforge."""
    if explicit:
        return Path(explicit).expanduser()
    if os.environ.get("AUDIOFORGE_HOME"):
        return Path(os.environ["AUDIOFORGE_HOME"]).expanduser()
    if (ROOT / "pyproject.toml").exists() and (ROOT / "audioforge").is_dir():
        return ROOT / "models"
    return Path.home() / ".cache" / "audioforge"


def find_model(key: str, directory: str | Path | None = None) -> Path | None:
    """Where a component's output is, searching the models dir, then this checkout's legacy runs/ and data/ paths."""
    c = COMPONENTS[key]
    cands = [models_dir(directory) / c.output]
    legacy = {"asr": ROOT / "runs" / SERVED, "asr_0p6b": ROOT / "runs" / SERVED_0P6B,
              "tsvad_0p6b": ROOT / "assets" / c.output, "lid_0p6b": ROOT / "assets" / c.output,
              "sortformer": ROOT / "runs" / c.output,
              "nemotron3": ROOT / "runs" / c.output, "silero": DATA_ROOT / "silero" / c.output,
              "tsvad": ROOT / "assets" / c.output, "lid": ROOT / "assets" / c.output}
    cands.append(legacy.get(key, DATA_ROOT / "nemo" / c.filename))
    return next((p for p in cands if p.exists()), None)


# --------------------------------------------------------------------------- hashing
def sha256_file(path: str | Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def state_hash(state_dict) -> str:
    """sha256 over (key, dtype, shape, bytes) of every tensor in key order: equal iff the weights are identical."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- heads delta
def export_heads(served: str | Path, base_nemo: str | Path, out: str | Path, base_key: str = "asr") -> dict:
    """Write the served model's non-NVIDIA tensors + full config + the merged model's tensor hash (maintainers)."""
    import torch

    from .nemo_import import import_nemo
    from .train import load_model
    s, b = load_model(served, "cpu"), import_nemo(base_nemo)
    ss, bs = s.state_dict(), b.state_dict()
    changed = [k for k in bs if k in ss and not torch.equal(bs[k], ss[k])]
    if changed or set(bs) - set(ss):
        raise ValueError(f"served model is not the base + new tensors: changed {changed[:3]}")
    tensors = {k: v.clone() for k, v in ss.items() if k not in bs}
    blob = {"format": "audioforge-heads/1", "config": s.cfg, "tensors": tensors,
            "base": {"repo": COMPONENTS[base_key].repo, "revision": COMPONENTS[base_key].revision,
                     "sha256": COMPONENTS[base_key].sha256},
            "state_hash": state_hash(ss), "source": str(Path(served).name)}
    torch.save(blob, out)
    return {"tensors": len(tensors), "mb": round(sum(v.numel() * v.element_size() for v in tensors.values()) / 1e6, 1),
            "state_hash": blob["state_hash"]}


def heads_path(explicit: str | Path | None, directory: Path, version: str = HEADS_VERSION,
               registry: dict | None = None) -> Path:
    """The heads file of ``version`` (in ``registry``: HEADS, or HEADS_0P6B for the 0.6B core): explicit path, the
    repo's assets/, the models dir, else downloaded from the release; a registry file (not an explicit path) must match
    its pinned size and sha256."""
    name, size, sha, _ = (registry or HEADS)[version]
    if explicit:
        return Path(explicit)
    for p in (ROOT / "assets" / name, directory / name):
        if p.exists():
            if p.stat().st_size != size or sha256_file(p) != sha:
                raise RuntimeError(f"{p}: size / sha256 do not match heads v{version} ({sha[:12]})")
            return p
    dst = directory / name
    print(f"  fetching heads from {RELEASE_URL}/{name}")
    try:
        _http_get(f"{RELEASE_URL}/{name}", dst)
    except OSError as e:
        raise RuntimeError(f"{name}: could not download {RELEASE_URL}/{name} ({e}); it ships in the repository's "
                           f"assets/") from e
    if sha256_file(dst) != sha:
        dst.unlink()
        raise RuntimeError(f"{name}: sha256 != pinned {sha}; deleted")
    return dst


def build_served(base_nemo: str | Path, heads: str | Path, out: str | Path) -> str:
    """NVIDIA hybrid streaming .nemo + heads file -> the served .afm; asserts the tensor hash; returns it."""
    import torch

    from .model import SpeechModel
    from .nemo_import import import_nemo
    from .train import save_model
    blob = torch.load(heads, map_location="cpu", weights_only=True)
    if blob.get("format") != "audioforge-heads/1":
        raise ValueError(f"{heads}: not an audioforge heads file")
    base = import_nemo(base_nemo)
    sd = base.state_dict()
    sd.update(blob["tensors"])
    model = SpeechModel(blob["config"], base.tokenizer)
    model.load_state_dict(sd, strict=True)
    got = state_hash(model.state_dict())
    if got != blob["state_hash"]:
        raise RuntimeError(f"rebuilt model hash {got[:12]} != recorded {blob['state_hash'][:12]}")
    _save_atomic(save_model, model.eval(), out)
    return got


def _save_atomic(save, model, out: Path) -> None:
    """Write through ``<out>.part`` and rename, so an interrupted conversion never leaves a file that ``install``
    would later take for a finished model."""
    import tarfile
    tmp = Path(out).with_name(Path(out).name + ".part")
    save(model, tmp)
    try:  # read the whole gzip stream back: a corrupt write (seen once on an exFAT volume) fails here, not at serve time
        with tarfile.open(tmp, "r:gz") as tar:
            for m in tar:
                if m.isfile():
                    with tar.extractfile(m) as f:
                        while f.read(1 << 22):
                            pass
    except (tarfile.TarError, OSError, EOFError) as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{out}: the written archive does not read back ({e}); nothing kept, re-run") from e
    tmp.replace(out)


# --------------------------------------------------------------------------- download
def _http_get(url: str, dst: Path):
    import urllib.request
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "audioforge-download"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if sys.stderr.isatty() and total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr)
        if sys.stderr.isatty() and total:
            print(file=sys.stderr)
    tmp.replace(dst)


def fetch(c: Component, directory: Path, local: list[Path]) -> tuple[Path, bool]:
    """(path of the verified source file, whether we downloaded it). Local copies are used in place."""
    for p in (d / name for d in local for name in dict.fromkeys([c.filename, Path(c.output).name])):
        if p.exists():
            print(f"  {c.filename}: using local copy {p}, checking sha256 ...")
            if sha256_file(p) != c.sha256:
                raise RuntimeError(f"{p}: sha256 mismatch (expected {c.sha256})")
            return p, False
    dst = directory / "nemo" / c.filename if c.filename.endswith(".nemo") else directory / c.output
    if dst.exists() and dst.stat().st_size == c.size and sha256_file(dst) == c.sha256:
        return dst, False
    print(f"  downloading {c.filename} ({c.size / 1e6:.0f} MB) from {c.source}")
    if c.repo:
        from huggingface_hub import hf_hub_download
        p = Path(hf_hub_download(c.repo, c.filename, revision=c.revision, local_dir=str(dst.parent)))
        if p != dst:
            shutil.move(str(p), dst)
    else:
        try:
            _http_get(str(c.url), dst)
        except OSError as e:  # urllib errors are OSErrors: one clear line instead of a traceback
            raise RuntimeError(f"{c.filename}: could not download {c.url} ({e}). It ships in the repository's assets/; "
                               f"from a checkout run audioforge-download there, or pass --from-local DIR") from e
    got = sha256_file(dst)
    if got != c.sha256:
        dst.unlink()
        raise RuntimeError(f"{c.filename}: sha256 {got} != pinned {c.sha256}; deleted")
    return dst, True


def accept_licenses(comps: list[Component], yes: bool) -> None:
    print("\nModel weights are not part of audioforge's code licence. You are about to use:")
    for c in comps:
        print(f"  - {c.title}\n      licence: {c.license} ({c.license_url})\n      source:  {c.source}")
    if "CC-BY-4.0" in {c.license for c in comps}:
        print("  CC-BY-4.0 requires attribution to NVIDIA when you redistribute the weights or outputs built on them.")
    if yes or os.environ.get("AUDIOFORGE_ACCEPT_LICENSES") == "1":
        print("Licences accepted (--yes / AUDIOFORGE_ACCEPT_LICENSES=1).\n")
        return
    if not sys.stdin.isatty():
        sys.exit("Not a terminal: re-run with --yes (or AUDIOFORGE_ACCEPT_LICENSES=1) to accept the licences above.")
    if input("Accept these licences? [y/N] ").strip().lower() not in ("y", "yes"):
        sys.exit("Licences not accepted; nothing downloaded.")


def install(keys: list[str], directory: Path, local: list[Path], yes: bool, keep_nemo: bool,
            heads: str | None = None, force: bool = False, heads_version: str = HEADS_VERSION) -> dict[str, Path]:
    import dataclasses
    comps = [COMPONENTS[k] if not (k == "asr" and heads_version != HEADS_VERSION)
             else dataclasses.replace(COMPONENTS[k], output=HEADS[heads_version][3]) for k in keys]
    missing = [c.key for c in comps if not c.sha256]
    if missing:
        raise RuntimeError(f"{', '.join(missing)}: no pinned file yet in this version of audioforge "
                           f"(research/CORE_0P6B.md); nothing downloaded")
    todo = [c for c in comps if force or not (directory / c.output).exists()]
    for c in comps:
        if c not in todo:
            print(f"[{c.key}] already present: {directory / c.output}")
    if todo:
        accept_licenses(todo, yes)
    directory.mkdir(parents=True, exist_ok=True)
    for c in todo:
        t0 = time.time()
        print(f"[{c.key}] {c.title}")
        src, downloaded = fetch(c, directory, local)
        out = directory / c.output
        out.parent.mkdir(parents=True, exist_ok=True)
        if c.convert == "served" and c.key == "asr_0p6b":
            h = build_served(src, heads_path(None, directory, HEADS_0P6B_VERSION, HEADS_0P6B), out)
            print(f"  built {out} (tensor hash {h[:16]}, identical to the measured model)")
        elif c.convert == "served":
            h = build_served(src, heads_path(heads, directory, heads_version), out)
            print(f"  built {out} (tensor hash {h[:16]}, identical to the measured model)")
        elif c.convert == "afm":
            from .nemo_import import import_nemo
            from .train import save_model
            _save_atomic(save_model, import_nemo(src), out)
            print(f"  converted -> {out}")
        elif src.resolve() != out.resolve():
            if out.exists() or out.is_symlink():
                out.unlink()
            out.symlink_to(src.resolve())
            print(f"  linked {out} -> {src}")
        if c.convert in ("served", "afm") and downloaded and not keep_nemo:
            src.unlink()
        print(f"  done in {time.time() - t0:.0f}s")
    return {c.key: directory / c.output for c in comps}


def list_components():
    print(f"{'key':<11} {'download MB':>11} {'on disk MB':>10}  {'licence':<18} used for")
    for c in COMPONENTS.values():
        print(f"{c.key:<11} {c.size / 1e6:>11.0f} {c.output_mb:>10}  {c.license:<18} {c.used_for}")
    print("\nDefault set (single-model mode): asr + tsvad + lid; --core 0.6b: asr_0p6b + tsvad_0p6b + lid_0p6b instead; "
          "--diarizer adds a diarizer for room mode. "
          "Peak disk during a default install ~0.9 GB (the .nemo files are deleted after conversion unless --keep-nemo).")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="audioforge-download", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="models directory (default: $AUDIOFORGE_HOME, <repo>/models, "
                                                "or ~/.cache/audioforge)")
    ap.add_argument("--core", choices=list(CORES), default=CORE_DEFAULT,
                    help="which streaming core: 115m (default; NVIDIA FastConformer 114M, CC-BY-4.0, real time on 2 CPU "
                         "threads) | 0.6b (NVIDIA nemotron-speech-streaming-en-0.6b, NVIDIA Open Model License, 2.5 GB; "
                         "about half the meeting WER; 1 real-time stream per 2 CPU threads (115M: 4), research/CORE_0P6B.md)")
    ap.add_argument("--diarizer", choices=sorted(DIARIZERS), default=None,
                    help="also fetch a diarizer for room mode (audioforge-serve --mode room): nemotron3 = "
                         "Nemotron-3-Diarization (OpenMDW-1.1, the room-mode default), sortformer = Streaming Sortformer "
                         "v2 (CC-BY-4.0). Default: none (single-model mode needs no diarizer)")
    ap.add_argument("--with", dest="extra", nargs="*", default=[], choices=list(OPTIONAL) + ["all"],
                    help="optional models: tdt_v3 (2.5 GB), titanet, ambernet, silero, or all")
    ap.add_argument("--only", nargs="*", choices=list(COMPONENTS), help="exactly these components")
    ap.add_argument("--from-local", nargs="*", default=[], metavar="DIR",
                    help="directories holding already-downloaded files (sha256-checked, used in place)")
    ap.add_argument("--data-root", default=None,
                    help="also reuse files under DIR/nemo and DIR/silero (default: $AUDIOFORGE_DATA, else <repo>/data)")
    ap.add_argument("--heads", default=None, help=f"path to {HEADS_FILE} (default: assets/ or the release asset)")
    ap.add_argument("--heads-version", choices=sorted(HEADS), default=HEADS_VERSION,
                    help="0.2 (default, ships: block-4 VAD head -> stage1_served_v2.afm) | 0.1 (the 2026-09-27 "
                         "measured checkpoint -> stage1_served.afm, for reproducing research/)")
    ap.add_argument("--keep-nemo", action="store_true", help="keep downloaded .nemo files after conversion")
    ap.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    ap.add_argument("--yes", "-y", action="store_true", help="accept the model licences non-interactively")
    ap.add_argument("--list", action="store_true", help="list components and exit")
    ap.add_argument("--export-heads", nargs=2, metavar=("SERVED_AFM", "OUT"),
                    help="(maintainers) write the heads file from a served .afm; the base .nemo comes from "
                         "--from-local or is downloaded")
    a = ap.parse_args(argv)
    if a.list:
        list_components()
        return 0
    directory = models_dir(a.dir)
    local = [Path(p).expanduser() for p in a.from_local]
    data_root = Path(a.data_root).expanduser() if a.data_root else DATA_ROOT
    local += [d for d in (data_root / "nemo", data_root / "silero", ROOT / "assets") if d.is_dir() and d not in local]
    if a.export_heads:
        base_key = core_keys(a.core)[0]
        src, _ = fetch(COMPONENTS[base_key], directory, local)
        print(export_heads(a.export_heads[0], src, a.export_heads[1], base_key))
        return 0
    if a.only:
        keys = list(a.only)
    else:
        extra = list(OPTIONAL) if "all" in a.extra else a.extra
        asr_k, tsvad_k, lid_k = core_keys(a.core)
        keys = [asr_k, tsvad_k] + ([lid_k] if a.core != CORE_DEFAULT else []) \
            + ([DIARIZERS[a.diarizer]] if a.diarizer else [])
        keys += [k for k in extra if k not in keys]
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    paths = install(keys, directory, local, a.yes, a.keep_nemo, a.heads, a.force, a.heads_version)
    print("\nModels:")
    for k, p in paths.items():
        size = p.resolve().stat().st_size / 1e6 if p.exists() else 0
        print(f"  {k:<11} {p}  ({size:.0f} MB)")
    missing = [k for k in ("tsvad", "lid", "tsvad_0p6b", "lid_0p6b") if k in paths and not paths[k].exists()]
    if missing:
        sys.exit(f"audioforge-download: {', '.join(missing)} missing after install; single-model mode cannot start")
    diar = [k for k in DIARIZERS if k in paths]
    extra = f" --mode room --diarizer {diar[0]}" if diar else ""
    extra += "" if a.core == CORE_DEFAULT else f" --core {a.core}"
    print(f"\nStart the server:  audioforge-serve{extra}" + ("" if a.dir is None else f" --models-dir {directory}"))
    if not diar:
        print("  single-model mode: send the user's stored voice print ({\"type\": \"enroll\", \"embedding\": [...]}, "
              "from >= 5 s of clean speech, 10 s for meetings) - docs/CONFIGURATION.md section 13")
    return 0


if __name__ == "__main__":
    sys.exit(main())
