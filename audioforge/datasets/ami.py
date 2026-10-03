"""AMI Meeting Corpus (CC-BY-4.0) as audioforge list-of-dict data: diarization and speaker-aware
end-of-turn on real conversation.

Source: headset-mix audio (``<meeting>.Mix-Headset.wav``, 16 kHz mono) + AMI manual annotations v1.6.2
(``words/<meeting>.<agent>.words.xml``: per-speaker word timings from forced alignment of the manual
transcript). Splits: the Full-corpus-ASR partition as distributed by the BUT/pyannote AMI diarization
setup (``lists/{train,dev,test}.meetings.txt``; we call ``test`` "eval").

Labels are built from WORD TIMINGS ONLY (no energy VAD, no vocal sounds - as the BUT "only_words"
diarization reference). Times are in seconds; an interval [s, e) covers 80 ms frames
``int(s / 0.08) .. ceil(e / 0.08) - 1`` (``data.rttm_to_frames`` convention), and every example has
``T == ToneLanguage.n_frames(len(audio))``.

  activity      union of a speaker's word intervals; words closer than ``act_bridge`` (0 s = only
                touching/overlapping words, the BUT convention) are merged.
  run           maximal sequence of one speaker's words with inter-word gaps < ``turn_gap`` (0.5 s).
  backchannel   a run of <= ``bc_max_words`` (2) words lasting <= ``bc_max_sec`` (1.0 s) ("yeah", "mm-hmm").
  turn          ``turn_def="gap"``: a run (the literal definition: gap >= turn_gap ends the turn).
                ``turn_def="floor"`` (default): consecutive runs of the same speaker are one turn unless
                the pause between them is >= ``max_hold`` (2.0 s) or another speaker produces a
                non-backchannel run inside the pause (someone else took the floor). With "gap" every
                pause >= 0.5 s is labelled a turn end even when the same speaker simply continues -
                exactly the error a silence timeout makes - so it cannot be used to measure it.
  hesitation    a silence between consecutive words of one turn lasting >= ``hes_gap`` (0.3 s).
  switch gap    for a turn followed by another speaker's (non-backchannel) turn: next start - turn end
                (negative = overlap). This is the true end-of-turn gap.

Modes (``AMI.examples(mode)``):
  diar  fixed windows (``window_sec`` 20 s, ``hop_sec`` 10 s): audio, spk_targets (T,4) arrival order.
  turn  one example per non-backchannel turn: the window runs from max(turn start - ``lead_sec``,
        previous word of the same speaker) to min(turn end + ``trail_sec``, next word of the same
        speaker), capped at ``window_sec``, so the primary speaks exactly one turn in it. This is what
        the turn head's label needs (``heads.turn.eot_targets``: EOT = every frame after the last active
        frame of ``spk_act``). Keys as ``conversation.conversation``: spk_targets (column 0 = primary,
        then arrival order), spk_act = primary_act, eot, turn_end_frame, onset_frame, text, speaker
        (global id), plus ``hes`` (T,) = 1 on the primary's within-turn hesitation frames.
  asr   single-speaker stretches (runs with no other speaker's word within 0.2 s, split at word
        boundaries to <= ``asr_max_sec``): audio, text, speaker, vad (word activity), eou.

Audio of each meeting is decoded once to ``<root>/cache/<meeting>.f32.npy`` and memory-mapped;
examples hold views into it (no copies), like ``librispeech.py``.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..data import ToneLanguage, synthetic_dataset
from ..metrics import pct_dict
from ..paths import DATA_ROOT

log = logging.getLogger(__name__)

SR = 16000
FRAME_SEC = 0.08
FRAME = 1280
DEFAULT_ROOT = DATA_ROOT / "ami"
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
ANNOT_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ANNOT_MD5 = "b9db091820ca3bd2d8f41a774e43d475"  # md5 of ami_public_manual_1.6.2.zip as served 2026-09-25
SPLIT_URL = "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/lists/{s}.meetings.txt"
SPLIT_FILES = {"train": "train", "dev": "dev", "eval": "test"}  # our name -> list file name
# Default subset (~12 h): one meeting per scenario series / site, mid-series meetings (not the short
# kick-off "a" sessions), all 4-speaker. Larger subsets append the remaining meetings in list order.
DEFAULT_MEETINGS = {
    "train": ["IS1000b", "IS1003c", "IS1006b", "ES2003b", "ES2007c", "ES2012b", "ES2015c", "TS3006b",
              "TS3009c", "TS3011b", "IN1002", "EN2003a"],
    "dev": ["IS1008b", "ES2011b", "TS3004b", "IB4002"],
    "eval": ["IS1009b", "ES2004b", "TS3003b", "EN2002a"],
}
# md5 / bytes of the Mix-Headset wavs, pinned from the first download (the AMI site publishes none).
WAV_MD5: dict[str, tuple[str, int]] = {
    "EN2002a": ("0795d1f0226fa00624025f67baa0a733", 68566744),
    "EN2003a": ("a91763602aa6b2d0474b275cc23d0e62", 71688920),
    "ES2003b": ("b0d60fc801883900a26bd49819ef8e2c", 67514072),
    "ES2004b": ("dc0e89aa95e37b65c8c5bdd9f661b1c3", 75055832),
    "ES2007c": ("62aee84f4f02d9d070e6e350f739b3b1", 76077100),
    "ES2011b": ("ce1d1655e87b59d2ea0352a06cf5b362", 50600664),
    "ES2012b": ("03996c05b928e6a9a50a1a24f75b65f4", 71669378),
    "ES2015c": ("b3edced6f2ea94c0679ac27d100b31cb", 68348632),
    "IB4002": ("cebcfe0daf8eb172e5adc7b4f4bc036f", 60235820),
    "IN1002": ("5f933dcd463ed1adf076319df667cbe9", 79120600),
    "IS1000b": ("7a07d849156750aff90d95227d336626", 74997452),
    "IS1003c": ("ae050b0271d0f284cc2e28cdac8d730e", 60044093),
    "IS1006b": ("dd6c55520d8864b13eedabbaf179d21d", 69124140),
    "IS1008b": ("ae8ef5f46396fdd3902b959d67761b29", 56593427),
    "IS1009b": ("50fdc537383bb3109a3b7bdaedbbd53d", 65676093),
    "TS3003b": ("2fd856127bf22152c7ae7e8bae3e85eb", 70729772),
    "TS3004b": ("97ba877f0c95c7c5f6ac4e6ae5f57ac8", 71873920),
    "TS3006b": ("0b198dfcc3b8c3be9e612f486a260a48", 75324800),
    "TS3009c": ("a9c9095307bcac19659aab8fb788358e", 82560044),
    "TS3011b": ("b73d78e7983069a32e873dbc6cfe6996", 70854700),
}
MAX_RUN_SEC = 600.0  # search horizon for runs overlapping a pause (no AMI run is longer)
LABEL_DEFAULTS = dict(turn_gap=0.5, hes_gap=0.3, turn_def="floor", max_hold=2.0, bc_max_sec=1.0,
                      bc_max_words=2, act_bridge=0.0)


def _root(root) -> Path:
    return Path(root) if root else DEFAULT_ROOT


# --------------------------------------------------------------------------- download
def _fetch(url: str, dst: Path, retries: int = 20, deadline: float | None = None) -> bool:
    """Resumable single-stream download to ``dst`` (via ``dst.part``). False if ``deadline`` hit."""
    if dst.exists():
        return True
    part = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
        total = int(r.headers["Content-Length"])
    for _ in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have >= total:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=30) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:
                    raise RuntimeError(f"server ignored Range (HTTP {r.status})")
                while b := r.read(1 << 20):
                    f.write(b)
                    if deadline and time.time() > deadline:
                        return False
        except (OSError, RuntimeError) as e:
            log.info(f"  {dst.name}: {type(e).__name__}: {e} (retrying)")
            time.sleep(2)
    if part.stat().st_size != total:
        raise RuntimeError(f"{url}: got {part.stat().st_size} of {total} bytes")
    os.replace(part, dst)
    return True


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def fetch_split_lists(root=None) -> dict[str, list[str]]:
    root = _root(root)
    d = root / "lists"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for ours, theirs in SPLIT_FILES.items():
        p = d / f"{theirs}.meetings.txt"
        if not p.exists():
            _fetch(SPLIT_URL.format(s=theirs), p)
        out[ours] = p.read_text().split()
    return out


def split_lists(root=None) -> dict[str, list[str]]:
    d = _root(root) / "lists"
    if all((d / f"{t}.meetings.txt").exists() for t in SPLIT_FILES.values()):
        return {o: (d / f"{t}.meetings.txt").read_text().split() for o, t in SPLIT_FILES.items()}
    return fetch_split_lists(root)


def subset(n: dict[str, int] | None = None, root=None) -> dict[str, list[str]]:
    """Meetings per split: the DEFAULT_MEETINGS first, then the rest of the official list in order."""
    n = n or {k: len(v) for k, v in DEFAULT_MEETINGS.items()}
    lists = split_lists(root)
    out = {}
    for s, k in n.items():
        order = DEFAULT_MEETINGS[s] + [m for m in lists[s] if m not in DEFAULT_MEETINGS[s]]
        assert set(DEFAULT_MEETINGS[s]) <= set(lists[s]), s
        out[s] = order[:k]
    return out


def download_annotations(root=None) -> Path:
    """ami_public_manual_1.6.2.zip (md5-checked) -> <root>/annotations/{words,segments,corpusResources}."""
    root = _root(root)
    ann = root / "annotations"
    if (ann / "words").exists() and (ann / "corpusResources" / "meetings.xml").exists():
        return ann
    z = root / "ami_public_manual_1.6.2.zip"
    root.mkdir(parents=True, exist_ok=True)
    _fetch(ANNOT_URL, z)
    if _md5(z) != ANNOT_MD5:
        raise RuntimeError(f"md5 mismatch for {z} (delete it and retry)")
    with zipfile.ZipFile(z) as f:
        names = [x for x in f.namelist() if x.split("/")[0] in ("words", "segments", "corpusResources")
                 or x in ("LICENCE.txt", "MANIFEST_MANUAL.txt", "00README_MANUAL.txt")]
        f.extractall(ann, names)
    return ann


def download_audio(meetings: list[str], root=None, connections: int = 4, max_minutes: float | None = None) -> dict:
    """Mix-Headset wavs -> <root>/audio/<m>.Mix-Headset.wav, ``connections`` meetings in parallel.
    Verified against WAV_MD5 (pinned) or the md5 recorded in <root>/audio/md5.json on first download,
    plus a header check (16 kHz mono PCM, byte count consistent). Returns {meeting: status}."""
    import soundfile as sf
    root = _root(root)
    adir = root / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    reg_path = adir / "md5.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    lock = threading.Lock()
    deadline = time.time() + 60 * max_minutes if max_minutes else None

    def one(m):
        dst = adir / f"{m}.Mix-Headset.wav"
        if not _fetch(AUDIO_URL.format(m=m), dst, deadline=deadline):
            return m, "partial (deadline; rerun to resume)"
        info = sf.info(str(dst))
        if info.samplerate != SR or info.channels != 1 or info.frames * 2 + 44 > dst.stat().st_size + 64:
            raise RuntimeError(f"{dst}: unexpected format {info}")
        with lock:
            known = WAV_MD5.get(m) or (tuple(reg[m]) if m in reg else None)
        if known and known[1] != dst.stat().st_size:
            raise RuntimeError(f"{dst}: size {dst.stat().st_size} != pinned {known[1]}")
        h = _md5(dst)
        if known and h != known[0]:
            raise RuntimeError(f"{dst}: md5 {h} != pinned {known[0]} (delete it and retry)")
        with lock:
            reg[m] = [h, dst.stat().st_size]
            reg_path.write_text(json.dumps(reg, indent=1, sort_keys=True))
        return m, "ok" if known else "ok (md5 recorded)"

    with ThreadPoolExecutor(max_workers=max(1, min(4, connections))) as ex:
        return dict(ex.map(one, meetings))


# --------------------------------------------------------------------------- annotation parsing
def normalize(tok: str) -> str:
    """Lowercase; spelled acronyms 'T_V_' -> 'tv'; hyphens -> spaces; keep [a-z0-9' ]."""
    s = tok.lower().replace("_", "").replace("-", " ")
    return " ".join(re.sub(r"[^a-z0-9' ]", "", s).split())


def parse_words_xml(src) -> list[tuple[float, float, str]]:
    """words.xml (path or XML string) -> sorted [(start, end, normalized text)] for timed, non-punctuation
    <w> elements. Vocal sounds, gaps, disfluency markers and punctuation tokens are dropped."""
    root = ET.fromstring(src) if isinstance(src, str) and src.lstrip().startswith("<") else ET.parse(src).getroot()
    out = []
    for w in root.iter("w"):
        if w.get("punc") == "true" or w.get("starttime") is None or w.get("endtime") is None:
            continue
        s, e = float(w.get("starttime")), float(w.get("endtime"))
        out.append((s, max(s, e), normalize(w.text or "")))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def speaker_names(ann: Path) -> dict[str, dict[str, str]]:
    """meetings.xml -> {meeting: {agent letter: global speaker name}}."""
    p = Path(ann) / "corpusResources" / "meetings.xml"
    if not p.exists():
        return {}
    return {m.get("observation"): {s.get("nxt_agent"): s.get("global_name") for s in m.iter("speaker")}
            for m in ET.parse(p).getroot().iter("meeting")}


def global_speakers(ann: Path) -> list[str]:
    """Sorted global speaker names of the whole corpus (stable integer ids for any subset)."""
    return sorted({g for d in speaker_names(ann).values() for g in d.values() if g})


def meeting_words(ann: Path, meeting: str) -> dict[str, list]:
    """{speaker name: words} (global names when meetings.xml is present, else '<meeting>.<agent>')."""
    names = speaker_names(ann).get(meeting, {})
    out = {}
    for p in sorted((Path(ann) / "words").glob(f"{meeting}.*.words.xml")):
        agent = p.name.split(".")[1]
        w = parse_words_xml(p)
        if w:
            out[names.get(agent) or f"{meeting}.{agent}"] = w
    return out


# --------------------------------------------------------------------------- label construction
def activity(words, bridge: float = 0.0) -> list[tuple[float, float]]:
    """Union of word intervals; gaps <= ``bridge`` are closed."""
    out: list[list[float]] = []
    for s, e, _ in sorted(words):
        if out and s - out[-1][1] <= bridge:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def _runs(words, gap):
    runs, cur, end = [], [], -1e9
    for w in words:
        if cur and w[0] - end >= gap:
            runs.append(cur)
            cur = []
        cur.append(w)
        end = max(end, w[1]) if len(cur) > 1 else w[1]
    return runs + ([cur] if cur else [])


def _turn(spk, words, hes_gap, bc_max_sec, bc_max_words):
    hes, end = [], words[0][1]
    for w in words[1:]:
        if w[0] - end >= hes_gap:
            hes.append((end, w[0]))
        end = max(end, w[1])
    start = words[0][0]
    return dict(speaker=spk, start=start, end=end, words=words, hes=hes, n_words=len(words),
                text=" ".join(t for _, _, t in words if t),
                bc=len(words) <= bc_max_words and end - start <= bc_max_sec)


ACTION_DEFAULTS = dict(hes_gap=LABEL_DEFAULTS["hes_gap"], hold="<HOLD>", yield_="<YIELD>")


def action_transcript(words, hes_gap: float = 0.3, hold: str = "<HOLD>", yield_: str = "<YIELD>") -> str:
    """Transcript of one speaker's turn with conversational action tokens.

    ``words`` = the turn's [(start, end, text)] (in window order; text may be "" for a dropped token). Rules, the same
    labels eot-bench uses (``_turn`` / ``turn_examples``):
      <HOLD>   between two consecutive words whose silence (next start - running max end) is >= ``hes_gap`` (0.3 s):
               exactly the turn's ``hes`` intervals, i.e. the within-turn pauses of the primary. Under the default
               floor definition a pause of up to max_hold (2 s) with nobody taking the floor stays inside the turn,
               so those pauses are HOLDs too.
      <YIELD>  once, after the last word: the labelled turn end (``turn_end_frame``), i.e. the end of the primary's
               last word, after which the floor is open or taken by someone else.
    Words with empty text still count for the gaps but add no token."""
    out, end = [], None
    for s, e, t in words:
        if end is not None and s - end >= hes_gap:
            out.append(hold)
        if t:
            out.append(t)
        end = e if end is None else max(end, e)
    if end is not None:
        out.append(yield_)
    return " ".join(out)


def window_action_transcript(turns: list[dict], a: float, b: float, hes_gap: float = 0.3, hold: str = "<HOLD>",
                             yield_: str = "<YIELD>") -> str:
    """Speaker-UNAWARE action transcript of a window [a, b) of a meeting (mirrors what
    a single-channel EOU model is trained on). ``turns`` = the meeting's turns (speaker_turns: every speaker, floor
    definition, backchannels included). Rules:
      words    every speaker's words with start < b and end > a, in time order by start (ties: by end, then speaker);
               overlapping speech is therefore interleaved word by word.
      <HOLD>   between two consecutive words of the SAME turn whose silence (next start - running max end) is
               >= hes_gap, both words inside the window; placed at the time the pause starts (the running max end).
      <YIELD>  at the end of every turn (any speaker, backchannels included) whose end lies inside the window
               (end <= b); placed at the turn end (= its last word's end).
    Events are sorted by (time, kind) with tokens before words at equal times, so a <YIELD> at 3.20 s precedes a
    word starting at 3.20 s. A turn cut by the window start keeps only its words inside it (no HOLD across the
    boundary); a turn cut by the window end gets no <YIELD>."""
    ev = []  # (time, rank, order, token): rank 0 tokens, 1 words
    n = 0
    for t in turns:
        if t["end"] <= a or t["start"] >= b:
            continue
        end = None  # running max end of the turn's words so far (as _turn's hesitation rule)
        for s, e, w in t["words"]:
            inside = e > a and s < b
            if end is not None and inside and end > a and s - end >= hes_gap:
                ev.append((end, 0, n, hold))
                n += 1
            if inside and w:
                ev.append((s, 1, n, w))
                n += 1
            end = e if end is None else max(end, e)
        if t["end"] <= b:
            ev.append((t["end"], 0, n, yield_))
            n += 1
    ev.sort(key=lambda x: (x[0], x[1], x[2]))
    return " ".join(x[3] for x in ev)


def speaker_turns(words_by_spk: dict, turn_gap=0.5, hes_gap=0.3, turn_def="floor", max_hold=2.0,
                  bc_max_sec=1.0, bc_max_words=2, **_) -> list[dict]:
    """All turns of a meeting sorted by start (definitions in the module docstring). Each turn:
    speaker, start, end, words, text, hes [(start, end)], n_words, bc (backchannel)."""
    assert turn_def in ("gap", "floor"), turn_def
    runs = {s: [_turn(s, r, hes_gap, bc_max_sec, bc_max_words) for r in _runs(w, turn_gap)]
            for s, w in words_by_spk.items()}
    if turn_def == "floor":
        floor = sorted((r["start"], r["end"], s) for s, rs in runs.items() for r in rs if not r["bc"])
        starts = [f[0] for f in floor]
        for s, rs in runs.items():
            merged = []
            for r in rs:
                if merged:
                    p = merged[-1]
                    a, b = p["end"], r["start"]
                    # another speaker's non-backchannel run intersecting the pause (a, b) takes the floor
                    lo, hi = bisect.bisect_left(starts, a - MAX_RUN_SEC), bisect.bisect_left(starts, b)
                    taken = any(floor[i][2] != s and floor[i][1] > a for i in range(lo, hi))
                    if b - a < max_hold and not taken:
                        merged[-1] = _turn(s, p["words"] + r["words"], hes_gap, bc_max_sec, bc_max_words)
                        continue
                merged.append(r)
            runs[s] = merged
    return sorted((t for rs in runs.values() for t in rs), key=lambda t: (t["start"], t["end"]))


def next_turn(turns: list[dict], i: int) -> dict | None:
    """The next non-backchannel turn (any speaker) that starts after turn i starts and ends after it ends."""
    t = turns[i]
    for u in turns[i + 1:]:
        if not u["bc"] and u["start"] > t["start"] and u["end"] > t["end"]:
            return u
    return None


def frames(intervals, T: int, offset: float = 0.0) -> np.ndarray:
    """(T,) float32 mask of [s, e) intervals (seconds, shifted by -offset) at 80 ms."""
    y = np.zeros(T, np.float32)
    for s, e in intervals:
        a, b = max(0, int((s - offset) / FRAME_SEC)), min(T, int(math.ceil((e - offset) / FRAME_SEC - 1e-9)))
        if b > a:
            y[a:b] = 1
    return y


def spk_matrix(acts: dict, T: int, offset: float, end: float, max_spks: int = 4, first=None):
    """(T, max_spks) activity; columns: ``first`` (if given) then arrival order within [offset, end).
    Returns (matrix, column order, number of speakers dropped for lack of slots)."""
    active = {}
    for s, iv in acts.items():
        on = [a for a, b in iv if b > offset and a < end]
        if on:
            active[s] = max(offset, min(on))
    order = sorted(active, key=lambda s: (active[s], s))
    if first is not None:
        order = [first] + [s for s in order if s != first]
    y = np.zeros((T, max_spks), np.float32)
    for k, s in enumerate(order[:max_spks]):
        y[:, k] = frames(acts.get(s, []), T, offset)
    return y, order[:max_spks], max(0, len(order) - max_spks)


# --------------------------------------------------------------------------- dataset
class AMI:
    """A list of meetings (audio optional: labels and ``stats()`` only need the annotations)."""

    def __init__(self, meetings: list[str], root=None, audio: bool = True, verbose: bool = True, **labels):
        self.root, self.meetings, self.verbose = _root(root), list(meetings), verbose
        self.cfg = {**LABEL_DEFAULTS, **{k: v for k, v in labels.items() if v is not None}}
        self.ann = self.root / "annotations"
        spk = global_speakers(self.ann)
        self.speaker_ids = spk
        self.spk_index = {s: i for i, s in enumerate(spk)}
        self.words = {m: meeting_words(self.ann, m) for m in self.meetings}
        missing = [m for m in self.meetings if not self.words[m]]
        if missing:
            raise FileNotFoundError(f"no word annotations for {missing} under {self.ann}")
        self.turns = {m: speaker_turns(self.words[m], **self.cfg) for m in self.meetings}
        self.acts = {m: {s: activity(w, self.cfg["act_bridge"]) for s, w in self.words[m].items()}
                     for m in self.meetings}
        self._audio: dict[str, np.ndarray] = {}
        if audio:
            for m in self.meetings:
                self._audio[m] = self._load_audio(m)

    # ----------------------------------------------------------------- audio cache
    def _load_audio(self, m: str) -> np.ndarray:
        npy = self.root / "cache" / f"{m}.f32.npy"
        if not npy.exists():
            import soundfile as sf
            wav = self.root / "audio" / f"{m}.Mix-Headset.wav"
            if not wav.exists():
                raise FileNotFoundError(f"{wav} missing: prepare the AMI corpus under $AUDIOFORGE_DATA first")
            x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            assert sr == SR, (wav, sr)
            npy.parent.mkdir(parents=True, exist_ok=True)
            tmp = npy.with_suffix(".tmp.npy")
            np.save(tmp, np.ascontiguousarray(x.mean(1)))
            os.replace(tmp, npy)
            if self.verbose:
                log.info(f"  cache {m}: {len(x) / SR / 60:.1f} min -> {npy}")
        return np.load(npy, mmap_mode="c")

    def duration(self, m: str) -> float:
        if m in self._audio:
            return len(self._audio[m]) / SR
        return max(e for iv in self.acts[m].values() for _, e in iv)

    def _clip(self, m, a, b):
        x = self._audio[m]
        return x[int(round(a * SR)): int(round(b * SR))]

    def gid(self, name: str) -> int:
        return self.spk_index.get(name, -1)

    # ----------------------------------------------------------------- modes
    def diar(self, window_sec=20.0, hop_sec=10.0, max_spks=4, meetings=None) -> list[dict]:
        out = []
        for m in meetings or self.meetings:
            dur = self.duration(m)
            starts = np.arange(0.0, max(dur - window_sec, 0.0) + 1e-6, hop_sec)
            for a in starts:
                b = min(dur, a + window_sec)
                x = self._clip(m, a, b)
                T = ToneLanguage.n_frames(len(x))
                y, order, dropped = spk_matrix(self.acts[m], T, a, b, max_spks)
                out.append(dict(audio=x, spk_targets=y, meeting=m, start=float(a),
                                speakers=[self.gid(s) for s in order], dropped_speakers=dropped))
        return out

    def turn_examples(self, window_sec=20.0, lead_sec=4.0, trail_sec=2.0, min_trail=1.0, max_spks=4,
                      meetings=None, action_tokens: dict | bool | None = None) -> list[dict]:
        """``action_tokens`` (None = off; True or {hes_gap, hold, yield_, speakers}): ``text`` becomes the primary
        turn's action_transcript (words inside the window, <HOLD> at within-turn pauses, <YIELD> at the end) or, with
        ``speakers: all``, the speaker-unaware window_action_transcript of every speaker; the primary's plain words
        stay in ``text_plain``."""
        act = None if not action_tokens else {**ACTION_DEFAULTS, **(action_tokens if isinstance(action_tokens, dict) else {})}
        out = []
        for m in meetings or self.meetings:
            turns, dur = self.turns[m], self.duration(m)
            by_spk: dict[str, list] = {}
            for t in turns:
                by_spk.setdefault(t["speaker"], []).append(t)
            for ts in by_spk.values():  # same-speaker neighbours bound the window
                for j, t in enumerate(ts):
                    t["_prev_end"] = ts[j - 1]["end"] if j else 0.0
                    t["_next_start"] = ts[j + 1]["start"] if j + 1 < len(ts) else dur
            for t in turns:
                if t["bc"]:
                    continue
                prev_end, next_start = t["_prev_end"], t["_next_start"]
                b = min(dur, t["end"] + trail_sec, next_start)
                if b - t["end"] < min_trail:
                    continue
                a = max(0.0, prev_end, t["start"] - lead_sec, b - window_sec)
                x = self._clip(m, a, b)
                T = ToneLanguage.n_frames(len(x))
                prim = frames(activity([w for w in t["words"] if w[1] > a], self.cfg["act_bridge"]), T, a)
                if not prim.any():
                    continue
                y, order, dropped = spk_matrix(self.acts[m], T, a, b, max_spks, first=t["speaker"])
                y[:, 0] = prim
                nz = np.nonzero(prim)[0]
                onset, end = int(nz[0]), int(nz[-1]) + 1
                eot = np.zeros(T, np.float32)
                eot[end:] = 1
                hes = frames([h for h in t["hes"] if h[1] > a], T, a) * (1 - prim)  # silent frames only
                ww = [w for w in t["words"] if w[1] > a]  # only words inside the window
                ex = dict(audio=x, spk_targets=y, spk_act=prim, primary_act=prim, eot=eot, hes=hes,
                          turn_end_frame=end, onset_frame=onset, text=" ".join(w[2] for w in ww),
                          speaker=self.gid(t["speaker"]), meeting=m, start=float(a),
                          onset_clipped=bool(t["start"] < a), n_hesitations=len(t["hes"]),
                          dropped_speakers=dropped)
                if act:
                    ex["text_plain"] = ex["text"]
                    if act.get("speakers", "primary") == "all":  # speaker-unaware: every speaker, time order
                        ex["text"] = window_action_transcript(turns, a, b, act["hes_gap"], act["hold"], act["yield_"])
                    else:
                        ex["text"] = action_transcript(ww, act["hes_gap"], act["hold"], act["yield_"])
                out.append(ex)
        return out

    def asr(self, min_sec=1.0, max_sec=15.0, guard=0.2, meetings=None, action_tokens: dict | bool | None = None) -> list[dict]:
        """``action_tokens``: ``text`` gets <HOLD> at >= hes_gap pauses inside the
        segment and, at the segment end, <YIELD> if the segment's last word ends a floor-definition turn of its speaker
        (self.turns) or <HOLD> if the run ends but the turn continues (a >= turn_gap pause with nobody taking the floor);
        a chunk cut by max_sec gets no end token. The audio still stops 0.1 s after the last word, so the end token
        must be predicted from the words / prosody, not from the following silence."""
        from .librispeech import eou_targets
        act = None if not action_tokens else {**ACTION_DEFAULTS, **(action_tokens if isinstance(action_tokens, dict) else {})}
        out = []
        for m in meetings or self.meetings:
            turn_ends = {(t["speaker"], round(t["end"], 4)) for t in self.turns[m]}
            for s, ws in self.words[m].items():
                oth = sorted(iv for o, ivs in self.acts[m].items() if o != s for iv in ivs)
                ostart = [iv[0] for iv in oth]
                for run in _runs(ws, self.cfg["turn_gap"]):
                    # split at word boundaries into <= max_sec chunks
                    chunks, cur = [], []
                    for w in run:
                        if cur and w[1] - cur[0][0] > max_sec:
                            chunks.append(cur)
                            cur = []
                        cur.append(w)
                    chunks.append(cur)
                    for c in chunks:
                        a, b = c[0][0], max(w[1] for w in c)
                        if b - a < min_sec or not any(w[2] for w in c):
                            continue
                        lo, hi = bisect.bisect_left(ostart, a - guard - MAX_RUN_SEC), bisect.bisect_left(ostart, b + guard)
                        if any(oth[i][1] > a - guard for i in range(lo, hi)):
                            continue  # someone else speaks within guard: not single-speaker
                        a0, b0 = max(0.0, a - 0.1), b + 0.1
                        x = self._clip(m, a0, b0)
                        T = ToneLanguage.n_frames(len(x))
                        vad = frames(activity(c, self.cfg["act_bridge"]), T, a0)
                        text = " ".join(w[2] for w in c if w[2])
                        ex = dict(audio=x, text=text, speaker=self.gid(s), vad=vad, eou=eou_targets(vad), meeting=m,
                                  start=float(a0), duration=len(x) / SR)
                        if act:
                            body = action_transcript(c, act["hes_gap"], act["hold"], "").strip()
                            if c[-1] is run[-1]:  # the run ends here: turn end (yield) or a within-turn pause (hold)
                                end_tok = act["yield_"] if (s, round(b, 4)) in turn_ends else act["hold"]
                                body = f"{body} {end_tok}"
                            ex["text_plain"], ex["text"], ex["end_token"] = text, body, body.split()[-1] if body else ""
                        out.append(ex)
        return out

    def examples(self, mode: str = "turn", **kw) -> list[dict]:
        return {"turn": self.turn_examples, "diar": self.diar, "asr": self.asr}[mode](**kw)

    # ----------------------------------------------------------------- statistics
    def stats(self, taus=(0.3, 0.5, 0.7, 1.0, 1.5, 2.0)) -> dict:
        return corpus_stats({m: self.words[m] for m in self.meetings}, {m: self.duration(m) for m in self.meetings},
                            self.cfg, taus)


_pct = pct_dict  # {p10..p90}, 3 dp


def corpus_stats(words: dict, durations: dict, cfg: dict | None = None, taus=(0.3, 0.5, 0.7, 1.0, 1.5, 2.0)) -> dict:
    """Turn / hesitation / overlap statistics over meetings (annotation only).

    switch_gap: next other-speaker turn start - turn end (true end-of-turn gap; < 0 = overlap).
    hold_gap: silence before the SAME speaker's next turn when no one else took the floor in between
              (with turn_def=gap these are pauses >= turn_gap that the gap rule calls turn ends).
    hesitation: within-turn silences >= hes_gap; pause_ge_0.1: within-turn silences >= 0.1 s.
    timeout table: for a silence timeout tau, cutoff = share of turns containing a within-turn pause
    >= tau (a speaker-unaware timeout on the turn holder alone would fire inside them); too_late = share of
    switches where the next speaker started < tau after the end (the timeout had no chance to fire first)."""
    cfg = {**LABEL_DEFAULTS, **(cfg or {})}
    sw, hold, hes, pauses, n_turns, n_bc, lens, words_n = [], [], [], [], 0, 0, [], []
    tot_min, speech, ovl, cut = 0.0, 0.0, 0.0, {t: 0 for t in taus}
    spk = set()
    max_pause = []
    for m, wb in words.items():
        spk |= set(wb)
        turns = speaker_turns(wb, **cfg)
        dur = durations[m]
        tot_min += dur / 60
        for i, t in enumerate(turns):
            if t["bc"]:
                n_bc += 1
                continue
            n_turns += 1
            lens.append(t["end"] - t["start"])
            words_n.append(t["n_words"])
            hes += [b - a for a, b in t["hes"]]
            ws, end, mp = t["words"], t["words"][0][1], 0.0
            for w in ws[1:]:
                g = w[0] - end
                if g >= 0.1:
                    pauses.append(g)
                mp = max(mp, g)
                end = max(end, w[1])
            max_pause.append(mp)
            for tau in taus:
                cut[tau] += mp >= tau
            u = next_turn(turns, i)
            if u is not None:
                (sw if u["speaker"] != t["speaker"] else hold).append(u["start"] - t["end"])
        # overlap at 10 ms
        n = int(math.ceil(dur * 100)) + 1
        cnt = np.zeros(n, np.int16)
        for w in wb.values():
            m1 = np.zeros(n, bool)
            for a, b in activity(w, cfg["act_bridge"]):
                m1[int(a * 100): int(math.ceil(b * 100))] = True
            cnt += m1
        speech += float((cnt >= 1).sum()) / 100
        ovl += float((cnt >= 2).sum()) / 100
    swa = np.asarray(sw)
    pos = swa[swa > 0]
    hesa = np.asarray(hes)
    return dict(
        meetings=len(words), hours=round(tot_min / 60, 2), speakers=len(spk),
        label_cfg=cfg, turns=n_turns, backchannels=n_bc,
        turns_per_min=round(n_turns / max(tot_min, 1e-9), 2), backchannels_per_min=round(n_bc / max(tot_min, 1e-9), 2),
        turn_sec=_pct(lens), turn_words_median=float(np.median(words_n)) if words_n else 0,
        speech_frac=round(speech / (tot_min * 60), 3), overlap_frac_of_speech=round(ovl / max(speech, 1e-9), 3),
        switch_gap=dict(n=len(sw), overlap_share=round(float((swa < 0).mean()), 3) if len(sw) else None, **_pct(sw)),
        switch_gap_positive=dict(n=int(len(pos)), **_pct(pos)),
        hold_gap=dict(n=len(hold), **_pct(hold)),
        hesitation=dict(n=int(len(hesa)), per_turn=round(len(hesa) / max(1, n_turns), 3), **_pct(hesa)),
        pause_ge_0p1=dict(n=len(pauses), **_pct(pauses)),
        hesitation_longer_than_median_switch_gap=(round(float((hesa > np.median(pos)).mean()), 3)
                                                  if len(hesa) and len(pos) else None),
        timeout=[dict(tau=t, cutoff=round(cut[t] / max(1, n_turns), 3),
                      too_late=round(float((swa < t).mean()), 3) if len(sw) else None) for t in taus],
    )


# --------------------------------------------------------------------------- recipe hook
def recipe_data(cfg: dict, split: str) -> list[dict]:
    """``data: {ami: {...}}`` for ``audioforge.train.load_data``.

    Keys: root, mode (turn | diar | asr), train_split (train), val_split (dev), meetings ({train: [...],
    val: [...]} to override), n_meetings ({train: 12, val: 4}), label keys (turn_gap, hes_gap, turn_def,
    max_hold, bc_max_sec, bc_max_words, act_bridge), mode kwargs (window_sec, hop_sec, lead_sec, trail_sec,
    min_trail, asr_max_sec, asr_min_sec), n_train / n_val (seeded caps), seed, ext_tracks (turn mode: cached
    external-diarizer tracks -> spk_act_ext / spk_targets_ext, see attach_ext_tracks), action_tokens (turn mode:
    text with <HOLD> / <YIELD>, see AMI.turn_examples). With data.synthetic.n_<split>
    (hermetic smoke tests) a synthetic stand-in of the same kind is returned."""
    d = cfg["data"]
    ac = dict(d["ami"])
    mode = ac.get("mode", "turn")
    syn = d.get("synthetic") or {}
    if f"n_{split}" in syn:
        n, seed = syn[f"n_{split}"], 0 if split == "train" else 1
        if mode == "turn":
            from ..conversation import conversation_dataset
            return list(conversation_dataset(n, seed=seed))
        return synthetic_dataset({"diar": "diar", "asr": "multitask"}[mode], n, seed=seed)
    sname = ac.get("train_split", "train") if split == "train" else ac.get("val_split", "dev")
    key = "train" if split == "train" else "val"
    meetings = (ac.get("meetings") or {}).get(key)
    if not meetings:
        k = (ac.get("n_meetings") or {}).get(key, len(DEFAULT_MEETINGS[sname]))
        meetings = subset({sname: k}, ac.get("root"))[sname]
    labels = {k: ac[k] for k in LABEL_DEFAULTS if k in ac}
    ds = AMI(meetings, ac.get("root"), **labels)
    if mode == "diar":
        data = ds.diar(ac.get("window_sec", 20.0), ac.get("hop_sec", 10.0))
    elif mode == "turn":
        data = ds.turn_examples(ac.get("window_sec", 20.0), ac.get("lead_sec", 4.0), ac.get("trail_sec", 2.0),
                                ac.get("min_trail", 1.0), action_tokens=ac.get("action_tokens"))
    else:
        data = ds.asr(ac.get("asr_min_sec", 1.0), ac.get("asr_max_sec", 15.0), action_tokens=ac.get("action_tokens"))
    cap = ac.get(f"n_{split}")
    if cap and cap < len(data):
        idx = sorted(random.Random(ac.get("seed", 0)).sample(range(len(data)), cap))
        data = [data[i] for i in idx]
    ext = ""
    if mode == "turn" and ac.get("ext_tracks"):
        ext = " " + json.dumps(attach_ext_tracks(data, ac["ext_tracks"], sname, ac.get("root"), ac.get("trail_sec")))
    log.info(f"[ami] {mode}/{split}: {len(meetings)} meetings -> {len(data)} examples{ext}")
    return data


def attach_ext_tracks(data: list[dict], spec, split: str, root=None, trail_sec: float | None = None) -> dict:
    """``ext_tracks: offline | stream | {source: offline|stream, dir: <cache dir>, require: false, fallback: ...}``:
    attach the cached external-diarizer tracks (datasets/ext_tracks.py) to the
    turn examples as spk_act_ext (T,) (enrollment-picked column, oracle overlap on [onset, turn_end) only),
    spk_targets_ext (T, 4) and spk_prim_ext (4,) (one-hot of that column). Off (no key) by default; items without a
    track carry -1 (never used). ``stream`` prefers the streaming track (what the head sees at inference) and falls
    back to the offline track of an item without one (``fallback: offline``, the default for stream; ``fallback:
    null`` = streaming only). ``trail_sec`` (the recipe's; != 2 s) selects the window set's own cache dir
    <split>_trail<x> (ext_tracks.split_dirname) unless ``dir`` is given."""
    from . import ext_tracks as xt
    spec = {"source": spec} if isinstance(spec, str) else dict(spec)
    src = spec.get("source", "offline")
    fb = spec.get("fallback", "offline" if src == "stream" else None)
    res = xt.attach(data, root, split, src, spec.get("dir"), require=bool(spec.get("require", False)), fallback=fb,
                    trail_sec=trail_sec)
    return {"ext_tracks": src, **res}
