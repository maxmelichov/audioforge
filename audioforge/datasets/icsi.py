"""ICSI Meeting Corpus (CC-BY-4.0) as audioforge list-of-dict data, with the SAME labels as ``ami.py``.

Source (https://groups.inf.ed.ac.uk/ami/icsi/): 75 meetings / 71.7 h of natural research-group meetings
(3-10 speakers, 6-7 typical), recorded at ICSI Berkeley 2000-2002.
  audio        ``ICSIsignals/NXT/<m>.interaction.wav``: the distributed single-file HEADSET MIX, 16 kHz mono
               PCM16 (no mixing of the individual ``chan*.sph`` headset channels needed). Stored as
               ``<root>/audio/<m>.Mix-Headset.wav`` so ``ami.AMI`` reads it unchanged.
  annotations  ``ICSI_core_NXT.zip`` v1.0 (NXT format): ``Words/<m>.<agent>.words.xml`` with per-word times
               and ``Segments/<m>.<agent>.segs.xml`` (transcriber segments, ``participant`` = speaker tag).

ICSI words differ from AMI's in three ways, resolved by ``convert_words`` into AMI-format words XML under
``<root>/annotations/words`` (+ ``corpusResources/meetings.xml``), so that ``ami.AMI`` - its turn / hesitation
/ backchannel rules, window builders and statistics - runs on ICSI unchanged (``ICSI`` subclasses it):
  1. contractions, possessives and hyphenated words are split into pieces ("you" + "'re", "mm" - "hmm");
     pieces are re-joined into ONE token as AMI writes them ("you're", "mm-hmm" -> "mm hmm"), so the
     backchannel word count matches AMI's.
  2. a few multi-token stretches (628 tokens corpus-wide) are timed only at their outer edges (first token has
     only a start, last only an end, e.g. "two hour long" [698.64, 699.35]); the span is split over the tokens
     by character length.
  3. 5 % of words have no time at all (two thirds are the read digit strings "Transcript L-..., one two ...",
     the rest unaligned stretches and "@@" unintelligible speech). Their speaker's activity is unknown there,
     so each untimed stretch becomes an UNTIMED ZONE (its transcriber segment, narrowed to the neighbouring
     timed words of the same segment), as does everything after the last transcriber segment. Examples that
     overlap a zone are dropped (``exclude_untimed``), so no window has unlabelled speech.
Punctuation (c = CM . QM EXCLM HYPH quotes, SYM "-"), vocal / non-vocal sounds, pauses, comments and
disfluency markers are dropped, as in ami.py.

Speaker ids: ``SPEAKER_OFFSET`` (1000) + index into the sorted ICSI speaker tags of the whole corpus
(``me011`` ...), so they never collide with AMI's 0..189 and stay fixed for any subset.

Splits: ICSI has no official partition. We use the Kaldi / lhotse ICSI recipe split (Renals & Swietojanski,
HSCMA 2014): dev = {Bmr021, Bns001}, eval = {Bmr013, Bmr018, Bro021}, train = the other 70 meetings. It is
NOT speaker-disjoint (the same research groups meet repeatedly). See research/archive/ICSI.md.
"""
from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.sax.saxutils import escape

from ..paths import DATA_ROOT
from . import ami
from .ami import LABEL_DEFAULTS, SR, _fetch, _md5, normalize

log = logging.getLogger(__name__)

DEFAULT_ROOT = DATA_ROOT / "icsi"
BASE = "https://groups.inf.ed.ac.uk/ami"
AUDIO_URL = BASE + "/ICSIsignals/NXT/{m}.interaction.wav"
ANNOT_URL = BASE + "/ICSICorpusAnnotations/ICSI_core_NXT.zip"
ANNOT_MD5 = "e5e42cf5e5fbb8db22bc160e551465b7"  # ICSI_core_NXT.zip (19,488,722 bytes) as served 2026-09-26
SPEAKER_OFFSET = 1000
PUNC = {"CM", ".", "QM", "EXCLM", "HYPH", "LQUOTE", "RQUOTE", "QUOTE"}
NITE_ID = "{http://nite.sourceforge.net/}id"

# Kaldi egs/icsi (P. Swietojanski) / lhotse.recipes.icsi PARTITIONS; we call "test" "eval".
SPLITS = {
    "dev": ["Bmr021", "Bns001"],
    "eval": ["Bmr013", "Bmr018", "Bro021"],
}
ALL_MEETINGS = [
    "Bdb001", "Bed002", "Bed003", "Bed004", "Bed005", "Bed006", "Bed008", "Bed009", "Bed010", "Bed011", "Bed012",
    "Bed013", "Bed014", "Bed015", "Bed016", "Bed017", "Bmr001", "Bmr002", "Bmr003", "Bmr005", "Bmr006", "Bmr007",
    "Bmr008", "Bmr009", "Bmr010", "Bmr011", "Bmr012", "Bmr013", "Bmr014", "Bmr015", "Bmr016", "Bmr018", "Bmr019",
    "Bmr020", "Bmr021", "Bmr022", "Bmr023", "Bmr024", "Bmr025", "Bmr026", "Bmr027", "Bmr028", "Bmr029", "Bmr030",
    "Bmr031", "Bns001", "Bns002", "Bns003", "Bro003", "Bro004", "Bro005", "Bro007", "Bro008", "Bro010", "Bro011",
    "Bro012", "Bro013", "Bro014", "Bro015", "Bro016", "Bro017", "Bro018", "Bro019", "Bro021", "Bro022", "Bro023",
    "Bro024", "Bro025", "Bro026", "Bro027", "Bro028", "Bsr001", "Btr001", "Btr002", "Buw001"]
SPLITS["train"] = [m for m in ALL_MEETINGS if m not in SPLITS["dev"] + SPLITS["eval"]]
# Default train subset (~10.2 h): 12 meetings of 45-58 min from every meeting series (Bdb, Bed, Bmr, Bro, Bns,
# Bsr) whose word annotation covers the whole recording. Larger subsets append the rest in list order.
DEFAULT_MEETINGS = {
    "train": ["Bdb001", "Bed010", "Bed011", "Bmr009", "Bmr014", "Bmr022", "Bmr026", "Bro010", "Bro016", "Bro026",
              "Bns003", "Bsr001"],
    "dev": list(SPLITS["dev"]),
    "eval": list(SPLITS["eval"]),
}
# Content-Length of every <m>.interaction.wav (HTTP HEAD, 2026-09-26; 8.26 GB for all 75 meetings).
WAV_BYTES = dict(zip(ALL_MEETINGS, [
    96701996, 125309570, 142364460, 108770946, 126788140, 120283436, 164907480, 103184428, 102224344, 104860290,
    84298370, 114538200, 113427928, 155574316, 90525314, 76206722, 70044628, 87895852, 147986904, 136402136,
    150882604, 115308674, 131047724, 95906946, 104573826, 128711384, 80603608, 93038124, 96424748, 95177090,
    125937368, 109500716, 114756482, 107305688, 70782168, 95288280, 102701528, 99496664, 63616728, 86173656,
    93820290, 105993602, 83100632, 50227842, 103481644, 191361752, 197050156, 96224044, 167131608, 134644952,
    135492226, 68296066, 65585452, 92900140, 72583640, 127692588, 72906114, 111209176, 32474668, 96629634,
    108223532, 110145580, 124241196, 116116354, 101827458, 141616770, 151650008, 67921708, 109269634, 140077016,
    109490392, 110367276, 127310380, 170204972, 131490434]))
# md5 of the downloaded wavs, pinned from the first download (the site publishes none); also in audio/md5.json.
WAV_MD5: dict[str, str] = {
    "Bdb001": "ee9a1a9cc9835f257aef4f362b5dccf5",
    "Bed010": "c10e1be8a26cc1645ce92447a70ed95f",
    "Bed011": "c1e2b7fed0e396a94eaae4e9c826d964",
    "Bmr009": "5d51e5f92d189279e5916ccde0e23ef7",
    "Bmr013": "18a39f56472af15ef25ee7a498e94ea9",
    "Bmr014": "c07183ca210cf472fdd58ac6f0f41523",
    "Bmr018": "b3028a07061a52447184adf3719f97ea",
    "Bmr021": "42d73293da97030593d26a92141ce70e",
    "Bmr022": "c3d24c004eff4483ce84b9c9988340d4",
    "Bmr026": "2d717b9b27c824c9d17f62e5ef430778",
    "Bns001": "b351965fd4a307fd405ee80d79fbedc4",
    "Bns003": "f98085d9c6c3a4dfeb2a633d98ccb061",
    "Bro010": "2bcd166e9c7dd119e0da39a579cd6171",
    "Bro016": "dfe129fa5b108e583d0df330235d9fef",
    "Bro021": "6aa9d4662cfa018ee77ddc6d882588ff",
    "Bro026": "c8e01a04772c2b915372f1d08052fd8a",
    "Bsr001": "a17c501ebf52850cc671390804f9bf7b",
}


def _root(root) -> Path:
    return Path(root) if root else DEFAULT_ROOT


# --------------------------------------------------------------------------- ICSI NXT -> AMI-format words
def _t(x):
    return float(x) if x not in (None, "") else None


def _tokens(elements) -> list[dict]:
    """<w> elements of one words.xml (document order) -> tokens: pieces joined by a HYPH element or starting
    with an apostrophe ("'re", "'s") merge into one token. Token: text, start (first piece), end (last piece),
    id (first piece)."""
    toks, hyph = [], False
    for el in elements:
        if el.tag != "w":
            continue
        c, txt = el.get("c"), (el.text or "").strip()
        if c in PUNC or (c == "SYM" and txt == "-"):
            hyph = hyph or c == "HYPH"
            continue
        s, e = _t(el.get("starttime")), _t(el.get("endtime"))
        if toks and (hyph or txt.startswith("'")):
            p = toks[-1]
            p["text"] += ("-" if hyph else "") + txt
            p["end"] = e
        else:
            toks.append(dict(text=txt, start=s, end=e, id=el.get(NITE_ID)))
        hyph = False
    return toks


def convert_words(words_src, segs_src=None) -> tuple[list[tuple[float, float, str]], list[tuple[float, float]], dict]:
    """One ICSI words.xml (+ its segs.xml) -> (timed words [(start, end, normalized text)], untimed zones
    [(start, end)], counts). Paths or XML strings. See the module docstring for the rules."""
    def parse(src):
        return ET.fromstring(src) if isinstance(src, str) and src.lstrip().startswith("<") else ET.parse(src).getroot()

    root = parse(words_src)
    elements = list(root)
    order = {el.get(NITE_ID): i for i, el in enumerate(elements)}
    toks = _tokens(elements)
    words, untimed, n_chain, when = [], [], 0, {}  # when: token index -> (start, end) of timed tokens
    i = 0
    while i < len(toks):
        t = toks[i]
        if t["start"] is not None and t["end"] is not None:
            when[i] = (t["start"], max(t["start"], t["end"]))
            words.append((*when[i], normalize(t["text"])))
            i += 1
            continue
        if t["start"] is not None:  # chain: following tokens without a start, closed by the first with an end
            j = i + 1
            while j < len(toks) and toks[j]["start"] is None and toks[j]["end"] is None:
                j += 1
            if j < len(toks) and toks[j]["start"] is None and toks[j]["end"] is not None and toks[j]["end"] >= t["start"]:
                chain = toks[i:j + 1]
                s, e = t["start"], toks[j]["end"]
                w = [max(1, len(re.sub(r"\W", "", x["text"]))) for x in chain]
                acc = 0
                for q, (x, k) in enumerate(zip(chain, w)):
                    a = s + (e - s) * acc / sum(w)
                    acc += k
                    when[i + q] = (round(a, 3), round(s + (e - s) * acc / sum(w), 3))
                    words.append((*when[i + q], normalize(x["text"])))
                n_chain += len(chain)
                i = j + 1
                continue
        untimed.append(i)
        i += 1
    # untimed zones: the transcriber segment of each untimed token, narrowed to the timed neighbours inside it
    segs = []
    if segs_src is not None:
        for sg in parse(segs_src).iter("segment"):
            a, b = _t(sg.get("starttime")), _t(sg.get("endtime"))
            for ch in sg:
                m = re.search(r"#id\(([^)]*)\)(?:\.\.id\(([^)]*)\))?", ch.get("href", ""))
                if m and a is not None and b is not None and m.group(1) in order:
                    segs.append((order[m.group(1)], order.get(m.group(2) or m.group(1), order[m.group(1)]), a, b))
    zones, no_seg = [], 0
    for k in untimed:
        pos = order.get(toks[k]["id"], -1)
        seg = next(((a, b) for lo, hi, a, b in segs if lo <= pos <= hi), None)
        if seg is None:
            no_seg += 1
            continue
        a, b = seg
        prev = next((when[q][1] for q in range(k - 1, -1, -1) if q in when), None)
        nxt = next((when[q][0] for q in range(k + 1, len(toks)) if q in when), None)
        if prev is not None and a <= prev <= b:
            a = prev
        if nxt is not None and a <= nxt <= b:
            b = nxt
        zones.append((a, max(a, b)))
    words.sort(key=lambda x: (x[0], x[1]))
    # everything after the last transcriber segment is unannotated (segments > 120 s are bogus, e.g. Bro025)
    last = max((b for _, _, a, b in segs if b - a <= 120), default=max((e for _, e, _ in words), default=0.0))
    return words, _merge(zones), dict(tokens=len(toks), timed=len(words), chain=n_chain, untimed=len(untimed),
                                      untimed_no_segment=no_seg, last_segment_end=last)


def _merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def participant(segs_src) -> str | None:
    for sg in ET.parse(segs_src).getroot().iter("segment"):
        if sg.get("participant"):
            return sg.get("participant")
    return None


def convert_annotations(raw: Path, ann: Path, meetings=None) -> dict:
    """<raw>/ICSI/{Words,Segments} -> <ann>/words/<m>.<agent>.words.xml (AMI format, timed words only),
    <ann>/corpusResources/meetings.xml (agent -> speaker tag), <ann>/untimed.json ({m: [[s, e], ...]} incl.
    [last segment end, inf) as [end, -1]) and <ann>/conversion.json (counts)."""
    raw, ann = Path(raw) / "ICSI", Path(ann)
    (ann / "words").mkdir(parents=True, exist_ok=True)
    (ann / "corpusResources").mkdir(parents=True, exist_ok=True)
    meetings = meetings or sorted({p.name.split('.')[0] for p in (raw / 'Words').glob('*.words.xml')})
    names, zones, counts = {}, {}, {}
    for m in meetings:
        names[m], z, last, cnt = {}, [], 0.0, {}
        for p in sorted((raw / "Words").glob(f"{m}.*.words.xml")):
            agent = p.name.split(".")[1]
            segs = raw / "Segments" / f"{m}.{agent}.segs.xml"
            words, zz, c = convert_words(p, segs if segs.exists() else None)
            tag = participant(segs) if segs.exists() else None
            if tag or words:  # 3 agents (Bmr012.I, Bns001.H, Buw001.J) have empty files: no speaker id
                names[m][agent] = tag or f"{m}.{agent}"
            z += zz
            last = max(last, c.pop("last_segment_end"))
            for k, v in c.items():
                cnt[k] = cnt.get(k, 0) + v
            body = "".join(f'   <w nite:id="{m}.{agent}.w{i}" starttime="{s:.3f}" endtime="{e:.3f}">{escape(t)}</w>\n'
                           for i, (s, e, t) in enumerate(words))
            (ann / "words" / f"{m}.{agent}.words.xml").write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<nite:root nite:id="{m}.{agent}.words" xmlns:nite="http://nite.sourceforge.net/">\n{body}</nite:root>\n')
        zones[m] = [list(x) for x in _merge(z)] + [[round(last, 3), -1]]
        counts[m] = cnt
    xml = "".join(f'   <meeting observation="{m}">\n' + "".join(
        f'      <speaker nxt_agent="{a}" global_name="{escape(g)}"/>\n' for a, g in sorted(d.items())) + "   </meeting>\n"
        for m, d in names.items())
    (ann / "corpusResources" / "meetings.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<nite:root nite:id="meet.00" xmlns:nite="http://nite.sourceforge.net/">\n'
        f"{xml}</nite:root>\n")
    (ann / "untimed.json").write_text(json.dumps(zones, indent=0))
    (ann / "conversion.json").write_text(json.dumps(counts, indent=0))
    for f in ("LICENCE.txt", "00README.txt"):
        if (raw / f).exists():
            (ann / f).write_text((raw / f).read_text(errors="replace"))
    return counts


# --------------------------------------------------------------------------- download
def download_annotations(root=None) -> Path:
    """ICSI_core_NXT.zip (md5-checked) -> <root>/raw/ICSI/{Words,Segments,...} -> converted <root>/annotations."""
    root = _root(root)
    ann = root / "annotations"
    if (ann / "untimed.json").exists() and (ann / "corpusResources" / "meetings.xml").exists():
        return ann
    z = root / "ICSI_core_NXT.zip"
    root.mkdir(parents=True, exist_ok=True)
    _fetch(ANNOT_URL, z)
    if _md5(z) != ANNOT_MD5:
        raise RuntimeError(f"md5 mismatch for {z} (delete it and retry)")
    with zipfile.ZipFile(z) as f:
        keep = [x for x in f.namelist() if x.split("/")[1:2] and x.split("/")[1] in
                ("Words", "Segments", "speakers.xml", "LICENCE.txt", "00README.txt", "ICSI-metadata.xml")]
        f.extractall(root / "raw", keep)
    convert_annotations(root / "raw", ann)
    return ann


def download_audio(meetings: list[str], root=None, connections: int = 4, max_minutes: float | None = None) -> dict:
    """<m>.interaction.wav -> <root>/audio/<m>.Mix-Headset.wav (<= 4 in parallel, resumable). Checked: byte
    count vs WAV_BYTES (the served Content-Length), header (16 kHz mono), md5 vs WAV_MD5 / audio/md5.json."""
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
        if m in WAV_BYTES and dst.stat().st_size != WAV_BYTES[m]:
            raise RuntimeError(f"{dst}: {dst.stat().st_size} bytes != served {WAV_BYTES[m]} (delete it and retry)")
        info = sf.info(str(dst))
        if info.samplerate != SR or info.channels != 1:
            raise RuntimeError(f"{dst}: unexpected format {info}")
        st = dst.stat()
        with lock:
            known = WAV_MD5.get(m) or (reg[m][0] if m in reg else None)
            seen = m in reg and reg[m][1] == st.st_size and reg[m][2] == st.st_mtime
        if known and seen:
            return m, "ok"  # verified before, file untouched
        h = _md5(dst)
        if known and h != known:
            raise RuntimeError(f"{dst}: md5 {h} != pinned {known} (delete it and retry)")
        with lock:
            reg[m] = [h, dst.stat().st_size, dst.stat().st_mtime]
            reg_path.write_text(json.dumps(reg, indent=1, sort_keys=True))
        return m, "ok" if known else "ok (md5 recorded)"

    with ThreadPoolExecutor(max_workers=max(1, min(4, connections))) as ex:
        return dict(ex.map(one, meetings))


def subset(n: dict[str, int] | None = None) -> dict[str, list[str]]:
    """Meetings per split: DEFAULT_MEETINGS first, then the rest of the split in list order."""
    n = n or {k: len(v) for k, v in DEFAULT_MEETINGS.items()}
    return {s: (DEFAULT_MEETINGS[s] + [m for m in SPLITS[s] if m not in DEFAULT_MEETINGS[s]])[:k] for s, k in n.items()}


# --------------------------------------------------------------------------- dataset
class ICSI(ami.AMI):
    """ami.AMI on the converted ICSI annotations: identical label rules, example modes and statistics, plus
    namespaced speaker ids and the untimed-zone / >max_spks filters."""

    def __init__(self, meetings: list[str], root=None, audio: bool = True, verbose: bool = True,
                 exclude_untimed: bool = True, drop_overfull: bool = True, **labels):
        root = _root(root)
        if not (root / "annotations" / "untimed.json").exists():
            if (root / "raw" / "ICSI" / "Words").exists():
                convert_annotations(root / "raw", root / "annotations")
            else:
                raise FileNotFoundError(f"{root}/annotations missing: run scripts/research/prepare_icsi.py")
        zones = json.loads((root / "annotations" / "untimed.json").read_text())
        self.zones = {m: [(a, b if b >= 0 else float("inf")) for a, b in zones.get(m, [])] for m in meetings}
        self.exclude_untimed, self.drop_overfull = exclude_untimed, drop_overfull
        self.speaker_offset = SPEAKER_OFFSET
        self.filtered: dict[str, dict] = {}
        super().__init__(meetings, root, audio=audio, verbose=verbose, **labels)

    def _load_audio(self, m: str):
        try:
            return super()._load_audio(m)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{e} (ICSI: run scripts/research/prepare_icsi.py)") from None

    def gid(self, name: str) -> int:
        i = super().gid(name)
        return -1 if i < 0 else SPEAKER_OFFSET + i

    def untimed_sec(self, m: str) -> float:
        dur = self.duration(m)
        return sum(max(0.0, min(b, dur) - a) for a, b in self.zones[m])

    def _keep(self, ex: dict) -> str | None:
        if self.drop_overfull and ex.get("dropped_speakers", 0) > 0:
            return "overfull"
        if self.exclude_untimed:
            a = ex["start"]
            b = a + len(ex["audio"]) / SR
            if any(za < b and zb > a for za, zb in self.zones[ex["meeting"]]):
                return "untimed"
        return None

    def _filter(self, mode: str, data: list[dict]) -> list[dict]:
        why = [self._keep(e) for e in data]
        self.filtered[mode] = dict(total=len(data), untimed=why.count("untimed"), overfull=why.count("overfull"))
        return [e for e, w in zip(data, why) if w is None]

    def diar(self, *a, **kw) -> list[dict]:
        return self._filter("diar", super().diar(*a, **kw))

    def turn_examples(self, *a, **kw) -> list[dict]:
        return self._filter("turn", super().turn_examples(*a, **kw))

    def asr(self, *a, **kw) -> list[dict]:
        return self._filter("asr", super().asr(*a, **kw))

    def stats(self, *a, **kw) -> dict:
        st = super().stats(*a, **kw)
        dur = sum(self.duration(m) for m in self.meetings)
        unt = sum(self.untimed_sec(m) for m in self.meetings)
        timed_min = max(1e-9, (dur - unt) / 60)
        st.update(untimed_frac=round(unt / max(dur, 1e-9), 3), hours_timed=round((dur - unt) / 3600, 2),
                  turns_per_timed_min=round(st["turns"] / timed_min, 2),
                  backchannels_per_timed_min=round(st["backchannels"] / timed_min, 2),
                  speakers_per_meeting=[len(self.words[m]) for m in self.meetings])
        return st


# --------------------------------------------------------------------------- recipe hook
def recipe_data(cfg: dict, split: str) -> list[dict]:
    """``data: {icsi: {...}}`` for ``audioforge.train.load_data``; the same keys as ``ami.recipe_data``
    (root, mode turn|diar|asr, train_split, val_split, meetings, n_meetings, label keys, window/lead/trail/
    min_trail/hop, asr_min_sec/asr_max_sec, n_train/n_val, seed) plus exclude_untimed (true) and
    drop_overfull (true: drop windows with more active speakers than the 4 target columns), ext_tracks (turn
    mode: cached external-diarizer tracks from data/icsi/cache/sortformer/<split>, see attach_ext_tracks).
    With data.synthetic.n_<split> a synthetic stand-in is returned (hermetic smoke tests)."""
    d = cfg["data"]
    ic = dict(d["icsi"])
    mode = ic.get("mode", "turn")
    if f"n_{split}" in (d.get("synthetic") or {}):
        return ami.recipe_data({**cfg, "data": {**d, "ami": {"mode": mode}}}, split)
    sname = ic.get("train_split", "train") if split == "train" else ic.get("val_split", "dev")
    key = "train" if split == "train" else "val"
    meetings = (ic.get("meetings") or {}).get(key)
    if not meetings:
        meetings = subset({sname: (ic.get("n_meetings") or {}).get(key, len(DEFAULT_MEETINGS[sname]))})[sname]
    labels = {k: ic[k] for k in LABEL_DEFAULTS if k in ic}
    ds = ICSI(meetings, ic.get("root"), exclude_untimed=ic.get("exclude_untimed", True),
              drop_overfull=ic.get("drop_overfull", True), **labels)
    if mode == "diar":
        data = ds.diar(ic.get("window_sec", 20.0), ic.get("hop_sec", 10.0))
    elif mode == "turn":
        data = ds.turn_examples(ic.get("window_sec", 20.0), ic.get("lead_sec", 4.0), ic.get("trail_sec", 2.0),
                                ic.get("min_trail", 1.0))
    else:
        data = ds.asr(ic.get("asr_min_sec", 1.0), ic.get("asr_max_sec", 15.0))
    cap = ic.get(f"n_{split}")
    if cap and cap < len(data):
        idx = sorted(random.Random(ic.get("seed", 0)).sample(range(len(data)), cap))
        data = [data[i] for i in idx]
    ext = ""
    if mode == "turn" and ic.get("ext_tracks"):
        ext = " " + json.dumps(attach_ext_tracks(data, ic["ext_tracks"], sname, ic.get("root")))
    log.info(f"[icsi] {mode}/{split}: {len(meetings)} meetings -> {len(data)} examples {json.dumps(ds.filtered.get(mode))}"
          f"{ext}")
    return data


def attach_ext_tracks(data: list[dict], spec, split: str, root=None) -> dict:
    """ami.attach_ext_tracks (same spec: offline | stream | {source, dir, require, fallback}) with the cache dir
    defaulting to ICSI's own (<root or data/icsi>/cache/sortformer/<split>, scripts/research/make_sortformer_tracks.py
    --dataset icsi) instead of AMI's."""
    from . import ext_tracks as xt
    spec = {"source": spec} if isinstance(spec, str) else dict(spec)
    if not spec.get("dir"):
        spec["dir"] = str(xt.cache_dir(root, split, dataset="icsi"))
    return ami.attach_ext_tracks(data, spec, split, root)
