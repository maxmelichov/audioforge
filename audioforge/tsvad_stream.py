"""Live target-speaker activity for the turn head: ``serve --turn-input tsvad`` (research/IMPROVEMENTS.md section 1).

The TS-VAD head (``heads.tsvad.TSVADHead``, research/IMPROVE_115M.md Part A) reads block 4 of the ASR pass that the
server already runs (no extra encoder pass, no diarizer on the turn path) and, given the user's voice print, gives
per 80 ms frame [P(target), P(other)]. ``TSVADTrack`` keeps one session's head state and voice print:

* ``feed(block4 (1, n, D), vad (n,))`` -> (n, 2) probabilities, one per ASR frame, in order. The turn head is then fed
  exactly what the offline benchmark fed it (scripts/research/tsvad.py ``binding_inputs``): primary activity
  P(target), columns [P(target), P(other), 0, 0], primary column 0.
* the voice print (a unit-norm 192-d block-4 speaker-head embedding, ``heads.audio.SpeakerHead`` via
  ``enrollment.ColumnEmbedder``) comes from one of
  - ``set_print(e)``: a stored print (the product's enrolment; ``{"type": "enroll", "embedding": [...]}``),
  - ``arm(frame)``: collect the block-4 frames of the next ``print_s`` seconds of speech (served VAD > 0.5) from
    frame ``frame`` on and embed them (``after_agent_arm``: armed by every ``agent_end``; ``explicit`` without an
    embedding: armed by ``enroll``); ``dominant``: armed at frame 0 (the session's first voice).
  Until a print exists the head runs with its learned "no enrollment" vector, i.e. as a plain VAD (as trained). A new
  print is swapped in mid-stream (the GRU state is kept, ``TSVADHead.set_enrollment``). ``refresh_s`` > 0 keeps
  collecting after the first print and re-embeds over the most recent ``print_s`` seconds of speech the head assigns
  to the target (P(target) > 0.5 and P(other) < 0.5) every ``refresh_s`` seconds of such speech (research
  experiment 4).

``voiceprint(model, audio)`` computes a print offline from raw audio exactly as the benchmark's voice prints were
made (a separate block-4 pass over the clip, then the speaker head).
"""
from __future__ import annotations

import numpy as np
import torch

TAP = 3  # 0-based encoder block read by the TS-VAD head and the speaker head (block 4)
FRAME_S = 0.08


def load_tsvad(path, d_model: int = 512):
    """runs/tsvad_*.pt (scripts/research/tsvad.py train) -> TSVADHead (eval, cpu)."""
    from .heads.tsvad import TSVADHead
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = {k: v for k, v in ck["cfg"].items() if k not in ("type", "from_layers", "weight", "enroll_embedder")}
    h = TSVADHead(d_model, **c)
    h.load_state_dict(ck["state_dict"])
    return h.eval()


def embed_frames(spk_head, frames: np.ndarray) -> np.ndarray:
    """Block-4 frames (T, D) -> unit-norm voice print (E,) by the speaker head (enrollment.ColumnEmbedder)."""
    from .enrollment import ColumnEmbedder
    f = np.asarray(frames, np.float32)
    return ColumnEmbedder(spk_head)(f[None], np.ones((1, len(f)), bool))[0]


@torch.no_grad()
def voiceprint(model, audio: np.ndarray, att=(70, 1)) -> np.ndarray:
    """A voice print from raw 16 kHz audio (the benchmark's recipe: block-4 features of the clip, speaker head)."""
    from .modules.fastconformer import chunked_attention_mask
    enc = model.encoder
    dev = next(model.parameters()).device
    x, lens = model._pad([np.asarray(audio, np.float32)])
    feats, flen = model.preprocessor(x.to(dev), lens.to(dev))
    h, hl = enc.pre_encode(feats, flen)
    if enc.xscale is not None:
        h = h * enc.xscale
    T = h.shape[1]
    am = chunked_attention_mask(T, hl, list(att))
    pm = torch.arange(T, device=h.device)[None] >= hl[:, None]
    for li in range(model.layer_tap.get("spk", [TAP])[0] + 1):
        h, _, _ = enc.layers[li](h, am, pm)
    return embed_frames(model.heads["spk"], h[0, : int(hl[0])].float().cpu().numpy())


class TSVADTrack:
    def __init__(self, head, spk_head, print_s: float = 5.0, refresh_s: float = 0.0, vad_thr: float = 0.5,
                 min_print_s: float = 1.0):
        self.head, self.spk = head, spk_head
        self.need = max(1, int(round(print_s / FRAME_S)))
        self.min_need = min(self.need, max(1, int(round(min_print_s / FRAME_S))))
        self.refresh = int(round(refresh_s / FRAME_S)) if refresh_s else 0
        self.vad_thr = float(vad_thr)
        self.state = head.init_stream(None)
        self.n = 0  # frames processed
        self.armed_at: int | None = None  # collecting speech frames from this frame on
        self.buf: list[np.ndarray] = []  # collected block-4 frames (1, D)
        self.print: np.ndarray | None = None
        self.print_t: float | None = None  # audio time (frame end) the current print was taken
        self.n_prints = 0
        self.since_refresh = 0
        self.events: list[dict] = []  # {"t", "seconds", "source"} per new print (read by the server)

    @property
    def enrolled(self) -> bool:
        return self.print is not None

    def set_print(self, e, source: str = "explicit"):
        e = np.asarray(e, np.float32).reshape(-1)
        e = e / max(float(np.linalg.norm(e)), 1e-8)
        self.print = e
        self.head.set_enrollment(self.state, e)
        self.print_t = self.n * FRAME_S
        self.n_prints += 1
        self.events.append({"t": round(self.print_t, 3), "seconds": 0.0, "source": source})

    def arm(self, frame: int):
        """Collect the next print_s seconds of speech from ``frame`` on (frames already processed are not revisited:
        the arm is applied at the audio position it arrived at)."""
        self.armed_at = max(int(frame), 0)
        self.buf = []

    def _take_print(self, frames, source, v: int):
        e = embed_frames(self.spk, np.concatenate(frames, 0))
        self.print = e
        self.head.set_enrollment(self.state, e)  # applies from frame v + 1 on
        self.print_t = (v + 1) * FRAME_S
        self.n_prints += 1
        self.events.append({"t": round(self.print_t, 3), "seconds": round(len(frames) * FRAME_S, 2), "source": source})

    @torch.no_grad()
    def feed(self, block4: torch.Tensor, vad) -> np.ndarray:
        """block4 (1, n, D) frames of the ASR pass, vad (n,) served VAD -> (n, 2) [P(target), P(other)]."""
        out = []
        n = block4.shape[1]
        for j in range(n):
            f = block4[:, j: j + 1].float()
            p = self.head.step(f, self.state)[0, 0].numpy()  # the frame is scored with the print known before it
            v = self.n
            if self.armed_at is not None and v >= self.armed_at and float(vad[j]) > self.vad_thr:
                self.buf.append(f[0].numpy())
                if len(self.buf) >= self.need:
                    self._take_print(self.buf[: self.need], "arm", v)
                    self.armed_at, self.buf = None, []
            elif self.refresh and self.print is not None and self.armed_at is None:
                if p[0] > 0.5 and p[1] < 0.5:  # confidently the target alone
                    self.buf.append(f[0].numpy())
                    self.buf = self.buf[-self.need:]
                    self.since_refresh += 1
                    if self.since_refresh >= self.refresh and len(self.buf) >= self.min_need:
                        self._take_print(list(self.buf), "refresh", v)
                        self.since_refresh = 0
            out.append(p)
            self.n += 1
        return np.asarray(out, np.float32).reshape(n, 2)


class TSVADColumns:
    """``serve --diar-off`` (with ``--turn-input tsvad``): stands in for the per-session diarizer. Its rows are the TS-VAD
    track of the ASR frames already processed, [P(target), P(other), 0, 0], finalized with their 160 ms ASR chunk
    (``cfg`` chunk 2, right context 0), so frame.speakers / primary and the timeout policies read the enrolled user's
    activity and the Sortformer pass (2/3 of the server's compute) is not run."""

    def __init__(self, asr_stream, num_spks: int = 4):
        import types
        self.asr, self.S, self.done = asr_stream, int(num_spks), 0
        self.cfg = types.SimpleNamespace(chunk_len=2, chunk_right_context=0)

    def feed(self, samples=None, final: bool = False, skip: bool = False) -> torch.Tensor:
        tp = self.asr.tsvad_p
        n = len(tp)
        rows = np.zeros((max(0, n - self.done), self.S), np.float32)
        for i, v in enumerate(range(self.done, n)):
            rows[i, :2] = tp[v]
        self.done = n
        return torch.from_numpy(rows)
