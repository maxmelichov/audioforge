"""Dedicated open turn-detection models as per-frame end-of-turn score tracks (research/archive/BASELINES.md, "Turn detection").

Every baseline is turned into one score per 80 ms frame of an eot-bench v2 window (scripts/research/eval_stage1.py --bench v2),
so the SAME scorer (conversation.eot_outcomes / eval_stage1 cross-fitted 5 % FC operating points) applies to them and
to our detectors. Conventions (the repo's): frame t covers [0.08 t, 0.08 (t + 1)) s from the window start; a score on
frame t is a decision whose input ends inside frame t; the scorer's emission rule then says when it is available
(``emit(t) = t + 1`` for a decision taken on audio up to the end of frame t; a streaming model adds its chunking).
Model compute time is NOT in the emission rule (as for our rows); it is measured and reported separately.

What is wrapped (all run offline on the window audio, causally: nothing reads audio after the decision time):

* ``SileroVAD``: Silero VAD v5 ONNX (the model Pipecat and LiveKit's Silero plugin ship), 512-sample (32 ms) chunks
  at 16 kHz with the model's 64-sample context, via the ``silero-vad`` package's ``OnnxWrapper``.
* ``pipecat_vad``: Pipecat's ``VADAnalyzer`` state machine (QUIET / STARTING / SPEAKING / STOPPING; confidence 0.7,
  start 0.2 s, stop 0.2 s = 6 chunks each; the min_volume gate is not modelled).
* ``livekit_vad``: LiveKit's Silero plugin endpointing (activation 0.5, deactivation 0.35, min speech 0.05 s,
  min silence 0.55 s): speech-end times and the end-of-speech events.
* ``SmartTurn``: pipecat-ai/smart-turn v3.x ONNX (Whisper-tiny encoder + linear head): last <= 8 s of audio,
  left-padded with zeros to 8 s, Whisper log-mel (transformers ``WhisperFeatureExtractor(chunk_length=8)``,
  do_normalize) -> P(complete). Exactly smart-turn's ``inference.py`` / Pipecat's ``LocalSmartTurnAnalyzerV3``.
* ``LiveKitText``: LiveKit's text turn detector through the plugin's OWN runner classes
  (``livekit.plugins.turn_detector.english._EUORunnerEn`` / ``multilingual._EUORunnerMultilingual``: chat-template
  formatting, tokenizer truncation to 128 tokens, ONNX q8 session); only the session is rebuilt with 2 threads.
* ``LiveKitAudio``: LiveKit's audio end-of-turn model ``turn-detector-v1-mini`` (``livekit.local_inference.EOT``, the
  in-process model ``livekit.agents.inference.TurnDetector`` falls back to): last 1.2 s of int16 PCM -> P(turn ended).
* ``rnnt_frame_decode``: greedy RNNT decoding of a cache-aware streaming FastConformer, frame by frame, recording the
  tokens emitted on each encoder frame and the posterior of chosen token ids (Parakeet-Realtime-EOU's <EOU>). The
  encoder runs once over the window with its chunked_limited attention mask, which is what cache-aware streaming
  computes chunk by chunk (checked in tests/test_baselines_turn.py against ``model.StreamingSession``).

Score-track builders: ``held_scores`` (a model probability computed at a trigger and held until speech resumes,
0 elsewhere) and ``silence_frames`` (a VAD silence timeout as a score, in frames).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 0.08
CHUNK = 512  # Silero VAD window at 16 kHz (32 ms)
CHUNK_SEC = CHUNK / SR


def frame_of_time(tau: float) -> int:
    """The frame whose span contains decision time ``tau`` (s): the first frame t with (t + 1) * 0.08 >= tau, so the
    scorer's emission t + 1 is the first frame boundary at or after the decision."""
    return max(0, int(math.ceil(round(tau / FRAME, 9))) - 1)


# --------------------------------------------------------------------------- Silero VAD v5
class SileroVAD:
    """Silero VAD (ONNX) over 32 ms chunks; ``probs(audio)`` -> (n_chunks,) speech probabilities (stateful, from a
    reset state at the window start; a trailing partial chunk is dropped)."""

    def __init__(self, onnx_path: str | Path):
        from silero_vad.utils_vad import OnnxWrapper
        self.path = str(onnx_path)
        self.model = OnnxWrapper(self.path, force_onnx_cpu=True)

    def probs(self, audio: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(np.asarray(audio, np.float32))
        self.model.reset_states()
        n = len(x) // CHUNK
        out = np.empty(n, np.float32)
        for j in range(n):
            out[j] = float(self.model(x[j * CHUNK:(j + 1) * CHUNK], SR))
        return out



class SileroStream:
    """Silero VAD v5 ONNX for live use (audioforge.serve hybrid_silero / hybrid_dyn): one ORT session shared by every
    stream, the recurrent state and the 64-sample context per stream (``new_state``). ``step(state, chunk)`` with one
    512-sample 16 kHz chunk -> speech probability; a stream of steps from ``new_state()`` equals
    ``SileroVAD.probs`` (the silero-vad package's ``OnnxWrapper``) on the same audio."""

    CONTEXT = 64

    def __init__(self, onnx_path: str | Path, threads: int = 1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        self.path = str(onnx_path)
        self.session = ort.InferenceSession(self.path, sess_options=so, providers=["CPUExecutionProvider"])
        self._sr = np.array(SR, np.int64)

    def new_state(self) -> list:
        return [np.zeros((2, 1, 128), np.float32), np.zeros((1, self.CONTEXT), np.float32)]

    def step(self, state: list, chunk: np.ndarray) -> float:
        x = np.concatenate([state[1], np.asarray(chunk, np.float32).reshape(1, CHUNK)], 1)
        out, state[0] = self.session.run(None, {"input": x, "state": state[0], "sr": self._sr})
        state[1] = x[:, -self.CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])


class PipecatVADState:
    """Pipecat ``VADAnalyzer._run_analyzer`` as an incremental state machine (one 32 ms chunk per ``step``): states
    1 QUIET, 2 STARTING, 3 SPEAKING, 4 STOPPING; speech starts after ``start_secs`` of confidence >= ``confidence``
    and stops after ``stop_secs`` below it. ``step(conf)`` -> (state, stopped, started): ``stopped`` = the state just
    became QUIET from SPEAKING/STOPPING (the "user stopped speaking" trigger), ``started`` = it just became SPEAKING.
    ``speaking`` = STARTING or SPEAKING (``speech_chunks_pipecat``: STOPPING counts as silence). ``pipecat_vad`` runs
    it over a whole track; audioforge.serve runs it live (hybrid_silero / hybrid_dyn)."""

    QUIET, STARTING, SPEAKING, STOPPING = 1, 2, 3, 4

    def __init__(self, confidence: float = 0.7, start_secs: float = 0.2, stop_secs: float = 0.2):
        self.confidence = confidence
        self.start_n = round(start_secs / CHUNK_SEC)
        self.stop_n = round(stop_secs / CHUNK_SEC)
        self.state, self.sc, self.pc = self.QUIET, 0, 0

    @property
    def speaking(self) -> bool:
        return self.state in (self.STARTING, self.SPEAKING)

    def step(self, conf: float) -> tuple[int, bool, bool]:
        QUIET, STARTING, SPEAKING, STOPPING = self.QUIET, self.STARTING, self.SPEAKING, self.STOPPING
        st, prev = self.state, self.state
        if conf >= self.confidence:
            if st == QUIET:
                st, self.sc = STARTING, 1
            elif st == STARTING:
                self.sc += 1
            elif st == STOPPING:
                st, self.pc = SPEAKING, 0
        else:
            if st == STARTING:
                st, self.sc = QUIET, 0
            elif st == SPEAKING:
                st, self.pc = STOPPING, 1
            elif st == STOPPING:
                self.pc += 1
        if st == STARTING and self.sc >= self.start_n:
            st, self.sc = SPEAKING, 0
        if st == STOPPING and self.pc >= self.stop_n:
            st, self.pc = QUIET, 0
        self.state = st
        return st, st == QUIET and prev in (SPEAKING, STOPPING), st == SPEAKING and prev in (QUIET, STARTING)


def pipecat_vad(probs: np.ndarray, confidence: float = 0.7, start_secs: float = 0.2, stop_secs: float = 0.2) -> dict:
    """Pipecat ``VADAnalyzer._run_analyzer`` on per-chunk confidences (one chunk per call, so its end-of-call checks
    run after every chunk). Returns per-chunk ``state`` (1 QUIET, 2 STARTING, 3 SPEAKING, 4 STOPPING), ``stops``
    (chunks after which SPEAKING/STOPPING became QUIET: the "user stopped speaking" trigger) and ``starts`` (chunks
    after which the state became SPEAKING). The state machine is ``PipecatVADState``."""
    sm = PipecatVADState(confidence, start_secs, stop_secs)
    state = np.empty(len(probs), np.int8)
    stops, starts = [], []
    for j, c in enumerate(probs):
        st, stopped, started = sm.step(float(c))
        if stopped:
            stops.append(j)
        if started:
            starts.append(j)
        state[j] = st
    return dict(state=state, stops=np.array(stops, np.int64), starts=np.array(starts, np.int64))


def livekit_vad(probs: np.ndarray, activation: float = 0.5, deactivation: float | None = None,
                min_speech: float = 0.05, min_silence: float = 0.55) -> dict:
    """LiveKit Silero plugin endpointing on per-chunk probabilities: speech starts after ``min_speech`` of p >=
    activation, ends after ``min_silence`` of p < deactivation (default activation - 0.15). Returns ``speaking``
    (per chunk, the plugin's speaking flag), ``eos`` (chunks at which END_OF_SPEECH is emitted), ``speech_end``
    (for each eos, the time (s) the silence began = LiveKit's last_speaking_time) and ``sos`` (START_OF_SPEECH
    chunks)."""
    deact = max(activation - 0.15, 0.01) if deactivation is None else deactivation
    n_sp, n_si = max(1, round(min_speech / CHUNK_SEC)), max(1, round(min_silence / CHUNK_SEC))
    speaking, cnt_sp, cnt_si = False, 0, 0
    flag = np.zeros(len(probs), bool)
    eos, sos, ends = [], [], []
    for j, p in enumerate(probs):
        if not speaking:
            cnt_sp = cnt_sp + 1 if p >= activation else 0
            if cnt_sp >= n_sp:
                speaking, cnt_si = True, 0
                sos.append(j)
        else:
            cnt_si = cnt_si + 1 if p < deact else 0
            if cnt_si >= n_si:
                speaking, cnt_sp = False, 0
                eos.append(j)
                ends.append((j + 1 - cnt_si) * CHUNK_SEC)
        flag[j] = speaking
    return dict(speaking=flag, eos=np.array(eos, np.int64), sos=np.array(sos, np.int64), speech_end=np.array(ends))


def speech_chunks_pipecat(v: dict) -> np.ndarray:
    """Per-chunk 'user is speaking' of the Pipecat VAD (STARTING or SPEAKING; STOPPING counts as silence)."""
    return np.isin(v["state"], (2, 3))


def silence_frames(speech_chunk: np.ndarray, T: int) -> np.ndarray:
    """Silence-timeout score on the frame grid from a per-chunk speech flag: at frame t, the time (in frames) from
    the end of the last speech chunk to the end of frame t; 0 before the first speech chunk and on frames whose end
    lies inside speech (a deployable timeout arms at the first speech)."""
    ends = (np.nonzero(speech_chunk)[0] + 1) * CHUNK_SEC  # end times of speech chunks
    out = np.zeros(T, np.float32)
    if not len(ends):
        return out
    fe = (np.arange(T) + 1) * FRAME
    k = np.searchsorted(ends, fe + 1e-9, side="right") - 1  # last speech chunk ending at or before the frame end
    ok = k >= 0
    # the chunk containing the frame end is speech -> 0
    cj = np.minimum((fe / CHUNK_SEC - 1e-9).astype(np.int64), len(speech_chunk) - 1)
    in_speech = speech_chunk[np.clip(cj, 0, len(speech_chunk) - 1)] & (cj < len(speech_chunk))
    out[ok] = ((fe[ok] - ends[k[ok]]) / FRAME).astype(np.float32)
    out[in_speech] = 0.0
    return np.round(out, 4)


def held_scores(T: int, triggers: list[tuple[float, float]], resumes: np.ndarray) -> np.ndarray:
    """Score track of a model evaluated at triggers: ``triggers`` [(decision time s, value)], ``resumes`` sorted
    decision times (s) at which speech resumes. The value holds from the trigger's frame until the frame before the
    next resume (a resume in the same frame as a later trigger does not cancel it); 0 elsewhere."""
    out = np.zeros(T, np.float32)
    resumes = np.sort(np.asarray(resumes, np.float64))
    for tau, val in triggers:
        a = frame_of_time(tau)
        if a >= T:
            continue
        k = np.searchsorted(resumes, tau, side="right")
        b = T if k >= len(resumes) else frame_of_time(resumes[k])
        b = max(b, a + 1)
        out[a:min(b, T)] = val
    return out


def frame_silence_triggers(act: np.ndarray, k: int = 3, thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """On a frame-level activity track: (trigger frames = the frame on which a silent run after activity reaches k
    frames, resume frames = the first active frame after each trigger)."""
    a = np.asarray(act) > thr
    trig, res = [], []
    run, seen, armed = 0, False, False
    for t in range(len(a)):
        if a[t]:
            if armed:
                res.append(t)
                armed = False
            run, seen = 0, True
        elif seen:
            run += 1
            if run == k:
                trig.append(t)
                armed = True
    return np.array(trig, np.int64), np.array(res, np.int64)


def held_frames(T: int, trig: np.ndarray, vals, res: np.ndarray) -> np.ndarray:
    """held_scores on the frame grid: value from each trigger frame until the next resume frame (exclusive)."""
    out = np.zeros(T, np.float32)
    res = np.sort(np.asarray(res, np.int64))
    for t, v in zip(trig, vals):
        k = np.searchsorted(res, t, side="right")
        b = T if k >= len(res) else int(res[k])
        out[int(t):b] = v
    return out


# --------------------------------------------------------------------------- Pipecat smart-turn v3
class SmartTurn:
    """pipecat-ai/smart-turn v3.x ONNX. ``predict(audio)`` = P(complete) of the last <= 8 s (zero-padded on the left),
    as smart-turn's inference.py and Pipecat's LocalSmartTurnAnalyzerV3."""

    def __init__(self, onnx_path: str | Path, threads: int = 2):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.path = str(onnx_path)
        self.session = ort.InferenceSession(self.path, sess_options=so, providers=["CPUExecutionProvider"])
        self.fe = WhisperFeatureExtractor(chunk_length=8)
        self.n = 8 * SR

    def features(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, np.float32)[-self.n:]
        if len(x) < self.n:
            x = np.pad(x, (self.n - len(x), 0))
        f = self.fe(x, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=self.n,
                    truncation=True, do_normalize=True).input_features
        return f.astype(np.float32)  # (1, 80, 800)

    def predict(self, audio: np.ndarray) -> float:
        return float(self.session.run(None, {"input_features": self.features(audio)})[0].reshape(-1)[0])


# --------------------------------------------------------------------------- LiveKit
class LiveKitText:
    """LiveKit's text end-of-utterance model via the plugin's own runner (``model`` 'en' = EnglishModel v1.2.2-en,
    'multilingual' = MultilingualModel v0.4.1-intl). ``predict(text)`` = P(end of utterance) of a chat context with
    one user message (empty text -> 0.0: LiveKit runs no EOU without a transcript). Results are cached per text."""

    def __init__(self, model: str = "en", threads: int = 2):
        import onnxruntime as ort
        if model == "en":
            from livekit.plugins.turn_detector.english import _EUORunnerEn as R
        else:
            from livekit.plugins.turn_detector.multilingual import _EUORunnerMultilingual as R
        from huggingface_hub import hf_hub_download
        self.runner = R()
        self.runner.initialize()
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.add_session_config_entry("session.dynamic_block_base", "4")
        self.runner._session = ort.InferenceSession(self.runner._session._model_path,
                                                    providers=["CPUExecutionProvider"], sess_options=so)
        self.revision = R.model_revision()
        langs = json.load(open(hf_hub_download("livekit/turn-detector", "languages.json", revision=self.revision,
                                               local_files_only=True)))
        self.threshold = float((langs.get("en") or langs.get("english"))["threshold"])
        self.cache: dict[str, float] = {}
        self.calls, self.sec = 0, 0.0

    def predict(self, text: str) -> float:
        text = text.strip()
        if not text:
            return 0.0
        if text not in self.cache:
            t0 = time.perf_counter()
            out = json.loads(self.runner.run(json.dumps({"chat_ctx": [{"role": "user", "content": text}]}).encode()))
            self.sec += time.perf_counter() - t0
            self.calls += 1
            self.cache[text] = float(out["eou_probability"])
        return self.cache[text]


class LiveKitAudio:
    """LiveKit ``turn-detector-v1-mini`` (in-process audio EOT): P(turn ended) of the last 1.2 s (int16 PCM)."""

    THRESHOLD_EN = 0.36  # livekit.agents.inference.eot.languages.LOCAL_LANGUAGES["en"]

    def __init__(self):
        from livekit.local_inference import EOT, EOT_MAX_SAMPLES
        self.eot, self.n = EOT(), int(EOT_MAX_SAMPLES)

    def predict(self, audio: np.ndarray) -> float:
        x = np.clip(np.asarray(audio, np.float32)[-self.n:] * 32768.0, -32768, 32767).astype(np.int16)
        return float(self.eot.predict(x))


# --------------------------------------------------------------------------- RNNT frame-level decode (Parakeet EOU, our ASR)
def rnnt_frame_decode(model, audio: np.ndarray, head: str | None = None, watch_ids=(), att_context_size=None):
    """Greedy RNNT decode of one utterance, frame by frame (= model.StreamingSession._decode for a non-TDT head).

    Returns (tokens_per_frame: list[list[int]], logpost (T, len(watch_ids)) float64: per frame the max over the
    frame's symbol steps of log softmax P(id) for each watched id (log domain: the posteriors of a peaky RNNT
    underflow float32 far from an emission), emitted (T, len(watch_ids)) bool)."""
    import torch
    head = head or model.primary
    h = model.heads[head]
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(audio, np.float32))[None]
        enc, elen, hidden = model.encode(x, torch.tensor([x.shape[1]]), att_context_size, return_hidden=True)
        e = enc if model.head_cfg[head].get("condition_on_speaker") else model.head_input(head, enc, hidden)
        f = e[0, : int(elen[0])]
        V1 = h.vocab_size + 1
        g, st = h.pred(torch.tensor([[h.blank]]), None)
        fe = h.joint.enc(f)
        pp = h.joint.pred(g)
        T = f.shape[0]
        toks = [[] for _ in range(T)]
        post = np.full((T, len(watch_ids)), -np.inf)
        emitted = np.zeros((T, len(watch_ids)), bool)
        w = list(watch_ids)
        for t in range(T):
            n_sym = 0
            while True:
                z = h.joint.out(fe[t][None, None] + pp)[0, 0][:V1]
                if w:
                    pr = torch.log_softmax(z.double(), -1)[w].numpy()
                    post[t] = np.maximum(post[t], pr)
                k = int(z.argmax())
                if k == h.blank:
                    break
                toks[t].append(k)
                if k in w:
                    emitted[t, w.index(k)] = True
                g, st = h.pred(torch.tensor([[k]]), st)
                pp = h.joint.pred(g)
                n_sym += 1
                if n_sym >= h.max_symbols:
                    break
    return toks, post, emitted
