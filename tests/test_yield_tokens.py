"""Conversational action tokens <HOLD> / <YIELD> in the RNNT vocabulary (research/archive/YIELD_TOKENS.md): the transcript
builder (datasets/ami.action_transcript), the vocabulary extension (audioforge/vocab.py: ids preserved, blank moved,
old logits unchanged), special-aware SentencePiece encoding, and the decision-score extraction / pause-site scoring
used by scripts/research/bench_yield_tokens.py."""
import numpy as np
import pytest
import torch

from audioforge.datasets.ami import action_transcript
from audioforge.model import SpeechModel
from audioforge.tokenizer import SentencePieceTokenizer
from audioforge.vocab import extend_model_vocab, extend_spm


def test_action_transcript_rules():
    # gaps: 0.1 (no), 0.3 (hold: >= hes_gap), 0.29 (no); running max end handles an overlapping short word
    words = [(0.0, 0.5, "so"), (0.6, 1.0, "we"), (1.3, 1.6, "could"), (1.89, 2.4, "maybe")]
    assert action_transcript(words) == "so we <HOLD> could maybe <YIELD>"
    # empty-text words count for the gaps but add no token
    assert action_transcript([(0.0, 0.5, "so"), (0.55, 0.7, ""), (1.0, 1.4, "yes")]) == "so <HOLD> yes <YIELD>"
    assert action_transcript([(0.0, 0.5, "so"), (0.55, 0.9, ""), (1.0, 1.4, "yes")]) == "so yes <YIELD>"
    # a long word that spans a later word's start: the gap is measured from the running max end
    assert action_transcript([(0.0, 2.0, "well"), (1.0, 1.2, "um"), (2.2, 2.5, "ok")]) == "well um ok <YIELD>"
    assert action_transcript([(0.0, 2.0, "well"), (1.0, 1.2, "um"), (2.31, 2.5, "ok")]) == "well um <HOLD> ok <YIELD>"
    assert action_transcript([]) == ""
    assert action_transcript([(0.0, 0.4, "yeah")], hold="[H]", yield_="[Y]") == "yeah [Y]"
    # the tokens are the same labels the benchmark uses: hes intervals of datasets.ami._turn
    from audioforge.datasets.ami import _turn
    t = _turn("A", words, 0.3, 1.0, 2)
    assert t["hes"] == [(1.0, 1.3)] and action_transcript(words).count("<HOLD>") == len(t["hes"])


def _spm_tok():
    texts = ["the cat sat on the mat", "a dog ran", "we could maybe do that", "so yes ok well um"] * 4
    return SentencePieceTokenizer.train(texts, vocab_size=40)


def test_extend_spm_keeps_ids_and_encodes_specials_inline():
    tok = _spm_tok()
    V = tok.vocab_size
    new = SentencePieceTokenizer(extend_spm(tok.model_bytes, ["<HOLD>", "<YIELD>"]), tok.specials + ["<HOLD>", "<YIELD>"])
    assert new.vocab_size == V + 2 and new.token_id("<HOLD>") == V and new.token_id("<YIELD>") == V + 1
    for s in ["the cat sat", "we could maybe do that", "so yes ok"]:
        assert new.encode(s) == tok.encode(s)  # every existing id unchanged
    ids = new.encode("the cat <HOLD> sat on <YIELD>")
    assert ids == tok.encode("the cat") + [V] + tok.encode("sat on") + [V + 1]  # no stray "▁" piece
    assert new.decode(ids) == tok.decode(tok.encode("the cat sat on"))  # specials dropped on decode
    with pytest.raises(ValueError):
        extend_spm(new.model_bytes, ["<HOLD>"])


CFG = {"preprocessor": {"n_mels": 40, "normalize": "fixed", "dither": 0.0},
       "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 8, "causal": True,
                   "att_context_size": [8, 1], "dropout": 0.0},
       "heads": {"rnnt": {"type": "rnnt", "pred_hidden": 32, "joint_hidden": 32}, "ctc": {"type": "ctc"}}}


def test_extend_model_vocab_moves_blank_and_keeps_old_logits():
    torch.manual_seed(0)
    tok = _spm_tok()
    m = SpeechModel(CFG, tok).eval()
    V = tok.vocab_size
    new = extend_model_vocab(m, ["<HOLD>", "<YIELD>"]).eval()
    h0, h1 = m.heads["rnnt"], new.heads["rnnt"]
    assert h0.blank == V and h1.blank == V + 2 and new.heads["ctc"].blank == V + 2
    assert set(new.vocab_extension["resized"]) == {"heads.rnnt.pred.embed.weight", "heads.rnnt.joint.out.2.weight",
                                                   "heads.rnnt.joint.out.2.bias", "heads.ctc.proj.weight",
                                                   "heads.ctc.proj.bias"}
    f = torch.randn(1, 5, 32)
    y = torch.tensor([[1, 2, 3]])
    g0, _ = h0.pred(h0.pred.prepend_sos(y))
    g1, _ = h1.pred(h1.pred.prepend_sos(y))
    assert torch.allclose(g0, g1, atol=1e-6)  # SOS row moved with the blank
    z0, z1 = h0.joint(f, g0), h1.joint(f, g1)
    assert torch.allclose(z0[..., :V], z1[..., :V], atol=1e-6) and torch.allclose(z0[..., V], z1[..., V + 2], atol=1e-6)
    # new tokens start rare: their bias is the lowest token bias, their weights the token mean
    b = h1.joint.out[2].bias
    assert torch.allclose(b[V:V + 2], b[:V].min().expand(2))
    # greedy decoding of the same audio is unchanged (the new tokens never win at init)
    audio = [np.random.default_rng(0).standard_normal(8000).astype(np.float32) * 0.1]
    assert new.transcribe(audio, head="rnnt") == m.transcribe(audio, head="rnnt")
    assert new.transcribe(audio, head="ctc") == m.transcribe(audio, head="ctc")
    # training on an action transcript works end to end (labels reach the new ids)
    from audioforge.data import Collate
    batch = Collate(new.tokenizer)([{"audio": audio[0], "text": "the cat <HOLD> sat <YIELD>"}])
    assert int(batch["text"][0, -1]) == V + 1 and (batch["text"][0] == V).any()
    new.train()
    loss = new({**batch})["loss"]
    assert torch.isfinite(loss)


def test_frame_decode_scores_new_tokens():
    """rnnt_frame_decode with the new ids as watch ids: per-frame log posteriors (<= 0, -inf only where no symbol
    step ran) and emitted flags consistent with the tokens per frame."""
    from audioforge.baselines.turn import rnnt_frame_decode
    torch.manual_seed(0)
    tok = _spm_tok()
    m = extend_model_vocab(SpeechModel(CFG, tok), ["<HOLD>", "<YIELD>"]).eval()
    V = tok.vocab_size
    audio = np.random.default_rng(1).standard_normal(16000).astype(np.float32) * 0.1
    toks, post, em = rnnt_frame_decode(m, audio, watch_ids=[V + 1, V])
    T = len(toks)
    assert post.shape == (T, 2) and em.shape == (T, 2) and np.all(post <= 1e-9)
    assert np.all(np.isfinite(post))  # every frame runs at least one symbol step
    for t in range(T):
        assert em[t, 0] == (V + 1 in toks[t]) and em[t, 1] == (V in toks[t])



def test_pause_site_stats():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("byt", Path(__file__).resolve().parents[1] / "scripts" / "research" / "bench_yield_tokens.py")
    byt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(byt)
    T = 40
    hes = np.zeros(T, np.float32)
    hes[10:13] = 1  # one within-turn pause
    onset, end = 4, 25
    emY = np.zeros(T, bool)
    emH = np.zeros(T, bool)
    emH[11] = True  # HOLD inside the pause: a hit
    emY[27] = True  # YIELD after the end: a hit (within 6 s)
    emY[6] = True  # YIELD inside the turn: a false yield
    st = byt.pause_site_stats([dict(hes=hes, onset_frame=onset, turn_end_frame=end, spk_act=np.zeros(T))],
                              [(emY, emH)], post_frames=75)
    assert st["pauses"] == 1 and st["hold_at_pause"] == 1 and st["yield_at_pause"] == 0
    assert st["ends"] == 1 and st["yield_at_end"] == 1 and st["hold_at_end"] == 0
    assert st["yield_emissions_in_turn"] == 1 and st["yield_emissions_post_end"] == 1


def test_window_action_transcript_all_speakers():
    from audioforge.datasets.ami import speaker_turns, window_action_transcript
    words = {"A": [(0.0, 0.4, "so"), (0.5, 0.9, "we"), (1.3, 1.7, "go"), (5.0, 5.3, "ok")],   # A: turn 1 (pause 0.4 -> HOLD), turn 2 (gap 3.3 s >= max_hold)
             "B": [(1.6, 1.9, "yeah"), (2.5, 2.9, "right"), (3.0, 3.4, "then")]}             # B: runs "yeah" | "right then", merged by the floor rule (0.6 s pause, nobody took the floor)
    turns = speaker_turns(words)
    # whole meeting: words by start time, HOLDs at the pause starts (0.9 A, 1.9 B), YIELDs at 1.7 (A: after 'yeah', which starts at 1.6), 3.4 (B), 5.3 (A)
    assert window_action_transcript(turns, 0.0, 6.0) == "so we <HOLD> go yeah <YIELD> <HOLD> right then <YIELD> ok <YIELD>"
    # window cut inside A's first turn: no HOLD across the boundary, its words before 1.0 dropped
    assert window_action_transcript(turns, 1.0, 6.0) == "go yeah <YIELD> <HOLD> right then <YIELD> ok <YIELD>"
    # window ending before B's turn ends: B gets no YIELD; A's second turn is outside
    assert window_action_transcript(turns, 0.0, 3.2) == "so we <HOLD> go yeah <YIELD> <HOLD> right then"
    # a token at the same time as a word start sorts before the word
    w2 = {"A": [(0.0, 0.5, "hi")], "B": [(0.5, 0.9, "yo")]}
    assert window_action_transcript(speaker_turns(w2), 0.0, 2.0) == "hi <YIELD> yo <YIELD>"


def test_emission_timing():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("byt", Path(__file__).resolve().parents[1] / "scripts" / "research" / "bench_yield_tokens.py")
    byt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(byt)
    T = 60
    hes = np.zeros(T, np.float32)
    hes[10:15] = 1  # a 0.4 s in-turn pause
    convs = [dict(hes=hes, onset_frame=4, turn_end_frame=25), dict(hes=np.zeros(T, np.float32), onset_frame=4, turn_end_frame=25)]
    last_word_end = [25 * 0.08 - 0.03, 25 * 0.08]  # label word ends (s)
    emit = lambda t: (t // 2 + 1) * 2  # noqa: E731  160 ms chunk rule
    # turn 1: an in-turn firing (frame 8) and the first end firing on frame 26 -> emission frame 28 = 2.24 s, lag 0.27 s
    # turn 2: fires on frame 24 (inside the last 160 ms of the word) -> emission frame 26 = 2.08 s, lag 0.08 s
    r = byt.emission_timing(convs, last_word_end, [np.array([8, 26, 40]), np.array([24])], emit)
    assert r["turns_fired_at_end"] == 2 and r["turns_fired_in_turn"] == 1
    assert abs(r["lag_p10_p50_p90_s"][1] - (0.27 + 0.08) / 2) < 1e-6
    assert r["share_within_160ms_of_word_end"] == 0.5 and r["share_before_word_end"] == 0.0
    assert r["median_in_turn_pause_s_on_fired_turns"] == 0.4 and r["n_in_turn_pauses_on_fired_turns"] == 1
    assert sum(r["hist_160ms"]["counts"]) == 2
    # no firing at all
    r0 = byt.emission_timing(convs, last_word_end, [np.array([], np.int64)] * 2, emit)
    assert r0["turns_fired_at_end"] == 0 and r0["lag_p10_p50_p90_s"] is None


def test_trainer_row_lr_trains_only_new_rows():
    """trainer.row_lr: the listed rows of the listed tensors get their own lr, the other rows of those tensors stay
    exactly fixed, and the rest of the model still trains."""
    from audioforge.data import Collate
    from audioforge.train import Trainer
    torch.manual_seed(0)
    tok = _spm_tok()
    m = extend_model_vocab(SpeechModel(CFG, tok), ["<HOLD>", "<YIELD>"])
    V = tok.vocab_size
    cfg = {**CFG, "trainer": {"max_steps": 3, "batch_size": 2, "lr": 1e-3, "device": "cpu", "warmup_steps": 1,
                              "row_lr": {"params": ["heads.rnnt.joint.out.2.weight", "heads.rnnt.joint.out.2.bias"],
                                         "rows": [V, V + 1], "lr": 0.1}}}
    tr = Trainer(m, cfg)
    w0 = m.heads["rnnt"].joint.out[2].weight.detach().clone()
    b0 = m.heads["rnnt"].joint.out[2].bias.detach().clone()
    lstm0 = next(m.heads["rnnt"].pred.lstm.parameters()).detach().clone()
    audio = np.random.default_rng(0).standard_normal(8000).astype(np.float32) * 0.1
    data = [{"audio": audio, "text": "the cat <HOLD> sat <YIELD>"}, {"audio": audio, "text": "a dog ran <YIELD>"}]
    tr.fit(data, collate=Collate(m.tokenizer))
    w1, b1 = m.heads["rnnt"].joint.out[2].weight.detach(), m.heads["rnnt"].joint.out[2].bias.detach()
    keep = [i for i in range(V + 3) if i not in (V, V + 1)]
    assert torch.equal(w1[keep], w0[keep]) and torch.equal(b1[keep], b0[keep])  # old rows and blank untouched
    assert not torch.equal(w1[V:V + 2], w0[V:V + 2]) and (b1[V:V + 2] - b0[V:V + 2]).abs().max() > 0.05
    assert not torch.equal(next(m.heads["rnnt"].pred.lstm.parameters()).detach(), lstm0)  # the rest still trains
    assert len(tr.opt.param_groups) == 2 and tr.opt.param_groups[-1]["initial_lr"] == 0.1
