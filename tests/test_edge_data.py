"""Edge cases for audioforge.data: manifests, RTTM parsing, WAV loading, collation."""
import json

import numpy as np
import pytest
import torch

from audioforge.data import (
    Collate,
    load_wav,
    prompt_specials,
    read_manifest,
    rttm_to_frames,
    save_wav,
    synthetic_dataset,
)
from audioforge.metrics import frame_der
from audioforge.tokenizer import CharTokenizer


@pytest.fixture(autouse=True, scope="module")
def _threads_1():
    """1 torch thread for this module only (restored after it, not at import time for the session)."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_collate_custom_frame_key_trains_and_evaluates():
    from audioforge.model import SpeechModel
    from audioforge.train import Trainer
    data = synthetic_dataset("multitask", 4, seed=0)
    for ex in data:
        ex["speech"] = ex["vad"].copy()
    batch = Collate()(data)
    assert "speech" in batch and torch.equal(batch["speech"], batch["vad"])
    assert torch.equal(batch["speech_len"], batch["vad_len"])
    cfg = {"preprocessor": {"n_mels": 80},
           "encoder": {"d_model": 32, "n_layers": 1, "n_heads": 2, "subsampling_channels": 8},
           "heads": {"vad": {"type": "frame", "key": "vad"}, "sp": {"type": "frame", "key": "speech"}}}
    torch.manual_seed(0)
    m = SpeechModel(cfg)
    out = m.train()(batch)
    out["loss"].backward()
    assert "loss_sp" in out and m.heads["sp"].net.weight.grad is not None
    assert "acc_sp" in Trainer(m, {"trainer": {"device": "cpu"}}).evaluate(data)


def test_rttm_keeps_more_than_four_speakers(tmp_path):
    p = tmp_path / "f.rttm"
    p.write_text("".join(f"SPEAKER f 1 {i * 1.6:.2f} 1.60 <NA> <NA> s{i} <NA> <NA>\n" for i in range(5)))
    ref = rttm_to_frames(p, 100)
    assert ref.shape == (100, 5) and ref.any(1).sum() == 100 and ref[:, 4].sum() >= 20
    truth = np.zeros((100, 5), np.float32)
    for i in range(5):
        truth[i * 20:(i + 1) * 20, i] = 1
    assert frame_der(torch.tensor(truth[:, :4]), torch.tensor(ref)) == pytest.approx(0.2, abs=0.02)
    with pytest.warns(UserWarning, match="speakers"):
        assert rttm_to_frames(p, 100, max_spks=4).shape == (100, 4)
    # <= 4 speakers keeps the historical 4-column layout
    p.write_text("SPEAKER f 1 0.00 0.80 <NA> <NA> a <NA> <NA>\n")
    assert rttm_to_frames(p, 20).shape == (20, 4)


def test_read_manifest_rttm_many_speakers_keeps_full_reference(tmp_path):
    save_wav(tmp_path / "f.wav", np.zeros(16000 * 8, np.float32))
    (tmp_path / "f.rttm").write_text(
        "".join(f"SPEAKER f 1 {i * 1.6:.2f} 1.60 <NA> <NA> s{i} <NA> <NA>\n" for i in range(5)))
    (tmp_path / "m.json").write_text(json.dumps({"audio_filepath": str(tmp_path / "f.wav"),
                                                 "rttm_filepath": str(tmp_path / "f.rttm")}))
    with pytest.warns(UserWarning, match="speakers"):
        ex = read_manifest(tmp_path / "m.json")[0]
    assert ex["spk_targets"].shape[1] == 4  # training targets stay at the model's slot count
    assert ex["spk_targets_full"].shape[1] == 5 and ex["spk_targets_full"][:, 4].sum() > 0


def test_read_manifest_honours_offset_and_duration(tmp_path):
    x = np.random.default_rng(0).uniform(-.1, .1, 16000 * 10).astype(np.float32)
    save_wav(tmp_path / "long.wav", x)
    (tmp_path / "s.rttm").write_text("SPEAKER long 1 3.00 0.80 <NA> <NA> a <NA> <NA>\n")
    lines = [{"audio_filepath": str(tmp_path / "long.wav"), "offset": 3.0, "duration": 1.5, "text": "go",
              "rttm_filepath": str(tmp_path / "s.rttm")},
             {"audio_filepath": str(tmp_path / "long.wav"), "duration": 1000, "text": "go",
              "rttm_filepath": str(tmp_path / "s.rttm")}]  # no offset: placeholder duration, whole file
    (tmp_path / "m.json").write_text("\n".join(map(json.dumps, lines)))
    seg, whole = read_manifest(tmp_path / "m.json")
    assert len(seg["audio"]) == 24000 and len(whole["audio"]) == 160000
    ref = np.clip(np.round(x * 32767), -32768, 32767) / 32768
    assert np.allclose(seg["audio"], ref[48000:72000], atol=1e-4)
    assert seg["spk_targets"][:10, 0].sum() == 10  # RTTM times shifted by the offset
    assert whole["spk_targets"][:30, 0].sum() == 0 and whole["spk_targets"][38:47, 0].sum() == 9
    assert len(load_wav(tmp_path / "long.wav", offset=9.5)) == 8000


def test_collate_mixed_keys_is_order_independent(tmp_path):
    tok = CharTokenizer.train(["go og"], specials=prompt_specials(["en", "de"]))
    asr = dict(audio=np.zeros(1600, np.float32), text="go")
    ast = dict(audio=np.zeros(1600, np.float32), text="og", prompt="<|en|><|translate|><|de|><|pnc|>")
    for order in ([asr, ast], [ast, asr]):
        with pytest.raises(ValueError, match="prompt"):
            Collate(tok)(order)
    # read_manifest gives every line a prompt when any line asks for one
    save_wav(tmp_path / "a.wav", np.zeros(1600, np.float32))
    lines = [{"audio_filepath": str(tmp_path / "a.wav"), "text": "go"},
             {"audio_filepath": str(tmp_path / "a.wav"), "text": "og", "task": "ast", "target_lang": "de"}]
    (tmp_path / "m.json").write_text("\n".join(map(json.dumps, lines)))
    data = read_manifest(tmp_path / "m.json")
    assert data[0]["prompt"] == "<|en|><|transcribe|><|en|><|pnc|>"
    assert data[1]["prompt"] == "<|en|><|translate|><|de|><|pnc|>"
    for order in (data, data[::-1]):
        assert Collate(tok)(order)["prompt"].shape[0] == 2


def test_load_wav_reads_every_pcm_float_and_truncated_wav(tmp_path):
    sf = pytest.importorskip("soundfile")
    x = 0.5 * np.sin(np.arange(16000) / 5)
    for sub in ("PCM_16", "PCM_24", "FLOAT", "DOUBLE"):
        p = tmp_path / f"{sub}.wav"
        sf.write(p, x, 16000, subtype=sub)
        y = load_wav(p)
        assert len(y) == 16000 and y.dtype == np.float32 and np.allclose(y, x, atol=1e-3), sub
    p = tmp_path / "PCM_16.wav"
    with open(p, "r+b") as f:
        f.truncate(p.stat().st_size - 1)
    assert len(load_wav(p)) == 15999


def test_shared_rttm_is_split_by_file_id(tmp_path):
    for r in ("rec1", "rec2"):
        save_wav(tmp_path / f"{r}.wav", np.zeros(16000 * 4, np.float32))
    (tmp_path / "all.rttm").write_text("SPEAKER rec1 1 0.00 1.00 <NA> <NA> alice <NA> <NA>\n"
                                       "SPEAKER rec2 1 2.00 1.00 <NA> <NA> bob <NA> <NA>\n")
    (tmp_path / "m.json").write_text("\n".join(
        json.dumps({"audio_filepath": str(tmp_path / f"{r}.wav"), "rttm_filepath": str(tmp_path / "all.rttm")})
        for r in ("rec1", "rec2")))
    rec1, rec2 = read_manifest(tmp_path / "m.json")
    assert rec1["spk_targets"].sum(0).tolist() == [13.0, 0.0, 0.0, 0.0]
    assert rec2["spk_targets"].sum(0).tolist() == [13.0, 0.0, 0.0, 0.0]
    assert rec2["spk_targets"][25:38, 0].sum() == 13
    # a per-file RTTM whose id differs from the wav stem still works
    (tmp_path / "one.rttm").write_text("SPEAKER other_id 1 0.00 1.00 <NA> <NA> a <NA> <NA>\n")
    assert rttm_to_frames(tmp_path / "one.rttm", 50, uniq_id="rec1")[:, 0].sum() == 13
    with pytest.raises(ValueError, match="rec3"):
        rttm_to_frames(tmp_path / "all.rttm", 50, uniq_id="rec3")
    (tmp_path / "bad.rttm").write_text("SPEAKER rec1 1 0.00\n")
    with pytest.raises(ValueError, match="bad.rttm"):
        rttm_to_frames(tmp_path / "bad.rttm", 50)
