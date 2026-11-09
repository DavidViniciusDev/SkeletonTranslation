#!/usr/bin/env python3
"""Testes de --seed, --label-smoothing, dos parâmetros de decodificação e da
média de checkpoints (tools/average_checkpoints.py).

Dados sintéticos e um T5 minúsculo criado do zero: não exige a base nem
download de pesos.

Uso:
    python3 tests/test_seed_ls_decoding.py
    (também compatível com: pytest tests/test_seed_ls_decoding.py)
"""
import argparse
import json
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from skeltrans.common.layout import INPUT_DIM  # noqa: E402
from skeltrans.training.generation import add_generation_args, generation_kwargs  # noqa: E402
from skeltrans.training.trainer import seed_everything  # noqa: E402


def _model(seed=0):
    from transformers import T5Config, T5ForConditionalGeneration
    from skeltrans.training.models import SLTModel
    torch.manual_seed(seed)
    cfg = T5Config(vocab_size=64, d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1,
                   num_heads=2, d_kv=16, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1)
    return SLTModel(T5ForConditionalGeneration(cfg), d_model=16, nhead=2, num_layers=1)


def _batch(B=3, T=20, L=7):
    g = torch.Generator().manual_seed(1)
    feats = torch.randn(B, T, INPUT_DIM, generator=g)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[1, 15:] = True
    labels = torch.randint(2, 64, (B, L), generator=g)
    labels[0, 5:] = -100                       # padding ignorado na perda
    return feats, mask, labels


# --------------------------------------------------------------------------- #
# label smoothing
# --------------------------------------------------------------------------- #
def test_label_smoothing_zero_is_t5_loss():
    m = _model().eval()
    feats, mask, labels = _batch()
    with torch.no_grad():
        base, _ = m(feats, mask, labels)
        zero, _ = m(feats, mask, labels, label_smoothing=0.0)
    assert torch.equal(base, zero)


def test_label_smoothing_matches_manual_ce():
    m = _model().eval()
    feats, mask, labels = _batch()
    with torch.no_grad():
        base, logits = m(feats, mask, labels)
        ls, _ = m(feats, mask, labels, label_smoothing=0.1)
        flat = logits.reshape(-1, logits.size(-1))
        # a CE manual sem suavização reproduz a do T5 (mesma média) ...
        assert torch.allclose(F.cross_entropy(flat, labels.reshape(-1), ignore_index=-100),
                              base, atol=1e-6)
        # ... e a suavizada é a do PyTorch com label_smoothing
        ref = F.cross_entropy(flat, labels.reshape(-1), ignore_index=-100,
                              label_smoothing=0.1)
    assert torch.allclose(ls, ref, atol=1e-6) and not torch.allclose(ls, base)


def test_label_smoothing_backward():
    m = _model().train()
    feats, mask, labels = _batch()
    loss, _ = m(feats, mask, labels, label_smoothing=0.1)
    loss.backward()
    assert m.encoder.input_proj.weight.grad is not None


# --------------------------------------------------------------------------- #
# decodificação
# --------------------------------------------------------------------------- #
def test_generation_kwargs_only_what_was_set():
    ap = argparse.ArgumentParser()
    add_generation_args(ap)
    assert generation_kwargs(ap.parse_args([])) == {}
    assert generation_kwargs(ap.parse_args(["--length-penalty", "1.2",
                                            "--no-repeat-ngram-size", "3"])) == \
        {"length_penalty": 1.2, "no_repeat_ngram_size": 3}


def test_no_repeat_ngram_is_applied():
    m = _model().eval()
    feats, mask, _ = _batch()
    with torch.no_grad():
        ids = m.generate(feats, mask, max_new_tokens=12, num_beams=1, no_repeat_ngram_size=1)
    for row in ids.tolist():
        toks = [t for t in row[1:] if t not in (0, 1)]       # sem start/pad/eos
        assert len(toks) == len(set(toks)), row               # nenhum token repetido


def test_generate_defaults_unchanged():
    """Sem gen_kwargs, o generate dá o mesmo resultado de antes."""
    m = _model().eval()
    feats, mask, _ = _batch()
    with torch.no_grad():
        a = m.generate(feats, mask, max_new_tokens=8, num_beams=2)
        b = m.t5.generate(encoder_outputs=m._encode(feats, mask)[0],
                          attention_mask=m._encode(feats, mask)[1],
                          max_new_tokens=8, num_beams=2)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# semente
# --------------------------------------------------------------------------- #
def test_seed_everything_reproduces_init():
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# média de checkpoints
# --------------------------------------------------------------------------- #
def _save_epochs(tmp, n=3):
    from skeltrans.training.checkpoint import save_checkpoint
    models = []
    for e in range(1, n + 1):
        m = _model(seed=e)
        save_checkpoint(os.path.join(tmp, f"epoch{e}.pt"), m, e, {"model": {"d_model": 16}})
        models.append(m)
    return models


def test_average_is_mean_of_weights():
    from average_checkpoints import average, epoch_files
    with tempfile.TemporaryDirectory() as tmp:
        models = _save_epochs(tmp)
        files = epoch_files(tmp)
        assert sorted(files) == [1, 2, 3]
        avg, first = average([files[e] for e in sorted(files)])
        k = "encoder.input_proj.weight"
        mean = sum(m.state_dict()[k] for m in models) / 3
        assert torch.allclose(avg[k], mean, atol=1e-6)
        assert avg[k].dtype == models[0].state_dict()[k].dtype
        assert first["args"] == {"model": {"d_model": 16}}


def test_average_best_uses_validation_history():
    from average_checkpoints import pick_best
    with tempfile.TemporaryDirectory() as tmp:
        _save_epochs(tmp)
        with open(os.path.join(tmp, "eval_history.jsonl"), "w") as f:
            for e, b in ((1, 10.0), (2, 12.5), (3, 11.0)):
                f.write(json.dumps({"epoch": e, "BLEU-4": b, "val_loss": 5 - e}) + "\n")
        best2 = pick_best(tmp, 2, "BLEU-4")
        assert [os.path.basename(p) for p in best2] == ["epoch2.pt", "epoch3.pt"]
        low = pick_best(tmp, 1, "val_loss")                   # menor é melhor
        assert os.path.basename(low[0]) == "epoch3.pt"


def test_average_rejects_different_architectures():
    from average_checkpoints import average
    from skeltrans.training.checkpoint import save_checkpoint
    from skeltrans.training.models import SLTModel
    from transformers import T5Config, T5ForConditionalGeneration
    with tempfile.TemporaryDirectory() as tmp:
        _save_epochs(tmp, n=1)
        cfg = T5Config(vocab_size=64, d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1,
                       num_heads=2, d_kv=16, decoder_start_token_id=0, pad_token_id=0,
                       eos_token_id=1)
        other = SLTModel(T5ForConditionalGeneration(cfg), d_model=32, nhead=2, num_layers=1)
        save_checkpoint(os.path.join(tmp, "other.pt"), other, 1, {})
        try:
            average([os.path.join(tmp, "epoch1.pt"), os.path.join(tmp, "other.pt")])
        except SystemExit:
            return
    raise AssertionError("arquiteturas diferentes deveriam falhar")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} testes passaram")
