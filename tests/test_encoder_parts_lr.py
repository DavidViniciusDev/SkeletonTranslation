#!/usr/bin/env python3
"""Testes do encoder por partes (--encoder-type parts) e do lr separado
(--encoder-lr).

Dados sintéticos e um T5 minúsculo criado do zero: não exige a base nem
download de pesos.

Uso:
    python3 tests/test_encoder_parts_lr.py
    (também compatível com: pytest tests/test_encoder_parts_lr.py)
"""
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skeltrans.common.layout import INPUT_DIM, N_CHANNELS, POSE, RIGHT_HAND  # noqa: E402
from skeltrans.training.data.features import feature_dim  # noqa: E402
from skeltrans.training.models.landmark_encoder import LandmarkEncoder  # noqa: E402
from skeltrans.training.models.parts_projection import PartsProjection  # noqa: E402


def _tiny_t5():
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=64, d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1,
                   num_heads=2, d_kv=16, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1)
    return T5ForConditionalGeneration(cfg)


def _model(**kw):
    from skeltrans.training.models import SLTModel
    torch.manual_seed(0)
    return SLTModel(_tiny_t5(), d_model=16, nhead=2, num_layers=1, **kw)


# --------------------------------------------------------------------------- #
# Encoder por partes
# --------------------------------------------------------------------------- #
def test_parts_projection_shapes():
    p = PartsProjection(INPUT_DIM, d_model=32)
    assert p.extra is None and p.part_dim == 8
    assert p(torch.randn(2, 7, INPUT_DIM)).shape == (2, 7, 32)
    pe = PartsProjection(feature_dim("A,F"), d_model=32)
    assert pe.extra is not None and pe.extra_dim == feature_dim("A,F") - INPUT_DIM
    assert pe(torch.randn(2, 7, feature_dim("A,F"))).shape == (2, 7, 32)


def test_parts_streams_are_isolated():
    """Mudar só as colunas da face não altera a saída do fluxo da mão direita."""
    p = PartsProjection(INPUT_DIM, d_model=32).eval()
    x = torch.randn(1, 5, INPUT_DIM)
    y = x.clone()
    face_cols = p.cols["face"]
    y[..., face_cols] += 3.0
    rh = p.cols["mao_dir"]
    assert rh == slice(RIGHT_HAND.start * N_CHANNELS, RIGHT_HAND.stop * N_CHANNELS)
    assert p.cols["pose"] == slice(POSE.start * N_CHANNELS, POSE.stop * N_CHANNELS)
    with torch.no_grad():
        assert torch.equal(p.parts["mao_dir"](x[..., rh]), p.parts["mao_dir"](y[..., rh]))
        assert not torch.allclose(p(x), p(y))          # mas a saída fundida muda


def test_parts_rejects_short_input():
    try:
        PartsProjection(345, d_model=32)               # só posições (115 x 3)
    except ValueError:
        return
    raise AssertionError("entrada com menos de 1035 colunas deveria falhar")


def test_invalid_encoder_type():
    try:
        LandmarkEncoder(d_model=8, nhead=2, num_layers=1, encoder_type="gcn")
    except ValueError:
        return
    raise AssertionError("encoder_type invalido deveria falhar")


def test_parts_forward_backward_all_params_get_grad():
    model = _model(encoder_type="parts", ds_steps=1, extra_features="A,F")
    feats = torch.randn(2, 30, feature_dim("A,F"))
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[1, 20:] = True
    labels = torch.randint(2, 64, (2, 6))
    loss, _ = model(feats, mask, labels)
    loss.backward()
    proj = model.encoder.input_proj
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in proj.parameters())
    assert model.generate(feats, mask, max_new_tokens=4, num_beams=1).shape[0] == 2


def test_flat_keys_unchanged():
    """O 'flat' (padrão) continua com as chaves originais do state_dict."""
    keys = set(_model().state_dict())
    assert "encoder.input_proj.weight" in keys
    assert not any(k.startswith("encoder.input_proj.parts.") for k in keys)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def _roundtrip(model, meta_model):
    from skeltrans.training.checkpoint import load_model, save_checkpoint
    tmp = tempfile.mkdtemp()
    t5_path = os.path.join(tmp, "t5")
    model.t5.save_pretrained(t5_path)
    meta = {"model": {"t5_name": t5_path, "d_model": 16, "nhead": 2, "num_layers": 1,
                      **meta_model}}
    ck = os.path.join(tmp, "m.pt")
    save_checkpoint(ck, model, 1, meta)
    return load_model(ck, torch.device("cpu"))[0], ck, meta


def test_parts_checkpoint_roundtrip():
    model = _model(encoder_type="parts", extra_features="A,F", input_norm=True)
    D = feature_dim("A,F")
    model.encoder.set_input_stats(np.full(D, 0.1), np.full(D, 1.5))
    loaded, _, _ = _roundtrip(model, {"extra_features": "A,F", "input_norm": True,
                                      "encoder_type": "parts"})
    assert loaded.encoder.encoder_type == "parts"
    feats = torch.randn(1, 12, D)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model.encoder(feats, mask)[0], loaded.encoder(feats, mask)[0])


def test_encoder_type_detected_from_weights():
    """Os pesos mandam: config 'flat' com pesos 'parts' carrega como 'parts'."""
    loaded, _, _ = _roundtrip(_model(encoder_type="parts"), {"encoder_type": "flat"})
    assert loaded.encoder.encoder_type == "parts"
    # checkpoint antigo, sem a chave encoder_type no config -> 'flat'
    loaded, _, _ = _roundtrip(_model(), {})
    assert loaded.encoder.encoder_type == "flat"


def test_parts_checkpoint_dim_mismatch_fails():
    from skeltrans.training.checkpoint import load_model, save_checkpoint
    model = _model(encoder_type="parts", extra_features="A,F")
    _, ck, meta = _roundtrip(model, {"extra_features": "A,F"})
    meta["model"]["extra_features"] = "A"            # grupos que não batem com os pesos
    save_checkpoint(ck, model, 1, meta)
    try:
        load_model(ck, torch.device("cpu"))
    except ValueError:
        return
    raise AssertionError("checkpoint inconsistente deveria falhar")


# --------------------------------------------------------------------------- #
# lr separado
# --------------------------------------------------------------------------- #
def _trainer(model, **cfg_kw):
    from skeltrans.training.config import TrainConfig
    from skeltrans.training.trainer import Trainer
    return Trainer(model, None, TrainConfig(lr=1e-4, device="cpu", **cfg_kw),
                   torch.device("cpu"))


def test_single_group_by_default():
    model = _model()
    groups = _trainer(model)._param_groups()
    opt = torch.optim.AdamW(groups, lr=1e-4)
    assert len(opt.param_groups) == 1


def test_encoder_lr_groups():
    for kw in ({}, {"use_ctc": True, "gloss_vocab_size": 9}, {"encoder_type": "parts"}):
        model = _model(**kw)
        opt = torch.optim.AdamW(_trainer(model, encoder_lr=5e-4)._param_groups(), lr=1e-4)
        dec, enc = opt.param_groups
        assert (dec["name"], dec["lr"]) == ("decoder_t5", 1e-4)
        assert (enc["name"], enc["lr"]) == ("encoder", 5e-4)
        enc_ids = {id(p) for p in enc["params"]}
        assert all(id(p) in enc_ids for p in model.encoder.parameters())
        if model.ctc_head is not None:
            assert all(id(p) in enc_ids for p in model.ctc_head.parameters())
        assert all(id(p) not in enc_ids for p in model.t5.parameters())
        n = sum(len(g["params"]) for g in opt.param_groups)
        assert n == len(list(model.parameters())), kw


def test_scheduler_keeps_lr_ratio():
    """Warmup + decaimento linear escalam os dois lrs na mesma proporção."""
    from transformers import get_linear_schedule_with_warmup
    model = _model()
    opt = torch.optim.AdamW(_trainer(model, encoder_lr=5e-4)._param_groups(), lr=1e-4)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=100)
    for step in range(60):
        opt.step()
        sched.step()
        dec, enc = sched.get_last_lr()
        if dec > 0:
            assert abs(enc / dec - 5.0) < 1e-9, step


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} testes passaram")
