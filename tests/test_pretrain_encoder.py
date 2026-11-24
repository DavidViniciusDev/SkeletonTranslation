#!/usr/bin/env python3
"""Testes do pré-treino do encoder (masked pose modeling) e do --init-encoder.

Dados sintéticos e T5 minúsculo; não exige a base nem download de pesos.

Uso:
    python3 tests/test_pretrain_encoder.py
    (também compatível com: pytest tests/test_pretrain_encoder.py)
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skeltrans.common.layout import LEFT_HAND, N_POINTS, RIGHT_HAND  # noqa: E402
from skeltrans.training.config import ModelConfig  # noqa: E402
from skeltrans.training.data import KeypointDataset, LandmarkTextDataset  # noqa: E402
from skeltrans.training.data.features import feature_dim  # noqa: E402
from skeltrans.training.encoder_init import (encoder_spec, init_encoder_from_file,  # noqa: E402
                                             load_encoder_weights, read_encoder,
                                             save_encoder, spec_mismatches)
from skeltrans.training.models import SLTModel  # noqa: E402
from skeltrans.training.models.masked_pose import (BASE_DIM, MaskedPoseModel,  # noqa: E402
                                                   apply_masks, make_masks)

# encoder minúsculo usado em todos os testes
TINY = dict(d_model=32, nhead=4, num_layers=2, ds_steps=2)


def _pad_mask(lengths, T=None):
    T = T or max(lengths)
    m = torch.ones(len(lengths), T, dtype=torch.bool)
    for i, n in enumerate(lengths):
        m[i, :n] = False
    return m


def _tiny_t5():
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=64, d_model=48, d_ff=64, num_layers=1, num_decoder_layers=1,
                   num_heads=4, d_kv=12, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1)
    return T5ForConditionalGeneration(cfg)


def _pretrained_file(tmp, extra=("A", "F"), input_norm=True, **kw):
    """Grava um encoder.pt de um MaskedPoseModel recém-criado; devolve (path, modelo)."""
    hp = dict(TINY, **kw)
    model = MaskedPoseModel(feature_dim(extra), input_norm=input_norm, **hp)
    if input_norm:
        model.encoder.set_input_stats(torch.arange(feature_dim(extra)).float(),
                                      torch.full((feature_dim(extra),), 2.0))
    spec = encoder_spec(extra, input_norm, "flat", hp["d_model"], hp["nhead"],
                        hp["num_layers"], hp["ds_steps"])
    path = os.path.join(tmp, "encoder.pt")
    save_encoder(path, model.encoder, spec, {"epochs_run": 3})
    return path, model


def _model_cfg(extra="A,F", input_norm=True, **kw):
    hp = dict(TINY, **kw)
    return ModelConfig(extra_features=extra, input_norm=input_norm, encoder_type="flat",
                       **hp)


# --------------------------------------------------------------------------- #
# Mascaramento
# --------------------------------------------------------------------------- #

def test_masks_cover_ratio_and_skip_padding():
    pad = _pad_mask([100, 60, 30], T=100)
    gen = torch.Generator().manual_seed(0)
    pm, fm = make_masks(pad, ratio=0.4, span=8, mode="frames", generator=gen)
    assert pm.shape == (3, 100, N_POINTS) and fm.shape == (3, 100)
    for b, n in enumerate([100, 60, 30]):
        assert fm[b, :n].sum() >= round(0.4 * n), "cobertura abaixo do ratio"
        assert not fm[b, n:].any() and not pm[b, n:].any(), "mascarou padding"


def test_masks_reproducible_with_generator():
    pad = _pad_mask([50, 40])
    a = make_masks(pad, generator=torch.Generator().manual_seed(7))
    b = make_masks(pad, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_mask_modes():
    pad = _pad_mask([80])
    pm, fm = make_masks(pad, mode="frames", generator=torch.Generator().manual_seed(1))
    # frames: todo frame tocado tem os 115 pontos escondidos
    assert torch.equal(pm[0].any(-1), pm[0].all(-1))
    assert torch.equal(pm[0].any(-1), fm[0])
    pm, fm = make_masks(pad, mode="hands", generator=torch.Generator().manual_seed(1))
    hands = torch.zeros(N_POINTS, dtype=torch.bool)
    hands[LEFT_HAND] = True
    hands[RIGHT_HAND] = True
    assert pm.any() and not pm[..., ~hands].any(), "hands escondeu ponto fora das maos"
    assert torch.equal(pm[0].any(-1), fm[0]), "extras devem sumir onde a mao some"


def test_short_sentence_gets_one_span():
    pm, fm = make_masks(_pad_mask([3]), ratio=0.4, span=8,
                        generator=torch.Generator().manual_seed(0))
    assert fm[0].sum() == 3


def test_masks_reject_bad_args():
    for kw in ({"ratio": 0.0}, {"ratio": 1.0}, {"span": 0}, {"mode": "xx"}):
        try:
            make_masks(_pad_mask([10]), **kw)
        except ValueError:
            continue
        raise AssertionError(f"deveria recusar {kw}")


def test_apply_masks_zeroes_only_what_was_drawn():
    torch.manual_seed(0)
    feats = torch.randn(2, 10, feature_dim(("A",)))
    pm = torch.zeros(2, 10, N_POINTS, dtype=torch.bool)
    fm = torch.zeros(2, 10, dtype=torch.bool)
    pm[0, 2:4, 5] = True
    fm[1, 7] = True
    out = apply_masks(feats, pm, fm)
    base = out[..., :BASE_DIM].reshape(2, 10, N_POINTS, 9)
    assert (base[0, 2:4, 5] == 0).all()
    assert (out[1, 7, BASE_DIM:] == 0).all()
    changed = (out != feats)
    assert changed.sum() == 2 * 9 + feature_dim(("A",)) - BASE_DIM
    assert not torch.equal(out, feats) and feats.abs().min() > 0, "alterou a entrada original"


# --------------------------------------------------------------------------- #
# Modelo de reconstrução
# --------------------------------------------------------------------------- #

def test_forward_shapes_any_length_and_ds():
    for ds, T in ((0, 13), (1, 13), (2, 37)):
        torch.manual_seed(0)
        model = MaskedPoseModel(BASE_DIM, **dict(TINY, ds_steps=ds))
        feats = torch.randn(2, T, BASE_DIM)
        pad = _pad_mask([T, T - 5])
        pm, fm = make_masks(pad, generator=torch.Generator().manual_seed(0))
        loss, st = model(feats, pad, pm, fm)
        loss.backward()
        assert torch.isfinite(loss) and st["count"] > 0
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_loss_ignores_absent_points():
    model = MaskedPoseModel(BASE_DIM, **TINY)
    feats = torch.randn(1, 12, BASE_DIM)
    base = feats.view(1, 12, N_POINTS, 9)
    base[:, :, LEFT_HAND] = 0.0                       # mão esquerda nunca detectada
    pm = torch.zeros(1, 12, N_POINTS, dtype=torch.bool)
    pm[0, :, LEFT_HAND] = True                        # só ela escondida
    fm = pm.any(-1)
    loss, st = model(feats, _pad_mask([12]), pm, fm)
    assert st["count"] == 0 and float(loss) == 0.0


def test_pretraining_learns_structure():
    """Em sinais suaves no tempo, o modelo deve recuperar parte do trecho escondido."""
    torch.manual_seed(0)
    T, B = 48, 8
    t = torch.linspace(0, 6.28, T)

    def batch():
        phase = torch.rand(B, 1, 1) * 6.28
        freq = torch.randint(1, 3, (B, 1, N_POINTS)).float()
        pos = torch.sin(freq * t.view(1, T, 1) + phase)  # (B, T, 115)
        return pos.unsqueeze(-1).repeat(1, 1, 1, 9).reshape(B, T, BASE_DIM) + 2.0

    model = MaskedPoseModel(BASE_DIM, **dict(TINY, ds_steps=1, d_model=64))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(0)
    pad = _pad_mask([T] * B)
    for _ in range(150):
        pm, fm = make_masks(pad, ratio=0.3, span=4, mode="frames", generator=gen)
        loss, _ = model(batch(), pad, pm, fm)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pm, fm = make_masks(pad, ratio=0.3, span=4, mode="frames",
                            generator=torch.Generator().manual_seed(99))
        _, st = model(batch(), pad, pm, fm)
    explained = 1 - st["sq_err"] / st["sq_zero"]
    assert explained > 0.5, f"recuperou so {explained:.2f} do sinal escondido"


# --------------------------------------------------------------------------- #
# Arquivo encoder.pt e --init-encoder
# --------------------------------------------------------------------------- #

def test_save_and_read_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path, model = _pretrained_file(tmp)
        ckpt = read_encoder(path)
        assert ckpt["spec"]["input_dim"] == feature_dim(("A", "F"))
        assert ckpt["spec"]["extra_features"] == "A,F"
        assert not any(k.startswith("adapter.") for k in ckpt["encoder"])
        for k, v in model.encoder.state_dict().items():
            assert torch.equal(ckpt["encoder"][k], v)


def test_read_rejects_other_files():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "best.pt")
        torch.save({"model": {}, "epoch": 1, "args": {}}, path)   # parece um best.pt
        try:
            read_encoder(path)
        except ValueError as exc:
            assert "pretrain_encoder.py" in str(exc)
        else:
            raise AssertionError("deveria recusar um arquivo que nao e encoder.pt")


def test_init_encoder_copies_weights_and_input_stats():
    with tempfile.TemporaryDirectory() as tmp:
        path, pre = _pretrained_file(tmp)
        torch.manual_seed(123)
        slt = SLTModel(_tiny_t5(), extra_features=("A", "F"), input_norm=True, **TINY)
        adapter_before = slt.encoder.adapter.weight.clone()
        info = init_encoder_from_file(slt, path, _model_cfg())
    assert info["epochs_run"] == 3
    got = slt.encoder.state_dict()
    for k, v in pre.encoder.state_dict().items():
        assert torch.equal(got[k], v), f"{k} nao veio do pre-treino"
    assert torch.equal(slt.encoder.input_mean, torch.arange(feature_dim(("A", "F"))).float())
    # adaptador (ponte para o T5, 32 -> 48) continua o do SLT
    assert torch.equal(slt.encoder.adapter.weight, adapter_before)


def test_init_encoder_with_concept_mode_loads_frame_floor():
    with tempfile.TemporaryDirectory() as tmp:
        path, pre = _pretrained_file(tmp)
        slt = SLTModel(_tiny_t5(), extra_features=("A", "F"), input_norm=True,
                       use_ctc=True, gloss_vocab_size=10, temporal_mode="concept", **TINY)
        missing = load_encoder_weights(slt.encoder, read_encoder(path))
    assert any(k.startswith("concept_transformer.") for k in missing)
    assert torch.equal(slt.encoder.transformer.layers[0].linear1.weight,
                       pre.encoder.transformer.layers[0].linear1.weight)


def test_init_encoder_rejects_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _pretrained_file(tmp)
        slt = SLTModel(_tiny_t5(), extra_features=("A",), input_norm=True,
                       **dict(TINY, num_layers=3))
        try:
            init_encoder_from_file(slt, path, _model_cfg(extra="A", num_layers=3))
        except ValueError as exc:
            msg = str(exc)
            assert "num_layers" in msg and "extra_features" in msg and "--num-layers" in msg
        else:
            raise AssertionError("deveria recusar spec diferente")


def test_spec_mismatch_ignores_dropout_and_matches_itself():
    a = encoder_spec("A,F", True, "flat", 512, 8, 6, 2)
    assert spec_mismatches(a, dict(a)) == []
    assert spec_mismatches(a, encoder_spec(["F", "A"], True, "flat", 512, 8, 6, 2)) == []
    assert len(spec_mismatches(a, encoder_spec("A,F", False, "gcn", 512, 8, 6, 2))) == 2


# --------------------------------------------------------------------------- #
# CLI do SLT e do pré-treino
# --------------------------------------------------------------------------- #

def test_slt_cli_init_encoder_flag():
    from skeltrans.training.cli import (args_to_configs, build_parser,
                                        validate_init_encoder)
    base = ["--train-manifest", "x.json", "--extra-features", "A,F", "--input-norm",
            "--d-model", "32", "--nhead", "4", "--num-layers", "2", "--ds-steps", "2"]
    args = build_parser().parse_args(base)
    assert args_to_configs(args)[2].init_encoder is None and validate_init_encoder(args) is None
    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _pretrained_file(tmp)
        ok = build_parser().parse_args(base + ["--init-encoder", path])
        assert args_to_configs(ok)[2].init_encoder == path
        assert validate_init_encoder(ok) is None
        bad = build_parser().parse_args(base + ["--init-encoder", path, "--num-layers", "4"])
        assert "num_layers" in validate_init_encoder(bad)
        missing = build_parser().parse_args(base + ["--init-encoder", os.path.join(tmp, "x.pt")])
        assert "nao encontrado" in validate_init_encoder(missing)


def test_pretrain_cli_validation():
    from skeltrans.training.pretrain import args_to_config, build_parser, validate_args
    ok = build_parser().parse_args(["--manifests", "a.json", "b.json", "--extra-features",
                                    "f,a", "--mask-mode", "hands"])
    assert validate_args(ok) is None
    cfg = args_to_config(ok)
    assert cfg.manifests == ["a.json", "b.json"] and cfg.extra_features == "A,F"
    for bad in ([], ["--manifests", "a.json", "--mask-ratio", "1"],
                ["--manifests", "a.json", "--extra-features", "Z"],
                ["--manifests", "a.json", "--d-model", "30", "--nhead", "4"]):
        assert validate_args(build_parser().parse_args(bad)) is not None, bad


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def test_datasets_keypoints_only_and_unchanged_text_dataset():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        arr = rng.normal(size=(15, N_POINTS, 9)).astype(np.float32)
        np.save(os.path.join(tmp, "a.npy"), arr)
        with open(os.path.join(tmp, "sem_texto.json"), "w", encoding="utf-8") as f:
            json.dump([{"features": "a.npy"}], f)
        with open(os.path.join(tmp, "com_texto.json"), "w", encoding="utf-8") as f:
            json.dump([{"features": "a.npy", "text": "oi", "tokens": ["OI"]}], f)
        (x,) = KeypointDataset(os.path.join(tmp, "sem_texto.json"))[0]
        assert x.shape == (15, BASE_DIM) and torch.equal(x, torch.from_numpy(arr.reshape(15, -1)))
        feats, text = LandmarkTextDataset(os.path.join(tmp, "com_texto.json"))[0]
        assert text == "oi" and torch.equal(feats, x)
        feats, text, gloss = LandmarkTextDataset(os.path.join(tmp, "com_texto.json"),
                                                 return_gloss=True,
                                                 extra_features="A")[0]
        assert gloss == ["OI"] and feats.shape == (15, feature_dim(("A",)))


def test_pretrain_smoke_end_to_end():
    from skeltrans.training.pretrain import smoke_test
    smoke_test()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} testes passaram")
