#!/usr/bin/env python3
"""Testes das features extras de entrada (--extra-features / --input-norm).

Dados 100% sintéticos (posições aleatórias) e um T5 minúsculo criado do zero:
não exige a base nem download de pesos.

Uso:
    python3 tests/test_extra_features.py
    (também compatível com: pytest tests/test_extra_features.py)
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skeltrans.common.geometry import normalize_array  # noqa: E402
from skeltrans.common.layout import INPUT_DIM, LEFT_HAND, RIGHT_HAND  # noqa: E402
from skeltrans.training.data.features import (GROUP_DIMS, build_input,  # noqa: E402
                                              compute_extra, feature_dim,
                                              parse_groups)

ALL = "A,B,C,D,E,F"


def _sample(T=40, seed=0):
    """Frase sintética já normalizada pelo Passo 3: (T,115,9)."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.3, 0.7, (T, 115, 3))
    pts[:, 11] = [0.40, 0.5, 0.0]
    pts[:, 12] = [0.60, 0.5, 0.0]
    return normalize_array(pts).astype(np.float32)


def _cols(groups, g):
    """Fatia das colunas do grupo `g` dentro de compute_extra(groups)."""
    start = 0
    for k in parse_groups(groups):
        if k == g:
            return slice(start, start + GROUP_DIMS[k])
        start += GROUP_DIMS[k]
    raise KeyError(g)


def test_parse_groups():
    assert parse_groups("") == ()
    assert parse_groups("f, a,A") == ("A", "F")
    assert parse_groups(["F", "A"]) == ("A", "F")
    try:
        parse_groups("A,X")
    except ValueError:
        pass
    else:
        raise AssertionError("grupo invalido deveria falhar")


def test_dims():
    x = _sample()
    for g, d in GROUP_DIMS.items():
        assert compute_extra(x, g).shape == (40, d), g
    assert feature_dim(ALL) == INPUT_DIM + 214
    assert build_input(x, ALL).shape == (40, INPUT_DIM + 214)


def test_no_groups_is_original_input():
    """Sem grupos, a entrada é bit a bit a de antes (reshape para 1035)."""
    x = _sample()
    np.testing.assert_array_equal(build_input(x, ()), x.reshape(40, -1))


def test_flattened_input_matches():
    x = _sample()
    np.testing.assert_allclose(build_input(x.reshape(40, -1), ALL), build_input(x, ALL))


def test_scale_and_translation_invariance():
    """A (mão), B (orientação) e E (face) são razões/ângulos: invariantes a
    escalar e transladar o corpo inteiro."""
    x = _sample()
    y = x.copy()
    y[..., :3] = x[..., :3] * 3.7 + np.array([1.5, -2.0, 0.3], dtype=np.float32)
    for g in ("A", "B", "E"):
        np.testing.assert_allclose(compute_extra(y, g), compute_extra(x, g),
                                   rtol=1e-4, atol=1e-4, err_msg=g)


def test_handshape_independent_of_hand_position():
    """Mover só a mão não muda a configuração de mão (A)."""
    x = _sample()
    y = x.copy()
    y[:, LEFT_HAND, :3] += np.array([0.8, 0.4, 0.0], dtype=np.float32)
    np.testing.assert_allclose(compute_extra(y, "A"), compute_extra(x, "A"), atol=1e-4)


def test_straight_finger_angle_is_pi():
    x = _sample(T=1)
    hand = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):                       # todos os pontos numa reta
        hand[i] = [0.1 * (i + 1), 0.0, 0.0]
    x[0, LEFT_HAND, :3] = hand
    ang = compute_extra(x, "A")[0, 63:78]     # flexão da mão esquerda
    np.testing.assert_allclose(ang, np.pi, atol=1e-3)


def test_absent_hand_is_zero_and_flagged():
    x = _sample()
    x[:10, RIGHT_HAND] = 0.0                  # mão direita ausente nos 10 1ºs frames
    e = compute_extra(x, ALL)
    a = e[:, _cols(ALL, "A")]
    assert np.all(a[:10, 88:] == 0) and np.any(a[10:, 88:] != 0)
    b = e[:, _cols(ALL, "B")]
    assert np.all(b[:10, 3:] == 0)
    c = e[:, _cols(ALL, "C")]
    assert np.all(c[:10, 10:] == 0)           # 10 distâncias da mão dir + punho-punho
    d = e[:, _cols(ALL, "D")]
    assert np.all(d[:10, 1] == 0)
    f = e[:, _cols(ALL, "F")]
    assert np.all(f[:10, 2] == 0) and np.all(f[10:, 2] == 1)
    assert np.all(np.isfinite(e))


def _tiny_t5(tmp):
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=64, d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1,
                   num_heads=2, d_kv=16, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1)
    path = os.path.join(tmp, "t5")
    T5ForConditionalGeneration(cfg).save_pretrained(path)
    return path


def test_checkpoint_roundtrip_with_input_norm():
    """Salva um modelo com extras + input-norm e recarrega pelo load_model:
    mesma arquitetura, mesmas estatísticas e mesma saída."""
    import torch
    from transformers import T5ForConditionalGeneration

    from skeltrans.training.checkpoint import load_model, save_checkpoint
    from skeltrans.training.models import SLTModel

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        t5_path = _tiny_t5(tmp)
        model = SLTModel(T5ForConditionalGeneration.from_pretrained(t5_path), d_model=16,
                         nhead=2, num_layers=1, extra_features="A,F", input_norm=True)
        D = feature_dim("A,F")
        model.encoder.set_input_stats(np.full(D, 0.5), np.full(D, 2.0))
        meta = {"model": {"t5_name": t5_path, "d_model": 16, "nhead": 2, "num_layers": 1,
                          "extra_features": "A,F", "input_norm": True}}
        ck = os.path.join(tmp, "m.pt")
        save_checkpoint(ck, model, 1, meta)

        loaded, _, _ = load_model(ck, torch.device("cpu"))
        assert loaded.extra_features == ("A", "F")
        assert torch.allclose(loaded.encoder.input_std, torch.full((D,), 2.0))

        feats = torch.from_numpy(build_input(_sample(T=30), "A,F")).unsqueeze(0)
        mask = torch.zeros(1, 30, dtype=torch.bool)
        model.eval()
        with torch.no_grad():
            a, _ = model.encoder(feats, mask)
            b, _ = loaded.encoder(feats, mask)
        assert torch.allclose(a, b, atol=1e-6)

        # grupos gravados que não batem com os pesos -> erro explícito
        meta["model"]["extra_features"] = "A"
        save_checkpoint(ck, model, 1, meta)
        try:
            load_model(ck, torch.device("cpu"))
        except ValueError:
            pass
        else:
            raise AssertionError("checkpoint inconsistente deveria falhar")


def test_input_norm_keeps_padding_zero():
    import torch
    from skeltrans.training.models.landmark_encoder import LandmarkEncoder

    enc = LandmarkEncoder(input_dim=8, d_model=8, nhead=2, num_layers=1, input_norm=True)
    enc.set_input_stats(np.full(8, 3.0), np.full(8, 2.0))
    feats = torch.ones(1, 5, 8)
    mask = torch.tensor([[False, False, False, True, True]])
    seen = {}
    enc.input_proj.register_forward_hook(lambda m, i, o: seen.setdefault("x", i[0]))
    enc.eval()
    enc(feats, mask)
    assert torch.all(seen["x"][0, 3:] == 0)                 # padding continua 0
    assert torch.allclose(seen["x"][0, :3], torch.full((3, 8), -1.0))  # (1-3)/2


def test_legacy_encoder_state_dict_unchanged():
    """Sem input_norm, o encoder não ganha buffers (checkpoints antigos intactos)."""
    from skeltrans.training.models.landmark_encoder import LandmarkEncoder
    keys = LandmarkEncoder(d_model=8, nhead=2, num_layers=1).state_dict().keys()
    assert not any("input_mean" in k or "input_std" in k for k in keys)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} testes passaram")
