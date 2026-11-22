#!/usr/bin/env python3
"""Testes da data augmentation dos keypoints (--aug-speed/-spatial/-noise/-flip).

Dados sintéticos; não exige a base nem download de pesos.

Uso:
    python3 tests/test_augment.py
    (também compatível com: pytest tests/test_augment.py)
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skeltrans.common.geometry import normalize_array, present_mask, temporal_features  # noqa: E402
from skeltrans.common.layout import (FACE, FACE_MASK, L_SHOULDER, LEFT_HAND, N_POINTS,  # noqa: E402
                                     POSE, R_SHOULDER, RIGHT_HAND)
from skeltrans.training.data import LandmarkTextDataset  # noqa: E402
from skeltrans.training.data.augment import (FLIP_PERM, AugmentConfig, _speed,  # noqa: E402
                                             augment)
from skeltrans.training.data.features import feature_dim  # noqa: E402


def _frase(T=40, seed=0):
    """Frase sintética já normalizada (T, 115, 9), com a mão esq. ausente num trecho."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.3, 0.7, (T, N_POINTS, 3))
    pts[:, L_SHOULDER] = [0.40, 0.5, 0.0]
    pts[:, R_SHOULDER] = [0.60, 0.5, 0.0]
    pts[10:16, LEFT_HAND] = 0.0
    return normalize_array(pts)


TODAS = AugmentConfig(speed=0.2, spatial=0.2, noise=0.01, flip=True)


# --------------------------------------------------------------------------- #
# Comportamento padrão
# --------------------------------------------------------------------------- #
def test_disabled_returns_input_unchanged():
    x = _frase()
    assert augment(x, AugmentConfig(), np.random.default_rng(0)) is x


def test_dataset_without_augment_is_identical():
    man, d = _manifesto()
    a = LandmarkTextDataset(man, features_dir=d)[0][0]
    b = LandmarkTextDataset(man, features_dir=d, augment=AugmentConfig())[0][0]
    assert torch.equal(a, b)
    assert LandmarkTextDataset(man, features_dir=d, augment=AugmentConfig()).augment is None


# --------------------------------------------------------------------------- #
# Espelhamento
# --------------------------------------------------------------------------- #
def test_flip_permutation_is_involution_and_swaps_sides():
    assert np.array_equal(FLIP_PERM[FLIP_PERM], np.arange(N_POINTS))
    # mãos trocam por inteiro, ponto a ponto
    assert np.array_equal(FLIP_PERM[LEFT_HAND], np.arange(RIGHT_HAND.start, RIGHT_HAND.stop))
    # ombros trocam; nariz fica
    assert FLIP_PERM[POSE.start + L_SHOULDER] == POSE.start + R_SHOULDER
    assert FLIP_PERM[POSE.start + 0] == POSE.start + 0
    face = lambda mid: FACE.start + FACE_MASK.index(mid)  # noqa: E731
    assert FLIP_PERM[face(61)] == face(291)         # cantos da boca
    assert FLIP_PERM[face(33)] == face(263)         # cantos externos dos olhos
    assert FLIP_PERM[face(70)] == face(300)         # pontas das sobrancelhas
    assert FLIP_PERM[face(13)] == face(13)          # centro do lábio superior fica


def test_flip_mirrors_positions():
    x = _frase()
    cfg = AugmentConfig(flip=True)
    rng = np.random.default_rng(1)
    while True:                                     # acha um sorteio que espelha
        y = augment(x, cfg, rng)
        if not np.allclose(y[..., :3], x[..., :3]):
            break
    expected = x[:, FLIP_PERM, :3].copy()
    expected[..., 0] *= -1.0
    assert np.allclose(y[..., :3], expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Invariantes de todas as augmentations
# --------------------------------------------------------------------------- #
def test_absent_points_stay_absent():
    x = _frase()
    for seed in range(20):
        y = augment(x, AugmentConfig(spatial=0.3, noise=0.05), np.random.default_rng(seed))
        assert np.all(y[10:16, LEFT_HAND] == 0.0)
        assert np.array_equal(present_mask(y[..., :3]), present_mask(x[..., :3]))


def test_velocity_and_acceleration_recomputed():
    x = _frase()
    for seed in range(10):
        y = augment(x, TODAS, np.random.default_rng(seed))
        vel, acc = temporal_features(y[..., :3].astype(np.float64), present_mask(y[..., :3]))
        assert np.allclose(y[..., 3:6], vel, atol=1e-5)
        assert np.allclose(y[..., 6:9], acc, atol=1e-5)


def test_noise_only_on_present_points():
    x = _frase()
    y = augment(x, AugmentConfig(noise=0.01), np.random.default_rng(0))
    present = present_mask(x[..., :3])
    assert np.all(y[..., :3][~present] == 0.0)
    assert 0.0 < np.abs(y[..., :3] - x[..., :3])[present].mean() < 0.05


def test_spatial_is_linear_around_origin():
    """Rotação/escala/cisalhamento em XY, sem translação; Z não muda."""
    x = _frase()
    y = augment(x, AugmentConfig(spatial=0.2), np.random.default_rng(3))
    assert np.allclose(y[..., 2], x[..., 2])
    # mesma matriz em todos os pontos: resolve M a partir de 3 pontos e confere o resto
    p, q = x[0, :, :2], y[0, :, :2]
    M, *_ = np.linalg.lstsq(p, q, rcond=None)
    assert np.allclose(p @ M, q, atol=1e-5)


def test_speed_changes_length_within_bounds():
    x = _frase(T=100)
    for seed in range(20):
        y = augment(x, AugmentConfig(speed=0.2), np.random.default_rng(seed))
        assert 83 <= y.shape[0] <= 125               # 100 / 1,2 ... 100 / 0,8
    # fator 1: posições idênticas
    pos = x[..., :3].astype(np.float64)
    out, ok = _speed(pos, present_mask(pos), 1.0)
    assert np.allclose(out, pos) and np.array_equal(ok, present_mask(pos))


def test_speed_interpolation_respects_gaps():
    """Frame novo entre um presente e um ausente fica ausente (sem 'puxar' para 0)."""
    x = _frase(T=40)
    pos = x[..., :3].astype(np.float64)
    out, ok = _speed(pos, present_mask(pos), 0.5)    # desacelera: frames intermediários
    hand = ok[:, LEFT_HAND.start]
    # os frames 10..15 originais eram ausentes; no tempo novo, 9.5 e 15.5 também
    assert not hand[19] and not hand[31] and hand[18] and hand[32]
    assert np.all(out[~ok] == 0.0)


# --------------------------------------------------------------------------- #
# Dataset e CLI
# --------------------------------------------------------------------------- #
def _manifesto(n=3):
    d = tempfile.mkdtemp()
    items = []
    for i in range(n):
        np.save(os.path.join(d, f"f{i}.npy"), _frase(T=30 + 10 * i, seed=i))
        items.append({"features": f"f{i}.npy", "text": f"frase {i}", "tokens": ["A", "B"]})
    path = os.path.join(d, "m.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f)
    return path, d


def test_dataset_augments_and_builds_extras_after():
    man, d = _manifesto()
    ds = LandmarkTextDataset(man, features_dir=d, extra_features="A,F", augment=TODAS)
    torch.manual_seed(0)
    a = ds[0][0]
    torch.manual_seed(0)
    b = ds[0][0]
    torch.manual_seed(1)
    c = ds[0][0]
    assert a.shape[1] == feature_dim("A,F")
    assert torch.equal(a, b)                         # mesma semente do torch -> mesmo sorteio
    assert not torch.equal(a, c)                     # outra semente -> outro sorteio


def test_cli_flags_map_to_config_and_validate():
    from skeltrans.training.cli import args_to_configs, build_parser, main
    args = build_parser().parse_args(["--train-manifest", "x.json", "--aug-speed", "0.2",
                                      "--aug-spatial", "0.2", "--aug-noise", "0.001",
                                      "--aug-flip"])
    data = args_to_configs(args)[1]
    assert (data.aug_speed, data.aug_spatial, data.aug_noise, data.aug_flip) == (0.2, 0.2, 0.001, True)
    default = args_to_configs(build_parser().parse_args(["--train-manifest", "x.json"]))[1]
    assert (default.aug_speed, default.aug_spatial, default.aug_noise, default.aug_flip) == (0, 0, 0, False)
    for bad in (["--aug-speed", "1.0"], ["--aug-noise", "-1"]):
        try:
            main(["--train-manifest", "x.json"] + bad)
        except SystemExit as e:
            assert "--aug" in str(e.code)
            continue
        raise AssertionError(f"deveria recusar {bad}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} testes passaram")
