"""Data augmentation dos keypoints, só no treino (--aug-*).

Receita do SignCLIP (Jiang et al., 2024), que usa o mesmo extrator (MediaPipe
Holistic) e a mesma normalização pelos ombros deste projeto: espelhamento
horizontal, transformação espacial 2D (rotação, escala, cisalhamento), mudança
de velocidade por interpolação e ruído gaussiano.

Tudo opera sobre as POSIÇÕES normalizadas (canais 0:3 do Passo 3) e depois
recalcula velocidade e aceleração com a mesma função do Passo 3
(geometry.temporal_features), para os 9 canais continuarem coerentes. Ponto
ausente ([0,0,0]) continua ausente em todas as transformações:

- espacial e espelhamento são lineares, sem translação: zero vai em zero;
- o ruído só é somado aos pontos presentes;
- na velocidade, um frame novo só tem o ponto se ele existe nos dois frames
  originais vizinhos.

Roda ANTES das --extra-features, que são calculadas sobre os pontos já
aumentados.
"""

from dataclasses import dataclass

import numpy as np

from skeltrans.common.geometry import present_mask, temporal_features
from skeltrans.common.layout import (FACE, FACE_LEFT_EYE, FACE_LEFT_EYEBROW, FACE_MASK,
                                     FACE_RIGHT_EYE, FACE_RIGHT_EYEBROW, LEFT_HAND,
                                     N_POINTS, POSE, RIGHT_HAND)

# --- pares esquerda/direita para o espelhamento ---
# Pose (MediaPipe Pose): olhos, orelhas, boca, ombros, braços, mãos, pernas, pés.
_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
               (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30),
               (31, 32)]
# Face (ids do FaceMesh): pontos simétricos dos lábios; os do centro (0, 17, 13,
# 14) ficam onde estão. Olhos e sobrancelhas: ver _face_pairs().
_LIP_PAIRS = [(61, 291), (37, 267), (39, 269), (40, 270), (84, 314), (91, 321),
              (181, 405), (78, 308)]


def _face_pairs():
    """Pares simétricos da máscara facial, em ids do FaceMesh.

    Olho esq. [33 canto ext., 133 canto int., 159, 145, 153] espelha no olho
    dir. [362 canto int., 263 canto ext., 386, 374, 380]: os cantos trocam de
    posição na lista. Sobrancelhas: [70, 63, 105, 66, 107] espelham em
    [336, 296, 334, 293, 300] lida de trás para frente.
    """
    le, re = FACE_LEFT_EYE, FACE_RIGHT_EYE
    eyes = [(le[0], re[1]), (le[1], re[0])] + list(zip(le[2:], re[2:]))
    brows = list(zip(FACE_LEFT_EYEBROW, reversed(FACE_RIGHT_EYEBROW)))
    pairs = _LIP_PAIRS + eyes + brows
    assert all(a in FACE_MASK and b in FACE_MASK for a, b in pairs), "par fora da mascara"
    return pairs


def flip_permutation():
    """Permutação dos 115 pontos que troca esquerda e direita (involução)."""
    perm = np.arange(N_POINTS)
    pairs = [(POSE.start + a, POSE.start + b) for a, b in _POSE_PAIRS]
    pairs += [(LEFT_HAND.start + i, RIGHT_HAND.start + i) for i in range(LEFT_HAND.stop - LEFT_HAND.start)]
    pairs += [(FACE.start + FACE_MASK.index(a), FACE.start + FACE_MASK.index(b))
              for a, b in _face_pairs()]
    for a, b in pairs:
        perm[a], perm[b] = b, a
    return perm


FLIP_PERM = flip_permutation()


@dataclass
class AugmentConfig:
    """Intensidades das augmentations; 0/False desliga cada uma."""
    speed: float = 0.0      # fator de velocidade sorteado em [1 - speed, 1 + speed]
    spatial: float = 0.0    # desvio da rotação (rad), da escala e do cisalhamento 2D
    noise: float = 0.0      # desvio do ruído gaussiano nas posições (unidade: ombros)
    flip: bool = False      # espelha (troca esquerda/direita) com probabilidade 0,5

    @property
    def enabled(self):
        return bool(self.speed or self.spatial or self.noise or self.flip)


def _speed(pos, present, factor):
    """Reamostra no tempo: factor > 1 acelera (menos frames), < 1 desacelera."""
    T = pos.shape[0]
    t_new = np.arange(0.0, T - 1 + 1e-9, factor) if T > 1 else np.zeros(1)
    lo = np.floor(t_new).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (t_new - lo)[:, None, None]
    out = (1.0 - w) * pos[lo] + w * pos[hi]
    # ponto presente só se existe nos dois vizinhos (ou no exato, quando w = 0)
    exact = (t_new == lo)[:, None]
    ok = np.where(exact, present[lo], present[lo] & present[hi])
    out[~ok] = 0.0
    return out, ok


def _spatial_matrix(rng, sigma):
    """Rotação, escala e cisalhamento 2D (no plano XY), sorteados com desvio sigma."""
    a = rng.normal(0.0, sigma)
    s = 1.0 + rng.normal(0.0, sigma)
    sh = rng.normal(0.0, sigma)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    shear = np.array([[1.0, sh], [0.0, 1.0]])
    return s * rot @ shear


def augment(feats, cfg, rng):
    """(T, 115, C) normalizado -> (T', 115, C) aumentado. C = 9 (ou 3, só posições).

    Sem nenhuma augmentation ligada, devolve `feats` inalterado.
    """
    if not cfg.enabled:
        return feats
    feats = np.asarray(feats)
    pos = feats[..., :3].astype(np.float64)
    present = present_mask(pos)

    if cfg.flip and rng.random() < 0.5:
        pos = pos[:, FLIP_PERM].copy()
        present = present[:, FLIP_PERM]
        pos[..., 0] *= -1.0                     # origem = meio dos ombros: espelho em x = 0
    if cfg.spatial:
        M = _spatial_matrix(rng, cfg.spatial)
        pos[..., :2] = pos[..., :2] @ M.T       # linear: ausente (0) continua 0
    if cfg.noise:
        pos = pos + rng.normal(0.0, cfg.noise, pos.shape) * present[..., None]
    if cfg.speed:
        factor = rng.uniform(1.0 - cfg.speed, 1.0 + cfg.speed)
        pos, present = _speed(pos, present, factor)

    pos[~present] = 0.0
    channels = feats.shape[-1]
    if channels == 3:
        return pos.astype(np.float32)
    if channels != 9:
        raise ValueError(f"augment espera 3 ou 9 canais por ponto; recebeu {channels}")
    vel, acc = temporal_features(pos, present)          # mesma regra do Passo 3
    return np.concatenate([pos, vel, acc], axis=-1).astype(np.float32)
