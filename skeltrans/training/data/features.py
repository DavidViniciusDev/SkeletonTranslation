"""Features extras de entrada, calculadas sob demanda a partir das posições.

As features do Passo 3 (X,Y,Z + velocidade + aceleração dos 115 pontos) são
mantidas intactas; cada grupo abaixo ACRESCENTA colunas ao fim do vetor de
1035 de cada frame. Os grupos seguem os parâmetros fonológicos da Libras:

    A  configuração de mão  mão centrada no punho e dividida pelo tamanho da
                            palma (63), flexão nas 15 articulações dos dedos
                            (radianos) e distâncias entre as 5 pontas ÷ palma
                            (10) — por mão                              176
    B  orientação da palma  vetor normal unitário da palma, por mão       6
    C  ponto de articulação punho e ponta do indicador até nariz, boca,
                            peito e ombros (5 refs × 2 pontos × 2 mãos)
                            + distância punho-punho                      21
    D  movimento            velocidade escalar de cada punho              2
    E  não manuais          abertura da boca, dos olhos e altura das
                            sobrancelhas (razões entre distâncias)         5
    F  presença             1/0 por grupo detectado: pose, mão esq,
                            mão dir, face                                 4

A, B e E são razões ou ângulos: não dependem da escala do frame. Quando a mão
(ou a face) não foi detectada, as features dela ficam em 0; o grupo F diz à
rede que esse 0 significa "ausente".

Calcular aqui, no Dataset, em vez de gravar nos .npy, permite o ablation de
grupos sem reextrair a base: cada combinação é só uma flag (--extra-features).
"""

import numpy as np

from skeltrans.common.layout import (FACE, FACE_MASK, INPUT_DIM, L_SHOULDER,
                                     LEFT_HAND, N_POINTS, NOSE, POSE, R_SHOULDER,
                                     RIGHT_HAND)

EPS = 1e-6

# Dimensão acrescentada por grupo (a ordem das colunas segue a ordem das chaves).
GROUP_DIMS = {"A": 176, "B": 6, "C": 21, "D": 2, "E": 5, "F": 4}
GROUP_NAMES = {"A": "configuracao de mao", "B": "orientacao da palma",
               "C": "ponto de articulacao", "D": "movimento",
               "E": "nao manuais", "F": "presenca"}

# --- geometria da mão (índices do MediaPipe Hands, base 0 = punho) --- #
_FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
# 15 articulações: (anterior, vértice, seguinte) em cada dedo
_JOINTS = np.array([(f[i - 1], f[i], f[i + 1]) for f in _FINGERS for i in (1, 2, 3)])
_TIPS = [4, 8, 12, 16, 20]
_TIP_PAIRS = np.array([(a, b) for i, a in enumerate(_TIPS) for b in _TIPS[i + 1:]])
_WRIST, _INDEX_MCP, _MIDDLE_MCP, _PINKY_MCP, _INDEX_TIP = 0, 5, 9, 17, 8


def _face(mesh_id):
    """Posição, no vetor de 115 pontos, de um índice do FaceMesh da máscara."""
    return FACE.start + FACE_MASK.index(mesh_id)


def _norm(v):
    return np.linalg.norm(v, axis=-1)


def _point_ok(p):
    """(T,3) -> (T,) True onde o ponto foi detectado (ausente = [0,0,0])."""
    return np.abs(p).sum(axis=-1) > 0


def _group_ok(pts):
    """(T,K,3) -> (T,) True onde algum ponto do grupo foi detectado."""
    return np.abs(pts).sum(axis=(1, 2)) > 0


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def parse_groups(spec):
    """'A,f' / ['F','A'] / '' -> tupla canônica ordenada, ex.: ('A', 'F')."""
    if not spec:
        return ()
    items = spec if isinstance(spec, (list, tuple)) else str(spec).split(",")
    items = [s.strip().upper() for s in items if s and s.strip()]
    bad = sorted(set(items) - set(GROUP_DIMS))
    if bad:
        raise ValueError(f"grupo(s) de feature desconhecido(s): {bad}. "
                         f"Validos: {', '.join(GROUP_DIMS)}")
    return tuple(sorted(set(items)))


def format_groups(groups):
    """Inverso de parse_groups: ('A','F') -> 'A,F' (formato gravado no config)."""
    return ",".join(parse_groups(groups))


def feature_dim(groups=()):
    """Dimensão total da entrada do encoder: 1035 + colunas dos grupos."""
    return INPUT_DIM + sum(GROUP_DIMS[g] for g in parse_groups(groups))


def compute_extra(feats, groups):
    """Calcula as colunas extras de uma frase.

    feats: (T, 115, C) com C >= 3 (posições nos 3 primeiros canais; com C >= 6,
    a velocidade do Passo 3 nos canais 3:6). Retorna (T, D_extra) float32.
    """
    groups = parse_groups(groups)
    T = feats.shape[0]
    if not groups:
        return np.zeros((T, 0), dtype=np.float32)
    pos = np.asarray(feats[..., :3], dtype=np.float64)
    hands = [pos[:, LEFT_HAND], pos[:, RIGHT_HAND]]                      # 2 x (T,21,3)
    palm = [_norm(h[:, _MIDDLE_MCP] - h[:, _WRIST]) for h in hands]      # tamanho da palma
    hand_ok = [_group_ok(h) & (p > EPS) for h, p in zip(hands, palm)]
    out = []

    if "A" in groups:
        for h, p, ok in zip(hands, palm, hand_ok):
            ps = np.where(ok, p, 1.0)[:, None]                           # (T,1)
            rel = (h - h[:, :1]) / ps[:, :, None]                        # (T,21,3)
            v1 = h[:, _JOINTS[:, 0]] - h[:, _JOINTS[:, 1]]
            v2 = h[:, _JOINTS[:, 2]] - h[:, _JOINTS[:, 1]]
            # atan2(|v1 x v2|, v1.v2): estável perto de 0 e pi e sem EPS no
            # denominador (arccos(dot/(|v1||v2|+EPS)) enviesa ossos curtos)
            ang = np.arctan2(_norm(np.cross(v1, v2)), (v1 * v2).sum(-1))  # (T,15), pi = esticado
            tips = _norm(h[:, _TIP_PAIRS[:, 0]] - h[:, _TIP_PAIRS[:, 1]]) / ps  # (T,10)
            block = np.concatenate([rel.reshape(T, -1), ang, tips], axis=1)
            out.append(block * ok[:, None])

    if "B" in groups:
        for h, ok in zip(hands, hand_ok):
            n = np.cross(h[:, _INDEX_MCP] - h[:, _WRIST], h[:, _PINKY_MCP] - h[:, _WRIST])
            nn = _norm(n)[:, None]
            valid = ok[:, None] & (nn > EPS)
            out.append(np.where(valid, n / np.maximum(nn, EPS), 0.0))

    if "C" in groups:
        lip_up, lip_low = pos[:, _face(13)], pos[:, _face(14)]
        ls, rs = pos[:, POSE.start + L_SHOULDER], pos[:, POSE.start + R_SHOULDER]
        nose = pos[:, POSE.start + NOSE]
        refs = [(nose, _point_ok(nose)),
                ((lip_up + lip_low) / 2, _point_ok(lip_up) & _point_ok(lip_low)),
                ((ls + rs) / 2, _point_ok(ls) & _point_ok(rs)),                  # peito
                (ls, _point_ok(ls)), (rs, _point_ok(rs))]
        for h, ok in zip(hands, hand_ok):
            cols = [_norm(h[:, k] - ref) * (ok & ref_ok)
                    for k in (_WRIST, _INDEX_TIP) for ref, ref_ok in refs]
            out.append(np.stack(cols, axis=1))
        both = hand_ok[0] & hand_ok[1]
        out.append((_norm(hands[0][:, _WRIST] - hands[1][:, _WRIST]) * both)[:, None])

    if "D" in groups:
        if feats.shape[-1] >= 6:
            vel = np.asarray(feats[..., 3:6], dtype=np.float64)
        else:                                       # só posições: diferença finita
            vel = np.zeros_like(pos)
            vel[1:] = pos[1:] - pos[:-1]
        for sl, ok in zip((LEFT_HAND, RIGHT_HAND), hand_ok):
            out.append((_norm(vel[:, sl.start + _WRIST]) * ok)[:, None])

    if "E" in groups:
        face_ok = _group_ok(pos[:, FACE])

        def ratio(a, b, c, d):
            """|a-b| / |c-d| entre pontos do FaceMesh; 0 sem face ou sem escala."""
            den = _norm(pos[:, _face(c)] - pos[:, _face(d)])
            num = _norm(pos[:, _face(a)] - pos[:, _face(b)])
            return np.where(face_ok & (den > EPS), num / np.maximum(den, EPS), 0.0)

        out.append(np.stack([
            ratio(13, 14, 61, 291),      # abertura da boca ÷ largura da boca
            ratio(159, 145, 33, 133),    # abertura do olho esq ÷ largura do olho
            ratio(386, 374, 362, 263),   # abertura do olho dir
            ratio(105, 159, 33, 133),    # sobrancelha esq -> pálpebra ÷ largura do olho
            ratio(334, 386, 362, 263),   # sobrancelha dir
        ], axis=1))

    if "F" in groups:
        out.append(np.stack([_group_ok(pos[:, sl]) for sl in (POSE, LEFT_HAND, RIGHT_HAND, FACE)],
                            axis=1).astype(np.float64))

    extra = np.concatenate(out, axis=1)
    assert extra.shape[1] == sum(GROUP_DIMS[g] for g in groups), extra.shape
    return extra.astype(np.float32)


def build_input(feats, groups=()):
    """(T,115,C) ou achatado (T,115*C) -> (T, D) float32 pronto para o encoder:
    as colunas originais (achatadas) seguidas das extras dos `groups`."""
    arr = np.asarray(feats, dtype=np.float32)
    groups = parse_groups(groups)
    if arr.ndim == 2 and groups:
        arr = arr.reshape(arr.shape[0], N_POINTS, -1)
    base = arr.reshape(arr.shape[0], -1) if arr.ndim == 3 else arr
    if not groups:
        return base
    return np.concatenate([base, compute_extra(arr, groups)], axis=1)


def compute_input_stats(dataset, num_workers=0):
    """Média e desvio por coluna sobre TODOS os frames reais de `dataset`
    (o de treino: a validação nunca entra nas estatísticas).

    Colunas constantes (desvio ~0) ficam com desvio 1, para não dividir por
    zero nem amplificar ruído numérico. Devolve (mean, std) float32 (D,).
    """
    from torch.utils.data import DataLoader

    s = ss = None
    n = 0
    # batch_size=None: o DataLoader só paraleliza o __getitem__ (sem collate)
    for item in DataLoader(dataset, batch_size=None, num_workers=num_workers):
        x = item[0].numpy().astype(np.float64)        # (T, D)
        if s is None:
            s, ss = np.zeros(x.shape[1]), np.zeros(x.shape[1])
        s += x.sum(axis=0)
        ss += (x * x).sum(axis=0)
        n += x.shape[0]
    if not n:
        raise ValueError("dataset vazio: impossivel calcular as estatisticas de entrada.")
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean * mean, 0.0))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)
