"""Normalização geométrica + dinâmica temporal dos landmarks (Passo 3).

Fonte ÚNICA da matemática de normalização. Antes, `normalize_array` e seus
auxiliares viviam em normalize_landmarks.py e o build_sentence_features.py os
importava de lá (dependência frágil script→script). Agora os dois passos
importam daqui.

Transformações (ver plan.md, Passo 3):
  1. Centralização: origem = ponto médio dos ombros (padrão) ou o nariz.
  2. Escalonamento: divide pela distância euclidiana entre os ombros.
  3. Dinâmica: velocidade V_t = P_t - P_{t-1} e aceleração A_t = V_t - V_{t-1}.
  4. Concatenação -> 9 canais por ponto: (T,115,3) -> (T,115,9).

Tratamento de ausentes: pontos ausentes vêm como [0,0,0] (Passo 2). A máscara de
presença é calculada da ENTRADA crua e reimposta no fim; a dinâmica só é válida
quando o ponto está presente nos frames envolvidos.
"""

import numpy as np

from skeltrans.common.layout import L_SHOULDER, NOSE, R_SHOULDER

EPS = 1e-6


def present_mask(arr):
    """(T,115) bool: True onde o ponto NAO e o vetor zero (ausente)."""
    return np.any(arr != 0.0, axis=2)


def present_mask_frame(frame):
    """(115,) bool: True onde o ponto do frame NAO e o vetor zero (ausente)."""
    return np.any(frame != 0.0, axis=1)


def present_mask_frame_stack(y):
    """(N,115) bool para um stack (N,115,3)."""
    return np.any(y != 0.0, axis=2)


def centralize_and_scale(arr, present, origin="shoulders"):
    """Aplica centralizacao e escalonamento in-place-safe. Retorna (T,115,3)."""
    T = arr.shape[0]
    out = arr.astype(np.float64).copy()

    # --- origem por frame ---
    if origin == "nose":
        org = out[:, NOSE, :].copy()                      # (T,3)
        org_ok = present[:, NOSE]                          # (T,)
    else:  # shoulders midpoint
        ls, rs = out[:, L_SHOULDER, :], out[:, R_SHOULDER, :]
        org = (ls + rs) / 2.0
        org_ok = present[:, L_SHOULDER] & present[:, R_SHOULDER]
    # frames sem origem valida: nao translada (origem = 0)
    org[~org_ok] = 0.0
    out = out - org[:, None, :]                            # (T,115,3)

    # --- escala por frame (distancia entre ombros) ---
    ls, rs = arr[:, L_SHOULDER, :], arr[:, R_SHOULDER, :]  # usa cru (invariante a translacao)
    dist = np.linalg.norm(ls - rs, axis=1)                 # (T,)
    sh_ok = present[:, L_SHOULDER] & present[:, R_SHOULDER] & (dist > EPS)
    # substitui escalas invalidas pela ultima valida (ou 1.0)
    scale = np.ones(T, dtype=np.float64)
    last = 1.0
    for t in range(T):
        if sh_ok[t]:
            last = dist[t]
        scale[t] = last
    out = out / scale[:, None, None]
    return out


def temporal_features(pos, present):
    """Calcula velocidade e aceleracao (T,115,3) com mascara de presenca."""
    T = pos.shape[0]
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)

    vel[1:] = pos[1:] - pos[:-1]
    # velocidade valida so quando presente em t e t-1
    vel_valid = np.zeros((T, pos.shape[1]), dtype=bool)
    vel_valid[1:] = present[1:] & present[:-1]
    vel[~vel_valid] = 0.0

    acc[1:] = vel[1:] - vel[:-1]
    acc_valid = np.zeros_like(vel_valid)
    acc_valid[1:] = vel_valid[1:] & vel_valid[:-1]
    acc[~acc_valid] = 0.0
    return vel, acc


def normalize_array(arr, origin="shoulders"):
    """(T,115,3) -> (T,115,9) normalizado."""
    present = present_mask(arr)
    pos = centralize_and_scale(arr, present, origin=origin)
    # reimpoe ausentes como zero na posicao
    pos[~present] = 0.0
    vel, acc = temporal_features(pos, present)
    feats = np.concatenate([pos, vel, acc], axis=2)        # (T,115,9)
    return feats.astype(np.float32)
