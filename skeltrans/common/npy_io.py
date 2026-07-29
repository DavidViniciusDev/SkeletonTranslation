"""I/O e validação de arrays de landmarks (.npy).

Centraliza a checagem de shape (T, N_POINTS, n_coords) que estava repetida em
normalize_landmarks.py e build_sentence_features.py.
"""

import numpy as np

from skeltrans.common.layout import N_POINTS


def load_points(path, *, n_coords=3, dtype=None):
    """Carrega um .npy de landmarks e valida shape (T, N_POINTS, n_coords).

    dtype: se informado, converte o array (ex.: np.float64).
    Levanta ValueError se o shape for inesperado; FileNotFoundError se ausente.
    """
    arr = np.load(path)
    if dtype is not None:
        arr = arr.astype(dtype)
    if arr.ndim != 3 or arr.shape[1] != N_POINTS or arr.shape[2] != n_coords:
        raise ValueError(
            f"shape inesperado {arr.shape} em {path}, esperado (T,{N_POINTS},{n_coords})")
    return arr
