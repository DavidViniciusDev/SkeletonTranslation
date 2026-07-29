#!/usr/bin/env python3
"""Regressão de E3: a normalização unificada (skeltrans.common.geometry) deve
reproduzir, byte-a-byte, as saídas originais geradas antes da refatoração.

Referência (verdade): a árvore original SkeLTrans/, que permanece intacta.
  - landmarks_115_norm9/  <- saída do Passo 3 (normalize_array)

Uso:
    python3 tests/test_geometry_regression.py          # roda a checagem
    (também compatível com: pytest tests/test_geometry_regression.py)

Se os dados de referência não existirem, o teste é PULADO (não falha).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skeltrans.common.geometry import normalize_array  # noqa: E402

RAW_DIR = os.path.join(ROOT, "data", "raw", "landmarks_115")
REF_DIR = "/libras/Doutorado/SkeLTrans/landmarks_115_norm9"
N_SAMPLES = 25  # amostragem espaçada sobre landmarks_115


def _samples():
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".npy"))
    if not files:
        return []
    step = max(1, len(files) // N_SAMPLES)
    return files[::step][:N_SAMPLES]


def test_normalize_matches_reference():
    if not os.path.isdir(REF_DIR):
        print(f"SKIP: referência ausente ({REF_DIR})")
        return
    samples = _samples()
    assert samples, f"sem .npy em {RAW_DIR}"

    checked = mismatched = 0
    for name in samples:
        ref_path = os.path.join(REF_DIR, name)
        if not os.path.exists(ref_path):
            continue
        raw = np.load(os.path.join(RAW_DIR, name)).astype(np.float32)
        got = normalize_array(raw, origin="shoulders")
        ref = np.load(ref_path)
        checked += 1
        if not (got.shape == ref.shape and np.array_equal(got, ref)):
            mismatched += 1
            print(f"DIFERE: {name} shapes={got.shape}/{ref.shape}")

    assert checked > 0, "nenhuma amostra tinha referência para comparar"
    assert mismatched == 0, f"{mismatched}/{checked} amostras divergiram da referência"
    print(f"OK: {checked} amostras idênticas à referência ✔")


if __name__ == "__main__":
    test_normalize_matches_reference()
