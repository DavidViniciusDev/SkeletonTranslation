#!/usr/bin/env python3
"""Mede a taxa de deteccao do MediaPipe nos landmarks extraidos (saida do s3).

Motivacao: o s3 preenche com [0,0,0] todo ponto/componente que o MediaPipe nao
detectou no frame (zero-fill). Num corpus de video limpo e em close-up isso e
raro; no PHOENIX (sinalizador pequeno em 210x260, motion blur) pode atingir uma
fracao grande dos frames — sobretudo as MAOS, que sao justamente o canal com
mais carga linguistica.

Este relatorio quantifica isso ANTES de comprometer horas de extracao, lendo os
.npy (T,115,3) ja gravados e contando, por componente, em quantos frames todos
os pontos daquele bloco sao exatamente zero.

Uso:
    python3 tools/phoenix_detection_report.py --in-dir data_phoenix/raw/landmarks_115
    python3 tools/phoenix_detection_report.py --in-dir ... --per-file   # detalha
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fatias dos 115 pontos: fonte unica do projeto (evita duplicar os limites).
from skeltrans.common.layout import N_HAND, N_POINTS, N_POSE  # noqa: E402

COMPONENTS = {
    "pose": slice(0, N_POSE),
    "mao_esq": slice(N_POSE, N_POSE + N_HAND),
    "mao_dir": slice(N_POSE + N_HAND, N_POSE + 2 * N_HAND),
    "face": slice(N_POSE + 2 * N_HAND, N_POINTS),
}


def present_per_frame(arr, sl):
    """(T,) bool: True nos frames em que o componente `sl` foi detectado.

    Ausencia == bloco inteiro exatamente zero, que e a convencao de zero-fill
    gravada pelo s3. Um ponto isolado em zero nao conta como ausencia.
    """
    return np.any(arr[:, sl, :] != 0.0, axis=(1, 2))


def main():
    ap = argparse.ArgumentParser(description="Taxa de deteccao por componente nos .npy do s3.")
    ap.add_argument("--in-dir", required=True, help="pasta com os .npy (T,115,3) do s3")
    ap.add_argument("--per-file", action="store_true", help="imprime uma linha por arquivo")
    args = ap.parse_args()

    files = sorted(
        f for f in glob.glob(os.path.join(args.in_dir, "*.npy"))
        if not f.endswith("_pose_vis.npy")
    )
    if not files:
        raise SystemExit(f"ERRO: nenhum .npy em {args.in_dir}")

    total_frames = 0
    detected = {k: 0 for k in COMPONENTS}
    both_hands_missing = 0
    seq_lens = []
    bad = 0

    for f in files:
        try:
            arr = np.load(f)
        except Exception as e:  # noqa: BLE001 - um arquivo corrompido nao aborta o relatorio
            print(f"ILEGIVEL: {f}: {e}")
            bad += 1
            continue
        if arr.ndim != 3 or arr.shape[1] != N_POINTS:
            print(f"SHAPE INESPERADO: {f}: {arr.shape} (esperado (T,{N_POINTS},C))")
            bad += 1
            continue

        T = arr.shape[0]
        total_frames += T
        seq_lens.append(T)
        pres = {k: present_per_frame(arr, sl) for k, sl in COMPONENTS.items()}
        for k, v in pres.items():
            detected[k] += int(v.sum())
        both_hands_missing += int(np.sum(~pres["mao_esq"] & ~pres["mao_dir"]))

        if args.per_file:
            rates = " ".join(f"{k}={v.mean():.0%}" for k, v in pres.items())
            print(f"{os.path.basename(f):60s} T={T:4d} {rates}")

    if total_frames == 0:
        raise SystemExit("ERRO: nenhum frame legivel.")

    lens = np.array(seq_lens)
    print(f"\n{'='*64}\nArquivos: {len(files)} (ilegiveis: {bad}) | frames: {total_frames}")
    print(f"Duracao (frames): min={lens.min()} p50={int(np.median(lens))} "
          f"p95={int(np.percentile(lens, 95))} max={lens.max()} media={lens.mean():.1f}")
    print("\nTaxa de deteccao por componente (fracao de frames com o bloco nao-zerado):")
    for k in COMPONENTS:
        r = detected[k] / total_frames
        print(f"  {k:8s} {r:7.1%}   (zero-fill em {total_frames - detected[k]} frames)")
    print(f"\n  AMBAS as maos ausentes: {both_hands_missing/total_frames:.1%} "
          f"({both_hands_missing} frames) <- frames sem nenhuma informacao manual")
    print("=" * 64)


if __name__ == "__main__":
    main()
