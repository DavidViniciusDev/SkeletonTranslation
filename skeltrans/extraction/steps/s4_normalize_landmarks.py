#!/usr/bin/env python3
"""Passo 3 do plano: Engenharia de Caracteristicas (Normalizacao Geometrica).

Le os landmarks extraidos (arquivos .npy com shape (T, 115, 3)) e produz, para
cada um, uma matriz de caracteristicas (T, 115, 9) contendo:
    [X, Y, Z, Vx, Vy, Vz, Ax, Ay, Az]

Transformacoes aplicadas (por frame, exceto as derivadas que sao ao longo de T):
  1. Centralizacao: origem = ponto medio entre os ombros (padrao) ou o nariz;
     subtraida de todos os 115 pontos do frame.
  2. Escalonamento: divide todas as coordenadas pela distancia euclidiana entre
     o ombro esquerdo e o direito (invariancia de escala/camera).
  3. Dinamica temporal: Velocidade V_t = P_t - P_{t-1} e Aceleracao
     A_t = V_t - V_{t-1}.
  4. Concatenacao -> 9 canais por ponto.

Tratamento de ausentes (pontos preenchidos com [0,0,0] no Passo 2):
  - A mascara de "presente" e calculada a partir da ENTRADA crua (antes de
    normalizar) e reimposta no fim: pontos ausentes permanecem [0,0,0] na
    posicao, evitando que a centralizacao os "ressuscite".
  - Velocidade so e valida quando o ponto esta presente em t e em t-1; caso
    contrario e zerada (evita saltos fantasma). Aceleracao exige presenca em
    t, t-1 e t-2. Isso propaga a mascara para a dinamica.

Guard de divisao por zero: se os ombros estiverem ausentes (pose nao detectada)
a distancia ~ 0; nesse frame usa-se a ultima escala valida (ou 1.0 se nao houver).

Layout dos 115 pontos (mesmo do Passo 2):
    [  0: 33) Pose | [33:54) Mao esq | [54:75) Mao dir | [75:115) Face
Indices de referencia na pose: nariz=0, ombro esq=11, ombro dir=12.

Processamento paralelo: usa um pool de PROCESSOS (padrao 8) para normalizar os
arquivos em paralelo, com barra de progresso ao vivo.

Uso:
    python3 normalize_landmarks.py                       # landmarks_115 -> landmarks_115_norm9
    python3 normalize_landmarks.py --workers 4           # nº de processos (padrao: 8)
    python3 normalize_landmarks.py --origin nose
    python3 normalize_landmarks.py --file X.npy --out-dir /tmp/out
    python3 normalize_landmarks.py --overwrite
"""
import argparse
import glob
import os

import numpy as np

# Caminhos default centralizados em skeltrans/extraction/config.py
from skeltrans.extraction.config import PATHS

IN_DIR = PATHS.landmarks_dir
OUT_DIR = PATHS.landmarks_norm_dir

# Layout, geometria, I/O e paralelismo: fonte unica em skeltrans/common/
from skeltrans.common.geometry import normalize_array  # noqa: E402  (Passo 3, reutilizavel)
from skeltrans.common.npy_io import load_points  # noqa: E402
from skeltrans.common.parallel import fmt_eta, run_pool  # noqa: E402


def _normalize_one(task):
    """Normaliza um arquivo (executado em processo worker). Retorna (src, status, info)."""
    src, dst, origin = task
    try:
        arr = load_points(src, n_coords=3)
        feats = normalize_array(arr, origin=origin)
        np.save(dst, feats)
        return (src, "ok", feats.shape)
    except Exception as e:  # noqa: BLE001 - reporta erro por arquivo sem derrubar o pool
        return (src, "error", str(e))


def main():
    ap = argparse.ArgumentParser(description="Passo 3: normalizacao geometrica + dinamica (T,115,3)->(T,115,9).")
    ap.add_argument("--in-dir", default=IN_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--file", default=None, help="normaliza um unico .npy")
    ap.add_argument("--origin", choices=["shoulders", "nose"], default="shoulders",
                    help="origem da centralizacao (padrao: ponto medio dos ombros)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="numero de processos paralelos (padrao: 8)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.file:
        files = [args.file]
    else:
        # ignora arquivos auxiliares de visibilidade
        files = sorted(
            f for f in glob.glob(os.path.join(args.in_dir, "*.npy"))
            if not f.endswith("_pose_vis.npy")
        )

    # Pre-filtragem (resume) no processo principal: monta a lista de tarefas.
    tasks = []
    skipped = 0
    for f in files:
        dst = os.path.join(args.out_dir, os.path.basename(f))
        if not args.overwrite and os.path.exists(dst):
            skipped += 1
            continue
        tasks.append((f, dst, args.origin))

    total = len(tasks)
    workers = max(1, min(args.workers, total)) if total else 1
    print(f"Arquivos encontrados: {len(files)} | a normalizar: {total} | "
          f"pulados(ja existiam): {skipped} | origem: {args.origin} | "
          f"processos: {workers} | saida: {args.out_dir}")

    if total == 0:
        print("Nada a normalizar.")
        return

    ok, failed, elapsed = run_pool(tasks, _normalize_one, workers=workers, unit="arq/s")
    print(f"\nConcluido em {fmt_eta(elapsed)}. normalizados={ok} falhas={failed} "
          f"pulados(ja existiam)={skipped}")


if __name__ == "__main__":
    main()
