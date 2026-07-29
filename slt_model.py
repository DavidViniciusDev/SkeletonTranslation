#!/usr/bin/env python3
"""Passo 5 — Modelo de SLT (LandmarkEncoder + decoder PTT5).

Shim de compatibilidade: o código foi decomposto em skeltrans/training/
(config, models/, data/, device, checkpoint, trainer, diagnostics, cli).
Este arquivo preserva o comando original.

Uso:
    # validacao rapida da arquitetura (CPU, dados sinteticos, T5 minusculo)
    python3 slt_model.py --smoke-test

    # treino real (depois do Passo 4)
    python3 slt_model.py --train-manifest train.json --val-manifest val.json \
        --epochs 30 --batch-size 8 --out-dir checkpoints
"""
from skeltrans.training.cli import main

if __name__ == "__main__":
    main()
