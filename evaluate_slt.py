#!/usr/bin/env python3
"""Avaliacao do modelo SLT (BLEU-1..4 e METEOR) no conjunto de teste.

Shim de compatibilidade: o codigo real vive em skeltrans/training/evaluate.py
(e as metricas em skeltrans/training/metrics.py). Este arquivo preserva o
comando `python3 evaluate_slt.py ...`.

Uso:
    python3 evaluate_slt.py \\
        --checkpoint checkpoints/best.pt \\
        --test-manifest data/interim/test.json \\
        --features-dir data/features/sentence_features \\
        --out checkpoints/test_metrics.json
"""
from skeltrans.training.evaluate import main

if __name__ == "__main__":
    main()
