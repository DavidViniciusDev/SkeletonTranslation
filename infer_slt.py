#!/usr/bin/env python3
"""Inferencia SLT ponta-a-ponta: video .mp4 -> texto em portugues.

Shim de compatibilidade: o codigo real vive em skeltrans/training/infer.py.
Este arquivo preserva o comando `python3 infer_slt.py ...`.

Uso:
    python3 infer_slt.py \\
        --checkpoint checkpoints/best.pt \\
        --video /caminho/para/frase.mp4

    # varios videos de uma vez:
    python3 infer_slt.py --checkpoint checkpoints/best.pt --video a.mp4 b.mp4 c.mp4
"""
from skeltrans.training.infer import main

if __name__ == "__main__":
    main()
