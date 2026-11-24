#!/usr/bin/env python3
"""Pré-treino do encoder de landmarks por masked pose modeling.

Esconde trechos dos keypoints e treina o encoder a reconstruí-los, sem usar
glosas nem traduções. O resultado (<out-dir>/encoder.pt) é o ponto de
partida do treino de SLT:

    python3 slt_model.py ... --init-encoder checkpoints_pretrain/encoder.pt

Uso:
    # conferir a instalação (CPU, dados aleatórios)
    python3 pretrain_encoder.py --smoke-test

    # pré-treino real (mesmas flags de encoder do treino de SLT)
    python3 pretrain_encoder.py --manifests train.json outro.json \
        --val-manifest dev.json --features-dir feats \
        --extra-features A,B,C,D,E,F --input-norm --out-dir checkpoints_pretrain

Código em skeltrans/training/pretrain.py; guia em docs/encoders/pretrain.md.
"""
from skeltrans.training.pretrain import main

if __name__ == "__main__":
    main()
