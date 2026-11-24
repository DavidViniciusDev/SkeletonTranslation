#!/usr/bin/env python3
"""Pré-treino do decoder em Gloss2Text (glosas -> texto), só com texto.

Treina o T5 a traduzir as glosas verdadeiras dos manifestos, sem vídeo. O
resultado (<out-dir>/best) é o ponto de partida do treino de SLT, e o BLEU no
teste mede o teto do decoder:

    python3 slt_model.py ... --t5 checkpoints_gloss2text/best

Uso:
    # conferir a instalação (CPU, dados e tokenizer sintéticos, sem download)
    python3 pretrain_decoder.py --smoke-test

    # pré-treino real (mesmo T5 do treino de SLT)
    python3 pretrain_decoder.py --train-manifest train.json --val-manifest dev.json \
        --test-manifest test.json --t5 GermanT5/t5-efficient-gc4-german-base-nl36 \
        --out-dir checkpoints_gloss2text

Código em skeltrans/training/gloss2text.py; guia em docs/decoder/gloss2text.md.
"""
from skeltrans.training.gloss2text import main

if __name__ == "__main__":
    main()
