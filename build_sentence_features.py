#!/usr/bin/env python3
"""Shim de compatibilidade — código real em skeltrans/extraction/steps/.

Mantém o comando `python3 build_sentence_features.py` funcionando após a reorganização (E6).
"""
from skeltrans.extraction.steps.s5_build_sentence_features import main

if __name__ == "__main__":
    main()
