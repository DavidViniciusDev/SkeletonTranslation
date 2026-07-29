#!/usr/bin/env python3
"""Shim de compatibilidade — código real em skeltrans/extraction/steps/.

Mantém o comando `python3 build_vocab.py` funcionando após a reorganização (E6).
"""
from skeltrans.extraction.steps.s1a_build_vocab import main

if __name__ == "__main__":
    main()
