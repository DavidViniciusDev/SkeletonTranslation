#!/usr/bin/env python3
"""Shim de compatibilidade — código real em skeltrans/extraction/steps/.

Mantém o comando `python3 filter_sentences_by_vocab.py` funcionando após a reorganização (E6).
"""
from skeltrans.extraction.steps.s1b_filter_sentences import main

if __name__ == "__main__":
    main()
