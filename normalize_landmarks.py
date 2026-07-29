#!/usr/bin/env python3
"""Shim de compatibilidade — código real em skeltrans/extraction/steps/.

Mantém o comando `python3 normalize_landmarks.py` funcionando após a reorganização (E6).
"""
from skeltrans.extraction.steps.s4_normalize_landmarks import main

if __name__ == "__main__":
    main()
