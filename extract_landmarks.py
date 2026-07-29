#!/usr/bin/env python3
"""Shim de compatibilidade — código real em skeltrans/extraction/steps/.

Mantém o comando `python3 extract_landmarks.py` funcionando após a reorganização (E6).
"""
from skeltrans.extraction.steps.s3_extract_landmarks import main

if __name__ == "__main__":
    main()
