"""Normalização de texto compartilhada pela pipeline de extração.

Antes, a função `norm()` estava duplicada, idêntica, em
filter_sentences_by_vocab.py e build_sentence_videos.py. Passa a viver aqui.
"""

import unicodedata


def norm(s: str) -> str:
    """Maiúsculas, sem acentos, espaços normalizados.

    O vocabulário de vídeos do V-LIBRASIL é ASCII e maiúsculo; esta função
    coloca qualquer glosa na mesma forma para comparação por igualdade.
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())
