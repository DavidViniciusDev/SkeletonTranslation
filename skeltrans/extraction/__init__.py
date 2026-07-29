"""Etapa 1 — extração: da preparação de texto às features de frase.

Ordem do pipeline (ver skeltrans.extraction.config.STEPS):
  s1a build_vocab            CSV de glosas        -> vocabulário .txt
  s1b filter_sentences       glosas + vocab vídeo -> sentenças no vocab
  s2  build_sentence_videos  sentenças            -> sequências de vídeos
  s3  extract_landmarks      vídeos .mp4 OU frames -> landmarks (T,115,3)
  s4  normalize_landmarks    landmarks brutos     -> features (T,115,9)
  s5  build_sentence_features vídeos + landmarks  -> features de frase + manifesto
"""
