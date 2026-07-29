"""SkeletonTranslation: pipeline de SLT baseado em landmarks sintéticos.

Pacote raiz. Camadas:
  - common/     código compartilhado por extração e treino (layout, texto,
                geometria, I/O de .npy, paralelismo).
  - extraction/ (Etapa 1) da preparação de texto às features de frase.
  - training/   (Etapa 2) modelo Encoder-Decoder (LandmarkEncoder + PTT5).
"""
