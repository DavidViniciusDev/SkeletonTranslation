"""Etapa 2 — modelo de SLT (LandmarkEncoder + decoder PTT5) e treino.

Camadas:
  config     dataclasses de hiperparâmetros (ModelConfig/DataConfig/TrainConfig)
  models/    SinusoidalPositionalEncoding, LandmarkEncoder, SLTModel
  data/      LandmarkTextDataset, make_collate
  device     resolução de dispositivo (cuda/cpu)
  checkpoint salvar/carregar checkpoints
  trainer    Trainer (fit / evaluate)
  diagnostics smoke-test da arquitetura
  cli        interface de linha de comando
"""
