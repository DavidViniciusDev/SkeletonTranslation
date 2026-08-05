"""Configuração do modelo/treino em dataclasses.

Centraliza os defaults que antes viviam espalhados entre os `__init__` das
redes e os defaults do argparse. O cli.py traduz os argumentos para estas
estruturas; o núcleo (models/trainer) passa a não depender do argparse.
"""

from dataclasses import dataclass
from typing import Optional

# Decoder pré-treinado padrão (PTT5-base em português).
PTT5_NAME = "unicamp-dl/ptt5-base-portuguese-vocab"


@dataclass
class ModelConfig:
    """Arquitetura: LandmarkEncoder + decoder T5."""
    t5_name: str = PTT5_NAME
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.2
    # --- supervisão auxiliar de reconhecimento (CTC de glosas) --- #
    # use_ctc=False mantém a arquitetura original (nenhum head extra criado).
    # gloss_vocab_size é preenchido automaticamente pelo build_and_train quando
    # o CTC é ativado (tamanho inclui <blank> e <unk>).
    use_ctc: bool = False
    gloss_vocab_size: int = 0


@dataclass
class DataConfig:
    """Fontes de dados e parâmetros do DataLoader."""
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    features_dir: Optional[str] = None
    batch_size: int = 8
    num_workers: int = 4
    max_text_len: int = 64


@dataclass
class TrainConfig:
    """Otimização, agenda e checkpointing."""
    epochs: int = 30
    lr: float = 3e-5
    warmup_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 50
    out_dir: str = "checkpoints"
    device: Optional[str] = None
    # --- early stopping (patience=0 -> desligado, treina todas as épocas) --- #
    patience: int = 0
    min_delta: float = 0.0
    # --- peso da perda auxiliar de CTC (só aplicado quando use_ctc=True) --- #
    ctc_weight: float = 0.3
    # --- otimizações de VRAM: offload do encoder T5, bf16, AdamW 8-bit --- #
    # Desligado por padrão (comportamento original). Ver LOW_VRAM.md.
    low_vram: bool = False
