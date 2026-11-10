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
    # --- downsampling temporal do encoder --- #
    # Número de convoluções stride=2 antes da pilha Transformer: T -> T / 2^n.
    # Padrão 2 (4x): aproxima a razão frames/unidade-lexical do regime em que o
    # T5 foi pré-treinado e corta a memória de self-attention por 16.
    # ds_steps=0 restaura a arquitetura original (braço de controle do ablation).
    ds_steps: int = 2
    # --- supervisão auxiliar de reconhecimento (CTC de glosas) --- #
    # use_ctc=False mantém a arquitetura original (nenhum head extra criado).
    # gloss_vocab_size é preenchido automaticamente pelo build_and_train quando
    # o CTC é ativado (tamanho inclui <blank> e <unk>).
    use_ctc: bool = False
    gloss_vocab_size: int = 0
    # --- entrada do encoder --- #
    # Grupos de features extras calculados sob demanda ("A,F", ...; ver
    # training/data/features.py). "" mantém a entrada original de 1035 colunas.
    extra_features: str = ""
    # z-score por coluna com média/desvio do TREINO, guardados no checkpoint.
    input_norm: bool = False
    # Projeção de entrada do encoder: "flat" (Linear único sobre o frame, o
    # original), "parts" (uma MLP por parte do corpo + fusão) ou "gcn"
    # (convolução em grafo sobre o esqueleto + pooling por parte).
    encoder_type: str = "flat"
    # --- eixo temporal do encoder (ver models/concept_pooling.py) --- #
    # "stride" (o original): o --ds-steps junta frames vizinhos às cegas.
    # "concept": uma posição de memória por SINAL, com fronteiras vindas do CTC
    # (exige use_ctc). concept_layers = Transformer entre sinais; concept_queries
    # = vetores por sinal; concept_boundaries = fonte das fronteiras no treino
    # ("align": alinhamento forçado com as glosas; "greedy": argmax do CTC);
    # concept_greedy_prob = chance de usar o greedy no treino mesmo com glosas.
    temporal_mode: str = "stride"
    concept_layers: int = 2
    concept_queries: int = 1
    concept_boundaries: str = "align"
    concept_greedy_prob: float = 0.0


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
    lr: float = 1e-4
    # lr do encoder de landmarks (+ head CTC), que começa do zero; o decoder T5
    # pré-treinado fica com `lr`. None -> um lr só para tudo (o original).
    encoder_lr: Optional[float] = None
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
    # --- gradient checkpointing: recomputa ativações no backward --- #
    # Corta drasticamente a VRAM de ativações (sequências longas) ao custo de
    # ~25-30% de velocidade. Desligado por padrão. Ver LOW_VRAM.md.
    grad_checkpoint: bool = False
    # --- métricas de tradução (BLEU/METEOR) na VALIDAÇÃO durante o treino --- #
    # eval_every=0 desliga (comportamento original). Por padrão são só
    # informativas; com select_by != "val_loss" passam a decidir best.pt e o
    # early stopping.
    eval_every: int = 0
    eval_beams: int = 4          # 1 = greedy (rápido); o evaluate_slt.py usa 4
    eval_max_samples: int = 0    # 0 = validação inteira; N = só os N primeiros
    eval_max_new_tokens: int = 64
    # --- critério de seleção do best.pt e do early stopping --- #
    # "val_loss" (padrão) mantém o comportamento original. Uma métrica de
    # SELECT_METRICS exige eval_every > 0; aí min_delta é medido em pontos da
    # métrica e o checkpoint de menor val_loss vai para best_loss.pt.
    select_by: str = "val_loss"
    # --- regularização da perda de tradução --- #
    # Label smoothing na cross-entropy do decoder (só no treino; a val_loss
    # continua sendo a CE pura). 0.0 = desligado (o original).
    label_smoothing: float = 0.0
    # --- reprodutibilidade --- #
    # Semente de random/numpy/torch e da ordem dos lotes. None = sem semente
    # (o original). Na GPU o resultado fica muito próximo, mas não bit a bit:
    # algumas operações CUDA (ex.: o backward do CTC) não são determinísticas.
    seed: Optional[int] = None


# Critérios aceitos por TrainConfig.select_by -> chave no dict de métricas de
# compute_metrics. val_loss (None) é "menor é melhor"; as métricas, "maior é melhor".
SELECT_METRICS = {
    "val_loss": None,
    "bleu1": "BLEU-1",
    "bleu2": "BLEU-2",
    "bleu3": "BLEU-3",
    "bleu4": "BLEU-4",
    "meteor": "METEOR",
}
