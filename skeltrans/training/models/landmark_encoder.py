"""Encoder Espacial-Temporal de landmarks (do zero).

Downsampling temporal (``ds_steps``)
-----------------------------------
Uma frase sintetizada tem tipicamente 500-800 frames, enquanto a saída tem ~15
tokens. A cross-attention do decoder precisa então alinhar cada token contra
centenas de posições de memória quase idênticas entre si (a 30 fps, frames
vizinhos dentro de um mesmo sinal variam pouco) — um regime muito distante
daquele em que o T5 foi pré-treinado, onde 1 posição de origem ≈ 1 unidade de
significado.

``ds_steps`` empilha N convoluções ``stride=2`` que reduzem T por 2^N. Além de
aproximar a razão memória/saída do regime de pré-treino, corta a memória da
self-attention por 4^N (as matrizes (B*nhead, T, T) dominam a VRAM), o que
permite subir o ``--batch-size`` e rodar mais passos por hora de GPU.

``ds_steps=0`` mantém a arquitetura ORIGINAL bit-a-bit (nenhum módulo extra é
criado, nenhuma chave nova aparece no state_dict), servindo como braço de
controle do ablation e preservando a compatibilidade com checkpoints antigos.
"""

import torch
import torch.nn as nn

from skeltrans.common.layout import INPUT_DIM
from skeltrans.training.models.concept_pooling import ConceptPooling, TEMPORAL_MODES
from skeltrans.training.models.graph_projection import GraphProjection, node_presence
from skeltrans.training.models.parts_projection import PartsProjection
from skeltrans.training.models.positional_encoding import SinusoidalPositionalEncoding

# Tipos de projeção de entrada (--encoder-type). 'flat' é o original.
ENCODER_TYPES = ("flat", "parts", "gcn")


def downsampled_lengths(lengths, ds_steps):
    """Comprimentos válidos após ``ds_steps`` convoluções stride=2.

    Espelha exatamente a aritmética de ``Conv1d(kernel_size=5, stride=2,
    padding=2)``, cuja saída é ``floor((L - 1) / 2) + 1 == ceil(L / 2)``.
    Mantém no mínimo 1 posição válida por amostra (sequências degeneradas não
    podem zerar o comprimento, ou o CTC e a cross-attention recebem máscara
    totalmente preenchida).
    """
    for _ in range(ds_steps):
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
    return lengths.clamp(min=1)


class LandmarkEncoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.2, out_dim=None, ds_steps=0,
                 input_norm=False, encoder_type="flat", temporal_mode="stride",
                 concept_layers=2, concept_queries=1):
        super().__init__()
        if encoder_type not in ENCODER_TYPES:
            raise ValueError(f"encoder_type desconhecido: {encoder_type!r} "
                             f"(opcoes: {', '.join(ENCODER_TYPES)}).")
        if temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"temporal_mode desconhecido: {temporal_mode!r} "
                             f"(opcoes: {', '.join(TEMPORAL_MODES)}).")
        self.encoder_type = encoder_type
        # Normalização por coluna da entrada (z-score com média/desvio do
        # treino). Fica em buffers: vai no state_dict e é aplicada sozinha na
        # avaliação/inferência. input_norm=False não cria nada -> state_dict
        # idêntico ao original.
        self.input_norm = bool(input_norm)
        if self.input_norm:
            self.register_buffer("input_mean", torch.zeros(input_dim))
            self.register_buffer("input_std", torch.ones(input_dim))
        # Projeção de entrada: 'flat' = Linear único sobre o frame achatado
        # (original, state_dict inalterado); 'parts' = uma MLP por parte do
        # corpo + fusão (ver parts_projection.py); 'gcn' = camadas GCN sobre o
        # grafo do esqueleto + pooling por parte (ver graph_projection.py).
        # Mesma saída (B, T, d_model).
        if encoder_type == "parts":
            self.input_proj = PartsProjection(input_dim, d_model)
        elif encoder_type == "gcn":
            self.input_proj = GraphProjection(input_dim, d_model)
        else:
            self.input_proj = nn.Linear(input_dim, d_model)
        # Conv1D temporal (kernel=3) preservando o comprimento T
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

        # Pilha de downsampling: ds_steps convolucoes stride=2 (T -> T / 2^ds_steps).
        # Com ds_steps=0 nenhum parametro e criado -> state_dict identico ao original.
        self.ds_steps = int(ds_steps)
        if self.ds_steps < 0:
            raise ValueError("ds_steps deve ser >= 0.")
        self.temporal_ds = nn.ModuleList(
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2)
            for _ in range(self.ds_steps)
        )
        self.ds_act = nn.GELU()
        # LayerNorm apos a pilha estabiliza a escala antes da atencao; so existe
        # quando ha downsampling (Identity nao cria chaves no state_dict).
        self.temporal_norm = nn.LayerNorm(d_model) if self.ds_steps else nn.Identity()

        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        # adaptador para o hidden do decoder (T5), se necessario
        out_dim = out_dim or d_model
        self.adapter = nn.Identity() if out_dim == d_model else nn.Linear(d_model, out_dim)
        # recomputa ativacoes no backward em vez de guarda-las (--grad-checkpoint);
        # nao cria parametros, entao checkpoints .pt continuam identicos
        self.grad_checkpoint = False

        # Modo do eixo temporal (--temporal-mode). 'stride': self.transformer
        # entrega a memoria direto ao decoder (o original). 'concept':
        # self.transformer vira o andar de FRAMES e, depois do pooling por
        # sinal, um segundo Transformer relaciona os sinais entre si (ver
        # concept_pooling.py). Criados por ultimo e so no 'concept': no
        # 'stride' o state_dict e a inicializacao ficam identicos ao original.
        self.temporal_mode = temporal_mode
        if temporal_mode == "concept":
            if concept_layers < 1:
                raise ValueError("concept_layers deve ser >= 1.")
            self.concept_pool = ConceptPooling(d_model, queries=concept_queries)
            self.concept_norm = nn.LayerNorm(d_model)
            self.concept_transformer = nn.TransformerEncoder(layer, num_layers=concept_layers)

    @torch.no_grad()
    def set_input_stats(self, mean, std):
        """Grava média/desvio por coluna (calculados no conjunto de treino)."""
        if not self.input_norm:
            raise RuntimeError("encoder criado sem input_norm=True.")
        self.input_mean.copy_(torch.as_tensor(mean, dtype=self.input_mean.dtype))
        self.input_std.copy_(torch.as_tensor(std, dtype=self.input_std.dtype))

    def downsample_mask(self, pad_mask, t_out):
        """Reconstrói a máscara de padding no comprimento reduzido.

        Deriva de ``pad_mask`` os comprimentos válidos, aplica a mesma aritmética
        das convoluções e remonta a máscara. Reconstruir a partir dos
        comprimentos (em vez de reamostrar a máscara) garante que o
        ``attention_mask`` entregue ao T5 e os ``input_lengths`` do CTC — ambos
        derivados daqui — fiquem consistentes com o tensor real. Divergência
        entre os dois faria o modelo atender a padding sem erro visível.
        """
        lengths = downsampled_lengths((~pad_mask).sum(dim=1), self.ds_steps)
        lengths = lengths.clamp(max=t_out)
        idx = torch.arange(t_out, device=pad_mask.device)
        return idx[None, :] >= lengths[:, None]      # True = padding

    def forward(self, feats, pad_mask, seg_ids=None):
        """feats: (B, T, input_dim) — 1035, ou mais com --extra-features;
        pad_mask: (B, T) True onde e padding.

        Retorna ``(memory, pad_mask)`` — a máscara acompanha a saída porque o
        comprimento temporal muda quando ``ds_steps > 0``. Com ``ds_steps=0`` a
        máscara devolvida é a própria entrada, inalterada.

        No modo 'concept', ``seg_ids`` (B, T') diz a que sinal pertence cada
        frame JÁ REDUZIDO; quem o calcula é o ``SLTModel`` (a partir do CTC sobre
        ``encode_frames``). A memória devolvida tem uma posição por sinal.
        """
        x, pad_mask = self.encode_frames(feats, pad_mask)
        if self.temporal_mode == "concept":
            if seg_ids is None:
                raise ValueError("temporal_mode='concept' exige seg_ids (fronteiras dos "
                                 "sinais); use SLTModel, que as calcula pelo CTC.")
            return self.encode_concepts(x, seg_ids)
        return self.adapter(x), pad_mask            # (B, T', out_dim), (B, T')

    def encode_concepts(self, x, seg_ids):
        """Andar de conceitos: (B, T', d_model) frames -> (B, N*K, out_dim) sinais."""
        c, mask = self.concept_pool(x, seg_ids)
        c = self.concept_norm(c)
        c = self.dropout(self.pos_enc(c))
        c = self.concept_transformer(c, src_key_padding_mask=mask)
        return self.adapter(c), mask

    def encode_frames(self, feats, pad_mask):
        """Andar de frames: projeção, conv temporal, ds-steps e Transformer.

        Devolve ``(x, pad_mask)`` em ``d_model``, ANTES do adaptador. No 'stride'
        o ``forward`` só aplica o adaptador por cima (o caminho original).
        """
        # o 'gcn' precisa saber quais pontos existem; depois do z-score o 0 de
        # um ponto ausente deixa de ser 0, então a presença vem da entrada crua
        present = node_presence(feats) if self.encoder_type == "gcn" else None
        if self.input_norm:
            feats = (feats - self.input_mean) / self.input_std
            # o padding continua exatamente 0 (sem o z-score viraria -media/desvio)
            feats = feats.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        if present is not None:
            # --grad-checkpoint vale também para as camadas GCN (B*T*115 nós)
            self.input_proj.grad_checkpoint = self.grad_checkpoint
            # o GCN roda só nos frames reais (o padding sai com o valor de um
            # frame vazio, idêntico ao que daria calculado)
            x = self.input_proj(feats, present=present, pad_mask=pad_mask)  # (B, T, d_model)
        else:
            x = self.input_proj(feats)                 # (B, T, d_model)
        x = self.temporal_conv(x.transpose(1, 2))  # (B, d_model, T) conv sobre o tempo
        for conv in self.temporal_ds:
            x = self.ds_act(conv(x))               # (B, d_model, T/2) por camada
        x = x.transpose(1, 2)                      # (B, T', d_model)
        x = self.temporal_norm(x)

        if self.ds_steps:
            pad_mask = self.downsample_mask(pad_mask, x.size(1))

        x = self.pos_enc(x)
        x = self.dropout(x)
        if self.grad_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            # camada a camada: evita guardar as matrizes de atencao (B*nhead,T,T),
            # que dominam a VRAM em sequencias longas
            for layer in self.transformer.layers:
                x = checkpoint(layer, x, None, pad_mask, use_reentrant=False)
        else:
            x = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, T', d_model)
        return x, pad_mask                          # (B, T', d_model), (B, T')
