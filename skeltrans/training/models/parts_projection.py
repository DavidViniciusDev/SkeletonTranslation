"""Projeção de entrada por partes do corpo (``--encoder-type parts``).

No encoder original (``flat``), os 1035 números de cada frame passam por um
único ``Linear(1035, d_model)``: para essa camada, dois dedos vizinhos não têm
relação maior do que um dedo e uma sobrancelha. Aqui, cada parte do corpo
ganha uma pequena rede própria antes da junção:

    pose     (33 pts x 9) -> MLP ->┐
    mão esq  (21 pts x 9) -> MLP ->│
    mão dir  (21 pts x 9) -> MLP ->├-> concatena -> Linear -> (d_model)
    face     (40 pts x 9) -> MLP ->│
    extras   (--extra-features) -> MLP ->┘   (só quando há extras)

Cada MLP é ``Linear -> GELU -> Linear`` com ``part_dim`` unidades. As mãos
deixam de disputar a mesma projeção com os 40 pontos da face. A saída tem a
mesma forma da do ``input_proj`` original, então o resto do encoder (conv
temporal, downsampling, Transformer) é o mesmo.

As colunas extras do ``--extra-features`` misturam partes (mão, face,
presença), então formam um fluxo à parte em vez de serem redistribuídas.
"""

import torch
import torch.nn as nn

from skeltrans.common.layout import (FACE, LEFT_HAND, N_CHANNELS, N_POINTS, POSE,
                                     RIGHT_HAND)

# Colunas de cada parte no vetor achatado (T, 115*9): a ordem é ponto-a-ponto,
# com os 9 canais de cada ponto em sequência (coluna = ponto*9 + canal).
PARTS = {"pose": POSE, "mao_esq": LEFT_HAND, "mao_dir": RIGHT_HAND, "face": FACE}
BASE_DIM = N_POINTS * N_CHANNELS          # 1035


def _mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))


class PartsProjection(nn.Module):
    def __init__(self, input_dim, d_model, part_dim=None):
        super().__init__()
        if input_dim < BASE_DIM:
            raise ValueError(f"encoder 'parts' exige as {BASE_DIM} colunas do Passo 3 "
                             f"(115 pontos x 9 canais); recebeu input_dim={input_dim}.")
        self.part_dim = int(part_dim or max(1, d_model // 4))
        self.cols = {name: slice(sl.start * N_CHANNELS, sl.stop * N_CHANNELS)
                     for name, sl in PARTS.items()}
        self.parts = nn.ModuleDict({name: _mlp(c.stop - c.start, self.part_dim)
                                    for name, c in self.cols.items()})
        self.extra_dim = input_dim - BASE_DIM
        self.extra = _mlp(self.extra_dim, self.part_dim) if self.extra_dim else None
        n_streams = len(self.parts) + (1 if self.extra is not None else 0)
        self.fuse = nn.Linear(n_streams * self.part_dim, d_model)

    def forward(self, x):
        """x: (B, T, input_dim) -> (B, T, d_model)."""
        outs = [self.parts[name](x[..., c]) for name, c in self.cols.items()]
        if self.extra is not None:
            outs.append(self.extra(x[..., BASE_DIM:]))
        return self.fuse(torch.cat(outs, dim=-1))
