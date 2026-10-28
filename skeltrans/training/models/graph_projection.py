"""Projeção de entrada por grafo do esqueleto (``--encoder-type gcn``).

No ``flat``, os 1035 números do frame entram juntos num ``Linear``; no
``parts``, cada parte do corpo tem sua MLP, mas dentro dela os pontos
continuam sem relação entre si. Aqui cada ponto é um NÓ com seus 9 canais e
cada camada mistura a informação de um nó só com os vizinhos no esqueleto
(dedo com dedo, dedo com a mão, mão com o pulso):

    frame (115 nós x 9)
      -> camadas GCN (vizinhança em 3 partições + grafo aprendido)
      -> média dos nós presentes de cada parte (pose, mão esq, mão dir, face)
      -> concatena (+ fluxo das --extra-features) -> Linear -> (d_model)

A saída tem a mesma forma da do ``input_proj`` original, então o resto do
encoder (conv temporal, downsampling, Transformer) é o mesmo.

Pontos ausentes (mão não detectada = 9 zeros) recebem tratamento explícito:
a agregação de cada partição é a média só sobre os vizinhos PRESENTES, os nós
ausentes continuam zerados em todas as camadas e o pooling ignora-os. Sem
isso, uma mão perdida pelo MediaPipe puxaria os vizinhos para zero.

Arestas e partições: ``skeltrans/common/skeleton_graph.py``.
"""

import torch
import torch.nn as nn

from skeltrans.common.layout import (FACE, LEFT_HAND, N_CHANNELS, N_POINTS, POSE,
                                     RIGHT_HAND)
from skeltrans.common.skeleton_graph import NUM_PARTITIONS, partitioned_adjacency

BASE_DIM = N_POINTS * N_CHANNELS          # 1035
PARTS = {"pose": POSE, "mao_esq": LEFT_HAND, "mao_dir": RIGHT_HAND, "face": FACE}
EPS = 1e-6


def node_presence(x):
    """(…, 1035 ou mais) -> (…, 115) bool: o ponto tem algum canal != 0.

    Precisa receber a entrada ANTES do --input-norm: depois do z-score, o
    0 de um ponto ausente vira -média/desvio e deixa de ser reconhecível.
    """
    nodes = x[..., :BASE_DIM].reshape(*x.shape[:-1], N_POINTS, N_CHANNELS)
    return nodes.ne(0).any(dim=-1)


def _mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))


class GCNLayer(nn.Module):
    """Uma camada GCN espacial (por frame) com partições e máscara de presença.

    Para cada partição k: agrega os vizinhos presentes com pesos Â_k + B_k
    (média renormalizada), concatena as K agregações e aplica um Linear
    (equivale a Σ_k Â_k X W_k). Depois: LayerNorm, GELU, residual.
    """

    def __init__(self, in_dim, out_dim, num_partitions=NUM_PARTITIONS):
        super().__init__()
        self.proj = nn.Linear(num_partitions * in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.res = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, h, A, B, present):
        """h: (B,T,V,C); A, B: (K,V,V); present: (B,T,V) float 0/1."""
        hm = h * present.unsqueeze(-1)                           # ausentes não contribuem
        # já na ordem (B,T,V,K,C): o reshape para K*C não precisa de cópia
        agg = torch.einsum("kvw,btwc->btvkc", A + B, hm)         # (B,T,V,K,C)
        # renormaliza cada partição pelos vizinhos presentes (grafo fixo):
        # média sobre quem existe, em vez de diluir com os zeros dos ausentes
        full = A.sum(-1).transpose(0, 1)                         # (V,K)
        seen = torch.einsum("kvw,btw->btvk", A, present)         # (B,T,V,K)
        scale = torch.where(seen > EPS, full / seen.clamp(min=EPS), torch.zeros_like(seen))
        agg = agg * scale.unsqueeze(-1)
        agg = agg.flatten(-2)                                    # (B,T,V,K*C)
        out = self.act(self.norm(self.proj(agg))) + self.res(h)
        return out * present.unsqueeze(-1)                       # ausente continua 0


class GraphProjection(nn.Module):
    def __init__(self, input_dim, d_model, part_dim=None, hidden_dim=None, num_layers=3):
        super().__init__()
        if input_dim < BASE_DIM:
            raise ValueError(f"encoder 'gcn' exige as {BASE_DIM} colunas do Passo 3 "
                             f"(115 pontos x 9 canais); recebeu input_dim={input_dim}.")
        self.part_dim = int(part_dim or max(1, d_model // 4))
        self.hidden_dim = int(hidden_dim or max(1, self.part_dim // 2))
        # grafo fixo (buffer: vai no state_dict, então o checkpoint guarda o
        # grafo com que foi treinado) + correção aprendida, começando em zero
        self.register_buffer("A", torch.as_tensor(partitioned_adjacency()))
        self.B = nn.Parameter(torch.zeros_like(self.A))
        dims = [N_CHANNELS] + [self.hidden_dim] * (num_layers - 1) + [self.part_dim]
        self.gcn_layers = nn.ModuleList(GCNLayer(i, o) for i, o in zip(dims, dims[1:]))
        self.extra_dim = input_dim - BASE_DIM
        self.extra = _mlp(self.extra_dim, self.part_dim) if self.extra_dim else None
        n_streams = len(PARTS) + (1 if self.extra is not None else 0)
        self.fuse = nn.Linear(n_streams * self.part_dim, d_model)
        # ligado pelo LandmarkEncoder quando --grad-checkpoint: as ativações de
        # cada camada (B*T*115 nós) são recalculadas no backward em vez de
        # guardadas; não cria parâmetros
        self.grad_checkpoint = False

    def describe(self):
        n = sum(p.numel() for p in self.parameters())
        return (f"{len(self.gcn_layers)} camadas GCN sobre {N_POINTS} nos, "
                f"{self.A.shape[0]} particoes, canais {N_CHANNELS}->{self.hidden_dim}->"
                f"{self.part_dim}; pooling por parte"
                f"{' + fluxo de extras' if self.extra is not None else ''} | "
                f"{n / 1e6:.2f}M parametros")

    def node_features(self, x, present=None):
        """(B,T,D) -> ((B,T,115,part_dim), presença (B,T,115) float)."""
        if present is None:
            present = node_presence(x)
        present = present.to(x.dtype)
        h = x[..., :BASE_DIM].reshape(*x.shape[:-1], N_POINTS, N_CHANNELS)
        A = self.A.to(x.dtype)
        B = self.B.to(x.dtype)
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for layer in self.gcn_layers:
            if use_ckpt:
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(layer, h, A, B, present, use_reentrant=False)
            else:
                h = layer(h, A, B, present)
        return h, present

    def forward(self, x, present=None, pad_mask=None):
        """x: (B, T, input_dim) -> (B, T, d_model).

        `present` (B,T,115): presença de cada ponto, calculada pelo encoder
        sobre a entrada crua (antes do --input-norm). Se None, deduz de `x`.

        `pad_mask` (B,T), True = padding: as camadas GCN rodam só nos frames
        reais. O custo aqui cresce com B*T*115 nós, e o padding até a maior
        frase do lote pode multiplicá-lo (no PHOENIX vão até 475 frames). Um
        frame de padding é todo zero -> todos os nós ausentes -> a saída é
        sempre o mesmo vetor, que é calculado uma vez e copiado: o resultado é
        idêntico ao de rodar o lote inteiro.
        """
        if present is None:
            present = node_presence(x)
        if pad_mask is None or not pad_mask.any():
            return self._project(x, present)
        valid = ~pad_mask
        out_valid = self._project(x[valid].unsqueeze(0), present[valid].unsqueeze(0))[0]
        zero = x.new_zeros(1, 1, x.shape[-1])
        out_pad = self._project(zero, zero.new_zeros(1, 1, N_POINTS))[0, 0]
        out = out_pad.expand(*pad_mask.shape, -1).clone()
        out[valid] = out_valid
        return out

    def _project(self, x, present):
        h, present = self.node_features(x, present)
        outs = []
        for sl in PARTS.values():
            m = present[..., sl].unsqueeze(-1)                   # (B,T,n,1)
            # média só sobre os nós presentes; parte inteira ausente -> 0
            outs.append((h[..., sl, :] * m).sum(-2) / m.sum(-2).clamp(min=1.0))
        if self.extra is not None:
            outs.append(self.extra(x[..., BASE_DIM:]))
        return self.fuse(torch.cat(outs, dim=-1))
