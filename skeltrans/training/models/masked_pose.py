"""Pré-treino do encoder por *masked pose modeling* (MPM).

A ideia é a do BERT, trocando palavras por keypoints: esconder trechos da
entrada e pedir ao encoder que os reconstrua a partir do resto. Não usa
glosas nem traduções, só keypoints, então qualquer vídeo de alguém
sinalizando vira dado de treino.

O encoder é o MESMO ``LandmarkEncoder`` do SLT (mesmo código, mesma
configuração). Depois do pré-treino, só os pesos dele são guardados
(``encoder_init.save_encoder``); a cabeça de reconstrução é descartada.

Mascaramento (``make_masks``)
-----------------------------
Cobre ~``ratio`` dos frames válidos de cada frase com trechos contíguos de
``span`` frames. Cada trecho é de um destes tipos:

- ``frames``: o frame inteiro some (os 115 pontos e as features extras);
- ``hands``: some só uma das mãos (sorteada) e as features extras.

``mixed`` (padrão) sorteia o tipo de cada trecho com chance 1/2. Esconder um
frame isolado seria fácil demais (basta interpolar os vizinhos); trechos
obrigam o encoder a entender o movimento.

As features extras (A–F) são calculadas a partir das posições: deixá-las
visíveis entregaria a resposta (ex.: a configuração da mão escondida). Por
isso elas somem em todo frame mascarado, de qualquer tipo.

O valor escondido é 0 na entrada crua, o mesmo de um ponto que o MediaPipe
não detectou. O encoder já está acostumado a esse valor e o ``gcn`` trata o
ponto como ausente.

Reconstrução
------------
Alvo: as 1035 colunas base (115 pontos × posição/velocidade/aceleração), no
espaço normalizado pelo ``--input-norm`` quando ativo. A perda é o erro
quadrático médio só nos pontos escondidos, em frames reais e com o ponto
presente (reconstruir um ponto que nem existia não ensina nada).

Com ``ds_steps`` > 0 o encoder devolve T' = T / 2^n posições; cada posição
reconstrói os 2^n frames que resume (cabeça ``Linear(d_model, 2^n · 1035)``).
"""

import torch
import torch.nn as nn

from skeltrans.common.layout import LEFT_HAND, N_CHANNELS, N_POINTS, RIGHT_HAND
from skeltrans.training.models.graph_projection import node_presence
from skeltrans.training.models.landmark_encoder import LandmarkEncoder

# Colunas base de um frame (115 pontos × 9 canais); as extras vêm depois.
BASE_DIM = N_POINTS * N_CHANNELS
MASK_MODES = ("mixed", "frames", "hands")


def make_masks(pad_mask, ratio=0.4, span=8, mode="mixed", generator=None):
    """Sorteia o que esconder em cada frase do lote.

    pad_mask: (B, T) bool, True = padding.
    Devolve ``(point_mask, frame_mask)``:
      - point_mask (B, T, 115) bool: pontos escondidos;
      - frame_mask (B, T) bool: frames com as features extras escondidas.

    Cada frase recebe trechos de ``span`` frames até cobrir ``ratio`` dos seus
    frames reais (no mínimo um trecho). ``generator`` (torch.Generator) torna
    o sorteio reprodutível, ex.: a validação usa sempre as mesmas máscaras.
    Roda na CPU; o chamador move o resultado para o dispositivo.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"mask mode desconhecido: {mode!r} (opcoes: {', '.join(MASK_MODES)}).")
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio deve estar em (0, 1).")
    if span < 1:
        raise ValueError("span deve ser >= 1.")
    B, T = pad_mask.shape
    point_mask = torch.zeros(B, T, N_POINTS, dtype=torch.bool)
    frame_mask = torch.zeros(B, T, dtype=torch.bool)
    lengths = (~pad_mask).sum(dim=1).tolist()

    def rand_int(high):
        """Inteiro em [0, high) com o gerador do chamador."""
        return int(torch.randint(0, max(1, high), (1,), generator=generator).item())

    for b, n in enumerate(lengths):
        n = int(n)
        if n == 0:
            continue
        width = min(span, n)
        target = max(1, round(ratio * n))
        covered = torch.zeros(n, dtype=torch.bool)
        # limite de tentativas: trechos sorteados podem se sobrepor
        for _ in range(4 * (n // width + 1)):
            start = rand_int(n - width + 1)
            stop = start + width
            kind = mode if mode != "mixed" else ("frames" if rand_int(2) == 0 else "hands")
            if kind == "frames":
                point_mask[b, start:stop, :] = True
            else:
                hand = LEFT_HAND if rand_int(2) == 0 else RIGHT_HAND
                point_mask[b, start:stop, hand] = True
            frame_mask[b, start:stop] = True
            covered[start:stop] = True
            if int(covered.sum()) >= target:
                break
    return point_mask, frame_mask


def apply_masks(feats, point_mask, frame_mask):
    """Zera na entrada crua o que foi sorteado (devolve uma cópia).

    feats: (B, T, 1035 + extras). Os pontos de ``point_mask`` têm os 9 canais
    zerados; as colunas extras são zeradas nos frames de ``frame_mask``.
    """
    B, T, D = feats.shape
    base = feats[..., :BASE_DIM].reshape(B, T, N_POINTS, N_CHANNELS)
    base = base.masked_fill(point_mask.unsqueeze(-1), 0.0).reshape(B, T, BASE_DIM)
    if D == BASE_DIM:
        return base
    extra = feats[..., BASE_DIM:].masked_fill(frame_mask.unsqueeze(-1), 0.0)
    return torch.cat([base, extra], dim=-1)


class MaskedPoseModel(nn.Module):
    """``LandmarkEncoder`` + cabeça de reconstrução dos keypoints escondidos.

    O encoder é criado sem adaptador (``out_dim = d_model``) e no modo
    'stride': o adaptador liga o encoder ao T5 e o andar de conceitos depende
    do CTC, então os dois começam do zero no treino de SLT.
    """

    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=6, dropout=0.2,
                 ds_steps=2, input_norm=False, encoder_type="flat"):
        super().__init__()
        self.encoder = LandmarkEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
            dropout=dropout, out_dim=None, ds_steps=ds_steps, input_norm=input_norm,
            encoder_type=encoder_type, temporal_mode="stride")
        self.factor = 2 ** int(ds_steps)
        self.head = nn.Linear(d_model, self.factor * BASE_DIM)

    def forward(self, feats, pad_mask, point_mask, frame_mask):
        """Devolve ``(loss, stats)``.

        ``stats`` traz somas para agregar várias passadas: ``sq_err`` (erro
        quadrático somado), ``sq_zero`` (o mesmo erro se o modelo previsse 0,
        a "variância" do alvo) e ``count`` (pontos avaliados). A razão
        ``sq_err / sq_zero`` é a fração do sinal que o modelo NÃO explicou.
        """
        B, T, _ = feats.shape
        masked = apply_masks(feats, point_mask, frame_mask)
        x, _ = self.encoder(masked, pad_mask)                  # (B, T', d_model)
        pred = self.head(x).float()                            # (B, T', f * 1035)
        pred = pred.reshape(B, -1, BASE_DIM)[:, :T]            # (B, T, 1035)
        pred = pred.reshape(B, T, N_POINTS, N_CHANNELS)

        target = feats[..., :BASE_DIM].float()
        enc = self.encoder
        if enc.input_norm:
            target = (target - enc.input_mean[:BASE_DIM]) / enc.input_std[:BASE_DIM]
        target = target.reshape(B, T, N_POINTS, N_CHANNELS)

        # só conta: ponto escondido, frame real e ponto presente na entrada crua
        weight = point_mask & node_presence(feats) & ~pad_mask.unsqueeze(-1)
        weight = weight.float()
        sq_err = ((pred - target) ** 2).mean(dim=-1)           # (B, T, 115)
        sq_zero = (target ** 2).mean(dim=-1)
        count = weight.sum()
        loss = (sq_err * weight).sum() / count.clamp(min=1.0)
        stats = {"sq_err": float((sq_err.detach() * weight).sum()),
                 "sq_zero": float((sq_zero * weight).sum()),
                 "count": float(count)}
        return loss, stats
