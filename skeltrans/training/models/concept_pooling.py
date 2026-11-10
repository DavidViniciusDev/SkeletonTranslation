"""Encoder por conceitos (``--temporal-mode concept``): frames -> sinais.

No modo ``stride`` (o original), o eixo temporal é encurtado às cegas: o
``--ds-steps`` junta 2^n frames vizinhos, sem saber onde um sinal termina. Aqui
a memória entregue ao decoder tem **uma posição por sinal**:

    frames -> Transformer de frames -> head CTC (glosas)
                     |                      |
                     |            fronteiras dos sinais
                     v                      v
              pooling por conceito  <-------+
                     -> Transformer de conceitos -> adapter -> decoder T5

Fronteiras
----------
O CTC é "pontudo": dispara em 1-2 frames por sinal e emite <blank> no resto.
Cada disparo marca um sinal; o corte entre dois sinais fica no ponto médio
entre os disparos, de modo que todo frame válido pertence a algum conceito.
Os disparos vêm de uma de duas fontes:

- ``align`` (treino, com as glosas verdadeiras): alinhamento forçado (Viterbi
  do CTC). Sempre dá exatamente 1 disparo por glosa, mesmo no início do
  treino, quando o CTC ainda só emite <blank>.
- ``greedy`` (inferência, ou treino sem glosas): argmax por frame, com o mesmo
  colapso do ``SLTModel.predict_glosses``. Frase sem nenhum disparo vira 1
  conceito (nunca 0: máscara vazia quebraria a atenção do decoder).

As fronteiras não são diferenciáveis, e não precisam ser: o gradiente chega aos
frames pelo pooling, e o CTC continua treinando quem as define.

Pooling
-------
Média ponderada por atenção DENTRO de cada segmento: o pico do sinal pesa mais
que a transição entre sinais. Com ``queries=K > 1``, cada sinal vira K vetores
(K pontuações independentes), o que alivia o gargalo de ~7 posições por frase.
"""

import torch
import torch.nn as nn

# Modos do eixo temporal (--temporal-mode) e fontes de fronteira do treino
# (--concept-boundaries). 'stride' é o original.
TEMPORAL_MODES = ("stride", "concept")
CONCEPT_BOUNDARIES = ("align", "greedy")

NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# Disparos do CTC (frames onde cada sinal aparece)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_spikes(logits, pad_mask, blank=0):
    """(B, T, V), (B, T) -> (B, T) bool: frames onde o greedy emite uma glosa.

    Mesma regra do ``predict_glosses``: argmax por frame, colapso de repetições
    consecutivas e descarte do <blank>. "A _ A" conta como dois disparos.
    """
    best = logits.argmax(dim=-1)                                   # (B, T)
    prev = torch.full_like(best, -1)
    prev[:, 1:] = best[:, :-1]
    return (best != blank) & (best != prev) & ~pad_mask


@torch.no_grad()
def forced_align_spikes(log_probs, pad_mask, targets, target_lengths, blank=0):
    """Viterbi do CTC em lote: frame do primeiro disparo de cada glosa-alvo.

    log_probs: (B, T, V) log-softmax; pad_mask: (B, T) True = padding;
    targets: (B, S) ids (0 = padding); target_lengths: (B,).
    Devolve (B, T) bool com exatamente ``target_lengths[b]`` disparos por
    amostra. Quem não tem alinhamento possível (frase mais curta que as
    glosas) fica com a linha toda False; ``spikes_to_seg_ids`` a trata como 1
    conceito, como no greedy sem disparos.

    Estados estendidos do CTC: ``_ g1 _ g2 _ ... gS _`` (2S+1). Em cada frame
    o caminho fica no estado, avança 1, ou pula o <blank> entre duas glosas
    DIFERENTES. Frames de padding repetem o estado (não mudam o caminho).
    """
    B, T, _ = log_probs.shape
    S = targets.size(1)
    L = 2 * S + 1
    dev = log_probs.device
    lp = log_probs.float()
    lengths = (~pad_mask).sum(dim=1)                               # (B,)
    tl = target_lengths.to(dev).long()

    ext = torch.full((B, L), blank, dtype=torch.long, device=dev)
    ext[:, 1::2] = targets.to(dev).long()
    emit = lp.gather(2, ext.unsqueeze(1).expand(B, T, L))          # (B, T, L)
    # estados além de 2*tl ficam inalcançáveis
    state_ok = torch.arange(L, device=dev)[None, :] < (2 * tl + 1)[:, None]
    emit = emit.masked_fill(~state_ok.unsqueeze(1), NEG_INF)
    # pular o blank: só para glosa (estado ímpar) diferente da anterior
    can_skip = torch.zeros(B, L, dtype=torch.bool, device=dev)
    can_skip[:, 3::2] = ext[:, 3::2] != ext[:, 1:-2:2]

    delta = torch.full((B, L), NEG_INF, device=dev)
    delta[:, 0] = emit[:, 0, 0]
    delta[:, 1] = torch.where(tl > 0, emit[:, 0, 1], torch.full_like(emit[:, 0, 1], NEG_INF))
    back = torch.zeros(B, T, L, dtype=torch.uint8, device=dev)     # 0 fica, 1 avança, 2 pula
    for t in range(1, T):
        stay = delta
        step = torch.full_like(delta, NEG_INF)
        step[:, 1:] = delta[:, :-1]
        skip = torch.full_like(delta, NEG_INF)
        skip[:, 2:] = delta[:, :-2]
        skip = skip.masked_fill(~can_skip, NEG_INF)
        cand = torch.stack([stay, step, skip], dim=-1)             # (B, L, 3)
        best, arg = cand.max(dim=-1)
        active = (t < lengths)[:, None]                            # padding: congela
        delta = torch.where(active, best + emit[:, t], delta)
        back[:, t] = torch.where(active, arg, torch.zeros_like(arg)).to(torch.uint8)

    # termina no último blank ou na última glosa
    last = 2 * tl
    end_blank = delta.gather(1, last[:, None]).squeeze(1)
    end_gloss = delta.gather(1, (last - 1).clamp(min=0)[:, None]).squeeze(1)
    end_gloss = torch.where(tl > 0, end_gloss, torch.full_like(end_gloss, NEG_INF))
    state = torch.where(end_gloss > end_blank, last - 1, last)
    feasible = torch.isfinite(torch.maximum(end_blank, end_gloss))

    path = torch.zeros(B, T, dtype=torch.long, device=dev)
    path[:, T - 1] = state
    for t in range(T - 1, 0, -1):
        state = state - back[:, t, :].gather(1, state[:, None]).squeeze(1).long()
        path[:, t - 1] = state

    # disparo = primeiro frame em cada estado de glosa (ímpar)
    prev = torch.full_like(path, -1)
    prev[:, 1:] = path[:, :-1]
    spikes = (path % 2 == 1) & (path != prev) & ~pad_mask
    return spikes & feasible[:, None]


# --------------------------------------------------------------------------- #
# Disparos -> segmentos
# --------------------------------------------------------------------------- #
def spikes_to_seg_ids(spikes, pad_mask):
    """(B, T) disparos -> (B, T) id do conceito de cada frame; -1 no padding.

    Corta no ponto médio entre disparos consecutivos: com disparos em 12, 51 e
    93 numa frase de 110 frames, os conceitos são 0-30, 31-71 e 72-109. Sem
    nenhum disparo, a frase inteira é o conceito 0.
    """
    B, T = spikes.shape
    t = torch.arange(T, device=spikes.device)
    seg = torch.zeros(B, T, dtype=torch.long, device=spikes.device)
    for b in range(B):
        pos = t[spikes[b]]
        if pos.numel() > 1:
            cuts = torch.div(pos[:-1] + pos[1:], 2, rounding_mode="floor")
            # frame t pertence ao conceito k = nº de cortes <= t
            seg[b] = torch.searchsorted(cuts, t, right=True)
    return seg.masked_fill(pad_mask, -1)


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #
class ConceptPooling(nn.Module):
    """Frames -> conceitos: K vetores por segmento, por atenção dentro dele.

    x: (B, T, d); seg_ids: (B, T) com ids 0..N-1 (contíguos e crescentes) e
    -1 no padding. Saída: (B, N_max*K, d) e máscara (B, N_max*K) True = pad,
    na ordem conceito-a-conceito (as K posições de um sinal ficam juntas).
    """

    def __init__(self, d_model, queries=1):
        super().__init__()
        if queries < 1:
            raise ValueError("concept_queries deve ser >= 1.")
        self.queries = int(queries)
        # uma pontuação por frame e por query: "quanto este frame representa o
        # seu sinal, do ponto de vista da query k"
        self.score = nn.Linear(d_model, self.queries)
        # com K > 1, cada posição ganha a identidade da sua query; com K = 1
        # nenhum parâmetro extra é criado
        self.query_embed = (nn.Parameter(torch.zeros(self.queries, d_model))
                            if self.queries > 1 else None)

    def forward(self, x, seg_ids):
        B, T, d = x.shape
        K = self.queries
        valid = seg_ids >= 0
        n_concepts = (seg_ids.max(dim=1).values + 1).clamp(min=1)  # (B,)
        N = int(n_concepts.max())
        ids = seg_ids.clamp(min=0)                                  # padding -> 0, peso 0

        # softmax por segmento, em fp32: exp(s - max) / soma do segmento
        s = self.score(x).float()                                   # (B, T, K)
        s = s.masked_fill(~valid.unsqueeze(-1), NEG_INF)
        s = s - s.amax(dim=1, keepdim=True).clamp(min=-1e4)
        w = torch.exp(s) * valid.unsqueeze(-1)                      # (B, T, K)
        idx = ids.unsqueeze(-1).expand(B, T, K)
        denom = torch.zeros(B, N, K, device=x.device).scatter_add_(1, idx, w)
        w = w / denom.gather(1, idx).clamp(min=1e-12)               # soma 1 por segmento

        # out[b, n, k] = soma_t w[b, t, k] * x[b, t]  (t no segmento n)
        out = torch.zeros(B, N, K, d, device=x.device, dtype=torch.float32)
        contrib = w.unsqueeze(-1) * x.float().unsqueeze(2)          # (B, T, K, d)
        out.scatter_add_(1, ids[:, :, None, None].expand(B, T, K, d), contrib)
        out = out.to(x.dtype)
        if self.query_embed is not None:
            out = out + self.query_embed.to(out.dtype)
        out = out.reshape(B, N * K, d)

        mask = torch.arange(N, device=x.device)[None, :] >= n_concepts[:, None]
        return out, mask.repeat_interleave(K, dim=1)
