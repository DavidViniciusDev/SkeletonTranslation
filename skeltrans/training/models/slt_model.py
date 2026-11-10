"""Modelo SLT completo: LandmarkEncoder + decoder T5 (cross-attention).

Opcionalmente, uma cabeça de reconhecimento por CTC (glosas) pode ser anexada
sobre a saída do encoder como supervisão auxiliar. Ela é criada apenas quando
``use_ctc=True``; com o padrão (``use_ctc=False``) a arquitetura é idêntica à
original e o ``forward`` mantém exatamente o mesmo contrato ``(loss, logits)``.

Com ``temporal_mode='concept'`` o CTC passa a ter dois papéis: supervisão
auxiliar e fonte das fronteiras dos sinais. Ele roda sobre o andar de FRAMES
do encoder; a memória do decoder tem uma posição por sinal (ver
models/concept_pooling.py). No 'stride' (padrão) nada muda.
"""

import random

import torch

from skeltrans.training.data.features import feature_dim, parse_groups
from skeltrans.training.models.concept_pooling import (CONCEPT_BOUNDARIES,
                                                       forced_align_spikes,
                                                       greedy_spikes, spikes_to_seg_ids)
from skeltrans.training.models.landmark_encoder import LandmarkEncoder

import torch.nn as nn
import torch.nn.functional as F


class SLTModel(nn.Module):
    def __init__(self, t5, d_model=512, nhead=8, num_layers=6, dropout=0.2,
                 use_ctc=False, gloss_vocab_size=0, ds_steps=0,
                 extra_features=(), input_norm=False, encoder_type="flat",
                 temporal_mode="stride", concept_layers=2, concept_queries=1,
                 concept_boundaries="align", concept_greedy_prob=0.0):
        super().__init__()
        self.t5 = t5
        t5_hidden = t5.config.d_model
        # grupos de features extras (ver training/data/features.py): definem a
        # dimensão da entrada e ficam no modelo para que avaliação/inferência
        # montem a MESMA entrada do treino. () -> 1035, a entrada original.
        self.extra_features = parse_groups(extra_features)
        if temporal_mode == "concept" and not use_ctc:
            raise ValueError("temporal_mode='concept' exige use_ctc=True: o CTC define "
                             "as fronteiras dos sinais.")
        if concept_boundaries not in CONCEPT_BOUNDARIES:
            raise ValueError(f"concept_boundaries desconhecido: {concept_boundaries!r} "
                             f"(opcoes: {', '.join(CONCEPT_BOUNDARIES)}).")
        if not 0.0 <= concept_greedy_prob <= 1.0:
            raise ValueError("concept_greedy_prob deve estar em [0, 1].")
        self.encoder = LandmarkEncoder(
            input_dim=feature_dim(self.extra_features), d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout, out_dim=t5_hidden,
            ds_steps=ds_steps, input_norm=input_norm, encoder_type=encoder_type,
            temporal_mode=temporal_mode, concept_layers=concept_layers,
            concept_queries=concept_queries,
        )
        # Fronteiras no TREINO (sem parâmetros; não vão no state_dict): 'align'
        # usa as glosas verdadeiras quando o lote as traz, com probabilidade
        # concept_greedy_prob de usar o greedy mesmo assim (aproxima o treino da
        # inferência, que nunca tem glosas). Avaliação e inferência: greedy.
        self.concept_boundaries = concept_boundaries
        self.concept_greedy_prob = float(concept_greedy_prob)
        # Cabeça CTC opcional (blank fica no índice 0 do vocabulário de glosas).
        self.use_ctc = bool(use_ctc)
        if self.use_ctc:
            if gloss_vocab_size <= 0:
                raise ValueError("use_ctc=True exige gloss_vocab_size > 0.")
            self.ctc_head = nn.Linear(t5_hidden, gloss_vocab_size)
        else:
            self.ctc_head = None

    def _encode(self, feats, pad_mask):
        """(memória do decoder, attention_mask, máscara) — o contrato original."""
        return self._encode_full(feats, pad_mask)[:3]

    def _encode_full(self, feats, pad_mask, gloss_targets=None, gloss_lengths=None):
        """Como ``_encode``, mais ``(ctc_memory, ctc_mask)``: onde o CTC roda.

        No 'stride' é a própria memória do decoder (o caminho original). No
        'concept' é a saída do andar de frames, porque depois do pooling sobram
        só ~N posições, pouco para o CTC; as glosas, quando vêm, servem às
        fronteiras 'align'.
        """
        from transformers.modeling_outputs import BaseModelOutput
        # o DataLoader entrega fp32; acompanha o dtype dos pesos (bf16 quando
        # --low-vram converte o modelo na inferência)
        feats = feats.to(next(self.encoder.input_proj.parameters()).dtype)
        if self.encoder.temporal_mode == "concept":
            x, frame_mask = self.encoder.encode_frames(feats, pad_mask)
            frame_mem = self.encoder.adapter(x)                   # (B, T', t5_hidden)
            seg_ids = self._concept_segments(frame_mem, frame_mask,
                                             gloss_targets, gloss_lengths)
            memory, pad_mask = self.encoder.encode_concepts(x, seg_ids)
            ctc_mem, ctc_mask = frame_mem, frame_mask
        else:
            # com ds_steps>0 o encoder encurta o eixo temporal e devolve a máscara
            # já reduzida; o attention_mask do T5 e o CTC derivam DESSA máscara,
            # nunca da original, sob pena de atender a padding silenciosamente.
            memory, pad_mask = self.encoder(feats, pad_mask)    # (B, T', t5_hidden)
            ctc_mem, ctc_mask = memory, pad_mask
        attn = (~pad_mask).long()                              # 1 = valido, 0 = pad
        return BaseModelOutput(last_hidden_state=memory), attn, pad_mask, (ctc_mem, ctc_mask)

    @torch.no_grad()
    def _concept_segments(self, frame_mem, frame_mask, gloss_targets, gloss_lengths):
        """Id do sinal de cada frame (B, T'), a partir dos disparos do CTC."""
        logits = self._ctc_logits(frame_mem).float()
        use_align = (self.training and self.concept_boundaries == "align"
                     and gloss_targets is not None
                     and random.random() >= self.concept_greedy_prob)
        if use_align:
            spikes = forced_align_spikes(logits.log_softmax(-1), frame_mask,
                                         gloss_targets, gloss_lengths)
        else:
            spikes = greedy_spikes(logits, frame_mask)
        return spikes_to_seg_ids(spikes, frame_mask)

    def _ctc_logits(self, memory):
        """Logits do head CTC sobre a saída do encoder: (B, T, V).

        Fatorado do `_ctc_loss` para que treino (perda) e avaliação
        (reconhecimento de glosas) usem exatamente o mesmo caminho.
        """
        if not self.use_ctc or self.ctc_head is None:
            raise RuntimeError(
                "modelo sem head CTC: reconstrua com use_ctc=True "
                "(ou avalie um checkpoint treinado com --use-ctc).")
        return self.ctc_head(memory)

    def _ctc_loss(self, memory, pad_mask, gloss_targets, gloss_lengths):
        """CTC entre a saída do encoder e a sequência de glosas.

        memory: (B, T, t5_hidden); pad_mask: (B, T) True=padding;
        gloss_targets: (B, S) ids (0 = blank/padding); gloss_lengths: (B,).
        """
        logits = self._ctc_logits(memory)                      # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)                  # (T, B, V) p/ CTCLoss
        input_lengths = (~pad_mask).sum(dim=1).to(torch.long)  # frames válidos
        return F.ctc_loss(
            log_probs, gloss_targets,
            input_lengths, gloss_lengths.to(torch.long),
            blank=0, zero_infinity=True,
        )

    def forward(self, feats, pad_mask, labels,
                gloss_targets=None, gloss_lengths=None, ctc_weight=0.0,
                label_smoothing=0.0):
        """Retorna (loss, logits).

        Parâmetros de CTC são opcionais: sem eles (ou com o head desativado), o
        comportamento é idêntico ao modelo original — só a perda de tradução.
        Quando o CTC está ativo e os alvos são fornecidos, a perda devolvida é
        ``ce + ctc_weight * ctc``; ``logits`` continua sendo o do decoder.

        ``label_smoothing > 0`` recalcula a CE de tradução com suavização
        (mesma média sobre os tokens não ignorados que o T5 usa). Com 0 a perda
        é exatamente a ``out.loss`` do T5, como antes.
        """
        enc_out, attn, _, (ctc_mem, ctc_mask) = self._encode_full(
            feats, pad_mask, gloss_targets, gloss_lengths)
        out = self.t5(encoder_outputs=enc_out, attention_mask=attn, labels=labels)
        loss = out.loss
        if label_smoothing > 0:
            logits = out.logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   labels.reshape(-1), ignore_index=-100,
                                   label_smoothing=label_smoothing)
        if (self.use_ctc and self.ctc_head is not None
                and gloss_targets is not None and ctc_weight > 0):
            ctc = self._ctc_loss(ctc_mem, ctc_mask, gloss_targets, gloss_lengths)
            loss = loss + ctc_weight * ctc
        return loss, out.logits

    @torch.no_grad()
    def predict_glosses(self, feats, pad_mask):
        """Reconhecimento de glosas por decodificação CTC *greedy*.

        Devolve uma lista de B listas de ids (já sem <blank> e sem repetições),
        pronta para `GlossVocab.decode`. É o que permite medir WER de glosas —
        a métrica nativa do PHOENIX — a partir do head auxiliar de CTC.

        O procedimento greedy padrão do CTC tem três etapas, nesta ordem:
          1. argmax por frame;
          2. colapso de repetições CONSECUTIVAS (o CTC emite o mesmo rótulo em
             frames vizinhos para um único sinal);
          3. remoção dos <blank>.
        A ordem importa: colapsar depois de remover os blanks fundiria dois
        sinais iguais legitimamente separados por um blank (ex.: "IX IX").

        Só os frames válidos entram: `pad_mask` delimita o comprimento real de
        cada item, senão o padding do lote geraria glosas fantasma.
        """
        # com ds_steps>0 os logits têm T' < T posições: os comprimentos vêm da
        # máscara REDUZIDA devolvida pelo encoder, nunca da original. No
        # 'concept' o CTC lê o andar de frames (ctc_mem), não a memória por sinal.
        _, _, _, (ctc_mem, enc_pad_mask) = self._encode_full(feats, pad_mask)
        logits = self._ctc_logits(ctc_mem)                     # (B, T', V)
        best = logits.argmax(dim=-1)                           # (B, T')
        lengths = (~enc_pad_mask).sum(dim=1).tolist()           # frames válidos
        out = []
        for b, n in enumerate(lengths):
            seq, prev = [], None
            for t in range(int(n)):
                k = int(best[b, t])
                if k != prev:                  # 2) colapsa repetições
                    if k != 0:                 # 3) descarta <blank> (índice 0)
                        seq.append(k)
                    prev = k
            out.append(seq)
        return out

    @torch.no_grad()
    def generate(self, feats, pad_mask, max_new_tokens=64, num_beams=4, **gen_kwargs):
        """Gera os ids da tradução. ``gen_kwargs`` (ex.: ``length_penalty``,
        ``no_repeat_ngram_size``) vão direto para o ``generate`` do T5; sem
        eles, valem os padrões do ``generation_config`` do modelo, como antes."""
        enc_out, attn, _ = self._encode(feats, pad_mask)
        return self.t5.generate(
            encoder_outputs=enc_out, attention_mask=attn,
            max_new_tokens=max_new_tokens, num_beams=num_beams, **gen_kwargs,
        )
