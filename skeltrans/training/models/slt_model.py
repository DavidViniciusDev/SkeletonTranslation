"""Modelo SLT completo: LandmarkEncoder + decoder T5 (cross-attention).

Opcionalmente, uma cabeça de reconhecimento por CTC (glosas) pode ser anexada
sobre a saída do encoder como supervisão auxiliar. Ela é criada apenas quando
``use_ctc=True``; com o padrão (``use_ctc=False``) a arquitetura é idêntica à
original e o ``forward`` mantém exatamente o mesmo contrato ``(loss, logits)``.
"""

import torch

from skeltrans.common.layout import INPUT_DIM
from skeltrans.training.models.landmark_encoder import LandmarkEncoder

import torch.nn as nn
import torch.nn.functional as F


class SLTModel(nn.Module):
    def __init__(self, t5, d_model=512, nhead=8, num_layers=6, dropout=0.2,
                 use_ctc=False, gloss_vocab_size=0, ds_steps=0):
        super().__init__()
        self.t5 = t5
        t5_hidden = t5.config.d_model
        self.encoder = LandmarkEncoder(
            input_dim=INPUT_DIM, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout, out_dim=t5_hidden,
            ds_steps=ds_steps,
        )
        # Cabeça CTC opcional (blank fica no índice 0 do vocabulário de glosas).
        self.use_ctc = bool(use_ctc)
        if self.use_ctc:
            if gloss_vocab_size <= 0:
                raise ValueError("use_ctc=True exige gloss_vocab_size > 0.")
            self.ctc_head = nn.Linear(t5_hidden, gloss_vocab_size)
        else:
            self.ctc_head = None

    def _encode(self, feats, pad_mask):
        from transformers.modeling_outputs import BaseModelOutput
        # o DataLoader entrega fp32; acompanha o dtype dos pesos (bf16 quando
        # --low-vram converte o modelo na inferência)
        feats = feats.to(self.encoder.input_proj.weight.dtype)
        # com ds_steps>0 o encoder encurta o eixo temporal e devolve a máscara já
        # reduzida; o attention_mask do T5 e o CTC derivam DESSA máscara, nunca
        # da original, sob pena de atender a padding silenciosamente.
        memory, pad_mask = self.encoder(feats, pad_mask)        # (B, T', t5_hidden)
        attn = (~pad_mask).long()                              # 1 = valido, 0 = pad
        return BaseModelOutput(last_hidden_state=memory), attn, pad_mask

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
                gloss_targets=None, gloss_lengths=None, ctc_weight=0.0):
        """Retorna (loss, logits).

        Parâmetros de CTC são opcionais: sem eles (ou com o head desativado), o
        comportamento é idêntico ao modelo original — só a perda de tradução.
        Quando o CTC está ativo e os alvos são fornecidos, a perda devolvida é
        ``ce + ctc_weight * ctc``; ``logits`` continua sendo o do decoder.
        """
        enc_out, attn, enc_pad_mask = self._encode(feats, pad_mask)
        out = self.t5(encoder_outputs=enc_out, attention_mask=attn, labels=labels)
        loss = out.loss
        if (self.use_ctc and self.ctc_head is not None
                and gloss_targets is not None and ctc_weight > 0):
            memory = enc_out.last_hidden_state
            ctc = self._ctc_loss(memory, enc_pad_mask, gloss_targets, gloss_lengths)
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
        # máscara REDUZIDA devolvida pelo encoder, nunca da original
        enc_out, _, enc_pad_mask = self._encode(feats, pad_mask)
        logits = self._ctc_logits(enc_out.last_hidden_state)   # (B, T', V)
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
    def generate(self, feats, pad_mask, max_new_tokens=64, num_beams=4):
        enc_out, attn, _ = self._encode(feats, pad_mask)
        return self.t5.generate(
            encoder_outputs=enc_out, attention_mask=attn,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
        )
