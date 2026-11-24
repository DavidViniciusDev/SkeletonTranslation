# Ideia: encoder em múltiplos fluxos com fusão tardia

> **Status:** ideia registrada, ainda não implementada.
> **Data:** 2026-11-23.
> **Motivação:** as substituições dominam o WER (≈23 de 37,5 na v15), e o
> `--encoder-type parts` (que separa as partes só dentro de um frame) não ajudou.
> **Objetivo:** reduzir as substituições separando mãos e corpo/rosto **ao longo
> do tempo**, e não só na projeção de cada frame.

---

## 1. A analogia

Três peritos num tribunal: um de impressões digitais, um de câmeras, um de
depoimentos. Cada um estuda só a sua evidência e escreve um laudo; o juiz lê
os três e decide. O contrário seria jogar tudo numa pilha e pedir a um só
perito que analise tudo junto desde o início. Ele tende a se apoiar no que é
mais fácil e ignorar o resto.

Hoje o SkeLTrans faz a "pilha única": pose, mãos e rosto viram um vetor por
frame logo na primeira camada (no `flat` e também no `parts`).

---

## 2. A proposta

```
HOJE (fusão cedo):
pose + mãos + rosto ─► um vetor por frame ─► Transformer ─► CTC ─► adapter ─► T5

FUSÃO TARDIA:
                      ┌─► fluxo MÃOS:  proj ─► conv/ds ─► Transformer ─┬─► CTC mãos (só treino)
keypoints ─► separa ──┤                                               │
                      └─► fluxo CORPO: proj ─► conv/ds ─► Transformer ─┼─► CTC corpo (só treino)
                                                                      ▼
                                       concatena ─► Linear ─► CTC conjunto ─► adapter ─► T5
```

- Cada fluxo tem **o seu encoder temporal inteiro** (projeção, convoluções,
  `--ds-steps` e Transformer).
- Cada fluxo tem um **CTC auxiliar próprio** no treino, que o obriga a
  reconhecer sinais sozinho.
- A fusão acontece **depois** dos Transformers; do adapter em diante nada muda.

---

## 3. Por que pode funcionar

1. **As partes carregam informações diferentes, em ritmos diferentes.**
   - As mãos dizem **qual** sinal: configuração, orientação, movimento. Mudam
     rápido e são detalhadas.
   - O rosto e o corpo carregam a gramática: pergunta, negação, intensidade.
     Mudam devagar e cobrem vários sinais.

   Num Transformer único, os dois ritmos disputam a mesma atenção.
2. **Nenhuma parte "abafa" a outra.** Com um CTC por fluxo, o fluxo das mãos não
   pode depender do corpo para acertar, e vice-versa.
3. **Robustez a falhas do MediaPipe.** Quando a mão some num frame, o fluxo do
   corpo continua inteiro. Na fusão cedo, o vetor do frame inteiro sai
   estragado.
4. **Precedente:** o TwoStream-SLT (Chen et al., NeurIPS 2022) usa dois fluxos
   (vídeo e keypoints), cada um com seu CTC, e esteve entre os melhores no
   PHOENIX-2014T. Aqui seriam dois fluxos de keypoints.

### Diferença para o `parts`, que não ajudou

| | `parts` | fusão tardia |
|---|---|---|
| Onde separa | só a projeção de **um frame** | o encoder **inteiro**, ao longo do tempo |
| Quando junta | logo depois da projeção | depois dos Transformers |
| Supervisão | um CTC | um CTC por fluxo + o conjunto |

---

## 4. Riscos

- **Mais parâmetros com os mesmos poucos dados:** dois Transformers podem
  sobreajustar. Para comparar com justiça, usar 3 camadas por fluxo (6 no
  total, como a v15).
- **Custo:** ~2× o tempo e a memória do encoder.
- **O `parts` não ajudou:** separar por parte pode não ser o que falta. A aposta
  é que a separação **no tempo** faz diferença; não é garantido.

---

## 5. Como implementar sem quebrar nada

Mesmo padrão do `--temporal-mode concept`:

```
--encoder-streams single         # padrão: exatamente o encoder de hoje
--encoder-streams hands,body     # fusão tardia: mãos | corpo + rosto
--stream-ctc-weight 0.5          # peso dos CTCs de cada fluxo (o conjunto segue com --ctc-weight)
--stream-layers 3                # camadas do Transformer de cada fluxo
```

| Garantia | Como |
|---|---|
| `single` bit a bit igual ao de hoje | nenhum parâmetro novo no modo padrão; teste comparando com o HEAD |
| Checkpoints antigos carregam | modo detectado pelos pesos (`encoder.streams.*`) em `checkpoint.py` |
| `evaluate_slt.py` e `infer_slt.py` sem flags novas | o mesmo mecanismo de detecção |
| Combinações inválidas abortam com mensagem | ex.: `--stream-ctc-weight > 0` sem `--use-ctc` |
| `--temporal-mode concept` continua funcionando | as fronteiras vêm do CTC conjunto |
| `--init-encoder` | o pré-treino passaria a gerar os dois fluxos; versão 2 do formato do `encoder.pt` |

Arquivos que mudariam: `models/landmark_encoder.py` (um `StreamEncoder` por
fluxo + fusão), `models/slt_model.py` (CTCs por fluxo na perda),
`checkpoint.py` (detecção), `cli.py`/`config.py` (flags), `diagnostics.py`
(smoke test) e um `tests/test_encoder_streams.py`.

### Decisões em aberto

1. **Para onde vão as `--extra-features`:** A (configuração de mão), B
   (orientação da palma) e parte de C (ponto de articulação) são das mãos; D
   (movimento), E (não manuais) e F (presença) se dividem entre os dois.
   Conferir em `skeltrans/training/data/features.py` antes de separar.
2. **Projeção de cada fluxo:** começar com `flat` em cada um.
3. **Fusão:** concatenação + `Linear` (simples) ou cross-attention entre os
   fluxos (como as conexões laterais do TwoStream-SLT), numa segunda versão.

---

## 6. Como medir

- Braço de controle: `stride`, receita da v15, mesma semente.
- Olhar **WER_sub** primeiro (o alvo da ideia) e depois o BLEU-4.
- Comparar o WER dos CTCs de cada fluxo: se o das mãos sozinho já estiver perto
  do conjunto, o corpo e o rosto contribuem pouco; isso também é um achado.
