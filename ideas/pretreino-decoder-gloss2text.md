# Ideia: pré-treino do decoder em Gloss2Text

> **Status:** ideia registrada, ainda não implementada.
> **Motivação:** resultados do PHOENIX-2014T até nov/2026.
> **Objetivo:** aumentar o BLEU-4 atacando o lado da **tradução** (decoder), que passou a ser o gargalo.

---

## 1. Por que agora

Nos últimos treinos, o **encoder melhora a cada execução**, mas o **BLEU estaciona**:

| Execução | WER (glosas) | WER_sub | BLEU-4 (teste) |
|---|---|---|---|
| linha de base | 45 | — | 16,51 |
| v10 (extras A,B,C) | 43,46 | 28,17 | 17,29 |
| v11 (+F, `--input-norm`) | 40,01 | 26,13 | 17,81 |
| A–F + `--input-norm` | **38,53** | **24,25** | 17,72 |

O WER caiu ~6,5 pontos, mas o BLEU-4 parou em ~17,7. O decoder recebe glosas
cada vez melhores e não converte esse ganho em frases melhores. Encoders
alternativos (`parts`, `gcn`) não ajudaram, o que reforça que o próximo ganho
deve vir do lado da tradução.

---

## 2. O problema: o decoder aprende duas tarefas ao mesmo tempo

Hoje o decoder T5 precisa aprender, com ~7 mil frases:

1. **Ler o encoder de landmarks.** Os vetores do encoder são uma
   "linguagem" nova, que não existia no pré-treino do T5.
2. **Traduzir da gramática de sinais para o alemão.** As glosas têm gramática
   própria: outra ordem, sem artigos, sem conjugação.

```
glosas:  MORGEN  NORD  REGEN
alemão:  "Morgen regnet es im Norden."
```

O decoder precisa reordenar, inserir palavras ("es", "im") e conjugar
("regnet"). Hoje ele aprende isso **misturado** com a tarefa 1 e a partir de
uma entrada **ruidosa** (o encoder ainda erra ~38% das glosas).

---

## 3. A ideia: separar as tarefas em duas etapas

### Etapa 1 — Gloss2Text (só texto)

O T5 é treinado como um tradutor comum, de texto para texto:

```
entrada:  "MORGEN NORD REGEN"
saída:    "Morgen regnet es im Norden."
```

- **Dados:** os próprios manifestos (`tokens` → `text`). Não precisa de vídeo.
- **Custo:** baixo. São frases curtas e sem features, de minutos a ~1 h de GPU.
- **O que ele aprende:** a gramática da tradução (tarefa 2), com entradas **limpas**.

### Etapa 2 — o treino de sinais, partindo do T5 da etapa 1

```
Etapa 1:  [encoder de texto do T5] → [decoder T5]   "glosas → alemão"
                                         │  reaproveitado
Etapa 2:  [encoder de landmarks]   → [decoder T5]   "vídeo → alemão"
```

O decoder já chega sabendo traduzir do "espaço das glosas". O encoder de
landmarks só precisa aprender a produzir algo que esse decoder entenda
(tarefa 1). Cada parte fica com um trabalho mais simples.

---

## 4. Conceitos e detalhes importantes

### 4.1 O encoder de texto do T5 é descartado na etapa 2

Na etapa 1, o decoder aprende a ler as representações que o **encoder de
texto do T5** produz a partir das glosas. Na etapa 2, a arquitetura usa o
encoder de landmarks no lugar dele (`encoder_outputs`), então o encoder de
texto deixa de ser usado. O encoder de landmarks precisa aprender a produzir
representações **parecidas** com as que o decoder espera.

O que ajuda nessa "imitação":

- **CTC de glosas** (já usado, `--use-ctc`): empurra o encoder a codificar a
  identidade de cada sinal, ou seja, algo próximo de glosas.
- **Perda de alinhamento (opcional, mais avançada):** passar as glosas
  verdadeiras pelo encoder de texto do T5 da etapa 1 e aproximar a saída do
  encoder de landmarks dessas representações (por exemplo, distância média
  entre as sequências após um pooling, ou uma perda contrastiva). É a ideia
  de um "mapeador" entre o espaço visual e o espaço da linguagem.

### 4.2 Glosas com ruído na etapa 1

Na etapa 2, o decoder recebe informação com erros (WER ~38%). Se a etapa 1
usar só glosas perfeitas, o decoder pode ficar frágil a esses erros.

**Truque:** treinar a etapa 1 com glosas corrompidas aleatoriamente, numa taxa
parecida com o WER do encoder, imitando a decomposição real (hoje: muita
substituição, alguma deleção, pouca inserção):

- **substituição:** trocar uma glosa por outra do vocabulário;
- **deleção:** remover uma glosa;
- **inserção:** inserir uma glosa aleatória.

Assim o decoder aprende a traduzir mesmo com glosas imperfeitas, que é a
situação real da etapa 2.

### 4.3 O diagnóstico de teto vem de graça

O BLEU da etapa 1 no **teste**, com glosas **verdadeiras**, é o **teto** do
decoder: o melhor que ele traduz se o encoder fosse perfeito.

- **Teto ~22–25:** a meta de BLEU-4 30 está acima do que esse decoder
  consegue. A prioridade passa a ser trocar ou ampliar o decoder (ex.: mBART).
- **Teto bem acima de 17,7:** há espaço; a perda está na ponte entre encoder e
  decoder, e o pré-treino (etapa 2) e o alinhamento (4.1) são o caminho.

Na literatura, o Gloss2Text no PHOENIX-2014T costuma ficar na faixa de 20 e
poucos de BLEU-4 com Transformers treinados do zero, e um pouco acima com
modelos de linguagem pré-treinados.

### 4.4 Tokenização das glosas

As glosas do PHOENIX vêm em maiúsculas e com marcações (`__ON__`, `loc-`,
`IX`, sufixos numéricos). O tokenizer do GermanT5 vai quebrá-las em subpalavras.
Vale testar:

- glosas em minúsculas (aproximam as subpalavras do alemão comum);
- remover ou normalizar marcações sem conteúdo lexical (ex.: `__ON__`, `__OFF__`).

### 4.5 Mesmo tokenizer nas duas etapas

A etapa 2 carrega o T5 da etapa 1 com `--t5 <pasta>`. O tokenizer salvo junto
precisa ser o mesmo do GermanT5 original. Se as glosas forem acrescentadas como
tokens novos, o vocabulário muda, e isso precisa ser tratado nas duas etapas.

---

## 5. Plano de implementação

1. **Script `gloss2text`** (ex.: `skeltrans/training/gloss2text.py`, com um
   atalho na raiz como `evaluate_slt.py`):
   - lê os manifestos (`tokens` e `text`) de treino, validação e teste;
   - treina `T5ForConditionalGeneration` de texto para texto (AdamW, warmup
     linear, como no treino atual);
   - avalia BLEU/METEOR na validação a cada época e escolhe o melhor (como
     `--select-by bleu4`);
   - salva o melhor modelo **no formato do Hugging Face** (`save_pretrained`
     com o tokenizer), para ser usado direto como `--t5`;
   - no fim, avalia no teste e grava o **teto** em JSON;
   - flags previstas: `--lowercase-glosses`, `--gloss-noise <taxa>`,
     `--label-smoothing`, `--seed`, `--eval-beams`.
2. **Treino de sinais:** nenhuma mudança obrigatória. Usar
   `--t5 checkpoints_gloss2text/best`.
3. **Opcional, depois:** perda de alinhamento (4.1) como flag do treino de sinais.
4. **Testes:** dados sintéticos + T5 minúsculo, como nos testes atuais, e
   regressão bit a bit garantindo que o treino de sinais sem as flags novas não muda.

---

## 6. Plano de experimentos

| # | Experimento | Pergunta |
|---|---|---|
| 1 | Gloss2Text com glosas limpas, teste | Qual é o **teto** do decoder? |
| 2 | Treino de sinais com `--t5` = T5 do exp. 1 | O pré-treino melhora o BLEU? |
| 3 | Gloss2Text com glosas ruidosas (taxa ≈ WER) → treino de sinais | O ruído deixa o decoder mais robusto? |
| 4 | Glosas em minúsculas vs. originais | A tokenização importa? |
| 5 | (opcional) + perda de alinhamento | Aproximar os espaços ajuda? |

Para cada um, comparar **BLEU-4**, **METEOR** e **WER** com a melhor
configuração atual, de preferência com 2–3 valores de `--seed`.

---

## 7. Riscos

- **Overfitting na etapa 1:** o corpus é pequeno; usar early stopping pela
  validação e label smoothing.
- **Esquecimento na etapa 2:** um `--lr` alto no decoder pode apagar o que ele
  aprendeu na etapa 1; manter o decoder com lr baixo (o `--encoder-lr` já
  permite lr maior só no encoder).
- **Descompasso entre os espaços:** se o encoder de landmarks não se aproximar
  do espaço que o decoder espera, o ganho fica pequeno; aí a perda de
  alinhamento (4.1) passa a ser necessária.

---

## 8. Referências (conferir antes de citar)

- Camgöz et al., *Neural Sign Language Translation*, CVPR 2018 — introduz o
  PHOENIX-2014T e os cenários Sign2Gloss2Text e Gloss2Text.
- Yin & Read, *Better Sign Language Translation with STMC-Transformer*,
  COLING 2020 — Gloss2Text forte no PHOENIX-2014T.
- Chen et al., *A Simple Multi-Modality Transfer Learning Baseline for Sign
  Language Translation*, CVPR 2022 — pré-treino progressivo: encoder visual em
  Sign2Gloss, modelo de linguagem (mBART) em Gloss2Text, e um mapeador entre os dois.
