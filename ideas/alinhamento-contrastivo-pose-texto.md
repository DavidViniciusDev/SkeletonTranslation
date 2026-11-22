# Ideia: alinhamento contrastivo entre pose e texto (inspirado no SignCLIP)

> **Status:** ideia registrada, ainda não implementada.
> **Origem:** leitura do SignCLIP (Jiang et al., 2024), nov/2026.
> **Objetivo:** aproximar a representação do encoder de landmarks da representação da linguagem, para o decoder aproveitar melhor o que o encoder já reconhece.

---

## 1. O que é o SignCLIP

Repete a ideia do CLIP para línguas de sinais: um **encoder de pose** e um
**encoder de texto** (BERT) são treinados juntos para que um vídeo e a frase que
o descreve fiquem **próximos** num espaço comum, e pares que não combinam
fiquem **longe**. A perda é contrastiva (InfoNCE).

| Ponto | SignCLIP |
|---|---|
| Entrada | keypoints do **MediaPipe Holistic** (rosto reduzido ao contorno: 203 pontos), normalizados pelos ombros |
| Dados | Spreadthesign: ~457 mil pares vídeo–texto, ~41 línguas de sinais (**não público**, exige licença) |
| Tarefa | busca vídeo ↔ texto e reconhecimento de sinais isolados (**não faz tradução**) |
| Código | fork do fairseq (`J22Melody/fairseq`, `examples/MMPT`) |

**Já aproveitado neste projeto:**

- a **receita de augmentation** (velocidade, espacial 2D, ruído, espelhamento) → `--aug-*`, ver `AUGMENTATION.md`;
- a **padronização dos keypoints** pela média e desvio da base foi o melhor resultado entre as variantes do artigo, o que valida o `--input-norm`;
- em soletração, keypoints do MediaPipe **superaram características de vídeo** (I3D, S3D): argumento para a escolha por esqueleto na tese.

---

## 2. A ideia para este projeto

Hoje o encoder aprende a produzir a memória só pelo que o decoder consegue
traduzir dela (CE) e pelo CTC de glosas. Nada pede explicitamente que essa
memória "fale a língua" do decoder. A proposta: uma perda que puxe a
representação da frase em sinais para perto da representação da frase em
texto.

```
frase em sinais ──► LandmarkEncoder ──► pooling ──► z_sinal ─┐
                                                              ├─► InfoNCE: par certo perto, os outros longe
frase em texto  ──► encoder de texto ──► pooling ──► z_texto ─┘
   (alemão ou glosas)
```

### Variantes

| # | Variante | Custo | Observação |
|---|---|---|---|
| A | **Etapa de pré-treino** do encoder (só InfoNCE) antes do treino de tradução | médio | parecida com o pré-treino "VLP" de trabalhos de tradução sem glosas (GFSLT-VLP), que tiveram ganhos com vídeo RGB |
| B | **Perda auxiliar** somada ao treino atual: `CE + λ·CTC + μ·InfoNCE` | médio | um treino só; mais um peso para calibrar |
| C | Alinhar com o **encoder de texto do T5 da etapa Gloss2Text** | alto | é a "perda de alinhamento" da nota `pretreino-decoder-gloss2text.md` (seção 4.1); o contrastivo é um jeito de implementá-la |
| D | Usar o **encoder pré-treinado do SignCLIP** como ponto de partida | alto, incerto | pesos públicos não confirmados; outro conjunto de pontos (203); contexto de 256 frames (o PHOENIX vai até 475); sinais isolados, não frases |

### O que alinhar com o quê

- **Com o alemão:** é o alvo final, mas a ordem e o vocabulário diferem muito dos sinais.
- **Com as glosas:** mais próximas do que o encoder vê (mesma ordem, um sinal por glosa), e combinam com o modo `concept` (uma posição por sinal), que deixa a memória com uma estrutura parecida com a das glosas.

Começaria pelas glosas: é o caso em que o alinhamento é mais fácil de aprender.

---

## 3. Detalhes que importam

- **Lotes grandes:** a InfoNCE compara cada par com os outros do lote (os
  negativos). O SignCLIP usou lotes de 448. Nas TITAN V, com o decoder grande,
  isso não cabe; alternativas: fila de negativos (*memory bank*), ou a etapa de
  pré-treino (variante A) sem o decoder, o que libera memória.
- **Pooling:** reduzir a sequência a um vetor (média sobre as posições válidas,
  respeitando a máscara reduzida pelo downsampling) — ou alinhar por sinal no
  modo `concept`.
- **Encoder de texto:** BERT alemão, ou o encoder do próprio T5 (que hoje nunca
  executa). Usar o do T5 manteria tudo no mesmo espaço do decoder.
- **Temperatura** da InfoNCE: hiperparâmetro sensível; aprender junto, como no CLIP.

---

## 4. Plano

1. **Variante A com glosas:** pré-treinar o `LandmarkEncoder` com InfoNCE contra
   o encoder de texto do T5 (congelado) sobre as glosas; depois o treino de
   tradução atual, partindo desse encoder.
2. Comparar com a melhor config (v15), 2–3 seeds: **BLEU-4**, **METEOR** e **WER**.
3. Se ajudar: testar a variante B (perda auxiliar) e o alinhamento com o alemão.
4. Implementação: flag de pré-treino no treino (ex.: `--pretrain-contrastive`),
   salvando só o encoder, e uma flag para carregá-lo no treino de tradução.
   Testes com dados sintéticos e T5 minúsculo, e regressão garantindo que, sem
   as flags, nada muda.

---

## 5. Riscos

- **Tamanho do lote:** com poucos negativos, a InfoNCE aprende pouco.
- **Base pequena:** ~7 mil frases é pouco para aprendizado contrastivo; o
  SignCLIP usou ~457 mil pares.
- **Objetivo desalinhado:** aproximar representações não garante tradução
  melhor; medir sempre pelo BLEU e pelo WER, não pela perda contrastiva.

---

## 6. Referências (conferir antes de citar)

- Jiang, Sant, Moryossef, Müller, Sennrich e Ebling, *SignCLIP: Connecting Text and Sign Language by Contrastive Learning*, arXiv 2407.01264, 2024.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021.
- Zhou et al., *Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining* (GFSLT-VLP), ICCV 2023.
