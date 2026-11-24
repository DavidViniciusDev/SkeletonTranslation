# Encoder pré-treinado por masked pose modeling (`pretrain_encoder.py` + `--init-encoder`)

> Preparação (`$PY`, `$DATA`, `$FEAT`) e receita base (`$BASE`) em
> [README.md](README.md). Código:
> [skeltrans/training/models/masked_pose.py](../../skeltrans/training/models/masked_pose.py)
> (modelo e máscaras), [skeltrans/training/pretrain.py](../../skeltrans/training/pretrain.py)
> (loop e CLI) e [skeltrans/training/encoder_init.py](../../skeltrans/training/encoder_init.py)
> (arquivo `encoder.pt` e `--init-encoder`).

## O que é

Hoje o encoder de landmarks começa com pesos aleatórios e aprende tudo com
~7 mil frases do PHOENIX. O pré-treino dá a ele uma fase anterior, **sem
rótulos**: esconder trechos dos keypoints e pedir que os reconstrua a partir do
resto, como o BERT faz com palavras.

```
FASE 1: pretrain_encoder.py (sem rótulo)       FASE 2: slt_model.py (o treino de hoje)
keypoints com trechos escondidos               keypoints
        │                                              │
     ENCODER ──► cabeça de reconstrução           ENCODER ──► CTC + adapter + T5
        │                                              ▲
  erro só nos trechos escondidos                       │
        │                                              │
        └──► encoder.pt ─────── --init-encoder ────────┘
```

Para reconstruir a mão escondida no frame 47, o encoder precisa saber como
mãos se movem, que dedos se dobram juntos e o que costuma vir antes e depois.
É o mesmo conhecimento que separa sinais parecidos, e as **substituições** são
o maior erro do WER (≈23 de 37,5 na v15).

## Onde ficam os pesos

| Arquivo | Gerado por | Contém | Usado por |
|---|---|---|---|
| `checkpoints_pretrain\encoder.pt` | `pretrain_encoder.py` | só o encoder (sem adaptador) + a spec + estatísticas do `--input-norm` | `slt_model.py --init-encoder`, **uma vez**, no início do treino |
| `checkpoints_phoenix_xxx\best.pt` | `slt_model.py` | o modelo inteiro (encoder ajustado + CTC + T5) | avaliação e inferência, como sempre |

A inferência **não muda**: o `best.pt` é autossuficiente.

## Passo 1: pré-treinar

```powershell
& $PY pretrain_encoder.py `
    --manifests $DATA\interim\train.json `
    --val-manifest $DATA\interim\dev.json `
    --features-dir $FEAT `
    --extra-features A,B,C,D,E,F --input-norm `
    --d-model 512 --nhead 8 --num-layers 6 --ds-steps 2 `
    --mask-mode mixed --mask-ratio 0.4 --mask-span 8 `
    --epochs 100 --patience 10 --batch-size 32 --lr 5e-4 --warmup-steps 1000 `
    --low-vram --grad-checkpoint --seed 1 `
    --out-dir checkpoints_pretrain
```

O log de cada época mostra quanto do sinal escondido o encoder recuperou
(números ilustrativos):

```
[epoch 12] train_mse=0.4210 val_mse=0.3982 (prever 0: 1.0031; recuperado 60.3%)
  -> encoder.pt (melhor ate agora, epoca 12)
```

- **`prever 0`** é o erro de um modelo que não sabe nada (com `--input-norm`, ≈1).
- **`recuperado`** é `1 − val_mse / prever 0`. Se ficar perto de 0%, o
  encoder não está aprendendo; acima de 50%, já entende o movimento.

**Mais dados sem rótulo** é onde o ganho deve aparecer. Qualquer manifesto
com `.npy` no mesmo formato (115 × 9, mesma normalização) serve, inclusive
sem texto:

```powershell
& $PY pretrain_encoder.py --manifests $DATA\interim\train.json outro_dataset\manifest.json libras_sintetico\manifest.json ...
```

Os keypoints do conjunto de teste também poderiam entrar (não há rótulo
envolvido), mas o script avisa. Declare isso na tese ou deixe o teste de fora.

## Passo 2: treinar o SLT a partir do encoder pré-treinado

As flags de **encoder** precisam ser iguais às do pré-treino: `--extra-features`,
`--input-norm`, `--encoder-type`, `--d-model`, `--nhead`, `--num-layers` e
`--ds-steps`. O resto da receita não muda:

```powershell
& $PY slt_model.py @BASE --init-encoder checkpoints_pretrain\encoder.pt `
    --out-dir checkpoints_phoenix_mpm
```

O log confirma (números ilustrativos):

```
[init-encoder] pesos do encoder vindos de checkpoints_pretrain\encoder.pt (pre-treino: 34 epocas, mascara mixed 0.4, val_mse 0.39...); adaptador e CTC comecam do zero
[input-norm] usando a media/desvio do encoder pre-treinado
```

Se alguma flag não bater, o treino **nem começa**, e a mensagem diz qual:

```
--init-encoder nao combina com este treino (use as mesmas flags do pre-treino):
  num_layers: pre-treino=6, treino=4 (--num-layers)
```

O `--dropout` pode mudar entre as fases.

### Combinações

- **Com `--temporal-mode concept`:** pré-treine com `--num-layers` igual ao
  andar de frames (ex.: 4). O andar de conceitos começa do zero.
- **Com `--encoder-type parts` ou `gcn`:** pré-treine com o mesmo tipo.
- **`--encoder-lr`:** o encoder já não começa do zero. Se o treino ficar
  instável no início, teste um `--encoder-lr` menor (ex.: `1e-4` em vez de `3e-4`).

## Avaliar

Igual a qualquer outro treino (o `best.pt` contém tudo):

```powershell
& $PY evaluate_slt.py --checkpoint checkpoints_phoenix_mpm\best.pt `
    --test-manifest $DATA\interim\test.json --features-dir $FEAT `
    --wer --bertscore de --out checkpoints_phoenix_mpm\test_metrics.json
```

## Ablações sugeridas

Compare com a v15 (BLEU-4 19,22, WER 37,5), mesma receita e mesma semente:

| # | Pergunta | Mudança |
|---|---|---|
| 1 | O pré-treino ajuda com os mesmos dados? | pré-treino só no `train.json`; SLT com `--init-encoder` |
| 2 | Mais keypoints sem rótulo ajudam mais? | pré-treino com `train.json` + outros manifestos |
| 3 | Qual máscara ensina mais? | `--mask-mode frames` × `hands` × `mixed` |
| 4 | Quanto esconder? | `--mask-ratio 0.25` × `0.4` × `0.6` |

**O que observar:** o pré-treino deve reduzir primeiro as **substituições** do
WER. Se o WER cair e o BLEU não, o gargalo continua sendo a tradução (ver
`ideas/pretreino-decoder-gloss2text.md`).

## Como funciona (resumo técnico)

- **Máscaras:** trechos contíguos de `--mask-span` frames até cobrir
  `--mask-ratio` dos frames reais de cada frase. `frames` esconde o frame
  inteiro; `hands`, uma das mãos (sorteada); `mixed` sorteia o tipo por trecho.
  O valor escondido é 0, o mesmo de um ponto não detectado.
- **Features extras:** são calculadas das posições e entregariam a resposta;
  somem em todo frame mascarado.
- **Alvo:** as 1035 colunas base (posição, velocidade e aceleração), no espaço do
  `--input-norm` quando ativo. Erro quadrático só nos pontos escondidos,
  presentes e em frames reais.
- **Downsampling:** com `--ds-steps n`, cada posição do encoder reconstrói os
  2ⁿ frames que resume.
- **Arquivo:** `encoder.pt` guarda só tensores e tipos simples e é lido com
  `weights_only=True` (não executa código do arquivo).

## Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `nao e um encoder pre-treinado do SkeLTrans` | passou um `best.pt` | use o `encoder.pt` do `pretrain_encoder.py` |
| `nao combina com este treino` | flags de encoder diferentes | repita no SLT as flags do pré-treino |
| `recuperado` perto de 0% | máscara fácil ou difícil demais, lr alto | teste `--mask-ratio 0.25` ou `--lr 3e-4` |
| OOM no pré-treino | cabeça de reconstrução (2ⁿ × 1035 saídas por posição) | `--batch-size 16`, `--low-vram --grad-checkpoint` |

## Testes

```powershell
& $PY pretrain_encoder.py --smoke-test
& $PY tests\test_pretrain_encoder.py
```

Os 19 testes cobrem: máscaras (cobertura, padding, modos, reprodutibilidade),
reconstrução (formatos com qualquer `--ds-steps`, pontos ausentes fora da
perda, aprendizado em sinais sintéticos), o arquivo `encoder.pt` (ida e volta,
rejeição de outros arquivos), o `--init-encoder` (pesos e estatísticas
copiados, adaptador intocado, modo `concept`, spec diferente recusada), as
duas CLIs e o Dataset sem texto.
