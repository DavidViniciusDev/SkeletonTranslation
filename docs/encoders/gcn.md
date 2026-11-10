# Encoder por grafo do esqueleto (`--encoder-type gcn`)

> Preparação (`$PY`, `$DATA`, `$FEAT`) e receita base (`$BASE`) em
> [README.md](README.md). Documentação completa: [ENCODER_GCN.md](../../ENCODER_GCN.md).

## O que é

Cada um dos 115 pontos vira um **nó** com seus 9 canais (posição, velocidade,
aceleração). Cada camada mistura a informação de um nó só com os **vizinhos no
esqueleto**: dedo com dedo, dedo com a mão, mão com o pulso. A forma da mão
passa a ser representada pela própria estrutura da rede.

```
frame (115 nós × 9 canais)
  ─► 3 camadas GCN (vizinhança em 3 partições, estilo ST-GCN, + grafo aprendido)
  ─► média dos nós PRESENTES de cada parte (pose · mão esq · mão dir · face)
  ─► + fluxo das --extra-features ─► concatena ─► Linear ─► d_model
```

Pontos ausentes (mão não detectada) não diluem os vizinhos: a média de cada
vizinhança usa só os nós presentes.

## Treinar

```powershell
& $PY slt_model.py @BASE --encoder-type gcn --out-dir checkpoints_phoenix_gcn
```

Combinado com o modo por conceitos:

```powershell
& $PY slt_model.py @BASE --encoder-type gcn --temporal-mode concept `
    --num-layers 4 --concept-layers 2 --out-dir checkpoints_phoenix_gcn_concept
```

## Avaliar

```powershell
& $PY evaluate_slt.py --checkpoint checkpoints_phoenix_gcn\best.pt `
    --test-manifest $DATA\interim\test.json --features-dir $FEAT `
    --out checkpoints_phoenix_gcn\test_metrics.json
```

O tipo `gcn` é detectado pelos pesos (`encoder.input_proj.gcn_layers.*`).

## Custo

Medido no encoder isolado (RTX 5070 Ti, bf16, `--grad-checkpoint`, lote de 32,
`--extra-features A,B,C,F`):

| Lote | `flat` | `gcn` |
|---|---|---|
| 110 frames por frase | 20 ms · 0,3 GB | 141 ms · 2,2 GB |
| comprimentos realistas (máx. 475 frames) | 44 ms · 0,6 GB | 191 ms · 2,3 GB |

A GCN é a variante mais cara em tempo e memória. Se faltar VRAM, baixe
`--batch-size` (sobrescreva o da receita: `--batch-size 16`).

## Resultado registrado

Na nota [ideas/pretreino-decoder-gloss2text.md](../../ideas/pretreino-decoder-gloss2text.md),
`gcn` não melhorou o BLEU em relação à v11.

## Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| OOM | 115 nós por frame em todos os frames | `--batch-size 16`, mantenha `--grad-checkpoint --low-vram` |
| `input_dim` menor que 1035 | entrada sem as 115 × 9 colunas base | confira `$FEAT` (precisa ser `landmarks_115_norm9`) |

## Testes

```powershell
& $PY tests\test_encoder_gcn.py
```
