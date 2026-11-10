# Encoder por partes do corpo (`--encoder-type parts`)

> Preparação (`$PY`, `$DATA`, `$FEAT`) e receita base (`$BASE`) em
> [README.md](README.md). Documentação completa: [ENCODER_PARTS.md](../../ENCODER_PARTS.md).

## O que é

No `flat`, os 1242 números de um frame passam por um único `Linear`: para essa
camada, dois dedos vizinhos não têm mais relação do que um dedo e uma
sobrancelha. No `parts`, cada parte do corpo tem a sua MLP, e só depois as
partes se juntam:

```
pose     (33 pts × 9) ─► MLP ─┐
mão esq  (21 pts × 9) ─► MLP ─┤
mão dir  (21 pts × 9) ─► MLP ─┼─► concatena ─► Linear ─► d_model
face     (40 pts × 9) ─► MLP ─┤
extras (--extra-features) ─► MLP ─┘
```

O resto do encoder (conv temporal, `--ds-steps`, Transformer, CTC) não muda.

## Treinar

```powershell
& $PY slt_model.py @BASE --encoder-type parts --out-dir checkpoints_phoenix_parts
```

Combinado com o modo por conceitos (projeção por partes + memória por sinal):

```powershell
& $PY slt_model.py @BASE --encoder-type parts --temporal-mode concept `
    --num-layers 4 --concept-layers 2 --out-dir checkpoints_phoenix_parts_concept
```

## Avaliar

```powershell
& $PY evaluate_slt.py --checkpoint checkpoints_phoenix_parts\best.pt `
    --test-manifest $DATA\interim\test.json --features-dir $FEAT `
    --out checkpoints_phoenix_parts\test_metrics.json
```

O tipo `parts` é detectado pelos pesos (`encoder.input_proj.parts.*`).

## Custo

- **Parâmetros:** a projeção tem **menos** parâmetros que a `flat` (cerca de
  0,5 M contra 0,6 M com `d_model 512`). Se houver ganho, ele vem da
  estrutura, não de mais capacidade.
- **Tempo:** praticamente o mesmo da `flat`. As MLPs atuam por frame, antes
  da atenção.

## Resultado registrado

Na nota [ideas/pretreino-decoder-gloss2text.md](../../ideas/pretreino-decoder-gloss2text.md),
`parts` não melhorou o BLEU em relação à v11. Vale manter como braço de
ablação e testar combinado com `--temporal-mode concept`.

## Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `checkpoint inconsistente: input_proj espera ...` | `--extra-features` do config não bate com os pesos | use o checkpoint com a mesma entrada do treino |
| Aviso `config diz encoder_type=flat, mas os pesos ...` | config antigo/editado | nada a fazer: os pesos mandam |

## Testes

```powershell
& $PY tests\test_encoder_parts_lr.py
```
