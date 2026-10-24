# Encoder por partes do corpo (`--encoder-type parts`)

`--encoder-type` escolhe como o encoder transforma cada frame no vetor de
`d_model` números que segue para a convolução temporal e o Transformer. O
resto do encoder (conv temporal, `--ds-steps`, Transformer, adaptador para o
T5) **não muda**.

| `--encoder-type` | Projeção de entrada |
|---|---|
| `flat` (padrão) | um `Linear(1035, d_model)` sobre o frame achatado — o original |
| `parts` | uma MLP por parte do corpo + uma camada de fusão |

O código fica em
[`skeltrans/training/models/parts_projection.py`](skeltrans/training/models/parts_projection.py).

## Motivação

No `flat`, os 1035 números do frame entram juntos numa soma ponderada: para
essa camada, dois dedos vizinhos não têm mais relação do que um dedo e uma
sobrancelha, e as mãos disputam a mesma projeção com os 40 pontos da face. No
`parts`, cada parte ganha uma rede própria antes da junção:

```
pose     (33 pontos x 9 canais) -> MLP ->┐
mão esq  (21 pontos x 9 canais) -> MLP ->│
mão dir  (21 pontos x 9 canais) -> MLP ->├-> concatena -> Linear -> d_model
face     (40 pontos x 9 canais) -> MLP ->│
extras   (--extra-features)     -> MLP ->┘   (só se houver extras)
```

- Cada MLP é `Linear -> GELU -> Linear`, com `d_model / 4` unidades (128 com
  `--d-model 512`).
- As colunas do `--extra-features` misturam partes (mão, face, presença),
  então formam um **fluxo próprio** em vez de serem redistribuídas.
- Com `--input-norm`, a normalização é aplicada antes da divisão em partes,
  como no `flat`.

## Uso

```bash
# braço de controle e braço com partes, mesmos hiperparâmetros no resto
python3 slt_model.py ... --encoder-type flat  --out-dir ck_flat
python3 slt_model.py ... --encoder-type parts --out-dir ck_parts

# combinado com as features extras e o lr separado
python3 slt_model.py ... --encoder-type parts --extra-features A,F --input-norm \
    --lr 5e-5 --encoder-lr 5e-4 --out-dir ck_parts_AF
```

Para conferir a instalação sem dados: `python3 slt_model.py --smoke-test --encoder-type parts`.

## Custo

- **Parâmetros:** com `d_model 512`, a projeção por partes tem **menos**
  parâmetros que a `flat`: 0,46M contra 0,53M (com `--extra-features A,F`,
  0,57M contra 0,62M). Cada MLP só enxerga as colunas da sua parte, em vez de
  um `Linear` ligando todas as 1035 colunas a todas as 512 saídas. Então o
  ganho, se houver, vem da **estrutura**, não de mais capacidade. O total do
  modelo é dominado pelo T5 e pelo Transformer do encoder.
- **Tempo:** as MLPs atuam por frame, antes da atenção; o custo extra é
  pequeno perto da atenção e do decoder.

## Checkpoints

- O tipo do encoder é gravado no config (`args.model.encoder_type`), mas, ao
  carregar, **os pesos mandam**: um checkpoint com as chaves
  `encoder.input_proj.parts.*` é reconstruído como `parts`, mesmo que o config
  diga outra coisa (com um aviso). Isso evita carregar um `parts` num `flat`
  com a projeção de entrada aleatória.
- `evaluate_slt.py` e `infer_slt.py` não têm flags novas.
- Checkpoints `flat` e `parts` não são intercambiáveis: a projeção de entrada
  tem pesos diferentes. Checkpoints anteriores à flag carregam como `flat`.
- Com o padrão (`flat`), a arquitetura, o `state_dict` e o treino são
  idênticos aos de antes — verificado bit a bit num treino com semente fixa.

## Testes

```bash
python3 tests/test_encoder_parts_lr.py
```

Cobre: formas da saída com e sem extras, isolamento entre os fluxos (mudar a
face não altera o fluxo da mão), forward/backward com gradiente em todos os
pesos, chaves do `flat` inalteradas, ciclo salvar → carregar, detecção do tipo
pelos pesos e erro com grupos incompatíveis. Também cobre o `--encoder-lr`:
grupos e lrs corretos, todos os parâmetros cobertos (incluindo o head CTC) e a
proporção entre os lrs mantida pelo warmup e pelo decaimento.
