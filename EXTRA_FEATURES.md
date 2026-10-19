# Features extras de entrada e normalização (`--extra-features`, `--input-norm`)

Por padrão o encoder recebe, por frame, as 1035 colunas do Passo 3: posição,
velocidade e aceleração dos 115 pontos. Estas duas flags, **desligadas por
padrão**, mudam essa entrada sem reextrair a base:

- `--extra-features GRUPOS` acrescenta colunas calculadas **sob demanda** (no
  `Dataset`, ao carregar cada `.npy`) a partir das posições;
- `--input-norm` normaliza cada coluna (z-score) com média e desvio do treino.

O código fica em [`skeltrans/training/data/features.py`](skeltrans/training/data/features.py).

## Grupos

Os grupos seguem os parâmetros fonológicos da Libras:

| Grupo | Parâmetro | O que calcula | Colunas |
|---|---|---|---|
| `A` | Configuração de mão | mão centrada no punho e ÷ tamanho da palma (63), flexão nas 15 articulações dos dedos em radianos (π = esticado), distâncias entre as 5 pontas ÷ palma (10) — por mão | 176 |
| `B` | Orientação da palma | vetor normal unitário da palma, por mão | 6 |
| `C` | Ponto de articulação | punho e ponta do indicador até nariz, boca, peito e cada ombro (por mão) + punho a punho | 21 |
| `D` | Movimento | velocidade escalar de cada punho | 2 |
| `E` | Não manuais | abertura da boca, abertura dos olhos e altura das sobrancelhas (razões) | 5 |
| `F` | Presença | 1/0 por grupo detectado: pose, mão esq., mão dir., face | 4 |

Com todos os grupos, a entrada passa de 1035 para **1249** colunas. A ordem das
letras não importa (`F,A` = `A,F`).

- `A`, `B` e `E` são razões ou ângulos: **não dependem da escala** do frame.
- Mão (ou face) não detectada → as features dela ficam em 0, e o grupo `F`
  indica que esse 0 significa "ausente". Por isso `A` deve vir com `F`.
- As escalas dos grupos são diferentes (radianos, razões, 0/1), então use
  `--input-norm` junto.

## Normalização (`--input-norm`)

Antes do treino começar, o rank 0 percorre o conjunto de **treino** uma vez e
calcula média e desvio de cada coluna sobre todos os frames reais. A
validação e o teste nunca entram nessas estatísticas. Colunas constantes ficam
com desvio 1.

As estatísticas ficam em buffers do encoder (`encoder.input_mean`,
`encoder.input_std`), ou seja, **dentro do `.pt`**. O padding continua 0
depois da normalização.

## Uso

```bash
# ablation, uma execução por linha (mesmos hiperparâmetros no resto)
python3 slt_model.py ... --input-norm                       --out-dir ck_norm
python3 slt_model.py ... --input-norm --extra-features A,F  --out-dir ck_AF
python3 slt_model.py ... --input-norm --extra-features A,C,F --out-dir ck_ACF
```

Acompanhe com `--eval-every` (BLEU/METEOR na validação) e, com `--use-ctc`,
pelo WER de glosas no `evaluate_slt.py --wer`.

## Avaliação, inferência e checkpoints

- `evaluate_slt.py` e `infer_slt.py` **não têm flags novas**: os grupos ficam
  gravados no checkpoint (`args.model.extra_features`), e a normalização vem
  nos buffers. A entrada é montada pela mesma função do treino.
- Ao carregar, a dimensão do `input_proj` é conferida contra os grupos
  gravados; se não baterem, o carregamento falha com mensagem explícita.
- Checkpoints anteriores a estas flags carregam como antes: 1035 colunas, sem
  normalização.
- Sem as flags, a entrada e o `state_dict` são idênticos aos de antes.

## Custo

Calcular todos os grupos leva ~2,6 ms por frase de ~600 frames num núcleo de
CPU. Isso roda nos workers do `DataLoader`, em paralelo com a GPU. Se o uso da
GPU (`nvidia-smi`) cair durante o treino, aumente `--num-workers`. A
normalização percorre o treino uma vez no início (usando os mesmos workers).

## Testes

```bash
python3 tests/test_extra_features.py
```

Cobre: dimensões, entrada idêntica sem grupos, invariância à escala e à
translação, mão ausente zerada e sinalizada, ângulo de dedo esticado = π,
padding em 0 após a normalização e o ciclo salvar → carregar checkpoint.
