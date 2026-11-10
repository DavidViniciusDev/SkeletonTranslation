# Encoder por conceitos (`--temporal-mode concept`)

> Preparação (`$PY`, `$DATA`, `$FEAT`) e receita base (`$BASE`) em
> [README.md](README.md). Diagrama didático:
> [documentations/diagrama-encoder-conceitos.drawio](../../documentations/diagrama-encoder-conceitos.drawio).
> Código: [skeltrans/training/models/concept_pooling.py](../../skeltrans/training/models/concept_pooling.py).

## O que é

No modo `stride` (o original), o `--ds-steps` encurta o vídeo **às cegas**:
junta 2^n frames vizinhos sem saber onde um sinal termina. O decoder T5 recebe
dezenas de "fatias" sem significado próprio.

No modo `concept`, a memória do decoder tem **uma posição por sinal**:

```
frames ─► projeção ─► conv / ds-steps ─► Transformer de FRAMES ─┬─► head CTC ─► disparos
                                                                │                  │
                                                                ▼                  ▼
                                                    pooling por sinal ◄── fronteiras (meio entre disparos)
                                                                │
                                                                ▼
                                             Transformer de CONCEITOS ─► adapter ─► T5
```

1. O **Transformer de frames** (`--num-layers`) contextualiza cada frame.
2. O **CTC** roda sobre esses frames e dispara em 1–2 frames por sinal.
3. O corte entre dois sinais fica no **ponto médio entre os disparos**, e todo
   frame pertence a algum sinal.
4. O **pooling** resume cada sinal em `K` vetores (média ponderada por atenção:
   o pico do sinal pesa mais que a transição).
5. O **Transformer de conceitos** (`--concept-layers`) relaciona os sinais.
6. O T5 faz cross-attention sobre ~N·K posições (N = nº de sinais) em vez de T/4.

## De onde vêm as fronteiras

| Fase | Fonte | Como |
|---|---|---|
| Treino (`--concept-boundaries align`, padrão) | glosas verdadeiras | alinhamento forçado do CTC (Viterbi): **exatamente 1 disparo por glosa**, inclusive no início do treino |
| Treino (`--concept-boundaries greedy`) | o próprio CTC | argmax por frame, como na inferência |
| Validação e inferência | o próprio CTC | sempre greedy (não há glosas) |

`--concept-greedy-prob p` usa o greedy em uma fração `p` dos lotes de treino,
mesmo com as glosas disponíveis, para o treino ver as fronteiras imperfeitas
que vai encontrar na inferência.

## Requisitos

- `--use-ctc` com `--ctc-weight > 0`: o CTC define as fronteiras. A CLI aborta
  com mensagem clara se faltar.
- O manifesto precisa do campo `tokens` (glosas), que o PHOENIX já tem.
- Depois do `--ds-steps`, cada frase precisa de mais frames que glosas. Com
  `--ds-steps 2`, a mediana do PHOENIX (110 frames → 27) fica bem acima das
  ~7 glosas por frase. Frase sem alinhamento possível vira 1 sinal, sem erro.

## Flags

| Flag | Padrão | O que faz |
|---|---|---|
| `--temporal-mode` | `stride` | `concept` liga o modo por sinal |
| `--concept-layers` | 2 | camadas do Transformer entre sinais |
| `--concept-queries` | 1 | vetores por sinal (K); K > 1 alivia o gargalo de poucas posições |
| `--concept-boundaries` | `align` | fonte das fronteiras no treino |
| `--concept-greedy-prob` | 0 | fração dos lotes de treino com fronteiras do greedy |
| `--num-layers` | 6 | no `concept`, é o andar de **frames** |

## Treinar

Configuração principal, com o mesmo total de camadas da v11 (4 de frames + 2
de conceitos = 6):

```powershell
& $PY slt_model.py @BASE --temporal-mode concept `
    --num-layers 4 --concept-layers 2 `
    --out-dir checkpoints_phoenix_concept
```

O log confirma o modo no início:

```
[concept] memoria por sinal | fronteiras no treino: align (greedy p=0.0); validacao/inferencia: greedy | 4 camadas de frames + 2 de conceitos | 1 vetor(es) por sinal
```

## Ablações sugeridas

Rode cada uma contra a v11 (`checkpoints_phoenix_flat`), mudando só o indicado:

| # | Pergunta | Flags além de `@BASE --temporal-mode concept --num-layers 4` |
|---|---|---|
| 1 | O modo por sinal ajuda? | `--concept-layers 2` |
| 2 | Uma posição por sinal é pouca? | `--concept-layers 2 --concept-queries 2` (e `4`) |
| 3 | A diferença treino × inferência pesa? | `--concept-layers 2 --concept-greedy-prob 0.3` |
| 4 | Menos downsampling ajuda o CTC a achar os sinais? | `--concept-layers 2 --ds-steps 1` |
| 5 | Combina com a projeção espacial? | `--concept-layers 2 --encoder-type parts` (ou `gcn`) |

Exemplo (ablação 2):

```powershell
& $PY slt_model.py @BASE --temporal-mode concept --num-layers 4 `
    --concept-layers 2 --concept-queries 2 --out-dir checkpoints_phoenix_concept_k2
```

**O que observar:** se o WER melhora e o BLEU não, as fronteiras estão boas
mas o gargalo é a tradução (ver a nota de pré-treino Gloss2Text em `ideas/`).
Se o BLEU cai com K=1 e sobe com K=2–4, uma posição por sinal era pouca
informação para o decoder.

## Avaliar

```powershell
& $PY evaluate_slt.py --checkpoint checkpoints_phoenix_concept\best.pt `
    --test-manifest $DATA\interim\test.json --features-dir $FEAT `
    --out checkpoints_phoenix_concept\test_metrics.json
```

- O modo `concept`, `--concept-layers` e `--concept-queries` são detectados
  pelos pesos (`encoder.concept_pool.*`, `encoder.concept_transformer.*`).
- O WER de glosas é calculado sobre o andar de **frames**, onde o CTC roda.
- A validação durante o treino (`val_loss`, `--eval-every`) já usa as
  fronteiras do greedy, ou seja, mede o modelo como ele será usado.

## Custo

- **Parâmetros:** com `--num-layers 4 --concept-layers 2`, o encoder tem o
  mesmo número de camadas Transformer da v11. O pooling acrescenta poucos
  pesos: uma pontuação por query (`d_model × K`), um LayerNorm e, com K > 1,
  um embedding por query.
- **Atenção:** o andar de conceitos atende a ~7·K posições (barato), e o
  cross-attention do T5 fica bem menor que no `stride`.
- **Alinhamento forçado:** um Viterbi em lote, com laço sobre o tempo. Medido
  com lote de 32 e 30 glosas: ~20–30 ms com `--ds-steps 2` (T' ≈ 120) e
  ~65–110 ms sem downsampling (T ≈ 480), em CPU e GPU. É pequeno perto de um
  passo do GermanT5.

## Compatibilidade

- Com `--temporal-mode stride` (o padrão), o modelo é **bit a bit idêntico**
  ao de antes da flag: mesmas chaves, pesos, perda e gradientes, verificado
  para `flat`, `parts` e `gcn`.
- Checkpoints antigos carregam como `stride`. Um checkpoint `concept` com o
  config dizendo `stride` carrega como `concept`, com aviso (os pesos mandam).

## Limitações desta versão

- As fronteiras do greedy dependem de o CTC estar bem treinado. Nas primeiras
  épocas, a validação pode ver frases inteiras como 1 sinal. É esperado e
  some conforme o WER cai.
- Fronteiras conhecidas do Libras sintético (`given`, a partir dos offsets da
  montagem das frases) e a cabeça de "próximo conceito" ainda não foram
  implementadas.
- Não há avaliação com fronteiras **oráculo** (vindas das glosas) no teste;
  seria a medida do teto do método.

## Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `--temporal-mode concept requer --use-ctc` | CTC desligado | acrescente `--use-ctc` |
| `requer --ctc-weight > 0` | CTC sem treino → fronteiras aleatórias | use o peso da receita (1.0) |
| BLEU muito baixo nas primeiras épocas | greedy ainda não dispara | espere o WER cair; ou `--concept-greedy-prob 0` (padrão) |
| `temporal_mode='concept' exige seg_ids` | chamada direta a `model.encoder(...)` | use o `SLTModel`, que calcula as fronteiras |

## Testes

```powershell
& $PY tests\test_encoder_concept.py
```

Os 22 testes cobrem: disparos do greedy, Viterbi contra busca exaustiva,
glosas repetidas, frase sem alinhamento possível, pooling (isolamento entre
sinais e padding), gradiente em todos os módulos, avaliação/geração pelo
greedy, checkpoints e validação da CLI.
