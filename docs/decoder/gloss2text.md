# Pré-treino do decoder em Gloss2Text

`pretrain_decoder.py` treina o T5 a traduzir **glosas → texto**, só com texto,
antes do treino de sinais. A ideia, os conceitos e o plano de experimentos
estão em [ideas/pretreino-decoder-gloss2text.md](../../ideas/pretreino-decoder-gloss2text.md).
Código em [`skeltrans/training/gloss2text.py`](../../skeltrans/training/gloss2text.py).

---

## 1. Por que

No SLT, o decoder aprende duas coisas ao mesmo tempo: **ler o encoder de
landmarks** e **traduzir a gramática de sinais para o alemão**. Até a v17, o
WER caiu a cada treino enquanto o BLEU estacionou: o gargalo é a tradução.

O pré-treino separa as duas tarefas:

```
Etapa 1 (pretrain_decoder.py):  [encoder de texto do T5] → [decoder]   glosas → alemão
                                                              │ reaproveitado
Etapa 2 (slt_model.py --t5):    [encoder de landmarks]   → [decoder]   vídeo → alemão
```

Na etapa 1, o decoder aprende a gramática da tradução com entradas limpas. Na
etapa 2, o encoder de texto do T5 não é usado (a memória vem do encoder de
landmarks); o decoder já chega sabendo traduzir do "espaço das glosas".

---

## 2. Etapa 1: o pré-treino

Use o **mesmo T5** do treino de sinais.

```bash
python3 pretrain_decoder.py \
    --train-manifest data_phoenix/interim/train.json \
    --val-manifest   data_phoenix/interim/dev.json \
    --test-manifest  data_phoenix/interim/test.json \
    --t5 GermanT5/t5-efficient-gc4-german-base-nl36 \
    --epochs 30 --batch-size 32 --lr 1e-4 --warmup-steps 200 \
    --label-smoothing 0.1 --eval-beams 4 --select-by bleu4 --patience 10 \
    --low-vram --seed 1 --out-dir checkpoints_gloss2text
```

É rápido: são frases curtas e sem features.

**Saída** (em `--out-dir`):

| Arquivo | Conteúdo |
|---|---|
| `best/` | o modelo da melhor época na **validação**, no formato do Hugging Face (pesos + tokenizer). É o que vai no `--t5` da etapa 2. |
| `teto.json` | BLEU/METEOR no **teste** com as glosas verdadeiras, e as traduções frase a frase (só com `--test-manifest`). |
| `gloss2text_history.jsonl` | perda e métricas da validação por época. |
| `gloss2text_config.json` | a configuração e o comando usados. |

### O teto do decoder

O BLEU do `teto.json` é o melhor que esse decoder traduz se recebesse glosas
**perfeitas**:

- **Teto bem acima do BLEU do SLT (19,22 na v15):** há espaço do lado da
  tradução; o pré-treino (etapa 2) é o próximo passo.
- **Teto perto do BLEU do SLT, ou bem abaixo de 30:** a limitação é o próprio
  decoder; a meta de 30 exige um decoder maior ou diferente.

O teste não participa de nenhuma escolha (a época vem da validação). Ele só
mede o teto, uma vez, no fim.

---

## 3. Etapa 2: o SLT a partir do decoder pré-treinado

Só troque o `--t5` pela pasta `best/`; o resto é a receita de sempre:

```bash
python3 slt_model.py "${BASE[@]}" --t5 checkpoints_gloss2text/best \
    --out-dir checkpoints_phoenix_g2t
```

Não há flag nova no `slt_model.py`: ele carrega a pasta como qualquer T5.
`evaluate_slt.py` e `infer_slt.py` também não mudam; o checkpoint guarda o
caminho do T5.

- **Mantenha o lr do decoder baixo** (`--lr 5e-5`, como na v15, com
  `--encoder-lr 3e-4`). Um lr alto pode apagar o que o decoder aprendeu na etapa 1.
- **Não apague nem mova** a pasta `best/` enquanto usar os checkpoints da
  etapa 2: a avaliação e a inferência carregam o T5 dela.
- Combina com o pré-treino do encoder: `--t5 checkpoints_gloss2text/best
  --init-encoder checkpoints_pretrain/encoder.pt`.

---

## 4. Opções das glosas

| Flag | O que faz |
|---|---|
| `--gloss-noise P` | corrompe cada glosa do **treino** com probabilidade `P`, imitando os erros do encoder (ex.: `0.35` ≈ WER de 35%). Validação e teste ficam limpos. Treina o decoder a traduzir mesmo com glosas imperfeitas, que é a situação real da etapa 2. |
| `--noise-mix S,D,I` | pesos de substituição, deleção e inserção no ruído. Padrão `0.6,0.3,0.1`, a decomposição do WER da v15/v17. |
| `--lowercase-glosses` | glosas em minúsculas (aproxima as subpalavras do texto comum). |
| `--drop-markers` | remove marcações sem conteúdo lexical no formato `__X__` (ex.: `__ON__`). |

O ruído é sorteado de novo a cada lote, então cada época vê erros diferentes.
Com `--num-workers 0` (o padrão) e `--seed`, ele é reproduzível.

---

## 5. Plano de experimentos

Cada linha muda uma coisa; compare BLEU-4, METEOR e WER com a v15 (mesma
semente).

| # | Etapa 1 | Etapa 2 | Pergunta |
|---|---|---|---|
| 1 | glosas limpas | — | Qual é o teto do decoder (`teto.json`)? |
| 2 | glosas limpas | v15 com `--t5 .../best` | O pré-treino melhora o BLEU? |
| 3 | `--gloss-noise 0.35` | v15 com `--t5 .../best` | O ruído deixa o decoder mais robusto? |
| 4 | `--lowercase-glosses` (+ `--drop-markers`) | v15 com `--t5 .../best` | A tokenização das glosas importa? |

Registre os resultados no [TIMELINE.md](../../TIMELINE.md) (seção 2).

---

## 6. Testes

```bash
python3 pretrain_decoder.py --smoke-test     # CPU, sem dados nem download
python3 tests/test_pretrain_decoder.py       # 11 testes
```

Os testes cobrem: normalização e ruído das glosas, leitura dos manifestos,
lotes e rótulos, label smoothing, um treino que **aprende de verdade** (o T5
minúsculo decora um corpus sintético até BLEU-4 100), o `teto.json` e o uso do
`best/` como `--t5` do SLT, com os pesos do decoder carregados exatamente.
