# Avaliação — métricas de tradução no conjunto de teste

Como medir a qualidade de um modelo SLT **já treinado** sobre um conjunto de
teste, gerando as traduções e calculando métricas de tradução automática.

> Etapa 2 do projeto (modelo). Corresponde ao **Passo 6** do
> [`plan.md`](plan.md).

---

## 1. O que a avaliação faz

`evaluate_slt.py`:

1. Carrega o checkpoint treinado (a arquitetura é **reconstruída
   automaticamente** a partir dos hiperparâmetros gravados em `ckpt['args']`).
2. Carrega o tokenizer (do diretório do checkpoint, onde o treino o salva).
3. Percorre o manifesto de teste, gera a tradução de cada frase (beam search) e
   coleta hipóteses (`hyp`) × referências (`ref`).
4. Calcula e imprime as métricas; opcionalmente grava um JSON com métricas **e**
   todas as predições.

O código real vive em [`skeltrans/training/evaluate.py`](skeltrans/training/evaluate.py)
(métricas em [`skeltrans/training/metrics.py`](skeltrans/training/metrics.py));
`evaluate_slt.py` na raiz é apenas um *shim*.

---

## 2. Métricas

| Métrica       | Fonte      | Escala | Observação |
|---------------|------------|--------|------------|
| BLEU-1..4     | `sacrebleu`| 0–100  | Cumulativos, com `effective_order` (robusto para frases curtas). |
| METEOR        | `nltk`     | 0–1    | Média no corpus; tokenização simples (minúsculas + palavras). |

### Dependências de métrica

```bash
pip install sacrebleu nltk
# corpora do METEOR (uma vez):
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 3. Pré-requisitos: o manifesto de teste

A avaliação precisa de um `test.json` no mesmo formato dos manifestos de treino
(Etapa 1, s5). O protocolo do projeto é **signer-independent** (intérpretes de
teste disjuntos dos de treino). Gere os *splits* com `split_manifest.py`:

```bash
python3 split_manifest.py \
    --manifest data/features/sentence_features/manifest.json \
    --out-dir data/interim \
    --train-interpreters 0,1 --test-interpreters 2 \
    --val-ratio 0.1 --seed 42
```

Isso produz `train.json`, `val.json` e `test.json`. Detalhes do protocolo (e a
opção `--test-exclude-train-sentences`, teste disjunto também em texto) estão no
cabeçalho de [`split_manifest.py`](split_manifest.py).

---

## 4. Uso

```bash
python3 evaluate_slt.py \
    --checkpoint checkpoints/best.pt \
    --test-manifest data/interim/test.json \
    --features-dir data/features/sentence_features \
    --out checkpoints/test_metrics.json
```

Equivalente, pelo módulo (sem o shim):

```bash
python -m skeltrans.training.evaluate \
    --checkpoint checkpoints/best.pt \
    --test-manifest data/interim/test.json \
    --features-dir data/features/sentence_features
```

### Saída esperada

```
Dispositivo: cuda
Checkpoint: checkpoints/best.pt | T5: unicamp-dl/ptt5-base-portuguese-vocab | epoca: 27
Teste: 512 exemplos (data/interim/test.json)
  ... 160/512 traduzidos
  ...
== Metricas (conjunto de teste) ==
  BLEU-1  : 41.2
  BLEU-2  : 28.7
  BLEU-3  : 20.5
  BLEU-4  : 15.1
  METEOR  : 0.3820

Predicoes + metricas salvas em: checkpoints/test_metrics.json
```

> Os números acima são **ilustrativos** — servem só para mostrar o formato.

---

## 5. Argumentos

| Argumento          | Padrão        | Descrição |
|--------------------|---------------|-----------|
| `--checkpoint`     | (obrigatório) | Caminho do `.pt` treinado (ex.: `best.pt`). |
| `--test-manifest`  | (obrigatório) | Manifesto de teste (`test.json`). |
| `--features-dir`   | (auto)        | Pasta dos `.npy` de features. Se informado, cada feature é buscada como `<features-dir>/<basename>`; senão, os caminhos relativos do manifesto são resolvidos em relação a ele. |
| `--batch-size`     | `8`           | Frases por lote na geração. |
| `--num-beams`      | `4`           | Largura do *beam search*. |
| `--max-new-tokens` | `64`          | Máximo de tokens gerados por frase. |
| `--num-workers`    | `2`           | Processos do DataLoader. |
| `--t5`             | (auto)        | Override do identificador/checkpoint do T5. |
| `--tokenizer`      | (auto)        | Override do tokenizer (padrão: dir do checkpoint, senão o T5). |
| `--device`         | (auto)        | `cuda` ou `cpu`. |
| `--out`            | (nenhum)      | JSON de saída com métricas + predições. |

---

## 6. O arquivo de saída (`--out`)

O JSON gravado permite inspecionar as traduções uma a uma (análise de erros) e
reproduzir os números:

```json
{
  "checkpoint": "/.../checkpoints/best.pt",
  "test_manifest": "/.../data/interim/test.json",
  "num_examples": 512,
  "gen": { "num_beams": 4, "max_new_tokens": 64 },
  "metrics": { "BLEU-1": 41.2, "BLEU-2": 28.7, "BLEU-3": 20.5, "BLEU-4": 15.1, "METEOR": 0.382 },
  "predictions": [
    { "ref": "o menino gosta de jogar bola", "hyp": "o menino gosta de bola" },
    ...
  ]
}
```

---

## 7. Boas práticas de comparação

- **Fixe a geração** ao comparar modelos: use os mesmos `--num-beams` e
  `--max-new-tokens` entre todas as execuções (eles são registrados em
  `gen` no JSON de saída).
- Avalie sempre o **mesmo `test.json`** (mesmo `--seed` no split) para que os
  números sejam comparáveis.
- Para *ablation studies*, salve cada execução com um `--out` distinto
  (ex.: `checkpoints/test_metrics_baseline.json`) e compare os JSONs.

---

## 8. Solução de problemas

| Sintoma | Causa provável / correção |
|---------|---------------------------|
| `ModuleNotFoundError: sacrebleu` / `nltk` | Instale as dependências de métrica (seção 2). |
| Erro do METEOR sobre `wordnet` | Baixe os corpora: `nltk.download('wordnet'); nltk.download('omw-1.4')`. |
| `KeyError` sobre chaves do manifesto | O `test.json` deve ter itens com `feature_file`/`features` e `pt_br`/`text` (ver `LandmarkTextDataset`). |
| `.npy` não encontrado | Passe `--features-dir` apontando para a pasta correta das features. |
| Métricas muito baixas | Confira que o checkpoint é o `best.pt` e que o `test.json` foi gerado com a mesma normalização usada no treino. |
