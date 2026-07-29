# INSTRUCTIONS — Como usar o SkeletonTranslation

Guia prático de ponta a ponta: do vídeo bruto em Libras à tradução em português.
O projeto tem **duas etapas** independentes:

1. **Extração** (`skeltrans.extraction`) — transforma o corpus e os vídeos em
   *features* de frase prontas para treino.
2. **Treino/Modelo** (`skeltrans.training`) — treina o Encoder-Decoder
   (LandmarkEncoder + PTT5) que traduz as features em texto.

> Para a visão geral da arquitetura de pastas, veja [STRUCTURE.md](STRUCTURE.md).

---

## 1. Pré-requisitos

- **Python 3.10+** (testado em 3.12).
- Dependências: `torch`, `transformers`, `mediapipe`, `opencv-python`,
  `numpy`, `scipy`.

### Ambiente

Todos os comandos assumem que você está **na raiz do projeto**
(`Doutorado/SkeletonTranslation/`), pois é lá que vive o pacote `skeltrans/` e a
árvore `data/`.

```bash
cd Doutorado/SkeletonTranslation

# criar um ambiente virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers mediapipe opencv-python numpy scipy
```

> Já existe um `.venv` pronto na pasta irmã `../SkeLTrans/.venv`. Se quiser
> reaproveitá-lo sem instalar nada, use o interpretador dele diretamente:
> `../SkeLTrans/.venv/bin/python -m skeltrans.extraction.pipeline ...`

### GPU / CPU

Se a GPU estiver ocupada ou incompatível, force CPU:

```bash
# no treino / smoke-test:
python3 slt_model.py --smoke-test --device cpu

# para esconder a GPU de qualquer passo:
CUDA_VISIBLE_DEVICES="" python3 slt_model.py --smoke-test
```

---

## 2. Layout dos dados (`data/`)

```
data/
├── raw/        entradas + landmarks extraídos
│   ├── pt_br2libras_gloss.csv            corpus texto→glosa (fonte)
│   ├── pt_br2libras_gloss.json           idem, em JSON
│   ├── videos_words_by_word.json         vocabulário de vídeos do V-LIBRASIL
│   └── landmarks_115/                    .npy (T,115,3) por sinal (saída do s3)
├── interim/    intermediários gerados
│   ├── libras_gloss_vocab.txt
│   ├── pt_br2libras_gloss_in_vocab.json
│   ├── pt_br2libras_gloss_sentence_videos.json
│   └── train.json · val.json             (splits, se gerados aqui)
└── features/   features finais
    ├── landmarks_115_norm9/              .npy (T,115,9) normalizados (s4)
    └── sentence_features/                features de frase + manifest.json (s5)
```

Os caminhos default vivem em [`skeltrans/extraction/config.py`](skeltrans/extraction/config.py)
(`DATA_ROOT` + dataclass `Paths`). Para outro layout, edite `DATA_ROOT` ou passe
os caminhos por CLI em cada passo.

---

## 3. Etapa 1 — Extração

### 3.1 Rodar o pipeline inteiro (um comando)

```bash
python -m skeltrans.extraction.pipeline
```

Executa, em ordem: **s1a → s1b → s2 → s3 → s4 → s5**. Cada passo roda em seu
próprio processo (isolando o multiprocessing do MediaPipe/normalização) e o
pipeline para no primeiro erro.

### 3.2 Controlar o trecho executado

```bash
python -m skeltrans.extraction.pipeline --list           # lista os passos e sai
python -m skeltrans.extraction.pipeline --dry-run         # mostra os comandos, sem executar
python -m skeltrans.extraction.pipeline --from s3         # do s3 até o fim
python -m skeltrans.extraction.pipeline --to s2           # do início até o s2
python -m skeltrans.extraction.pipeline --from s3 --to s4 # apenas s3 e s4
python -m skeltrans.extraction.pipeline --only s4         # somente um passo

# passar argumentos a UM passo (só com --only), após ' -- ':
python -m skeltrans.extraction.pipeline --only s3 -- --workers 8 --limit 100
```

### 3.3 Os passos, um a um

Cada passo também roda isolado — pelo **shim** na raiz (`python3 <nome>.py`) ou
por **módulo** (`python -m skeltrans.extraction.steps.<passo>`). Os defaults já
apontam para `data/`.

| # | Shim | Entrada → Saída | Principais opções |
|---|------|-----------------|-------------------|
| s1a | `build_vocab.py` | `raw/*.csv` → `interim/…vocab.txt` | `--csv --out` |
| s1b | `filter_sentences_by_vocab.py` | glosas + vocab → `interim/…in_vocab.json` | `--gloss-json --vocab-json --out` |
| s2  | `build_sentence_videos.py` | in_vocab → `interim/…sentence_videos.json` | `--gloss-json --vocab-json --out` |
| s3  | `extract_landmarks.py` | vídeos `.mp4` **ou** diretórios de frames → `raw/landmarks_115/*.npy` | `--workers --limit --video --overwrite --with-visibility` |
| s4  | `normalize_landmarks.py` | landmarks → `features/landmarks_115_norm9/` | `--workers --origin --file --overwrite` |
| s5  | `build_sentence_features.py` | vídeos+landmarks → `features/sentence_features/` + manifesto | `--limit --transition-frames --interp-mode --feature-mode --variants --anchor --origin` |

Exemplos úteis:

```bash
# s3: reprocessar um único vídeo (útil p/ depurar), 4 processos
python3 extract_landmarks.py --video /libras/v-librasil/videos_words_augmentation/0-casa.mp4 --workers 4

# s3: extrair de um diretório de frames (ex.: PHOENIX-2014T) em vez de um vídeo
python3 extract_landmarks.py --video /dados/phoenix/features/fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/ --workers 4

# s4: normalizar tudo com origem no nariz em vez dos ombros
python3 normalize_landmarks.py --origin nose

# s5: gerar só as 20 primeiras sentenças, colagem "seca" (ablation, sem interpolação)
python3 build_sentence_features.py --limit 20 --interp-mode none
```

> **Dica de retomada (resume):** s3, s4 e s5 pulam arquivos que já existem na
> saída. Use `--overwrite` para reprocessar.

### 3.4 Saída da extração

Ao fim do s5 você terá, em `data/features/sentence_features/`:
- um `.npy` por sentença/variante (a sequência contínua de features),
- `manifest.json` (mapa feature → texto-alvo + metadados),
- `manifest.csv` e `manifest.stats.json`.

O `manifest.json` é a entrada do treino.

---

## 4. Dividir em treino / validação / teste

`split_manifest.py` faz um split **signer-independent** (intérprete held-out):

```bash
python3 split_manifest.py \
    --manifest data/features/sentence_features/manifest.json \
    --out-dir data/interim \
    --train-interpreters 0,1 --test-interpreters 2 \
    --val-ratio 0.1 --seed 42
```

- **train.json** — intérpretes de treino, *todas* as augmentations.
- **val.json** — intérpretes de treino, só a base (`none`); separado do treino por `sentence_id`.
- **test.json** — intérprete(s) de teste, só a base; nunca visto no treino.
- `--test-exclude-train-sentences` torna o teste disjunto também em texto.

---

## 5. Etapa 2 — Treino do modelo

### 5.1 Validar a arquitetura (sem dados, sem download)

```bash
python3 slt_model.py --smoke-test --device cpu
```

Monta um T5 minúsculo e roda forward/backward/generate com dados aleatórios.
Serve para conferir a instalação e a integração encoder↔decoder em segundos.

### 5.2 Treino real

```bash
python3 slt_model.py \
    --train-manifest data/interim/train.json \
    --val-manifest   data/interim/val.json \
    --epochs 30 --batch-size 8 --lr 3e-5 \
    --out-dir checkpoints
```

Salva `checkpoints/epochN.pt` a cada época e `checkpoints/best.pt` quando a
perda de validação melhora (junto com o tokenizer).

### 5.3 Opções principais do treino

| Grupo | Flags |
|-------|-------|
| Dados | `--train-manifest --val-manifest --features-dir --max-text-len` |
| Modelo | `--t5 --d-model --nhead --num-layers --dropout` |
| Otimização | `--epochs --batch-size --lr --warmup-steps --grad-clip` |
| Execução | `--num-workers --log-every --out-dir --device` |

Veja a ajuda completa (com explicação de cada flag):

```bash
python3 slt_model.py --help
```

> **`--features-dir`**: se o `manifest.json` e os `.npy` estiverem em pastas
> diferentes, aponte aqui a pasta dos `.npy` (cada feature é buscada por
> *basename*).

---

## 6. Estatísticas e testes

```bash
# estatísticas do corpus (nº de sentenças, vocabulário, durações, ...)
python3 dataset_stats.py                       # usa data/interim/…sentence_videos.json
python3 dataset_stats.py outro_arquivo.json

# teste de regressão da normalização (compara com a referência em ../SkeLTrans/)
python3 tests/test_geometry_regression.py
```

---

## 7. Fluxo completo, do zero (resumo)

```bash
cd Doutorado/SkeletonTranslation
source .venv/bin/activate                       # ou use ../SkeLTrans/.venv

# 1) extração completa (texto → features de frase)
python -m skeltrans.extraction.pipeline

# 2) split em train/val/test
python3 split_manifest.py --out-dir data/interim \
    --train-interpreters 0,1 --test-interpreters 2

# 3) treino
python3 slt_model.py \
    --train-manifest data/interim/train.json \
    --val-manifest   data/interim/val.json \
    --epochs 30 --out-dir checkpoints
```

---

## 8. Problemas comuns

| Sintoma | Causa provável / solução |
|---------|--------------------------|
| `CUDA error: out of memory` no smoke-test | GPU ocupada → use `--device cpu` ou `CUDA_VISIBLE_DEVICES=""`. |
| `ModuleNotFoundError: skeltrans` | Rode a partir da **raiz** do projeto (onde está a pasta `skeltrans/`). |
| Passo não encontra o arquivo de entrada | Confira se os dados estão em `data/` (ver seção 2) ou passe o caminho por CLI. |
| `mediapipe nao instalado` no s3 | `pip install mediapipe opencv-python`. |
| s3/s4/s5 "pulando" tudo | Saída já existe → use `--overwrite` para reprocessar. |
| Warning de *compute capability* do PyTorch | GPU antiga não suportada pelo build do torch; rode em CPU. |
