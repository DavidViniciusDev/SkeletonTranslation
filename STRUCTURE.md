# Estrutura do projeto (após refatoração)

Pacote único `skeltrans/` em três camadas + shims de compatibilidade na raiz.

```
skeltrans/
├── common/                 # camada COMPARTILHADA (extração + treino)
│   ├── layout.py           # 115 pontos: N_POINTS, fatias, N_CHANNELS, INPUT_DIM, FACE_MASK
│   ├── text.py             # norm() (maiúsculas, sem acentos)
│   ├── geometry.py         # normalização geométrica + dinâmica (Passo 3) + máscaras
│   ├── npy_io.py           # load_points() + validação de shape (T,115,C)
│   └── parallel.py         # run_pool() (pool de processos + barra de progresso)
├── extraction/             # ETAPA 1
│   ├── config.py           # PATHS (árvore data/) + STEPS (ordem do pipeline)
│   ├── pipeline.py         # ORQUESTRADOR (--from/--to/--only/--list/--dry-run)
│   ├── frame_sources.py    # fontes de frames RGB: vídeo .mp4 OU diretório de imagens
│   └── steps/
│       ├── s1a_build_vocab.py
│       ├── s1b_filter_sentences.py
│       ├── s2_build_sentence_videos.py
│       ├── s3_extract_landmarks.py
│       ├── s4_normalize_landmarks.py
│       └── s5_build_sentence_features.py
└── training/               # ETAPA 2
    ├── config.py           # ModelConfig / DataConfig / TrainConfig
    ├── models/             # SinusoidalPositionalEncoding, LandmarkEncoder, SLTModel
    ├── data/               # LandmarkTextDataset, make_collate
    ├── device.py · checkpoint.py · trainer.py · diagnostics.py · cli.py
    ├── metrics.py          # BLEU-1..4 (sacrebleu) + METEOR (nltk)
    ├── evaluate.py         # avaliação no teste (métricas + predições)  → EVALUATION.md
    ├── infer.py            # inferência vídeo .mp4 → texto pt-br         → INFERENCE.md
    └── ...

data/                       # DADOS (separados do código — E8)
├── raw/                    # entradas externas + landmarks_115 (extraídos)
├── interim/                # vocab, sentenças filtradas, sentence_videos, train/val
└── features/               # landmarks normalizados + features de frase + manifesto

# Shims na raiz (preservam os comandos originais):
build_vocab.py · filter_sentences_by_vocab.py · build_sentence_videos.py
extract_landmarks.py · normalize_landmarks.py · build_sentence_features.py
slt_model.py · split_manifest.py · dataset_stats.py
evaluate_slt.py · infer_slt.py
```

## Como rodar

### Etapa 1 — extração (um comando)

```bash
# pipeline inteiro (s1a → s5), lendo/gravando em data/
python -m skeltrans.extraction.pipeline

# apenas um trecho, ou um passo com argumentos próprios
python -m skeltrans.extraction.pipeline --from s3 --to s4
python -m skeltrans.extraction.pipeline --only s3 -- --workers 8
python -m skeltrans.extraction.pipeline --list        # lista os passos
python -m skeltrans.extraction.pipeline --dry-run     # mostra os comandos
```

Cada passo também roda isolado, pelo shim (`python3 extract_landmarks.py ...`)
ou por módulo (`python -m skeltrans.extraction.steps.s3_extract_landmarks ...`).

### Etapa 2 — modelo / treino

```bash
python3 slt_model.py --smoke-test                     # valida a arquitetura (CPU)
python3 slt_model.py \
    --train-manifest data/features/sentence_features/manifest.json \
    --val-manifest ... --epochs 30 --out-dir checkpoints
```

### Etapa 2 — avaliação e inferência

```bash
# métricas (BLEU-1..4 + METEOR) no conjunto de teste — ver EVALUATION.md
python3 evaluate_slt.py --checkpoint checkpoints/best.pt \
    --test-manifest data/interim/test.json \
    --features-dir data/features/sentence_features --out checkpoints/test_metrics.json

# traduzir um vídeo .mp4 ponta-a-ponta — ver INFERENCE.md
python3 infer_slt.py --checkpoint checkpoints/best.pt --video frase.mp4
```

## Caminhos

Todos os caminhos default vivem em `skeltrans/extraction/config.py` (`DATA_ROOT`
+ dataclass `Paths`). Para usar outro layout, ajuste `DATA_ROOT` ou passe os
caminhos por CLI em cada passo.

## Testes de regressão

```bash
python3 tests/test_geometry_regression.py   # normalização == referência SkeLTrans/
python3 slt_model.py --smoke-test --device cpu
```
