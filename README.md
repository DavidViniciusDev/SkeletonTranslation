# SkeLTrans — Pipeline de Dados para SLT baseado em Landmarks Sintéticos

Este repositório documenta e implementa as etapas de **engenharia de dados** da
pesquisa descrita em [`plan.md`](plan.md): usar landmarks esqueléticos de sinais
isolados do **V-LIBRASIL** para sintetizar dados de treino de um modelo de
**Sign Language Translation (SLT)** — tradução de Libras contínua para Português.

O objetivo deste README é permitir que **outra pessoa reproduza** o pipeline do
zero. Ele cobre os **Passos 1 a 5** do plano (do texto/vídeo bruto até o
treinamento do modelo de tradução). O **Passo 6** (protocolo de avaliação e
*ablation study*) ainda não está implementado.

---

## 1. Visão geral do pipeline

O fluxo transforma dois insumos brutos (um corpus de glosas e um banco de vídeos
de sinais isolados) em matrizes de características, sintetiza frases contínuas e
treina o modelo de tradução:

```
  pt_br2libras_gloss.csv            videos_words_by_word.json
  (127k sentenças pt→glosa)         (vocabulário de vídeos: 1.300 sinais)
         │                                   │
         ▼  [Etapa A]                        │
  libras_gloss_vocab.txt                     │
  (vocabulário das glosas)                   │
                                             │
         └──────────────┬────────────────────┘
                        ▼  [Etapa B]
        pt_br2libras_gloss_in_vocab.json
        (sentenças 100% cobertas pelo vocabulário + expansão de "&")
                        │
                        ▼  [Etapa C]
        pt_br2libras_gloss_sentence_videos.json
        (cada sentença → sequências de vídeos a concatenar)
                        │
                        ▼  [Etapa D]  MediaPipe Holistic
              landmarks_115/*.npy
              (por vídeo isolado: T × 115 × 3  →  X,Y,Z)
                        │
                        ▼  [Etapa E]  Normalização geométrica + dinâmica
            landmarks_115_norm9/*.npy
            (por vídeo isolado: T × 115 × 9)
                        │
                        ▼  [Etapa F]  Síntese de frases (Keyframe Blending)
            sentence_features/*.npy + manifest.json
            (por FRASE inteira: T × 115 × 9, sinais colados + transições suaves)
                        │
                        ▼  [Etapa G]  Modelo SLT (treino)
            checkpoints/best.pt
            (LandmarkEncoder + decoder PTT5  →  texto em Português)
```

Correspondência com o `plan.md`:

| Etapa deste README | Passo do plano | Script |
|---|---|---|
| A. Vocabulário das glosas | Passo 1 (sanitização/indexação) | `build_vocab.py` |
| B. Filtragem por cobertura + expansão `&` | Passo 1 / pré-Passo 4 | `filter_sentences_by_vocab.py` |
| C. Mapeamento sentença → vídeos | Passo 1.3 (indexação) | `build_sentence_videos.py` |
| D. Extração de landmarks | Passo 2 | `extract_landmarks.py` |
| E. Normalização geométrica | Passo 3 | `normalize_landmarks.py` |
| F. Síntese de frases (Keyframe Blending) | Passo 4 | `build_sentence_features.py` |
| G. Modelo SLT + treino | Passo 5 | `slt_model.py` |
| — Avaliação (BLEU/ROUGE/chrF) + ablação | Passo 6 | *(não implementado)* |

---

## 2. Pré-requisitos

### 2.1. Dados de entrada (não versionados)

| Arquivo / pasta | Descrição |
|---|---|
| `pt_br2libras_gloss.csv` | Corpus paralelo pt-br → glosa (colunas: `pt-br`, `libras-gloss`, `is_government_source`, `english_translation`). |
| `pt_br2libras_gloss.json` | Mesmo corpus em JSON (lista de objetos). |
| `videos_words_by_word.json` | Vocabulário de vídeos: `{ "PALAVRA": [ {video, begin, end, duration, word, word_english}, ... ] }`. |
| `/libras/v-librasil/videos_words_augmentation/` | Vídeos `.mp4` dos sinais com *data augmentation*. |
| `/libras/v-librasil/videos_words/` e `regular_videos_words/` | Vídeos **originais** (sem augmentation) — necessários para a *base* de teste (ver Etapa D). |

### 2.2. Ambiente Python

Python **3.12** (testado em 3.12.3). Como o sistema é "externally managed",
usamos um **virtualenv** local:

```bash
cd /libras/Doutorado/SkeLTrans
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# Etapas A–F (dados/landmarks/síntese)
.venv/bin/pip install "mediapipe==0.10.14" opencv-python-headless numpy scipy

# Etapa G (modelo/treino) — usa a GPU
.venv/bin/pip install torch transformers sentencepiece sacrebleu evaluate accelerate
```

> **Por que `mediapipe==0.10.14`?** As versões novas (0.10.30+) **removeram** a
> API legada `mediapipe.solutions.holistic` (com `model_complexity` e
> `refine_face_landmarks`), exigida pelo plano. A 0.10.14 ainda a expõe e tem
> wheel para Python 3.12.

Versões usadas na referência: `mediapipe 0.10.14`, `opencv 4.13`, `numpy 2.5`,
`scipy 1.18`, `torch 2.12 (CUDA)`, `transformers 5.13`.

> Todos os comandos abaixo assumem `.venv/bin/python`. Máquina de referência:
> 2× NVIDIA TITAN V (12 GB). O MediaPipe Holistic legado roda em **CPU/TFLite**
> (Etapa D não usa GPU); o treino do modelo (Etapa G) usa **GPU/CUDA**.

---

## 3. Etapas do pipeline

### Etapa A — Vocabulário das glosas

**Script:** [`build_vocab.py`](build_vocab.py)
**Objetivo:** levantar o tamanho e a lista de "palavras" (tokens) da coluna
`libras-gloss`.

```bash
.venv/bin/python build_vocab.py
```

**Entrada:** `pt_br2libras_gloss.csv`
**Saída:** `libras_gloss_vocab.txt` (uma linha `palavra<TAB>frequência`, ordenado).

**Tokenização:** cada token separado por espaço é uma entrada. Construções
especiais são **preservadas** como um único token:
- `[PONTO]`, `[INTERROGAÇÃO]` → marcadores de pontuação;
- `A&B` (ex.: `BRASIL&PAÍS`) → sinal composto/desambiguação;
- `A_B` (ex.: `NÃO_TER`) → sinal multi-palavra.

**Resultado de referência:** ~**55.630** tokens únicos, ~1,07 milhão de tokens
no total (127.349 sentenças).

---

### Etapa B — Filtragem por cobertura de vocabulário + expansão de `&`

**Script:** [`filter_sentences_by_vocab.py`](filter_sentences_by_vocab.py)
**Objetivo:** manter apenas as sentenças em que **todas** as palavras existem no
vocabulário de vídeos, e **aumentar** o número de sentenças separando os tokens
compostos com `&`.

```bash
.venv/bin/python filter_sentences_by_vocab.py
```

**Entrada:** `pt_br2libras_gloss.json` + `videos_words_by_word.json`
**Saída:** `pt_br2libras_gloss_in_vocab.json`

**Regras de correspondência (importante para reprodutibilidade):**
1. **Normalização:** MAIÚSCULAS + remoção de acentos (o vocabulário de vídeos é
   ASCII/maiúsculo). Ex.: `CIDADÃO` → `CIDADAO`.
2. **Pontuação** (`[...]`): ignorada — não precisa existir no vocabulário.
3. **Compostos `&`:** divididos; cada parte precisa existir no vocabulário.
4. **Multi-palavra `_`:** o `_` vira espaço e casa com entradas multi-palavra do
   vocabulário; se não casar, tenta o token sem o `_`.
5. Uma sentença só é mantida se tiver ≥1 sinal real e **todos** no vocabulário.

**Expansão de `&`:** para cada sentença mantida, gera-se o **produto cartesiano**
das escolhas dos tokens compostos. Ex.: `DOAR&OBJETO` vira duas variantes
(`DOAR` e `OBJETO`); duas ocorrências de `&` geram 4 variantes.

**Resultado de referência:** 2.650 sentenças cobertas → **3.019** após expansão.

---

### Etapa C — Mapeamento sentença → sequências de vídeos

**Script:** [`build_sentence_videos.py`](build_sentence_videos.py)
**Objetivo:** para cada sentença, montar as **sequências de vídeos** (um vídeo por
palavra, na ordem da glosa) que, concatenadas, formam o vídeo da frase.

```bash
.venv/bin/python build_sentence_videos.py
```

**Entrada:** `pt_br2libras_gloss_in_vocab.json` + `videos_words_by_word.json`
**Saída:** `pt_br2libras_gloss_sentence_videos.json`

**Padrão dos nomes de vídeo:** `{intérprete}-{palavra}{-augmentation}.mp4`
- **intérprete:** número inicial (`0`, `1`, `2`) — cada número é um sinalizador;
- **augmentation:** `horizontal-flip`, `upsample`, `downsample` e combinações,
  ou `none`.

**Regra de consistência (crucial):** uma sequência usa o **mesmo intérprete** e a
**mesma augmentation** em **todas** as palavras (`0` só cola com `0`; `flip` só
com `flip`). Só entram combinações que existem em **todas** as palavras da
sentença (interseção) — evita sequência quebrada.

**Estrutura de cada registro:**
- `base`: intérprete `0` **sem** augmentation → reservado para a **etapa de teste**;
- `sentences`: todas as demais combinações consistentes (treino/augmentation),
  **excluindo** a base;
- cada sequência tem `interpreter`, `augmentation`, `num_videos`,
  `total_duration` e `videos[]` (com `gloss`, `word`, `video`, `begin`, `end`,
  `duration`) na ordem da glosa. `[PONTO]`/`[INTERROGAÇÃO]` são ignorados.

**Resultado de referência:** 3.019 registros; **50.957** sequências augmentadas
em `sentences` (todos com base completa).

---

### Etapa D — Extração de landmarks (MediaPipe Holistic)

**Script:** [`extract_landmarks.py`](extract_landmarks.py)
**Objetivo:** extrair o esqueleto de **cada item individual** frame a frame.

```bash
# processa todos os vídeos únicos referenciados, 8 processos, com resume
.venv/bin/python extract_landmarks.py

# opções úteis
.venv/bin/python extract_landmarks.py --workers 4        # nº de processos
.venv/bin/python extract_landmarks.py --limit 10         # só os 10 primeiros
.venv/bin/python extract_landmarks.py --video X.mp4      # um único vídeo
.venv/bin/python extract_landmarks.py --video frames/    # ou um diretório de frames
.venv/bin/python extract_landmarks.py --overwrite        # reprocessa existentes
.venv/bin/python extract_landmarks.py --with-visibility  # salva visibilidade da pose
```

**Entrada:** `pt_br2libras_gloss_sentence_videos.json` (coleta os **10.404 vídeos
únicos** referenciados) + os `.mp4`.
**Saída:** `landmarks_115/<nome-do-vídeo>.npy`, shape **`(T, 115, 3)`** (float32).

> **Dois modos de entrada.** Cada item pode ser um **arquivo de vídeo**
> (`.mp4`, `.mov`, …) — o caso do V-LIBRASIL — **ou** um **diretório de frames**
> (imagens `.png`/`.jpg`… ordenadas por nome), o formato de bases como a
> **PHOENIX-2014T**. A fonte é escolhida automaticamente pelo tipo do caminho
> (`skeltrans/extraction/frame_sources.py::open_frames`): diretório → sequência
> de imagens; arquivo → vídeo. O restante do passo (MediaPipe, os 115 pontos,
> zero-fill, resume, paralelismo) é **idêntico** nos dois modos. Para plugar uma
> base nova baseada em outro container (shards, HDF5, …), basta acrescentar um
> gerador em `frame_sources.py` — o s3 não muda.

**Configuração do MediaPipe:** `mp.solutions.holistic.Holistic` com
`model_complexity=2`, `refine_face_landmarks=True`,
`min_detection_confidence=0.5`, `min_tracking_confidence=0.5`.

**Os 115 pontos por frame (X, Y, Z):**

| Faixa de índice | Componente | Pontos |
|---|---|---|
| `[0:33)` | Pose | 33 |
| `[33:54)` | Mão esquerda | 21 |
| `[54:75)` | Mão direita | 21 |
| `[75:115)` | Face (máscara) | 40 |

**Máscara facial (40 pontos):** índices fixos do FaceMesh focados em
**lábios (20) + olhos (10) + sobrancelhas (10)** — as expressões não-manuais.
Os índices exatos estão em `FACE_MASK` no script.

**Decisões de implementação (importantes para reproduzir igual):**
- **Zero-fill:** frames/componentes não detectados (ex.: mãos ocultas) são
  preenchidos com `[0.0, 0.0, 0.0]`, evitando artefatos de congelamento. **O zero
  passa a significar "ausente"** — usado na Etapa E.
- **Visibilidade da pose:** o Holistic fornece um 4º valor (`visibility`) por
  ponto de pose, mas o shape exigido é `(T,115,3)`; por isso guardamos só
  `X,Y,Z`. Com `--with-visibility`, a visibilidade vai para um arquivo paralelo
  `*_pose_vis.npy` de shape `(T,33)`.
- **Fallback de diretório:** os vídeos `none` (a *base*, ex.: `0-qualquer.mp4`)
  **não existem** em `videos_words_augmentation/` — só as variantes augmentadas.
  Os 1.734 originais estão em `videos_words/` (ou `regular_videos_words/`). O
  script tenta esses diretórios automaticamente (`FALLBACK_DIRS`).
- **Paralelismo:** pool de **processos** (padrão 8), não threads — o Holistic não
  é thread-safe e é CPU-bound. Cada processo cria sua própria instância. Usa o
  start method `spawn` (seguro com libs nativas) e exibe **barra de progresso**
  com ETA. O `resume` pula `.npy` já existentes.

> ⏱️ São ~10 mil vídeos com o modelo "heavy" em CPU: espere **várias horas**.
> O `resume` permite interromper e retomar sem reprocessar.

---

### Etapa E — Normalização geométrica + features temporais (Passo 3)

**Script:** [`normalize_landmarks.py`](normalize_landmarks.py)
**Objetivo:** tornar os landmarks invariantes à escala/posição da câmera e
adicionar a **dinâmica** (velocidade e aceleração). Opera **direto nos `.npy`**,
sem reprocessar vídeo.

```bash
.venv/bin/python normalize_landmarks.py                 # landmarks_115 -> landmarks_115_norm9
.venv/bin/python normalize_landmarks.py --workers 8     # nº de processos (padrão: 8)
.venv/bin/python normalize_landmarks.py --origin nose   # origem = nariz (padrão: ombros)
.venv/bin/python normalize_landmarks.py --file X.npy --out-dir /tmp/out
.venv/bin/python normalize_landmarks.py --overwrite
```

**Entrada:** `landmarks_115/*.npy` `(T,115,3)`
**Saída:** `landmarks_115_norm9/*.npy` **`(T, 115, 9)`** — canais
`[X, Y, Z, Vx, Vy, Vz, Ax, Ay, Az]`.

> **Paralelismo:** pool de processos (padrão 8, via `--workers`) com barra de
> progresso e `resume`. Como esta etapa é numpy puro e muito I/O de disco, o
> ganho satura rápido — o gargalo tende a ser o disco, não a CPU.

**Transformações:**
1. **Centralização:** origem = ponto médio dos ombros (pose `11`/`12`; ou nariz
   `0` via `--origin nose`), subtraída de todos os pontos por frame.
2. **Escalonamento:** divide tudo pela distância euclidiana ombro-ombro
   (invariância de escala).
3. **Dinâmica:** velocidade `V_t = P_t − P_{t-1}` e aceleração `A_t = V_t − V_{t-1}`.
4. **Concatenação** → 9 canais por ponto.

**Por que a dinâmica?** Em Libras, o *movimento* é discriminativo (dois sinais
podem ter a mesma configuração de mão e diferir só na qualidade do movimento).
Entregar velocidade/aceleração prontas ajuda o modelo a convergir com poucos
dados e torna mensurável o efeito da interpolação (*Keyframe Blending*, Passo 4).

**Tratamento dos pontos ausentes (não pule isto):**
- A **máscara de "presente"** é calculada na entrada crua (antes de normalizar) e
  reimposta no fim: pontos ausentes continuam `[0,0,0]` na posição — senão a
  centralização os "ressuscitaria".
- A **velocidade** só é válida quando o ponto está presente em `t` **e** `t-1`
  (aceleração exige `t`, `t-1`, `t-2`); caso contrário é zerada — evita "movimento
  fantasma" nas bordas dos zeros.
- **Guard de divisão por zero:** se a pose (ombros) estiver ausente num frame,
  usa-se a última escala válida (ou `1.0`).

---

### Etapa F — Síntese de frases: *Keyframe Blending* (Passo 4)

**Script:** [`build_sentence_features.py`](build_sentence_features.py)
**Objetivo:** para cada frase, **colar os sinais isolados** numa sequência
contínua, inserindo uma **janela de transição suave** entre sinais consecutivos
(o núcleo de inovação da pesquisa). Cada frase vira **um** `.npy`.

```bash
.venv/bin/python build_sentence_features.py \
    --sentences-json pt_br2libras_gloss_sentence_videos.json \
    --landmarks-dir landmarks_115 \
    --out-dir sentence_features \
    --manifest sentence_features/manifest.json

# controle do ablation study (colagem "seca", sem interpolação):
.venv/bin/python build_sentence_features.py --interp-mode none --out-dir sentence_features_dry

# teste rápido em poucas frases:
.venv/bin/python build_sentence_features.py --limit 20
```

**Entrada:** `pt_br2libras_gloss_sentence_videos.json` (Etapa C) + os `.npy` de
sinais isolados (Etapa D, `landmarks_115/`).
**Saídas** (em `--out-dir`):
- `<id>.npy` — uma sequência contínua por frase/variante;
- `manifest.json` — `{config, items}` com o mapeamento feature → frase + metadados;
- `manifest.csv` — versão plana para o DataLoader;
- `manifest.stats.json` — **taxa de descarte** e contagens (para o paper).

**Parâmetros principais:**
- `--transition-frames K` (padrão **5**): tamanho da janela de transição.
- `--interp-mode {lerp,cubic,none}` (padrão `lerp`): interpolação **linear**,
  **spline cúbica** (`scipy`) ou **nenhuma** (colagem seca = controle da ablação).
- `--feature-mode {positions,normalized}` (padrão `normalized`): salva `(T,115,3)`
  ou aplica a **normalização do Passo 3 sobre a sequência já colada** → `(T,115,9)`.
- `--variants {all,base}`: gera todas as augmentations ou só a base limpa.

**Decisões importantes:**
- **Normalização depois da colagem:** no modo `normalized`, a normalização (Etapa E)
  é aplicada **sobre a frase contínua**, não sobre cada sinal isolado — assim as
  derivadas (velocidade/aceleração) atravessam corretamente as transições.
- **Ausentes na transição:** um ponto só é interpolado quando está presente **nos
  dois lados** da fronteira; senão permanece `[0,0,0]` (preserva a semântica de
  "ausente" e evita movimento fantasma).
- **`sentence_id`:** todas as variantes de augmentation de uma mesma frase
  compartilham o mesmo id — para que o split treino/val/teste (Passo 6) seja feito
  **por sentença**, sem vazamento de dados.

---

### Etapa G — Modelo SLT: Encoder de landmarks + decoder PTT5 (Passo 5)

**Script:** [`slt_model.py`](slt_model.py)
**Objetivo:** treinar o tradutor *End-to-End*: um **Encoder Espaço-Temporal** de
landmarks alimentando, via *cross-attention*, o **decoder pré-treinado do PTT5**
que gera o texto em Português.

```bash
# validação rápida da arquitetura (CPU, dados sintéticos, sem rede):
.venv/bin/python slt_model.py --smoke-test

# treino real (após a Etapa F):
.venv/bin/python slt_model.py \
    --train-manifest train.json --val-manifest val.json \
    --epochs 30 --batch-size 8 --out-dir checkpoints
```

**Entrada:** um manifesto JSON `[{"features": "<.npy>", "text": "<pt-br>"}]`, onde
cada `.npy` é a sequência `(T,115,9)` de uma **frase inteira** (Etapa F).
**Saída:** `checkpoints/epochN.pt` e `checkpoints/best.pt` + tokenizer salvo.

**Arquitetura:**
1. **Encoder Espaço-Temporal** (do zero): projeção linear `115×9=1035 → D_model`
   (512) + **Conv1D** temporal (`kernel=3`) + **positional encoding** senoidal +
   **6** `TransformerEncoderLayer` (8 cabeças, `dropout=0.2`).
2. **Decoder PTT5** (`unicamp-dl/ptt5-base-portuguese-vocab`): a memória do encoder
   entra no *cross-attention* via `encoder_outputs`; o T5 pula seu próprio encoder.
   Um **adaptador `Linear(512→768)`** casa o `D_model` do encoder com o hidden do
   PTT5 (768).
3. **Treino:** perda **CrossEntropy ignorando padding** (`labels` com `pad=-100`),
   **AdamW `lr=3e-5`** + *linear decay*, `dropout=0.2`, grad clipping, checkpoint
   por época e `best.pt`.

**É fine-tuning?** Sim, do decoder (transfer learning cross-modal); o encoder de
landmarks e o adaptador são treinados do zero. Ver discussão sobre riscos (taxa
única vs. diferenciada, congelar o decoder) na Seção 6.

> ⚠️ **Integração F → G (atenção ao reproduzir):** o manifesto da Etapa F usa o
> formato `{config, items:[{feature_file, pt_br, sentence_id, ...}]}`, enquanto o
> `slt_model.py` espera uma **lista plana** `[{features, text}]`. É preciso um
> passo de conversão (extrair `items`, renomear `feature_file→features` e
> `pt_br→text`) e o **split por `sentence_id`** (Passo 6) antes do treino. Esse
> conversor ainda não existe.

---

## 4. Ordem de reprodução (resumo)

```bash
cd /libras/Doutorado/SkeLTrans

# 0) ambiente
python3 -m venv .venv
.venv/bin/pip install "mediapipe==0.10.14" opencv-python-headless numpy scipy
.venv/bin/pip install torch transformers sentencepiece sacrebleu evaluate accelerate

# A) vocabulário das glosas
.venv/bin/python build_vocab.py

# B) filtra sentenças cobertas pelo vocabulário e expande "&"
.venv/bin/python filter_sentences_by_vocab.py

# C) mapeia sentenças -> sequências de vídeos (base + augmentations)
.venv/bin/python build_sentence_videos.py

# D) extrai landmarks (T,115,3) — várias horas; suporta resume
.venv/bin/python extract_landmarks.py --workers 8

# E) normaliza geometria + dinâmica (T,115,9)
.venv/bin/python normalize_landmarks.py --workers 8

# F) sintetiza frases contínuas (Keyframe Blending) + manifesto
.venv/bin/python build_sentence_features.py \
    --sentences-json pt_br2libras_gloss_sentence_videos.json \
    --landmarks-dir landmarks_115 --out-dir sentence_features \
    --manifest sentence_features/manifest.json

# (F→G) converter o manifesto p/ [{features,text}] + split por sentence_id  [pendente]

# G) treina o modelo SLT (LandmarkEncoder + PTT5) — usa GPU
.venv/bin/python slt_model.py --train-manifest train.json --val-manifest val.json \
    --epochs 30 --batch-size 8 --out-dir checkpoints
```

---

## 5. Artefatos produzidos

| Arquivo / pasta | Etapa | Conteúdo |
|---|---|---|
| `libras_gloss_vocab.txt` | A | Vocabulário das glosas (`palavra<TAB>freq`). |
| `pt_br2libras_gloss_in_vocab.json` | B | Sentenças cobertas pelo vocabulário, com `&` expandido. |
| `pt_br2libras_gloss_sentence_videos.json` | C | Sentenças → sequências de vídeos (`base` + `sentences`). |
| `landmarks_115/*.npy` | D | Landmarks por vídeo, `(T, 115, 3)`. |
| `landmarks_115/*_pose_vis.npy` | D | (Opcional) visibilidade da pose, `(T, 33)`. |
| `landmarks_115_norm9/*.npy` | E | Características normalizadas, `(T, 115, 9)`. |
| `sentence_features/*.npy` | F | Sequência contínua por frase, `(T, 115, 9)`. |
| `sentence_features/manifest.{json,csv}` | F | Mapeamento feature → frase + metadados. |
| `sentence_features/manifest.stats.json` | F | Taxa de descarte e contagens (paper). |
| `checkpoints/best.pt` + `epochN.pt` | G | Pesos do modelo SLT treinado. |

---

## 6. Notas e decisões de projeto

- **Normalização de acento** é obrigatória ao casar glosas (com acento) contra o
  vocabulário de vídeos (ASCII). Sem isso, quase nada casa.
- **Cobertura do vocabulário é baixa de propósito:** o banco de vídeos tem só
  ~1.300 sinais, então a maioria das 127k sentenças é descartada. A **taxa de
  descarte** deve ser reportada no paper (Passo 4.2 do plano).
- **`base` vs `sentences`:** a base (intérprete `0`, sem augmentation) é reservada
  para **teste**; as augmentations são material de **treino**. Isso ajuda a evitar
  vazamento de dados na avaliação (Passo 6.1).
- **Índices `.npy` por vídeo**, não por palavra: assim uma frase é montada
  concatenando os `.npy` das suas palavras (Etapa F).
- **Fine-tuning cross-modal (Etapa G):** o decoder PTT5 é *fine-tunado* (transfer
  learning) enquanto o encoder de landmarks e o adaptador são treinados **do zero**.
  Treinar os dois com uma **taxa única** (`3e-5`, como no plano) tem um risco: é
  lento demais para o encoder aleatório e, no início, o *cross-attention* recebe
  features ruidosas que podem degradar os pesos bons do decoder. Alternativas
  naturais para o *ablation study*: **taxas diferenciadas** (encoder alto, decoder
  baixo), **congelar o decoder** no começo, ou **LoRA/PEFT**. O plano pede full
  fine-tuning com taxa única, e é assim que o `slt_model.py` está por padrão.

## 7. Próximos passos (ainda não implementados)

- **Conversor F → G:** achatar `manifest.json` (`items` → `[{features, text}]`) e
  fazer o **split treino/val/teste por `sentence_id`** (evita vazamento).
- **Passo 6:** protocolo de avaliação — **BLEU-4** (SacreBLEU), **ROUGE-L**,
  **chrF** — e o *ablation study* (colagem "seca" vs. *Keyframe Blending*;
  posição-only vs. +dinâmica; taxa única vs. diferenciada/decoder congelado).
