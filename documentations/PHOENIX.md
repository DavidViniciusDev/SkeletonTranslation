# Rodando o SkeletonTranslation com a base PHOENIX

**Data:** 2026-09-01
**Objetivo:** treinar o modelo de SLT deste projeto (LandmarkEncoder + decoder T5)
na base alemã RWTH-PHOENIX-Weather, reaproveitando o pipeline existente.

Este documento é didático de propósito: explica **por que** cada decisão foi
tomada, não só quais comandos rodar. Se você só quer os comandos, vá para a
[seção 6](#6-passo-a-passo-reproduzível).

---

## 1. Por que o PHOENIX precisa de adaptação

O pipeline original foi desenhado para a **Libras**, num cenário de escassez de
dados. Ele *sintetiza* frases sinalizadas: pega um corpus de texto paralelo
(português ↔ glosa), pega um léxico de vídeos de sinais **isolados**
(V-LIBRASIL) e cola os sinais na ordem que a glosa manda, interpolando
transições. Daí os seis passos:

| Passo | O que faz | Serve para o PHOENIX? |
|---|---|---|
| s1a | monta vocabulário de glosas | ❌ |
| s1b | filtra sentenças cujas glosas existem no léxico | ❌ |
| s2  | monta a sequência de vídeos de cada frase | ❌ |
| s3  | extrai landmarks (MediaPipe) → `(T,115,3)` | ✅ |
| s4  | normaliza geometria + dinâmica → `(T,115,9)` | ✅ |
| s5  | cola sinais isolados (*Keyframe Blending*) | ❌ |

**Os quatro passos marcados com ❌ existem só para a síntese.** No PHOENIX cada
amostra **já é** uma frase contínua, sinalizada de verdade por um intérprete
humano. Não há nada para montar nem colar. Sobram s3 e s4 — que são
justamente os dois passos agnósticos de base.

Isso significa que a adaptação **não exigiu alterar uma linha do pacote
`skeltrans/`**. O projeto já havia sido escrito prevendo isso:
`skeltrans/extraction/frame_sources.py::open_frames()` despacha pelo tipo do
caminho — diretório vira sequência de imagens, arquivo vira vídeo — e os
caminhos de s3/s4 são todos sobreponíveis por CLI.

---

## 2. Achado crítico: PHOENIX-2014 ≠ PHOENIX-2014T

Existem duas bases com nome parecido, e **só uma serve para tradução.**

### PHOENIX-2014 (`phoenix2014-release`) — não serve

É o corpus de **reconhecimento**. A anotação é:

```
id|folder|signer|annotation
01April_2010_Thursday_heute_default-0|.../1/*.png|Signer04|__ON__ LIEB ZUSCHAUER ABEND ...
```

Só existe `annotation` = **sequência de glosas**. Não há frase em alemão. Como
o modelo deste projeto traduz landmarks → **texto falado**, com esta base o
único alvo possível seria a própria glosa — o que é *reconhecimento*
(sign2gloss), não tradução (sign2text). Seria preciso reenquadrar o
experimento na tese.

Há também um bloqueio técnico: os frames ficam em `<id>/1/*.png`. A função
`out_name()` do s3 nomeia a saída pelo *basename* do caminho, e o basename de
todos é `1` → **todas as sequências gerariam `1.npy`, sobrescrevendo-se.**

### PHOENIX-2014T (`PHOENIX-2014-T-release-v3`) — é esta

```
name|video|start|end|speaker|orth|translation
11August_2010_Wednesday_tagesschau-1|...|Signer08|JETZT WETTER MORGEN DONNERSTAG ZWOELF FEBRUAR|und nun die wettervorhersage für morgen donnerstag den zwölften august
```

Tem as duas colunas que importam:

- **`translation`** — frase em alemão → **alvo do decoder** (é o que faz disto SLT)
- **`orth`** — sequência de glosas → **alvo da supervisão auxiliar de CTC**

E o layout de frames é `<split>/<name>/images0001.png`: o nome do diretório é
único (verificado: 8.257 nomes, 8.257 únicos), então cada frase gera seu
próprio `.npy` sem colisão.

> **Lição:** confira sempre se o corpus tem a coluna de tradução antes de
> planejar um experimento de SLT. O nome da pasta não basta.

O suporte à variante 2014 foi mantido no script (`--variant 2014`), contornando
a colisão com *symlinks* — mas o padrão é 2014T.

---

## 3. Achado crítico: o torch instalado não roda nas GPUs desta máquina

O primeiro teste de treino falhou com `CUBLAS_STATUS_ARCH_MISMATCH`. Isolando
com um matmul mínimo:

```
torch     : 2.12.1+cu130          (venv ../SkeLTrans/.venv)
arch_list : ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
GPUs      : 2x NVIDIA TITAN V  →  sm_70
→ CUDA error: no kernel image is available for execution on the device
```

As TITAN V são **Volta (`sm_70`)**, arquitetura removida dos builds CUDA 13 do
PyTorch. **Nenhuma operação de GPU funciona nesse venv** — e isso não é
específico do PHOENIX: atinge igualmente o treino do V-LIBRASIL. O driver
(580.95.05) está correto; o problema é só a lista de arquiteturas compilada no
*wheel*.

A solução não exigiu instalar torch: o ambiente conda `new_bmt` já tinha um
torch com suporte a Volta. Clonamos (para não mexer num ambiente de outro
trabalho) e completamos as dependências de NLP que faltavam.

> **Lição:** `torch.cuda.is_available()` retornando `True` **não** garante que a
> GPU funciona. Ele só verifica driver e runtime. Para saber de verdade, cheque
> `torch.cuda.get_arch_list()` contra a *compute capability* da placa, ou rode
> um matmul de verdade.

### O ambiente final

```bash
conda create -n skeltrans-cu121 --clone new_bmt -y
/home/dvsilva/miniconda3/envs/skeltrans-cu121/bin/pip install \
    "transformers==4.44.2" sentencepiece sacrebleu nltk bitsandbytes
```

| Pacote | Versão | Observação |
|---|---|---|
| python | 3.9.25 | abaixo do 3.10+ que o INSTRUCTIONS.md declara; funcionou |
| torch | 2.3.1 | inclui `sm_70` — este é o requisito que manda |
| transformers | **4.44.2** | pinado; ver abaixo |
| sentencepiece | 0.2.2 | tokenizer do mT5 |
| sacrebleu | 2.6.0 | BLEU-1..4 |
| nltk | 3.9.2 | METEOR |
| bitsandbytes | 0.48.2 | AdamW 8-bit do `--low-vram`; funciona em Volta |

**Por que o `transformers` está pinado em 4.44.2.** A 4.57 recusa carregar
arquivos `.bin` com torch < 2.6 (CVE-2025-32434, `check_torch_load_is_safe`), e
o `google/mt5-small` só publica `pytorch_model.bin` no `main` — o
`model.safetensors` que aparece no cache local vem de um *revision* de PR, não
da branch principal. Como o torch está preso em 2.3.1 pelo hardware, alinhar o
`transformers` à mesma época resolve de uma vez, para qualquer modelo.

O venv antigo (`../SkeLTrans/.venv`) **continua servindo para a extração**
(s3/s4), que usa MediaPipe/OpenCV na CPU e não toca a GPU: mediapipe 0.10.14,
opencv 4.13.0.

---

## 4. Achado: o decoder precisa mudar

O default do projeto é `unicamp-dl/ptt5-base-portuguese-vocab`: um T5 com
SentencePiece **português**. O alvo do PHOENIX é **alemão**. Com o vocabulário
errado, o alemão se fragmenta em subwords ruins e o fine-tuning parte de um
prior linguístico que não tem nada a ver com a tarefa.

Trocamos por **`google/mt5-small`** (multilíngue, cobre alemão), via `--t5`.

Por que `small` e não `base`: o mT5 tem vocabulário de 250k tokens, então o
`mt5-base` chega a ~580M parâmetros só de embedding + camadas. Somado ao
encoder de 6 camadas em sequências de até 475 frames, isso aperta os 11,8 GiB
das TITAN V. Comece pelo `small`, valide, e só então tente o `base` com batch
menor.

> **Nota:** o `trainer.py` instancia `T5ForConditionalGeneration` fixo, então o
> `transformers` avisa ao carregar um checkpoint `mt5`. Carrega os 192/192 pesos
> e funciona (o `lm_head` destravado é o correto para mT5). Trocar por
> `AutoModelForSeq2SeqLM` — 1 linha — silenciaria o aviso.

---

## 5. Os scripts criados

Ficam em [`tools/`](../tools/) e são os únicos artefatos novos de código.

### `tools/phoenix_prepare.py`

Traduz as anotações do PHOENIX nos dois artefatos que faltavam:

1. **`frames.json`** — a lista de diretórios de frames na forma que
   `collect_videos()` do s3 espera. Isso é o truque que evita mexer no s3:
   em vez de dar ao s3 uma nova flag, emitimos o formato que ele já lê.

   ```json
   [{"base": {"videos": [{"video": "/caminho/para/<name>"}]}}]
   ```

2. **`train.json` / `dev.json` / `test.json`** — manifestos no esquema legado
   do `LandmarkTextDataset`:

   ```json
   [{"features": "../features/landmarks_115_norm9/<name>.npy",
     "text": "und nun die wettervorhersage für morgen donnerstag ...",
     "tokens": ["JETZT","WETTER","MORGEN","DONNERSTAG","ZWOELF","FEBRUAR"],
     "name": "...", "signer": "Signer08", "split": "train"}]
   ```

   `features` é gravado **relativo ao próprio manifesto**, então funciona com
   ou sem `--features-dir` no treino. `tokens` é o campo que a supervisão CTC lê.

Opções úteis: `--variant {2014T,2014}`, `--features-subdir`, e `--sample N`,
que emite também um `frames.sample.json` com N sequências espalhadas por **todo**
o corpus (passo constante, determinístico). O `--sample` existe porque
`--limit N` do s3 pegaria os N primeiros em ordem alfabética — todos do mesmo
mês e de poucos sinalizadores, o que daria uma estimativa enviesada.

**Não usamos o `split_manifest.py`.** Ele implementa o protocolo do V-LIBRASIL
(intérprete *held-out* + augmentations). O PHOENIX já vem com splits oficiais, e
usá-los é o que torna a métrica comparável à literatura.

### `tools/phoenix_detection_report.py`

O s3 preenche com `[0,0,0]` todo componente que o MediaPipe não detectou no
frame (*zero-fill*). Numa base de vídeo limpo e em close-up isso é raro; no
PHOENIX (sinalizador pequeno em 210×260, motion blur) pode atingir uma fração
grande dos frames — sobretudo as **mãos**, o canal de maior carga linguística.

O script lê os `.npy` e conta, por componente, em quantos frames o bloco inteiro
é exatamente zero. Serve para medir isso **antes** de comprometer horas de
extração, e depois para ter o número definitivo da seção de limitações.

---

## 6. Passo a passo reproduzível

```bash
cd /libras/Doutorado/SkeletonTranslation

EXTRACT=../SkeLTrans/.venv/bin/python                              # CPU: MediaPipe
TRAIN=/home/dvsilva/miniconda3/envs/skeltrans-cu121/bin            # GPU: torch sm_70
```

### 6.1 Preparar manifestos

```bash
$EXTRACT tools/phoenix_prepare.py \
    --phoenix-root /libras/slt/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
    --out-dir data_phoenix --sample 50
```

Resultado: `train 7096 / dev 519 / test 642`, **0 ausentes**, `frames.json` com
8.257 sequências.

### 6.2 (recomendado) Medir a qualidade da extração numa amostra

Antes de gastar ~5 h, extraia 50 sequências e olhe a taxa de detecção:

```bash
$EXTRACT extract_landmarks.py --json data_phoenix/interim/frames.sample.json \
    --out data_phoenix/raw/landmarks_115 --workers 20
$EXTRACT tools/phoenix_detection_report.py --in-dir data_phoenix/raw/landmarks_115
```

Os `.npy` da amostra são reaproveitados na extração completa (s3 tem *resume*).

### 6.3 Extração completa (s3) — o passo caro

```bash
nohup $EXTRACT extract_landmarks.py \
    --json data_phoenix/interim/frames.json \
    --out  data_phoenix/raw/landmarks_115 \
    --workers 20 > extraction.phoenix.log 2>&1 &

# acompanhar (a barra de progresso usa \r; o tr converte para linhas)
tr '\r' '\n' < extraction.phoenix.log | grep -a 'vid/s' | tail -1
```

Se cair, **reexecute o mesmo comando** — o s3 pula os `.npy` que já existem.

### 6.4 Normalização (s4)

```bash
$EXTRACT normalize_landmarks.py \
    --in-dir  data_phoenix/raw/landmarks_115 \
    --out-dir data_phoenix/features/landmarks_115_norm9 \
    --workers 20
```

### 6.5 Treino

```bash
nohup $TRAIN/torchrun --nproc_per_node=2 slt_model.py \
    --train-manifest data_phoenix/interim/train.json \
    --val-manifest   data_phoenix/interim/dev.json \
    --features-dir   data_phoenix/features/landmarks_115_norm9 \
    --t5 google/mt5-small --use-ctc --ctc-weight 0.3 \
    --grad-checkpoint --low-vram \
    --epochs 60 --batch-size 4 --lr 5e-5 --warmup-steps 1000 \
    --max-text-len 64 --patience 8 \
    --out-dir checkpoints_phoenix --num-workers 8 --log-every 100 \
    > training.phoenix.log 2>&1 &
```

### 6.6 Avaliação no test oficial

```bash
$TRAIN/python evaluate_slt.py \
    --checkpoint    checkpoints_phoenix/best.pt \
    --test-manifest data_phoenix/interim/test.json \
    --features-dir  data_phoenix/features/landmarks_115_norm9 \
    --out           checkpoints_phoenix/test_metrics.json
```

---

## 7. Escolhas de hiperparâmetro, e por quê

| Flag | Valor | Motivo |
|---|---|---|
| `--t5` | `google/mt5-small` | alvo é alemão; o default é português (§4) |
| `--use-ctc` | ligado | o PHOENIX tem glosas reais alinhadas (`orth`). Isso reproduz a receita *Sign2(Gloss+Text)* da literatura e é o maior ganho disponível sem escrever código |
| `--batch-size` | 4 (×2 GPUs = 8) | mediana 110 frames, p95 205, **máx 475**; atenção é O(T²) e o encoder não faz downsampling temporal |
| `--grad-checkpoint` | ligado | praticamente obrigatório com T até 475: corta a VRAM de ativações ao custo de ~25-30% de velocidade |
| `--low-vram` | ligado | offload do encoder T5 + bf16 + AdamW 8-bit |
| `--max-text-len` | 64 | traduções têm média 14 palavras, máx 52 → 64 subwords cobre quase tudo |
| `--lr` | 5e-5 | encoder treina do zero, decoder é fine-tune; 3e-5 (default) é conservador demais |
| `--patience` | 8 | 7k amostras; early stopping evita queimar 60 épocas em vão |

**Sobre padding:** o collate preenche até o maior `T` do lote. Um batch que
mistura T=16 e T=475 desperdiça muito. Se o throughput incomodar, ordenar o
manifesto por duração (*bucketing*) é a otimização de maior retorno.

**Sobre `--low-vram` em DDP:** o log avisa que o offload do encoder T5 vira
*congelamento*. Isso é inofensivo aqui: o `slt_model.py` passa
`encoder_outputs=enc_out`, ou seja, o encoder do T5 é contornado e a memória do
cross-attention vem toda do `LandmarkEncoder`. Congelar peso que nunca é usado
não custa nada.

**Sobre bf16 em Volta:** o autocast bf16 ativa, mas em Volta é emulado (não há
tensor core bf16). Economiza memória; não acelera.

---

## 8. Resultados medidos

### Custo

| Etapa | Tempo | Notas |
|---|---|---|
| s3 (extração) | **4 h 39 min** | 8.207 sequências, `--workers 20`, **0 falhas** |
| s4 (normalização) | **1 min 29 s** | 8.207 arquivos, 0 falhas |
| disco | 1,3 GB + 3,7 GB | `landmarks_115` + `landmarks_115_norm9` |

Throughput medido: 4,3 fps por processo; ~0,36 CPU-s por frame; 947.756 frames
no total (train 827k / dev 56k / test 65k).

### Qualidade da extração

Corpus completo, 947.756 frames:

| Componente | Taxa de detecção | Zero-fill |
|---|---|---|
| pose | **100,0 %** | 1 frame |
| face | **99,1 %** | 8.540 frames |
| mão esquerda | **72,4 %** | 261.753 frames |
| mão direita | **75,9 %** | 227.936 frames |
| **ambas as mãos ausentes** | — | **10,4 %** (98.555 frames) |

Duração das sequências: min 16, mediana 110, p95 205, máx 475 frames.

**A amostra de 50 previu o corpus inteiro dentro de ~1 ponto percentual** em
todos os componentes (mão esq 71,8 % → 72,4 %; ambas ausentes 11,6 % → 10,4 %).
Ou seja: o `--sample 50` é um preditor confiável e vale sempre rodar antes.

### Upscaling não resolve (experimento negativo)

Hipótese testada: as mãos têm ~35 px em 210×260, então ampliar os frames
melhoraria a detecção. **Não melhora.** Mesmo conjunto, `INTER_CUBIC`:

| escala | mão esq | mão dir | ambas ausentes | fps |
|---|---|---|---|---|
| 1× | 61,3 % | 76,9 % | 14,9 % | 7,2 |
| 2× | 61,7 % | 76,4 % | 14,6 % | 7,2 |
| 3× | 60,4 % | 77,3 % | 14,7 % | 7,2 |

Diferenças < 1,3 pp, dentro do ruído — e o **fps idêntico** revela o mecanismo:
o MediaPipe redimensiona internamente para o tamanho fixo do modelo, então
upscalar por interpolação não acrescenta informação nenhuma. O limite é perda
real de detalhe e motion blur no material original.

**Registrado aqui para ninguém repetir o experimento.** Se a detecção precisar
melhorar, o caminho é trocar o extrator (MMPose/HRNet whole-body, que é o que a
literatura de esqueleto no PHOENIX usa), não mexer na resolução de entrada.

### Validação do treino

```
[smoke] TUDO OK ✔  (arquitetura do Passo 5 valida ponta-a-ponta)

Dispositivo: cuda:0 | DDP com 2 processos (1 GPU cada) | batch efetivo = 4 x 2
CTC ativo | vocabulario de glosas: 136 rotulos (inclui <blank>/<unk>)
[grad-checkpoint] ativo · [low-vram] bf16 ativo · AdamW 8-bit (bitsandbytes) ativo
[epoch 1] train_loss=63.1218 val_loss=18.1337 → Treino concluido.
```

Validado numa sub-amostra de 43 frases: mT5 carrega (192/192 pesos), tokenizer
alemão funciona, `tokens` vira vocabulário CTC, collate empacota os dois alvos,
checkpoint grava, DDP nas 2 GPUs roda. No corpus completo o vocabulário de
glosas é de **1.085 rótulos** (train).

---

## 9. Limitações a carregar para a tese

**Reporte a taxa de detecção.** Com ~74 % de detecção manual e 10,4 % dos frames
sem nenhuma informação de mão, há duas reduções de informação empilhadas:

1. **pixels → esqueleto** — deliberada, é a escolha metodológica do trabalho. É
   o que dá invariância a sinalizador e o que torna a síntese de Libras possível
   (concatenar esqueletos funciona; concatenar features CNN de vídeos diferentes
   não).
2. **esqueleto → esqueleto incompleto** — acidental, é limitação do MediaPipe em
   210×260. Não é propriedade do método.

Sem reportar (2), um leitor atribui toda a diferença de BLEU ao método. Com ela,
você mostra quanto é ruído de medição.

**Sobre comparações:** os baselines canônicos do PHOENIX-2014T (Camgöz et al. e
sucessores) **devem** estar na tabela — omiti-los parece evasivo. Mas explicite
o regime de entrada de cada linha (features CNN vs. esqueleto), porque são
problemas de informação diferente. E note que, dentro do próprio Camgöz, as
linhas *Sign2Text* e *Sign2(Gloss+Text)* têm resultados bem distintos; como aqui
se usa `--use-ctc`, a comparação análoga é com a segunda.
**Confira os números nos papers** — inclusive quais são dev e quais são test,
que é confusão comum nessa literatura.

**Mantenha a proporção.** O PHOENIX é experimento secundário. A pergunta que ele
responde não é "eu bato o SOTA?", é "meu encoder aprende em sinalização contínua
real, ou só nas frases sintéticas?". Para isso, a comparação mais informativa é
entre o próprio modelo nas duas bases — é ela que mede o *domain shift* que o
`INSIGHT-TECHNICAL.md` coloca como objeto de estudo.

**Experimento sugerido.** O `--per-file` do `phoenix_detection_report.py` dá a
taxa de detecção por sequência. Partindo o test em terços por qualidade de
extração e avaliando o BLEU em cada um, mede-se diretamente quanto a extração
custa — e sustenta o argumento de que um extrator melhor elevaria o resultado
sem mudar a arquitetura. Transforma uma limitação em achado quantificado.

---

## 10. Troubleshooting

| Sintoma | Causa | Solução |
|---|---|---|
| `CUBLAS_STATUS_ARCH_MISMATCH` ou `no kernel image is available` | wheel do torch sem `sm_70` | use o env `skeltrans-cu121`; confira com `torch.cuda.get_arch_list()` |
| `ValueError: ... upgrade torch to at least v2.6 ... CVE-2025-32434` | `transformers` ≥ 4.56 + torch < 2.6 + checkpoint `.bin` | `pip install "transformers==4.44.2"` |
| Todos os `.npy` saem com o mesmo nome | variante 2014, frames em `<id>/1/` | use `--variant 2014` (cria symlinks) ou o 2014T |
| `diretorio sem imagens reconhecidas` | caminho aponta para o pai do diretório de frames | confira `--features-subdir` |
| Extração interrompida | — | reexecute o mesmo comando; s3 e s4 têm *resume* (`--overwrite` força) |
| OOM no treino | T até 475, atenção O(T²) | `--grad-checkpoint --low-vram`, baixe `--batch-size`, use `mt5-small` |
| `bitsandbytes nao instalado` | dependência opcional do `--low-vram` | `pip install bitsandbytes` (funciona em Volta) |
| Log de progresso parece uma linha só | barra usa `\r` | `tr '\r' '\n' < arquivo \| grep -a 'vid/s' \| tail -1` |

---

## 11. Árvore de arquivos gerada

```
Doutorado/SkeletonTranslation/
├── documentations/
│   └── PHOENIX.md                        este documento
├── tools/
│   ├── phoenix_prepare.py                CSV → frames.json + manifestos
│   └── phoenix_detection_report.py       taxa de detecção / zero-fill
├── data_phoenix/
│   ├── interim/
│   │   ├── frames.json                   8.257 diretórios (entrada do s3)
│   │   ├── frames.sample.json            50 sequências (amostra)
│   │   ├── train.json                    7.096 itens
│   │   ├── dev.json                        519 itens
│   │   └── test.json                       642 itens
│   ├── raw/landmarks_115/                8.257 .npy (T,115,3) — 1,3 GB
│   └── features/landmarks_115_norm9/     8.257 .npy (T,115,9) — 3,7 GB
├── extraction.phoenix.log
└── checkpoints_phoenix/                  (após o treino)
```

`data_phoenix/` é deliberadamente separado de `data/` para não misturar
artefatos do PHOENIX com os do V-LIBRASIL. Como s3 e s4 aceitam todos os
caminhos por CLI, não foi preciso editar `skeltrans/extraction/config.py`.
