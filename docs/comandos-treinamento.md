# Comandos de treinamento, módulo a módulo

Todos os comandos para treinar, avaliar e usar cada módulo do SkeLTrans no
**PHOENIX-2014T**, em duas versões:

- **PowerShell**: máquina local Windows (`.venv\Scripts\python.exe`);
- **bash**: servidor Linux (`python3`, `torchrun`).

Cada módulo é **uma mudança sobre a receita base**. Para comparar resultados,
mude uma coisa por vez e mantenha a semente.

| # | Módulo | O que muda | Guia |
|---|---|---|---|
| 0 | [Preparação e testes](#0-preparação) | — | — |
| 1 | [Receita base (v15)](#1-receita-base-v15) | — | [docs/encoders/README.md](encoders/README.md) |
| 2 | [Encoder `flat`](#2-encoder-flat-linha-de-base) | nada (é a linha de base) | — |
| 3 | [Encoder `parts`](#3-encoder-parts) | projeção por parte do corpo | [parts.md](encoders/parts.md) |
| 4 | [Encoder `gcn`](#4-encoder-gcn) | grafo do esqueleto | [gcn.md](encoders/gcn.md) |
| 5 | [Modo `concept`](#5-modo-concept) | uma posição de memória por sinal | [concept.md](encoders/concept.md) |
| 6 | [Pré-treino do encoder](#6-pré-treino-do-encoder-masked-pose-modeling) | encoder começa pré-treinado | [pretrain.md](encoders/pretrain.md) |
| 7 | [Augmentation](#7-augmentation-dos-keypoints) | dados variados a cada época | [AUGMENTATION.md](../AUGMENTATION.md) |
| 8 | [Multi-GPU](#8-várias-gpus-ddp) | mesmo treino em N GPUs | [MULTI_GPU.md](../MULTI_GPU.md) |
| 9 | [Depois do treino](#9-depois-do-treino) | avaliação, média, decodificação, inferência | [EVALUATION.md](../EVALUATION.md) |

---

## 0. Preparação

Uma vez por terminal. Ajuste `DATA` para onde estão os dados do PHOENIX
(layout em [documentations/PHOENIX.md](../documentations/PHOENIX.md)).

**PowerShell**

```powershell
$PY   = ".venv\Scripts\python.exe"
$DATA = "data_phoenix"
$FEAT = "$DATA\features\landmarks_115_norm9"
$env:PYTHONIOENCODING = "utf-8"     # evita UnicodeEncodeError nos prints
```

**bash**

```bash
PY=python3
DATA=data_phoenix
FEAT=$DATA/features/landmarks_115_norm9
```

### Conferir a instalação (sem dados, CPU, ~1 min cada)

```powershell
& $PY slt_model.py --smoke-test --device cpu --use-ctc --ds-steps 2
& $PY slt_model.py --smoke-test --device cpu --encoder-type parts
& $PY slt_model.py --smoke-test --device cpu --encoder-type gcn --use-ctc
& $PY slt_model.py --smoke-test --device cpu --temporal-mode concept --ds-steps 2
& $PY pretrain_encoder.py --smoke-test
```

### Testes unitários

```powershell
Get-ChildItem tests\test_*.py | ForEach-Object { & $PY $_.FullName }
```

```bash
for t in tests/test_*.py; do $PY "$t"; done
```

---

## 1. Receita base (v15)

O melhor resultado até agora: **BLEU-4 19,22, WER 37,5**. Todos os módulos
abaixo partem dela.

**PowerShell**

```powershell
$BASE = @(
  "--train-manifest", "$DATA\interim\train.json",
  "--val-manifest",   "$DATA\interim\dev.json",
  "--features-dir",   $FEAT,
  "--t5", "GermanT5/t5-efficient-gc4-german-base-nl36",
  "--d-model", "512", "--num-layers", "6", "--ds-steps", "2",
  "--use-ctc", "--ctc-weight", "1.0",
  "--extra-features", "A,B,C,D,E,F", "--input-norm",
  "--batch-size", "32", "--lr", "5e-5", "--encoder-lr", "3e-4", "--warmup-steps", "800",
  "--label-smoothing", "0.1",
  "--epochs", "70", "--patience", "20",
  "--eval-every", "1", "--eval-beams", "4", "--select-by", "bleu4",
  "--low-vram", "--grad-checkpoint",
  "--seed", "1"
)
```

**bash**

```bash
BASE=(
  --train-manifest "$DATA/interim/train.json"
  --val-manifest   "$DATA/interim/dev.json"
  --features-dir   "$FEAT"
  --t5 GermanT5/t5-efficient-gc4-german-base-nl36
  --d-model 512 --num-layers 6 --ds-steps 2
  --use-ctc --ctc-weight 1.0
  --extra-features A,B,C,D,E,F --input-norm
  --batch-size 32 --lr 5e-5 --encoder-lr 3e-4 --warmup-steps 800
  --label-smoothing 0.1
  --epochs 70 --patience 20
  --eval-every 1 --eval-beams 4 --select-by bleu4
  --low-vram --grad-checkpoint
  --seed 1
)
```

Uma flag repetida depois de `@BASE` / `"${BASE[@]}"` vale a última
ocorrência. Por exemplo, `@BASE --lr 3e-5` troca só o lr. No servidor,
escolha a GPU com `--device cuda:1`.

---

## 2. Encoder `flat` (linha de base)

Um `Linear` único sobre as colunas do frame: o encoder original.

```powershell
& $PY slt_model.py @BASE --out-dir checkpoints_phoenix_flat
```

```bash
$PY slt_model.py "${BASE[@]}" --out-dir checkpoints_phoenix_flat
```

---

## 3. Encoder `parts`

Uma MLP por parte do corpo (pose, mão esq., mão dir., rosto, extras) + fusão.

```powershell
& $PY slt_model.py @BASE --encoder-type parts --out-dir checkpoints_phoenix_parts
```

```bash
$PY slt_model.py "${BASE[@]}" --encoder-type parts --out-dir checkpoints_phoenix_parts
```

---

## 4. Encoder `gcn`

Convolução em grafo sobre os 115 pontos do esqueleto. É o mais caro: se faltar
memória, acrescente `--batch-size 16`.

```powershell
& $PY slt_model.py @BASE --encoder-type gcn --out-dir checkpoints_phoenix_gcn
```

```bash
$PY slt_model.py "${BASE[@]}" --encoder-type gcn --out-dir checkpoints_phoenix_gcn
```

---

## 5. Modo `concept`

Uma posição de memória por **sinal**, com fronteiras vindas do CTC. Exige
`--use-ctc` (já na receita). `--num-layers` passa a ser o andar de frames.

**Configuração principal** (4 + 2 = 6 camadas, como a receita):

```powershell
& $PY slt_model.py @BASE --temporal-mode concept --num-layers 4 --concept-layers 2 `
    --out-dir checkpoints_phoenix_concept
```

```bash
$PY slt_model.py "${BASE[@]}" --temporal-mode concept --num-layers 4 --concept-layers 2 \
    --out-dir checkpoints_phoenix_concept
```

**Ablações** (acrescente à linha acima, com outro `--out-dir`):

| Pergunta | Flags extras | `--out-dir` sugerido |
|---|---|---|
| Uma posição por sinal é pouca? | `--concept-queries 2` | `checkpoints_phoenix_concept_k2` |
| Treinar com as fronteiras da inferência ajuda? | `--concept-greedy-prob 0.3` | `checkpoints_phoenix_concept_gp03` |
| As deleções vêm de ter menos camadas de frames? | `--num-layers 6` | `checkpoints_phoenix_concept_l6` |
| Menos downsampling ajuda o CTC? | `--ds-steps 1` | `checkpoints_phoenix_concept_ds1` |

---

## 6. Pré-treino do encoder (masked pose modeling)

Duas fases. Na 1ª, o encoder aprende a reconstruir keypoints escondidos, sem
rótulos. Na 2ª, o treino de SLT começa com esses pesos.

**Fase 1: pré-treino.** As flags de encoder (`--extra-features`, `--input-norm`,
`--encoder-type`, `--d-model`, `--nhead`, `--num-layers`, `--ds-steps`) precisam
ser **iguais** às da fase 2.

```powershell
& $PY pretrain_encoder.py `
    --manifests $DATA\interim\train.json --val-manifest $DATA\interim\dev.json `
    --features-dir $FEAT --extra-features A,B,C,D,E,F --input-norm `
    --d-model 512 --nhead 8 --num-layers 6 --ds-steps 2 `
    --mask-mode mixed --mask-ratio 0.4 --mask-span 8 `
    --epochs 100 --patience 10 --batch-size 32 --lr 5e-4 --warmup-steps 1000 `
    --low-vram --grad-checkpoint --seed 1 --out-dir checkpoints_pretrain
```

```bash
$PY pretrain_encoder.py \
    --manifests "$DATA/interim/train.json" --val-manifest "$DATA/interim/dev.json" \
    --features-dir "$FEAT" --extra-features A,B,C,D,E,F --input-norm \
    --d-model 512 --nhead 8 --num-layers 6 --ds-steps 2 \
    --mask-mode mixed --mask-ratio 0.4 --mask-span 8 \
    --epochs 100 --patience 10 --batch-size 32 --lr 5e-4 --warmup-steps 1000 \
    --low-vram --grad-checkpoint --seed 1 --out-dir checkpoints_pretrain
```

Com mais keypoints sem rótulo, liste vários manifestos em `--manifests`
(ex.: `train.json outro_dataset.json libras_sintetico.json`). O pré-treino roda
em uma GPU (sem `torchrun`).

**Fase 2: SLT a partir do encoder pré-treinado.**

```powershell
& $PY slt_model.py @BASE --init-encoder checkpoints_pretrain\encoder.pt `
    --out-dir checkpoints_phoenix_mpm
```

```bash
$PY slt_model.py "${BASE[@]}" --init-encoder checkpoints_pretrain/encoder.pt \
    --out-dir checkpoints_phoenix_mpm
```

**Com o modo `concept`:** pré-treine com `--num-layers 4` e use na fase 2
`--temporal-mode concept --num-layers 4 --concept-layers 2 --init-encoder ...`.

---

## 7. Augmentation dos keypoints

Só no treino. Teste uma de cada vez e depois a combinação:

```powershell
& $PY slt_model.py @BASE --aug-speed 0.2     --out-dir checkpoints_phoenix_aug_speed
& $PY slt_model.py @BASE --aug-spatial 0.2   --out-dir checkpoints_phoenix_aug_spatial
& $PY slt_model.py @BASE --aug-noise 0.001   --out-dir checkpoints_phoenix_aug_noise
& $PY slt_model.py @BASE --aug-flip          --out-dir checkpoints_phoenix_aug_flip
& $PY slt_model.py @BASE --aug-speed 0.2 --aug-spatial 0.2 --aug-noise 0.001 `
    --out-dir checkpoints_phoenix_aug_all
```

```bash
$PY slt_model.py "${BASE[@]}" --aug-speed 0.2 --aug-spatial 0.2 --aug-noise 0.001 \
    --out-dir checkpoints_phoenix_aug_all
```

A augmentation também vale no pré-treino (mesmas flags `--aug-*` no
`pretrain_encoder.py`).

---

## 8. Várias GPUs (DDP)

Qualquer comando de `slt_model.py` acima roda em N GPUs trocando `python3` por
`torchrun` (só no servidor Linux). `--batch-size` passa a ser **por GPU**, e
`--device` é ignorado.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 slt_model.py "${BASE[@]}" \
    --out-dir checkpoints_phoenix_flat_ddp
```

Se travar antes da época 1: `NCCL_P2P_DISABLE=1 torchrun ...`.

---

## 9. Depois do treino

Os comandos são iguais para todos os módulos: o tipo de encoder, o modo e o
pré-treino ficam gravados no `best.pt`. Troque `CK` pela pasta do treino.

**PowerShell**

```powershell
$CK = "checkpoints_phoenix_flat"

# avaliação no TESTE (BLEU, METEOR, WER de glosas, BERTScore)
& $PY evaluate_slt.py --checkpoint $CK\best.pt `
    --test-manifest $DATA\interim\test.json --features-dir $FEAT `
    --wer --bertscore de --out $CK\test_metrics.json

# média dos 5 melhores checkpoints pela métrica da validação
& $PY tools\average_checkpoints.py --dir $CK --best 5 --metric BLEU-4 --out $CK\avg_best5.pt

# ajuste da decodificação na VALIDAÇÃO (nunca no teste)
& $PY tools\tune_decoding.py --checkpoint $CK\best.pt `
    --val-manifest $DATA\interim\dev.json --features-dir $FEAT `
    --num-beams 4 5 --length-penalty 0.8 1.0 1.2 --no-repeat-ngram-size 0 3 `
    --out $CK\tune_decoding.json

# tradução de um vídeo
& $PY infer_slt.py --checkpoint $CK\best.pt --video frase.mp4 --aspect 210x260
```

**bash**

```bash
CK=checkpoints_phoenix_flat
$PY evaluate_slt.py --checkpoint $CK/best.pt \
    --test-manifest "$DATA/interim/test.json" --features-dir "$FEAT" \
    --wer --bertscore de --out $CK/test_metrics.json
$PY tools/average_checkpoints.py --dir $CK --best 5 --metric BLEU-4 --out $CK/avg_best5.pt
$PY tools/tune_decoding.py --checkpoint $CK/best.pt \
    --val-manifest "$DATA/interim/dev.json" --features-dir "$FEAT" \
    --num-beams 4 5 --length-penalty 0.8 1.0 1.2 --no-repeat-ngram-size 0 3 \
    --out $CK/tune_decoding.json
$PY infer_slt.py --checkpoint $CK/best.pt --video frase.mp4 --aspect 210x260
```

O `--aspect 210x260` corrige a proporção da imagem do PHOENIX. Use-o só se as
features do treino também foram normalizadas com ele (ver
[documentations/PHOENIX.md](../documentations/PHOENIX.md), seção 6.4).

Depois de avaliar, registre o resultado no [TIMELINE.md](../TIMELINE.md) (seção 2).

---

## Combinações

Os módulos são independentes e se somam na mesma linha. Exemplos:

```powershell
# gcn + concept
& $PY slt_model.py @BASE --encoder-type gcn --temporal-mode concept --num-layers 4 --concept-layers 2 `
    --out-dir checkpoints_phoenix_gcn_concept

# pré-treino + augmentation
& $PY slt_model.py @BASE --init-encoder checkpoints_pretrain\encoder.pt `
    --aug-speed 0.2 --aug-spatial 0.2 --out-dir checkpoints_phoenix_mpm_aug
```

Só combine depois de medir cada módulo sozinho: assim se sabe de onde veio o
efeito.
