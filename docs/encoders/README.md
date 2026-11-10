# Guias de execução dos encoders

Como treinar, avaliar e comparar cada variante do encoder de landmarks no
**PHOENIX-2014T**, rodando **localmente no Windows** (`.venv\Scripts\python.exe`,
PowerShell).

| Guia | Flag | O que muda | Eixo |
|---|---|---|---|
| [parts.md](parts.md) | `--encoder-type parts` | uma MLP por parte do corpo (pose, mãos, face) + fusão | espacial (dentro de um frame) |
| [gcn.md](gcn.md) | `--encoder-type gcn` | convolução em grafo sobre o esqueleto + média por parte | espacial (dentro de um frame) |
| [concept.md](concept.md) | `--temporal-mode concept` | uma posição de memória por **sinal**, com fronteiras vindas do CTC | temporal (entre frames) |

**Espacial × temporal.** `--encoder-type` decide como os 115 pontos de **um
frame** viram um vetor. `--temporal-mode` decide como os **frames** viram a
memória que o decoder lê. Os dois são independentes e se combinam, por exemplo
`--encoder-type gcn --temporal-mode concept`.

```
keypoints (T, 1242) ─► projeção espacial ─► conv temporal / ds-steps ─► Transformer ─► [conceitos] ─► T5
                       flat | parts | gcn                                              só no concept
```

---

## 1. Preparação (uma vez por terminal)

Os dados do PHOENIX seguem o layout de
[documentations/PHOENIX.md](../../documentations/PHOENIX.md). Aponte `$DATA` para
a pasta onde eles estão nesta máquina:

```powershell
$PY   = ".venv\Scripts\python.exe"
$DATA = "data_phoenix"            # ajuste para o caminho local
$FEAT = "$DATA\features\landmarks_115_norm9"
```

## 2. Receita base (v11)

Todos os guias partem da configuração da **v11**, o melhor resultado até agora
(BLEU-4 17,81, WER 40,01). Para comparar encoders, **mude só a flag do
encoder** e mantenha o resto, inclusive a semente:

```powershell
$BASE = @(
  "--train-manifest", "$DATA\interim\train.json",
  "--val-manifest",   "$DATA\interim\dev.json",
  "--features-dir",   $FEAT,
  "--t5", "GermanT5/t5-efficient-gc4-german-base-nl36",
  "--d-model", "512", "--num-layers", "6", "--ds-steps", "2",
  "--use-ctc", "--ctc-weight", "1.0",
  "--extra-features", "A,B,C,F", "--input-norm",
  "--batch-size", "32", "--lr", "1e-4", "--encoder-lr", "3e-4", "--warmup-steps", "800",
  "--epochs", "70", "--patience", "20",
  "--eval-every", "1", "--eval-beams", "4", "--select-by", "bleu4",
  "--low-vram", "--grad-checkpoint",
  "--seed", "42"                  # a v11 rodou sem semente; fixe uma para comparar braços
)

# linha de base (flat, o original)
& $PY slt_model.py @BASE --out-dir checkpoints_phoenix_flat
```

`@BASE` é o *splatting* do PowerShell: expande o array como argumentos. Uma
flag repetida depois de `@BASE` vale a última ocorrência (é o comportamento do
`argparse`), então dá para sobrescrever qualquer item da receita na própria
linha.

## 3. Avaliação e inferência: iguais para todos

`evaluate_slt.py` e `infer_slt.py` **não têm flags de encoder**: o tipo
(`flat`/`parts`/`gcn`) e o modo (`stride`/`concept`) são detectados pelos
pesos do checkpoint.

```powershell
& $PY evaluate_slt.py `
    --checkpoint    checkpoints_phoenix_concept\best.pt `
    --test-manifest $DATA\interim\test.json `
    --features-dir  $FEAT `
    --out           checkpoints_phoenix_concept\test_metrics.json

& $PY infer_slt.py --checkpoint checkpoints_phoenix_concept\best.pt --video frase.mp4
```

Detalhes das métricas em [EVALUATION.md](../../EVALUATION.md); inferência a
partir de vídeo em [INFERENCE.md](../../INFERENCE.md).

## 4. Conferir a instalação sem dados

Cada variante tem um *smoke test* (T5 minúsculo, dados aleatórios, CPU):

```powershell
& $PY slt_model.py --smoke-test --device cpu --encoder-type parts
& $PY slt_model.py --smoke-test --device cpu --encoder-type gcn --use-ctc
& $PY slt_model.py --smoke-test --device cpu --temporal-mode concept --ds-steps 2
```

Se aparecer `UnicodeEncodeError` no `print` final (o caractere ✔ no console
cp1252), rode antes `$env:PYTHONIOENCODING = "utf-8"`. Todas as verificações
anteriores à linha final já passaram.

## 5. Protocolo de comparação

1. **Uma mudança por vez** em relação à receita base, mesma `--seed`.
2. Reporte **BLEU-4 e WER** do conjunto de teste. O WER mostra se o encoder
   reconhece melhor os sinais; o BLEU, se isso vira tradução melhor. A v11
   mostrou que os dois podem andar separados.
3. Se possível, 2–3 sementes por braço: diferenças de ~0,5 BLEU estão dentro
   da variação entre execuções.
4. Guarde o `training_config.json` que o treino grava em `--out-dir`.

## 6. Testes

```powershell
& $PY tests\test_encoder_parts_lr.py
& $PY tests\test_encoder_gcn.py
& $PY tests\test_encoder_concept.py
```
