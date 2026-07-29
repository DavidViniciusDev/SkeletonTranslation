# Inferência — vídeo de Libras → texto em Português

Como usar um modelo SLT **já treinado** para traduzir um vídeo `.mp4` de uma
frase sinalizada em Libras para texto em Português (pt-br).

> Etapa 2 do projeto (modelo). Se você ainda não treinou, veja
> [`STRUCTURE.md`](STRUCTURE.md) (seção *Etapa 2 — modelo / treino*).

---

## 1. O que a inferência faz

`infer_slt.py` executa **todo o pipeline ponta-a-ponta** sobre cada vídeo, sem
exigir nenhum pré-processamento manual:

```
vídeo .mp4
   │  1. MediaPipe Holistic  (reaproveita a extração da Etapa 1, s3)
   ▼
landmarks (T, 115, 3)
   │  2. normalização geométrica + dinâmica  (Etapa 1, s4)
   ▼
features (T, 115, 9) → achatadas em (T, 1035)
   │  3. LandmarkEncoder + decoder PTT5  (Etapa 2)  — geração com beam search
   ▼
texto em Português
```

> **Sem *keyframe blending*.** O vídeo de entrada já é uma sentença contínua
> real; o *blending* de keyframes existe apenas para **sintetizar** frases a
> partir de sinais isolados durante o treino, e por isso **não** é aplicado aqui.

O código real vive em [`skeltrans/training/infer.py`](skeltrans/training/infer.py);
`infer_slt.py` na raiz é apenas um *shim* que preserva o comando.

---

## 2. Pré-requisitos

- Um **checkpoint treinado** (ex.: `checkpoints/best.pt`).
- O **tokenizer** — o treino já o salva no diretório do checkpoint (`best.pt`,
  `spiece.model`, `tokenizer_config.json`, …). A inferência o encontra
  automaticamente; nada a fazer.
- **MediaPipe** e **OpenCV** instalados (necessários para ler o vídeo e extrair
  landmarks):

  ```bash
  pip install mediapipe opencv-python
  ```

  As demais dependências (torch, transformers) são as mesmas do treino.

---

## 3. Uso

### Um único vídeo

```bash
python3 infer_slt.py \
    --checkpoint checkpoints/best.pt \
    --video /caminho/para/frase.mp4
```

### Vários vídeos de uma vez

```bash
python3 infer_slt.py \
    --checkpoint checkpoints/best.pt \
    --video a.mp4 b.mp4 c.mp4
```

Equivalente, pelo módulo (sem o shim):

```bash
python -m skeltrans.training.infer --checkpoint checkpoints/best.pt --video frase.mp4
```

### Saída esperada

```
Dispositivo: cuda
Checkpoint: checkpoints/best.pt | T5: unicamp-dl/ptt5-base-portuguese-vocab

[frase.mp4]  (87 frames)
  -> o menino gosta de jogar bola
```

Se um vídeo falhar (arquivo ausente, sem frames legíveis, etc.), o erro é
reportado **por vídeo** e o processamento segue para os demais.

---

## 4. Argumentos

| Argumento          | Padrão        | Descrição |
|--------------------|---------------|-----------|
| `--checkpoint`     | (obrigatório) | Caminho do `.pt` treinado. |
| `--video`          | (obrigatório) | Um ou mais caminhos de vídeo `.mp4`. |
| `--origin`         | `shoulders`   | Origem da centralização na normalização. **Deve casar com o valor usado no treino** (`shoulders` ou `nose`). |
| `--num-beams`      | `4`           | Largura do *beam search* na geração. |
| `--max-new-tokens` | `64`          | Máximo de tokens gerados por frase. |
| `--t5`             | (auto)        | Override do identificador/checkpoint do decoder T5. Por padrão usa o que está gravado no checkpoint. |
| `--tokenizer`      | (auto)        | Override do tokenizer. Por padrão: diretório do checkpoint, senão o T5. |
| `--device`         | (auto)        | `cuda` ou `cpu`. Se omitido, usa CUDA quando disponível. |

> **Atenção à `--origin`.** A arquitetura é reconstruída a partir dos
> hiperparâmetros gravados no checkpoint, mas a **origem da normalização** não é
> gravada. Use o mesmo valor da extração/treino (o padrão do projeto é
> `shoulders`), senão as features ficam inconsistentes e a tradução degrada.

---

## 5. Como saber se está correto

- Rode primeiro sobre um vídeo do **conjunto de teste** cuja tradução de
  referência você conhece e compare a saída.
- Para uma medida quantitativa (BLEU/METEOR) sobre um conjunto inteiro, use a
  **avaliação** em vez da inferência vídeo a vídeo — veja
  [`EVALUATION.md`](EVALUATION.md).

---

## 6. Solução de problemas

| Sintoma | Causa provável / correção |
|---------|---------------------------|
| `mediapipe nao instalado` | `pip install mediapipe opencv-python`. |
| `nao foi possivel abrir o video` | Caminho errado ou codec não suportado pelo OpenCV/ffmpeg. |
| `video sem frames legiveis` | Vídeo corrompido ou vazio. |
| Tradução sem sentido | Confirme `--origin` igual ao do treino e que `--checkpoint` é o `best.pt` correto. |
| Muito lento na CPU | Use `--device cuda`; o gargalo costuma ser o MediaPipe, que roda na CPU. |
