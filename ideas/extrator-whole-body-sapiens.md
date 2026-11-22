# Ideia: trocar o extrator de keypoints por um whole-body (Sapiens)

> **Status:** ideia registrada, ainda não implementada.
> **Motivação:** qualidade da extração com o MediaPipe no PHOENIX-2014T (nov/2026).
> **Objetivo:** pontos melhores (sobretudo das mãos), sem mudar nada do treino.

---

## 1. Por que considerar

O MediaPipe Holistic é o elo mais fraco da entrada:

- detecta as mãos em só **~74% dos frames** do PHOENIX, e em **~10%** não vê nenhuma das duas;
- as mãos têm ~35 px em imagens de 210 × 260, e ampliar a imagem não ajuda (experimento negativo no `documentations/PHOENIX.md`);
- nos melhores runs, a maior parte dos erros de glosa são **substituições** (o modelo vê o sinal, mas confunde qual é), o que aponta para a configuração de mão.

Extratores *whole-body* treinados em bases maiores tendem a ser mais robustos em mãos pequenas e borradas. É a alavanca com mais espaço que mantém a ideia do projeto (entrada só de esqueleto).

**Pré-requisito já feito:** as duas correções baratas do MediaPipe (estado do Holistic zerado entre vídeos e `--aspect`). Qualquer comparação deve ser contra o MediaPipe **corrigido**, senão o ganho do Sapiens fica superestimado.

---

## 2. O Sapiens (o que a documentação diz)

Fonte: `docs/POSE_README.md` do repositório `facebookresearch/sapiens`.

| Ponto | Sapiens |
|---|---|
| Formatos de pontos | **17** (corpo, COCO), **133** (corpo + rosto + mãos + pés, COCO-WholeBody), **308** (com rosto denso de 274 pontos) |
| Modelos | 0.3B, 0.6B, 1B e 2B parâmetros (o de 308 vai até 1B) |
| Saída | **2D**, um **JSON** por imagem |
| Entrada | **pasta de imagens**, via scripts de shell (`keypoints133.sh`, …) |
| Pesos | Hugging Face |

**Não informado na página** (verificar antes de começar): estrutura do JSON, resolução de entrada, detector de pessoa, dependências, memória de GPU, velocidade e **licença** (ver o `LICENSE` do repositório; se for não comercial, serve para pesquisa, mas registrar na tese).

---

## 3. Estratégia: trocar só a fonte dos pontos

O resto do projeto depende de um contrato único: arquivos **`(T, 115, 3)` na ordem de `skeltrans/common/layout.py`, com `[0, 0, 0]` = ausente**. Se o Sapiens produzir arquivos nesse formato, normalização, features extras, encoders, grafo, treino e avaliação funcionam sem alteração.

```
frames PNG ──► Sapiens (ambiente próprio, GPU) ──► JSON por frame
                                                      │
              tools/sapiens_to_landmarks.py  ◄────────┘   lê só JSON; não importa o Sapiens
                     │  mapeia 133 → 115, aplica limiar de confiança, ausente = 0
                     ▼
        landmarks_115_sapiens/<vídeo>.npy  (T, 115, 3)
                     │
                     ▼   s4 normalize_landmarks.py (o de hoje, SEM --aspect)
        landmarks_115_sapiens_norm9/  ──►  treino com --features-dir apontando para cá
```

- **Ambientes separados:** o Sapiens roda num ambiente próprio. O ambiente do servidor está fixado em torch 2.3.1 por causa das TITAN V; misturar seria arriscado.
- **O projeto ganha só um conversor**, sem dependência nova. Dados antigos e novos ficam em pastas diferentes.

---

## 4. Mapeamento 133 → 115 (começar pelo COCO-WholeBody)

Índices do COCO-WholeBody: 0–16 corpo, 17–22 pés, 23–90 rosto (68 pontos), 91–111 mão esquerda, 112–132 mão direita.

| Bloco do projeto | Origem no COCO-WholeBody |
|---|---|
| Mãos (21 + 21) | 91–111 e 112–132, **na mesma ordem** de 21 do MediaPipe (punho, depois cada dedo da base à ponta) |
| Lábios (20) | os 20 pontos de lábios do rosto de 68 (12 externos + 8 internos) |
| Sobrancelhas (5 + 5) | as 5 + 5 do rosto de 68 |
| Olhos (5 + 5) | 5 dos 6 pontos de cada olho |
| Pose (33) | ombros, cotovelos, pulsos, quadris, joelhos, tornozelos, calcanhares e pés direto; cantos de olhos e boca do rosto de 68; os pontos de mão da pose (17–22) a partir das mãos detalhadas; sem equivalente → ausente |

**Cuidados no mapeamento:**

- O rosto precisa de uma **tabela ponto a ponto**: o `FACE_MASK` usa índices do FaceMesh, e as features **C** e **E** dependem de pontos específicos (centros dos lábios 13/14, cantos 61/291, pálpebras 159/145 e 386/374, cantos dos olhos 33/133 e 362/263, sobrancelhas 105/334).
- Conferir a convenção de **esquerda/direita** (do ponto de vista da pessoa, como no MediaPipe).
- Validar com teste e **desenhando os pontos** sobre alguns frames.

O formato de **308** (rosto de 274 pontos) é interessante para as expressões não manuais, mas a lista de índices fica em `pose/configs/_base_/datasets/goliath.py` e precisa ser estudada. Segundo passo.

---

## 5. Cuidados que mudam o uso

| Ponto | Consequência |
|---|---|
| **Sem profundidade (z = 0)** | ângulos dos dedos (A) e distâncias (C) continuam válidos no plano; o **vetor normal da palma (B) degenera** → usar as extras **sem B** |
| **Sempre devolve todos os pontos** | "ausente" vem de um **limiar de confiança** no conversor (parâmetro a calibrar na amostra), para o grupo F e o GCN continuarem funcionando |
| **Coordenadas em pixels** | guardar dividido pela **altura** da imagem (eixos na mesma escala) → **não usar `--aspect`** |
| **Treino e inferência com o mesmo extrator** | um modelo treinado com pontos do Sapiens não funciona com pontos do MediaPipe; o `infer_slt.py` (vídeo → texto) precisaria do Sapiens antes. Avaliar no teste do PHOENIX não tem problema |
| **Custo** | centenas de milhões a bilhões de parâmetros × ~950 mil frames; nas TITAN V, pode levar **dias** com os modelos maiores → começar pelo 0.3B ou 0.6B e medir |

---

## 6. Plano

1. **Piloto no servidor:** ambiente próprio para o Sapiens, modelo de 133 pontos (0.3B ou 0.6B), `keypoints133.sh` em ~50 vídeos do PHOENIX. Anotar o tempo por frame.
2. **Conversor** `tools/sapiens_to_landmarks.py`, escrito **a partir de um JSON real do piloto** (o formato não está documentado):
   - mapeamento 133 → 115, limiar de confiança, ordem dos frames, frames sem pessoa;
   - testes com JSON sintético; sem importar o Sapiens.
   - Estimativa: ~1 dia depois de ter o JSON.
3. **Comparação na amostra** contra o MediaPipe **corrigido** (estimativa: ~meio dia para o script):
   - taxa de detecção das mãos (reaproveitar a lógica do `tools/phoenix_detection_report.py`);
   - tremor: deslocamento dos pontos da mão entre frames num trecho parado;
   - inspeção visual: pontos desenhados sobre os frames de ~5 vídeos.
4. **Se o ganho for claro:** extração completa → s4 → treino com a melhor config (v15), mudando só o `--features-dir` e tirando o **B** das extras.

---

## 7. Experimentos

| # | Experimento | Pergunta |
|---|---|---|
| 1 | Piloto: detecção, tremor e visual (MediaPipe corrigido × Sapiens 0.3B/0.6B) | O Sapiens vê melhor as mãos no PHOENIX? |
| 2 | Treino v15 com pontos do Sapiens (extras sem B) | A extração melhor reduz o WER e sobe o BLEU? |
| 3 | Limiar de confiança (2–3 valores) | Quanto do ganho depende de marcar bem os ausentes? |
| 4 | (depois) formato 308, rosto denso | O rosto detalhado ajuda nas expressões não manuais? |

Comparar **WER (com sub/del/ins)**, **BLEU-4** e **METEOR**, de preferência com 2–3 seeds. Espera-se que o ganho apareça primeiro nas **substituições** e **deleções** do WER.

---

## 8. Riscos

- **Custo de extração alto:** pode inviabilizar os modelos maiores nas TITAN V.
- **Domínio:** o Sapiens não foi avaliado em língua de sinais (a página não traz métrica de mãos); o piloto existe para descobrir isso antes de gastar a extração completa.
- **Mapeamento errado:** um ponto do rosto trocado não quebra o treino, só piora as features C/E em silêncio → teste + inspeção visual.
- **Perda do z:** se a profundidade estiver ajudando hoje, parte do ganho se perde; o experimento 2 mede o efeito líquido.

---

## 9. Alternativas

- **RTMPose / DWPose / RTMW** (MMPose, também COCO-WholeBody 133): mesmo conversor e mesmo mapeamento, modelos mais leves. A biblioteca `rtmlib` roda esses modelos via ONNX sem instalar o MMPose (conferir se continua mantida). Bom plano B se o Sapiens for pesado demais.
- **MediaPipe corrigido + melhorias de pós-processamento:** interpolar falhas curtas das mãos e suavizar no tempo antes das derivadas (velocidade/aceleração). Mais barato, ganho provavelmente menor.

---

## 10. Referências (conferir antes de citar)

- Khirodkar et al., *Sapiens: Foundation for Human Vision Models*, ECCV 2024 — repositório `facebookresearch/sapiens`, `docs/POSE_README.md`.
- Jin et al., *Whole-Body Human Pose Estimation in the Wild*, ECCV 2020 — o formato COCO-WholeBody (133 pontos).
- Jiang et al., *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose*, 2023 — alternativa mais leve.
