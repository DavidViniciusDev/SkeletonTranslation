# Timeline do SkeLTrans

Histórico das ideias implementadas no framework: **o quê**, **quando**, **por
quê** e, quando houver, **o efeito medido**. A fonte das datas é o git; os
resultados são os do conjunto de **teste** do PHOENIX-2014T informados em cada
treino.

**Como atualizar:** a cada alteração no framework, acrescente uma entrada no
**topo** da seção 1, com:

```
### AAAA-MM-DD · Título curto
- **Tipo:** Funcionalidade | Correção | Ferramenta | Infraestrutura | Documentação | Ideia
- **Commit:** `abc1234`
- **O quê:** o que mudou (flags, arquivos principais).
- **Por quê:** o problema ou a hipótese.
- **Padrão:** se muda o comportamento sem flags (ou "desligado; sem a flag, nada muda").
- **Resultado:** efeito medido, quando houver (ou "a medir").
- **Docs:** onde está documentado.
```

Resultados de treino novos entram também na tabela da seção 2.

---

## 1. Linha do tempo (mais recente primeiro)

### 2026-11-23 · Página de comandos de treinamento
- **Tipo:** Documentação
- **Commit:** a definir (ainda não commitado)
- **O quê:** `docs/comandos-treinamento.md` reúne, em PowerShell e bash, os comandos de cada módulo (receita base v15, `flat`, `parts`, `gcn`, `concept`, pré-treino, augmentation, multi-GPU, avaliação, média de checkpoints, ajuste da decodificação e inferência). A receita base de `docs/encoders/README.md` passou da v11 para a v15.
- **Por quê:** ter num lugar só tudo o que é preciso para rodar cada experimento.
- **Padrão:** só documentação.
- **Docs:** `docs/comandos-treinamento.md`.

### 2026-11-23 · Ideia: encoder em múltiplos fluxos com fusão tardia
- **Tipo:** Ideia
- **Commit:** a definir (ainda não commitado)
- **O quê:** nota `ideas/encoder-multifluxo-fusao-tardia.md`: um encoder temporal para as mãos e outro para corpo + rosto, cada um com CTC auxiliar, fundidos antes do T5; plano para implementar atrás de `--encoder-streams` sem mudar o padrão.
- **Docs:** pasta `ideas/` (ver seção 3).

### 2026-11-23 · Pré-treino do encoder por masked pose modeling
- **Tipo:** Funcionalidade
- **Commit:** a definir (ainda não commitado)
- **O quê:** `pretrain_encoder.py` (código em `skeltrans/training/pretrain.py` e `models/masked_pose.py`) esconde trechos dos keypoints (`--mask-mode frames|hands|mixed`, `--mask-ratio`, `--mask-span`) e treina o `LandmarkEncoder` a reconstruí-los, sem glosas nem traduções. Gera `encoder.pt` (só o encoder + spec + estatísticas do `--input-norm`; lido com `weights_only=True`). No SLT, `--init-encoder <encoder.pt>` carrega esses pesos como ponto de partida, conferindo antes que `--extra-features`, `--input-norm`, `--encoder-type`, `--d-model`, `--nhead`, `--num-layers` e `--ds-steps` são os mesmos. Novo `KeypointDataset` (manifestos sem texto).
- **Por quê:** ~7 mil frases é pouco para um encoder que começa do zero; reconstruir keypoints escondidos ensina a estrutura do movimento, o que deve reduzir as substituições (≈23 dos 37,5 de WER da v15). Permite usar keypoints sem rótulo de outros datasets.
- **Padrão:** desligado; sem `--init-encoder`, o treino é o mesmo de antes. Avaliação e inferência não mudam (o `best.pt` contém tudo).
- **Resultado:** a medir (comparar com a v15, mesma receita e semente).
- **Docs:** `docs/encoders/pretrain.md`; testes em `tests/test_pretrain_encoder.py` (19).

### 2026-11-22 · Tour guiado pelo código
- **Tipo:** Ferramenta
- **Commit:** `208754c`
- **O quê:** `tools/tour_um_lote.py` passa um lote pelo código real e imprime formas, máscaras e perdas em cada etapa (Dataset, collate, encoder, T5, CTC, otimizador, geração). Roda sem dados (`--sintetico`) ou com um manifesto real.
- **Por quê:** acompanhar e explicar o que o código faz, lendo-o pelo caminho de um lote.
- **Padrão:** ferramenta à parte; não muda o treino.
- **Docs:** guia "Tour guiado pelo código" (biblioteca Explicador).

### 2026-11-22 · Data augmentation dos keypoints
- **Tipo:** Funcionalidade
- **Commit:** `a499026`
- **O quê:** `--aug-speed`, `--aug-spatial`, `--aug-noise`, `--aug-flip` (`skeltrans/training/data/augment.py`), só no treino. Recalcula velocidade/aceleração, preserva ausentes, roda antes das `--extra-features`.
- **Por quê:** ~7 mil frases de treino é pouco para um encoder do zero; receita do SignCLIP, com o mesmo extrator e a mesma normalização.
- **Padrão:** desligado; sem as flags, cada frase sai idêntica.
- **Resultado:** a medir (sugestão: uma por vez sobre o v15, começando por `--aug-speed 0.2`).
- **Docs:** `AUGMENTATION.md`.

### 2026-11-22 · Correções da extração: estado do Holistic e proporção da imagem
- **Tipo:** Correção
- **Commit:** `5b0d086`
- **O quê:** (1) o estado do MediaPipe Holistic passa a ser zerado antes de cada vídeo (s3 e `infer_slt.py`); (2) `--aspect` (s4, s5, infer) leva X e Z à escala de Y antes de normalizar.
- **Por quê:** (1) o rastreamento e a suavização vazavam de um vídeo para o seguinte, e o resultado dependia do vídeo anterior, do `--workers` e da ordem; (2) o MediaPipe divide X pela largura e Y pela altura, o que distorce ângulos e distâncias no PHOENIX (210 × 260).
- **Padrão:** (1) muda a extração (os `.npy` antigos têm o problema; reextrair para corrigir); (2) desligado, saída idêntica byte a byte sem a flag.
- **Resultado:** a medir (renormalizar com `--aspect 210x260`; depois reextrair).
- **Docs:** `documentations/PHOENIX.md` (seções 6.3 e 6.4).

### 2026-11-22 · Ideias: extrator whole-body e alinhamento contrastivo
- **Tipo:** Ideia
- **Commit:** `76fbd1b`
- **O quê:** notas `ideas/extrator-whole-body-sapiens.md` e `ideas/alinhamento-contrastivo-pose-texto.md`.
- **Docs:** pasta `ideas/` (ver seção 3).

### 2026-11-10 · Encoder por conceitos
- **Tipo:** Funcionalidade
- **Commit:** `ced4880`
- **O quê:** `--temporal-mode concept` (+ `--concept-layers`, `--concept-queries`, `--concept-boundaries`, `--concept-greedy-prob`): uma posição de memória por sinal, com fronteiras dadas pelo CTC (alinhamento forçado no treino, greedy na inferência).
- **Por quê:** aproximar a memória do decoder de "um vetor por sinal", como as glosas.
- **Padrão:** `stride` (o original), bit a bit idêntico.
- **Resultado:** 1º treino (lr 3e-5, label smoothing 0,1, K=1, 4 + 2 camadas): BLEU-4 17,98, WER 38,25 (del 12,15). Contra a v16 (`stride`, mesma config): −0,78 de BLEU-4 e mais deleções (10,6 → 12,15). Com K=1, não ajudou; faltam as ablações K=2, greedy-prob e 6 camadas de frames.
- **Docs:** `docs/encoders/concept.md`.

### 2026-11-10 · Diagramas da arquitetura e ideia de pré-treino Gloss2Text
- **Tipo:** Documentação + Ideia
- **Commit:** `60fd917`
- **O quê:** diagramas draw.io (arquitetura versões 1 e 2 e exemplos didáticos) em `documentations/`; nota `ideas/pretreino-decoder-gloss2text.md`.

### 2026-11-09 · Seed, label smoothing, média de checkpoints e ajuste da decodificação
- **Tipo:** Funcionalidade + Ferramenta
- **Commit:** `767c04a`
- **O quê:** `--seed`; `--label-smoothing`; `tools/average_checkpoints.py`; `tools/tune_decoding.py`; `--length-penalty` e `--no-repeat-ngram-size` na avaliação e na inferência.
- **Por quê:** reprodutibilidade, regularização da tradução e ganhos "de graça" na geração.
- **Padrão:** treino bit a bit idêntico com os padrões.
- **Resultado:** v15 (com `--label-smoothing 0.1` e lr do decoder 5e-5): **BLEU-4 19,22**, maior salto até então (+1,5).
- **Docs:** `EVALUATION.md`, `INSTRUCTIONS.md`.

### 2026-10-28 · Encoder por grafo do esqueleto
- **Tipo:** Funcionalidade
- **Commit:** `b7d1a34`
- **O quê:** `--encoder-type gcn`: 115 nós, 128 arestas escritas à mão, 3 partições (ST-GCN), grafo aprendido, pooling por parte. Ausentes não diluem vizinhos; roda só nos frames reais.
- **Por quê:** dar ao encoder a estrutura do esqueleto (forma da mão), atacando as substituições do WER.
- **Padrão:** `flat` (o original).
- **Resultado:** não melhorou o BLEU frente ao `flat`.
- **Docs:** `ENCODER_GCN.md`.

### 2026-10-24 · Lr separado e encoder por partes
- **Tipo:** Funcionalidade
- **Commit:** `f2a92f2`
- **O quê:** `--encoder-lr` (lr próprio para encoder + CTC); `--encoder-type parts` (uma MLP por parte do corpo + fusão).
- **Por quê:** o encoder começa do zero e o decoder é pré-treinado; partes do corpo com redes próprias.
- **Padrão:** um lr só e `flat`; bit a bit idêntico.
- **Resultado:** lr separado entrou no v10 (BLEU-4 17,29); `parts` não melhorou o BLEU.
- **Docs:** `ENCODER_PARTS.md`.

### 2026-10-19 · Critério de seleção por métrica
- **Tipo:** Funcionalidade
- **Commit:** `5312063`
- **O quê:** `--select-by {val_loss,bleu1..4,meteor}`; `best_loss.pt` para comparação; `--patience` em épocas desde a melhor avaliação; broadcast da métrica no DDP.
- **Por quê:** o menor val_loss não é o maior BLEU, e o early stopping pela loss cortava treinos com o BLEU ainda subindo.
- **Padrão:** `val_loss` (o original).
- **Docs:** `INSTRUCTIONS.md`, `MULTI_GPU.md`.

### 2026-10-19 · Estatísticas das features
- **Tipo:** Ferramenta
- **Commit:** `3926b37`
- **O quê:** `tools/landmark_stats.py`: distribuição de T, simulação de padding por lote e magnitude por grupo/canal.

### 2026-10-19 · Features extras e normalização da entrada
- **Tipo:** Funcionalidade
- **Commit:** `17636ad`
- **O quê:** `--extra-features` (grupos A–F: configuração de mão, orientação da palma, ponto de articulação, movimento, não manuais, presença) e `--input-norm` (z-score com estatísticas do treino, gravado no `.pt`).
- **Por quê:** entregar ao modelo os parâmetros fonológicos prontos e pôr as colunas na mesma escala.
- **Padrão:** desligado; entrada e `state_dict` idênticos.
- **Resultado:** v10 (A,B,C) BLEU-4 17,29 / WER 43,5; v11 (A,B,C,F + input-norm) 17,81 / 40,0; A–F + input-norm 17,72 / 38,5.
- **Docs:** `EXTRA_FEATURES.md`.

### 2026-10-13 · BLEU durante o treino e dump da configuração
- **Tipo:** Funcionalidade
- **Commit:** `19325aa`
- **O quê:** `--eval-every`, `--eval-beams`, `--eval-max-samples` (`eval_history.jsonl`); `training_config.json` com comando, versões, commit e hardware.
- **Por quê:** acompanhar a métrica de tradução, e não só a loss; registrar exatamente como cada treino foi feito.
- **Docs:** `INSTRUCTIONS.md`.

### 2026-10-08 · Correções e documentação do PHOENIX
- **Tipo:** Correção + Documentação
- **Commit:** `e530bb3`
- **O quê:** ajustes em checkpoint/modelo; `documentations/PHOENIX.md` (adaptação, ambiente, passo a passo) e `EVALUATION.md`.

### 2026-09-29 · Downsampling temporal no encoder
- **Tipo:** Funcionalidade
- **Commit:** `e2b5921`
- **O quê:** `--ds-steps` (convoluções stride 2 antes do Transformer: T → T/2ⁿ), com a máscara reduzida propagada para o T5 e o CTC.
- **Por quê:** frases longas e redundantes; atenção mais barata; proporção frames/palavras mais próxima do pré-treino do T5.
- **Resultado:** `--ds-steps 8` colapsava as frases do PHOENIX para 1 posição (BLEU-4 5,4, WER 99); com `--ds-steps 2`, BLEU-4 11,38.

### 2026-09-29 · Suporte ao PHOENIX-2014T
- **Tipo:** Funcionalidade + Ferramenta
- **Commit:** `3af9b2c`
- **O quê:** `tools/phoenix_prepare.py` (manifestos com `text` e `tokens`), `tools/phoenix_detection_report.py`, WER de glosas (`--wer`) e BERTScore (`--bertscore`) na avaliação.
- **Por quê:** levar o pipeline do V-LIBRASIL ao PHOENIX, base com resultados comparáveis na literatura.

### 2026-08-05 · Treino multi-GPU (DDP)
- **Tipo:** Infraestrutura
- **Commit:** `67bd932`
- **O quê:** `torchrun` → um processo por GPU, DistributedSampler, val_loss global, salvamento só no rank 0.
- **Padrão:** `python3` continua em 1 GPU.
- **Docs:** `MULTI_GPU.md`.

### 2026-08-05 · Gradient checkpointing
- **Tipo:** Infraestrutura
- **Commit:** `c4641bf`
- **O quê:** `--grad-checkpoint` no LandmarkEncoder e no decoder T5.
- **Por quê:** falta de memória por ativações de atenção em sequências longas.
- **Docs:** `LOW_VRAM.md`.

### 2026-08-05 · Economia de memória de GPU
- **Tipo:** Infraestrutura
- **Commit:** `e853104`
- **O quê:** `--low-vram`: offload do encoder T5, bf16 e AdamW 8-bit.
- **Docs:** `LOW_VRAM.md`.

### 2026-07-29 · Estrutura do projeto
- **Tipo:** Funcionalidade
- **Commit:** `1ad2940` (e `483d498`, README)
- **O quê:** pipeline de dados do V-LIBRASIL (s1–s5: vocabulário, filtro, mapeamento sentença → vídeos, extração MediaPipe, normalização, síntese de frases por keyframe blending) e o modelo SLT (LandmarkEncoder + decoder T5, CTC opcional, early stopping, avaliação e inferência).

---

## 2. Marcos de resultado (PHOENIX-2014T, teste)

| Data | Treino | Mudança principal | BLEU-4 | WER |
|---|---|---|---|---|
| ~2026-10-15 | v7 | `--ds-steps 8` (frases colapsadas) | 5,4 | 99,3 |
| 2026-10-19 | linha de base | `--ds-steps 2`, lote 64, `--max-text-len 64` | 11,38 | — |
| 2026-10-20 | — | (config não registrada) | 16,51 | 45 |
| 2026-10-25 | v10 | extras A,B,C · CTC 1,0 · `--encoder-lr` · `--select-by bleu4` | 17,29 | 43,46 |
| 2026-10-26 | v11 | + F · `--input-norm` | 17,81 | 40,01 |
| 2026-11-09 | — | extras A–F · `--input-norm` | 17,72 | 38,53 |
| 2026-11-11 | **v15** | lr do decoder 5e-5 · `--label-smoothing 0.1` | **19,22** | 37,5 |
| 2026-11-13 | v16 | v15 com `--lr 3e-5` | 18,76 | 37,97 |
| 2026-11-13 | concept v1 | `--temporal-mode concept` (lr 3e-5) | 17,98 | 38,25 |

Padrão observado até nov/2026: o WER cai a cada treino, mas o BLEU-4 cresce
pouco; o gargalo passou para o lado da tradução (decoder).

---

## 3. Ideias registradas (ainda não implementadas)

| Data | Ideia | Nota |
|---|---|---|
| 2026-11-10 | Pré-treino do decoder em Gloss2Text (e medição do teto do decoder) | `ideas/pretreino-decoder-gloss2text.md` |
| 2026-11-22 | Extrator whole-body (Sapiens / RTMPose) convertido para o formato de 115 pontos | `ideas/extrator-whole-body-sapiens.md` |
| 2026-11-22 | Alinhamento contrastivo entre pose e texto (SignCLIP) | `ideas/alinhamento-contrastivo-pose-texto.md` |
| 2026-11-23 | Encoder em múltiplos fluxos (mãos × corpo + rosto) com fusão tardia e CTC por fluxo | `ideas/encoder-multifluxo-fusao-tardia.md` |

Quando uma ideia for implementada, ela ganha uma entrada na seção 1 e sai desta tabela.
