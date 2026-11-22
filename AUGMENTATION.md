# Data augmentation dos keypoints (`--aug-*`)

Quatro augmentations, **desligadas por padrão**, aplicadas **só ao conjunto de
treino**, a cada vez que uma frase é carregada (a mesma frase sai diferente em
cada época). Validação, avaliação e as estatísticas do `--input-norm` nunca
recebem augmentation.

O código fica em
[`skeltrans/training/data/augment.py`](skeltrans/training/data/augment.py).

## Motivação

O PHOENIX-2014T tem ~7 mil frases de treino: pouco para um encoder treinado do
zero. Augmentation cria variações plausíveis da mesma frase (outro ritmo, outro
ângulo de câmera, outra mão dominante) e ajuda contra o superajuste.

A receita segue o **SignCLIP** (Jiang et al., 2024), que usa o mesmo extrator
(MediaPipe Holistic) e a mesma normalização pelos ombros deste projeto. Lá, a
augmentation de velocidade foi uma das que mais ajudou.

## As augmentations

| Flag | O que faz | Valor de referência |
|---|---|---|
| `--aug-speed F` | reamostra a frase no tempo por interpolação, com fator sorteado em [1−F, 1+F]: a frase fica mais rápida (menos frames) ou mais lenta | `0.2` |
| `--aug-spatial S` | rotação (em radianos), escala e cisalhamento no plano XY, sorteados com desvio S, em torno do meio dos ombros | `0.2` (SignCLIP) |
| `--aug-noise S` | ruído gaussiano nas posições, com desvio S em unidades da largura dos ombros | `0.001` (SignCLIP) |
| `--aug-flip` | espelha metade das frases: inverte o X **e troca esquerda e direita** (mãos, pose e rosto) | — |

## Como funciona

1. A frase chega normalizada pelo Passo 3: `(T, 115, 9)`.
2. As augmentations agem só nas **posições** (canais 0–2).
3. **Velocidade e aceleração são recalculadas** com a mesma função do Passo 3,
   para os 9 canais continuarem coerentes (uma frase acelerada tem velocidades
   maiores, não as antigas).
4. As `--extra-features` são calculadas **depois**, sobre os pontos já aumentados.

**Pontos ausentes continuam ausentes** (`[0,0,0]`), o que mantém o grupo F e o
`--encoder-type gcn` corretos:

- espacial e espelhamento são lineares e sem translação: zero vai em zero;
- o ruído só é somado aos pontos presentes;
- na velocidade, um frame novo só tem o ponto se ele existe nos dois frames
  originais vizinhos (não interpola em direção ao zero).

**Sorteios por worker:** a semente de cada sorteio vem do gerador do torch, que
o DataLoader inicializa de forma diferente em cada worker (e que o `--seed`
fixa). O gerador global do numpy repetiria os mesmos sorteios em todos os
workers.

## Uso

```bash
# uma augmentation por vez, sobre a melhor config (mesmos hiperparâmetros no resto)
python3 slt_model.py ... --aug-speed 0.2                   --out-dir ck_aug_speed
python3 slt_model.py ... --aug-spatial 0.2                 --out-dir ck_aug_spatial
python3 slt_model.py ... --aug-noise 0.001                 --out-dir ck_aug_noise
python3 slt_model.py ... --aug-flip                        --out-dir ck_aug_flip

# combinadas
python3 slt_model.py ... --aug-speed 0.2 --aug-spatial 0.2 --aug-noise 0.001 --out-dir ck_aug_all
```

O log do treino mostra as augmentations ativas (`[augment] ...`), e elas ficam
gravadas no `training_config.json` e na configuração dos checkpoints.

## Cuidados

- **Velocidade e CTC:** acelerar encurta a frase. Com `--aug-speed 0.2` e
  `--ds-steps 2`, uma frase de 110 frames vira ~92, ou ~23 posições: ainda
  muito acima das ~7 glosas. Valores grandes (perto de 1) poderiam deixar
  frases curtas sem posições suficientes para o CTC.
- **Espelhamento e a língua de sinais:** espelhar transforma o sinalizador em
  canhoto. Os sinais continuam válidos, mas a mão dominante muda; é uma
  augmentation comum, porém vale testar isolada.
- **`--aug-spatial` em radianos:** 0,2 rad ≈ 11°.
- Mudam o treino, não a avaliação: compare sempre pelo dev/teste sem augmentation.

## Testes

```bash
python3 tests/test_augment.py
```

Cobre:

- **padrão:** sem augmentation, item idêntico ao original;
- **espelhamento:** permutação que troca esquerda/direita é uma involução (mãos,
  ombros, cantos da boca, olhos, sobrancelhas) e o espelho inverte o X;
- **invariantes:** ausentes continuam ausentes, velocidade/aceleração
  recalculadas, ruído só nos presentes, espacial linear sem mexer no Z;
- **velocidade:** comprimento dentro do esperado, fator 1 idêntico, falhas
  respeitadas na interpolação;
- **integração:** extras calculadas depois da augmentation, sorteio
  reprodutível pela semente do torch, flags da CLI e validações.
