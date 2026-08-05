# `--low-vram` — economia de memória de GPU

Parâmetro opcional, **desligado por padrão**, disponível no treino
(`slt_model.py`), na inferência (`infer_slt.py`) e na avaliação
(`evaluate_slt.py`). Sem ele, nada muda: o comportamento é exatamente o
original. Com ele, três otimizações de memória são ativadas de uma vez —
liberando tipicamente **1,5 a 2,5 GB de VRAM no treino** e **mais da metade
da memória de pesos na inferência**, sem alterar a arquitetura nem o formato
dos checkpoints.

> Se o treino ainda estourar a memória (OOM) com `--low-vram`, veja também a
> flag complementar [`--grad-checkpoint`](#e-quando-o-oom-continua--grad-checkpoint-só-no-treino),
> que ataca as ativações de sequências longas.

```bash
# treino
python3 slt_model.py --train-manifest train.json --val-manifest val.json \
    --epochs 30 --batch-size 8 --out-dir checkpoints --low-vram

# inferência
python3 infer_slt.py --checkpoint checkpoints/best.pt --video frase.mp4 --low-vram

# avaliação
python3 evaluate_slt.py --checkpoint checkpoints/best.pt \
    --test-manifest test.json --low-vram
```

---

## Por que o modelo gasta tanta VRAM?

O `SLTModel` soma **~244M parâmetros**: o PTT5-base inteiro (~223M) mais o
`LandmarkEncoder` treinado do zero (~21M). Em fp32 (o padrão do PyTorch),
cada parâmetro ocupa 4 bytes — mas no **treino** o custo real é bem maior,
porque cada parâmetro treinável carrega três "sombras":

| Componente                        | Bytes por parâmetro | Para quê serve                          |
|-----------------------------------|--------------------:|-----------------------------------------|
| Peso (fp32)                       | 4                   | o próprio modelo                        |
| Gradiente (fp32)                  | 4                   | resultado do backward                   |
| Estado 1 do AdamW (momento, fp32) | 4                   | média móvel dos gradientes              |
| Estado 2 do AdamW (variância, fp32)| 4                  | média móvel dos gradientes ao quadrado  |

Ou seja: **16 bytes por parâmetro treinável**, antes mesmo de processar o
primeiro batch. Somam-se a isso as **ativações** (os tensores intermediários
guardados durante o forward para calcular o backward), que crescem com o
`--batch-size` e com o comprimento `T` das sequências de landmarks.

O `--low-vram` ataca três dessas frentes, uma por otimização.

---

## Otimização 1 — Offload do encoder do T5 (o "passageiro fantasma")

**A observação-chave:** nesta arquitetura, o encoder do PTT5 **nunca executa**.

Num T5 normal, o fluxo é `texto → encoder T5 → decoder T5`. Aqui, quem faz o
papel de encoder é o `LandmarkEncoder` (que lê os esqueletos), e sua saída é
injetada diretamente no decoder via `encoder_outputs`:

```
fluxo normal do T5:      texto ──► encoder T5 ──► decoder T5 ──► texto
fluxo do SLTModel:    landmarks ──► LandmarkEncoder ──► decoder T5 ──► texto
                                                   ▲
                                    encoder T5 fica ocioso na GPU!
```

Quando `encoder_outputs` já vem pronto, o HuggingFace **pula o encoder do T5
completamente** — tanto no `forward` (treino) quanto no `generate`
(inferência). Mas os ~84M parâmetros dos 12 blocos dele (~340 MB em fp32)
continuavam ocupando VRAM à toa.

Com `--low-vram`, esses blocos são movidos para a RAM da CPU. Como nunca são
executados, **o resultado é bit a bit idêntico** — é economia de graça.

> Detalhe de implementação: só os *blocos* do encoder saem da GPU. A tabela de
> embeddings (`t5.shared`) fica, porque é o mesmo objeto usado pelo decoder.

**Economia: ~340 MB (fp32) / ~170 MB (bf16). Efeito na qualidade: nenhum.**

## Otimização 2 — bfloat16 (números pela metade)

Todo tensor fp32 ocupa 4 bytes por número. O formato **bfloat16 (bf16)** usa
2 bytes, mantendo a mesma faixa de valores do fp32 (mesmo expoente, menos
precisão na mantissa). Isso o torna muito mais estável que o fp16 clássico —
relevante aqui, porque o T5 é conhecido por gerar `NaN` em fp16.

O parâmetro aplica bf16 de forma diferente em cada modo:

- **Treino:** usa *autocast* — os pesos continuam em fp32 (o otimizador
  precisa da precisão), mas o forward/backward roda em bf16. A maior economia
  é nas **ativações**, justamente a parte que cresce com o batch. Na prática,
  quase metade da memória de ativações, com bônus de velocidade (tensor cores).
- **Inferência:** converte os **próprios pesos** para bf16 — sem otimizador e
  sem gradientes, isso é seguro e corta a memória de pesos pela metade. O KV
  cache do beam search também cai pela metade.

**Requisito:** GPU NVIDIA Ampere ou mais nova (RTX 30xx/40xx, A10G, A100).
Em GPUs sem suporte (ex.: T4) ou na CPU, o código detecta, avisa
(`[low-vram] dispositivo sem suporte a bf16...`) e segue em fp32 — as outras
duas otimizações continuam valendo.

**Economia: ~50% das ativações (treino) / ~50% dos pesos (inferência).
Efeito na qualidade: desprezível (bf16 é o padrão de treino de LLMs modernos).**

## Otimização 3 — AdamW 8-bit (só no treino)

Como a tabela lá em cima mostra, **metade** dos 16 bytes/parâmetro do treino
são os dois estados do AdamW. A biblioteca
[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
implementa um AdamW que guarda esses estados **quantizados em 8 bits**
(1 byte em vez de 4 por estado), fazendo a dequantização por bloco na hora do
update. É a parte do `--low-vram` que é *quantização* no sentido estrito.

No modelo, ~160M parâmetros recebem gradiente (decoder + embeddings +
`LandmarkEncoder`): os estados do otimizador caem de ~1,28 GB para ~0,32 GB.

**Requisito:** `pip install bitsandbytes`. Se não estiver instalado, o código
avisa e usa o AdamW padrão — o treino funciona igual, sem essa economia.

**Economia: ~0,95 GB. Efeito na qualidade: desprezível (a literatura do
bitsandbytes mostra paridade com o Adam fp32).**

---

## Resumo da economia

Estimativas para PTT5-base + LandmarkEncoder padrão (batch 8):

| Onde        | O quê                       | Sem `--low-vram` | Com `--low-vram` |
|-------------|-----------------------------|-----------------:|-----------------:|
| Treino      | Encoder T5 ocioso           | ~340 MB          | 0 (vai p/ CPU)   |
| Treino      | Estados do AdamW            | ~1,28 GB         | ~0,32 GB         |
| Treino      | Ativações (forward/backward)| ~100%            | ~50–60% (bf16)   |
| Inferência  | Pesos do modelo             | ~0,98 GB         | ~0,32 GB         |
| Inferência  | KV cache / beam search      | ~100%            | ~50% (bf16)      |

Na prática: **~1,5 a 2,5 GB liberados no treino** (dependendo do batch) e
**~0,7 GB + metade das ativações na inferência**.

## O que NÃO muda

- **Padrão desligado:** sem `--low-vram`, nenhuma linha nova executa.
- **Arquitetura:** nenhuma camada é adicionada, removida ou congelada.
- **Checkpoints:** mesmo formato de sempre. Um checkpoint treinado com
  `--low-vram` carrega sem a flag, e vice-versa (os pesos são salvos/carregados
  em fp32 normalmente; a conversão bf16 da inferência acontece só em memória).
- **Compatibilidade dos comandos:** é uma flag aditiva; todos os outros
  parâmetros funcionam igual.

## E quando o OOM continua? — `--grad-checkpoint` (só no treino)

Se mesmo com `--low-vram` o treino estoura a memória, o culpado quase sempre
são as **ativações da atenção com sequências longas**. Cada uma das 6 camadas
do `LandmarkEncoder` guarda, para o backward, uma matriz de atenção de formato
`(B × nhead, T, T)` — e `T` é o número de *frames* da maior frase do lote.
O custo cresce com o **quadrado** de `T`:

| T (frames) | Uma matriz de atenção (batch 8, 8 heads, bf16) | × 6 camadas |
|-----------:|-----------------------------------------------:|------------:|
| 500        | ~32 MB                                          | ~0,2 GB     |
| 1500       | ~288 MB                                         | ~1,7 GB     |
| 3000       | ~1,15 GB                                        | ~6,9 GB     |

Ou seja: um único lote com frases de ~3000 frames consome mais VRAM em
ativações do que o modelo inteiro em pesos + otimizador. É exatamente o perfil
do erro `Tried to allocate 1.23 GiB` com quase 10 GB já ocupados.

A flag `--grad-checkpoint` resolve trocando memória por computação: as
ativações **não são guardadas** — no backward, cada camada refaz seu forward
para recomputá-las na hora. Aplica-se ao `LandmarkEncoder` e ao decoder T5.

```bash
python3 slt_model.py --train-manifest train.json --val-manifest val.json \
    --out-dir checkpoints --low-vram --grad-checkpoint
```

- **Economia:** a VRAM de ativações cai para uma fração pequena (fica ~1 camada
  em vez de todas); é a otimização mais eficaz contra OOM de sequência longa.
- **Custo:** ~25–30% mais lento por época (um forward extra no backward).
- **Qualidade:** nenhuma — os gradientes são matematicamente idênticos.
- **Por que é uma flag separada do `--low-vram`?** Porque tem custo real de
  velocidade; as três otimizações do `--low-vram` são praticamente de graça.
- Só tem efeito no treino; na inferência não há backward nem ativações guardadas.

## Perguntas frequentes

**Isso reduz a qualidade da tradução (BLEU/METEOR)?**
O offload do encoder é matematicamente neutro. bf16 e AdamW 8-bit introduzem
diferenças numéricas minúsculas — na prática, dentro da variação normal entre
duas execuções de treino. Se quiser conferir, treine uma vez com e uma vez sem
a flag e compare a `val_loss`/BLEU.

**Minha GPU é uma T4 / GTX 16xx. Vale a pena?**
Sim: você perde só a parte do bf16 (a flag avisa e mantém fp32), mas ganha o
offload do encoder e o otimizador 8-bit — juntos, ~1,3 GB no treino.

**Preciso instalar algo?**
Só o `bitsandbytes`, e apenas se quiser o otimizador 8-bit no treino. Offload
e bf16 usam só o PyTorch.

**Ainda estou sem VRAM suficiente. E agora?**
Na ordem: (1) adicione `--grad-checkpoint` (seção acima); (2) reduza o
`--batch-size` (4 ou 2); (3) instale o `bitsandbytes` se ainda não instalou;
(4) se a máquina tiver mais de uma GPU, treine com DDP — o lote é dividido
entre as placas e as ativações por GPU caem na proporção (ver
[`MULTI_GPU.md`](MULTI_GPU.md)). Depois disso, os próximos passos (mais
invasivos, fora do escopo destas flags) seriam acúmulo de gradiente e
LoRA/QLoRA — congelar o PTT5 quantizado em 4 bits e treinar só adaptadores +
o `LandmarkEncoder`.
