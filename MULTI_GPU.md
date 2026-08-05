# Treino multi-GPU (DDP) — `torchrun`

O treino suporta várias GPUs na mesma máquina via **DistributedDataParallel
(DDP)**, o mecanismo padrão do PyTorch. Não existe flag nova: **o modo é
escolhido pelo comando de lançamento**, e o treino em 1 GPU continua
funcionando exatamente como sempre.

```bash
# 1 GPU (ou CPU) — comportamento original, nada mudou:
python3 slt_model.py --train-manifest train.json --val-manifest val.json \
    --out-dir checkpoints --low-vram --grad-checkpoint

# 2 GPUs — mesmo comando, trocando só o lançador:
torchrun --nproc_per_node=2 slt_model.py \
    --train-manifest train.json --val-manifest val.json \
    --out-dir checkpoints --low-vram --grad-checkpoint
```

O `torchrun` já vem instalado junto com o PyTorch — não há dependência nova.

---

## Como funciona

O DDP cria **um processo por GPU**, cada um com uma cópia completa do modelo.
A cada passo:

```
                    dataset (dividido pelo DistributedSampler)
                       │                          │
                 metade A do lote           metade B do lote
                       │                          │
                       ▼                          ▼
              ┌─────────────────┐        ┌─────────────────┐
              │  GPU 0: modelo  │        │  GPU 1: modelo  │
              │ forward+backward│        │ forward+backward│
              └────────┬────────┘        └────────┬────────┘
                       │      gradientes são      │
                       └──── promediados entre ───┘
                             as GPUs (all-reduce)
                       │                          │
                  optimizer.step()           optimizer.step()
                  (pesos idênticos)          (pesos idênticos)
```

Como os gradientes são sincronizados antes de cada `optimizer.step()`, os
pesos permanecem **idênticos** nas duas GPUs o treino inteiro — o resultado é
matematicamente equivalente a treinar em uma GPU com o lote dobrado.

## O que muda na prática

- **`--batch-size` passa a ser POR GPU.** Com `--batch-size 8` e 2 GPUs, o
  lote efetivo é 16. Para manter o lote efetivo de 8 de antes (e cortar a
  memória de ativações por GPU pela metade), use `--batch-size 4`.
- **Velocidade:** perto de 2× por época (menos um pequeno custo de
  sincronização dos gradientes via PCIe).
- **Memória por GPU:** as **ativações** caem proporcionalmente ao lote por
  GPU — é o que resolve OOM de sequências longas. Já **pesos + otimizador são
  replicados** em cada GPU (DDP não divide o modelo), então essa parte não
  diminui.
- **Logs e checkpoints:** só o processo 0 imprime e salva. Os `.pt` têm
  exatamente o mesmo formato de sempre (o estado salvo é o do `SLTModel`, sem
  o invólucro do DDP) — inferência e avaliação carregam sem mudança nenhuma.
- **`val_loss` e early stopping:** cada GPU avalia sua fatia da validação e a
  média é global (all-reduce); todos os processos veem o mesmo número e param
  juntos.
- **`--device` é ignorado** sob `torchrun` (cada processo usa a GPU do seu
  `LOCAL_RANK`). Para escolher QUAIS GPUs usar:
  `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ...`

## Combinação com as outras flags de memória

| Flag                | Em 1 GPU                          | Em DDP (2 GPUs)                                        |
|---------------------|-----------------------------------|--------------------------------------------------------|
| `--low-vram` bf16   | autocast bf16                     | igual (por processo)                                   |
| `--low-vram` AdamW 8-bit | estados 8-bit                | igual (por processo)                                   |
| `--low-vram` offload do encoder T5 | blocos vão p/ CPU  | **substituído por congelamento** (ver abaixo)          |
| `--grad-checkpoint` | recomputa ativações no backward   | igual (compatível com DDP)                             |

Sobre a linha do offload: o DDP exige que todos os parâmetros do modelo
estejam na GPU do processo, então os blocos do encoder do T5 não podem ir
para a CPU. Em vez disso eles são **congelados** (`requires_grad=False`) —
o que também é necessário porque o DDP exige que todo parâmetro treinável
receba gradiente, e esses blocos nunca executam. O congelamento é neutro:
esses pesos jamais recebiam gradiente mesmo. O custo é manter ~340 MB deles
parados em cada GPU (o preço de usar DDP).

## Receita recomendada para 2× TITAN V (12 GB)

```bash
pip install bitsandbytes   # uma vez, no ambiente de treino

torchrun --nproc_per_node=2 slt_model.py \
    --train-manifest train.json --val-manifest val.json \
    --out-dir checkpoints \
    --low-vram --grad-checkpoint --batch-size 4
```

`--batch-size 4` × 2 GPUs mantém o lote efetivo em 8 (mesma dinâmica de treino
dos seus experimentos anteriores, checkpoints comparáveis) com metade das
ativações em cada placa. Se sobrar folga no `nvidia-smi`, suba para
`--batch-size 8` (lote efetivo 16 — nesse caso vale acompanhar a `val_loss`,
pois a dinâmica de treino muda um pouco).

## Solução de problemas

| Sintoma | O que fazer |
|---|---|
| Trava/congela logo no início (antes da época 1) | Comunicação P2P entre as GPUs falhando: rode com `NCCL_P2P_DISABLE=1 torchrun ...` |
| `RuntimeError: ... find_unused_parameters` | Indica parâmetro sem gradiente; abra uma issue com o comando usado (o caso conhecido — CTC com peso 0 — já é tratado automaticamente) |
| Quer ver o log dos dois processos | `torchrun --nproc_per_node=2 --log-dir logs ...` grava stdout/stderr por rank |
| Época não embaralha diferente | Já tratado: o `DistributedSampler.set_epoch` é chamado a cada época |
