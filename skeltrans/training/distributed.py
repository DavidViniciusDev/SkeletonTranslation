"""Suporte a treino multi-GPU com DistributedDataParallel (DDP).

O modo é escolhido pelo LANÇADOR, não por flag de código:

    # 1 GPU (ou CPU) — comportamento original, nada muda:
    python3 slt_model.py --train-manifest ...

    # N GPUs na mesma máquina — 1 processo por GPU:
    torchrun --nproc_per_node=2 slt_model.py --train-manifest ...

O torchrun exporta RANK/WORLD_SIZE/LOCAL_RANK no ambiente de cada processo;
quando essas variáveis existem, o build_and_train entra no grupo NCCL e cada
processo treina na sua GPU. Sem elas, todas as funções deste módulo viram
no-ops e o fluxo é idêntico ao de sempre. Detalhes em MULTI_GPU.md.
"""

import os

import torch
import torch.distributed as dist


def is_distributed():
    """True quando o processo foi lançado pelo torchrun com mais de 1 processo."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed():
    """Entra no grupo NCCL e devolve (device, rank, world_size).

    Cada processo fica preso à GPU do seu LOCAL_RANK (0, 1, ...).
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size()


def is_main():
    """True no processo 0 (ou em treino não-distribuído): quem loga e salva."""
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def mean_across_ranks(total, count, device):
    """Média global de somas locais.

    Em DDP cada rank avalia só o seu shard da validação; o all_reduce garante
    que todos vejam a MESMA val_loss — essencial para o early stopping tomar
    a mesma decisão em todos os processos. Sem DDP, é só a divisão local.
    """
    if not dist.is_initialized():
        return total / max(1, count)
    t = torch.tensor([total, float(count)], device=device)
    dist.all_reduce(t)
    return (t[0] / t[1].clamp(min=1)).item()
