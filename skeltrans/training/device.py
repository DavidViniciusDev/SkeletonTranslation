"""Resolução do dispositivo de execução."""

import torch


def resolve_device(device=None):
    """Resolve o dispositivo.

    - device explícito ('cuda'/'cpu'): usado como está.
    - None: CUDA se disponível, senão CPU.
    """
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
