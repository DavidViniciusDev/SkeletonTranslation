"""Dump da configuração completa de um treino em ``<out_dir>/training_config.json``.

O ``meta`` gravado dentro de cada ``.pt`` guarda só as dataclasses de config.
Este arquivo complementa com o contexto da execução — comando, versões, commit
do git, hardware, tamanhos dos dados e da agenda — para que um treino possa ser
reproduzido e comparado sem abrir o checkpoint.

Tudo aqui é *best-effort*: qualquer falha ao coletar ou gravar vira um aviso e
NUNCA interrompe o treino. O formato dos checkpoints não é alterado.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime

TRAINING_CONFIG_FILE = "training_config.json"

# Raiz do repositório (skeltrans/training/run_info.py -> ../../)
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(*args):
    """Executa um comando git no repositório; devolve a saída ou None."""
    try:
        out = subprocess.run(["git", *args], cwd=_REPO_DIR, capture_output=True,
                             text=True, timeout=5, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def git_info():
    """Commit, branch e se havia alterações não commitadas (código ≠ commit)."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def environment_info(device):
    """Versões de Python/bibliotecas e o hardware usado."""
    import torch

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    try:
        if getattr(device, "type", str(device)) == "cuda" and torch.cuda.is_available():
            idx = device.index if getattr(device, "index", None) is not None else 0
            info["gpu_name"] = torch.cuda.get_device_name(idx)
            info["gpu_count_visible"] = torch.cuda.device_count()
    except Exception:
        pass
    # só variáveis que descrevem a topologia do treino (nada de segredos)
    for key in ("CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        if key in os.environ:
            info[key] = os.environ[key]
    return info


def parameter_counts(model):
    """Total de parâmetros e quantos são treináveis (encoder, T5, CTC)."""
    def count(params):
        params = list(params)
        return {"total": sum(p.numel() for p in params),
                "trainable": sum(p.numel() for p in params if p.requires_grad)}

    counts = {"model": count(model.parameters())}
    for name in ("encoder", "t5", "ctc_head"):
        sub = getattr(model, name, None)
        if sub is not None:
            counts[name] = count(sub.parameters())
    return counts


def save_training_config(out_dir, config, run=None):
    """Grava ``{"config": ..., "run": ...}`` em ``out_dir/training_config.json``.

    Escrita atômica (arquivo temporário + rename) para nunca deixar um JSON
    pela metade. Devolve o caminho gravado, ou None se falhar.
    """
    path = os.path.join(out_dir, TRAINING_CONFIG_FILE)
    try:
        os.makedirs(out_dir, exist_ok=True)
        payload = {"config": config, "run": run or {}}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            # default=str cobre tipos não-JSON (torch.device, Path, ...)
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
        return path
    except Exception as exc:
        print(f"[config] AVISO: nao foi possivel gravar {path}: {exc}")
        return None


def now_iso():
    """Data/hora local com fuso, ex.: 2026-10-13T14:02:11-03:00."""
    return datetime.now().astimezone().isoformat(timespec="seconds")
