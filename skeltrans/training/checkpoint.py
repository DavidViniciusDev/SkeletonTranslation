"""Salvamento e carregamento de checkpoints (centraliza a lógica de I/O)."""

import os

import torch


def save_checkpoint(path, model, epoch, meta):
    """Grava {model, epoch, args} em `path`.

    `meta` é um dict serializável com os hiperparâmetros (mantém a chave 'args'
    por compatibilidade com o formato original dos checkpoints).
    """
    torch.save({"model": model.state_dict(), "epoch": epoch, "args": meta}, path)


def _model_hparams(meta):
    """Extrai os hiperparâmetros de arquitetura do `meta` gravado no checkpoint.

    Aceita dois formatos:
      - atual : meta = {"model": {t5_name, d_model, nhead, num_layers, dropout}, ...}
      - legado: meta = {t5|t5_name, d_model, nhead, num_layers, dropout, ...}
    """
    return meta.get("model", meta) if isinstance(meta, dict) else {}


def load_model(checkpoint, device, t5_override=None):
    """Reconstrói o SLTModel a partir de um checkpoint e carrega os pesos.

    Retorna (model, ckpt, t5_name): `model` já em `device` e em modo eval;
    `ckpt` é o dict bruto salvo (útil para ler ckpt['epoch']); `t5_name` é o
    identificador do decoder efetivamente usado.
    """
    from transformers import T5ForConditionalGeneration

    from skeltrans.training.config import PTT5_NAME
    from skeltrans.training.models import SLTModel

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    hp = _model_hparams(ckpt.get("args", {}) or {})
    t5_name = t5_override or hp.get("t5_name") or hp.get("t5") or PTT5_NAME

    t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
    model = SLTModel(
        t5,
        d_model=hp.get("d_model", 512),
        nhead=hp.get("nhead", 8),
        num_layers=hp.get("num_layers", 6),
        dropout=hp.get("dropout", 0.2),
        # reconstrói o head CTC apenas se o checkpoint foi treinado com ele
        use_ctc=hp.get("use_ctc", False),
        gloss_vocab_size=hp.get("gloss_vocab_size", 0),
    )
    # strict=False garante retrocompatibilidade: checkpoints antigos (sem head
    # CTC) carregam num modelo novo, e o head CTC (irrelevante na inferência) é
    # ignorado quando ausente. Chaves inesperadas/faltantes são reportadas.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[checkpoint] chaves ausentes ({len(missing)}): {missing[:4]}...")
    if unexpected:
        print(f"[checkpoint] chaves inesperadas ({len(unexpected)}): {unexpected[:4]}...")
    model.to(device).eval()
    return model, ckpt, t5_name


def load_tokenizer(checkpoint, t5_name, tokenizer_override=None):
    """Carrega o tokenizer.

    Prioridade: override explícito > diretório do checkpoint (onde o Trainer
    salva o tokenizer junto do best.pt) > identificador do T5.
    """
    from transformers import AutoTokenizer

    if tokenizer_override:
        return AutoTokenizer.from_pretrained(tokenizer_override)
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
    if os.path.exists(os.path.join(ckpt_dir, "tokenizer_config.json")) or \
       os.path.exists(os.path.join(ckpt_dir, "spiece.model")):
        return AutoTokenizer.from_pretrained(ckpt_dir)
    return AutoTokenizer.from_pretrained(t5_name)
