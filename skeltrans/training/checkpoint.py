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


def _infer_ds_steps(state_dict, meta_ds_steps):
    """Deduz ``ds_steps`` dos pesos salvos, e não do config gravado.

    Alguns treinos gravaram ``ds_steps`` no config sem que o trainer o
    repassasse ao SLTModel: os pesos vieram de um encoder SEM downsampling.
    Confiar no config montaria convoluções extras com pesos aleatórios
    (``strict=False`` só avisaria das chaves ausentes). Contar as chaves
    ``encoder.temporal_ds.N.weight`` reconstrói a arquitetura realmente treinada;
    checkpoints anteriores à flag não têm essas chaves e caem em 0, como antes.
    """
    prefix = "encoder.temporal_ds."
    idx = {k[len(prefix):].split(".", 1)[0] for k in state_dict
           if k.startswith(prefix) and k.endswith(".weight")}
    ds_steps = len(idx)
    if meta_ds_steps is not None and int(meta_ds_steps) != ds_steps:
        print(f"[checkpoint] AVISO: config diz ds_steps={meta_ds_steps}, mas os "
              f"pesos correspondem a ds_steps={ds_steps}; usando {ds_steps}.")
    return ds_steps


def load_model(checkpoint, device, t5_override=None, low_vram=False):
    """Reconstrói o SLTModel a partir de um checkpoint e carrega os pesos.

    Retorna (model, ckpt, t5_name): `model` já em `device` e em modo eval;
    `ckpt` é o dict bruto salvo (útil para ler ckpt['epoch']); `t5_name` é o
    identificador do decoder efetivamente usado.

    `low_vram=True` aplica as otimizações de memória da inferência (pesos em
    bf16 quando a GPU suporta + offload do encoder do T5, que não é usado).
    Ver LOW_VRAM.md.
    """
    from transformers import T5ForConditionalGeneration

    from skeltrans.training.config import PTT5_NAME
    from skeltrans.training.models import SLTModel

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    hp = _model_hparams(ckpt.get("args", {}) or {})
    t5_name = t5_override or hp.get("t5_name") or hp.get("t5") or PTT5_NAME

    ds_steps = _infer_ds_steps(ckpt["model"], hp.get("ds_steps"))

    t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
    model = SLTModel(
        t5,
        d_model=hp.get("d_model", 512),
        nhead=hp.get("nhead", 8),
        num_layers=hp.get("num_layers", 6),
        dropout=hp.get("dropout", 0.2),
        # o state_dict é a fonte da verdade (ver _infer_ds_steps)
        ds_steps=ds_steps,
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
    if low_vram:
        from skeltrans.training.low_vram import (cast_bf16_for_inference,
                                                 offload_t5_encoder)
        cast_bf16_for_inference(model, device)
        offload_t5_encoder(model.t5)
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
