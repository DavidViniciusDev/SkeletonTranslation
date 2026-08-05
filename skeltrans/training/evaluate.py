#!/usr/bin/env python3
"""Etapa 2 (avaliacao): mede BLEU-1..4 e METEOR no conjunto de teste.

Carrega um checkpoint treinado (skeltrans.training), gera as traducoes para um
manifesto de teste e calcula as metricas de traducao.

A arquitetura e reconstruida automaticamente a partir dos hiperparametros
gravados no checkpoint (ckpt['args']). O tokenizer e carregado do diretorio do
checkpoint (onde o Trainer o salva junto do best.pt) ou, na falta, do nome do T5.

Uso (pelo shim na raiz):
    python3 evaluate_slt.py \\
        --checkpoint checkpoints/best.pt \\
        --test-manifest data/interim/test.json \\
        --features-dir data/features/sentence_features \\
        --out checkpoints/test_metrics.json

Ou pelo modulo:
    python -m skeltrans.training.evaluate --checkpoint ... --test-manifest ...
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from skeltrans.training.checkpoint import load_model, load_tokenizer
from skeltrans.training.data import LandmarkTextDataset
from skeltrans.training.device import resolve_device
from skeltrans.training.metrics import compute_metrics


# --------------------------------------------------------------------------- #
# Collate de avaliacao: paga as features e devolve os textos de referencia
# --------------------------------------------------------------------------- #
def make_eval_collate():
    def collate(batch):
        feats, texts = zip(*batch)
        lengths = [f.shape[0] for f in feats]
        T = max(lengths)
        B = len(feats)
        D = feats[0].shape[1]
        padded = torch.zeros(B, T, D, dtype=torch.float32)
        pad_mask = torch.ones(B, T, dtype=torch.bool)  # True = padding
        for i, f in enumerate(feats):
            padded[i, : f.shape[0]] = f
            pad_mask[i, : f.shape[0]] = False
        return padded, pad_mask, list(texts)
    return collate


# --------------------------------------------------------------------------- #
# Loop de avaliacao
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run(args):
    device = resolve_device(args.device)
    print(f"Dispositivo: {device}")

    model, ckpt, t5_name = load_model(args.checkpoint, device, args.t5,
                                      low_vram=args.low_vram)
    tokenizer = load_tokenizer(args.checkpoint, t5_name, args.tokenizer)
    print(f"Checkpoint: {args.checkpoint} | T5: {t5_name} | epoca: {ckpt.get('epoch', '?')}")

    ds = LandmarkTextDataset(args.test_manifest, features_dir=args.features_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=make_eval_collate(), num_workers=args.num_workers)
    print(f"Teste: {len(ds)} exemplos ({args.test_manifest})")

    hyps, refs = [], []
    for feats, pad_mask, texts in dl:
        feats, pad_mask = feats.to(device), pad_mask.to(device)
        ids = model.generate(feats, pad_mask,
                             max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
        preds = tokenizer.batch_decode(ids, skip_special_tokens=True)
        hyps.extend(p.strip() for p in preds)
        refs.extend(t.strip() for t in texts)
        if len(hyps) % (args.batch_size * 20) == 0:
            print(f"  ... {len(hyps)}/{len(ds)} traduzidos")

    metrics = compute_metrics(hyps, refs)

    print("\n== Metricas (conjunto de teste) ==")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v}")

    if args.out:
        payload = {
            "checkpoint": os.path.abspath(args.checkpoint),
            "test_manifest": os.path.abspath(args.test_manifest),
            "num_examples": len(hyps),
            "gen": {"num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens},
            "metrics": metrics,
            "predictions": [{"ref": r, "hyp": h} for r, h in zip(refs, hyps)],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nPredicoes + metricas salvas em: {args.out}")


def build_parser():
    ap = argparse.ArgumentParser(description="Avaliacao SLT: BLEU-1..4 e METEOR.")
    ap.add_argument("--checkpoint", required=True, help="caminho do .pt treinado (ex.: best.pt)")
    ap.add_argument("--test-manifest", required=True, help="manifesto de teste (test.json)")
    ap.add_argument("--features-dir", default=None,
                    help="pasta dos .npy de features (repassado ao Dataset)")
    ap.add_argument("--t5", default=None, help="override do nome/checkpoint do T5")
    ap.add_argument("--tokenizer", default=None,
                    help="override do tokenizer (padrao: dir do checkpoint, senao o T5)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None, help="cuda|cpu (auto se omitido)")
    ap.add_argument("--low-vram", action="store_true",
                    help="reduz o uso de VRAM: pesos em bfloat16 (GPU Ampere+) e offload do "
                         "encoder do T5, que nao e usado nesta arquitetura. Ver LOW_VRAM.md.")
    ap.add_argument("--out", default=None, help="JSON de saida com metricas + predicoes")
    return ap


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
