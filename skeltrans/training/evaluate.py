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
from skeltrans.training.metrics import (compute_bertscore, compute_metrics,
                                        compute_wer)


# --------------------------------------------------------------------------- #
# Collate de avaliacao: paga as features e devolve os textos de referencia
# --------------------------------------------------------------------------- #
def make_eval_collate(with_gloss=False):
    """Collate de avaliacao. Com `with_gloss`, o Dataset entrega tambem a lista
    de glosas de referencia (campo `tokens`), necessaria para o WER."""
    def collate(batch):
        if with_gloss:
            feats, texts, glosses = zip(*batch)
        else:
            feats, texts = zip(*batch)
            glosses = None
        lengths = [f.shape[0] for f in feats]
        T = max(lengths)
        B = len(feats)
        D = feats[0].shape[1]
        padded = torch.zeros(B, T, D, dtype=torch.float32)
        pad_mask = torch.ones(B, T, dtype=torch.bool)  # True = padding
        for i, f in enumerate(feats):
            padded[i, : f.shape[0]] = f
            pad_mask[i, : f.shape[0]] = False
        if with_gloss:
            return padded, pad_mask, list(texts), [list(g) for g in glosses]
        return padded, pad_mask, list(texts)
    return collate


def resolve_gloss_vocab(checkpoint, override=None):
    """Localiza o gloss_vocab.json. Mesma prioridade do tokenizer: override
    explicito > diretorio do checkpoint (onde o Trainer o salva) > None."""
    from skeltrans.training.gloss_vocab import GlossVocab

    if override:
        return GlossVocab.load(override)
    cand = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "gloss_vocab.json")
    return GlossVocab.load(cand) if os.path.exists(cand) else None


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

    # WER de glosas: exige --wer, um checkpoint com head CTC e o vocabulario.
    # Falha cedo e com mensagem explicita, em vez de silenciosamente omitir a
    # metrica pedida.
    gloss_vocab = None
    if args.wer:
        if not getattr(model, "use_ctc", False):
            raise SystemExit(
                "ERRO: --wer pedido, mas este checkpoint nao tem head CTC. "
                "O WER de glosas vem do reconhecimento auxiliar; treine com --use-ctc.")
        gloss_vocab = resolve_gloss_vocab(args.checkpoint, args.gloss_vocab)
        if gloss_vocab is None:
            raise SystemExit(
                "ERRO: --wer pedido, mas nao encontrei gloss_vocab.json no diretorio "
                f"do checkpoint ({os.path.dirname(os.path.abspath(args.checkpoint))}). "
                "Aponte com --gloss-vocab.")
        print(f"WER de glosas ativo | vocabulario: {len(gloss_vocab)} rotulos")

    ds = LandmarkTextDataset(args.test_manifest, features_dir=args.features_dir,
                             return_gloss=bool(args.wer))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=make_eval_collate(with_gloss=bool(args.wer)),
                    num_workers=args.num_workers)
    print(f"Teste: {len(ds)} exemplos ({args.test_manifest})")

    hyps, refs = [], []
    gloss_hyps, gloss_refs = [], []
    for batch in dl:
        feats, pad_mask, texts = batch[0], batch[1], batch[2]
        feats, pad_mask = feats.to(device), pad_mask.to(device)
        ids = model.generate(feats, pad_mask,
                             max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
        preds = tokenizer.batch_decode(ids, skip_special_tokens=True)
        hyps.extend(p.strip() for p in preds)
        refs.extend(t.strip() for t in texts)
        if gloss_vocab is not None:
            for seq in model.predict_glosses(feats, pad_mask):
                gloss_hyps.append(gloss_vocab.decode(seq))
            gloss_refs.extend(batch[3])
        if len(hyps) % (args.batch_size * 20) == 0:
            print(f"  ... {len(hyps)}/{len(ds)} traduzidos")

    metrics = compute_metrics(hyps, refs)

    if gloss_vocab is not None:
        metrics.update(compute_wer(gloss_hyps, gloss_refs))

    if args.bertscore:
        print(f"Calculando BERTScore (lang={args.bertscore}"
              f"{', model=' + args.bertscore_model if args.bertscore_model else ''}) ...")
        metrics.update(compute_bertscore(
            hyps, refs, lang=args.bertscore, model_type=args.bertscore_model,
            batch_size=args.bertscore_batch_size, device=device,
            rescale_with_baseline=args.bertscore_baseline))

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
        if gloss_vocab is not None:
            payload["gloss_predictions"] = [
                {"ref": " ".join(r), "hyp": " ".join(h)}
                for r, h in zip(gloss_refs, gloss_hyps)
            ]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nPredicoes + metricas salvas em: {args.out}")


def build_parser():
    ap = argparse.ArgumentParser(
        description="Avaliacao SLT: BLEU-1..4, METEOR, e opcionalmente WER de glosas e BERTScore.")
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

    # ------------------- reconhecimento (sign2gloss) ------------------- #
    ap.add_argument("--wer", action="store_true",
                    help="mede WER de glosas a partir do head CTC (decodificacao greedy), "
                         "com decomposicao em substituicoes/delecoes/insercoes. Exige um "
                         "checkpoint treinado com --use-ctc e o gloss_vocab.json. E a metrica "
                         "nativa do PHOENIX e serve de diagnostico: separa falha do encoder "
                         "(nao viu os sinais) de falha do decoder (viu, mas nao traduziu).")
    ap.add_argument("--gloss-vocab", default=None,
                    help="caminho do gloss_vocab.json (padrao: diretorio do checkpoint)")

    # ------------------------ BERTScore (opcional) --------------------- #
    ap.add_argument("--bertscore", default=None, metavar="LANG",
                    help="ativa o BERTScore e define o IDIOMA das referencias ('de' para o "
                         "PHOENIX, 'pt' para o V-LIBRASIL). Metrica neural reference-only; "
                         "requer 'pip install bert-score' e baixa um modelo na primeira vez.")
    ap.add_argument("--bertscore-model", default=None,
                    help="modelo HF especifico para o BERTScore, sobrepondo o default do idioma")
    ap.add_argument("--bertscore-batch-size", type=int, default=64,
                    help="batch do BERTScore (padrao: 64); reduza se faltar VRAM")
    ap.add_argument("--bertscore-baseline", action="store_true",
                    help="reescalona por baseline (espalha a faixa alta e estreita do BERTScore "
                         "cru, melhorando a legibilidade). Exige arquivo de baseline para o "
                         "idioma, que nao existe para todos - por isso e opt-in.")
    return ap


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
