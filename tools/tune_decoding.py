#!/usr/bin/env python3
"""Ajuste dos parâmetros de decodificação no conjunto de VALIDAÇÃO.

Carrega um checkpoint uma vez e gera as traduções da validação para cada
combinação de beams, length_penalty e no_repeat_ngram_size, medindo BLEU e
METEOR. Não retreina nada: só muda como o decoder escolhe as palavras.

Use SEMPRE a validação (dev). Escolher a decodificação olhando o teste
contamina o teste — a combinação vencedora daqui é que deve ser aplicada,
uma única vez, no evaluate_slt.py sobre o teste.

Uso:
    python3 tools/tune_decoding.py \\
        --checkpoint checkpoints_phoenixv11/best.pt \\
        --val-manifest data_phoenix/interim/dev.json \\
        --features-dir data_phoenix/features/landmarks_115_norm9 \\
        --num-beams 4 5 --length-penalty 0.8 1.0 1.2 --no-repeat-ngram-size 0 3 \\
        --out checkpoints_phoenixv11/tune_decoding.json
"""

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, Subset  # noqa: E402

from skeltrans.training.checkpoint import load_model, load_tokenizer  # noqa: E402
from skeltrans.training.data import LandmarkTextDataset  # noqa: E402
from skeltrans.training.device import resolve_device  # noqa: E402
from skeltrans.training.evaluate import make_eval_collate  # noqa: E402
from skeltrans.training.trainer import generate_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Ajuste da decodificacao na validacao.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-manifest", required=True,
                    help="manifesto de VALIDACAO (dev). Nunca o de teste.")
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--num-beams", type=int, nargs="+", default=[4])
    ap.add_argument("--length-penalty", type=float, nargs="+", default=[1.0])
    ap.add_argument("--no-repeat-ngram-size", type=int, nargs="+", default=[0],
                    help="0 = desligado")
    ap.add_argument("--metric", default="BLEU-4", help="metrica para ordenar. Padrao: BLEU-4")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="usa so os N primeiros exemplos (0 = validacao inteira)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--low-vram", action="store_true")
    ap.add_argument("--t5", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default=None, help="JSON com todos os resultados")
    args = ap.parse_args()
    if "test" in os.path.basename(args.val_manifest).lower():
        print("[AVISO] o manifesto parece ser de TESTE. Ajuste a decodificacao na validacao; "
              "o teste so deve ser usado uma vez, no fim.")

    device = resolve_device(args.device)
    model, ckpt, t5_name = load_model(args.checkpoint, device, args.t5, low_vram=args.low_vram)
    tokenizer = load_tokenizer(args.checkpoint, t5_name, args.tokenizer)
    ds = LandmarkTextDataset(args.val_manifest, features_dir=args.features_dir,
                             extra_features=getattr(model, "extra_features", ()))
    if args.max_samples and args.max_samples < len(ds):
        ds = Subset(ds, range(args.max_samples))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=make_eval_collate(), num_workers=args.num_workers)

    grid = list(itertools.product(args.num_beams, args.length_penalty,
                                  args.no_repeat_ngram_size))
    print(f"Checkpoint: {args.checkpoint} (epoca {ckpt.get('epoch', '?')}) | "
          f"validacao: {len(ds)} exemplos | {len(grid)} combinacoes\n")
    results = []
    for i, (beams, lp, nrng) in enumerate(grid, 1):
        t0 = time.time()
        metrics, _, _ = generate_metrics(model, tokenizer, dl, device, num_beams=beams,
                                         max_new_tokens=args.max_new_tokens,
                                         length_penalty=lp, no_repeat_ngram_size=nrng)
        results.append({"num_beams": beams, "length_penalty": lp,
                        "no_repeat_ngram_size": nrng, **metrics})
        print(f"[{i}/{len(grid)}] beams={beams} length_penalty={lp} no_repeat={nrng} -> "
              + " ".join(f"{k}={v:.2f}" for k, v in metrics.items())
              + f"  ({time.time() - t0:.0f}s)")

    results.sort(key=lambda r: r[args.metric], reverse=True)
    best = results[0]
    print(f"\n== Ranking por {args.metric} (validacao) ==")
    for r in results:
        mark = "  <- melhor" if r is best else ""
        print(f"  {r[args.metric]:6.2f}  beams={r['num_beams']} "
              f"length_penalty={r['length_penalty']} no_repeat={r['no_repeat_ngram_size']}{mark}")
    flags = (f"--num-beams {best['num_beams']} --length-penalty {best['length_penalty']} "
             f"--no-repeat-ngram-size {best['no_repeat_ngram_size']}")
    print(f"\nAplique no teste (uma vez): python3 evaluate_slt.py --checkpoint "
          f"{args.checkpoint} --test-manifest ... {flags}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": os.path.abspath(args.checkpoint),
                       "val_manifest": os.path.abspath(args.val_manifest),
                       "num_examples": len(ds), "metric": args.metric,
                       "best": best, "evaluate_flags": flags, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"Resultados em: {args.out}")


if __name__ == "__main__":
    main()
