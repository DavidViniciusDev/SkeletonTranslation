#!/usr/bin/env python3
"""Média dos pesos de vários checkpoints (checkpoint averaging).

No fim do treino, os pesos de épocas vizinhas oscilam em volta de um bom
ponto. A média desses pesos costuma cair mais perto do "centro" dessa região
do que qualquer época isolada, e em tradução rende tipicamente alguns décimos
de BLEU — sem retreinar nada.

O resultado é um .pt no MESMO formato do Trainer: avalie com o
evaluate_slt.py como qualquer outro checkpoint. Por padrão ele é gravado na
pasta dos checkpoints, onde o evaluate encontra o tokenizer e o
gloss_vocab.json.

Escolha das épocas (os epochN.pt que o Trainer grava a cada época):
  --last N     as N últimas épocas
  --best K     as K melhores épocas pela métrica de VALIDAÇÃO registrada no
               eval_history.jsonl (requer ter treinado com --eval-every)
  --checkpoints a.pt b.pt ...   uma lista explícita

Uso:
    python3 tools/average_checkpoints.py --dir checkpoints_phoenixv11 --best 5
    python3 tools/average_checkpoints.py --dir checkpoints_phoenixv11 --last 5
    python3 tools/average_checkpoints.py --checkpoints ck/epoch40.pt ck/epoch45.pt --out ck/avg.pt
"""

import argparse
import glob
import json
import os
import re
import sys

import torch

_EPOCH_RE = re.compile(r"epoch(\d+)\.pt$")


def epoch_files(ckpt_dir):
    """{época: caminho} dos epochN.pt da pasta."""
    out = {}
    for p in glob.glob(os.path.join(ckpt_dir, "epoch*.pt")):
        m = _EPOCH_RE.search(os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def pick_best(ckpt_dir, k, metric):
    """As k melhores épocas pelo eval_history.jsonl (val_loss: menor é melhor)."""
    hist_path = os.path.join(ckpt_dir, "eval_history.jsonl")
    if not os.path.exists(hist_path):
        sys.exit(f"--best exige {hist_path} (treine com --eval-every).")
    with open(hist_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if r.get(metric) is not None]
    if not rows:
        sys.exit(f"nenhuma linha do eval_history.jsonl tem a metrica {metric!r}.")
    rows.sort(key=lambda r: r[metric], reverse=(metric != "val_loss"))
    files = epoch_files(ckpt_dir)
    chosen = [(r["epoch"], r[metric]) for r in rows if r["epoch"] in files][:k]
    if len(chosen) < k:
        print(f"[aviso] so {len(chosen)} das {k} melhores epocas tem o epochN.pt na pasta.")
    for ep, val in chosen:
        print(f"  epoca {ep:4d}  {metric}={val}")
    return [files[ep] for ep, _ in sorted(chosen)]


def average(paths):
    """Média dos state_dicts, carregando um checkpoint por vez (pouca RAM)."""
    total, first, ref_shapes = None, None, None
    for i, p in enumerate(paths, 1):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"]
        if first is None:
            first = ck
            ref_shapes = {k: tuple(v.shape) for k, v in sd.items()}
            total = {k: v.double().clone() if v.is_floating_point() else v.clone()
                     for k, v in sd.items()}
        else:
            shapes = {k: tuple(v.shape) for k, v in sd.items()}
            if shapes != ref_shapes:
                sys.exit(f"{p} tem arquitetura diferente de {paths[0]} "
                         "(chaves ou formatos dos pesos nao batem).")
            for k, v in sd.items():
                if v.is_floating_point():
                    total[k] += v.double()
                elif not torch.equal(v, total[k]):
                    print(f"[aviso] tensor nao numerico {k} difere entre checkpoints; "
                          "mantendo o do primeiro.")
        print(f"  [{i}/{len(paths)}] {os.path.basename(p)} (epoca {ck.get('epoch', '?')})")
        del ck, sd
    n = len(paths)
    avg = {k: (v / n).to(first["model"][k].dtype) if v.is_floating_point() else v
           for k, v in total.items()}
    return avg, first


def main():
    ap = argparse.ArgumentParser(description="Media dos pesos de varios checkpoints.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoints", nargs="+", help="lista explicita de .pt")
    src.add_argument("--last", type=int, help="media das N ultimas epocas de --dir")
    src.add_argument("--best", type=int,
                     help="media das K melhores epocas de --dir pela metrica de validacao")
    ap.add_argument("--dir", help="pasta dos epochN.pt (para --last/--best)")
    ap.add_argument("--metric", default="BLEU-4",
                    help="chave do eval_history.jsonl usada por --best (BLEU-1..4, METEOR "
                         "ou val_loss). Padrao: BLEU-4")
    ap.add_argument("--out", default=None,
                    help="arquivo de saida (padrao: <dir>/avg_last<N>.pt ou avg_best<K>.pt)")
    args = ap.parse_args()

    if args.checkpoints:
        paths = args.checkpoints
        default_out = os.path.join(os.path.dirname(os.path.abspath(paths[0])), "avg.pt")
    else:
        if not args.dir:
            sys.exit("--last/--best exigem --dir.")
        n = args.last if args.last is not None else args.best
        if n < 1:
            sys.exit("--last/--best devem ser >= 1.")
        if args.last is not None:
            files = epoch_files(args.dir)
            paths = [files[e] for e in sorted(files)[-n:]]
            default_out = os.path.join(args.dir, f"avg_last{n}.pt")
        else:
            paths = pick_best(args.dir, n, args.metric)
            default_out = os.path.join(args.dir, f"avg_best{n}.pt")
    if not paths:
        sys.exit("nenhum checkpoint selecionado.")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"checkpoints nao encontrados: {missing}")

    print(f"Media de {len(paths)} checkpoint(s):")
    avg, first = average(paths)
    epochs = []
    for p in paths:
        m = _EPOCH_RE.search(os.path.basename(p))
        epochs.append(int(m.group(1)) if m else os.path.basename(p))
    out = args.out or default_out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    torch.save({"model": avg, "epoch": f"media{epochs}", "args": first.get("args", {}),
                "averaged_from": [os.path.abspath(p) for p in paths]}, out)
    print(f"\nCheckpoint medio salvo em: {out}")
    print(f"Avalie com: python3 evaluate_slt.py --checkpoint {out} --test-manifest ...")


if __name__ == "__main__":
    main()
