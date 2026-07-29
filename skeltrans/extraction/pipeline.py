#!/usr/bin/env python3
"""Orquestrador da extração: roda os passos s1a … s5 na ordem correta.

Cada passo roda em seu PRÓPRIO processo (`python -m skeltrans.extraction.steps.sN`),
isolando o multiprocessing 'spawn' usado pelo MediaPipe (s3) e pela normalização
(s4). A ordem canônica vem de skeltrans.extraction.config.STEPS.

Uso:
    python -m skeltrans.extraction.pipeline                # roda tudo (s1a..s5)
    python -m skeltrans.extraction.pipeline --list         # lista os passos
    python -m skeltrans.extraction.pipeline --from s3      # do s3 até o fim
    python -m skeltrans.extraction.pipeline --to s2        # do início até s2
    python -m skeltrans.extraction.pipeline --from s3 --to s4
    python -m skeltrans.extraction.pipeline --only s4 -- --workers 4  # args extras p/ o passo
    python -m skeltrans.extraction.pipeline --dry-run

Observação: argumentos após `--` só são repassados quando se usa --only (um
único passo), evitando mandar flags que um passo não conhece.
"""
import argparse
import subprocess
import sys

from skeltrans.extraction.config import STEPS


def _ids():
    return [s[0] for s in STEPS]


def _select(from_id, to_id, only):
    ids = _ids()
    if only:
        return [only]
    i0 = ids.index(from_id) if from_id else 0
    i1 = ids.index(to_id) if to_id else len(ids) - 1
    if i0 > i1:
        raise SystemExit(f"--from ({from_id}) vem depois de --to ({to_id}).")
    return ids[i0:i1 + 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Orquestra o pipeline de extracao (s1a..s5).")
    ap.add_argument("--from", dest="from_id", choices=_ids(), help="passo inicial (inclusive)")
    ap.add_argument("--to", dest="to_id", choices=_ids(), help="passo final (inclusive)")
    ap.add_argument("--only", choices=_ids(), help="roda apenas este passo")
    ap.add_argument("--list", action="store_true", help="lista os passos e sai")
    ap.add_argument("--dry-run", action="store_true", help="mostra os comandos sem executar")
    args, extra = ap.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.list:
        for sid, mod, desc in STEPS:
            print(f"  {sid:4s} {desc}   ({mod})")
        return 0

    if extra and not args.only:
        raise SystemExit("Argumentos extras só são permitidos com --only (passo único).")

    steps = {s[0]: s for s in STEPS}
    selected = _select(args.from_id, args.to_id, args.only)
    print(f"Passos a executar: {', '.join(selected)}")

    for sid in selected:
        _, mod, desc = steps[sid]
        cmd = [sys.executable, "-m", mod] + (extra if args.only else [])
        print(f"\n=== [{sid}] {desc} ===")
        print(f"$ {' '.join(cmd)}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nFALHOU no passo {sid} (exit {result.returncode}). Pipeline interrompido.")
            return result.returncode

    print("\nPipeline concluido." if not args.dry_run else "\n(dry-run) nada executado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
