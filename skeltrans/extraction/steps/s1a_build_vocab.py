#!/usr/bin/env python3
"""Constroi o vocabulario da coluna `libras-gloss` do CSV pt_br2libras_gloss.csv.

Cada token separado por espaco na glosa e tratado como uma entrada do
vocabulario (padrao para vocabularios de traducao de lingua de sinais).
Tokens especiais como [PONTO] e sinais compostos (BRASIL&PAIS) ou
multi-palavra (NAO_TER) sao preservados como estao.
"""
import argparse
import csv
import sys
from collections import Counter

from skeltrans.extraction.config import PATHS


def build_parser():
    ap = argparse.ArgumentParser(description="s1a: vocabulario das glosas (libras-gloss) do CSV.")
    ap.add_argument("--csv", default=PATHS.gloss_csv, help="CSV de entrada (coluna libras-gloss)")
    ap.add_argument("--out", default=PATHS.vocab_txt, help="TXT de saida (palavra<TAB>frequencia)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    counter = Counter()
    n_rows = 0
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "libras-gloss" not in reader.fieldnames:
            sys.exit(f"Coluna 'libras-gloss' nao encontrada. Colunas: {reader.fieldnames}")
        for row in reader:
            gloss = (row.get("libras-gloss") or "").strip()
            if not gloss:
                continue
            n_rows += 1
            for token in gloss.split():
                counter[token] += 1

    vocab = sorted(counter)

    # Grava o vocabulario (uma palavra por linha, com a frequencia)
    with open(args.out, "w", encoding="utf-8") as out:
        for word in vocab:
            out.write(f"{word}\t{counter[word]}\n")

    print(f"Linhas processadas (com glosa): {n_rows}")
    print(f"Tamanho do vocabulario (tokens unicos): {len(vocab)}")
    print(f"Total de tokens (com repeticao): {sum(counter.values())}")
    print(f"\nArquivo gerado: {args.out} (palavra<TAB>frequencia)")
    print("\n20 palavras mais frequentes:")
    for word, freq in counter.most_common(20):
        print(f"  {freq:7d}  {word}")


if __name__ == "__main__":
    main()
