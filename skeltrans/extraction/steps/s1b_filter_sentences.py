#!/usr/bin/env python3
"""Filtra sentencas de pt_br2libras_gloss.json mantendo apenas aquelas em que
TODAS as palavras da glosa (libras-gloss) existem no vocabulario de videos
definido por videos_words_by_word.json.

Politica de correspondencia:
  - Normalizacao: maiusculas + remocao de acentos (o vocabulario de videos e
    ASCII e maiusculo), comparacao por igualdade.
  - Tokens de pontuacao entre colchetes ([PONTO], [INTERROGACAO], ...) sao
    ignorados (nao precisam existir no vocabulario).
  - Tokens compostos com '&' (ex.: DOAR&OBJETO) sao divididos; cada parte
    precisa existir no vocabulario.
  - Tokens multi-palavra com '_' (ex.: TERCA_FEIRA) tem o '_' trocado por
    espaco e sao casados contra as entradas multi-palavra do vocabulario;
    se nao casar, tenta-se o token sem o '_'.

Uma sentenca so e mantida se tiver pelo menos um sinal "real" e todos eles
estiverem presentes no vocabulario.

Expansao por '&': para aumentar o numero de sentencas, cada sentenca mantida e
expandida separando os tokens compostos com '&' (ex.: DOAR&OBJETO). Gera-se o
produto cartesiano das escolhas: uma copia da sentenca para cada combinacao,
onde cada token composto e substituido por um de seus termos. Assim uma
sentenca com um token '&' vira duas, e os demais campos sao duplicados.
"""
import argparse
import json
import itertools

from skeltrans.common.text import norm  # (fonte unica de norm())
from skeltrans.extraction.config import PATHS


def part_in_vocab(part: str, vnorm: set) -> bool:
    """Verifica uma parte de token contra o vocabulario, expandindo '_'."""
    if norm(part.replace("_", " ")) in vnorm:
        return True
    if norm(part) in vnorm:
        return True
    return False


def gloss_fully_in_vocab(gloss: str, vnorm: set) -> bool:
    real_signs = 0
    for token in gloss.split():
        # ignora marcadores de pontuacao [PONTO], [INTERROGACAO], etc.
        if token.startswith("[") and token.endswith("]"):
            continue
        real_signs += 1
        # divide tokens compostos com '&'; cada parte precisa existir
        for part in token.split("&"):
            if not part:
                continue
            if not part_in_vocab(part, vnorm):
                return False
    return real_signs > 0


def expand_ampersand(gloss: str):
    """Gera as variantes da glosa separando os tokens compostos com '&'.

    Retorna a lista de glosas resultantes do produto cartesiano: para cada
    token 'A&B', escolhe-se A ou B; tokens sem '&' (inclusive [PONTO]) tem
    uma unica opcao. Uma sentenca com um token '&' produz duas variantes.
    """
    options = []
    for token in gloss.split():
        if "&" in token:
            parts = [p for p in token.split("&") if p]
            options.append(parts if parts else [token])
        else:
            options.append([token])
    return [" ".join(combo) for combo in itertools.product(*options)]


def build_parser():
    ap = argparse.ArgumentParser(
        description="s1b: filtra sentencas cujas glosas estao todas no vocabulario de videos.")
    ap.add_argument("--gloss-json", default=PATHS.gloss_json, help="sentencas de entrada")
    ap.add_argument("--vocab-json", default=PATHS.videos_words, help="vocabulario de videos")
    ap.add_argument("--out", default=PATHS.in_vocab_json, help="JSON de saida (sentencas no vocab)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.vocab_json, encoding="utf-8") as f:
        vocab = json.load(f)
    vnorm = {norm(k) for k in vocab}

    with open(args.gloss_json, encoding="utf-8") as f:
        data = json.load(f)

    kept = [r for r in data if gloss_fully_in_vocab(r.get("libras-gloss", ""), vnorm)]

    # Expande os tokens compostos com '&', duplicando a sentenca por combinacao.
    expanded = []
    for r in kept:
        for variant in expand_ampersand(r.get("libras-gloss", "")):
            new_r = dict(r)
            new_r["libras-gloss"] = variant
            expanded.append(new_r)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(expanded, f, ensure_ascii=False, indent=2)

    print(f"Vocabulario de videos: {len(vocab)} sinais")
    print(f"Sentencas no arquivo de entrada: {len(data)}")
    print(f"Sentencas mantidas (todas as palavras no vocabulario): {len(kept)}")
    print(f"Sentencas apos expansao dos tokens '&': {len(expanded)}")
    print(f"Arquivo gerado: {args.out}")


if __name__ == "__main__":
    main()
