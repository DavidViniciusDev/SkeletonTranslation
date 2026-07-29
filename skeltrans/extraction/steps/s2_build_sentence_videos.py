#!/usr/bin/env python3
"""Monta, para cada sentenca de pt_br2libras_gloss_in_vocab.json, as sequencias
de videos (de videos_words_by_word.json) necessarias para concatenar os sinais
individuais e formar um unico video da sentenca.

Padrao dos nomes de video: {interprete}-{palavra}{-augmentation}.mp4
  - interprete: numero inicial (0, 1, 2, ...)
  - augmentation: combinacao de 'horizontal-flip', 'upsample', 'downsample'
    (ou 'none' quando nenhum)

Regra de consistencia: uma sequencia (sentence) usa o MESMO interprete e a
MESMA augmentation em TODAS as palavras (0 so cola com 0; flip so cola com
flip; etc.). So sao geradas as combinacoes (interprete, augmentation) que
existem em todas as palavras da sentenca.

Saida por registro:
  - base: interprete 0 SEM augmentation (videos 0-palavra.mp4) -> etapa de teste
  - sentences: todas as demais combinacoes consistentes (treino), excluindo a base

Tokens de pontuacao entre colchetes ([PONTO], ...) sao ignorados (sem video).
"""
import argparse
import json
import os
import re

from skeltrans.common.text import norm  # (fonte unica de norm())
from skeltrans.extraction.config import BASE_AUGMENTATION, BASE_INTERPRETER, PATHS


def interpreter_of(video_path: str) -> str:
    m = re.match(r"^(\d+)-", os.path.basename(video_path))
    return m.group(1) if m else "?"


def augmentation_of(video_path: str) -> str:
    b = os.path.basename(video_path).lower()
    flip = "horizontal-flip" in b
    if "upsample" in b:
        scale = "upsample"
    elif "downsample" in b:
        scale = "downsample"
    else:
        scale = "none"
    if not flip and scale == "none":
        return "none"
    parts = (["horizontal-flip"] if flip else []) + ([scale] if scale != "none" else [])
    return "-".join(parts)


def build_norm_map(vocab: dict) -> dict:
    return {norm(k): k for k in vocab}


def lookup_key(token: str, nmap: dict):
    for cand in (norm(token), norm(token.replace("_", " "))):
        if cand in nmap:
            return nmap[cand]
    return None


def index_videos(entries: list) -> dict:
    """(interprete, augmentation) -> entrada de video da palavra."""
    idx = {}
    for e in entries:
        idx[(interpreter_of(e["video"]), augmentation_of(e["video"]))] = e
    return idx


def video_info(gloss_token: str, entry: dict) -> dict:
    return {
        "gloss": gloss_token,
        "word": entry.get("word"),
        "video": entry["video"],
        "begin": entry.get("begin"),
        "end": entry.get("end"),
        "duration": entry.get("duration"),
    }


def make_sequence(combo, tokens, indices) -> dict:
    interp, aug = combo
    videos = []
    total = 0.0
    for tok, idx in zip(tokens, indices):
        e = idx[combo]
        videos.append(video_info(tok, e))
        total += e.get("duration") or 0.0
    return {
        "interpreter": interp,
        "augmentation": aug,
        "num_videos": len(videos),
        "total_duration": round(total, 2),
        "videos": videos,
    }


def build_record(rec: dict, nmap: dict, vocab: dict):
    tokens = [
        t for t in rec.get("libras-gloss", "").split()
        if not (t.startswith("[") and t.endswith("]"))
    ]
    if not tokens:
        return None

    indices = []
    for tok in tokens:
        key = lookup_key(tok, nmap)
        if key is None:
            return None  # nao deveria ocorrer (entrada ja filtrada por vocabulario)
        indices.append(index_videos(vocab[key]))

    # combinacoes (interprete, augmentation) presentes em TODAS as palavras
    common = set(indices[0])
    for idx in indices[1:]:
        common &= set(idx)

    base_combo = (BASE_INTERPRETER, BASE_AUGMENTATION)
    base = make_sequence(base_combo, tokens, indices) if base_combo in common else None

    sentence_combos = sorted(c for c in common if c != base_combo)
    sentences = [make_sequence(c, tokens, indices) for c in sentence_combos]

    return {
        "pt-br": rec.get("pt-br"),
        "english_translation": rec.get("english_translation"),
        "libras-gloss": rec.get("libras-gloss"),
        "is_government_source": rec.get("is_government_source"),
        "tokens": tokens,
        "num_words": len(tokens),
        "base": base,
        "sentences": sentences,
    }


def build_parser():
    ap = argparse.ArgumentParser(
        description="s2: monta as sequencias de videos por sentenca (base + variantes).")
    ap.add_argument("--gloss-json", default=PATHS.in_vocab_json, help="sentencas no vocabulario")
    ap.add_argument("--vocab-json", default=PATHS.videos_words, help="vocabulario de videos")
    ap.add_argument("--out", default=PATHS.sentence_videos_json, help="JSON de saida")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.vocab_json, encoding="utf-8") as f:
        vocab = json.load(f)
    nmap = build_norm_map(vocab)

    with open(args.gloss_json, encoding="utf-8") as f:
        data = json.load(f)

    out = []
    no_base = 0
    total_sentences = 0
    for rec in data:
        built = build_record(rec, nmap, vocab)
        if built is None:
            continue
        if built["base"] is None:
            no_base += 1
        total_sentences += len(built["sentences"])
        out.append(built)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Registros de entrada: {len(data)}")
    print(f"Registros gerados: {len(out)}")
    print(f"Registros sem base (0-* limpo) completa: {no_base}")
    print(f"Total de sequencias 'sentences' (augmentadas, sem base): {total_sentences}")
    print(f"Arquivo gerado: {args.out}")


if __name__ == "__main__":
    main()
