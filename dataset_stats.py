#!/usr/bin/env python3
"""Estatísticas do dataset pt_br2libras_gloss_sentence_videos.json.

Uso:
    python dataset_stats.py [caminho_json]
"""

import json
import sys
import statistics
from collections import Counter


DEFAULT_PATH = "data/interim/pt_br2libras_gloss_sentence_videos.json"


def fmt(seconds):
    """Formata duração em segundos como m:ss."""
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:04.1f}s"


def main(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    n_sentences = len(data)

    # --- Vocabulário (a partir dos tokens da glosa) ---
    token_counter = Counter()
    words_per_sentence = []
    for entry in data:
        tokens = entry.get("tokens", [])
        token_counter.update(tokens)
        words_per_sentence.append(len(tokens))

    vocab_size = len(token_counter)
    total_tokens = sum(token_counter.values())

    # --- Durações dos vídeos ---
    # 'base' = versão original; 'sentences' = versões aumentadas.
    base_durations = []       # duração total por sentença (versão base)
    all_clip_durations = []   # duração de cada clipe de palavra individual
    aug_durations = []        # duração total por versão aumentada
    n_aug_versions = 0

    for entry in data:
        base = entry.get("base", {})
        if "total_duration" in base:
            base_durations.append(base["total_duration"])
        for vid in base.get("videos", []):
            all_clip_durations.append(vid.get("duration", 0.0))

        for sent in entry.get("sentences", []):
            n_aug_versions += 1
            if "total_duration" in sent:
                aug_durations.append(sent["total_duration"])
            for vid in sent.get("videos", []):
                all_clip_durations.append(vid.get("duration", 0.0))

    # --- Fontes governamentais ---
    gov = Counter(str(e.get("is_government_source")) for e in data)

    # --- Relatório ---
    print("=" * 60)
    print(f"ESTATÍSTICAS: {path}")
    print("=" * 60)

    print("\n## SENTENÇAS")
    print(f"  Número de sentenças:            {n_sentences}")
    print(f"  Fonte governamental (True/False): {dict(gov)}")

    print("\n## VOCABULÁRIO (glosas / tokens)")
    print(f"  Tamanho do vocabulário (únicos): {vocab_size}")
    print(f"  Total de tokens (com repetição): {total_tokens}")
    print(f"  Palavras por sentença - média:   {statistics.mean(words_per_sentence):.2f}")
    print(f"  Palavras por sentença - mediana: {statistics.median(words_per_sentence)}")
    print(f"  Palavras por sentença - min/max: {min(words_per_sentence)} / {max(words_per_sentence)}")

    print("\n  Top 10 palavras mais comuns:")
    for word, cnt in token_counter.most_common(10):
        print(f"    {cnt:5d}  {word}")

    print("\n  10 palavras menos comuns:")
    for word, cnt in token_counter.most_common()[-10:]:
        print(f"    {cnt:5d}  {word}")

    hapax = [w for w, c in token_counter.items() if c == 1]
    print(f"\n  Palavras que aparecem só 1 vez (hapax): {len(hapax)}")

    print("\n## VÍDEOS")
    print(f"  Versões aumentadas (augmentations): {n_aug_versions}")
    print(f"  Total de clipes de palavra:         {len(all_clip_durations)}")

    if base_durations:
        print("\n  Duração das sentenças (versão base):")
        print(f"    média:   {statistics.mean(base_durations):.2f}s  ({fmt(statistics.mean(base_durations))})")
        print(f"    mediana: {statistics.median(base_durations):.2f}s")
        print(f"    min/max: {min(base_durations):.2f}s / {max(base_durations):.2f}s")
        print(f"    total:   {sum(base_durations):.1f}s  ({fmt(sum(base_durations))})")

    if aug_durations:
        print("\n  Duração das versões aumentadas:")
        print(f"    média:   {statistics.mean(aug_durations):.2f}s")
        print(f"    min/max: {min(aug_durations):.2f}s / {max(aug_durations):.2f}s")

    if all_clip_durations:
        print("\n  Duração dos clipes individuais de palavra:")
        print(f"    média:   {statistics.mean(all_clip_durations):.2f}s")
        print(f"    mediana: {statistics.median(all_clip_durations):.2f}s")
        print(f"    min/max: {min(all_clip_durations):.2f}s / {max(all_clip_durations):.2f}s")

    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
