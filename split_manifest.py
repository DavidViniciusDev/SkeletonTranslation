#!/usr/bin/env python3
"""Quebra o manifest.json (Passo 4) em train.json, val.json e test.json.

Protocolo de avaliacao por INTERPRETE HELD-OUT (signer-independent):

  - Escolhe-se um conjunto de interpretes para TREINO+VALIDACAO
    (--train-interpreters) e outro, disjunto, reservado SO para TESTE
    (--test-interpreters). Assim o teste mede generalizacao para um sinalizador
    nunca visto no treino.

  - train.json: interpretes de treino, TODAS as augmentations (none, upsample,
                downsample, horizontal-flip, ...). E o unico conjunto que usa
                dados aumentados.
  - val.json:   interpretes de treino, SOMENTE base (augmentation == "none").
                As sentencas de val sao separadas das de treino no nivel de
                sentence_id (sem vazamento dentro do pool de treino).
  - test.json:  interpretes de teste, SOMENTE base (augmentation == "none").

Prevencao de vazamento:
  - A divisao train/val e feita por `sentence_id` (todas as variantes de uma
    mesma frase caem do mesmo lado), de forma deterministica (--seed).
  - test e naturalmente isolado por interprete. Opcionalmente
    (--test-exclude-train-sentences) removem-se do teste os `sentence_id` que
    tambem aparecem no treino, tornando o teste disjunto tambem em TEXTO — util
    para medir traducao sem memorizacao de sentenca (ver nota no relatorio).

O bloco 'config' original e preservado e enriquecido com os metadados do split.

Uso:
    python split_manifest.py --manifest manifest.json --out-dir splits \\
        --train-interpreters 0,1 --test-interpreters 2 --val-ratio 0.1 --seed 42

    # protocolo mais estrito (teste disjunto tambem em texto):
    python split_manifest.py --test-exclude-train-sentences
"""

import argparse
import json
import os
import random
from collections import Counter


def parse_interpreters(s):
    """'0,1' -> {'0','1'} (mantem como string, que e o tipo no manifesto)."""
    return {tok.strip() for tok in s.split(",") if tok.strip() != ""}


def is_base(item):
    return item.get("augmentation") == "none"


def sentence_id_of(item):
    # sentence_id e o identificador estavel da frase (compartilhado entre
    # interpretes e augmentations). Fallback para o texto se ausente.
    return item.get("sentence_id", item.get("pt_br"))


def main():
    ap = argparse.ArgumentParser(description="Split do manifest em train/val/test por interprete held-out.")
    ap.add_argument("--manifest", default="data/features/sentence_features/manifest.json",
                    help="manifest.json do Passo 4")
    ap.add_argument("--out-dir", default=None,
                    help="diretorio de saida (padrao: diretorio do manifest)")
    ap.add_argument("--train-interpreters", default="0,1",
                    help="interpretes usados em treino+validacao (ex.: '0,1')")
    ap.add_argument("--test-interpreters", default="2",
                    help="interprete(s) reservado(s) so para teste (ex.: '2')")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="fracao dos sentence_id de treino reservada para validacao")
    ap.add_argument("--seed", type=int, default=42, help="semente da divisao train/val")
    ap.add_argument("--test-exclude-train-sentences", action="store_true",
                    help="remove do teste os sentence_id que tambem aparecem no treino "
                         "(teste disjunto tambem em texto)")
    args = ap.parse_args()

    out_dir = args.out_dir or (os.path.dirname(os.path.abspath(args.manifest)) or ".")
    os.makedirs(out_dir, exist_ok=True)

    train_interp = parse_interpreters(args.train_interpreters)
    test_interp = parse_interpreters(args.test_interpreters)

    overlap = train_interp & test_interp
    if overlap:
        raise SystemExit(
            f"ERRO: interpretes {sorted(overlap)} estao em treino E teste. "
            f"O held-out exige conjuntos disjuntos.")

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    config = dict(manifest.get("config", {}))
    items = manifest.get("items", [])

    # Sanidade: quais interpretes existem de fato
    present = set(it.get("interpreter") for it in items)
    missing = (train_interp | test_interp) - present
    if missing:
        print(f"AVISO: interpretes pedidos ausentes no manifest: {sorted(missing)} "
              f"(presentes: {sorted(present)})")

    # 1) sentence_ids do pool de treino -> divide em train/val por sentenca
    train_pool_sids = sorted({
        sentence_id_of(it) for it in items if it.get("interpreter") in train_interp
    }, key=lambda x: (str(type(x)), x))
    rng = random.Random(args.seed)
    rng.shuffle(train_pool_sids)
    n_val = int(round(len(train_pool_sids) * args.val_ratio))
    val_sids = set(train_pool_sids[:n_val])
    train_sids = set(train_pool_sids[n_val:])

    # 2) monta os conjuntos
    train_items = [
        it for it in items
        if it.get("interpreter") in train_interp and sentence_id_of(it) in train_sids
    ]  # todas as augmentations
    val_items = [
        it for it in items
        if it.get("interpreter") in train_interp
        and sentence_id_of(it) in val_sids and is_base(it)
    ]  # so base
    test_items = [
        it for it in items
        if it.get("interpreter") in test_interp and is_base(it)
    ]  # so base

    if args.test_exclude_train_sentences:
        before = len(test_items)
        test_items = [it for it in test_items if sentence_id_of(it) not in train_sids]
        print(f"test: removidos {before - len(test_items)} itens cujo sentence_id "
              f"aparece no treino (--test-exclude-train-sentences)")

    # 3) metadados do split no config
    split_meta = {
        "protocol": "held-out-interpreter",
        "train_interpreters": sorted(train_interp),
        "test_interpreters": sorted(test_interp),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "test_exclude_train_sentences": args.test_exclude_train_sentences,
        "val_only_base": True,
        "test_only_base": True,
    }

    outputs = {
        "train.json": train_items,
        "val.json": val_items,
        "test.json": test_items,
    }
    for name, subset in outputs.items():
        payload = {"config": {**config, "split": {**split_meta, "subset": name.split(".")[0]}},
                   "items": subset}
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # --- Relatorio ---
    print(f"\nManifest: {args.manifest}  (itens totais: {len(items)})")
    print(f"Saida:    {out_dir}")
    print(f"Protocolo: treino/val = interpretes {sorted(train_interp)} | "
          f"teste = interpretes {sorted(test_interp)}")
    print(f"sentence_id no pool de treino: {len(train_pool_sids)} "
          f"(train={len(train_sids)}, val={len(val_sids)} @ ratio {args.val_ratio})")

    def report(name, subset):
        interp = Counter(it.get("interpreter") for it in subset)
        aug = Counter(it.get("augmentation") for it in subset)
        sids = {sentence_id_of(it) for it in subset}
        print(f"\n{name}: {len(subset)} itens | sentence_id unicos: {len(sids)}")
        print(f"  interpretes : {dict(interp)}")
        print(f"  augmentations: {dict(aug)}")
        return sids

    tr = report("train.json", train_items)
    va = report("val.json", val_items)
    te = report("test.json", test_items)

    # checagens de vazamento
    print("\n-- Checagem de vazamento (sentence_id) --")
    print(f"  train ∩ val  : {len(tr & va)}  (esperado 0)")
    if not args.test_exclude_train_sentences:
        print(f"  train ∩ test : {len(tr & te)}  (>0 e esperado: mesmo texto, "
              f"interprete diferente — held-out de sinalizador; use "
              f"--test-exclude-train-sentences para zerar)")
    else:
        print(f"  train ∩ test : {len(tr & te)}  (esperado 0)")


if __name__ == "__main__":
    main()
