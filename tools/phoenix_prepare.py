#!/usr/bin/env python3
"""Adaptador RWTH-PHOENIX-Weather -> artefatos do SkeletonTranslation.

Le as anotacoes oficiais do PHOENIX e emite os dois artefatos que faltam para
reaproveitar o pipeline existente (s3 + s4 + treino) sem alterar o pacote
`skeltrans`:

  1. `frames.json`  — lista de diretorios de frames na forma que o s3
     (`collect_videos`) espera. Como cada item do PHOENIX ja e uma FRASE
     CONTINUA, os passos de sintese Libras (s1a/s1b/s2/s5) nao se aplicam.
  2. `train.json` / `dev.json` / `test.json` — manifestos de treino no esquema
     legado de `LandmarkTextDataset`: {"features", "text", "tokens"}.
       - `text`   = traducao em alemao  (alvo do decoder; so existe no 2014T)
       - `tokens` = sequencia de glosas (alvo da supervisao auxiliar de CTC)

Usa os SPLITS OFICIAIS do PHOENIX (train/dev/test), e nao o protocolo de
interprete held-out do `split_manifest.py` — que e especifico do V-LIBRASIL e
pressupoe augmentations. Manter os splits oficiais e o que torna a metrica
comparavel a literatura.

Duas variantes de corpus:

  --variant 2014T (padrao)  PHOENIX-2014-T: tem `translation` (alemao) -> SLT.
                            Frames em `<split>/<name>/images0001.png`; o nome
                            do diretorio e unico, entao o s3 grava um .npy por
                            frase sem colisao.

  --variant 2014            PHOENIX-2014 (multisigner/SI5): NAO tem traducao,
                            so glosas -> o alvo passa a ser a glosa
                            (reconhecimento, nao traducao). Frames em
                            `<split>/<id>/1/*.png`: o basename de todos e "1",
                            e `out_name()` do s3 geraria "1.npy" para TODAS as
                            sequencias. Para evitar isso sem tocar no s3, sao
                            criados symlinks `<out>/frames/<id> -> .../<id>/1`.

Uso:
    python3 tools/phoenix_prepare.py \\
        --phoenix-root /caminho/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
        --out-dir data_phoenix

    # amostra estratificada p/ medir a taxa de deteccao antes da extracao longa
    python3 tools/phoenix_prepare.py --phoenix-root ... --out-dir data_phoenix \\
        --sample 50
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Nomes das colunas por variante. O PHOENIX usa CSV delimitado por '|'.
#   2014T: name|video|start|end|speaker|orth|translation
#   2014 : id|folder|signer|annotation
SCHEMAS = {
    "2014T": {"name": "name", "signer": "speaker", "gloss": "orth", "text": "translation"},
    "2014": {"name": "id", "signer": "signer", "gloss": "annotation", "text": None},
}

SPLITS = ("train", "dev", "test")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def read_corpus(csv_path, schema):
    """Le um <split>.corpus.csv do PHOENIX e devolve a lista de registros.

    Valida a presenca das colunas esperadas e rejeita linhas sem alvo textual,
    em vez de propagar um item vazio para o treino.
    """
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        missing = [c for c in schema.values() if c and c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"ERRO: {csv_path} nao tem as colunas {missing}. "
                f"Colunas encontradas: {reader.fieldnames}. "
                f"Confira se --variant casa com o corpus.")
        rows = []
        for lineno, row in enumerate(reader, start=2):
            name = (row.get(schema["name"]) or "").strip()
            gloss = (row.get(schema["gloss"]) or "").strip()
            # No 2014 nao existe traducao: o alvo textual passa a ser a glosa.
            text = (row.get(schema["text"]) or "").strip() if schema["text"] else gloss
            if not name:
                print(f"AVISO: {csv_path}:{lineno} sem identificador — ignorada")
                continue
            if not text:
                print(f"AVISO: {csv_path}:{lineno} ({name}) sem texto-alvo — ignorada")
                continue
            rows.append({
                "name": name,
                "signer": (row.get(schema["signer"]) or "").strip(),
                "gloss_tokens": gloss.split(),
                "text": text,
            })
    return rows


def frames_dir_for(phoenix_root, features_subdir, split, name, variant):
    """Diretorio que contem de fato os .png da sequencia `name`."""
    base = Path(phoenix_root) / "features" / features_subdir / split / name
    # No 2014 os frames vivem um nivel abaixo, em `<id>/1/`.
    return base / "1" if variant == "2014" else base


def has_images(path):
    try:
        with os.scandir(path) as it:
            return any(
                e.is_file() and os.path.splitext(e.name)[1].lower() in IMAGE_EXTS
                for e in it
            )
    except (FileNotFoundError, NotADirectoryError):
        return False


def link_for_2014(out_dir, name, real_dir):
    """Cria `<out>/frames/<name>` -> `real_dir`, contornando a colisao de nomes.

    O s3 nomeia a saida pelo basename do caminho; apontando para um link cujo
    basename e o `name` da sequencia, cada frase gera seu proprio .npy sem que
    o s3 precise mudar.
    """
    link_root = Path(out_dir) / "frames"
    link_root.mkdir(parents=True, exist_ok=True)
    link = link_root / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(real_dir).resolve(), target_is_directory=True)
    return link


def stride_sample(items, n):
    """Amostra `n` itens espalhados por toda a lista (passo constante).

    Deterministico e sem depender de aleatoriedade: cobre o corpus inteiro em
    vez dos N primeiros nomes em ordem alfabetica (que no PHOENIX seriam todos
    do mesmo mes e de poucos sinalizadores).
    """
    if n <= 0 or n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(
        description="Gera frames.json (entrada do s3) e os manifestos de treino a partir do PHOENIX.")
    ap.add_argument("--phoenix-root", required=True,
                    help="raiz do corpus: a pasta que contem features/ e annotations/ "
                         "(ex.: .../PHOENIX-2014-T-release-v3/PHOENIX-2014-T)")
    ap.add_argument("--out-dir", default="data_phoenix",
                    help="arvore de saida (interim/ e features/). Padrao: data_phoenix")
    ap.add_argument("--variant", choices=sorted(SCHEMAS), default="2014T",
                    help="2014T (com traducao alema -> SLT) ou 2014 (so glosas -> reconhecimento)")
    ap.add_argument("--features-subdir", default="fullFrame-210x260px",
                    help="subpasta de features/ com os frames. Padrao: fullFrame-210x260px")
    ap.add_argument("--landmarks-norm-dir", default=None,
                    help="pasta dos .npy normalizados (s4), usada para o caminho relativo "
                         "gravado nos manifestos. Padrao: <out-dir>/features/landmarks_115_norm9")
    ap.add_argument("--sample", type=int, default=0,
                    help="emite tambem frames.sample.json com N sequencias espalhadas por todo "
                         "o corpus, para medir custo/qualidade antes da extracao completa")
    args = ap.parse_args()

    root = Path(args.phoenix_root).resolve()
    annot_dir = root / "annotations" / "manual"
    if not annot_dir.is_dir():
        raise SystemExit(f"ERRO: nao encontrei {annot_dir}. --phoenix-root aponta para a raiz errada?")

    schema = SCHEMAS[args.variant]
    out_dir = Path(args.out_dir)
    interim = out_dir / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    norm_dir = Path(args.landmarks_norm_dir) if args.landmarks_norm_dir \
        else out_dir / "features" / "landmarks_115_norm9"

    if args.variant == "2014":
        print("AVISO: a variante 2014 nao possui traducao para lingua falada; o alvo textual\n"
              "       sera a SEQUENCIA DE GLOSAS. Isso e reconhecimento (sign2gloss), nao\n"
              "       traducao (sign2text). Para SLT use --variant 2014T.\n")

    all_dirs = []          # caminhos de frames p/ o frames.json (na ordem do corpus)
    manifest_counts = {}
    missing_total = 0

    for split in SPLITS:
        csv_path = annot_dir / f"{split}.corpus.csv"
        if not csv_path.is_file():
            print(f"AVISO: {csv_path} nao existe — split '{split}' ignorado")
            continue
        rows = read_corpus(csv_path, schema)

        items, missing = [], 0
        for rec in rows:
            real_dir = frames_dir_for(root, args.features_subdir, split, rec["name"], args.variant)
            if not has_images(real_dir):
                missing += 1
                print(f"AUSENTE: {split}/{rec['name']} -> {real_dir}")
                continue
            # O s3 nomeia o .npy pelo basename do caminho: no 2014 usamos um
            # symlink cujo basename e o nome da sequencia.
            frames_path = link_for_2014(out_dir, rec["name"], real_dir) \
                if args.variant == "2014" else real_dir
            all_dirs.append(str(frames_path))

            # Caminho relativo ao proprio manifesto: funciona com OU sem
            # --features-dir no treino (o dataset resolve os dois casos).
            feat_rel = os.path.relpath(norm_dir / f"{rec['name']}.npy", interim)
            items.append({
                "features": feat_rel,
                "text": rec["text"],
                "tokens": rec["gloss_tokens"],   # alvo do CTC (--use-ctc)
                "name": rec["name"],
                "signer": rec["signer"],
                "split": split,
            })

        out_path = interim / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(items, fh, ensure_ascii=False, indent=2)
        manifest_counts[split] = len(items)
        missing_total += missing
        print(f"{split:5s}: {len(items):5d} itens -> {out_path}"
              + (f"  ({missing} ausentes)" if missing else ""))

    if not all_dirs:
        raise SystemExit("ERRO: nenhuma sequencia valida encontrada — nada a fazer.")

    # frames.json: o s3 le rec['base']['videos'][*]['video']. Um unico registro
    # com todos os diretorios basta (collect_videos deduplica e ordena).
    frames_json = interim / "frames.json"
    payload = [{"base": {"videos": [{"video": d} for d in all_dirs]}}]
    with open(frames_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nframes.json: {len(all_dirs)} sequencias -> {frames_json}")

    if args.sample:
        sample = stride_sample(all_dirs, args.sample)
        sample_json = interim / "frames.sample.json"
        with open(sample_json, "w", encoding="utf-8") as fh:
            json.dump([{"base": {"videos": [{"video": d} for d in sample]}}],
                      fh, ensure_ascii=False, indent=2)
        print(f"frames.sample.json: {len(sample)} sequencias -> {sample_json}")

    total = sum(manifest_counts.values())
    print(f"\nTotal: {total} frases | ausentes: {missing_total} | variante: {args.variant}")
    print(f"Alvo textual: {'traducao (alemao)' if schema['text'] else 'glosas (reconhecimento)'}")
    print(f"\nProximo passo (extracao):\n"
          f"  python3 extract_landmarks.py --json {frames_json} "
          f"--out {out_dir}/raw/landmarks_115 --workers 20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
