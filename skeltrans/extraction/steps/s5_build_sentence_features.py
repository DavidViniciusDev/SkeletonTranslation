#!/usr/bin/env python3
"""Passo 4 do plano: Sintese de Frases e Interpolacao (Keyframe Blending).

Consome o arquivo de sentencas ja resolvido em videos
(`pt_br2libras_gloss_sentence_videos.json`, produzido pelos itens 4.1-4.3) e,
para cada sentenca valida, concatena os landmarks dos sinais isolados inserindo
uma janela de transicao suave entre sinais consecutivos. O resultado e uma
sequencia continua por sentenca (um arquivo .npy) pronta para alimentar o
Encoder do Passo 5.

Pipeline por sentenca/variante:
  1. Para cada token da frase, carrega o .npy do sinal isolado a partir de
     --landmarks-dir (o nome do arquivo e o basename do video sem .mp4).
  2. Entre o fim do sinal atual e o inicio do proximo, cria K frames de
     transicao (K = --transition-frames, padrao 5) interpolando as coordenadas:
        - lerp : interpolacao linear entre o ultimo frame de A e o primeiro de B
        - cubic: spline cubica (scipy) usando uma janela de ancoras em cada lado
        - none : sem transicao (colagem "seca" -> controle do ablation study)
  3. Concatena tudo numa sequencia continua.
  4. feature-mode:
        - positions : salva posicoes (T,115,3)
        - normalized: aplica a normalizacao geometrica + dinamica do Passo 3
                      SOBRE a sequencia continua (mais correto que normalizar
                      cada sinal isolado, pois as derivadas atravessam as
                      transicoes) -> (T,115,9)

Tratamento de ausentes: pontos ausentes vem como [0,0,0] (Passo 2). Um ponto so
e interpolado na transicao quando esta presente NOS DOIS lados da fronteira;
caso contrario permanece [0,0,0], preservando a semantica de "ausente" e
evitando movimento fantasma em direcao a origem.

Variantes (augmentation): cada sentenca do JSON traz `base` (augmentation
"none") e uma lista `sentences` (upsample, downsample, horizontal-flip, ...).
  --variants all  -> gera base + todas as variantes (data augmentation)
  --variants base -> gera apenas a base limpa (augmentation "none")

Saidas (em --out-dir):
  <out-dir>/<id>.npy                          uma sequencia por sentenca/variante
  --manifest (JSON)                           mapeamento feature -> frase + metadados
  <manifest sem ext>.csv                      versao plana p/ o DataLoader
  <manifest sem ext>.stats.json               taxa de descarte e contagens (paper)

O campo `sentence_id` no manifesto identifica a sentenca de origem: TODAS as
variantes de augmentation de uma mesma frase compartilham o mesmo id, para que
o split treino/val/teste (Passo 6) seja feito por sentenca e nao vaze dados.

Uso:
    python3 build_sentence_features.py \
        --sentences-json pt_br2libras_gloss_sentence_videos.json \
        --landmarks-dir landmarks_115 \
        --out-dir sentence_features \
        --manifest sentence_features/manifest.json

    # controle do ablation study (colagem seca, sem interpolacao):
    python3 build_sentence_features.py --interp-mode none --out-dir sentence_features_dry

    # teste rapido em poucas sentencas:
    python3 build_sentence_features.py --limit 20
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

# Garante a raiz (onde vive o pacote skeltrans/) no path, mesmo se o modulo for
# invocado por caminho direto. Raiz = 4 niveis acima (steps/extraction/skeltrans/).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Layout e geometria: fonte unica em skeltrans/common/ (antes, normalize_array
# vinha de normalize_landmarks.py — dependencia script->script, agora removida).
from skeltrans.common.layout import N_COORDS  # noqa: E402
from skeltrans.common.npy_io import load_points  # noqa: E402
from skeltrans.common.geometry import (  # noqa: E402
    present_mask_frame,
    present_mask_frame_stack,
)
from skeltrans.extraction.config import PATHS  # noqa: E402
try:
    from skeltrans.common.geometry import normalize_array  # noqa: E402  (Passo 3)
except Exception as e:  # pragma: no cover
    normalize_array = None
    _NORM_IMPORT_ERR = e


# --------------------------------------------------------------------------- #
# I/O de landmarks
# --------------------------------------------------------------------------- #
def video_to_npy(video_path, landmarks_dir):
    """Mapeia o caminho do video para o .npy correspondente na pasta de landmarks."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(landmarks_dir, stem + ".npy")


def load_sign(video_path, landmarks_dir):
    """Carrega (T,115,3) de um sinal isolado. Levanta FileNotFoundError se ausente."""
    path = video_to_npy(video_path, landmarks_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load_points(path, n_coords=N_COORDS, dtype=np.float64)


# present_mask_frame / present_mask_frame_stack: importados de skeltrans.common.geometry


# --------------------------------------------------------------------------- #
# Transicoes (Keyframe Blending)
# --------------------------------------------------------------------------- #
def transition_lerp(a, b, k):
    """K frames de transicao por interpolacao linear entre a[-1] e b[0].

    a, b: (T,115,3). Retorna (K,115,3). Ponto so interpolado se presente nos
    dois lados; caso contrario fica [0,0,0].
    """
    end, start = a[-1], b[0]                               # (115,3)
    both = present_mask_frame(end) & present_mask_frame(start)  # (115,)
    alphas = (np.arange(1, k + 1) / (k + 1.0))[:, None, None]    # (K,1,1)
    trans = (1.0 - alphas) * end[None] + alphas * start[None]    # (K,115,3)
    trans[:, ~both, :] = 0.0
    return trans


def transition_cubic(a, b, k, anchor):
    """K frames de transicao por spline cubica (scipy) com janela de ancoras.

    Usa os ultimos `anchor` frames de A e os primeiros `anchor` de B como nos,
    deixando um vao de K frames no meio; a spline cubica preenche o vao. Pontos
    sem presenca completa nos nos recorrem ao lerp; pontos ausentes na fronteira
    ficam zerados.
    """
    from scipy.interpolate import CubicSpline

    wa = min(anchor, a.shape[0])
    wb = min(anchor, b.shape[0])
    a_tail = a[-wa:]                                       # (wa,115,3)
    b_head = b[:wb]                                        # (wb,115,3)

    # Eixo temporal: A ocupa ...,-1,0 ; vao 1..K ; B ocupa K+1,...
    t_a = np.arange(-wa + 1, 1)                            # termina em 0
    t_b = np.arange(k + 1, k + 1 + wb)                     # comeca em K+1
    t_knots = np.concatenate([t_a, t_b])                  # (wa+wb,)
    y = np.concatenate([a_tail, b_head], axis=0)          # (wa+wb,115,3)
    t_eval = np.arange(1, k + 1)                           # (K,)

    cs = CubicSpline(t_knots, y, axis=0)
    trans = cs(t_eval)                                    # (K,115,3)

    # Fallback lerp e mascara de presenca, por ponto.
    end, start = a[-1], b[0]
    both = present_mask_frame(end) & present_mask_frame(start)   # (115,)
    knots_present = np.all(present_mask_frame_stack(y), axis=0)   # (115,) presente em TODOS os nos
    lerp = transition_lerp(a, b, k)                              # (K,115,3)
    use_lerp = both & ~knots_present
    trans[:, use_lerp, :] = lerp[:, use_lerp, :]
    trans[:, ~both, :] = 0.0
    return trans


def build_sequence(videos, landmarks_dir, k, interp_mode, anchor):
    """Concatena os sinais de uma variante numa sequencia continua (T,115,3).

    videos: lista de dicts com chave 'video'. Levanta FileNotFoundError no
    primeiro sinal ausente (a variante inteira e entao descartada pelo chamador).
    Retorna (seq, meta) onde meta traz contagens de frames e transicoes.
    """
    segments = []
    n_sign_frames = 0
    n_trans_frames = 0
    prev = None
    for v in videos:
        arr = load_sign(v["video"], landmarks_dir)
        if prev is not None and k > 0 and interp_mode != "none":
            if interp_mode == "cubic":
                trans = transition_cubic(prev, arr, k, anchor)
            else:
                trans = transition_lerp(prev, arr, k)
            segments.append(trans)
            n_trans_frames += trans.shape[0]
        segments.append(arr)
        n_sign_frames += arr.shape[0]
        prev = arr

    seq = np.concatenate(segments, axis=0)
    meta = {
        "num_signs": len(videos),
        "sign_frames": int(n_sign_frames),
        "transition_frames": int(n_trans_frames),
    }
    return seq, meta


# --------------------------------------------------------------------------- #
# Selecao de variantes
# --------------------------------------------------------------------------- #
def iter_variants(sentence, which):
    """Gera (collection_dict) das variantes escolhidas de uma sentenca.

    `base` = augmentation "none"; `sentences` = variantes augmentadas.
    which: 'all' -> base + sentences ; 'base' -> apenas base.
    """
    base = sentence.get("base")
    if base:
        yield base
    if which == "all":
        for s in sentence.get("sentences", []):
            yield s


def safe_id(idx, interpreter, augmentation):
    """Nome de arquivo estavel e unico por (sentenca, interprete, augmentation)."""
    aug = augmentation if augmentation else "none"
    aug = aug.replace(" ", "-")
    return f"sent{idx:06d}_i{interpreter}_{aug}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Passo 4: Keyframe Blending -> features continuas por sentenca.")
    ap.add_argument("--sentences-json", default=PATHS.sentence_videos_json,
                    help="JSON de sentencas resolvidas em videos (saida dos itens 4.1-4.3).")
    ap.add_argument("--landmarks-dir", default=PATHS.landmarks_dir,
                    help="Pasta com os .npy dos sinais isolados (T,115,3).")
    ap.add_argument("--out-dir", default=PATHS.features_dir,
                    help="Pasta de saida das sequencias continuas (.npy).")
    ap.add_argument("--manifest", default=None,
                    help="Caminho do manifesto JSON. Padrao: <out-dir>/manifest.json")
    ap.add_argument("--transition-frames", type=int, default=5,
                    help="K frames de transicao entre sinais (padrao 5).")
    ap.add_argument("--interp-mode", choices=["lerp", "cubic", "none"], default="lerp",
                    help="Tipo de interpolacao. 'none' = colagem seca (ablation).")
    ap.add_argument("--feature-mode", choices=["positions", "normalized"], default="normalized",
                    help="'positions' (T,115,3) ou 'normalized' (T,115,9) do Passo 3.")
    ap.add_argument("--variants", choices=["all", "base"], default="all",
                    help="'all' = base + augmentations; 'base' = so a base limpa.")
    ap.add_argument("--anchor", type=int, default=3,
                    help="Janela de frames de ancora em cada lado para a spline cubica.")
    ap.add_argument("--origin", choices=["shoulders", "nose"], default="shoulders",
                    help="Origem da centralizacao (repassado ao Passo 3).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Processa apenas as N primeiras sentencas (0 = todas). Util p/ teste.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Regrava .npy existentes (padrao: pula).")
    args = ap.parse_args()

    if args.feature_mode == "normalized" and normalize_array is None:
        sys.exit(f"ERRO: nao consegui importar normalize_landmarks: {_NORM_IMPORT_ERR}")

    manifest_path = args.manifest or os.path.join(args.out_dir, "manifest.json")
    stem = os.path.splitext(manifest_path)[0]
    csv_path = stem + ".csv"
    stats_path = stem + ".stats.json"

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.sentences_json, encoding="utf-8") as fh:
        sentences = json.load(fh)
    if args.limit:
        sentences = sentences[: args.limit]

    print(f"Sentencas de entrada : {len(sentences)}")
    print(f"Landmarks            : {args.landmarks_dir}")
    print(f"Saida                : {args.out_dir}")
    print(f"Config               : K={args.transition_frames} interp={args.interp_mode} "
          f"feat={args.feature_mode} variants={args.variants}")

    records = []
    n_variants_total = 0
    n_generated = 0
    n_skipped_missing = 0
    n_skipped_exists = 0
    n_error = 0
    missing_signs = {}  # token/video -> contagem de descartes causados

    for idx, sent in enumerate(sentences):
        pt_br = sent.get("pt-br", "")
        gloss = sent.get("libras-gloss", "")
        english = sent.get("english_translation", "")
        tokens = sent.get("tokens", [])

        for coll in iter_variants(sent, args.variants):
            n_variants_total += 1
            interpreter = coll.get("interpreter", "0")
            augmentation = coll.get("augmentation", "none")
            videos = coll.get("videos", [])
            if not videos:
                continue

            fid = safe_id(idx, interpreter, augmentation)
            out_npy = os.path.join(args.out_dir, fid + ".npy")

            if not args.overwrite and os.path.exists(out_npy):
                n_skipped_exists += 1
                continue

            # Constroi a sequencia; descarta a variante inteira se faltar algum sinal.
            try:
                seq, seq_meta = build_sequence(
                    videos, args.landmarks_dir,
                    args.transition_frames, args.interp_mode, args.anchor,
                )
            except FileNotFoundError as e:
                n_skipped_missing += 1
                key = os.path.basename(str(e))
                missing_signs[key] = missing_signs.get(key, 0) + 1
                continue
            except Exception as e:
                n_error += 1
                print(f"[{idx}] ERRO em {fid}: {e}")
                continue

            if args.feature_mode == "normalized":
                feats = normalize_array(seq.astype(np.float32), origin=args.origin)
            else:
                feats = seq.astype(np.float32)

            np.save(out_npy, feats)
            n_generated += 1

            records.append({
                "id": fid,
                "feature_file": os.path.relpath(out_npy, os.path.dirname(manifest_path) or "."),
                "sentence_id": idx,
                "pt_br": pt_br,
                "english_translation": english,
                "libras_gloss": gloss,
                "tokens": tokens,
                "num_tokens": len(tokens),
                "interpreter": interpreter,
                "augmentation": augmentation,
                "num_frames": int(feats.shape[0]),
                "feature_dim": int(feats.shape[1] * feats.shape[2]),
                "n_points": int(feats.shape[1]),
                "n_channels": int(feats.shape[2]),
                "sign_frames": seq_meta["sign_frames"],
                "transition_frames": seq_meta["transition_frames"],
                "source_videos": [os.path.basename(v["video"]) for v in videos],
            })

            if n_generated % 200 == 0:
                print(f"  ... geradas {n_generated} sequencias")

    # ----- manifesto (JSON rico) ----- #
    config = {
        "sentences_json": args.sentences_json,
        "landmarks_dir": args.landmarks_dir,
        "out_dir": args.out_dir,
        "transition_frames": args.transition_frames,
        "interp_mode": args.interp_mode,
        "feature_mode": args.feature_mode,
        "variants": args.variants,
        "anchor": args.anchor,
        "origin": args.origin,
        "numpy": np.__version__,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"config": config, "items": records}, fh, ensure_ascii=False, indent=2)

    # ----- CSV plano p/ DataLoader ----- #
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "feature_file", "sentence_id", "pt_br", "libras_gloss",
                    "interpreter", "augmentation", "num_frames", "n_channels", "num_tokens"])
        for r in records:
            w.writerow([r["id"], r["feature_file"], r["sentence_id"], r["pt_br"],
                        r["libras_gloss"], r["interpreter"], r["augmentation"],
                        r["num_frames"], r["n_channels"], r["num_tokens"]])

    # ----- estatisticas p/ o paper ----- #
    discard_rate = (n_skipped_missing / n_variants_total) if n_variants_total else 0.0
    top_missing = sorted(missing_signs.items(), key=lambda kv: -kv[1])[:30]
    stats = {
        "config": config,
        "input_sentences": len(sentences),
        "variants_considered": n_variants_total,
        "generated": n_generated,
        "skipped_missing_landmark": n_skipped_missing,
        "skipped_already_exists": n_skipped_exists,
        "errors": n_error,
        "discard_rate_missing_landmark": round(discard_rate, 4),
        "unique_missing_signs": len(missing_signs),
        "top_missing_signs": [{"sign": k, "discarded_variants": v} for k, v in top_missing],
    }
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print("\n== Concluido ==")
    print(f"Variantes consideradas : {n_variants_total}")
    print(f"Sequencias geradas     : {n_generated}")
    print(f"Descartadas (faltou npy): {n_skipped_missing}  (taxa {discard_rate:.2%})")
    print(f"Puladas (ja existiam)  : {n_skipped_exists}")
    print(f"Erros                  : {n_error}")
    print(f"Manifesto : {manifest_path}")
    print(f"CSV       : {csv_path}")
    print(f"Stats     : {stats_path}")


if __name__ == "__main__":
    main()
