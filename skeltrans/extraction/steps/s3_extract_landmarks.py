#!/usr/bin/env python3
"""Extrai landmarks (esqueleto) de cada item referenciado usando MediaPipe
Holistic. Cada item pode ser um arquivo de VIDEO (V-LIBRASIL) ou um DIRETORIO
de frames (ex.: PHOENIX-2014T) — a fonte e escolhida automaticamente por
skeltrans.extraction.frame_sources.open_frames. Referencia default: os videos
de pt_br2libras_gloss_sentence_videos.json.

Configuracao do MediaPipe:
    model_complexity=2, refine_face_landmarks=True

Saida por frame: matriz (115, 3) com (X, Y, Z) normalizados, na ordem:
    [  0: 33)  Pose          -> 33 pontos
    [ 33: 54)  Mao esquerda  -> 21 pontos
    [ 54: 75)  Mao direita   -> 21 pontos
    [ 75:115)  Face (mascara)-> 40 pontos (olhos, sobrancelhas, labios)
Total: 33 + 21 + 21 + 40 = 115 pontos.

Cada video vira um arquivo .npy com shape (T, 115, 3) (float32), nomeado pelo
nome do video (sem extensao). Frames/componentes ausentes (ex.: maos nao
detectadas) sao preenchidos com vetores zero [0.0, 0.0, 0.0] para evitar
congelamento.

Observacao sobre a Pose: o MediaPipe fornece tambem 'visibility', mas o shape
exigido e (T, 115, 3); portanto guardamos apenas (X, Y, Z). Use --with-visibility
para salvar a visibilidade da pose num arquivo paralelo *_pose_vis.npy (T, 33).

Processamento paralelo: usa um pool de PROCESSOS (padrao 8). O MediaPipe Holistic
nao e thread-safe e e CPU-bound, entao processos (e nao threads) dao speedup real;
cada processo cria sua propria instancia do Holistic. O progresso e exibido ao vivo.

Uso:
    python3 extract_landmarks.py                 # processa tudo (com resume), 8 processos
    python3 extract_landmarks.py --workers 4     # ajusta o numero de processos
    python3 extract_landmarks.py --limit 5       # processa apenas 5 videos
    python3 extract_landmarks.py --video X.mp4   # processa um unico video
    python3 extract_landmarks.py --overwrite     # reprocessa mesmo se .npy existir
"""
import argparse
import json
import os
import sys

import numpy as np

# Paralelismo + progresso: fonte unica em skeltrans/common/parallel.py
from skeltrans.common.parallel import fmt_eta, run_pool

# Reduz o ruido de log do MediaPipe/glog e do ffmpeg (OpenCV) para manter a
# barra de progresso limpa. -8 = AV_LOG_QUIET (silencia avisos de swscaler/decode).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

# ---------------------------------------------------------------------------
# Configuracao (caminhos default centralizados em skeltrans/extraction/config.py)
# ---------------------------------------------------------------------------
from skeltrans.extraction.config import PATHS

JSON_PATH = PATHS.sentence_videos_json
OUT_DIR = PATHS.landmarks_dir

# Diretorios alternativos: os videos sem augmentation (base/teste) nao estao em
# videos_words_augmentation/, e sim nestas pastas. Se o caminho do JSON nao
# existir, procura-se o mesmo nome de arquivo nestes diretorios (em ordem).
FALLBACK_DIRS = [
    "/libras/v-librasil/videos_words",
    "/libras/v-librasil/regular_videos_words",
]


def resolve_video(path):
    """Resolve o caminho real do video, tentando os diretorios de fallback."""
    if os.path.exists(path):
        return path
    bn = os.path.basename(path)
    for d in FALLBACK_DIRS:
        cand = os.path.join(d, bn)
        if os.path.exists(cand):
            return cand
    return None

# Layout dos 115 pontos e mascara facial: fonte unica em skeltrans/common/layout.py
from skeltrans.common.layout import (  # noqa: E402
    FACE_MASK,
    N_FACE,
    N_HAND,
    N_POINTS as N_TOTAL,
    N_POSE,
)


# ---------------------------------------------------------------------------
# Coleta dos videos
# ---------------------------------------------------------------------------
def collect_videos(json_path):
    """Retorna a lista ordenada de caminhos de video individuais (sem repeticao)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    vids = set()
    for rec in data:
        if rec.get("base"):
            for v in rec["base"]["videos"]:
                vids.add(v["video"])
        for seq in rec.get("sentences", []):
            for v in seq["videos"]:
                vids.add(v["video"])
    return sorted(vids)


# ---------------------------------------------------------------------------
# Extracao por frame
# ---------------------------------------------------------------------------
def landmarks_xyz(landmark_list, indices, n_expected):
    """Converte um landmark_list do MediaPipe em array (len(indices ou n), 3).

    Se landmark_list for None, retorna zeros. `indices` (ou None para todos).
    """
    if indices is None:
        n = n_expected
    else:
        n = len(indices)
    if landmark_list is None:
        return np.zeros((n, 3), dtype=np.float32)
    lms = landmark_list.landmark
    out = np.zeros((n, 3), dtype=np.float32)
    if indices is None:
        for i in range(min(n, len(lms))):
            lm = lms[i]
            out[i] = (lm.x, lm.y, lm.z)
    else:
        for j, idx in enumerate(indices):
            if idx < len(lms):
                lm = lms[idx]
                out[j] = (lm.x, lm.y, lm.z)
    return out


def pose_visibility(results):
    if results.pose_landmarks is None:
        return np.zeros((N_POSE,), dtype=np.float32)
    return np.array(
        [lm.visibility for lm in results.pose_landmarks.landmark[:N_POSE]],
        dtype=np.float32,
    )


def frame_to_matrix(results):
    """Monta a matriz (115, 3) de um frame a partir do resultado do Holistic."""
    pose = landmarks_xyz(results.pose_landmarks, None, N_POSE)
    left = landmarks_xyz(results.left_hand_landmarks, None, N_HAND)
    right = landmarks_xyz(results.right_hand_landmarks, None, N_HAND)
    face = landmarks_xyz(results.face_landmarks, FACE_MASK, N_FACE)
    return np.concatenate([pose, left, right, face], axis=0)  # (115, 3)


def process_frames(frames, holistic, with_visibility=False):
    """Extrai landmarks de uma SEQUENCIA de frames RGB (H,W,3).

    `frames` e qualquer iteravel de arrays RGB — vindo de um video, de um
    diretorio de imagens, etc. (ver skeltrans.extraction.frame_sources). Assim
    o s3 nao conhece a fonte: so o formato dos frames.

    Retorna (arr (T,115,3), vis (T,33) ou None).
    """
    matrices = []
    vis_frames = [] if with_visibility else None
    for rgb in frames:
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        matrices.append(frame_to_matrix(results))
        if with_visibility:
            vis_frames.append(pose_visibility(results))
    if not matrices:
        raise ValueError("fonte sem frames legiveis")
    arr = np.stack(matrices, axis=0).astype(np.float32)  # (T, 115, 3)
    vis = np.stack(vis_frames, axis=0).astype(np.float32) if with_visibility else None
    return arr, vis


def process_video(path, holistic, with_visibility=False):
    """Compatibilidade: processa um unico arquivo de VIDEO.

    Mantido para chamadores que passam um caminho de video (ex.: infer.py).
    Delega para process_frames() sobre a fonte de video. Para bases baseadas em
    diretorios de imagens, use open_frames()/process_frames() diretamente.
    """
    from skeltrans.extraction.frame_sources import video_frames

    return process_frames(video_frames(path), holistic, with_visibility)


def out_name(video_path):
    # normpath remove barra final: assim um diretorio de frames ".../1/" vira
    # o nome "1" (e um arquivo "X.mp4" continua virando "X").
    return os.path.splitext(os.path.basename(os.path.normpath(video_path)))[0]


# ---------------------------------------------------------------------------
# Worker de processo (pool)
# ---------------------------------------------------------------------------
_WORKER_HOLISTIC = None  # instancia por processo (criada uma unica vez)


def _init_worker():
    """Inicializador de cada processo: cria uma instancia propria do Holistic."""
    global _WORKER_HOLISTIC
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp

    _WORKER_HOLISTIC = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _process_one(task):
    """Processa um video (executado em processo worker). Retorna (vid, status, info)."""
    vid, real, dst, out_dir, with_visibility = task
    try:
        from skeltrans.extraction.frame_sources import open_frames

        arr, vis = process_frames(open_frames(real), _WORKER_HOLISTIC, with_visibility)
        np.save(dst, arr)
        if vis is not None:
            np.save(os.path.join(out_dir, out_name(vid) + "_pose_vis.npy"), vis)
        return (vid, "ok", arr.shape)
    except Exception as e:  # noqa: BLE001 - reporta erro por video sem derrubar o pool
        return (vid, "error", str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Extrai landmarks 115x3 por video (MediaPipe Holistic).")
    ap.add_argument("--json", default=JSON_PATH)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--video", default=None,
                    help="processa um unico caminho: arquivo de video (ex.: X.mp4) OU "
                         "diretorio de frames (imagens ordenadas por nome)")
    ap.add_argument("--limit", type=int, default=None, help="processa apenas os N primeiros")
    ap.add_argument("--overwrite", action="store_true", help="reprocessa mesmo se o .npy existir")
    ap.add_argument("--with-visibility", action="store_true",
                    help="salva tambem a visibilidade da pose em *_pose_vis.npy (T,33)")
    ap.add_argument("--workers", type=int, default=8, help="numero de processos paralelos (padrao: 8)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.video:
        videos = [args.video]
    else:
        videos = collect_videos(args.json)
        if args.limit:
            videos = videos[: args.limit]

    try:
        import mediapipe  # noqa: F401 - apenas verifica disponibilidade
    except ImportError:
        sys.exit("mediapipe nao instalado. Instale com: pip install mediapipe opencv-python")

    # Pre-filtragem (resume + ausentes) no processo principal: monta a lista de
    # tarefas efetivas e ja contabiliza pulados/ausentes.
    tasks = []
    skipped = missing = 0
    for vid in videos:
        dst = os.path.join(args.out, out_name(vid) + ".npy")
        if not args.overwrite and os.path.exists(dst):
            skipped += 1
            continue
        real = resolve_video(vid)
        if real is None:
            missing += 1
            print(f"AUSENTE: {vid}")
            continue
        tasks.append((vid, real, dst, args.out, args.with_visibility))

    total = len(tasks)
    workers = max(1, min(args.workers, total)) if total else 1
    print(f"Videos referenciados: {len(videos)} | a processar: {total} | "
          f"pulados(ja existiam): {skipped} | ausentes: {missing} | processos: {workers}")

    if total == 0:
        print("Nada a processar.")
        return

    ok, failed, elapsed = run_pool(
        tasks, _process_one, workers=workers, unit="vid/s", initializer=_init_worker)
    print(f"\nConcluido em {fmt_eta(elapsed)}. processados={ok} falhas={failed} "
          f"pulados(ja existiam)={skipped} ausentes={missing}")


if __name__ == "__main__":
    main()
