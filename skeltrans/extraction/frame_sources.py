"""Fontes de frames RGB para a extração de landmarks (Passo 3).

Desacopla *de onde vêm os frames* de *como extrair landmarks*. Assim o s3
processa qualquer base que saiba entregar uma sequência de frames RGB:

  - vídeos .mp4/.mov/... (V-LIBRASIL)        -> video_frames()
  - diretórios de imagens .png/.jpg/... (PHOENIX-2014T) -> image_dir_frames()

Cada fonte é um GERADOR que produz arrays RGB (H, W, 3) na mesma convenção que
o MediaPipe espera (o inverso do BGR do OpenCV). O dispatcher open_frames()
escolhe a fonte pelo tipo do caminho — diretório vira sequência de imagens,
arquivo com extensão de vídeo vira vídeo — mantendo o comportamento atual do
LIBRAS inalterado (todo caminho hoje é um arquivo de vídeo).

Extensão futura: para um container exótico (shards .tar/WebDataset, HDF5, ...),
basta escrever mais um gerador e um ramo em open_frames(); nada no s3 muda.
"""

import os

# Extensões reconhecidas pelo dispatcher. Minúsculas; a comparação normaliza.
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def video_frames(path):
    """Gera frames RGB de um arquivo de vídeo (via OpenCV VideoCapture)."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"nao foi possivel abrir o video: {path}")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def image_dir_frames(path):
    """Gera frames RGB de um diretório de imagens, ordenadas por nome.

    A ordem lexicográfica dos nomes define a ordem temporal — o padrão dos
    datasets baseados em frames (ex.: PHOENIX nomeia `images0001.png`, ...).
    """
    import cv2

    names = sorted(
        f for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not names:
        raise ValueError(f"diretorio sem imagens reconhecidas: {path}")
    for name in names:
        full = os.path.join(path, name)
        bgr = cv2.imread(full)
        if bgr is None:
            raise IOError(f"nao foi possivel ler o frame: {full}")
        yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def open_frames(ref):
    """Escolhe a fonte de frames pelo tipo do caminho `ref`.

    - diretório existente                 -> image_dir_frames (base por frames)
    - arquivo com extensão de vídeo       -> video_frames (base por vídeo)

    Levanta ValueError se o caminho não casar com nenhuma fonte conhecida.
    """
    if os.path.isdir(ref):
        return image_dir_frames(ref)
    if os.path.splitext(ref)[1].lower() in VIDEO_EXTS:
        return video_frames(ref)
    raise ValueError(
        f"fonte de frames nao reconhecida: {ref} "
        f"(esperado um diretorio de imagens ou arquivo {sorted(VIDEO_EXTS)})"
    )
