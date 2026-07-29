#!/usr/bin/env python3
"""Inferencia ponta-a-ponta: video .mp4 -> texto em portugues.

Executa todo o pipeline sobre um unico video de uma frase sinalizada:

    1. Extracao de landmarks (MediaPipe Holistic, 115 pontos)   [Etapa 1, s3]
    2. Normalizacao geometrica + dinamica -> (T,115,9)          [Etapa 1, s4]
    3. Encoder de landmarks + decoder PTT5 -> geracao de texto  [Etapa 2]

Nao aplica o blending de keyframes (usado apenas para SINTETIZAR frases a partir
de sinais isolados no treino): o video de entrada ja e uma sentenca continua.

Uso (pelo shim na raiz):
    python3 infer_slt.py \\
        --checkpoint checkpoints/best.pt \\
        --video /caminho/para/frase.mp4

    # varios videos de uma vez:
    python3 infer_slt.py --checkpoint checkpoints/best.pt --video a.mp4 b.mp4 c.mp4

Ou pelo modulo:
    python -m skeltrans.training.infer --checkpoint ... --video ...
"""

import argparse
import os

import numpy as np
import torch

from skeltrans.common.geometry import normalize_array
from skeltrans.extraction.steps.s3_extract_landmarks import process_video
from skeltrans.training.checkpoint import load_model, load_tokenizer
from skeltrans.training.device import resolve_device


def build_holistic():
    """Instancia o MediaPipe Holistic com a mesma config da extracao (s3)."""
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def video_to_features(video_path, holistic, origin="shoulders"):
    """video .mp4 -> (T, 1035) pronto para o encoder (Etapa 1, s3+s4)."""
    raw, _ = process_video(video_path, holistic)                    # (T,115,3)
    feats = normalize_array(raw.astype(np.float32), origin=origin)  # (T,115,9)
    return feats.reshape(feats.shape[0], -1)                        # (T,1035)


@torch.no_grad()
def translate(model, tokenizer, feats_np, device, num_beams=4, max_new_tokens=64):
    """(T,1035) -> string traduzida."""
    feats = torch.from_numpy(feats_np).unsqueeze(0).to(device)      # (1,T,1035)
    pad_mask = torch.zeros(1, feats.shape[1], dtype=torch.bool, device=device)  # sem padding
    ids = model.generate(feats, pad_mask,
                         max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


def run(args):
    device = resolve_device(args.device)
    print(f"Dispositivo: {device}")

    model, ckpt, t5_name = load_model(args.checkpoint, device, args.t5)
    tokenizer = load_tokenizer(args.checkpoint, t5_name, args.tokenizer)
    print(f"Checkpoint: {args.checkpoint} | T5: {t5_name}\n")

    holistic = build_holistic()
    try:
        for path in args.video:
            if not os.path.exists(path):
                print(f"[ERRO] video nao encontrado: {path}")
                continue
            try:
                feats = video_to_features(path, holistic, origin=args.origin)
                text = translate(model, tokenizer, feats, device,
                                 num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
            except Exception as e:  # noqa: BLE001 - reporta erro por video sem abortar o lote
                print(f"[ERRO] {os.path.basename(path)}: {e}")
                continue
            print(f"[{os.path.basename(path)}]  ({feats.shape[0]} frames)")
            print(f"  -> {text}\n")
    finally:
        holistic.close()


def build_parser():
    ap = argparse.ArgumentParser(description="Inferencia SLT: video .mp4 -> texto pt-br.")
    ap.add_argument("--checkpoint", required=True, help="caminho do .pt treinado")
    ap.add_argument("--video", required=True, nargs="+", help="um ou mais videos .mp4")
    ap.add_argument("--t5", default=None, help="override do nome/checkpoint do T5")
    ap.add_argument("--tokenizer", default=None, help="override do tokenizer")
    ap.add_argument("--origin", choices=["shoulders", "nose"], default="shoulders",
                    help="origem da centralizacao (deve casar com o treino; padrao shoulders)")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda|cpu (auto se omitido)")
    return ap


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
