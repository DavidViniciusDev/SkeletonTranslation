#!/usr/bin/env python3
"""Testes das correções da extração:

1. Estado do MediaPipe Holistic zerado antes de cada vídeo (s3 e infer).
2. Correção de proporção da imagem (--aspect) na normalização (s4/s5/infer).

Sem MediaPipe, OpenCV nem dados: o Holistic é substituído por um falso que
guarda estado entre frames, como o modo vídeo do verdadeiro.

Uso:
    python3 tests/test_extraction_fixes.py
    (também compatível com: pytest tests/test_extraction_fixes.py)
"""
import os
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skeltrans.common.geometry import correct_aspect, normalize_array, parse_aspect  # noqa: E402
from skeltrans.common.layout import L_SHOULDER, N_POINTS, R_SHOULDER  # noqa: E402

import skeltrans.extraction.frame_sources as frame_sources  # noqa: E402
import skeltrans.extraction.steps.s3_extract_landmarks as s3  # noqa: E402

PHOENIX = 210 / 260


def _pixels(T=12, seed=0, W=210, H=260):
    """Esqueleto sintético em PIXELS (x em [0,W], y em [0,H], z na escala de x)."""
    rng = np.random.default_rng(seed)
    px = np.zeros((T, N_POINTS, 3))
    px[..., 0] = rng.uniform(0.2 * W, 0.8 * W, (T, N_POINTS))
    px[..., 1] = rng.uniform(0.2 * H, 0.8 * H, (T, N_POINTS))
    px[..., 2] = rng.uniform(-0.1 * W, 0.1 * W, (T, N_POINTS))
    px[:, L_SHOULDER] = [0.65 * W, 0.45 * H, 0.0]
    px[:, R_SHOULDER] = [0.35 * W, 0.45 * H, 0.0]
    return px


def _mediapipe_coords(px, W=210, H=260):
    """Como o MediaPipe devolve: x/W, y/H e z na escala de x (z/W)."""
    out = px.copy()
    out[..., 0] /= W
    out[..., 1] /= H
    out[..., 2] /= W
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. Correção de proporção
# --------------------------------------------------------------------------- #
def test_default_is_unchanged():
    """Sem aspect (ou aspect=1), a saída é idêntica, bit a bit, à de antes."""
    raw = _mediapipe_coords(_pixels())
    base = normalize_array(raw)
    assert np.array_equal(base, normalize_array(raw, aspect=None))
    assert np.array_equal(base, normalize_array(raw, aspect=1.0))
    assert np.array_equal(base, normalize_array(raw, origin="shoulders"))


def test_aspect_recovers_geometry_in_pixels():
    """Corrigir a proporção dá o mesmo que normalizar os pontos em pixels:
    a única diferença entre os dois é uma escala uniforme, que a normalização
    pelos ombros remove."""
    px = _pixels()
    raw = _mediapipe_coords(px)
    corrected = normalize_array(raw, aspect=PHOENIX)
    reference = normalize_array(px.astype(np.float32))
    assert np.allclose(corrected, reference, atol=1e-5)
    # e sem a correção a geometria fica, de fato, distorcida
    assert not np.allclose(normalize_array(raw), reference, atol=1e-3)


def test_aspect_fixes_angles():
    """Um segmento a 45° em pixels só aparece a 45° depois da correção."""
    px = _pixels(T=1)
    px[0, 40] = px[0, 33] + [20.0, 20.0, 0.0]          # mão esq.: ponto 7 a 45° do punho
    raw = _mediapipe_coords(px)

    def angle(feats):
        v = feats[0, 40, :2] - feats[0, 33, :2]
        return np.degrees(np.arctan2(v[1], v[0]))

    assert abs(angle(normalize_array(raw, aspect=PHOENIX)) - 45.0) < 1e-3
    assert abs(angle(normalize_array(raw)) - 45.0) > 5.0   # ~39° no espaço esticado


def test_absent_points_stay_zero():
    raw = _mediapipe_coords(_pixels())
    raw[3:6, 33:54] = 0.0                              # mão esq. ausente em 3 frames
    out = normalize_array(raw, aspect=PHOENIX)
    assert np.all(out[3:6, 33:54] == 0.0)
    assert np.all(correct_aspect(raw, PHOENIX)[3:6, 33:54] == 0.0)


def test_parse_aspect():
    assert abs(parse_aspect("210x260") - PHOENIX) < 1e-12
    assert abs(parse_aspect("1280X720") - 1280 / 720) < 1e-12
    assert parse_aspect("0.8") == 0.8
    for bad in ("abc", "210x0", "0", "-1", "x260"):
        try:
            parse_aspect(bad)
        except ValueError:
            continue
        raise AssertionError(f"deveria recusar {bad!r}")


# --------------------------------------------------------------------------- #
# 1. Estado do Holistic entre vídeos
# --------------------------------------------------------------------------- #
def _landmarks(n, value):
    lm = types.SimpleNamespace(x=value, y=value, z=0.0, visibility=1.0)
    return types.SimpleNamespace(landmark=[lm] * n)


class HolisticFalso:
    """Imita o modo vídeo: a saída de cada frame depende do frame anterior
    (como o rastreamento e a suavização do verdadeiro)."""

    def __init__(self, com_reset=True):
        self.estado = 0.0
        self.fechado = False
        if not com_reset:
            self.reset = None                           # simula versão sem reset()

    def process(self, rgb):
        self.estado = 0.5 * self.estado + float(rgb.mean())   # memória do passado
        return types.SimpleNamespace(
            pose_landmarks=_landmarks(33, self.estado), left_hand_landmarks=None,
            right_hand_landmarks=None, face_landmarks=None)

    def reset(self):                                    # noqa: F811 (sobrescrito acima se None)
        self.estado = 0.0

    def close(self):
        self.fechado = True


def _videos():
    return {"A": [np.full((4, 4, 3), 200, np.uint8)] * 5,      # vídeo "claro"
            "B": [np.full((4, 4, 3), 10, np.uint8)] * 5}       # vídeo "escuro"


def _extrair(sequencia, holistic):
    """Roda o _process_one do s3 para cada vídeo, com o mesmo Holistic de worker."""
    videos = _videos()
    saved = frame_sources.open_frames
    frame_sources.open_frames = lambda ref: iter(videos[ref])
    s3._WORKER_HOLISTIC = holistic
    out_dir = tempfile.mkdtemp()
    try:
        res = {}
        for vid in sequencia:
            dst = os.path.join(out_dir, f"{vid}.npy")
            status = s3._process_one((vid, vid, dst, out_dir, False))
            assert status[1] == "ok", status
            res[vid] = np.load(dst)
        return res
    finally:
        frame_sources.open_frames = saved


def test_video_result_does_not_depend_on_previous_video():
    sozinho = _extrair(["B"], HolisticFalso())["B"]
    depois_de_a = _extrair(["A", "B"], HolisticFalso())["B"]
    assert np.array_equal(sozinho, depois_de_a)


def test_without_reset_recreates_instance():
    """Sem reset(): fecha a instância antiga e cria outra pelo build_holistic."""
    criados = []
    saved = s3.build_holistic
    s3.build_holistic = lambda: criados.append(HolisticFalso(com_reset=False)) or criados[-1]
    try:
        inicial = HolisticFalso(com_reset=False)
        sozinho = _extrair(["B"], HolisticFalso(com_reset=False))["B"]
        depois_de_a = _extrair(["A", "B"], inicial)["B"]
    finally:
        s3.build_holistic = saved
    assert inicial.fechado
    assert np.array_equal(sozinho, depois_de_a)


def test_old_behavior_really_leaked():
    """Controle: sem zerar o estado, o vídeo B herda o A (o bug corrigido)."""
    videos = _videos()
    h = HolisticFalso()
    s3.process_frames(iter(videos["A"]), h)
    vazado, _ = s3.process_frames(iter(videos["B"]), h)
    limpo, _ = s3.process_frames(iter(videos["B"]), HolisticFalso())
    assert not np.array_equal(vazado, limpo)


def test_infer_uses_same_holistic_config():
    import skeltrans.training.infer as infer
    assert infer.build_holistic is s3.build_holistic
    assert infer.fresh_holistic is s3.fresh_holistic
    assert infer.resolve_aspect(None, "x.mp4") is None
    assert abs(infer.resolve_aspect("210x260", "x.mp4") - PHOENIX) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} testes passaram")
