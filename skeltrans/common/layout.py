"""Layout dos landmarks — fonte ÚNICA de verdade da geometria dos 115 pontos.

Toda a pipeline (extração e treino) importa daqui, em vez de redefinir as
mesmas constantes. Antes, N_POINTS/N_CHANNELS, as fatias dos grupos e os
índices de referência estavam duplicados em extract_landmarks.py,
normalize_landmarks.py, build_sentence_features.py e slt_model.py.

Ordem canônica dos 115 pontos por frame:
    [  0: 33) Pose          -> 33 pontos
    [ 33: 54) Mão esquerda  -> 21 pontos
    [ 54: 75) Mão direita   -> 21 pontos
    [ 75:115) Face (máscara) -> 40 pontos
Total: 33 + 21 + 21 + 40 = 115.
"""

# --- contagem por grupo ---
N_POSE = 33
N_HAND = 21
N_FACE = 40
N_POINTS = N_POSE + 2 * N_HAND + N_FACE          # 115

# --- coordenadas e canais ---
N_COORDS = 3                                      # X, Y, Z (saída do Passo 2)
N_CHANNELS = 9                                    # X,Y,Z, Vx,Vy,Vz, Ax,Ay,Az (Passo 3)
INPUT_DIM = N_POINTS * N_CHANNELS                 # 1035 (entrada do encoder, Passo 5)

# --- fatias (slices) de cada grupo no eixo dos pontos ---
POSE = slice(0, N_POSE)                                       # [0:33)
LEFT_HAND = slice(N_POSE, N_POSE + N_HAND)                    # [33:54)
RIGHT_HAND = slice(N_POSE + N_HAND, N_POSE + 2 * N_HAND)      # [54:75)
FACE = slice(N_POSE + 2 * N_HAND, N_POINTS)                  # [75:115)

# --- índices de referência (dentro do bloco Pose, base 0) ---
NOSE = 0
L_SHOULDER = 11
R_SHOULDER = 12

# --- máscara facial do MediaPipe (40 índices do FaceMesh) ---
# Focada em olhos, sobrancelhas e lábios (expressões não-manuais).
# Todos < 468, válidos com refine_face_landmarks=True.
FACE_LIPS = [
    61, 291, 0, 17, 37, 39, 40, 267, 269, 270,      # contorno externo (superior/inferior)
    84, 91, 181, 314, 321, 405,                      # contorno externo inferior
    13, 14, 78, 308,                                 # contorno interno (centro/cantos)
]                                                    # 20 pontos
FACE_LEFT_EYE = [33, 133, 159, 145, 153]             # cantos + palpebra sup/inf
FACE_RIGHT_EYE = [362, 263, 386, 374, 380]
FACE_LEFT_EYEBROW = [70, 63, 105, 66, 107]
FACE_RIGHT_EYEBROW = [336, 296, 334, 293, 300]
FACE_MASK = (
    FACE_LIPS
    + FACE_LEFT_EYE + FACE_RIGHT_EYE
    + FACE_LEFT_EYEBROW + FACE_RIGHT_EYEBROW
)
assert len(FACE_MASK) == N_FACE, f"mascara facial tem {len(FACE_MASK)} pontos, esperado {N_FACE}"
assert len(set(FACE_MASK)) == N_FACE, "ha indices repetidos na mascara facial"
