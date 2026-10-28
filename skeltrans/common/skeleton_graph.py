"""Grafo do esqueleto de 115 pontos (para ``--encoder-type gcn``).

Nós = os 115 pontos de ``layout.py``, na mesma ordem do vetor de entrada.
Arestas = os ossos/contornos que ligam pontos vizinhos:

- Pose (0-32): conexões do MediaPipe Pose, mais nariz-cantos da boca e
  nariz-ombros. Nas conexões oficiais a boca fica solta e a cabeça inteira
  não se liga ao tronco; sem esse "pescoço", rosto e face ficariam isolados.
- Mãos (33-53, 54-74): as 21 conexões do MediaPipe Hands, em cada mão.
- Face (75-114): contornos dos lábios, de cada olho e de cada sobrancelha,
  restritos aos 40 pontos de FACE_MASK (as conexões oficiais do FaceMesh
  passam por pontos que não foram selecionados).
- Entre blocos: pulso da pose -> punho da mão (cada lado) e nariz da pose ->
  lábio superior e cantos internos dos olhos. Sem essas ligações, mãos, corpo
  e face seriam grafos separados.

As arestas são listadas à mão (não derivadas) e conferidas em
tests/test_encoder_gcn.py: um erro aqui não quebra o treino, só piora o
resultado em silêncio.
"""

from collections import deque

import numpy as np

from skeltrans.common.layout import (FACE, FACE_MASK, LEFT_HAND, L_SHOULDER, N_POINTS,
                                     NOSE, POSE, R_SHOULDER, RIGHT_HAND)

# --- pose: índices do MediaPipe Pose (base 0 do bloco) ---
POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),     # rosto
    (9, 10), (0, 9), (0, 10),                                           # boca (+ ao nariz)
    (0, 11), (0, 12),                                                   # "pescoço" (não oficial)
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),                   # ombros e braços
    (15, 17), (15, 19), (15, 21), (17, 19),                             # mão esq. da pose
    (16, 18), (16, 20), (16, 22), (18, 20),                             # mão dir. da pose
    (11, 23), (12, 24), (23, 24),                                       # tronco
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),                   # perna esq.
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),                   # perna dir.
]

# --- mão: índices do MediaPipe Hands (0 = punho) ---
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),                                     # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),                                     # indicador
    (5, 9), (9, 10), (10, 11), (11, 12),                                # médio
    (9, 13), (13, 14), (14, 15), (15, 16),                              # anelar
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),                    # mínimo + palma
]

# --- face: ids do FaceMesh, só os presentes em FACE_MASK, na ordem do contorno ---
FACE_EDGES_MESH = [
    # lábio externo superior (61 -> 291) e inferior (61 -> 291)
    (61, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270), (270, 291),
    (61, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 291),
    # lábio interno (78 -> 308, por cima e por baixo) e cantos interno-externo
    (78, 13), (13, 308), (78, 14), (14, 308), (61, 78), (291, 308),
    # olho esquerdo: pálpebra superior e inferior entre os cantos 33 e 133
    (33, 159), (159, 133), (33, 145), (145, 153), (153, 133),
    # olho direito: entre os cantos 263 e 362
    (263, 386), (386, 362), (263, 374), (374, 380), (380, 362),
    # sobrancelhas (cadeias) e ligação sobrancelha -> pálpebra superior
    (70, 63), (63, 105), (105, 66), (66, 107),
    (300, 293), (293, 334), (334, 296), (296, 336),
    (105, 159), (334, 386),
]

# --- ligações entre blocos: (nó da pose, id) ---
POSE_TO_HAND = [(15, LEFT_HAND.start), (16, RIGHT_HAND.start)]   # pulso da pose -> punho
POSE_TO_FACE_MESH = [(NOSE, 0), (NOSE, 133), (NOSE, 362)]        # nariz -> lábio sup., olhos

# Número de partições de vizinhança (estratégia "espacial" do ST-GCN):
# 0 = o próprio nó (e vizinhos à mesma distância do centro), 1 = vizinhos mais
# perto do centro do corpo (centrípetos), 2 = mais longe (centrífugos).
NUM_PARTITIONS = 3
CENTER_NODES = (POSE.start + L_SHOULDER, POSE.start + R_SHOULDER)


def _face_node(mesh_id):
    """Índice no vetor (75-114) do ponto `mesh_id` do FaceMesh."""
    return FACE.start + FACE_MASK.index(mesh_id)


def skeleton_edges():
    """Lista de arestas (i, j), i < j, em índices do vetor de 115 pontos."""
    edges = [(POSE.start + a, POSE.start + b) for a, b in POSE_EDGES]
    for hand in (LEFT_HAND, RIGHT_HAND):
        edges += [(hand.start + a, hand.start + b) for a, b in HAND_EDGES]
    edges += [(_face_node(a), _face_node(b)) for a, b in FACE_EDGES_MESH]
    edges += [(POSE.start + p, h) for p, h in POSE_TO_HAND]
    edges += [(POSE.start + p, _face_node(m)) for p, m in POSE_TO_FACE_MESH]
    edges = sorted({(min(i, j), max(i, j)) for i, j in edges})
    assert all(0 <= i < j < N_POINTS for i, j in edges), "aresta fora dos 115 nós"
    return edges


def hop_distance(edges, sources=CENTER_NODES, n=N_POINTS):
    """Distância em arestas de cada nó até o centro (os dois ombros), por BFS."""
    nbrs = [[] for _ in range(n)]
    for i, j in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    dist = np.full(n, -1, dtype=np.int64)
    queue = deque(sources)
    for s in sources:
        dist[s] = 0
    while queue:
        v = queue.popleft()
        for w in nbrs[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                queue.append(w)
    return dist


def partitioned_adjacency(edges=None, n=N_POINTS):
    """Matrizes de vizinhança (NUM_PARTITIONS, n, n), normalizadas.

    Cada aresta i-j entra na partição conforme a distância ao centro:
    j mais perto do centro que i -> centrípeta; mais longe -> centrífuga;
    igual -> junto com o próprio nó. A normalização é simétrica, pelo grau no
    grafo completo (com laço): Â_k = D^-1/2 A_k D^-1/2, de modo que a soma das
    partições é a vizinhança normalizada usual do GCN.
    """
    edges = skeleton_edges() if edges is None else edges
    dist = hop_distance(edges, n=n)
    if (dist < 0).any():
        raise ValueError(f"grafo desconexo: nós sem caminho até o centro {np.where(dist < 0)[0]}")
    A = np.zeros((NUM_PARTITIONS, n, n), dtype=np.float32)
    A[0] = np.eye(n, dtype=np.float32)
    for i, j in edges:
        for a, b in ((i, j), (j, i)):           # a recebe de b
            if dist[b] < dist[a]:
                A[1, a, b] = 1.0
            elif dist[b] > dist[a]:
                A[2, a, b] = 1.0
            else:
                A[0, a, b] = 1.0
    deg = A.sum(axis=(0, 2))                    # grau com laço
    d = 1.0 / np.sqrt(deg)
    return A * d[None, :, None] * d[None, None, :]
