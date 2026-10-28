#!/usr/bin/env python3
"""Testes do encoder por grafo (--encoder-type gcn) e do grafo do esqueleto.

Dados sintéticos e um T5 minúsculo criado do zero: não exige a base nem
download de pesos.

Uso:
    python3 tests/test_encoder_gcn.py
    (também compatível com: pytest tests/test_encoder_gcn.py)
"""
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skeltrans.common.layout import (FACE, INPUT_DIM, LEFT_HAND, N_CHANNELS,  # noqa: E402
                                     N_POINTS, POSE, RIGHT_HAND)
from skeltrans.common.skeleton_graph import (hop_distance, partitioned_adjacency,  # noqa: E402
                                             skeleton_edges)
from skeltrans.training.data.features import feature_dim  # noqa: E402
from skeltrans.training.models.graph_projection import (GraphProjection,  # noqa: E402
                                                        node_presence)
from skeltrans.training.models.landmark_encoder import LandmarkEncoder  # noqa: E402

LH, RH = LEFT_HAND.start, RIGHT_HAND.start


def _tiny_t5():
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=64, d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1,
                   num_heads=2, d_kv=16, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1)
    return T5ForConditionalGeneration(cfg)


def _model(**kw):
    from skeltrans.training.models import SLTModel
    torch.manual_seed(0)
    return SLTModel(_tiny_t5(), d_model=16, nhead=2, num_layers=1, **kw)


def _frames(B=2, T=6, seed=0):
    """Entrada sintética com todos os pontos presentes (nenhum canal = 0)."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, T, INPUT_DIM, generator=g) + 0.1


# --------------------------------------------------------------------------- #
# Grafo
# --------------------------------------------------------------------------- #
def test_graph_matches_hand_written_edges():
    """Confere arestas-chave contra uma lista escrita à mão."""
    e = set(skeleton_edges())
    expected = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # ombros e braços
        (0, 11), (0, 12),                                   # pescoço acrescentado
        (15, LH), (16, RH),                                 # pulso da pose -> punho
        (LH, LH + 1), (LH + 3, LH + 4), (LH + 7, LH + 8),   # polegar e indicador esq.
        (LH, LH + 17), (LH + 5, LH + 9),                    # palma
        (RH + 19, RH + 20),                                 # mínimo dir.
    ]
    missing = [x for x in expected if x not in e]
    assert not missing, f"arestas esperadas ausentes: {missing}"
    # nada liga as mãos diretamente à face ou uma mão à outra
    for i, j in e:
        blocks = {b for b, sl in (("lh", LEFT_HAND), ("rh", RIGHT_HAND), ("face", FACE))
                  for v in (i, j) if sl.start <= v < sl.stop}
        assert blocks not in ({"lh", "face"}, {"rh", "face"}, {"lh", "rh"}), (i, j)


def test_graph_counts():
    e = skeleton_edges()
    in_block = lambda sl: sum(sl.start <= i < sl.stop and sl.start <= j < sl.stop for i, j in e)  # noqa: E731
    assert in_block(LEFT_HAND) == in_block(RIGHT_HAND) == 21   # 21 conexões do MediaPipe Hands
    # face: lábios 16 + 4 + 2, olhos 5 + 5, sobrancelhas 4 + 4, sobrancelha-pálpebra 2
    assert in_block(FACE) == 42
    assert in_block(POSE) == 39                                  # 35 oficiais + boca-nariz 2 + pescoço 2
    # 39 + 2 x 21 + 42 + entre blocos (2 pulso-punho + 3 nariz-face)
    assert len(e) == len(set(e)) == 39 + 42 + 42 + 5 == 128


def test_graph_connected_and_hops():
    dist = hop_distance(skeleton_edges())
    assert (dist >= 0).all(), "grafo desconexo"
    assert dist[11] == dist[12] == 0                    # ombros = centro
    assert dist[LH] == dist[RH] == 3                    # ombro -> cotovelo -> pulso -> punho
    assert dist[LH + 8] == 7                            # ponta do indicador


def test_adjacency_normalized_and_partitioned():
    A = partitioned_adjacency()
    assert A.shape == (3, N_POINTS, N_POINTS) and np.isfinite(A).all()
    S = A.sum(0)
    assert np.allclose(S, S.T)                          # simétrica
    assert np.all(np.diag(A[0]) > 0)                    # laço na partição 0
    # cada direção de cada aresta cai em exatamente uma partição
    nz = (A > 0).sum(0)
    for i, j in skeleton_edges():
        assert nz[i, j] == 1 and nz[j, i] == 1
    # centrípeta e centrífuga são transpostas uma da outra
    assert np.array_equal(A[1] > 0, (A[2] > 0).T)


# --------------------------------------------------------------------------- #
# Projeção
# --------------------------------------------------------------------------- #
def test_shapes_with_and_without_extras():
    p = GraphProjection(INPUT_DIM, d_model=32)
    assert p.extra is None and p.part_dim == 8 and p.hidden_dim == 4
    assert p(_frames()).shape == (2, 6, 32)
    D = feature_dim("A,F")
    pe = GraphProjection(D, d_model=32)
    assert pe.extra is not None and pe.extra_dim == D - INPUT_DIM
    assert pe(torch.rand(2, 6, D) + 0.1).shape == (2, 6, 32)


def test_rejects_short_input():
    try:
        GraphProjection(100, d_model=32)
    except ValueError:
        return
    raise AssertionError("entrada com menos de 1035 colunas deveria falhar")


def test_absent_nodes_stay_zero_and_dont_dilute_neighbors():
    """Mão ausente: seus nós ficam 0 e o punho da pose não é puxado para 0.

    Com entrada constante, a média renormalizada sobre os vizinhos presentes
    dá o mesmo valor com ou sem um vizinho ausente (desde que reste algum
    vizinho na mesma partição).
    """
    torch.manual_seed(0)
    p = GraphProjection(INPUT_DIM, d_model=32).eval()
    x = torch.ones(1, 1, INPUT_DIM)
    y = x.clone()
    nodes = y.view(1, 1, N_POINTS, N_CHANNELS)
    nodes[:, :, LH + 5] = 0.0                            # só a base do indicador some
    with torch.no_grad():
        h1, _ = p.node_features(x)
        h2, pres = p.node_features(y)
    assert pres[0, 0, LH + 5] == 0 and torch.all(h2[0, 0, LH + 5] == 0)
    # o punho (vizinhos centrífugos: 1, 5, 17) ainda tem 1 e 17 presentes; na
    # 1ª camada o valor agregado é o mesmo -> confere a saída da 1ª camada
    layer = p.gcn_layers[0]
    h0 = x.view(1, 1, N_POINTS, N_CHANNELS)
    with torch.no_grad():
        a = layer(h0, p.A, p.B, node_presence(x).float())
        b = layer(y.view(1, 1, N_POINTS, N_CHANNELS), p.A, p.B, node_presence(y).float())
    assert torch.allclose(a[0, 0, LH], b[0, 0, LH], atol=1e-6)


def test_whole_hand_absent_pools_to_zero():
    p = GraphProjection(INPUT_DIM, d_model=32).eval()
    x = _frames(1, 3)
    x.view(1, 3, N_POINTS, N_CHANNELS)[:, :, LEFT_HAND] = 0.0
    seen = {}
    # hook que devolve algo != None substituiria a saída: só registra
    p.fuse.register_forward_hook(lambda m, i, o: seen.update(x=i[0]))
    with torch.no_grad():
        p(x)
    pooled = seen["x"].view(1, 3, 4, p.part_dim)         # pose, mão esq, mão dir, face
    assert torch.all(pooled[:, :, 1] == 0)
    assert torch.all(pooled[:, :, 2].abs().sum(-1) > 0)


def test_hands_independent_within_receptive_field():
    """Com 3 camadas (alcance de 3 saltos), mudar a mão esquerda não altera
    os nós da mão direita: o caminho entre elas passa por 7 arestas."""
    p = GraphProjection(INPUT_DIM, d_model=32).eval()
    x = _frames(1, 2)
    y = x.clone()
    y.view(1, 2, N_POINTS, N_CHANNELS)[:, :, LEFT_HAND] += 2.0
    with torch.no_grad():
        hx, _ = p.node_features(x)
        hy, _ = p.node_features(y)
    assert torch.allclose(hx[..., RIGHT_HAND, :], hy[..., RIGHT_HAND, :])
    assert not torch.allclose(hx[..., LEFT_HAND, :], hy[..., LEFT_HAND, :])


def test_skipping_padding_gives_identical_output():
    """Rodar o GCN só nos frames reais dá o mesmo resultado que o lote inteiro,
    inclusive nas posições de padding (e com o fluxo de extras)."""
    D = feature_dim("A,F")
    torch.manual_seed(0)
    p = GraphProjection(D, d_model=32).eval()
    x = torch.rand(3, 9, D) + 0.1
    pad = torch.zeros(3, 9, dtype=torch.bool)
    pad[0, 6:] = True
    pad[2, 2:] = True
    x[pad] = 0.0                                        # padding = frames zerados (collate)
    with torch.no_grad():
        assert torch.allclose(p(x), p(x, pad_mask=pad), atol=1e-6)
    # e o gradiente também chega igual aos pesos
    p.train()
    p(x).sum().backward()
    g1 = [q.grad.clone() for q in p.parameters()]
    p.zero_grad()
    p(x, pad_mask=pad).sum().backward()
    assert all(torch.allclose(a, q.grad, atol=1e-5) for a, q in zip(g1, p.parameters()))


def test_grad_checkpoint_same_gradients():
    torch.manual_seed(0)
    p = GraphProjection(INPUT_DIM, d_model=32).train()
    x = _frames(2, 5)
    p(x).sum().backward()
    g1 = [q.grad.clone() for q in p.parameters()]
    p.zero_grad()
    p.grad_checkpoint = True
    p(x).sum().backward()
    assert all(torch.allclose(a, q.grad, atol=1e-6) for a, q in zip(g1, p.parameters()))


def test_presence_uses_raw_input_with_input_norm():
    """Com --input-norm, a presença precisa vir da entrada ANTES do z-score."""
    enc = LandmarkEncoder(d_model=16, nhead=2, num_layers=1, input_norm=True,
                          encoder_type="gcn").eval()
    enc.set_input_stats(np.full(INPUT_DIM, 0.5), np.full(INPUT_DIM, 2.0))
    feats = _frames(1, 4)
    feats.view(1, 4, N_POINTS, N_CHANNELS)[:, :, RIGHT_HAND] = 0.0
    seen = {}
    enc.input_proj.register_forward_hook(
        lambda m, args, kwargs, out: seen.update(present=kwargs["present"]),
        with_kwargs=True)
    with torch.no_grad():
        enc(feats, torch.zeros(1, 4, dtype=torch.bool))
    present = seen["present"][0]
    assert not present[:, RIGHT_HAND].any()                 # mão dir. reconhecida como ausente
    assert present[:, POSE].all() and present[:, LEFT_HAND].all()


def test_forward_backward_all_params_get_grad():
    model = _model(encoder_type="gcn", ds_steps=1, extra_features="A,F")
    D = feature_dim("A,F")
    feats = torch.rand(2, 30, D) + 0.1
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[1, 20:] = True
    labels = torch.randint(2, 64, (2, 6))
    loss, _ = model(feats, mask, labels)
    loss.backward()
    proj = model.encoder.input_proj
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in proj.parameters())
    assert proj.B.grad.abs().sum() > 0                      # o grafo aprendido também aprende
    assert model.generate(feats, mask, max_new_tokens=4, num_beams=1).shape[0] == 2


def test_other_encoder_types_unchanged():
    """flat e parts não ganham chaves do gcn."""
    for et in ("flat", "parts"):
        keys = set(_model(encoder_type=et).state_dict())
        assert not any(k.startswith("encoder.input_proj.gcn_layers.") for k in keys), et


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def _roundtrip(model, meta_model):
    from skeltrans.training.checkpoint import load_model, save_checkpoint
    tmp = tempfile.mkdtemp()
    t5_path = os.path.join(tmp, "t5")
    model.t5.save_pretrained(t5_path)
    meta = {"model": {"t5_name": t5_path, "d_model": 16, "nhead": 2, "num_layers": 1,
                      **meta_model}}
    ck = os.path.join(tmp, "m.pt")
    save_checkpoint(ck, model, 1, meta)
    return load_model(ck, torch.device("cpu"))[0], ck, meta


def test_gcn_checkpoint_roundtrip():
    model = _model(encoder_type="gcn", extra_features="A,F", input_norm=True)
    D = feature_dim("A,F")
    model.encoder.set_input_stats(np.full(D, 0.1), np.full(D, 1.5))
    with torch.no_grad():
        model.encoder.input_proj.B.add_(0.01)               # grafo aprendido != 0
    loaded, _, _ = _roundtrip(model, {"extra_features": "A,F", "input_norm": True,
                                      "encoder_type": "gcn"})
    assert loaded.encoder.encoder_type == "gcn"
    assert torch.equal(loaded.encoder.input_proj.B, model.encoder.input_proj.B)
    feats = torch.rand(1, 12, D) + 0.1
    mask = torch.zeros(1, 12, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model.encoder(feats, mask)[0], loaded.encoder(feats, mask)[0])


def test_gcn_detected_from_weights():
    """Os pesos mandam: config 'flat' com pesos 'gcn' carrega como 'gcn'."""
    loaded, _, _ = _roundtrip(_model(encoder_type="gcn"), {"encoder_type": "flat"})
    assert loaded.encoder.encoder_type == "gcn"


def test_gcn_checkpoint_dim_mismatch_fails():
    from skeltrans.training.checkpoint import load_model, save_checkpoint
    model = _model(encoder_type="gcn", extra_features="A,F")
    _, ck, meta = _roundtrip(model, {"extra_features": "A,F"})
    meta["model"]["extra_features"] = "A"                   # grupos que não batem com os pesos
    save_checkpoint(ck, model, 1, meta)
    try:
        load_model(ck, torch.device("cpu"))
    except ValueError:
        return
    raise AssertionError("checkpoint inconsistente deveria falhar")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} testes passaram")
