#!/usr/bin/env python3
"""Testes do encoder por conceitos (--temporal-mode concept).

Dados sintéticos e um T5 minúsculo criado do zero: não exige a base nem
download de pesos.

Uso:
    python3 tests/test_encoder_concept.py
    (também compatível com: pytest tests/test_encoder_concept.py)
"""
import argparse
import itertools
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skeltrans.common.layout import INPUT_DIM  # noqa: E402
from skeltrans.training.models.concept_pooling import (ConceptPooling,  # noqa: E402
                                                       forced_align_spikes,
                                                       greedy_spikes, spikes_to_seg_ids)
from skeltrans.training.models.landmark_encoder import LandmarkEncoder  # noqa: E402

V = 8                                   # vocabulário de glosas fictício (0 = blank)


def _tiny_t5():
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=64, d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1,
                   num_heads=2, d_kv=16, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1)
    return T5ForConditionalGeneration(cfg)


def _model(**kw):
    from skeltrans.training.models import SLTModel
    torch.manual_seed(0)
    kw.setdefault("use_ctc", True)
    kw.setdefault("gloss_vocab_size", V)
    return SLTModel(_tiny_t5(), d_model=16, nhead=2, num_layers=1, **kw)


def _batch(lengths=(30, 18), S=(3, 2), seed=0):
    g = torch.Generator().manual_seed(seed)
    B, T = len(lengths), max(lengths)
    feats = torch.rand(B, T, INPUT_DIM, generator=g) + 0.1
    mask = torch.ones(B, T, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = False
    gloss = torch.zeros(B, max(S), dtype=torch.long)
    for i, s in enumerate(S):
        gloss[i, :s] = torch.randint(1, V, (s,), generator=g)
    labels = torch.randint(2, 64, (B, 6), generator=g)
    return feats, mask, labels, gloss, torch.tensor(S)


def _peaky_log_probs(T, spikes_at, labels):
    """Log-probs em que o blank domina, salvo `labels[i]` no frame `spikes_at[i]`."""
    lp = torch.full((T, V), -8.0)
    lp[:, 0] = -0.01
    for t, k in zip(spikes_at, labels):
        lp[t, 0] = -8.0
        lp[t, k] = -0.01
    return lp


# --------------------------------------------------------------------------- #
# Disparos e segmentos
# --------------------------------------------------------------------------- #
def test_greedy_spikes_follow_ctc_collapse():
    """'A A _ A B B' -> disparos em A(0), A(3), B(4): igual ao predict_glosses."""
    best = torch.tensor([[1, 1, 0, 1, 2, 2, 0, 0]])
    logits = torch.nn.functional.one_hot(best, V).float()
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 7] = True
    spikes = greedy_spikes(logits, mask)
    assert spikes[0].nonzero().flatten().tolist() == [0, 3, 4]


def test_seg_ids_cut_at_midpoints():
    T = 120
    spikes = torch.zeros(1, T, dtype=torch.bool)
    spikes[0, [12, 51, 93]] = True
    mask = torch.zeros(1, T, dtype=torch.bool)
    mask[0, 110:] = True
    seg = spikes_to_seg_ids(spikes, mask)[0]
    # cortes em (12+51)//2 = 31 e (51+93)//2 = 72
    assert seg[:31].eq(0).all() and seg[31:72].eq(1).all() and seg[72:110].eq(2).all()
    assert seg[110:].eq(-1).all()


def test_seg_ids_without_spikes_is_one_concept():
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[1, 4:] = True
    seg = spikes_to_seg_ids(torch.zeros(2, 10, dtype=torch.bool), mask)
    assert seg[0].eq(0).all()
    assert seg[1, :4].eq(0).all() and seg[1, 4:].eq(-1).all()


def test_forced_align_finds_the_peaks():
    lp = _peaky_log_probs(20, [3, 9, 15], [4, 2, 6])[None]
    mask = torch.zeros(1, 20, dtype=torch.bool)
    spikes = forced_align_spikes(lp, mask, torch.tensor([[4, 2, 6]]), torch.tensor([3]))
    assert spikes[0].nonzero().flatten().tolist() == [3, 9, 15]


def test_forced_align_one_spike_per_gloss_even_untrained():
    """CTC aleatório (início do treino): ainda assim exatamente S disparos."""
    torch.manual_seed(1)
    lp = torch.randn(3, 25, V).log_softmax(-1)
    mask = torch.zeros(3, 25, dtype=torch.bool)
    mask[1, 15:] = True
    targets = torch.tensor([[1, 1, 3, 5], [2, 7, 0, 0], [6, 0, 0, 0]])   # inclui glosa repetida
    tl = torch.tensor([4, 2, 1])
    spikes = forced_align_spikes(lp, mask, targets, tl)
    assert spikes.sum(dim=1).tolist() == [4, 2, 1]
    assert not spikes[1, 15:].any()                                     # nada no padding


def test_forced_align_batched_equals_single():
    torch.manual_seed(2)
    lp = torch.randn(2, 16, V).log_softmax(-1)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[1, 9:] = True
    targets = torch.tensor([[3, 4, 3], [5, 6, 0]])
    tl = torch.tensor([3, 2])
    both = forced_align_spikes(lp, mask, targets, tl)
    for b in range(2):
        L = int((~mask[b]).sum())
        one = forced_align_spikes(lp[b:b + 1, :L], mask[b:b + 1, :L],
                                  targets[b:b + 1, :tl[b]], tl[b:b + 1])
        assert torch.equal(both[b, :L], one[0]), b


def _collapse(seq):
    out, prev = [], None
    for k in seq:
        if k != prev and k != 0:
            out.append(k)
        prev = k
    return out


def test_forced_align_matches_brute_force():
    """Em T pequeno, o Viterbi acha o mesmo caminho que a busca exaustiva."""
    torch.manual_seed(3)
    T, target = 6, [2, 2]                                  # repetida: exige blank no meio
    lp = torch.randn(T, V).log_softmax(-1)
    best_score, best_path = None, None
    for path in itertools.product([0, 2], repeat=T):
        if _collapse(path) != target:
            continue
        score = sum(float(lp[t, k]) for t, k in enumerate(path))
        if best_score is None or score > best_score:
            best_score, best_path = score, path
    expected = [t for t, k in enumerate(best_path)
                if k != 0 and (t == 0 or best_path[t - 1] != k)]
    spikes = forced_align_spikes(lp[None], torch.zeros(1, T, dtype=torch.bool),
                                 torch.tensor([target]), torch.tensor([2]))
    assert spikes[0].nonzero().flatten().tolist() == expected


def test_forced_align_infeasible_falls_back_to_one_concept():
    """2 glosas iguais precisam de 3 frames (A _ A); com 2 frames não há caminho."""
    lp = torch.randn(1, 2, V).log_softmax(-1)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    spikes = forced_align_spikes(lp, mask, torch.tensor([[3, 3]]), torch.tensor([2]))
    assert not spikes.any()
    assert spikes_to_seg_ids(spikes, mask)[0].tolist() == [0, 0]


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #
def test_pooling_shapes_and_mask():
    pool = ConceptPooling(8, queries=2)
    x = torch.randn(2, 10, 8)
    seg = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
                        [0, 0, 1, 1, 1, 1, -1, -1, -1, -1]])
    out, mask = pool(x, seg)
    assert out.shape == (2, 3 * 2, 8)
    assert mask.tolist() == [[False] * 6, [False] * 4 + [True] * 2]


def test_pooling_constant_segment_returns_the_constant():
    """Pesos somam 1 por segmento: frames iguais -> o próprio vetor."""
    pool = ConceptPooling(4)
    v0, v1 = torch.randn(4), torch.randn(4)
    x = torch.stack([v0, v0, v0, v1, v1])[None]
    out, _ = pool(x, torch.tensor([[0, 0, 0, 1, 1]]))
    assert torch.allclose(out[0, 0], v0, atol=1e-6) and torch.allclose(out[0, 1], v1, atol=1e-6)


def test_pooling_segments_and_padding_are_isolated():
    pool = ConceptPooling(4)
    x = torch.randn(1, 8, 4)
    seg = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]])
    ref, _ = pool(x, seg)
    y = x.clone()
    y[0, 3:5] += 5.0                                     # mexe só no sinal 1
    y[0, 6:] = 100.0                                     # e no padding
    out, _ = pool(y, seg)
    assert torch.allclose(out[0, 0], ref[0, 0])
    assert not torch.allclose(out[0, 1], ref[0, 1])


# --------------------------------------------------------------------------- #
# Encoder / SLTModel
# --------------------------------------------------------------------------- #
def test_stride_has_no_concept_keys_and_same_output():
    enc = LandmarkEncoder(d_model=16, nhead=2, num_layers=1, ds_steps=1)
    assert not any("concept" in k for k in enc.state_dict())
    enc.eval()
    feats, mask, *_ = _batch()
    with torch.no_grad():
        mem, m = enc(feats, mask)
        x, m2 = enc.encode_frames(feats, mask)
    assert torch.equal(mem, enc.adapter(x)) and torch.equal(m, m2)


def test_concept_requires_ctc_and_valid_options():
    for kw in ({"use_ctc": False}, {"concept_boundaries": "xyz"},
               {"concept_greedy_prob": 1.5}, {"concept_layers": 0}):
        try:
            _model(temporal_mode="concept", **kw)
        except ValueError:
            continue
        raise AssertionError(f"deveria falhar: {kw}")
    try:
        LandmarkEncoder(d_model=8, nhead=2, num_layers=1, temporal_mode="xyz")
    except ValueError:
        return
    raise AssertionError("temporal_mode inválido deveria falhar")


def test_encoder_forward_without_seg_ids_fails_in_concept():
    enc = LandmarkEncoder(d_model=16, nhead=2, num_layers=1, temporal_mode="concept")
    feats, mask, *_ = _batch()
    try:
        enc(feats, mask)
    except ValueError:
        return
    raise AssertionError("modo concept sem seg_ids deveria falhar")


def test_align_gives_one_position_per_gloss():
    for ds, K in ((0, 1), (1, 2)):
        model = _model(temporal_mode="concept", ds_steps=ds, concept_queries=K)
        model.train()
        feats, mask, _, gloss, glen = _batch()
        with torch.no_grad():
            enc_out, attn, cmask, (ctc_mem, ctc_mask) = model._encode_full(
                feats, mask, gloss, glen)
        assert (~cmask).sum(dim=1).tolist() == (glen * K).tolist(), (ds, K)
        assert attn.shape == cmask.shape == enc_out.last_hidden_state.shape[:2]
        # o CTC fica no andar de frames, com os comprimentos do ds-steps
        assert ctc_mem.shape[1] == ctc_mask.shape[1] > cmask.shape[1]


def test_greedy_prob_one_ignores_glosses():
    model = _model(temporal_mode="concept", concept_greedy_prob=1.0)
    model.train()
    feats, mask, _, gloss, glen = _batch()
    with torch.no_grad():
        x, fmask = model.encoder.encode_frames(feats, mask)
        expected = spikes_to_seg_ids(
            greedy_spikes(model.ctc_head(model.encoder.adapter(x)), fmask), fmask)
        got = model._concept_segments(model.encoder.adapter(x), fmask, gloss, glen)
    assert torch.equal(got, expected)


def test_forward_backward_all_params_get_grad():
    model = _model(temporal_mode="concept", ds_steps=1, concept_queries=2)
    model.train()
    feats, mask, labels, gloss, glen = _batch()
    loss, logits = model(feats, mask, labels, gloss_targets=gloss, gloss_lengths=glen,
                         ctc_weight=0.5)
    loss.backward()
    assert torch.isfinite(loss)
    enc = model.encoder
    for name, mod in (("concept_pool", enc.concept_pool),
                      ("concept_transformer", enc.concept_transformer),
                      ("frames", enc.transformer), ("ctc_head", model.ctc_head)):
        grads = [p.grad for p in mod.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert any(g.abs().sum() > 0 for g in grads), name


def test_eval_paths_use_greedy():
    model = _model(temporal_mode="concept", ds_steps=1)
    model.eval()
    feats, mask, labels, *_ = _batch()
    loss, _ = model(feats, mask, labels)                 # val_loss: sem glosas
    assert torch.isfinite(loss)
    assert model.generate(feats, mask, max_new_tokens=4, num_beams=2).shape[0] == 2
    preds = model.predict_glosses(feats, mask)
    n_frames = (~model.encoder.encode_frames(feats, mask)[1]).sum(dim=1).tolist()
    assert all(len(p) <= n for p, n in zip(preds, n_frames))


# --------------------------------------------------------------------------- #
# Checkpoints e CLI
# --------------------------------------------------------------------------- #
def _roundtrip(model, meta_model):
    from skeltrans.training.checkpoint import load_model, save_checkpoint
    tmp = tempfile.mkdtemp()
    t5_path = os.path.join(tmp, "t5")
    model.t5.save_pretrained(t5_path)
    meta = {"model": {"t5_name": t5_path, "d_model": 16, "nhead": 2, "num_layers": 1,
                      "use_ctc": True, "gloss_vocab_size": V, **meta_model}}
    ck = os.path.join(tmp, "m.pt")
    save_checkpoint(ck, model, 1, meta)
    return load_model(ck, torch.device("cpu"))[0]


def test_concept_checkpoint_roundtrip_detected_from_weights():
    model = _model(temporal_mode="concept", concept_layers=3, concept_queries=2)
    with torch.no_grad():
        model.encoder.concept_pool.query_embed.add_(0.1)
    # o config diz 'stride', mas os pesos são de 'concept': os pesos mandam
    loaded = _roundtrip(model, {"temporal_mode": "stride"})
    enc = loaded.encoder
    assert enc.temporal_mode == "concept"
    assert len(enc.concept_transformer.layers) == 3 and enc.concept_pool.queries == 2
    model.eval()
    feats, mask, *_ = _batch()
    with torch.no_grad():
        a, b = model._encode(feats, mask), loaded._encode(feats, mask)
    assert torch.allclose(a[0].last_hidden_state, b[0].last_hidden_state)
    assert torch.equal(a[2], b[2])


def test_stride_checkpoint_stays_stride():
    loaded = _roundtrip(_model(), {})
    assert loaded.encoder.temporal_mode == "stride"
    assert not hasattr(loaded.encoder, "concept_pool")


def test_cli_validation():
    from skeltrans.training.cli import build_parser, validate_concept_args
    ap = build_parser()

    def check(argv):
        return validate_concept_args(ap.parse_args(["--train-manifest", "x.json"] + argv))
    assert check([]) is None                                           # stride: nada a validar
    assert check(["--temporal-mode", "concept"]) is not None           # sem --use-ctc
    assert check(["--temporal-mode", "concept", "--use-ctc", "--ctc-weight", "0"]) is not None
    assert check(["--temporal-mode", "concept", "--use-ctc", "--concept-queries", "0"])
    assert check(["--temporal-mode", "concept", "--use-ctc"]) is None
    assert isinstance(ap.parse_args([]), argparse.Namespace)


def test_smoke_test_concept():
    from skeltrans.training.diagnostics import smoke_test
    smoke_test(device="cpu", ds_steps=1, temporal_mode="concept", concept_queries=2)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} testes passaram")
