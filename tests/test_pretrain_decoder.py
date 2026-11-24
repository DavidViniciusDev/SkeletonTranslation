#!/usr/bin/env python3
"""Testes do pré-treino do decoder em Gloss2Text (pretrain_decoder.py).

Tudo offline: manifestos sintéticos, tokenizer de palavras montado localmente
e um T5 minúsculo criado do zero.

Uso:
    python3 tests/test_pretrain_decoder.py
    (também compatível com: pytest tests/test_pretrain_decoder.py)
"""
import argparse
import json
import os
import random
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skeltrans.training.gloss2text import (Gloss2TextCollate, Gloss2TextConfig,  # noqa: E402
                                           GlossTextDataset, build_and_train, build_parser,
                                           corrupt_glosses, normalize_glosses, parse_noise_mix,
                                           seq2seq_loss, toy_tokenizer, validate_args,
                                           write_toy_corpus)

VOCAB = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# glosas
# --------------------------------------------------------------------------- #
def test_normalize_glosses():
    g = ["__ON__", "MORGEN", "REGEN", "__OFF__"]
    assert normalize_glosses(g) == g
    assert normalize_glosses(g, drop_markers=True) == ["MORGEN", "REGEN"]
    assert normalize_glosses(g, lowercase=True, drop_markers=True) == ["morgen", "regen"]


def test_parse_noise_mix():
    assert parse_noise_mix("6,3,1") == (0.6, 0.3, 0.1)
    for bad in ("1,2", "a,b,c", "0,0,0", "-1,1,1"):
        try:
            parse_noise_mix(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} deveria falhar")


def test_corrupt_zero_is_identity():
    g = ["A", "B", "C"]
    assert corrupt_glosses(g, VOCAB, 0.0, (1, 0, 0), random.Random(0)) == g


def test_corrupt_operations():
    g = ["A", "B", "C", "D"] * 5
    sub = corrupt_glosses(g, ["X"], 1.0, (1, 0, 0), random.Random(0))
    assert sub == ["X"] * len(g)                            # tudo substituído
    assert corrupt_glosses(g, ["X"], 1.0, (0, 1, 0), random.Random(0)) == []   # tudo removido
    ins = corrupt_glosses(g, ["X"], 1.0, (0, 0, 1), random.Random(0))
    assert ins == [t for x in g for t in (x, "X")]           # glosa + inserção


def test_corrupt_is_reproducible_and_partial():
    g = ["A", "B", "C", "D"] * 50
    a = corrupt_glosses(g, VOCAB, 0.3, parse_noise_mix("6,3,1"), random.Random(7))
    b = corrupt_glosses(g, VOCAB, 0.3, parse_noise_mix("6,3,1"), random.Random(7))
    assert a == b and a != g
    assert all(t in VOCAB for t in a)


# --------------------------------------------------------------------------- #
# dados
# --------------------------------------------------------------------------- #
def test_dataset_reads_and_skips():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "m.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"items": [
                {"tokens": ["B", "A"], "text": "frase um"},
                {"gloss_tokens": "C D", "pt_br": "frase dois"},      # chaves alternativas
                {"tokens": [], "text": "sem glosas"},
                {"tokens": ["A"]},                                  # sem texto
            ]}, f)
        ds = GlossTextDataset(p)
    assert len(ds) == 2 and ds.skipped == 2
    assert ds[1] == (["C", "D"], "frase dois")
    assert ds.vocab() == ["A", "B", "C", "D"]


def test_collate_labels_and_noise():
    with tempfile.TemporaryDirectory() as tmp:
        tok = toy_tokenizer(VOCAB + ["um", "dois", "tres"], tmp)
    batch = [(["A", "B"], "um dois"), (["C"], "tres")]
    ids, attn, labels, srcs, texts = Gloss2TextCollate(tok, 16, 16)(batch)
    assert srcs == ["A B", "C"] and texts == ["um dois", "tres"]
    assert ids.shape == attn.shape and (labels[1, 2:] == -100).all()
    assert labels[0, 2] == tok.eos_token_id                  # </s> no fim do alvo
    noisy = Gloss2TextCollate(tok, 16, 16, vocab=["D"], noise=1.0, mix=(1, 0, 0),
                              rng=random.Random(0))(batch)
    assert noisy[3] == ["D D", "D"]                          # só a fonte muda
    assert torch.equal(noisy[2], labels)


def test_seq2seq_loss_smoothing():
    from transformers import T5Config, T5ForConditionalGeneration
    torch.manual_seed(0)
    m = T5ForConditionalGeneration(T5Config(vocab_size=20, d_model=16, d_ff=32, num_layers=1,
                                            num_decoder_layers=1, num_heads=2, d_kv=8,
                                            decoder_start_token_id=0, pad_token_id=0,
                                            eos_token_id=1)).eval()
    ids = torch.randint(3, 20, (2, 5))
    labels = torch.randint(3, 20, (2, 4))
    labels[1, 2:] = -100
    attn = torch.ones_like(ids)
    with torch.no_grad():
        base = seq2seq_loss(m, ids, attn, labels)
        assert torch.equal(base, m(input_ids=ids, attention_mask=attn, labels=labels).loss)
        logits = m(input_ids=ids, attention_mask=attn, labels=labels).logits
        ref = F.cross_entropy(logits.reshape(-1, 20), labels.reshape(-1), ignore_index=-100,
                              label_smoothing=0.1)
        assert torch.allclose(seq2seq_loss(m, ids, attn, labels, 0.1), ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# treino
# --------------------------------------------------------------------------- #
def test_learns_and_writes_outputs():
    """O treino aprende de verdade: o T5 minúsculo decora o corpus sintético."""
    with tempfile.TemporaryDirectory() as tmp:
        train, val, t5_dir = write_toy_corpus(tmp, n_train=16, n_val=4)
        out = os.path.join(tmp, "out")
        summary = build_and_train(Gloss2TextConfig(
            train_manifest=train, val_manifest=val, test_manifest=val, t5_name=t5_dir,
            epochs=40, batch_size=4, lr=3e-3, warmup_steps=5, label_smoothing=0.0,
            patience=0, eval_beams=1, log_every=10_000, out_dir=out, device="cpu", seed=0))
        assert summary["teto"]["BLEU-4"] > 80, summary["teto"]
        for name in ("gloss2text_config.json", "gloss2text_history.jsonl", "teto.json",
                     os.path.join("best", "config.json"), os.path.join("best", "tokenizer.json")):
            assert os.path.exists(os.path.join(out, name)), name
        teto = json.load(open(os.path.join(out, "teto.json"), encoding="utf-8"))
        assert teto["num_examples"] == 4 and {"src", "ref", "hyp"} <= set(teto["predictions"][0])


def test_best_dir_works_as_slt_t5():
    """best/ é carregado pelo mesmo caminho do slt_model.py (--t5 <pasta>)."""
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from skeltrans.common.layout import INPUT_DIM
    from skeltrans.training.models import SLTModel
    with tempfile.TemporaryDirectory() as tmp:
        train, val, t5_dir = write_toy_corpus(tmp)
        out = os.path.join(tmp, "out")
        build_and_train(Gloss2TextConfig(train_manifest=train, val_manifest=val, t5_name=t5_dir,
                                         epochs=1, batch_size=4, warmup_steps=1, eval_beams=1,
                                         log_every=10_000, out_dir=out, device="cpu", seed=0))
        best = os.path.join(out, "best")
        tok = AutoTokenizer.from_pretrained(best)
        t5 = T5ForConditionalGeneration.from_pretrained(best)
        trained = T5ForConditionalGeneration.from_pretrained(best).state_dict()
        slt = SLTModel(t5, d_model=16, nhead=2, num_layers=1)
        # o decoder do SLT carrega exatamente os pesos do pré-treino
        k = "decoder.block.0.layer.0.SelfAttention.q.weight"
        assert torch.equal(slt.t5.state_dict()[k], trained[k])
        labels = tok(["morgen regnet es im norden"], return_tensors="pt").input_ids
        loss, _ = slt(torch.randn(1, 12, INPUT_DIM), torch.zeros(1, 12, dtype=torch.bool), labels)
        assert torch.isfinite(loss)


def test_validate_args():
    ap = build_parser()
    ok = ["--train-manifest", "t.json", "--val-manifest", "v.json"]
    assert validate_args(ap.parse_args(ok)) is None
    for extra, frag in ((["--gloss-noise", "1.0"], "--gloss-noise"),
                        (["--label-smoothing", "1"], "--label-smoothing"),
                        (["--noise-mix", "1,1"], "--noise-mix"),
                        (["--eval-beams", "0"], "--eval-beams")):
        msg = validate_args(ap.parse_args(ok + extra))
        assert msg and frag in msg, (extra, msg)
    assert "--train-manifest" in validate_args(ap.parse_args([]))
    assert "--val-manifest" in validate_args(ap.parse_args(["--train-manifest", "t.json"]))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} testes passaram")
