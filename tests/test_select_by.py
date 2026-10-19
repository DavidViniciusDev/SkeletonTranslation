#!/usr/bin/env python3
"""Testes do critério de seleção do best.pt e do early stopping (--select-by).

O laço real do Trainer.fit roda com um modelo de 1 parâmetro e com
`evaluate`/`generate_metrics` trocados por versões que devolvem val_loss e
métricas roteirizadas por época: não exige dados, T5 nem GPU.

Uso:
    python3 tests/test_select_by.py
    (também compatível com: pytest tests/test_select_by.py)
"""
import glob
import os
import random
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import skeltrans.training.trainer as tr  # noqa: E402
from skeltrans.training.config import TrainConfig  # noqa: E402
from skeltrans.training.trainer import BestTracker, Trainer  # noqa: E402


# --------------------------------------------------------------------------- #
# BestTracker
# --------------------------------------------------------------------------- #
def test_tracker_lower_is_better():
    t = BestTracker(higher_is_better=False)
    assert t.update(5.0, 1) and t.update(4.0, 2)
    assert not t.update(4.0, 3)              # empate não é melhora
    assert t.best == 4.0 and t.best_epoch == 2
    assert t.epochs_since_best(5) == 3


def test_tracker_higher_is_better_with_min_delta():
    t = BestTracker(higher_is_better=True, min_delta=0.5)
    assert t.update(10.0, 2)
    assert not t.update(10.4, 4)             # +0.4 < min_delta
    assert t.update(10.6, 6)
    assert t.best_epoch == 6


def test_tracker_none_and_nan_never_improve():
    t = BestTracker(higher_is_better=True)
    assert not t.update(None, 1)
    assert not t.update(float("nan"), 2)
    assert t.best_epoch == 0 and t.epochs_since_best(2) == 2


def _original_loop(losses, patience, min_delta):
    """Cópia da lógica ANTERIOR do fit (val_loss), para comparação."""
    best_val, sem, best_epochs = float("inf"), 0, []
    for epoch, v in enumerate(losses, 1):
        if v < best_val - min_delta:
            best_val, sem = v, 0
            best_epochs.append(epoch)
        else:
            sem += 1
            if patience > 0 and sem >= patience:
                return epoch, best_epochs
    return len(losses), best_epochs


def _tracker_loop(losses, patience, min_delta):
    """Mesma decisão, agora com BestTracker (como no fit novo)."""
    t, best_epochs = BestTracker(higher_is_better=False, min_delta=min_delta), []
    for epoch, v in enumerate(losses, 1):
        if t.update(v, epoch):
            best_epochs.append(epoch)
        elif patience > 0 and t.epochs_since_best(epoch) >= patience:
            return epoch, best_epochs
    return len(losses), best_epochs


def test_val_loss_logic_identical_to_original():
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(1, 40)
        losses = [rng.choice([rng.uniform(0, 5), float("nan")]) if rng.random() < 0.05
                  else rng.uniform(0, 5) for _ in range(n)]
        patience = rng.randint(0, 6)
        min_delta = rng.choice([0.0, 0.01, 0.5])
        assert _original_loop(losses, patience, min_delta) == \
            _tracker_loop(losses, patience, min_delta), (losses, patience, min_delta)


# --------------------------------------------------------------------------- #
# Trainer.fit com valores roteirizados
# --------------------------------------------------------------------------- #
class _Dummy(nn.Module):
    """Modelo mínimo com a assinatura do SLTModel.forward."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, feats, pad_mask, labels, gloss_targets=None,
                gloss_lengths=None, ctc_weight=0.0):
        return (self.w * feats.mean()) ** 2, None


class _Tok:
    def save_pretrained(self, out_dir):
        pass


def _loader():
    ds = TensorDataset(torch.ones(4, 5, 3), torch.zeros(4, 5, dtype=torch.bool),
                       torch.zeros(4, 3, dtype=torch.long))
    return DataLoader(ds, batch_size=2)


def _run(losses, metric_by_epoch, **cfg):
    """Roda o fit com val_loss[epoca-1] e métricas por época; devolve
    (épocas treinadas, época do best.pt, época do best_loss.pt ou None)."""
    out = tempfile.mkdtemp()
    trainer = Trainer(_Dummy(), _Tok(), TrainConfig(
        epochs=len(losses), lr=1e-3, warmup_steps=0, log_every=10_000, out_dir=out, **cfg),
        torch.device("cpu"))

    def fake_evaluate(model, dl, device, use_bf16=False):
        return losses[trainer._epoch - 1]

    def fake_generate(model, tok, dl, device, num_beams=1, max_new_tokens=64, use_bf16=False):
        value = metric_by_epoch[trainer._epoch]
        return {"BLEU-1": value, "BLEU-4": value, "METEOR": value}, ["h"], ["r"]

    saved = (tr.evaluate, tr.generate_metrics)
    tr.evaluate, tr.generate_metrics = fake_evaluate, fake_generate
    try:
        dl = _loader()
        trainer.fit(dl, val_dl=dl, eval_dl=dl)
    finally:
        tr.evaluate, tr.generate_metrics = saved

    def epoch_of(name):
        path = os.path.join(out, name)
        return torch.load(path, weights_only=False)["epoch"] if os.path.exists(path) else None

    trained = len(glob.glob(os.path.join(out, "epoch*.pt")))
    return trained, epoch_of("best.pt"), epoch_of("best_loss.pt")


LOSSES = [5.0, 4.0, 3.0, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3]   # mínimo na época 3
BLEU = {e: v for e, v in enumerate([1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6], 1)}  # pico na época 6


def test_fit_default_is_val_loss():
    trained, best, best_loss = _run(LOSSES, BLEU, patience=3, eval_every=1)
    assert (trained, best, best_loss) == (6, 3, None)   # para 3 épocas após a época 3


def test_fit_select_by_bleu4():
    trained, best, best_loss = _run(LOSSES, BLEU, patience=3, eval_every=1, select_by="bleu4")
    assert best == 6          # best.pt = maior BLEU-4
    assert best_loss == 3     # best_loss.pt = menor val_loss
    assert trained == 9       # 3 épocas sem melhora de BLEU após a época 6


def test_fit_select_by_bleu4_counts_epochs_with_eval_every_2():
    bleu = {2: 1.0, 4: 3.0, 6: 2.0, 8: 2.5, 10: 2.0, 12: 2.0}
    trained, best, _ = _run(LOSSES, bleu, patience=4, eval_every=2, select_by="bleu4")
    assert best == 4
    assert trained == 8       # na época 8 a melhor avaliação ficou 4 épocas para trás


def test_fit_select_by_meteor_and_min_delta():
    meteor = {e: v for e, v in enumerate([10, 11, 11.05, 11.08, 11.09, 11.1], 1)}
    trained, best, _ = _run(LOSSES[:6], meteor, patience=3, eval_every=1,
                            select_by="meteor", min_delta=0.2)
    assert best == 2          # ganhos depois da época 2 ficam abaixo de min_delta
    assert trained == 5


def test_fit_without_patience_trains_all_epochs():
    trained, best, _ = _run(LOSSES, BLEU, patience=0, eval_every=1, select_by="bleu4")
    assert trained == len(LOSSES) and best == 6


def test_fit_metric_requires_eval_every():
    try:
        _run(LOSSES[:2], BLEU, select_by="bleu4", eval_every=0)
    except ValueError as exc:
        assert "eval_every" in str(exc)
    else:
        raise AssertionError("esperava ValueError sem eval_every")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli_exit(argv):
    from skeltrans.training.cli import main
    try:
        main(argv)
    except SystemExit as exc:
        return str(exc.code)
    raise AssertionError("esperava SystemExit")


def test_cli_rejects_metric_without_eval_every():
    msg = _cli_exit(["--train-manifest", "x.json", "--val-manifest", "y.json",
                     "--select-by", "bleu4"])
    assert "--eval-every" in msg


def test_cli_rejects_unknown_criterion():
    assert _cli_exit(["--train-manifest", "x.json", "--select-by", "wer"]) == "2"


def test_cli_maps_select_by_to_config():
    from skeltrans.training.cli import args_to_configs, build_parser
    args = build_parser().parse_args(["--train-manifest", "x.json", "--select-by", "meteor"])
    assert args_to_configs(args)[2].select_by == "meteor"
    args = build_parser().parse_args(["--train-manifest", "x.json"])
    assert args_to_configs(args)[2].select_by == "val_loss"   # padrão inalterado


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} testes passaram")
