"""Pré-treino do encoder de landmarks por masked pose modeling (MPM).

Entrada: um ou mais manifestos (só os .npy de keypoints importam; texto e
glosas são ignorados). Saída, em ``--out-dir``:

- ``encoder.pt``: pesos do encoder na época de menor erro de reconstrução
  na validação (ou na última, sem validação). É o arquivo do
  ``slt_model.py --init-encoder``;
- ``encoder_last.pt``: o da última época;
- ``pretrain_history.jsonl``: perda e erro por época;
- ``pretrain_config.json``: a configuração usada.

O modelo e o mascaramento estão em ``models/masked_pose.py``; o formato do
``encoder.pt``, em ``encoder_init.py``. Guia: docs/encoders/pretrain.md.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader

from skeltrans.training.config import ModelConfig
from skeltrans.training.data.augment import AugmentConfig
from skeltrans.training.data.dataset import KeypointDataset
from skeltrans.training.data.features import (compute_input_stats, feature_dim,
                                              format_groups, parse_groups)
from skeltrans.training.device import resolve_device
from skeltrans.training.encoder_init import encoder_spec, save_encoder
from skeltrans.training.low_vram import autocast_ctx, bf16_supported, make_optimizer
from skeltrans.training.models.landmark_encoder import ENCODER_TYPES
from skeltrans.training.models.masked_pose import (MASK_MODES, MaskedPoseModel,
                                                   make_masks)

_MC = ModelConfig()
# Semente fixa das máscaras da validação: as mesmas em todas as épocas, para
# que o erro de uma época seja comparável com o da outra.
VAL_MASK_SEED = 1234


@dataclass
class PretrainConfig:
    """Tudo o que define um pré-treino (gravado em pretrain_config.json)."""
    manifests: List[str] = field(default_factory=list)
    val_manifest: Optional[str] = None
    features_dir: Optional[str] = None
    # --- encoder: precisa ser igual ao do treino de SLT (ver encoder_init.SPEC_FIELDS) --- #
    extra_features: str = ""
    input_norm: bool = False
    encoder_type: str = "flat"
    d_model: int = _MC.d_model
    nhead: int = _MC.nhead
    num_layers: int = _MC.num_layers
    ds_steps: int = _MC.ds_steps
    dropout: float = _MC.dropout
    # --- mascaramento --- #
    mask_mode: str = "mixed"
    mask_ratio: float = 0.4
    mask_span: int = 8
    # --- otimização --- #
    epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    patience: int = 10
    seed: Optional[int] = None
    # --- augmentation (só no treino do pré-treino) --- #
    aug_speed: float = 0.0
    aug_spatial: float = 0.0
    aug_noise: float = 0.0
    aug_flip: bool = False
    # --- execução --- #
    num_workers: int = 4
    log_every: int = 50
    out_dir: str = "checkpoints_pretrain"
    device: Optional[str] = None
    low_vram: bool = False
    grad_checkpoint: bool = False


def collate_keypoints(batch):
    """Lote de itens ``(features,)`` -> ``(padded, pad_mask)`` (True = padding)."""
    feats = [item[0] for item in batch]
    T = max(f.shape[0] for f in feats)
    padded = torch.zeros(len(feats), T, feats[0].shape[1], dtype=torch.float32)
    pad_mask = torch.ones(len(feats), T, dtype=torch.bool)
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
        pad_mask[i, : f.shape[0]] = False
    return padded, pad_mask


def _datasets(manifests, cfg, groups, augment=None):
    parts = [KeypointDataset(m, features_dir=cfg.features_dir, extra_features=groups,
                             augment=augment) for m in manifests]
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


@torch.no_grad()
def evaluate_reconstruction(model, dl, device, cfg, use_bf16=False):
    """Erro de reconstrução na validação, com máscaras fixas.

    Devolve ``(mse, mse_zero)``: o erro do modelo e o de prever sempre 0 (no
    espaço normalizado, ~1 = a variância do alvo). ``1 - mse / mse_zero`` é a
    fração do sinal escondido que o modelo conseguiu recuperar.
    """
    model.eval()
    gen = torch.Generator().manual_seed(VAL_MASK_SEED)
    sq_err = sq_zero = count = 0.0
    for feats, pad_mask in dl:
        pm, fm = make_masks(pad_mask, cfg.mask_ratio, cfg.mask_span, cfg.mask_mode, gen)
        with autocast_ctx(use_bf16):
            _, st = model(feats.to(device), pad_mask.to(device), pm.to(device), fm.to(device))
        sq_err += st["sq_err"]
        sq_zero += st["sq_zero"]
        count += st["count"]
    n = max(count, 1.0)
    return sq_err / n, sq_zero / n


def build_and_pretrain(cfg):
    """Monta dados e modelo a partir de ``cfg`` e pré-treina. Devolve o resumo."""
    from transformers import get_linear_schedule_with_warmup

    from skeltrans.training.trainer import seed_everything

    if not cfg.manifests:
        raise ValueError("informe ao menos um manifesto em --manifests.")
    device = resolve_device(cfg.device)
    print(f"Dispositivo: {device}")
    if cfg.seed is not None:
        seed_everything(cfg.seed)
        print(f"[seed] semente {cfg.seed}")
    if any("test" in os.path.basename(m).lower() for m in cfg.manifests):
        print("[pretrain] AVISO: um manifesto parece ser de TESTE. O pre-treino nao usa "
              "rotulos, mas declare isso na tese ou deixe o teste de fora.")

    groups = parse_groups(cfg.extra_features)
    aug = AugmentConfig(speed=cfg.aug_speed, spatial=cfg.aug_spatial,
                        noise=cfg.aug_noise, flip=cfg.aug_flip)
    train_ds = _datasets(cfg.manifests, cfg, groups, augment=aug)
    print(f"[pretrain] {len(train_ds)} frases de {len(cfg.manifests)} manifesto(s) | "
          f"entrada = {feature_dim(groups)} colunas | mascara {cfg.mask_mode} "
          f"{cfg.mask_ratio:g} em trechos de {cfg.mask_span} frames")
    if aug.enabled:
        print(f"[augment] velocidade ±{aug.speed:g} | espacial σ={aug.spatial:g} | "
              f"ruido σ={aug.noise:g} | espelhamento={'sim' if aug.flip else 'nao'}")

    model = MaskedPoseModel(feature_dim(groups), d_model=cfg.d_model, nhead=cfg.nhead,
                            num_layers=cfg.num_layers, dropout=cfg.dropout,
                            ds_steps=cfg.ds_steps, input_norm=cfg.input_norm,
                            encoder_type=cfg.encoder_type)
    if cfg.input_norm:
        # estatísticas sobre os manifestos de pré-treino, sem augmentation;
        # viajam no encoder.pt e o treino de SLT as mantém
        print("[input-norm] calculando media/desvio por coluna ...")
        mean, std = compute_input_stats(_datasets(cfg.manifests, cfg, groups),
                                        num_workers=cfg.num_workers)
        model.encoder.set_input_stats(mean, std)
    model.to(device)
    if cfg.grad_checkpoint:
        model.encoder.grad_checkpoint = True
    use_bf16 = cfg.low_vram and bf16_supported(device)

    loader_kw = {"generator": torch.Generator().manual_seed(cfg.seed)} if cfg.seed is not None else {}
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=collate_keypoints, num_workers=cfg.num_workers,
                          **loader_kw)
    val_dl = None
    if cfg.val_manifest:
        val_dl = DataLoader(_datasets([cfg.val_manifest], cfg, groups),
                            batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate_keypoints, num_workers=cfg.num_workers)

    optim = make_optimizer(model.parameters(), cfg.lr, low_vram=cfg.low_vram)
    total_steps = len(train_dl) * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, cfg.warmup_steps, total_steps)

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "pretrain_config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "command": [sys.executable] + sys.argv},
                  f, ensure_ascii=False, indent=2)
    spec = encoder_spec(groups, cfg.input_norm, cfg.encoder_type, cfg.d_model,
                        cfg.nhead, cfg.num_layers, cfg.ds_steps)
    history_path = os.path.join(cfg.out_dir, "pretrain_history.jsonl")

    # máscaras do treino: gerador próprio, semeado pelo torch (fixo com --seed)
    mask_gen = torch.Generator().manual_seed(int(torch.randint(0, 2**62, (1,)).item()))
    best, best_epoch, epoch = float("inf"), 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for step, (feats, pad_mask) in enumerate(train_dl, 1):
            pm, fm = make_masks(pad_mask, cfg.mask_ratio, cfg.mask_span, cfg.mask_mode, mask_gen)
            with autocast_ctx(use_bf16):
                loss, _ = model(feats.to(device), pad_mask.to(device),
                                pm.to(device), fm.to(device))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            running += loss.item()
            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} "
                      f"mse {running / step:.4f} lr {sched.get_last_lr()[0]:.2e}")
        train_mse = running / max(1, len(train_dl))

        record = {"epoch": epoch, "train_mse": train_mse}
        msg = f"[epoch {epoch}] train_mse={train_mse:.4f}"
        if val_dl is not None:
            val_mse, val_zero = evaluate_reconstruction(model, val_dl, device, cfg, use_bf16)
            explained = 1.0 - val_mse / val_zero if val_zero > 0 else 0.0
            record.update(val_mse=val_mse, val_mse_zero=val_zero, val_explained=explained)
            msg += (f" val_mse={val_mse:.4f} (prever 0: {val_zero:.4f}; "
                    f"recuperado {100 * explained:.1f}%)")
            score = val_mse
        else:
            score = train_mse
        print(msg)
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        info = {"epochs_run": epoch, "mask_mode": cfg.mask_mode,
                "mask_ratio": cfg.mask_ratio, "mask_span": cfg.mask_span,
                "manifests": list(cfg.manifests), "train_sentences": len(train_ds),
                "train_mse": train_mse, "best_val_mse": record.get("val_mse"),
                "dropout": cfg.dropout, "seed": cfg.seed}
        save_encoder(os.path.join(cfg.out_dir, "encoder_last.pt"), model.encoder, spec, info)
        if score < best:
            best, best_epoch = score, epoch
            save_encoder(os.path.join(cfg.out_dir, "encoder.pt"), model.encoder, spec, info)
            print(f"  -> encoder.pt (melhor ate agora, epoca {epoch})")
        elif cfg.patience and epoch - best_epoch >= cfg.patience:
            print(f"[early stopping] {epoch - best_epoch} epocas sem melhora "
                  f"(melhor na epoca {best_epoch}).")
            break
    print(f"Pre-treino concluido. Use: slt_model.py ... --init-encoder "
          f"{os.path.join(cfg.out_dir, 'encoder.pt')}")
    return {"best": best, "best_epoch": best_epoch, "epochs_run": epoch}


# ------------------------------------------------------------------------- #
# Linha de comando
# ------------------------------------------------------------------------- #

def build_parser():
    d = PretrainConfig()
    ap = argparse.ArgumentParser(
        description="Pre-treino do encoder de landmarks por masked pose modeling. "
                    "Gera <out-dir>/encoder.pt para o slt_model.py --init-encoder.")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Roda um pre-treino minusculo com dados aleatorios na CPU e confere "
                         "o encoder.pt gerado. Nao precisa de dados.")
    ap.add_argument("--manifests", nargs="+", default=[],
                    help="Um ou mais manifestos JSON (mesmo formato do treino de SLT). So os "
                         ".npy importam: texto e glosas sao ignorados, entao servem manifestos "
                         "sem traducao.")
    ap.add_argument("--val-manifest", default=None,
                    help="Manifesto para medir o erro de reconstrucao (mascaras fixas). Escolhe "
                         "o encoder.pt e o early stopping. Sem ele, vale o erro de treino.")
    ap.add_argument("--features-dir", default=None,
                    help="Pasta dos .npy (mesmo sentido do slt_model.py). Vale para todos os "
                         "manifestos.")
    g = ap.add_argument_group("encoder (precisa ser igual ao do treino de SLT)")
    g.add_argument("--extra-features", default="", metavar="GRUPOS",
                   help="Grupos de features extras (A-F), como no slt_model.py.")
    g.add_argument("--input-norm", action="store_true",
                   help="z-score por coluna; as estatisticas vao no encoder.pt e o treino de SLT "
                        "as mantem.")
    g.add_argument("--encoder-type", default=d.encoder_type, choices=list(ENCODER_TYPES))
    g.add_argument("--d-model", type=int, default=d.d_model)
    g.add_argument("--nhead", type=int, default=d.nhead)
    g.add_argument("--num-layers", type=int, default=d.num_layers,
                   help="Camadas do Transformer. Com --temporal-mode concept no SLT, e o andar "
                        "de frames (ex.: 4).")
    g.add_argument("--ds-steps", type=int, default=d.ds_steps)
    g.add_argument("--dropout", type=float, default=d.dropout)
    m = ap.add_argument_group("mascaramento")
    m.add_argument("--mask-mode", default=d.mask_mode, choices=list(MASK_MODES),
                   help="'frames': some o frame inteiro; 'hands': some uma das maos; 'mixed' "
                        "(padrao): sorteia o tipo de cada trecho. As features extras somem em "
                        "todo frame mascarado.")
    m.add_argument("--mask-ratio", type=float, default=d.mask_ratio,
                   help="Fracao dos frames de cada frase coberta por trechos mascarados. "
                        "Padrao: 0.4.")
    m.add_argument("--mask-span", type=int, default=d.mask_span,
                   help="Tamanho de cada trecho, em frames. Padrao: 8 (cerca de meio sinal a "
                        "25 fps; precisa ser >= 2^ds-steps para esconder posicoes inteiras).")
    o = ap.add_argument_group("otimizacao")
    o.add_argument("--epochs", type=int, default=d.epochs)
    o.add_argument("--batch-size", type=int, default=d.batch_size)
    o.add_argument("--lr", type=float, default=d.lr)
    o.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    o.add_argument("--grad-clip", type=float, default=d.grad_clip)
    o.add_argument("--patience", type=int, default=d.patience,
                   help="Epocas sem melhora antes de parar. 0 desliga. Padrao: 10.")
    o.add_argument("--seed", type=int, default=None)
    a = ap.add_argument_group("augmentation (como no slt_model.py)")
    a.add_argument("--aug-speed", type=float, default=0.0, metavar="F")
    a.add_argument("--aug-spatial", type=float, default=0.0, metavar="S")
    a.add_argument("--aug-noise", type=float, default=0.0, metavar="S")
    a.add_argument("--aug-flip", action="store_true")
    e = ap.add_argument_group("execucao")
    e.add_argument("--num-workers", type=int, default=d.num_workers)
    e.add_argument("--log-every", type=int, default=d.log_every)
    e.add_argument("--out-dir", default=d.out_dir)
    e.add_argument("--device", default=None)
    e.add_argument("--low-vram", action="store_true",
                   help="autocast bf16 (GPU Ampere+) e AdamW 8-bit, se disponivel.")
    e.add_argument("--grad-checkpoint", action="store_true")
    return ap


def args_to_config(args):
    return PretrainConfig(
        manifests=list(args.manifests), val_manifest=args.val_manifest,
        features_dir=args.features_dir, extra_features=format_groups(args.extra_features),
        input_norm=args.input_norm, encoder_type=args.encoder_type, d_model=args.d_model,
        nhead=args.nhead, num_layers=args.num_layers, ds_steps=args.ds_steps,
        dropout=args.dropout, mask_mode=args.mask_mode, mask_ratio=args.mask_ratio,
        mask_span=args.mask_span, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
        patience=args.patience, seed=args.seed, aug_speed=args.aug_speed,
        aug_spatial=args.aug_spatial, aug_noise=args.aug_noise, aug_flip=args.aug_flip,
        num_workers=args.num_workers, log_every=args.log_every, out_dir=args.out_dir,
        device=args.device, low_vram=args.low_vram, grad_checkpoint=args.grad_checkpoint)


def validate_args(args):
    """Mensagem de erro para argumentos inválidos (ou None)."""
    if not args.manifests:
        return "Faltou --manifests (ou use --smoke-test)."
    if not 0.0 < args.mask_ratio < 1.0:
        return "--mask-ratio deve estar em (0, 1)."
    if args.mask_span < 1:
        return "--mask-span deve ser >= 1."
    if args.d_model % args.nhead:
        return "--d-model precisa ser divisivel por --nhead."
    if not 0.0 <= args.aug_speed < 1.0 or args.aug_spatial < 0 or args.aug_noise < 0:
        return "--aug-speed deve estar em [0, 1); --aug-spatial e --aug-noise >= 0."
    try:
        parse_groups(args.extra_features)
    except ValueError as exc:
        return f"--extra-features: {exc}"
    return None


def smoke_test():
    """Pré-treino minúsculo com frases aleatórias (CPU), para conferir a instalação.

    Tudo acontece numa pasta temporária, apagada no fim.
    """
    import tempfile

    import numpy as np

    from skeltrans.training.encoder_init import read_encoder

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "out")
        items = []
        for i in range(6):
            path = os.path.join(tmp, f"f{i}.npy")
            np.save(path, rng.normal(size=(int(rng.integers(20, 40)), 115, 9)).astype(np.float32))
            items.append({"features": path})
        manifest = os.path.join(tmp, "m.json")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(items, f)
        cfg = PretrainConfig(manifests=[manifest], val_manifest=manifest, extra_features="A,F",
                             input_norm=True, d_model=32, nhead=4, num_layers=2, ds_steps=2,
                             epochs=2, batch_size=3, warmup_steps=1, num_workers=0,
                             log_every=1000, out_dir=out_dir, device="cpu", seed=0)
        summary = build_and_pretrain(cfg)
        ckpt = read_encoder(os.path.join(out_dir, "encoder.pt"))
    assert ckpt["spec"]["input_dim"] == feature_dim(("A", "F"))
    assert not any(k.startswith("adapter.") for k in ckpt["encoder"])
    print(f"[smoke] OK: encoder.pt valido (melhor epoca {summary['best_epoch']}).")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        smoke_test()
        return
    error = validate_args(args)
    if error:
        sys.exit(error)
    build_and_pretrain(args_to_config(args))


if __name__ == "__main__":
    main()
