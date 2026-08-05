"""Loop de treino/avaliação (extraído da antiga função train())."""

import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from skeltrans.training.checkpoint import save_checkpoint
from skeltrans.training.data import LandmarkTextDataset, make_collate
from skeltrans.training.device import resolve_device
from skeltrans.training.low_vram import (autocast_ctx, bf16_supported,
                                         make_optimizer, offload_t5_encoder)
from skeltrans.training.models import SLTModel


@torch.no_grad()
def evaluate(model, dl, device, use_bf16=False):
    """Perda de validação — sempre só a de tradução (CE), sem o termo de CTC,
    para que val_loss seja comparável entre execuções com e sem CTC."""
    model.eval()
    total = 0.0
    for batch in dl:
        feats, pad_mask, labels = batch[0], batch[1], batch[2]
        feats, pad_mask, labels = feats.to(device), pad_mask.to(device), labels.to(device)
        with autocast_ctx(use_bf16):
            loss, _ = model(feats, pad_mask, labels)
        total += loss.item()
    return total / max(1, len(dl))


class Trainer:
    """Treina um SLTModel com AdamW + agenda linear, salvando checkpoints."""

    def __init__(self, model, tokenizer, train_cfg, device, meta=None):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = train_cfg
        self.device = device
        self.meta = meta or {}
        # bf16 só quando --low-vram foi pedido E a GPU suporta (Ampere+).
        low_vram = bool(getattr(train_cfg, "low_vram", False))
        self.use_bf16 = low_vram and bf16_supported(device)
        if self.use_bf16:
            print("[low-vram] autocast bf16 ativo no forward do treino")
        elif low_vram:
            print("[low-vram] dispositivo sem suporte a bf16; treino segue em fp32")

    def _run_batch(self, batch):
        """Executa um passo forward, aceitando lotes com ou sem alvos de CTC."""
        if len(batch) == 5:                       # (feats, pad_mask, labels, gloss, glen)
            feats, pad_mask, labels, gloss, glen = batch
            gloss = gloss.to(self.device)
            glen = glen.to(self.device)
        else:                                     # (feats, pad_mask, labels)
            feats, pad_mask, labels = batch
            gloss = glen = None
        feats = feats.to(self.device)
        pad_mask = pad_mask.to(self.device)
        labels = labels.to(self.device)
        with autocast_ctx(self.use_bf16):
            loss, _ = self.model(feats, pad_mask, labels,
                                 gloss_targets=gloss, gloss_lengths=glen,
                                 ctc_weight=getattr(self.cfg, "ctc_weight", 0.0))
        return loss

    def fit(self, train_dl, val_dl=None):
        from transformers import get_linear_schedule_with_warmup

        cfg = self.cfg
        optim = make_optimizer(self.model.parameters(), cfg.lr,
                               low_vram=getattr(cfg, "low_vram", False))
        total_steps = len(train_dl) * cfg.epochs
        sched = get_linear_schedule_with_warmup(optim, cfg.warmup_steps, total_steps)

        os.makedirs(cfg.out_dir, exist_ok=True)
        best_val = float("inf")
        epochs_sem_melhora = 0
        patience = getattr(cfg, "patience", 0) or 0
        min_delta = getattr(cfg, "min_delta", 0.0) or 0.0
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            running = 0.0
            for step, batch in enumerate(train_dl, 1):
                loss = self._run_batch(batch)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optim.step()
                sched.step()
                running += loss.item()
                if step % cfg.log_every == 0:
                    print(f"epoch {epoch} step {step}/{len(train_dl)} "
                          f"loss {running/step:.4f} lr {sched.get_last_lr()[0]:.2e}")
            train_loss = running / max(1, len(train_dl))

            val_loss = (evaluate(self.model, val_dl, self.device, use_bf16=self.use_bf16)
                        if val_dl else None)
            msg = f"[epoch {epoch}] train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f" val_loss={val_loss:.4f}"
            print(msg)

            # checkpoint da época
            save_checkpoint(os.path.join(cfg.out_dir, f"epoch{epoch}.pt"),
                            self.model, epoch, self.meta)

            # seleção do melhor + contagem para early stopping
            if val_loss is not None:
                improved = val_loss < best_val - min_delta
                if improved:
                    best_val = val_loss
                    epochs_sem_melhora = 0
                    save_checkpoint(os.path.join(cfg.out_dir, "best.pt"),
                                    self.model, epoch, self.meta)
                    self.tokenizer.save_pretrained(cfg.out_dir)
                else:
                    epochs_sem_melhora += 1
                    if patience > 0 and epochs_sem_melhora >= patience:
                        print(f"[early stopping] parando na epoca {epoch}: "
                              f"{epochs_sem_melhora} epocas sem melhora "
                              f"(melhor val_loss={best_val:.4f}).")
                        break
        print("Treino concluido.")


def build_and_train(model_cfg, data_cfg, train_cfg):
    """Monta tokenizer, modelo e dataloaders a partir das configs e treina."""
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    device = resolve_device(train_cfg.device)
    print(f"Dispositivo: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.t5_name)
    t5 = T5ForConditionalGeneration.from_pretrained(model_cfg.t5_name)

    # Supervisão auxiliar de CTC (opcional): constrói o vocabulário de glosas a
    # partir do manifesto de treino e dimensiona o head. Salvo em disco/meta.
    gloss_vocab = None
    if model_cfg.use_ctc:
        from skeltrans.training.gloss_vocab import GlossVocab
        gloss_vocab = GlossVocab.build_from_manifest(data_cfg.train_manifest)
        model_cfg.gloss_vocab_size = len(gloss_vocab)
        os.makedirs(train_cfg.out_dir, exist_ok=True)
        gloss_vocab.save(os.path.join(train_cfg.out_dir, "gloss_vocab.json"))
        print(f"CTC ativo | vocabulario de glosas: {len(gloss_vocab)} rotulos "
              f"(inclui <blank>/<unk>) | peso={train_cfg.ctc_weight}")

    model = SLTModel(t5, d_model=model_cfg.d_model, nhead=model_cfg.nhead,
                     num_layers=model_cfg.num_layers, dropout=model_cfg.dropout,
                     use_ctc=model_cfg.use_ctc,
                     gloss_vocab_size=model_cfg.gloss_vocab_size).to(device)

    # O encoder do T5 nunca executa (a memória vem do LandmarkEncoder via
    # encoder_outputs); com --low-vram seus blocos saem da GPU.
    if getattr(train_cfg, "low_vram", False):
        offload_t5_encoder(t5)

    # Gradient checkpointing: recomputa ativações no backward em vez de
    # guardá-las — essencial quando o OOM vem de sequências longas (a atenção
    # do LandmarkEncoder guarda matrizes (B*nhead, T, T) por camada).
    if getattr(train_cfg, "grad_checkpoint", False):
        model.encoder.grad_checkpoint = True
        t5.config.use_cache = False  # incompatível com checkpointing (só treino)
        t5.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[grad-checkpoint] ativo no LandmarkEncoder e no decoder T5 "
              "(menos VRAM de ativacoes, ~25-30% mais lento)")

    # Treino: collate com glosas quando CTC ativo. Validação: sempre sem glosas
    # (val_loss = só CE, comparável entre configurações).
    train_collate = make_collate(tokenizer, data_cfg.max_text_len, gloss_vocab=gloss_vocab)
    val_collate = make_collate(tokenizer, data_cfg.max_text_len)
    train_ds = LandmarkTextDataset(data_cfg.train_manifest, features_dir=data_cfg.features_dir,
                                   return_gloss=model_cfg.use_ctc)
    train_dl = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=True,
                          collate_fn=train_collate, num_workers=data_cfg.num_workers)
    val_dl = None
    if data_cfg.val_manifest:
        val_ds = LandmarkTextDataset(data_cfg.val_manifest, features_dir=data_cfg.features_dir)
        val_dl = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False,
                            collate_fn=val_collate, num_workers=data_cfg.num_workers)

    meta = {"model": asdict(model_cfg), "data": asdict(data_cfg), "train": asdict(train_cfg)}
    Trainer(model, tokenizer, train_cfg, device, meta=meta).fit(train_dl, val_dl)
