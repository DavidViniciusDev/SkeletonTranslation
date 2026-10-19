"""Loop de treino/avaliação (extraído da antiga função train())."""

import json
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from skeltrans.training.checkpoint import save_checkpoint
from skeltrans.training.data import LandmarkTextDataset, make_collate
from skeltrans.training.data.features import (GROUP_NAMES, compute_input_stats,
                                              feature_dim, parse_groups)
from skeltrans.training.device import resolve_device
from skeltrans.training.config import SELECT_METRICS
from skeltrans.training.distributed import (barrier, broadcast_from_main, cleanup,
                                            init_distributed, is_distributed,
                                            is_main, mean_across_ranks)
from skeltrans.training.low_vram import (autocast_ctx, bf16_supported,
                                         make_optimizer, offload_t5_encoder)
from skeltrans.training.models import SLTModel


@torch.no_grad()
def evaluate(model, dl, device, use_bf16=False):
    """Perda de validação — sempre só a de tradução (CE), sem o termo de CTC,
    para que val_loss seja comparável entre execuções com e sem CTC.
    Em DDP cada rank avalia o seu shard e a média é global (all_reduce)."""
    model.eval()
    total = 0.0
    for batch in dl:
        feats, pad_mask, labels = batch[0], batch[1], batch[2]
        feats, pad_mask, labels = feats.to(device), pad_mask.to(device), labels.to(device)
        with autocast_ctx(use_bf16):
            loss, _ = model(feats, pad_mask, labels)
        total += loss.item()
    return mean_across_ranks(total, len(dl), device)


@torch.no_grad()
def generate_metrics(model, tokenizer, dl, device, num_beams=1, max_new_tokens=64,
                     use_bf16=False):
    """Gera traduções para `dl` e mede BLEU-1..4 e METEOR (mesmas funções do
    evaluate_slt.py). `model` é o SLTModel cru (o wrapper DDP não expõe
    .generate); `dl` usa o collate de avaliação, que devolve os textos.
    Devolve (metrics, hyps, refs)."""
    from skeltrans.training.metrics import compute_metrics

    model.eval()
    # --grad-checkpoint desliga o KV-cache do T5, o que deixa o generate muito
    # lento; religa só durante a geração e restaura em seguida.
    t5 = model.t5
    prev_cache = t5.config.use_cache
    t5.config.use_cache = True
    hyps, refs = [], []
    try:
        for batch in dl:
            feats, pad_mask, texts = batch[0].to(device), batch[1].to(device), batch[2]
            with autocast_ctx(use_bf16):
                ids = model.generate(feats, pad_mask, max_new_tokens=max_new_tokens,
                                     num_beams=num_beams)
            preds = tokenizer.batch_decode(ids, skip_special_tokens=True)
            hyps.extend(p.strip() for p in preds)
            refs.extend(t.strip() for t in texts)
    finally:
        t5.config.use_cache = prev_cache
    return compute_metrics(hyps, refs), hyps, refs


class BestTracker:
    """Guarda o melhor valor de um critério e a época em que ocorreu.

    ``higher_is_better`` define o sentido (métricas de tradução: True;
    val_loss: False). Uma melhora só conta se superar o melhor por mais que
    ``min_delta``. Começa em ±inf, como o ``best_val`` original: o primeiro
    valor válido sempre melhora, e NaN nunca melhora.
    """

    def __init__(self, higher_is_better, min_delta=0.0):
        self.higher = higher_is_better
        self.min_delta = min_delta
        self.best = float("-inf") if higher_is_better else float("inf")
        self.best_epoch = 0          # 0 = nenhuma melhora ainda

    def update(self, value, epoch):
        """Registra `value` da época `epoch`; devolve True se melhorou."""
        if value is None:
            return False
        if self.higher:
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch = value, epoch
        return improved

    def epochs_since_best(self, epoch):
        """Épocas desde a última melhora (conta desde o início se nunca houve)."""
        return epoch - self.best_epoch


class Trainer:
    """Treina um SLTModel com AdamW + agenda linear, salvando checkpoints."""

    def __init__(self, model, tokenizer, train_cfg, device, meta=None, run_info=None):
        self.model = model
        # com DDP, o SLTModel real fica em model.module; é ele que vai no .pt
        self.raw_model = getattr(model, "module", model)
        self.tokenizer = tokenizer
        self.cfg = train_cfg
        self.device = device
        self.meta = meta or {}
        # contexto da execução para o training_config.json (não vai no .pt)
        self.run_info = run_info or {}
        self.is_main = is_main()  # em DDP, só o rank 0 imprime e salva
        # bf16 só quando --low-vram foi pedido E a GPU suporta (Ampere+).
        low_vram = bool(getattr(train_cfg, "low_vram", False))
        self.use_bf16 = low_vram and bf16_supported(device)
        if self.use_bf16 and self.is_main:
            print("[low-vram] autocast bf16 ativo no forward do treino")
        elif low_vram and not self.use_bf16 and self.is_main:
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

    def _eval_metrics(self, epoch, eval_dl, val_loss):
        """Métricas de tradução na validação (--eval-every). Só o rank 0 gera;
        os demais esperam no barrier. Devolve o dict de métricas no rank 0 e
        None nos demais (quem decide com elas replica o valor via broadcast)."""
        cfg = self.cfg
        metrics = None
        if self.is_main:
            num_beams = getattr(cfg, "eval_beams", 1)
            metrics, hyps, refs = generate_metrics(
                self.raw_model, self.tokenizer, eval_dl, self.device,
                num_beams=num_beams,
                max_new_tokens=getattr(cfg, "eval_max_new_tokens", 64),
                use_bf16=self.use_bf16)
            print(f"[epoch {epoch}] val ({len(hyps)} ex., beams={num_beams}) "
                  + " ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
            for h, r in list(zip(hyps, refs))[:2]:   # amostra qualitativa
                print(f"    ref: {r}\n    hyp: {h}")
            with open(os.path.join(cfg.out_dir, "eval_history.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(json.dumps({"epoch": epoch, "val_loss": val_loss,
                                    "num_examples": len(hyps), "num_beams": num_beams,
                                    **metrics}, ensure_ascii=False) + "\n")
        barrier()
        return metrics

    def _save_best(self, name):
        """Salva o modelo atual como `name` (+ tokenizer) no rank 0."""
        if self.is_main:
            save_checkpoint(os.path.join(self.cfg.out_dir, name),
                            self.raw_model, self._epoch, self.meta)
            self.tokenizer.save_pretrained(self.cfg.out_dir)

    def _dump_config(self, optim, total_steps, train_dl, val_dl, eval_dl):
        """Grava <out_dir>/training_config.json com toda a configuração do treino.

        Best-effort: qualquer erro vira aviso e o treino continua normalmente.
        """
        from skeltrans.training.run_info import (environment_info, git_info,
                                                 parameter_counts,
                                                 save_training_config)
        try:
            run = dict(self.run_info)
            run.update({
                "git": git_info(),
                "environment": environment_info(self.device),
                "parameters": parameter_counts(self.raw_model),
                "precision": "bf16 (autocast)" if self.use_bf16 else "fp32",
                "optimizer": type(optim).__name__,
                "scheduler": "linear com warmup (get_linear_schedule_with_warmup)",
                # em DDP, len(train_dl) é o número de passos POR processo
                "steps_per_epoch": len(train_dl),
                "total_steps": total_steps,
                "val_batches": len(val_dl) if val_dl is not None else None,
                "eval_examples": len(eval_dl.dataset) if eval_dl is not None else None,
            })
            path = save_training_config(self.cfg.out_dir, self.meta, run)
            if path:
                print(f"Configuracao do treino salva em {path}")
        except Exception as exc:
            print(f"[config] AVISO: falha ao montar o training_config.json: {exc}")

    def fit(self, train_dl, val_dl=None, eval_dl=None):
        from transformers import get_linear_schedule_with_warmup

        cfg = self.cfg
        # A decisão de avaliar precisa ser IGUAL em todos os ranks (senão o
        # barrier fica desemparelhado e o DDP trava): depende de val_dl, que
        # existe em todos, e não de eval_dl, que só o rank 0 monta.
        eval_every = (getattr(cfg, "eval_every", 0) or 0) if val_dl is not None else 0
        if eval_every and self.is_main and eval_dl is None:
            raise ValueError("eval_every > 0 exige eval_dl no processo principal.")
        if eval_every and self.is_main:
            # falha AGORA se faltar sacrebleu/nltk/WordNet, e não depois de N épocas
            from skeltrans.training.metrics import compute_metrics
            compute_metrics(["teste"], ["teste"])

        # Critério do best.pt / early stopping (--select-by). Com uma métrica de
        # tradução, a val_loss continua acompanhada e salva em best_loss.pt.
        select_by = getattr(cfg, "select_by", "val_loss") or "val_loss"
        if select_by not in SELECT_METRICS:
            raise ValueError(f"select_by desconhecido: {select_by!r} "
                             f"(opcoes: {', '.join(SELECT_METRICS)}).")
        metric_key = SELECT_METRICS[select_by]
        if metric_key and not eval_every:
            raise ValueError(f"select_by={select_by} exige eval_every > 0 e val_dl "
                             "(a metrica so e calculada nas epocas de avaliacao).")
        optim = make_optimizer(self.model.parameters(), cfg.lr,
                               low_vram=getattr(cfg, "low_vram", False),
                               verbose=self.is_main)
        total_steps = len(train_dl) * cfg.epochs
        sched = get_linear_schedule_with_warmup(optim, cfg.warmup_steps, total_steps)

        os.makedirs(cfg.out_dir, exist_ok=True)
        if self.is_main:
            self._dump_config(optim, total_steps, train_dl, val_dl, eval_dl)
        patience = getattr(cfg, "patience", 0) or 0
        min_delta = getattr(cfg, "min_delta", 0.0) or 0.0
        # min_delta está na unidade do critério escolhido; com uma métrica, a
        # val_loss (só para o best_loss.pt) usa qualquer melhora.
        loss_best = BestTracker(higher_is_better=False,
                                min_delta=0.0 if metric_key else min_delta)
        metric_best = BestTracker(higher_is_better=True, min_delta=min_delta)
        for epoch in range(1, cfg.epochs + 1):
            self._epoch = epoch   # usado por _save_best
            self.model.train()
            # em DDP, o sampler precisa da época para variar o embaralhamento
            if hasattr(train_dl.sampler, "set_epoch"):
                train_dl.sampler.set_epoch(epoch)
            running = 0.0
            for step, batch in enumerate(train_dl, 1):
                loss = self._run_batch(batch)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optim.step()
                sched.step()
                running += loss.item()
                if step % cfg.log_every == 0 and self.is_main:
                    print(f"epoch {epoch} step {step}/{len(train_dl)} "
                          f"loss {running/step:.4f} lr {sched.get_last_lr()[0]:.2e}")
            train_loss = running / max(1, len(train_dl))

            # val_loss é global em DDP (all_reduce) -> igual em todos os ranks,
            # logo as decisões de best/early-stopping abaixo são consistentes.
            val_loss = (evaluate(self.model, val_dl, self.device, use_bf16=self.use_bf16)
                        if val_dl else None)
            if self.is_main:
                msg = f"[epoch {epoch}] train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.4f}"
                print(msg)

            # métricas de tradução a cada --eval-every épocas (e na última)
            evaluated = bool(eval_every) and (epoch % eval_every == 0 or epoch == cfg.epochs)
            metrics = self._eval_metrics(epoch, eval_dl, val_loss) if evaluated else None

            # checkpoint da época (rank 0; o estado do modelo é idêntico nos ranks)
            if self.is_main:
                save_checkpoint(os.path.join(cfg.out_dir, f"epoch{epoch}.pt"),
                                self.raw_model, epoch, self.meta)

            # seleção do melhor + contagem para early stopping
            if metric_key is None:
                # --select-by val_loss (padrão): comportamento original
                if val_loss is None:
                    continue
                if loss_best.update(val_loss, epoch):
                    self._save_best("best.pt")
                    continue
                sem_melhora = loss_best.epochs_since_best(epoch)
                if patience > 0 and sem_melhora >= patience:
                    if self.is_main:
                        print(f"[early stopping] parando na epoca {epoch}: "
                              f"{sem_melhora} epocas sem melhora "
                              f"(melhor val_loss={loss_best.best:.4f}).")
                    break
            else:
                # --select-by <métrica>: a val_loss segue só para o best_loss.pt
                if val_loss is not None and loss_best.update(val_loss, epoch):
                    self._save_best("best_loss.pt")
                if not evaluated:
                    continue  # a métrica só existe nas épocas de avaliação
                # valor do rank 0 replicado: todos os ranks decidem igual
                value = broadcast_from_main(
                    metrics.get(metric_key) if metrics else None, self.device)
                if metric_best.update(value, epoch):
                    self._save_best("best.pt")
                    if self.is_main:
                        print(f"[select-by {select_by}] novo melhor {metric_key}="
                              f"{value:.2f} na epoca {epoch} -> best.pt")
                    continue
                sem_melhora = metric_best.epochs_since_best(epoch)
                if patience > 0 and sem_melhora >= patience:
                    if self.is_main:
                        print(f"[early stopping] parando na epoca {epoch}: "
                              f"{sem_melhora} epocas sem melhora de {metric_key} "
                              f"(melhor={metric_best.best:.2f} na epoca "
                              f"{metric_best.best_epoch}).")
                    break
        if self.is_main:
            print("Treino concluido.")


def build_and_train(model_cfg, data_cfg, train_cfg):
    """Monta tokenizer, modelo e dataloaders a partir das configs e treina.

    Multi-GPU: quando lançado via ``torchrun --nproc_per_node=N``, entra em
    DDP automaticamente (1 processo por GPU, lote dividido, gradientes
    sincronizados). Lançado com ``python3``, treina em 1 GPU/CPU como sempre.
    Ver MULTI_GPU.md.
    """
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    if is_distributed():
        device, rank, world = init_distributed()
        main = rank == 0
        if main:
            print(f"Dispositivo: {device} | DDP com {world} processos (1 GPU cada) | "
                  f"batch efetivo = {data_cfg.batch_size} x {world}")
    else:
        device = resolve_device(train_cfg.device)
        main = True
        world = 1
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
        if main:
            os.makedirs(train_cfg.out_dir, exist_ok=True)
            gloss_vocab.save(os.path.join(train_cfg.out_dir, "gloss_vocab.json"))
            print(f"CTC ativo | vocabulario de glosas: {len(gloss_vocab)} rotulos "
                  f"(inclui <blank>/<unk>) | peso={train_cfg.ctc_weight}")

    extra = parse_groups(model_cfg.extra_features)
    model = SLTModel(t5, d_model=model_cfg.d_model, nhead=model_cfg.nhead,
                     num_layers=model_cfg.num_layers, dropout=model_cfg.dropout,
                     ds_steps=model_cfg.ds_steps,
                     use_ctc=model_cfg.use_ctc,
                     gloss_vocab_size=model_cfg.gloss_vocab_size,
                     extra_features=extra,
                     input_norm=model_cfg.input_norm).to(device)
    if main and extra:
        print(f"[features] extras: {','.join(extra)} "
              f"({', '.join(GROUP_NAMES[g] for g in extra)}) | entrada = "
              f"{feature_dim(extra)} colunas")

    # Normalização da entrada: média/desvio por coluna sobre o TREINO. Só o
    # rank 0 calcula; em DDP o construtor do DistributedDataParallel
    # transmite os buffers do rank 0 para os demais.
    if model_cfg.input_norm and main:
        print("[input-norm] calculando media/desvio por coluna no conjunto de treino ...")
        stats_ds = LandmarkTextDataset(data_cfg.train_manifest,
                                       features_dir=data_cfg.features_dir,
                                       extra_features=extra)
        mean, std = compute_input_stats(stats_ds, num_workers=data_cfg.num_workers)
        model.encoder.set_input_stats(mean, std)
        print(f"[input-norm] {len(stats_ds)} frases | desvio por coluna: "
              f"min {std.min():.3g}, mediana {float(np.median(std)):.3g}, max {std.max():.3g}")

    # O encoder do T5 nunca executa (a memória vem do LandmarkEncoder via
    # encoder_outputs); com --low-vram seus blocos saem da GPU. Em DDP o
    # offload não é possível (todos os parâmetros precisam estar na GPU do
    # processo), então os blocos são congelados logo abaixo.
    if getattr(train_cfg, "low_vram", False) and not is_distributed():
        offload_t5_encoder(t5)

    # Gradient checkpointing: recomputa ativações no backward em vez de
    # guardá-las — essencial quando o OOM vem de sequências longas (a atenção
    # do LandmarkEncoder guarda matrizes (B*nhead, T, T) por camada).
    if getattr(train_cfg, "grad_checkpoint", False):
        model.encoder.grad_checkpoint = True
        t5.config.use_cache = False  # incompatível com checkpointing (só treino)
        t5.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if main:
            print("[grad-checkpoint] ativo no LandmarkEncoder e no decoder T5 "
                  "(menos VRAM de ativacoes, ~25-30% mais lento)")

    if is_distributed():
        # O DDP exige que todo parâmetro treinável receba gradiente em cada
        # passo. Os blocos do encoder do T5 nunca recebem (nunca executam),
        # então são congelados — neutro: eles jamais aprendem nada mesmo.
        for p in t5.encoder.block.parameters():
            p.requires_grad_(False)
        for p in t5.encoder.final_layer_norm.parameters():
            p.requires_grad_(False)
        if main and getattr(train_cfg, "low_vram", False):
            print("[low-vram] DDP ativo: offload do encoder T5 substituido por "
                  "congelamento (DDP exige os parametros na GPU do processo)")
        from torch.nn.parallel import DistributedDataParallel as DDP
        # head CTC com peso 0 na perda ficaria sem gradiente; nesse caso o DDP
        # precisa rastrear parâmetros não usados (custa um pouco por passo)
        unused = bool(model_cfg.use_ctc and not train_cfg.ctc_weight)
        model = DDP(model, device_ids=[device.index], find_unused_parameters=unused)

    # Treino: collate com glosas quando CTC ativo. Validação: sempre sem glosas
    # (val_loss = só CE, comparável entre configurações).
    train_collate = make_collate(tokenizer, data_cfg.max_text_len, gloss_vocab=gloss_vocab)
    val_collate = make_collate(tokenizer, data_cfg.max_text_len)
    train_ds = LandmarkTextDataset(data_cfg.train_manifest, features_dir=data_cfg.features_dir,
                                   return_gloss=model_cfg.use_ctc, extra_features=extra)
    # Em DDP, o DistributedSampler dá a cada processo uma fatia distinta do
    # dataset por época (batch_size passa a ser POR GPU).
    train_sampler = None
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_dl = DataLoader(train_ds, batch_size=data_cfg.batch_size,
                          shuffle=(train_sampler is None), sampler=train_sampler,
                          collate_fn=train_collate, num_workers=data_cfg.num_workers)
    val_dl = None
    if data_cfg.val_manifest:
        val_ds = LandmarkTextDataset(data_cfg.val_manifest, features_dir=data_cfg.features_dir,
                                     extra_features=extra)
        val_sampler = None
        if is_distributed():
            from torch.utils.data.distributed import DistributedSampler
            val_sampler = DistributedSampler(val_ds, shuffle=False)
        val_dl = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False,
                            sampler=val_sampler,
                            collate_fn=val_collate, num_workers=data_cfg.num_workers)

    # Métricas de geração (--eval-every): DataLoader extra sobre a MESMA
    # validação, com o collate de avaliação (devolve os textos de referência).
    # Só o rank 0 gera, então só ele o monta, sobre a validação inteira.
    eval_dl = None
    if getattr(train_cfg, "eval_every", 0) and data_cfg.val_manifest and main:
        from skeltrans.training.evaluate import make_eval_collate
        eval_ds = LandmarkTextDataset(data_cfg.val_manifest, features_dir=data_cfg.features_dir,
                                      extra_features=extra)
        n_max = getattr(train_cfg, "eval_max_samples", 0) or 0
        if n_max and n_max < len(eval_ds):
            eval_ds = Subset(eval_ds, range(n_max))
        eval_dl = DataLoader(eval_ds, batch_size=data_cfg.batch_size, shuffle=False,
                             collate_fn=make_eval_collate(),
                             num_workers=data_cfg.num_workers)
        print(f"[eval-every] BLEU/METEOR na validacao a cada {train_cfg.eval_every} "
              f"epoca(s) | {len(eval_ds)} exemplos | beams={train_cfg.eval_beams}")

    meta = {"model": asdict(model_cfg), "data": asdict(data_cfg), "train": asdict(train_cfg)}

    # Contexto da execução para o training_config.json (o meta acima, gravado
    # nos .pt, fica inalterado). Falhar aqui nunca impede o treino.
    run_info = {}
    try:
        from skeltrans.training.run_info import now_iso
        run_info = {
            "started_at": now_iso(),
            "command": [sys.executable] + sys.argv,
            "distributed": is_distributed(),
            "world_size": world,
            "effective_batch_size": data_cfg.batch_size * world,
            "train_examples": len(train_ds),
            "val_examples": len(val_dl.dataset) if val_dl is not None else None,
        }
    except Exception as exc:
        if main:
            print(f"[config] AVISO: falha ao coletar dados da execucao: {exc}")

    try:
        Trainer(model, tokenizer, train_cfg, device, meta=meta,
                run_info=run_info).fit(train_dl, val_dl, eval_dl)
    finally:
        cleanup()
