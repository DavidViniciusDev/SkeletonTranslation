"""Pré-treino do decoder em Gloss2Text (glosas -> texto), só com texto.

O T5 é treinado como um tradutor comum, das glosas verdadeiras (campo
``tokens`` dos manifestos) para a tradução (campo ``text``), sem vídeo. Dois
usos (ver ideas/pretreino-decoder-gloss2text.md):

1. **Teto do decoder:** o BLEU no teste com glosas verdadeiras é o melhor que
   esse decoder traduz se o encoder fosse perfeito.
2. **Ponto de partida do SLT:** o modelo salvo em ``<out-dir>/best`` é uma pasta
   do Hugging Face (pesos + tokenizer), usada direto no treino de sinais:

       python3 slt_model.py ... --t5 checkpoints_gloss2text/best

   No SLT, o encoder de texto do T5 não é usado (a memória vem do encoder de
   landmarks); o que se aproveita é o decoder, que já sabe traduzir do
   "espaço das glosas".

Saída, em ``--out-dir``:

- ``best/``: o modelo da melhor época na validação (``--select-by``);
- ``gloss2text_history.jsonl``: perda e métricas por época;
- ``gloss2text_config.json``: a configuração usada;
- ``teto.json``: métricas e traduções no teste (só com ``--test-manifest``).

O ruído nas glosas (``--gloss-noise``) só vale no treino: validação e teste
usam as glosas verdadeiras.
"""

import argparse
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from skeltrans.training.config import SELECT_METRICS, ModelConfig
from skeltrans.training.data.dataset import LandmarkTextDataset
from skeltrans.training.device import resolve_device
from skeltrans.training.low_vram import autocast_ctx, bf16_supported, make_optimizer

NOISE_OPS = ("sub", "del", "ins")
# Proporção padrão das operações do ruído, imitando a decomposição do WER da
# v15/v17 no PHOENIX (substituição ~62%, deleção ~28%, inserção ~10%).
DEFAULT_NOISE_MIX = "0.6,0.3,0.1"
# Marcações sem conteúdo lexical, como __ON__ / __OFF__ / __EMOTION__.
_MARKER_RE = re.compile(r"^__.*__$")


@dataclass
class Gloss2TextConfig:
    """Tudo o que define um pré-treino Gloss2Text (gravado em gloss2text_config.json)."""
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    test_manifest: Optional[str] = None
    t5_name: str = ModelConfig().t5_name
    # --- glosas --- #
    lowercase_glosses: bool = False
    drop_markers: bool = False
    gloss_noise: float = 0.0
    noise_mix: str = DEFAULT_NOISE_MIX
    max_src_len: int = 128
    max_text_len: int = 64
    # --- otimização --- #
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    patience: int = 10
    select_by: str = "bleu4"
    eval_beams: int = 4
    seed: Optional[int] = None
    # --- execução --- #
    num_workers: int = 0
    log_every: int = 50
    out_dir: str = "checkpoints_gloss2text"
    device: Optional[str] = None
    low_vram: bool = False
    grad_checkpoint: bool = False


# ------------------------------------------------------------------------- #
# Glosas: normalização e ruído
# ------------------------------------------------------------------------- #

def normalize_glosses(tokens, lowercase=False, drop_markers=False):
    """Lista de glosas -> lista normalizada (minúsculas, sem marcações __X__)."""
    out = [t for t in tokens if not (drop_markers and _MARKER_RE.match(t))]
    return [t.lower() for t in out] if lowercase else list(out)


def parse_noise_mix(spec):
    """'0.6,0.3,0.1' -> (0.6, 0.3, 0.1) normalizado (substituição, deleção, inserção)."""
    try:
        vals = [float(v) for v in str(spec).split(",")]
    except ValueError as exc:
        raise ValueError(f"--noise-mix invalido: {spec!r}") from exc
    if len(vals) != 3 or any(v < 0 for v in vals) or sum(vals) <= 0:
        raise ValueError(f"--noise-mix precisa de 3 pesos >= 0 com soma > 0 "
                         f"(substituicao,delecao,insercao); recebeu {spec!r}")
    total = sum(vals)
    return tuple(v / total for v in vals)


def corrupt_glosses(tokens, vocab, p, mix, rng):
    """Corrompe cada glosa com probabilidade ``p``, imitando os erros do encoder.

    A operação é sorteada pelos pesos ``mix`` (substituição, deleção,
    inserção): substituir por outra glosa do vocabulário, remover, ou manter e
    inserir uma glosa aleatória depois dela. ``rng`` é um ``random.Random``.
    """
    if p <= 0:
        return list(tokens)
    out = []
    for g in tokens:
        if rng.random() >= p:
            out.append(g)
            continue
        op = rng.choices(NOISE_OPS, weights=mix)[0]
        if op == "sub":
            out.append(rng.choice(vocab))
        elif op == "ins":
            out.extend([g, rng.choice(vocab)])
        # "del": a glosa simplesmente some
    return out


# ------------------------------------------------------------------------- #
# Dados
# ------------------------------------------------------------------------- #

class GlossTextDataset(Dataset):
    """Pares (glosas, texto) de um manifesto, sem features.

    Aceita os mesmos esquemas e chaves do ``LandmarkTextDataset`` (glosas em
    ``tokens``/``gloss_tokens``, texto em ``text``/``pt_br``/...). Itens sem
    glosas ou sem texto são ignorados (e contados em ``skipped``).
    """

    def __init__(self, manifest_path, lowercase=False, drop_markers=False):
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        items = data["items"] if isinstance(data, dict) else data
        self.pairs, self.skipped = [], 0
        for it in items:
            gloss = next((it[k] for k in LandmarkTextDataset._GLOSS_KEYS if k in it), None)
            text = next((it[k] for k in LandmarkTextDataset._TEXT_KEYS if k in it), None)
            if isinstance(gloss, str):
                gloss = gloss.split()
            gloss = normalize_glosses(gloss or [], lowercase, drop_markers)
            if not gloss or not (text and str(text).strip()):
                self.skipped += 1
                continue
            self.pairs.append((gloss, str(text).strip()))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]

    def vocab(self):
        """Glosas distintas do conjunto (ordenadas, para o ruído ser reproduzível)."""
        return sorted({g for gloss, _ in self.pairs for g in gloss})


class Gloss2TextCollate:
    """Lote de pares -> (input_ids, attention_mask, labels, fontes, textos).

    ``labels`` tem -100 no padding (ignorado na perda). Com ``noise > 0``, as
    glosas são corrompidas a cada lote (cada época vê um ruído diferente).
    """

    def __init__(self, tokenizer, max_src_len, max_text_len, vocab=None, noise=0.0,
                 mix=None, rng=None):
        self.tok = tokenizer
        self.max_src_len, self.max_text_len = max_src_len, max_text_len
        self.vocab, self.noise = vocab, noise
        self.mix = mix or parse_noise_mix(DEFAULT_NOISE_MIX)
        self.rng = rng or random.Random(0)

    def __call__(self, batch):
        glosses = [g for g, _ in batch]
        if self.noise > 0:
            glosses = [corrupt_glosses(g, self.vocab, self.noise, self.mix, self.rng)
                       for g in glosses]
        srcs = [" ".join(g) for g in glosses]
        texts = [t for _, t in batch]
        enc = self.tok(srcs, padding=True, truncation=True, max_length=self.max_src_len,
                       return_tensors="pt")
        labels = self.tok(texts, padding=True, truncation=True, max_length=self.max_text_len,
                          return_tensors="pt").input_ids
        labels[labels == self.tok.pad_token_id] = -100
        return enc.input_ids, enc.attention_mask, labels, srcs, texts


# ------------------------------------------------------------------------- #
# Treino e avaliação
# ------------------------------------------------------------------------- #

def seq2seq_loss(model, input_ids, attention_mask, labels, label_smoothing=0.0):
    """Perda de tradução do T5; com ``label_smoothing > 0``, recalculada com
    suavização (mesma média sobre os tokens não ignorados, como no SLT)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    if label_smoothing <= 0:
        return out.loss
    logits = out.logits
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1),
                           ignore_index=-100, label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate(model, tokenizer, dl, device, num_beams=4, max_new_tokens=64, use_bf16=False):
    """Perda (cross-entropy pura) e BLEU/METEOR de ``dl``.

    Devolve ``(val_loss, metrics, rows)``, com ``rows`` = [{src, ref, hyp}].
    """
    from skeltrans.training.metrics import compute_metrics

    model.eval()
    # --grad-checkpoint desliga o KV-cache; religa só durante a geração
    prev_cache = model.config.use_cache
    model.config.use_cache = True
    total, n_batches, rows = 0.0, 0, []
    try:
        for input_ids, attn, labels, srcs, texts in dl:
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            with autocast_ctx(use_bf16):
                total += seq2seq_loss(model, input_ids, attn, labels).item()
                ids = model.generate(input_ids=input_ids, attention_mask=attn,
                                     num_beams=num_beams, max_new_tokens=max_new_tokens)
            n_batches += 1
            hyps = tokenizer.batch_decode(ids, skip_special_tokens=True)
            rows.extend({"src": s, "ref": t.strip(), "hyp": h.strip()}
                        for s, t, h in zip(srcs, texts, hyps))
    finally:
        model.config.use_cache = prev_cache
    metrics = compute_metrics([r["hyp"] for r in rows], [r["ref"] for r in rows])
    return total / max(1, n_batches), metrics, rows


def _loader(ds, collate, cfg, shuffle=False, seed=None):
    kw = {"generator": torch.Generator().manual_seed(seed)} if (shuffle and seed is not None) else {}
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate,
                      num_workers=cfg.num_workers, **kw)


def build_and_train(cfg):
    """Treina o Gloss2Text descrito por ``cfg``. Devolve um resumo (dict)."""
    from transformers import (AutoTokenizer, T5ForConditionalGeneration,
                              get_linear_schedule_with_warmup)

    from skeltrans.training.trainer import BestTracker, seed_everything

    if not cfg.train_manifest:
        raise ValueError("informe --train-manifest.")
    device = resolve_device(cfg.device)
    print(f"Dispositivo: {device}")
    if cfg.seed is not None:
        seed_everything(cfg.seed)
        print(f"[seed] semente {cfg.seed}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.t5_name)
    model = T5ForConditionalGeneration.from_pretrained(cfg.t5_name).to(device)
    norm = {"lowercase": cfg.lowercase_glosses, "drop_markers": cfg.drop_markers}
    train_ds = GlossTextDataset(cfg.train_manifest, **norm)
    if not len(train_ds):
        raise ValueError(f"{cfg.train_manifest} nao tem pares (glosas, texto).")
    vocab = train_ds.vocab()
    mix = parse_noise_mix(cfg.noise_mix)
    print(f"[gloss2text] T5: {cfg.t5_name} | treino: {len(train_ds)} pares "
          f"({train_ds.skipped} ignorados) | {len(vocab)} glosas distintas")
    if cfg.gloss_noise > 0:
        print(f"[gloss2text] ruido nas glosas do treino: p={cfg.gloss_noise:g} "
              f"(substituicao/delecao/insercao = {mix[0]:.2f}/{mix[1]:.2f}/{mix[2]:.2f})")

    noise_rng = random.Random(cfg.seed if cfg.seed is not None else 0)
    train_dl = _loader(train_ds, Gloss2TextCollate(tokenizer, cfg.max_src_len, cfg.max_text_len,
                                                   vocab, cfg.gloss_noise, mix, noise_rng),
                       cfg, shuffle=True, seed=cfg.seed)
    clean = Gloss2TextCollate(tokenizer, cfg.max_src_len, cfg.max_text_len)
    val_dl = None
    if cfg.val_manifest:
        val_ds = GlossTextDataset(cfg.val_manifest, **norm)
        val_dl = _loader(val_ds, clean, cfg)
        print(f"[gloss2text] validacao: {len(val_ds)} pares")

    if cfg.grad_checkpoint:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    use_bf16 = cfg.low_vram and bf16_supported(device)
    optim = make_optimizer(model.parameters(), cfg.lr, low_vram=cfg.low_vram)
    total_steps = len(train_dl) * cfg.epochs
    sched = get_linear_schedule_with_warmup(optim, cfg.warmup_steps, total_steps)

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "gloss2text_config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "command": [sys.executable] + sys.argv},
                  f, ensure_ascii=False, indent=2)
    history_path = os.path.join(cfg.out_dir, "gloss2text_history.jsonl")
    best_dir = os.path.join(cfg.out_dir, "best")
    metric_key = SELECT_METRICS[cfg.select_by]          # None -> val_loss
    tracker = BestTracker(higher_is_better=metric_key is not None)

    epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for step, (input_ids, attn, labels, _, _) in enumerate(train_dl, 1):
            with autocast_ctx(use_bf16):
                loss = seq2seq_loss(model, input_ids.to(device), attn.to(device),
                                    labels.to(device), cfg.label_smoothing)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            running += loss.item()
            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} "
                      f"loss {running / step:.4f} lr {sched.get_last_lr()[0]:.2e}")
        train_loss = running / max(1, len(train_dl))

        record = {"epoch": epoch, "train_loss": train_loss}
        msg = f"[epoch {epoch}] train_loss={train_loss:.4f}"
        if val_dl is not None:
            val_loss, metrics, _ = evaluate(model, tokenizer, val_dl, device, cfg.eval_beams,
                                            cfg.max_text_len, use_bf16)
            record.update(val_loss=val_loss, **metrics)
            msg += f" val_loss={val_loss:.4f} " + " ".join(f"{k}={v:.2f}" for k, v in metrics.items())
            score = metrics[metric_key] if metric_key else val_loss
        else:
            score = -train_loss if metric_key else train_loss   # sem validação: perda de treino
        print(msg)
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if tracker.update(score, epoch):
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  -> best/ (melhor ate agora, epoca {epoch})")
        elif cfg.patience and tracker.epochs_since_best(epoch) >= cfg.patience:
            print(f"[early stopping] {tracker.epochs_since_best(epoch)} epocas sem melhora "
                  f"(melhor na epoca {tracker.best_epoch}).")
            break

    summary = {"best_epoch": tracker.best_epoch, "best_score": tracker.best,
               "select_by": cfg.select_by, "epochs_run": epoch, "best_dir": best_dir}
    if cfg.test_manifest:
        summary["teto"] = evaluate_test(cfg, best_dir, device, use_bf16)
    print(f"Pre-treino Gloss2Text concluido (melhor epoca {tracker.best_epoch}). Use no SLT:\n"
          f"  python3 slt_model.py ... --t5 {best_dir}")
    return summary


def evaluate_test(cfg, best_dir, device, use_bf16=False):
    """Teto do decoder: o melhor modelo traduzindo as glosas VERDADEIRAS do teste.

    O teste não participa de nenhuma escolha (a época vem da validação); ele só
    mede o teto, gravado em ``teto.json`` com as traduções.
    """
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(best_dir)
    model = T5ForConditionalGeneration.from_pretrained(best_dir).to(device)
    test_ds = GlossTextDataset(cfg.test_manifest, lowercase=cfg.lowercase_glosses,
                               drop_markers=cfg.drop_markers)
    dl = _loader(test_ds, Gloss2TextCollate(tokenizer, cfg.max_src_len, cfg.max_text_len), cfg)
    test_loss, metrics, rows = evaluate(model, tokenizer, dl, device, cfg.eval_beams,
                                        cfg.max_text_len, use_bf16)
    print("\n== Teto do decoder (teste, glosas verdadeiras) ==")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v}")
    with open(os.path.join(cfg.out_dir, "teto.json"), "w", encoding="utf-8") as f:
        json.dump({"test_manifest": os.path.abspath(cfg.test_manifest), "best_dir": best_dir,
                   "num_examples": len(rows), "num_beams": cfg.eval_beams,
                   "test_loss": test_loss, "metrics": metrics, "predictions": rows},
                  f, ensure_ascii=False, indent=2)
    return metrics


# ------------------------------------------------------------------------- #
# Linha de comando
# ------------------------------------------------------------------------- #

def build_parser():
    d = Gloss2TextConfig()
    ap = argparse.ArgumentParser(
        description="Pre-treino do decoder em Gloss2Text (glosas -> texto). Gera "
                    "<out-dir>/best, usado no slt_model.py com --t5, e mede o teto do decoder.")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Roda um Gloss2Text minusculo na CPU, com dados e tokenizer sinteticos "
                         "(sem download), e confere que o best/ funciona como --t5 do SLT.")
    ap.add_argument("--train-manifest", help="Manifesto de treino (usa 'tokens' e 'text').")
    ap.add_argument("--val-manifest",
                    help="Manifesto de validacao: escolhe a melhor epoca (--select-by) e o early "
                         "stopping. Sem ele, vale a perda de treino.")
    ap.add_argument("--test-manifest",
                    help="Manifesto de teste: so mede o TETO do decoder no fim (teto.json); nao "
                         "participa de nenhuma escolha.")
    ap.add_argument("--t5", default=d.t5_name,
                    help="T5 de partida (o mesmo que sera usado no SLT). Padrao: o do slt_model.py.")
    g = ap.add_argument_group("glosas")
    g.add_argument("--lowercase-glosses", action="store_true",
                   help="Glosas em minusculas (aproxima as subpalavras do texto comum).")
    g.add_argument("--drop-markers", action="store_true",
                   help="Remove marcacoes sem conteudo lexical no formato __X__ (ex.: __ON__).")
    g.add_argument("--gloss-noise", type=float, default=d.gloss_noise, metavar="P",
                   help="Probabilidade de corromper cada glosa do TREINO, imitando os erros do "
                         "encoder (ex.: 0.35 ~ WER de 35%%). Validacao e teste ficam limpos. "
                         "Padrao: 0 (glosas verdadeiras).")
    g.add_argument("--noise-mix", default=d.noise_mix, metavar="S,D,I",
                   help="Pesos de substituicao, delecao e insercao no ruido. Padrao: "
                         f"{DEFAULT_NOISE_MIX} (a decomposicao do WER da v15/v17).")
    g.add_argument("--max-src-len", type=int, default=d.max_src_len,
                   help="Maximo de tokens das glosas na entrada. Padrao: 128.")
    g.add_argument("--max-text-len", type=int, default=d.max_text_len,
                   help="Maximo de tokens do texto (e da geracao). Padrao: 64, como no SLT.")
    o = ap.add_argument_group("otimizacao")
    o.add_argument("--epochs", type=int, default=d.epochs)
    o.add_argument("--batch-size", type=int, default=d.batch_size)
    o.add_argument("--lr", type=float, default=d.lr)
    o.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    o.add_argument("--grad-clip", type=float, default=d.grad_clip)
    o.add_argument("--label-smoothing", type=float, default=d.label_smoothing,
                   help="Padrao: 0.1 (como na receita v15 do SLT).")
    o.add_argument("--patience", type=int, default=d.patience,
                   help="Epocas sem melhora antes de parar. 0 desliga. Padrao: 10.")
    o.add_argument("--select-by", default=d.select_by, choices=list(SELECT_METRICS),
                   help="Criterio da melhor epoca na validacao. Padrao: bleu4.")
    o.add_argument("--eval-beams", type=int, default=d.eval_beams,
                   help="Beams na geracao da validacao e do teste. Padrao: 4.")
    o.add_argument("--seed", type=int, default=None)
    e = ap.add_argument_group("execucao")
    e.add_argument("--num-workers", type=int, default=d.num_workers,
                   help="Padrao: 0 (dados so de texto; com >0 o ruido deixa de ser reproduzivel).")
    e.add_argument("--log-every", type=int, default=d.log_every)
    e.add_argument("--out-dir", default=d.out_dir)
    e.add_argument("--device", default=None)
    e.add_argument("--low-vram", action="store_true",
                   help="autocast bf16 (GPU Ampere+) e AdamW 8-bit, se disponivel.")
    e.add_argument("--grad-checkpoint", action="store_true")
    return ap


def args_to_config(args):
    return Gloss2TextConfig(
        train_manifest=args.train_manifest, val_manifest=args.val_manifest,
        test_manifest=args.test_manifest, t5_name=args.t5,
        lowercase_glosses=args.lowercase_glosses, drop_markers=args.drop_markers,
        gloss_noise=args.gloss_noise, noise_mix=args.noise_mix, max_src_len=args.max_src_len,
        max_text_len=args.max_text_len, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing, patience=args.patience,
        select_by=args.select_by, eval_beams=args.eval_beams, seed=args.seed,
        num_workers=args.num_workers, log_every=args.log_every, out_dir=args.out_dir,
        device=args.device, low_vram=args.low_vram, grad_checkpoint=args.grad_checkpoint)


def validate_args(args):
    """Mensagem de erro para argumentos inválidos (ou None)."""
    if not args.train_manifest:
        return "Faltou --train-manifest (ou use --smoke-test)."
    if not 0.0 <= args.gloss_noise < 1.0:
        return "--gloss-noise deve estar em [0, 1)."
    if not 0.0 <= args.label_smoothing < 1.0:
        return "--label-smoothing deve estar em [0, 1)."
    if args.eval_beams < 1 or args.epochs < 1 or args.batch_size < 1:
        return "--eval-beams, --epochs e --batch-size devem ser >= 1."
    if SELECT_METRICS[args.select_by] and not args.val_manifest:
        return f"--select-by {args.select_by} requer --val-manifest."
    try:
        parse_noise_mix(args.noise_mix)
    except ValueError as exc:
        return str(exc)
    return None


def toy_tokenizer(words, save_dir):
    """Tokenizer de palavras inteiras, montado localmente (sem download).

    Só para o smoke test e os testes: <pad>=0 (o decoder_start do T5),
    </s>=1 acrescentado ao fim de cada frase, como nos tokenizers T5.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    for w in sorted(set(words)):
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="$A </s>",
                                                       special_tokens=[("</s>", 1)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>",
                                   unk_token="<unk>")
    fast.save_pretrained(save_dir)
    return fast


def write_toy_corpus(tmp, n_train=12, n_val=4):
    """Manifestos sintéticos (glosas -> frase) e um T5 minúsculo com tokenizer
    local, em ``tmp``. Devolve (train, val, t5_dir)."""
    from transformers import T5Config, T5ForConditionalGeneration

    pairs = [(["MORGEN", "REGEN", "NORD"], "morgen regnet es im norden"),
             (["HEUTE", "SONNE", "SUED"], "heute scheint im sueden die sonne"),
             (["__ON__", "WIND", "STARK", "__OFF__"], "der wind wird stark"),
             (["NACHT", "KALT"], "in der nacht wird es kalt")]

    def items(n):
        return [{"features": f"f{i}.npy", "tokens": pairs[i % len(pairs)][0],
                 "text": pairs[i % len(pairs)][1]} for i in range(n)]

    paths = []
    for name, n in (("train", n_train), ("val", n_val)):
        p = os.path.join(tmp, f"{name}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(items(n), f)
        paths.append(p)
    words = [w for g, t in pairs for w in g + [x.lower() for x in g] + t.split()]
    t5_dir = os.path.join(tmp, "t5")
    tok = toy_tokenizer(words, t5_dir)
    cfg = T5Config(vocab_size=len(tok), d_model=32, d_ff=64, num_layers=1, num_decoder_layers=1,
                   num_heads=2, d_kv=16, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    T5ForConditionalGeneration(cfg).save_pretrained(t5_dir)
    return paths[0], paths[1], t5_dir


def smoke_test():
    """Gloss2Text minúsculo (CPU) + conferência de que o best/ serve de --t5 no SLT."""
    import tempfile

    from transformers import T5ForConditionalGeneration

    from skeltrans.common.layout import INPUT_DIM
    from skeltrans.training.models import SLTModel

    with tempfile.TemporaryDirectory() as tmp:
        train, val, t5_dir = write_toy_corpus(tmp)
        out_dir = os.path.join(tmp, "out")
        cfg = Gloss2TextConfig(train_manifest=train, val_manifest=val, test_manifest=val,
                               t5_name=t5_dir, drop_markers=True, gloss_noise=0.2, epochs=2,
                               batch_size=4, warmup_steps=1, eval_beams=2, log_every=1000,
                               out_dir=out_dir, device="cpu", seed=0)
        summary = build_and_train(cfg)
        assert os.path.exists(os.path.join(out_dir, "teto.json"))
        # o best/ precisa funcionar como --t5 do treino de sinais
        t5 = T5ForConditionalGeneration.from_pretrained(summary["best_dir"])
        slt = SLTModel(t5, d_model=16, nhead=2, num_layers=1)
        feats = torch.randn(2, 20, INPUT_DIM)
        mask = torch.zeros(2, 20, dtype=torch.bool)
        loss, _ = slt(feats, mask, torch.tensor([[3, 4, 1], [5, 1, -100]]))
        assert torch.isfinite(loss)
    print(f"[smoke] OK: best/ valido como --t5 do SLT (melhor epoca {summary['best_epoch']}).")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        smoke_test()
        return
    error = validate_args(args)
    if error:
        sys.exit(error)
    build_and_train(args_to_config(args))


if __name__ == "__main__":
    main()
