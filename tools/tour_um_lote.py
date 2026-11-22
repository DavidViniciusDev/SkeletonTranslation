#!/usr/bin/env python3
"""Tour guiado: passa UM lote pelo código real do SkeLTrans e mostra cada parada.

Acompanha o guia "Tour guiado pelo código" (guia-tour-codigo-skeltrans.html):
cada bloco impresso tem o número da parada do guia. Nada aqui reimplementa o
modelo — o script chama as mesmas funções do treino (Dataset, collate,
SLTModel, CTC, otimizador) e só registra o que entra e sai de cada uma.

Dois modos:

  # sem dados nem download: frases sintéticas, T5 minúsculo, tokenizer de brinquedo
  python tools/tour_um_lote.py --sintetico

  # com uma frase de verdade do seu manifesto (baixa o --t5 se não estiver em cache)
  python tools/tour_um_lote.py --manifest data_phoenix/interim/dev.json \
      --features-dir data_phoenix/features/landmarks_115_norm9 \
      --t5 GermanT5/t5-efficient-gc4-german-base-nl36 --use-ctc --extra-features A,F

Flags de arquitetura (--ds-steps, --encoder-type, --extra-features,
--input-norm, --use-ctc) seguem o mesmo significado do slt_model.py.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skeltrans.common.layout import N_CHANNELS, N_POINTS  # noqa: E402
from skeltrans.training.data import LandmarkTextDataset, make_collate  # noqa: E402
from skeltrans.training.data.features import compute_input_stats, parse_groups  # noqa: E402
from skeltrans.training.gloss_vocab import GlossVocab  # noqa: E402
from skeltrans.training.models import SLTModel  # noqa: E402


def parada(n, titulo):
    print(f"\n{'=' * 78}\nPARADA {n} · {titulo}\n{'=' * 78}")


def forma(t):
    return tuple(t.shape) if hasattr(t, "shape") else type(t).__name__


# --------------------------------------------------------------------------- #
# Modo sintético: dados e tokenizer de brinquedo (sem rede, sem a base)
# --------------------------------------------------------------------------- #
class TokenizerBrinquedo:
    """Imita o necessário de um tokenizer HF: 1 token por palavra + </s>."""
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab = {"<pad>": 0, "</s>": 1}

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab))

    def __call__(self, texts, padding=True, truncation=True, max_length=64,
                 return_tensors="pt"):
        seqs = [[self._id(w) for w in t.split()][: max_length - 1] + [1] for t in texts]
        L = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (L - len(s)) for s in seqs])

        class Enc:
            input_ids = ids
        return Enc()

    def batch_decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self.vocab.items()}
        return [" ".join(inv.get(int(i), "?") for i in row if int(i) > 1) for row in ids]


def dados_sinteticos():
    """3 frases (T, 115, 9) em disco + manifesto no formato do phoenix_prepare."""
    rng = np.random.default_rng(0)
    pasta = tempfile.mkdtemp(prefix="tour_")
    frases = [("morgen regnet es im norden", ["MORGEN", "NORD", "REGEN"], 100),
              ("am tag sonne", ["TAG", "SONNE"], 60),
              ("nachts wind im sueden", ["NACHT", "WIND", "SUED"], 80)]
    itens = []
    for i, (texto, glosas, T) in enumerate(frases):
        arr = rng.normal(0, 0.5, (T, N_POINTS, N_CHANNELS)).astype(np.float32)
        arr[30:55, 33:54] = 0.0          # mão esquerda "some" no meio da 1ª frase
        np.save(os.path.join(pasta, f"frase{i}.npy"), arr)
        itens.append({"features": f"frase{i}.npy", "text": texto, "tokens": glosas})
    manifesto = os.path.join(pasta, "manifesto.json")
    with open(manifesto, "w", encoding="utf-8") as f:
        json.dump(itens, f, ensure_ascii=False)
    return manifesto, pasta


def t5_minusculo():
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=128, d_model=64, d_ff=128, num_layers=2, num_decoder_layers=2,
                   num_heads=4, d_kv=16, decoder_start_token_id=0, pad_token_id=0,
                   eos_token_id=1)
    return T5ForConditionalGeneration(cfg)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sintetico", action="store_true",
                    help="dados, T5 e tokenizer de brinquedo (padrão se --manifest faltar)")
    ap.add_argument("--manifest", help="manifesto JSON (ex.: o dev.json do PHOENIX)")
    ap.add_argument("--features-dir")
    ap.add_argument("--t5", help="decoder T5 (HF ou pasta); obrigatório sem --sintetico")
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=None,
                    help="padrão: 512 com --manifest, 32 no modo sintético")
    ap.add_argument("--ds-steps", type=int, default=2)
    ap.add_argument("--encoder-type", default="flat", choices=["flat", "parts", "gcn"])
    ap.add_argument("--extra-features", default="")
    ap.add_argument("--input-norm", action="store_true")
    ap.add_argument("--use-ctc", action="store_true")
    ap.add_argument("--ctc-weight", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sintetico = args.sintetico or not args.manifest
    torch.manual_seed(0)
    dev = torch.device(args.device)
    if sintetico:
        manifest, features_dir = dados_sinteticos()
        tokenizer, t5 = TokenizerBrinquedo(), t5_minusculo()
        args.use_ctc = True
        d_model = args.d_model or 32
    else:
        if not args.t5:
            sys.exit("--t5 é obrigatório com --manifest (use o mesmo do treino).")
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        manifest, features_dir = args.manifest, args.features_dir
        tokenizer = AutoTokenizer.from_pretrained(args.t5)
        t5 = T5ForConditionalGeneration.from_pretrained(args.t5)
        d_model = args.d_model or 512
    groups = parse_groups(args.extra_features)

    # ------------------------------------------------------------------ 1
    parada(1, "Montagem (trainer.build_and_train)")
    vocab = GlossVocab.build_from_manifest(manifest) if args.use_ctc else None
    model = SLTModel(t5, d_model=d_model, nhead=4 if d_model < 128 else 8,
                     num_layers=2 if d_model < 128 else 6, ds_steps=args.ds_steps,
                     use_ctc=args.use_ctc, gloss_vocab_size=len(vocab) if vocab else 0,
                     extra_features=groups, input_norm=args.input_norm,
                     encoder_type=args.encoder_type).to(dev)
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    t5_p = sum(p.numel() for p in model.t5.parameters())
    print(f"decoder T5: {t5_p / 1e6:.2f}M parâmetros (hidden {t5.config.d_model})")
    print(f"LandmarkEncoder ({args.encoder_type}, ds_steps={args.ds_steps}): {enc_p / 1e6:.2f}M")
    if vocab:
        print(f"vocabulário de glosas: {len(vocab)} rótulos (0 = <blank>, 1 = <unk>)")

    # ------------------------------------------------------------------ 2
    parada(2, "Um item do Dataset (data/dataset.py, __getitem__)")
    ds = LandmarkTextDataset(manifest, features_dir=features_dir,
                             return_gloss=args.use_ctc, extra_features=groups)
    item = ds[0]
    print(f"features: {forma(item[0])}  <- (T frames, 1035{' + extras' if groups else ''} colunas)")
    print(f"texto alvo: {item[1]!r}")
    if args.use_ctc:
        print(f"glosas: {item[2]}")
    print(f"colunas de base = {N_POINTS} pontos x {N_CHANNELS} canais; extras {groups or 'nenhum'}")

    if args.input_norm:
        mean, std = compute_input_stats(ds)
        model.encoder.set_input_stats(mean, std)
        print(f"input-norm: média/desvio de {len(mean)} colunas calculados no manifesto")

    # ------------------------------------------------------------------ 3
    parada(3, "O lote (data/collate.py)")
    collate = make_collate(tokenizer, 64, gloss_vocab=vocab)
    batch = collate([ds[i] for i in range(min(args.batch_size, len(ds)))])
    feats, pad_mask, labels = batch[0], batch[1], batch[2]
    print(f"feats    {forma(feats)}   <- todas as frases completadas até a maior")
    print(f"pad_mask {forma(pad_mask)}   <- True = frame falso; frames reais por frase: "
          f"{(~pad_mask).sum(1).tolist()}")
    print(f"labels   {forma(labels)}   <- ids do texto; -100 = padding (ignorado na perda)")
    print(f"           1ª frase: {labels[0].tolist()}")
    gloss = glen = None
    if len(batch) == 5:
        gloss, glen = batch[3], batch[4]
        print(f"gloss_targets {forma(gloss)} = {gloss.tolist()}  gloss_lengths = {glen.tolist()}")
    feats, pad_mask, labels = feats.to(dev), pad_mask.to(dev), labels.to(dev)
    if gloss is not None:
        gloss, glen = gloss.to(dev), glen.to(dev)

    # ------------------------------------------------------------------ 4-6
    parada("4-6", "Dentro do encoder (models/landmark_encoder.py)")
    enc = model.encoder
    marcas = []

    def gancho(nome):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            marcas.append((nome, forma(inp[0]), forma(o)))
        return f

    hooks = [enc.input_proj.register_forward_hook(gancho("projeção de entrada")),
             enc.temporal_conv.register_forward_hook(gancho("conv temporal k=3 (B, C, T)"))]
    for i, c in enumerate(enc.temporal_ds):
        hooks.append(c.register_forward_hook(gancho(f"downsampling {i + 1} (stride 2)")))
    hooks.append(enc.transformer.register_forward_hook(gancho("Transformer")))
    if not isinstance(enc.adapter, torch.nn.Identity):
        hooks.append(enc.adapter.register_forward_hook(gancho("adapter -> hidden do T5")))

    model.train()
    enc_out, attn, mem_mask, (ctc_mem, ctc_mask) = model._encode_full(feats, pad_mask,
                                                                      gloss, glen)
    for h in hooks:
        h.remove()
    for nome, a, b in marcas:
        print(f"{nome:32s} {str(a):18s} -> {b}")
    print(f"\nmemory {forma(enc_out.last_hidden_state)} | posições válidas por frase: "
          f"{(~mem_mask).sum(1).tolist()}  (frames reais / 2^{args.ds_steps}, arredondando para cima)")

    # ------------------------------------------------------------------ 7
    parada(7, "Decoder T5 e a perda de tradução (slt_model.forward)")
    out = model.t5(encoder_outputs=enc_out, attention_mask=attn, labels=labels)
    print(f"logits {forma(out.logits)}  <- (lote, tokens do alvo, vocabulário do T5)")
    print(f"CE (tradução) = {out.loss.item():.4f}")
    loss = out.loss

    # ------------------------------------------------------------------ 8
    if args.use_ctc and gloss is not None:
        parada(8, "CTC de glosas (slt_model._ctc_loss)")
        logits_ctc = model._ctc_logits(ctc_mem)
        print(f"logits CTC {forma(logits_ctc)}  <- (lote, posições, glosas + blank)")
        ctc = model._ctc_loss(ctc_mem, ctc_mask, gloss, glen)
        print(f"CTC = {ctc.item():.4f}   (posições {(~ctc_mask).sum(1).tolist()} "
              f"para glosas {glen.tolist()})")
        loss = loss + args.ctc_weight * ctc
        print(f"perda total = CE + {args.ctc_weight} x CTC = {loss.item():.4f}")

    # ------------------------------------------------------------------ 9
    parada(9, "Um passo de otimização (trainer.fit)")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optim.zero_grad()
    loss.backward()
    norma = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    g_enc = sum(p.grad.norm() ** 2 for p in model.encoder.parameters() if p.grad is not None) ** 0.5
    print(f"norma do gradiente (antes do corte em 1.0): {norma.item():.3f}")
    print(f"o encoder recebeu gradiente: {g_enc.item() > 0}")
    optim.step()
    print("pesos atualizados (optim.step); no treino real, o sched.step ajusta o lr")

    # ------------------------------------------------------------------ 10
    parada(10, "Usar o modelo: gerar e reconhecer (evaluate / predict_glosses)")
    model.eval()
    with torch.no_grad():
        ids = model.generate(feats, pad_mask, max_new_tokens=12, num_beams=1)
        print(f"tradução gerada (ids {forma(ids)}): {tokenizer.batch_decode(ids)[0]!r}")
        if args.use_ctc:
            preds = model.predict_glosses(feats, pad_mask)
            print(f"glosas reconhecidas (ids): {preds[0]}  -> {vocab.decode(preds[0])}")
    if sintetico:
        print("\n(modelo sem treino: as saídas são aleatórias — o que importa são as formas)")


if __name__ == "__main__":
    main()
