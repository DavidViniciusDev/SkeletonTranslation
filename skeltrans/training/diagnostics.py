"""Smoke test: valida forward/backward/generate sem rede nem dados reais."""

import torch

from skeltrans.common.layout import INPUT_DIM
from skeltrans.training.models import SLTModel


def smoke_test(device=None, use_ctc=False):
    """Valida a arquitetura ponta-a-ponta com um T5 minusculo e dados aleatorios.

    `device`: None (auto: CUDA se disponivel) ou 'cpu' para forcar CPU.
    `use_ctc`: quando True, tambem valida o head CTC (forward/backward) com um
    vocabulario de glosas sintetico. Padrao False (caminho original intacto).
    """
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
    print(f"[smoke] dispositivo: {dev} | use_ctc={use_ctc}")

    # T5 minusculo, inicializado do zero (sem download)
    cfg = T5Config(vocab_size=128, d_model=64, d_ff=128, num_layers=2,
                   num_decoder_layers=2, num_heads=4, d_kv=16, decoder_start_token_id=0,
                   pad_token_id=0, eos_token_id=1)
    t5 = T5ForConditionalGeneration(cfg)
    gloss_vocab_size = 12 if use_ctc else 0        # 12 rotulos ficticios (inclui blank)
    model = SLTModel(t5, d_model=32, nhead=4, num_layers=2, dropout=0.2,
                     use_ctc=use_ctc, gloss_vocab_size=gloss_vocab_size).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] parametros: {n_params/1e6:.2f}M | encoder d_model=32 -> t5 hidden={cfg.d_model}")

    # batch sintetico: 3 sequencias de comprimentos diferentes
    B, lengths = 3, [40, 25, 60]
    T = max(lengths)
    feats = torch.randn(B, T, INPUT_DIM)
    pad_mask = torch.ones(B, T, dtype=torch.bool)
    for i, L in enumerate(lengths):
        pad_mask[i, :L] = False
    labels = torch.randint(2, cfg.vocab_size, (B, 12))
    labels[:, -3:] = -100  # simula padding ignorado
    feats, pad_mask, labels = feats.to(dev), pad_mask.to(dev), labels.to(dev)

    # alvos de glosas sinteticos (ids em [1, V-1]; 0 e blank)
    gloss = glen = None
    ctc_weight = 0.0
    if use_ctc:
        S = 3
        gloss = torch.randint(1, gloss_vocab_size, (B, S)).to(dev)
        glen = torch.full((B,), S, dtype=torch.long).to(dev)
        ctc_weight = 0.3

    # forward + backward
    loss, logits = model(feats, pad_mask, labels,
                         gloss_targets=gloss, gloss_lengths=glen, ctc_weight=ctc_weight)
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())
    ctc_grad_ok = True
    if use_ctc:
        ctc_grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                          for p in model.ctc_head.parameters())
    print(f"[smoke] forward OK | loss={loss.item():.4f} | logits={tuple(logits.shape)} | "
          f"grad_encoder_ok={grad_ok}" + (f" | grad_ctc_ok={ctc_grad_ok}" if use_ctc else ""))

    # generate (inferencia via cross-attention)
    model.eval()
    out = model.generate(feats, pad_mask, max_new_tokens=10, num_beams=2)
    print(f"[smoke] generate OK | saida ids shape={tuple(out.shape)}")
    assert torch.isfinite(loss), "loss nao finita"
    assert logits.shape[0] == B
    assert ctc_grad_ok, "gradiente do head CTC invalido"
    print("[smoke] TUDO OK ✔  (arquitetura do Passo 5 valida ponta-a-ponta)")
