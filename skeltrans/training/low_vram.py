"""Otimizações opcionais de VRAM (parâmetro --low-vram; ver LOW_VRAM.md).

Três mecanismos independentes, todos atrás do mesmo parâmetro (desligado por
padrão — sem ele o comportamento é exatamente o original):

1. Offload do encoder do T5: o SLTModel injeta a saída do LandmarkEncoder
   direto como ``encoder_outputs`` do T5, então os blocos do encoder do PTT5
   nunca executam — podem sair da GPU sem alterar nenhum resultado.
2. bfloat16: autocast no treino / conversão dos pesos na inferência
   (exige GPU com suporte a bf16, i.e. Ampere ou mais nova).
3. Otimizador AdamW 8-bit (bitsandbytes) no treino: quantiza os estados do
   otimizador, que em fp32 custam 8 bytes por parâmetro treinável.

Cada função degrada graciosamente (com aviso impresso) quando o requisito não
está presente: GPU sem bf16 mantém fp32; bitsandbytes ausente mantém o AdamW
padrão. Nenhum mecanismo altera a arquitetura nem o formato dos checkpoints.
"""

import contextlib

import torch


def offload_t5_encoder(t5, target="cpu"):
    """Move os blocos do encoder do T5 (nunca executados) para fora da GPU.

    O embedding compartilhado (``t5.shared``) fica onde está: é o MESMO objeto
    usado pelo decoder (``t5.decoder.embed_tokens``), então movê-lo quebraria o
    decoder. Por isso o offload cobre os blocos + layer norm final, e não o
    módulo ``t5.encoder`` inteiro.
    """
    enc = getattr(t5, "encoder", None)
    if enc is None:
        return
    enc.block.to(target)
    enc.final_layer_norm.to(target)
    n = sum(p.numel() for p in enc.block.parameters())
    print(f"[low-vram] blocos do encoder do T5 movidos para '{target}' "
          f"({n / 1e6:.0f}M parametros fora da GPU)")


def bf16_supported(device):
    """True quando o dispositivo é uma GPU CUDA com suporte a bfloat16."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    return dev.type == "cuda" and torch.cuda.is_bf16_supported()


def autocast_ctx(enabled):
    """Contexto de autocast bf16 para o forward (no-op quando desligado)."""
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def make_optimizer(params, lr, low_vram=False, verbose=True):
    """AdamW padrão, ou AdamW 8-bit (bitsandbytes) quando ``low_vram=True``.

    ``verbose=False`` silencia os avisos (usado pelos ranks != 0 no DDP).
    """
    if low_vram:
        try:
            import bitsandbytes as bnb
            if verbose:
                print("[low-vram] otimizador AdamW 8-bit (bitsandbytes) ativo")
            return bnb.optim.AdamW8bit(params, lr=lr)
        except ImportError:
            if verbose:
                print("[low-vram] bitsandbytes nao instalado; usando AdamW padrao "
                      "(instale com 'pip install bitsandbytes' para estados 8-bit)")
    return torch.optim.AdamW(params, lr=lr)


def cast_bf16_for_inference(model, device):
    """Converte os pesos para bf16 na inferência (metade da VRAM dos pesos).

    Diferente do treino (autocast, pesos continuam fp32), na inferência não há
    otimizador nem gradientes, então converter os próprios pesos é seguro.
    Em CPU ou GPU sem bf16, mantém fp32 e apenas avisa.
    """
    if bf16_supported(device):
        model.to(torch.bfloat16)
        print("[low-vram] pesos convertidos para bfloat16")
    else:
        print("[low-vram] dispositivo sem suporte a bf16; pesos mantidos em fp32")
    return model
