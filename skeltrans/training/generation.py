"""Parâmetros de decodificação compartilhados (avaliação, inferência e ajuste).

Só os parâmetros informados são repassados ao ``generate`` do T5: sem eles,
valem os padrões do ``generation_config`` do próprio modelo, exatamente como
antes destas flags existirem.
"""


def add_generation_args(ap):
    """Acrescenta ao argparse as flags de decodificação além de beams/tokens."""
    ap.add_argument("--length-penalty", type=float, default=None,
                    help="penalidade de comprimento do beam search: > 1 favorece traducoes "
                         "mais longas, < 1 mais curtas (padrao: o do modelo, normalmente 1.0). "
                         "So tem efeito com --num-beams > 1. Ajuste na VALIDACAO "
                         "(tools/tune_decoding.py), nunca no teste.")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=None,
                    help="proibe repetir qualquer sequencia de N tokens na mesma traducao "
                         "(ex.: 3). Corta repeticoes do tipo 'regen regen regen'. Padrao: o do "
                         "modelo (normalmente desligado).")


def generation_kwargs(args):
    """Dict só com os parâmetros que o usuário definiu (None = padrão do modelo)."""
    out = {}
    if getattr(args, "length_penalty", None) is not None:
        out["length_penalty"] = args.length_penalty
    if getattr(args, "no_repeat_ngram_size", None) is not None:
        out["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    return out
