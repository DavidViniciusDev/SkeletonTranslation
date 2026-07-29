"""Metricas de traducao para a avaliacao do modelo SLT.

    - BLEU-1..4 cumulativos (via sacrebleu, escala 0-100)
    - METEOR medio no corpus (via nltk, escala 0-1)

Dependencias: sacrebleu, nltk (+ corpora 'wordnet' e 'omw-1.4'). Se faltar o
WordNet:  python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""

import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def simple_tokens(text):
    """Tokenizacao simples e deterministica (minusculas + palavras)."""
    return _TOKEN_RE.findall(text.lower())


def compute_bleu(hyps, refs):
    """BLEU-1..4 cumulativos (sacrebleu, escala 0-100)."""
    import sacrebleu

    out = {}
    for n in (1, 2, 3, 4):
        bleu = sacrebleu.BLEU(max_ngram_order=n, effective_order=True)
        out[f"BLEU-{n}"] = round(bleu.corpus_score(hyps, [refs]).score, 4)
    return out


def compute_meteor(hyps, refs):
    """METEOR medio no corpus (nltk, escala 0-1)."""
    from nltk.translate.meteor_score import meteor_score

    scores = [
        meteor_score([simple_tokens(r)], simple_tokens(h))
        for h, r in zip(hyps, refs)
    ]
    return round(sum(scores) / max(1, len(scores)), 4)


def compute_metrics(hyps, refs):
    """Reune todas as metricas num unico dict."""
    metrics = {}
    metrics.update(compute_bleu(hyps, refs))
    metrics["METEOR"] = compute_meteor(hyps, refs)
    return metrics
