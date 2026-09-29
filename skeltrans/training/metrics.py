"""Metricas de avaliacao do modelo SLT.

Traducao (sign2text):
    - BLEU-1..4 cumulativos (via sacrebleu, escala 0-100)
    - METEOR medio no corpus (via nltk, reescalado para 0-100)
    - BERTScore P/R/F1 (opcional, via bert-score, reescalado para 0-100)

Reconhecimento (sign2gloss):
    - WER de glosas + decomposicao em substituicoes/delecoes/insercoes

Todas as metricas ficam na MESMA escala 0-100. Atencao ao sentido: BLEU,
METEOR e BERTScore sao "maior e melhor"; WER e "menor e melhor".

Dependencias: sacrebleu, nltk (+ corpora 'wordnet' e 'omw-1.4'). Se faltar o
WordNet:  python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
O BERTScore e OPCIONAL (pip install bert-score) e so roda quando pedido.
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
    """METEOR medio no corpus (nltk), reescalado de 0-1 para 0-100.

    O `meteor_score` do nltk devolve uma fracao (0-1), enquanto o sacrebleu
    devolve BLEU ja em 0-100. Multiplicar por 100 aqui mantem as duas metricas
    na MESMA escala no dicionario de saida, evitando tabelas em que um valor
    parece duas ordens de grandeza pior que o outro so por causa da unidade.
    E tambem a convencao usada na literatura de SLT, o que torna os numeros
    diretamente comparaveis aos dos papers.
    """
    from nltk.translate.meteor_score import meteor_score

    scores = [
        meteor_score([simple_tokens(r)], simple_tokens(h))
        for h, r in zip(hyps, refs)
    ]
    return round(100.0 * sum(scores) / max(1, len(scores)), 4)


def compute_metrics(hyps, refs):
    """Reune todas as metricas num unico dict."""
    metrics = {}
    metrics.update(compute_bleu(hyps, refs))
    metrics["METEOR"] = compute_meteor(hyps, refs)
    return metrics


# --------------------------------------------------------------------------- #
# Reconhecimento: WER de glosas (metrica nativa do PHOENIX)
# --------------------------------------------------------------------------- #
def _edit_counts(ref, hyp):
    """Levenshtein com backtrace: devolve (sub, del, ins) entre duas listas.

    Custo unitario para as tres operacoes. O backtrace e necessario porque o
    WER convencionalmente reporta a decomposicao, nao so o total: numa base de
    reconhecimento de sinais, muitas DELECOES apontam para sinais nao
    detectados (mao ausente), enquanto muitas SUBSTITUICOES apontam para
    confusao entre sinais parecidos. Sao diagnosticos diferentes.
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = custo minimo alinhando ref[:i] com hyp[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i                      # so delecoes
    for j in range(1, m + 1):
        d[0][j] = j                      # so insercoes
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j - 1],   # substituicao
                                  d[i - 1][j],       # delecao
                                  d[i][j - 1])       # insercao
    # backtrace do canto (n,m) ate (0,0), contando as operacoes usadas
    sub = dele = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1                      # acerto
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins


def compute_wer(hyp_seqs, ref_seqs):
    """WER de glosas AGREGADO no corpus (escala 0-100, menor e melhor).

    `hyp_seqs`/`ref_seqs` sao listas de listas de glosas (tokens ja separados).

    A agregacao e por CONTAGEM no corpus inteiro --
    WER = (S + D + I) / N_ref -- e nao a media dos WER por frase. Essa e a
    definicao padrao (e a que o script oficial do PHOENIX usa): a media por
    frase daria peso desproporcional as frases curtas, em que um unico erro
    vale 100%.
    """
    S = D = I = N = 0
    for hyp, ref in zip(hyp_seqs, ref_seqs):
        s, d, i = _edit_counts(list(ref), list(hyp))
        S += s
        D += d
        I += i
        N += len(ref)
    if N == 0:
        return {"WER": 0.0, "WER_sub": 0.0, "WER_del": 0.0, "WER_ins": 0.0,
                "WER_ref_tokens": 0}
    return {
        "WER": round(100.0 * (S + D + I) / N, 4),
        "WER_sub": round(100.0 * S / N, 4),
        "WER_del": round(100.0 * D / N, 4),
        "WER_ins": round(100.0 * I / N, 4),
        "WER_ref_tokens": N,
    }


# --------------------------------------------------------------------------- #
# Traducao: BERTScore (opcional)
# --------------------------------------------------------------------------- #
def compute_bertscore(hyps, refs, lang, model_type=None, batch_size=64,
                      device=None, rescale_with_baseline=False):
    """BERTScore P/R/F1 medios (escala 0-100).

    E uma metrica NEURAL e REFERENCE-ONLY: compara embeddings contextuais de
    hipotese e referencia por casamento guloso, sem precisar de sentenca-fonte
    -- que em SLT nao existe (a fonte e um video). Por isso ela se aplica aqui,
    enquanto metricas como o COMET, que exigem a tripla (fonte, hipotese,
    referencia), nao se aplicam sem contorcao.

    `lang` seleciona o modelo default do bert-score para o idioma ('de' para o
    PHOENIX, 'pt' para o V-LIBRASIL); `model_type` sobrepoe com um modelo
    especifico da HF.

    Sobre `rescale_with_baseline`: o BERTScore cru vive numa faixa alta e
    estreita (tipicamente 0,8-0,95), o que dificulta ler diferenca entre
    modelos. O reescalonamento por baseline espalha a faixa e melhora a
    legibilidade, mas exige um arquivo de baseline para o idioma -- que nao
    existe para todos. Fica desligado por padrao para nao falhar em idiomas
    sem baseline.

    Levanta ImportError com instrucao de instalacao se o pacote faltar.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError as e:
        raise ImportError(
            "BERTScore pedido mas 'bert-score' nao esta instalado. "
            "Instale com: pip install bert-score") from e

    kwargs = {"batch_size": batch_size, "rescale_with_baseline": rescale_with_baseline}
    if model_type:
        kwargs["model_type"] = model_type
    else:
        kwargs["lang"] = lang
    if device:
        kwargs["device"] = str(device)

    P, R, F1 = bert_score_fn(hyps, refs, **kwargs)
    return {
        "BERTScore_P": round(100.0 * P.mean().item(), 4),
        "BERTScore_R": round(100.0 * R.mean().item(), 4),
        "BERTScore_F1": round(100.0 * F1.mean().item(), 4),
    }
