#!/usr/bin/env python3
"""Raio-X das features de entrada do encoder: comprimento e magnitude.

Responde duas perguntas sobre os .npy (T,115,9) que alimentam o modelo:

1. COMPRIMENTO (T, frames por frase): media, percentis, maximo e os maiores
   arquivos. Como o collate faz padding ate o MAIOR T de cada lote, uma frase
   muito longa faz o lote inteiro carregar padding; o relatorio simula lotes
   aleatorios com o --batch-size real e mede quanto do tensor vira padding
   (e quanto sobraria agrupando frases de tamanho parecido, o "bucketing").

2. MAGNITUDE dos valores, por grupo do corpo (pose, maos, face) e por canal
   (posicao X/Y/Z, velocidade, aceleracao): media, desvio, percentis de |x|,
   maximo, NaN/inf. As features entram CRUAS no input_proj (nao ha
   normalizacao de entrada), entao um canal com escala muito maior que os
   outros domina a projecao. Apos o Passo 3 as posicoes estao em unidades de
   "largura dos ombros": um ponto a mais de --pos-limit larguras do centro e
   fisicamente impossivel e indica frame com escala degenerada (ombros mal
   detectados -> divisao por uma distancia minuscula).

Convencao: [0,0,0] marca ponto AUSENTE (Passo 2/3). As medias/desvios usam
TODOS os valores (e o que a rede ve); os percentis usam so os valores nao
nulos (a distribuicao real de quem foi detectado).

Uso:
    python3 tools/landmark_stats.py --manifest data/interim/train.json \\
        --features-dir data/features/sentence_features
    python3 tools/landmark_stats.py --dir data/features/sentence_features --sample 2000
    python3 tools/landmark_stats.py --manifest train.json val.json --out stats.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fatias dos 115 pontos: fonte unica do projeto (evita duplicar os limites).
from skeltrans.common.layout import (FACE, LEFT_HAND, N_POINTS,  # noqa: E402
                                     POSE, RIGHT_HAND)

GROUPS = {"pose": POSE, "mao_esq": LEFT_HAND, "mao_dir": RIGHT_HAND, "face": FACE}
KINDS = {"pos": slice(0, 3), "vel": slice(3, 6), "acc": slice(6, 9)}

# Histograma log de |x| (1e-6 .. 1e6) para percentis aproximados sem guardar
# todos os valores na memoria. Bin 0 = abaixo de 1e-6; ultimo = acima de 1e6.
EDGES = np.logspace(-6, 6, 241)
PCTS = (50, 90, 99, 99.9)


# --------------------------------------------------------------------------- #
# Entrada: lista de .npy a partir de manifestos ou de um diretorio
# --------------------------------------------------------------------------- #
def files_from_manifest(path, features_dir):
    """Mesma resolucao de caminhos do LandmarkTextDataset (sem importar torch)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"] if isinstance(data, dict) else data
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for it in items:
        feat = next((it[k] for k in ("features", "feature_file", "feature", "npy") if k in it), None)
        if feat is None:
            raise KeyError(f"item sem caminho de feature em {path}: chaves {sorted(it)}")
        if features_dir:
            out.append(os.path.join(features_dir, os.path.basename(feat)))
        elif os.path.isabs(feat):
            out.append(feat)
        else:
            out.append(os.path.join(base, feat))
    return out


def load_features(path):
    """Carrega como (T,115,C). Aceita (T,115,9), (T,115,3) e o achatado (T,1035)."""
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2 and arr.shape[1] % N_POINTS == 0:
        arr = arr.reshape(arr.shape[0], N_POINTS, -1)
    if arr.ndim != 3 or arr.shape[1] != N_POINTS:
        raise ValueError(f"shape inesperado {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Acumulador por (grupo, canal)
# --------------------------------------------------------------------------- #
class Acc:
    def __init__(self):
        self.n = 0            # valores vistos
        self.n_zero = 0       # exatamente 0 (ausente ou parado)
        self.n_bad = 0        # NaN/inf
        self.s = 0.0          # soma (para a media)
        self.ss = 0.0         # soma dos quadrados (para o desvio)
        self.maxabs = 0.0
        self.hist = np.zeros(len(EDGES) + 1, dtype=np.int64)

    def add(self, x):
        x = x.ravel()
        finite = np.isfinite(x)
        self.n_bad += int(x.size - finite.sum())
        x = x[finite]
        a = np.abs(x)
        self.n += x.size
        self.n_zero += int((a == 0).sum())
        self.s += float(x.sum())
        self.ss += float((x * x).sum())
        if a.size:
            self.maxabs = max(self.maxabs, float(a.max()))
        nz = a[a > 0]
        self.hist += np.bincount(np.searchsorted(EDGES, nz, side="right"),
                                 minlength=len(EDGES) + 1)

    def percentile(self, q):
        """Percentil aproximado de |x| (nao nulos): borda superior do bin."""
        tot = self.hist.sum()
        if not tot:
            return 0.0
        idx = int(np.searchsorted(np.cumsum(self.hist), q / 100.0 * tot))
        return float(EDGES[min(idx, len(EDGES) - 1)])

    def summary(self):
        mean = self.s / self.n if self.n else 0.0
        var = max(0.0, self.ss / self.n - mean * mean) if self.n else 0.0
        out = {"media": mean, "desvio": var ** 0.5, "max_abs": self.maxabs,
               "frac_zero": self.n_zero / self.n if self.n else 0.0, "nan_inf": self.n_bad}
        out.update({f"p{q:g}_abs": self.percentile(q) for q in PCTS})
        return out


# --------------------------------------------------------------------------- #
# Padding: simulacao dos lotes do DataLoader
# --------------------------------------------------------------------------- #
def padding_sim(lengths, batch_size, seed=0, rounds=20):
    """Fracao de padding em lotes aleatorios (shuffle, como no treino) e com
    bucketing (frases de tamanho parecido no mesmo lote). Tambem o custo
    relativo de atencao, que cresce com (T_max do lote)^2."""
    L = np.asarray(lengths)
    rng = np.random.default_rng(seed)

    def stats(order):
        real = padded = att = 0.0
        tmax = []
        for i in range(0, len(order), batch_size):
            b = L[order[i:i + batch_size]]
            m = b.max()
            real += b.sum()
            padded += m * len(b)
            att += (m ** 2) * len(b)
            tmax.append(m)
        return 1 - real / padded, att, tmax

    shuf = [stats(rng.permutation(len(L))) for _ in range(rounds)]
    bucket = stats(np.argsort(L, kind="stable"))
    ideal_att = float((L.astype(np.float64) ** 2).sum())   # sem padding nenhum
    return {
        "batch_size": batch_size,
        "padding_aleatorio": float(np.mean([s[0] for s in shuf])),
        "padding_bucketing": float(bucket[0]),
        "atencao_vs_sem_padding_aleatorio": float(np.mean([s[1] for s in shuf]) / ideal_att),
        "atencao_vs_sem_padding_bucketing": float(bucket[1] / ideal_att),
        "T_max_lote_mediana_aleatorio": float(np.median(np.concatenate([s[2] for s in shuf]))),
    }


def describe(values):
    v = np.asarray(values, dtype=np.float64)
    out = {"n": int(v.size), "media": float(v.mean()), "desvio": float(v.std()),
           "min": float(v.min()), "max": float(v.max())}
    out.update({f"p{q:g}": float(np.percentile(v, q)) for q in (50, 90, 95, 99)})
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", nargs="+", help="um ou mais manifestos JSON (treino/val/teste)")
    src.add_argument("--dir", help="diretorio com os .npy (usa todos *.npy)")
    ap.add_argument("--features-dir", default=None,
                    help="sobrepoe a pasta dos .npy (mesma semantica do treino)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="lote usado na simulacao de padding (use o mesmo do treino). Padrao: 8")
    ap.add_argument("--sample", type=int, default=0,
                    help="analisa so N arquivos sorteados (0 = todos). Os comprimentos usam todos.")
    ap.add_argument("--pos-limit", type=float, default=10.0,
                    help="|posicao| acima disto (em larguras de ombro) conta como frame "
                         "outlier. Padrao: 10")
    ap.add_argument("--top", type=int, default=10, help="quantos arquivos extremos listar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="grava o relatorio completo em JSON")
    args = ap.parse_args()
    if args.batch_size < 1 or args.sample < 0 or args.top < 0 or args.pos_limit <= 0:
        sys.exit("--batch-size >= 1, --sample/--top >= 0 e --pos-limit > 0.")

    if args.manifest:
        files = [p for m in args.manifest for p in files_from_manifest(m, args.features_dir)]
    else:
        files = sorted(glob.glob(os.path.join(args.dir, "*.npy")))
    if not files:
        sys.exit("Nenhum .npy encontrado.")

    # --- 1) comprimentos: so le o cabecalho (mmap), barato mesmo com milhares ---
    lengths, missing = {}, []
    for p in files:
        try:
            lengths[p] = int(np.load(p, mmap_mode="r").shape[0])
        except (OSError, ValueError):
            missing.append(p)
    if not lengths:
        sys.exit(f"Nenhum .npy legivel ({len(missing)} faltando). Confira --features-dir.")

    # --- 2) magnitudes: le os valores (opcionalmente numa amostra) ---
    paths = list(lengths)
    if args.sample and args.sample < len(paths):
        rng = np.random.default_rng(args.seed)
        paths = [paths[i] for i in sorted(rng.choice(len(paths), args.sample, replace=False))]

    accs = {(g, k): Acc() for g in GROUPS for k in KINDS}
    per_file = []
    n_frames = n_out_frames = 0
    for i, p in enumerate(paths, 1):
        try:
            arr = load_features(p)
        except (OSError, ValueError) as e:
            print(f"  [aviso] ignorado {os.path.basename(p)}: {e}")
            continue
        kinds = {k: s for k, s in KINDS.items() if s.stop <= arr.shape[2]}
        worst = (0.0, None)
        for g, gs in GROUPS.items():
            for k, ks in kinds.items():
                block = arr[:, gs, ks]
                accs[(g, k)].add(block)
                fin = block[np.isfinite(block)]
                m = float(np.abs(fin).max()) if fin.size else 0.0
                if m > worst[0]:
                    worst = (m, f"{g}/{k}")
        pos = np.abs(arr[:, :, 0:3])
        out_frames = int((np.nan_to_num(pos, nan=np.inf).max(axis=(1, 2)) > args.pos_limit).sum())
        n_frames += arr.shape[0]
        n_out_frames += out_frames
        per_file.append({"arquivo": p, "T": arr.shape[0], "max_abs": worst[0],
                         "onde": worst[1], "frames_outlier": out_frames,
                         "nan_inf": int((~np.isfinite(arr)).sum())})
        if i % 500 == 0:
            print(f"  ... {i}/{len(paths)} arquivos")

    # --- relatorio ---
    L = list(lengths.values())
    len_stats = describe(L)
    by_len = sorted(lengths.items(), key=lambda kv: -kv[1])
    pad = padding_sim(L, args.batch_size, seed=args.seed)
    mag = {f"{g}/{k}": accs[(g, k)].summary() for g in GROUPS for k in KINDS
           if accs[(g, k)].n}

    print(f"\n== Arquivos: {len(lengths)} legiveis"
          + (f", {len(missing)} faltando" if missing else "")
          + (f" | magnitudes em amostra de {len(per_file)}" if len(per_file) < len(lengths) else ""))

    print("\n== 1) Comprimento das sequencias (T, em frames) ==")
    print("  media {media:.1f} | desvio {desvio:.1f} | min {min:.0f} | mediana {p50:.0f} | "
          "p90 {p90:.0f} | p95 {p95:.0f} | p99 {p99:.0f} | MAX {max:.0f}".format(**len_stats))
    print(f"  max/mediana = {len_stats['max'] / max(1, len_stats['p50']):.1f}x")
    print(f"  {args.top} mais longos:")
    for p, t in by_len[:args.top]:
        print(f"    {t:6d}  {os.path.basename(p)}")
    print(f"\n  Padding com --batch-size {pad['batch_size']}:")
    print(f"    lotes aleatorios (treino atual): {pad['padding_aleatorio']:.1%} do tensor e padding "
          f"| atencao custa {pad['atencao_vs_sem_padding_aleatorio']:.2f}x o necessario")
    print(f"    com bucketing por tamanho      : {pad['padding_bucketing']:.1%} "
          f"| {pad['atencao_vs_sem_padding_bucketing']:.2f}x")

    print("\n== 2) Magnitude dos valores (pos em larguras de ombro; percentis de |x| nao nulo) ==")
    hdr = f"  {'grupo/canal':16s} {'media':>8s} {'desvio':>8s} {'p50':>8s} {'p99':>8s} " \
          f"{'p99.9':>8s} {'MAX':>10s} {'zeros':>6s} {'nan/inf':>7s}"
    print(hdr)
    for name, s in mag.items():
        print(f"  {name:16s} {s['media']:8.3f} {s['desvio']:8.3f} {s['p50_abs']:8.3g} "
              f"{s['p99_abs']:8.3g} {s['p99.9_abs']:8.3g} {s['max_abs']:10.4g} "
              f"{s['frac_zero']:6.1%} {s['nan_inf']:7d}")
    print(f"\n  Frames com |pos| > {args.pos_limit:g} larguras de ombro: {n_out_frames} de "
          f"{n_frames} ({n_out_frames / max(1, n_frames):.3%}) em "
          f"{sum(1 for f in per_file if f['frames_outlier'])} arquivo(s)")
    print(f"  {args.top} arquivos com maior |valor|:")
    for f in sorted(per_file, key=lambda f: -f["max_abs"])[:args.top]:
        print(f"    {f['max_abs']:12.4g}  em {f['onde']:12s} | T={f['T']:5d} | "
              f"frames outlier={f['frames_outlier']:4d}  {os.path.basename(f['arquivo'])}")

    if args.out:
        report = {"n_arquivos": len(lengths), "faltando": missing, "comprimento": len_stats,
                  "mais_longos": [{"arquivo": p, "T": t} for p, t in by_len[:args.top]],
                  "padding": pad, "magnitude": mag, "pos_limit": args.pos_limit,
                  "frames_total": n_frames, "frames_outlier": n_out_frames,
                  "por_arquivo": per_file}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nRelatorio completo em: {args.out}")


if __name__ == "__main__":
    main()
