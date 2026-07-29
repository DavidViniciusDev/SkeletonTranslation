"""Execução paralela em processos + barra de progresso ao vivo.

Este boilerplate estava duplicado (quase verbatim) em extract_landmarks.py e
normalize_landmarks.py: `_fmt_eta`, `_render_progress` e o laço do
ProcessPoolExecutor. Fonte única aqui.

Contrato do worker: recebe uma `task` e retorna uma tupla (key, status, info),
onde status ∈ {"ok", "error"}. Em "error", `info` é a mensagem exibida.
"""

import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def fmt_eta(seconds):
    """Formata segundos como H:MM:SS (ou M:SS quando < 1h)."""
    seconds = int(max(0, seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _render(done, total, ok, failed, start_ts, unit):
    elapsed = time.time() - start_ts
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    pct = (100.0 * done / total) if total else 100.0
    bar_w = 30
    filled = int(bar_w * done / total) if total else bar_w
    bar = "#" * filled + "-" * (bar_w - filled)
    sys.stdout.write(
        f"\r[{bar}] {done}/{total} ({pct:5.1f}%) ok={ok} err={failed} "
        f"| {rate:5.1f} {unit} | ETA {fmt_eta(eta)}   "
    )
    sys.stdout.flush()


def run_pool(tasks, worker, *, workers, unit="arq/s", initializer=None, initargs=()):
    """Roda worker(task) em paralelo (contexto 'spawn'), com progresso ao vivo.

    'spawn' evita problemas de fork com bibliotecas nativas (numpy/BLAS,
    MediaPipe). Retorna (ok, failed, elapsed).
    """
    total = len(tasks)
    done = ok = failed = 0
    start_ts = time.time()
    ctx = multiprocessing.get_context("spawn")
    kwargs = {"max_workers": workers, "mp_context": ctx}
    if initializer is not None:
        kwargs["initializer"] = initializer
        kwargs["initargs"] = initargs
    with ProcessPoolExecutor(**kwargs) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        _render(done, total, ok, failed, start_ts, unit)
        for fut in as_completed(futures):
            key, status, info = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            else:
                failed += 1
                sys.stdout.write("\n")
                print(f"ERRO em {key}: {info}")
            _render(done, total, ok, failed, start_ts, unit)
    return ok, failed, time.time() - start_ts
