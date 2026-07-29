"""Configuração central da extração: caminhos default e ordem dos passos.

Antes, cada script tinha seus próprios caminhos hardcoded no topo do módulo
(CSV_PATH, GLOSS_JSON, OUT_JSON, IN_DIR, ...). Aqui ficam num só lugar, para
o orquestrador (pipeline.py) e para os defaults de CLI de cada passo.

Layout em data/ (E8), relativo à raiz do projeto:
  data/raw/       entradas externas + landmarks extraídos
  data/interim/   intermediários gerados (vocab, sentenças filtradas, etc.)
  data/features/  features finais (landmarks normalizados, features de frase)
"""

import os
from dataclasses import dataclass

# Raiz da árvore de dados e suas três divisões.
DATA_ROOT = "data"
RAW = os.path.join(DATA_ROOT, "raw")
INTERIM = os.path.join(DATA_ROOT, "interim")
FEATURES = os.path.join(DATA_ROOT, "features")


@dataclass(frozen=True)
class Paths:
    """Artefatos de entrada/saída do pipeline de extração."""

    # entradas externas + landmarks extraídos (raw/)
    gloss_csv: str = os.path.join(RAW, "pt_br2libras_gloss.csv")
    gloss_json: str = os.path.join(RAW, "pt_br2libras_gloss.json")
    videos_words: str = os.path.join(RAW, "videos_words_by_word.json")
    landmarks_dir: str = os.path.join(RAW, "landmarks_115")

    # intermediários gerados (interim/)
    vocab_txt: str = os.path.join(INTERIM, "libras_gloss_vocab.txt")
    in_vocab_json: str = os.path.join(INTERIM, "pt_br2libras_gloss_in_vocab.json")
    sentence_videos_json: str = os.path.join(INTERIM, "pt_br2libras_gloss_sentence_videos.json")

    # features finais (features/)
    landmarks_norm_dir: str = os.path.join(FEATURES, "landmarks_115_norm9")
    features_dir: str = os.path.join(FEATURES, "sentence_features")
    manifest_json: str = os.path.join(FEATURES, "sentence_features", "manifest.json")


# Instância default usada pelos passos e pelo orquestrador.
PATHS = Paths()

# Constantes de domínio do build_sentence_videos.
BASE_INTERPRETER = "0"
BASE_AUGMENTATION = "none"

# Ordem canônica dos passos: (id, módulo, descrição). O orquestrador usa
# os ids para --from/--to.
STEPS = [
    ("s1a", "skeltrans.extraction.steps.s1a_build_vocab", "Vocabulário das glosas"),
    ("s1b", "skeltrans.extraction.steps.s1b_filter_sentences", "Filtra sentenças pelo vocabulário de vídeos"),
    ("s2", "skeltrans.extraction.steps.s2_build_sentence_videos", "Monta sequências de vídeos por sentença"),
    ("s3", "skeltrans.extraction.steps.s3_extract_landmarks", "Extrai landmarks (MediaPipe) dos vídeos"),
    ("s4", "skeltrans.extraction.steps.s4_normalize_landmarks", "Normalização geométrica + dinâmica"),
    ("s5", "skeltrans.extraction.steps.s5_build_sentence_features", "Features de frase (Keyframe Blending) + manifesto"),
]
