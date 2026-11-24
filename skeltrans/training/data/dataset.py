"""Dataset que lê um manifesto JSON e devolve (features, texto-alvo)."""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from skeltrans.common.layout import N_POINTS
from skeltrans.training.data.augment import augment as augment_features
from skeltrans.training.data.features import build_input, parse_groups


class LandmarkTextDataset(Dataset):
    """Le um manifesto JSON e devolve (features, texto-alvo).

    Aceita dois esquemas:
      - legado : lista [{"features": "<.npy>", "text": "<pt-br>"}, ...]
      - Passo 4: dict  {"config": {...}, "items": [{"feature_file": "...",
                        "pt_br": "..."}, ...]} (saida de build_sentence_features.py)

    Caminhos de feature relativos sao resolvidos em relacao ao diretorio do
    proprio manifesto (como gravado pelo Passo 4).
    """

    _FEAT_KEYS = ("features", "feature_file", "feature", "npy")
    _TEXT_KEYS = ("text", "pt_br", "pt-br", "target")
    _GLOSS_KEYS = ("tokens", "gloss_tokens")

    def __init__(self, manifest_path, features_dir=None, return_gloss=False,
                 extra_features=(), augment=None):
        # features_dir sobrepoe a localizacao das .npy: quando informado, cada
        # feature e buscada como features_dir/<basename>. Caso contrario, os
        # caminhos relativos do manifesto sao resolvidos em relacao ao proprio
        # manifesto (e caminhos absolutos sao usados como estao).
        self.features_dir = os.path.abspath(features_dir) if features_dir else None
        self.base_dir = os.path.dirname(os.path.abspath(manifest_path))
        # return_gloss=True acrescenta a lista de glosas (tokens) ao item, usada
        # pela supervisão auxiliar de CTC. Padrão False -> retorno inalterado.
        self.return_gloss = return_gloss
        # grupos de features extras (ver features.py), calculados sob demanda e
        # acrescentados ao fim de cada frame. () -> entrada original (1035).
        self.extra_features = parse_groups(extra_features)
        # augmentation (AugmentConfig, ver augment.py): só o Dataset de TREINO
        # recebe; validação, avaliação e as estatísticas do --input-norm, não.
        # None (ou tudo desligado) -> item idêntico ao original.
        self.augment = augment if (augment is not None and augment.enabled) else None
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        # dict com "items" (Passo 4) ou lista pura (legado)
        self.items = data["items"] if isinstance(data, dict) else data
        if not isinstance(self.items, list):
            raise ValueError(
                f"Manifesto {manifest_path} nao contem uma lista de itens "
                f"(encontrado {type(self.items).__name__}).")

    @classmethod
    def _pick(cls, item, keys, path):
        for k in keys:
            if k in item:
                return item[k]
        raise KeyError(
            f"Nenhuma das chaves {keys} encontrada em um item de {path}. "
            f"Chaves presentes: {sorted(item)}")

    def _resolve(self, feat_path):
        if self.features_dir:
            return os.path.join(self.features_dir, os.path.basename(feat_path))
        if os.path.isabs(feat_path):
            return feat_path
        return os.path.join(self.base_dir, feat_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        text = self._pick(it, self._TEXT_KEYS, "manifesto")
        feats = self.load_features(i)
        if self.return_gloss:
            gloss = self._pick_optional(it, self._GLOSS_KEYS, default=[])
            return feats, text, list(gloss)
        return feats, text

    def load_features(self, i):
        """Entrada do encoder do item ``i``: (T, 1035 + extras) float32.

        Lê o .npy, aplica a augmentation (se houver) e monta as features
        extras. Não precisa do texto: o pré-treino do encoder
        (``KeypointDataset``) usa só este método.
        """
        it = self.items[i]
        feat = self._resolve(self._pick(it, self._FEAT_KEYS, "manifesto"))
        arr = np.load(feat).astype(np.float32)            # (T, 115, 9)
        if self.augment is not None:
            if arr.ndim == 2:                             # .npy já achatado (T, 1035)
                arr = arr.reshape(arr.shape[0], N_POINTS, -1)
            # semente vinda do torch: cada worker do DataLoader tem a sua (e o
            # --seed a fixa); o numpy global repetiria os sorteios entre workers
            rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
            arr = augment_features(arr, self.augment, rng)   # (T', 115, 9)
        if self.extra_features:
            arr = build_input(arr, self.extra_features)   # (T, 1035 + extras)
        elif arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)           # (T, 1035)
        return torch.from_numpy(arr)

    @classmethod
    def _pick_optional(cls, item, keys, default=None):
        for k in keys:
            if k in item:
                return item[k]
        return default


class KeypointDataset(LandmarkTextDataset):
    """Só os keypoints de um manifesto, sem texto nem glosas.

    Usado no pré-treino do encoder (``pretrain_encoder.py``): basta que cada
    item aponte para um .npy, então servem manifestos sem tradução. O item é
    a tupla ``(features,)``, para que ``compute_input_stats`` (que lê
    ``item[0]``) funcione igual ao do treino de SLT.
    """

    def __getitem__(self, i):
        return (self.load_features(i),)
