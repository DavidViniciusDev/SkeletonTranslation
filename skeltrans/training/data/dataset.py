"""Dataset que lê um manifesto JSON e devolve (features, texto-alvo)."""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


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

    def __init__(self, manifest_path, features_dir=None, return_gloss=False):
        # features_dir sobrepoe a localizacao das .npy: quando informado, cada
        # feature e buscada como features_dir/<basename>. Caso contrario, os
        # caminhos relativos do manifesto sao resolvidos em relacao ao proprio
        # manifesto (e caminhos absolutos sao usados como estao).
        self.features_dir = os.path.abspath(features_dir) if features_dir else None
        self.base_dir = os.path.dirname(os.path.abspath(manifest_path))
        # return_gloss=True acrescenta a lista de glosas (tokens) ao item, usada
        # pela supervisão auxiliar de CTC. Padrão False -> retorno inalterado.
        self.return_gloss = return_gloss
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
        feat = self._resolve(self._pick(it, self._FEAT_KEYS, "manifesto"))
        text = self._pick(it, self._TEXT_KEYS, "manifesto")
        arr = np.load(feat).astype(np.float32)            # (T, 115, 9)
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)           # (T, 1035)
        if self.return_gloss:
            gloss = self._pick_optional(it, self._GLOSS_KEYS, default=[])
            return torch.from_numpy(arr), text, list(gloss)
        return torch.from_numpy(arr), text

    @classmethod
    def _pick_optional(cls, item, keys, default=None):
        for k in keys:
            if k in item:
                return item[k]
        return default
