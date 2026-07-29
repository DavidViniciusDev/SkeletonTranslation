"""Vocabulário de glosas para o head CTC auxiliar (opcional).

Só é usado quando a supervisão de reconhecimento (CTC) está ativa. Constrói o
mapa glosa->id a partir do campo ``tokens`` de um manifesto de treino (o mesmo
formato lido por :class:`LandmarkTextDataset`).

Convenção de índices (importante para o CTC):
    0 -> <blank>   (rótulo em branco exigido pelo CTCLoss)
    1 -> <unk>     (glosa fora do vocabulário; defensivo)
    2.. -> glosas ordenadas alfabeticamente

Nada aqui é importado pelo caminho padrão de treino/inferência quando o CTC
está desligado, então o módulo é totalmente encapsulado.
"""

import json

BLANK_ID = 0
UNK_ID = 1
BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"

_FEAT_TOKENS_KEY = "tokens"


class GlossVocab:
    """Mapa bidirecional glosa<->id, com blank/unk reservados."""

    def __init__(self, glosses):
        # `glosses` é a lista de glosas "reais" (sem blank/unk).
        self.itos = [BLANK_TOKEN, UNK_TOKEN] + list(glosses)
        self.stoi = {g: i for i, g in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        """Converte uma lista de glosas em ids (glosa desconhecida -> <unk>)."""
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    # ------------------------------------------------------------------ #
    # Construção / persistência
    # ------------------------------------------------------------------ #
    @classmethod
    def build_from_manifest(cls, manifest_path):
        """Extrai as glosas distintas do campo ``tokens`` de um manifesto."""
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        items = data["items"] if isinstance(data, dict) else data
        vocab = set()
        for it in items:
            for tok in it.get(_FEAT_TOKENS_KEY, []) or []:
                vocab.add(tok)
        return cls(sorted(vocab))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            itos = json.load(f)["itos"]
        # reconstrói pulando blank/unk (os 2 primeiros) que o __init__ recria
        return cls(itos[2:])
