"""Collate: empacota um lote de (features, texto[, glosas]) com padding.

Sem `gloss_vocab` o comportamento é o original: devolve (padded, pad_mask,
labels). Com `gloss_vocab`, o dataset deve ter sido criado com
``return_gloss=True`` e cada item vira (features, texto, tokens); o collate
então também devolve (gloss_targets, gloss_lengths) para a perda de CTC.
"""

import torch


def make_collate(tokenizer, max_text_len=64, gloss_vocab=None):
    def collate(batch):
        if gloss_vocab is not None:
            feats, texts, gloss_lists = zip(*batch)
        else:
            feats, texts = zip(*batch)
            gloss_lists = None

        lengths = [f.shape[0] for f in feats]
        T = max(lengths)
        B = len(feats)
        D = feats[0].shape[1]
        padded = torch.zeros(B, T, D, dtype=torch.float32)
        pad_mask = torch.ones(B, T, dtype=torch.bool)     # True = padding
        for i, f in enumerate(feats):
            padded[i, : f.shape[0]] = f
            pad_mask[i, : f.shape[0]] = False
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_text_len, return_tensors="pt")
        labels = enc.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100    # ignora padding na perda

        if gloss_vocab is None:
            return padded, pad_mask, labels

        # alvos de CTC: ids por item, com padding a 0 (=blank, ignorado via lengths)
        encoded = [gloss_vocab.encode(g) for g in gloss_lists]
        gloss_lengths = torch.tensor([max(1, len(e)) for e in encoded], dtype=torch.long)
        S = int(gloss_lengths.max().item())
        gloss_targets = torch.zeros(B, S, dtype=torch.long)
        for i, e in enumerate(encoded):
            if e:
                gloss_targets[i, : len(e)] = torch.tensor(e, dtype=torch.long)
        return padded, pad_mask, labels, gloss_targets, gloss_lengths

    return collate
