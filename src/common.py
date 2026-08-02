"""
common.py — shared utilities for the embedding / clustering / classification
pipeline on the king penguin genome (*Aptenodytes patagonicus*, GCF_965638725.1)
using the CrossDNA and JanusDNA foundation models.

This module is deliberately model-agnostic: it imports identically in all three
Docker environments (analysis / crossdna / janusdna). Anything that needs torch
is imported lazily inside the function that uses it, so the analysis environment
stays light and GPU-free.
"""
from __future__ import annotations

import json
import os
import random
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Workspace layout
#
# Every path in the pipeline is derived from PENGUIN_ROOT so the code carries no
# machine-specific absolute paths. Set it in the shell, or accept the default.
# --------------------------------------------------------------------------- #
PENGUIN_ROOT = Path(os.environ.get("PENGUIN_ROOT", Path.home() / "penguin_glm"))
JANUSDNA_REPO = Path(os.environ.get("JANUSDNA_REPO", PENGUIN_ROOT.parent / "JanusDNA"))
CROSSDNA_REPO = Path(os.environ.get("CROSSDNA_REPO", PENGUIN_ROOT.parent / "CrossDNA"))


def workspace_path(*parts: str) -> Path:
    """Resolve a path relative to PENGUIN_ROOT."""
    return PENGUIN_ROOT.joinpath(*parts)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Sequence composition
#
# These are the confounding covariates. Recording them per window is what makes
# the "does the model encode function, or just composition?" question measurable
# rather than rhetorical.
# --------------------------------------------------------------------------- #
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def gc_content(seq: str) -> float:
    s = seq.upper()
    n = len(s)
    if n == 0:
        return 0.0
    return (s.count("G") + s.count("C")) / n


def n_fraction(seq: str) -> float:
    n = len(seq)
    return 0.0 if n == 0 else seq.upper().count("N") / n


def repeat_fraction(seq: str) -> float:
    """Fraction of soft-masked bases (lowercase in the RefSeq FASTA), used as a
    proxy for repeat content.

    Must be computed BEFORE uppercasing the sequence, or the signal is lost.
    """
    n = len(seq)
    if n == 0:
        return 0.0
    return sum(1 for c in seq if c.islower()) / n


# --------------------------------------------------------------------------- #
# k-mer baseline
#
# Raw composition with no learned representation. Model embeddings have to beat
# this to be worth anything; on this task the 4-mer random forest reaches
# MCC 0.243, which several zero-shot embeddings do not clear.
# --------------------------------------------------------------------------- #
def kmer_features(seqs: Sequence[str], k: int = 4) -> np.ndarray:
    idx = {"".join(p): i for i, p in enumerate(product("ACGT", repeat=k))}
    X = np.zeros((len(seqs), len(idx)), dtype=np.float32)
    for r, s in enumerate(seqs):
        s = s.upper()
        tot = 0
        for i in range(len(s) - k + 1):
            j = idx.get(s[i : i + k])  # k-mers containing N are skipped
            if j is not None:
                X[r, j] += 1
                tot += 1
        if tot:
            X[r] /= tot
    return X


# --------------------------------------------------------------------------- #
# Hidden-state pooling (mean / max / cls)
# --------------------------------------------------------------------------- #
def pool_hidden(hidden, attention_mask=None, mode: str = "mean"):
    """hidden: torch.Tensor [B, T, D]; attention_mask: [B, T] (1 = valid).

    Returns [B, D].
    """
    import torch

    if mode == "cls":
        return hidden[:, 0]

    if attention_mask is None:
        attention_mask = torch.ones(hidden.shape[:2], device=hidden.device)
    m = attention_mask.unsqueeze(-1).to(hidden.dtype)

    if mode == "max":
        neg = torch.finfo(hidden.dtype).min
        return hidden.masked_fill(m == 0, neg).max(dim=1).values

    # mean over valid tokens only
    summ = (hidden * m).sum(dim=1)
    cnt = m.sum(dim=1).clamp(min=1)
    return summ / cnt


# --------------------------------------------------------------------------- #
# Embedding I/O
#
# Deliberately minimal and stable (ids + X + meta). Labels and covariates stay
# in the window parquet and are joined by id downstream, so an embedding file
# never carries a stale copy of the annotation.
# --------------------------------------------------------------------------- #
def save_embeddings(path: str, ids, X, meta: dict | None = None) -> None:
    np.savez_compressed(
        path,
        ids=np.asarray(ids).astype(str),
        X=np.asarray(X, dtype=np.float32),
        meta=json.dumps(meta or {}),
    )


def load_embeddings(path: str):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    return d["ids"].astype(str), d["X"], meta


def align_to_ids(emb_ids, X, target_ids):
    """Reorder X to match the order of target_ids.

    Raises if any id is missing, rather than silently misaligning rows — a
    silent misalignment here would corrupt every metric downstream while still
    producing plausible-looking numbers.
    """
    pos = {i: k for k, i in enumerate(emb_ids)}
    try:
        order = [pos[i] for i in target_ids]
    except KeyError as e:
        raise KeyError(
            f"No embedding for id {e}. The embeddings do not cover all windows."
        ) from None
    return X[order]
