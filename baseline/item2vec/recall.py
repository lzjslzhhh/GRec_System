from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def build_topk_neighbors(
    item_ids: np.ndarray,
    emb: np.ndarray,
    topk: int,
    device: str = "cpu",
) -> Dict[int, List[Tuple[int, float]]]:
    item_ids = np.asarray(item_ids, dtype=np.int64)
    emb = np.asarray(emb, dtype=np.float32)
    if item_ids.ndim != 1 or emb.ndim != 2 or emb.shape[0] != item_ids.shape[0]:
        raise ValueError("Expected item_ids[N] and embeddings[N, D].")

    x = torch.from_numpy(emb).to(device)
    x = F.normalize(x, p=2, dim=1)
    sim = x @ x.T
    sim.fill_diagonal_(-1.0)

    n = sim.shape[0]
    if topk <= 0:
        topk = n - 1
    topk = min(topk, n - 1)

    val, idx = torch.topk(sim, k=topk, dim=1, largest=True, sorted=True)
    idx = idx.cpu().numpy()
    val = val.cpu().numpy()

    neighbors: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(n):
        src = int(item_ids[i])
        nbrs = [(int(item_ids[j]), float(s)) for j, s in zip(idx[i].tolist(), val[i].tolist())]
        neighbors[src] = nbrs
    return neighbors


def recall_from_neighbors(
    history: List[int],
    neighbors: Dict[int, List[Tuple[int, float]]],
    max_k: int,
    seen_filter: bool = True,
    recent_n: int = 10,
    pos_decay: float = 0.8,
) -> List[int]:
    if not history:
        return []

    seen = set(history) if seen_filter else set()
    cand_score = defaultdict(float)

    recent = history[-recent_n:] if recent_n > 0 else history
    weight = 1.0
    for it in reversed(recent):
        nbrs = neighbors.get(it)
        if not nbrs:
            weight *= pos_decay
            continue
        for j, s in nbrs:
            if seen_filter and j in seen:
                continue
            cand_score[j] += weight * s
        weight *= pos_decay

    ranked = sorted(cand_score.items(), key=lambda x: x[1], reverse=True)
    return [it for it, _ in ranked[:max_k]]


def reachable_set(
    history: List[int],
    neighbors: Dict[int, List[Tuple[int, float]]],
    seen_filter: bool = True,
    recent_n: int = 10,
) -> set:
    if not history:
        return set()
    seen = set(history) if seen_filter else set()
    recent = history[-recent_n:] if recent_n > 0 else history

    out = set()
    for it in recent:
        nbrs = neighbors.get(it)
        if not nbrs:
            continue
        for j, _ in nbrs:
            if seen_filter and j in seen:
                continue
            out.add(j)
    return out
