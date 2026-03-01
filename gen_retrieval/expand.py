import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_item2sid(path: str) -> Dict[int, int]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def build_sid2items(item2sid: Dict[int, int]) -> Dict[int, List[int]]:
    sid2items: Dict[int, List[int]] = defaultdict(list)
    for item, sid in item2sid.items():
        sid2items[int(sid)].append(int(item))
    return sid2items


def build_raw_train_interactions(
    raw_csv: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    user_col: str = "user_id",
    item_col: str = "video_id",
    ts_col: str = "timestamp",
) -> Tuple[List[Tuple[int, int, int]], Counter]:
    """
    Strictly build train interactions with per-user timeline split.
    Returns:
      - train_triplets: [(user_id, item_id, ts), ...]
      - train_item_pop: Counter(item_id -> count)
    """
    df = pd.read_csv(raw_csv)
    df = df[[user_col, item_col, ts_col]].dropna()
    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df[ts_col] = df[ts_col].astype("int64")
    df[user_col] = df[user_col].astype("int64")
    df[item_col] = df[item_col].astype("int64")

    # Stable order for same timestamp.
    df["__row__"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values([user_col, ts_col, "__row__"], kind="mergesort")

    train_triplets: List[Tuple[int, int, int]] = []
    train_item_pop: Counter = Counter()
    for uid, g in df.groupby(user_col, sort=False):
        seq = list(zip(g[item_col].tolist(), g[ts_col].tolist()))
        n = len(seq)
        if n < 3:
            continue
        n_train = max(int(n * train_ratio), 1)
        n_val = max(int(n * val_ratio), 1)
        if n - n_train - n_val <= 0:
            n_train = max(n - 2, 1)
            n_val = 1
        if n - n_train - n_val <= 0:
            continue
        train_seq = seq[:n_train]
        for item, ts in train_seq:
            train_triplets.append((int(uid), int(item), int(ts)))
            train_item_pop[int(item)] += 1
    return train_triplets, train_item_pop


def cluster_pop_order(
    sid2items: Dict[int, List[int]],
    train_item_pop: Counter,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for sid, items in sid2items.items():
        out[int(sid)] = sorted(items, key=lambda x: (-train_item_pop.get(x, 0), x))
    return out


def load_item2vec_embeddings(
    item_ids_npy: str,
    item_emb_npy: str,
) -> Dict[int, np.ndarray]:
    item_ids = np.load(item_ids_npy).astype(np.int64)
    item_emb = np.load(item_emb_npy).astype(np.float32)
    norms = np.linalg.norm(item_emb, axis=1, keepdims=True)
    item_emb = item_emb / np.maximum(norms, 1e-12)
    return {int(i): v for i, v in zip(item_ids.tolist(), item_emb)}


def expand_by_cluster_pop(
    token_rank: Iterable[int],
    sid2items_pop: Dict[int, List[int]],
    seen_items: set,
    topk_items: int,
) -> List[int]:
    out: List[int] = []
    used = set()
    for sid in token_rank:
        for item in sid2items_pop.get(int(sid), []):
            if item in seen_items or item in used:
                continue
            used.add(item)
            out.append(item)
            if len(out) >= topk_items:
                return out
    return out


def expand_by_cluster_embed(
    token_rank: Iterable[int],
    sid2items_pop: Dict[int, List[int]],
    item_vecs: Dict[int, np.ndarray],
    last_item: int,
    seen_items: set,
    topk_items: int,
) -> List[int]:
    """
    Rerank items inside each predicted cluster by cosine to last history item.
    If embedding missing for last_item or candidate item, fallback to pop order.
    """
    out: List[int] = []
    used = set()
    q = item_vecs.get(int(last_item))
    for sid in token_rank:
        base = sid2items_pop.get(int(sid), [])
        if q is None:
            ordered = base
        else:
            scored = []
            for item in base:
                v = item_vecs.get(int(item))
                sim = float(np.dot(q, v)) if v is not None else -2.0
                scored.append((sim, item))
            scored.sort(key=lambda x: (-x[0], x[1]))
            ordered = [it for _, it in scored]
        for item in ordered:
            if item in seen_items or item in used:
                continue
            used.add(item)
            out.append(item)
            if len(out) >= topk_items:
                return out
    return out


def target_in_train_ratio(test_rows: List[dict], train_items: set) -> float:
    if not test_rows:
        return 0.0
    c = 0
    for ex in test_rows:
        if int(ex["target"]) in train_items:
            c += 1
    return float(c / len(test_rows))


def target_in_item2sid_ratio(test_rows: List[dict], item2sid: Dict[int, int]) -> float:
    if not test_rows:
        return 0.0
    c = 0
    for ex in test_rows:
        if int(ex["target"]) in item2sid:
            c += 1
    return float(c / len(test_rows))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

