import json
import math
import os
from typing import Dict, Iterable, List


def recall_at_k(rank: List[int], target: int, k: int) -> float:
    return 1.0 if target in rank[:k] else 0.0


def ndcg_at_k(rank: List[int], target: int, k: int) -> float:
    if target in rank[:k]:
        pos = rank.index(target)
        return 1.0 / math.log2(pos + 2)
    return 0.0


def evaluate_rows(rows: Iterable[dict], ks: List[int]) -> Dict[str, float]:
    rec_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    users = 0

    for row in rows:
        rank = row["rank"]
        target = row["target"]
        users += 1
        for k in ks:
            rec_sum[k] += recall_at_k(rank, target, k)
            ndcg_sum[k] += ndcg_at_k(rank, target, k)

    metrics = {"users": users}
    for k in ks:
        metrics[f"Recall@{k}"] = rec_sum[k] / users if users else 0.0
        metrics[f"NDCG@{k}"] = ndcg_sum[k] / users if users else 0.0
    return metrics


def save_metrics(path: str, metrics: Dict[str, float]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
