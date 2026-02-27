# baseline_itemcf_from_raw.py
# ItemCF / ItemKNN recall baseline (no leakage): build item-item similarity ONLY from raw train interactions.
# Evaluate on processed jsonl (history + target) with seen-filter.

import argparse
import csv
import json
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Iterable


# ----------------------------
# Utils: metrics
# ----------------------------
def dcg_at_k(ranks: List[int], k: int) -> float:
    # ranks: list of 0/1 relevance
    s = 0.0
    for i, rel in enumerate(ranks[:k], start=1):
        if rel:
            s += 1.0 / math.log2(i + 1)
    return s

def ndcg_at_k(hit_rank: int, k: int) -> float:
    # hit_rank: 1-based position if hit else 0
    if hit_rank <= 0 or hit_rank > k:
        return 0.0
    # DCG of single hit at position hit_rank, IDCG=1 at position 1
    return (1.0 / math.log2(hit_rank + 1)) / 1.0

def recall_at_k(hit: bool) -> float:
    return 1.0 if hit else 0.0


# ----------------------------
# Read processed eval jsonl
# ----------------------------
def read_eval_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


# ----------------------------
# Read raw csv and build per-user time-ordered sequences (stable)
# ----------------------------
def read_raw_sequences(raw_csv: str) -> Dict[str, List[str]]:
    """
    Return user -> list of item ids in time order.
    Stable tie-break with row index.
    Expect columns: user_id, video_id, timestamp
    """
    rows = []
    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader):
            u = str(r["user_id"])
            it = str(r["video_id"])
            ts = int(float(r["timestamp"]))
            rows.append((u, ts, idx, it))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))  # user, timestamp, row_idx

    seqs = defaultdict(list)
    for u, _, __, it in rows:
        seqs[u].append(it)
    return seqs


def split_by_ratio(seq: List[str], train_ratio: float, val_ratio: float) -> Tuple[List[str], List[str], List[str]]:
    n = len(seq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = seq[:n_train]
    val = seq[n_train:n_train + n_val]
    test = seq[n_train + n_val:]
    return train, val, test


# ----------------------------
# ItemCF similarity (co-occurrence + cosine)
# ----------------------------
def build_itemcf_topk(
    user_train_seqs: Dict[str, List[str]],
    topk: int,
    co_window: int = 50,
    use_iuf: bool = False,
    min_co: int = 1,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build item-item similarity from user train sequences only.
    Similarity: cosine on co-occurrence matrix (optionally IUF).
    co_window: only consider co-occurrence within recent window per user (optional cap).
    Return item -> list of (neighbor_item, sim) sorted desc, length<=topk.
    """
    # co_count[i][j] and item_count[i]
    co = defaultdict(Counter)     # i -> Counter(j -> co)
    cnt = Counter()              # i -> occurrences

    for u, seq in user_train_seqs.items():
        if not seq:
            continue
        # optional cap window to avoid huge users
        if co_window is not None and co_window > 0 and len(seq) > co_window:
            seq = seq[-co_window:]

        unique_items = seq  # keep duplicates to count frequency; co-occurrence commonly uses unique per user too.
        # If you want "set per user" co-occurrence, replace with list(set(seq)) but keep order not needed here.
        # We'll use set for co edges to avoid quadratic blow-up on repeats:
        items = list(dict.fromkeys(unique_items))  # unique with order

        # IUF weight: penalize very active users / long sequences
        w = 1.0
        if use_iuf:
            w = 1.0 / math.log(1.0 + len(items))

        for i in items:
            cnt[i] += 1
        # pairwise co-occurrence
        L = len(items)
        for a in range(L):
            ia = items[a]
            for b in range(L):
                if a == b:
                    continue
                ib = items[b]
                co[ia][ib] += w

    # compute cosine sim
    topk_neighbors = {}
    for i, nbrs in co.items():
        scored = []
        denom_i = math.sqrt(cnt[i])
        if denom_i == 0:
            continue
        for j, cij in nbrs.items():
            if cij < min_co:
                continue
            denom_j = math.sqrt(cnt[j])
            if denom_j == 0:
                continue
            sim = float(cij) / (denom_i * denom_j)
            scored.append((j, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        if topk > 0:
            scored = scored[:topk]
        topk_neighbors[i] = scored

    return topk_neighbors


# ----------------------------
# Recall: aggregate neighbors from recent history
# ----------------------------
def itemcf_recall(
    history: List[str],
    itemcf_topk: Dict[str, List[Tuple[str, float]]],
    K: int,
    seen_filter: bool = True,
    recent_n: int = 10,
    pos_decay: float = 0.8,
) -> List[str]:
    """
    For a user history, take recent_n items, aggregate their neighbors with weighted score.
    pos_decay: weight decay for older interactions (closer to 1 => less decay).
    """
    if not history:
        return []

    seen = set(history) if seen_filter else set()
    cand_score = defaultdict(float)

    # use last recent_n interactions
    recent = history[-recent_n:] if recent_n > 0 else history
    # newest has idx -1, weight 1; older weight *= pos_decay
    weight = 1.0
    for it in reversed(recent):  # newest -> oldest
        nbrs = itemcf_topk.get(it)
        if not nbrs:
            weight *= pos_decay
            continue
        for j, s in nbrs:
            if seen_filter and j in seen:
                continue
            cand_score[j] += weight * s
        weight *= pos_decay

    ranked = sorted(cand_score.items(), key=lambda x: x[1], reverse=True)
    return [it for it, _ in ranked[:K]]


# ----------------------------
# Evaluate on test.jsonl
# ----------------------------
def evaluate(
    eval_data: List[dict],
    itemcf_topk: Dict[str, List[Tuple[str, float]]],
    ks: List[int],
    recent_n: int,
    pos_decay: float,
) -> Dict[str, float]:
    total = len(eval_data)
    hit_counts = {k: 0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}

    for ex in eval_data:
        history = ex["history"]
        target = ex["target"]

        maxK = max(ks)
        recs = itemcf_recall(
            history=history,
            itemcf_topk=itemcf_topk,
            K=maxK,
            seen_filter=True,
            recent_n=recent_n,
            pos_decay=pos_decay,
        )

        # find rank (1-based)
        hit_rank = 0
        for idx, it in enumerate(recs, start=1):
            if it == target:
                hit_rank = idx
                break

        for k in ks:
            hit = (hit_rank > 0 and hit_rank <= k)
            hit_counts[k] += 1 if hit else 0
            ndcg_sums[k] += ndcg_at_k(hit_rank, k)

    metrics = {}
    for k in ks:
        metrics[f"Recall@{k}"] = hit_counts[k] / total if total else 0.0
        metrics[f"NDCG@{k}"] = ndcg_sums[k] / total if total else 0.0
    metrics["users"] = total
    metrics["itemcf_size"] = len(itemcf_topk)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=str, default='./KuaiRec 2.0/data/small_matrix.csv')
    ap.add_argument("--eval_jsonl", type=str, default='./data/processed/small_matrix_sw/test.jsonl')

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--topk_sim", type=int, default=200, help="neighbors per item")
    ap.add_argument("--co_window", type=int, default=50, help="cap per-user train seq length for co-occurrence")
    ap.add_argument("--use_iuf", action="store_true")
    ap.add_argument("--min_co", type=float, default=1.0)

    ap.add_argument("--ks", type=str, default="20,50,100")
    ap.add_argument("--recent_n", type=int, default=10)
    ap.add_argument("--pos_decay", type=float, default=0.8)

    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    # 1) raw -> per-user sequence
    seqs = read_raw_sequences(args.raw_csv)

    # 2) split by user timeline; ONLY use train segment for statistics
    user_train = {}
    for u, seq in seqs.items():
        train, _, _ = split_by_ratio(seq, args.train_ratio, args.val_ratio)
        user_train[u] = train

    # 3) build itemcf topk neighbors from raw train interactions (no leakage)
    itemcf_topk = build_itemcf_topk(
        user_train_seqs=user_train,
        topk=args.topk_sim,
        co_window=args.co_window,
        use_iuf=args.use_iuf,
        min_co=args.min_co,
    )

    # 4) evaluate on processed test jsonl (history + target)
    eval_data = read_eval_jsonl(args.eval_jsonl)
    metrics = evaluate(
        eval_data=eval_data,
        itemcf_topk=itemcf_topk,
        ks=ks,
        recent_n=args.recent_n,
        pos_decay=args.pos_decay,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()