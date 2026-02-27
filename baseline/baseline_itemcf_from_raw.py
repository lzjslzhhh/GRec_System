import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from eval_recall_metrics import evaluate_rows, save_metrics


def read_eval_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def read_raw_sequences(raw_csv: str) -> Dict[int, List[int]]:
    rows = []
    dropped_invalid = 0
    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader):
            try:
                u_raw = r.get("user_id")
                it_raw = r.get("video_id")
                ts_raw = r.get("timestamp")
                if u_raw in (None, "") or it_raw in (None, "") or ts_raw in (None, ""):
                    dropped_invalid += 1
                    continue
                u = int(float(u_raw))
                it = int(float(it_raw))
                ts = int(float(ts_raw))
            except (TypeError, ValueError):
                dropped_invalid += 1
                continue
            rows.append((u, ts, idx, it))

    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    seqs = defaultdict(list)
    for u, _, __, it in rows:
        seqs[u].append(it)

    if dropped_invalid > 0:
        print(f"drop_invalid_rows: {dropped_invalid}")

    return seqs


def split_by_ratio(seq: List[int], train_ratio: float, val_ratio: float) -> Tuple[List[int], List[int], List[int]]:
    n = len(seq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = seq[:n_train]
    val = seq[n_train:n_train + n_val]
    test = seq[n_train + n_val:]
    return train, val, test


def build_itemcf_topk(
    user_train_seqs: Dict[int, List[int]],
    topk: int,
    co_window: int = 50,
    use_iuf: bool = False,
    min_co: float = 1.0,
) -> Dict[int, List[Tuple[int, float]]]:
    co = defaultdict(Counter)
    cnt = Counter()

    for _, seq in user_train_seqs.items():
        if not seq:
            continue
        if co_window is not None and co_window > 0 and len(seq) > co_window:
            seq = seq[-co_window:]

        items = list(dict.fromkeys(seq))
        w = 1.0
        if use_iuf and items:
            w = 1.0 / math.log(1.0 + len(items))

        for i in items:
            cnt[i] += 1

        L = len(items)
        for a in range(L):
            ia = items[a]
            for b in range(L):
                if a == b:
                    continue
                ib = items[b]
                co[ia][ib] += w

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


def itemcf_recall(
    history: List[int],
    itemcf_topk: Dict[int, List[Tuple[int, float]]],
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
    return [it for it, _ in ranked[:max_k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=str, default="./KuaiRec 2.0/data/small_matrix.csv")
    ap.add_argument("--eval_jsonl", type=str, default="./data/processed/small_matrix_sw/test.jsonl")
    ap.add_argument("--metrics_out", type=str, default="", help="output json for metrics")

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
    max_k = max(ks)

    seqs = read_raw_sequences(args.raw_csv)

    user_train = {}
    for u, seq in seqs.items():
        train, _, _ = split_by_ratio(seq, args.train_ratio, args.val_ratio)
        user_train[u] = train

    itemcf_topk = build_itemcf_topk(
        user_train_seqs=user_train,
        topk=args.topk_sim,
        co_window=args.co_window,
        use_iuf=args.use_iuf,
        min_co=args.min_co,
    )

    eval_data = read_eval_jsonl(args.eval_jsonl)
    rows = []
    for ex in eval_data:
        history = [int(x) for x in ex["history"]]
        target = int(ex["target"])
        recs = itemcf_recall(
            history=history,
            itemcf_topk=itemcf_topk,
            max_k=max_k,
            seen_filter=True,
            recent_n=args.recent_n,
            pos_decay=args.pos_decay,
        )
        rows.append({"rank": recs, "target": target})

    metrics = evaluate_rows(rows, ks)
    metrics["itemcf_size"] = len(itemcf_topk)
    metrics_out = args.metrics_out or os.path.join(os.path.dirname(args.eval_jsonl), "itemcf_metrics.json")
    save_metrics(metrics_out, metrics)

    print(f"users={metrics['users']}  itemcf_size={len(itemcf_topk)}")
    for k in ks:
        print(f"Recall@{k}: {metrics[f'Recall@{k}']:.6f}   NDCG@{k}: {metrics[f'NDCG@{k}']:.6f}")
    print(f"metrics_saved: {metrics_out}")


if __name__ == "__main__":
    main()
