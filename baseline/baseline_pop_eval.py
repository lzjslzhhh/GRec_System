import argparse
import json
import math
from collections import Counter
import pandas as pd

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def time_split_indices(n, train_ratio=0.8, val_ratio=0.1):
    if n < 3:
        return None
    n_train = max(int(n * train_ratio), 1)
    n_val = max(int(n * val_ratio), 1)
    if n - n_train - n_val <= 0:
        n_train = max(n - 2, 1)
        n_val = 1
        if n - n_train - n_val <= 0:
            return None
    return n_train, n_val

def recall_at_k(rank, target, k):
    return 1.0 if target in rank[:k] else 0.0

def ndcg_at_k(rank, target, k):
    if target in rank[:k]:
        p = rank.index(target)
        return 1.0 / math.log2(p + 2)
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=str, default='./KuaiRec 2.0/data/small_matrix.csv', help="small_matrix.csv or big_matrix.csv")
    ap.add_argument("--eval_jsonl", type=str, default='./data/processed/small_matrix_sw/test.jsonl', help="processed/.../test.jsonl or val.jsonl")
    ap.add_argument("--user_col", type=str, default="user_id")
    ap.add_argument("--item_col", type=str, default="video_id")
    ap.add_argument("--ts_col", type=str, default="timestamp")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--ks", type=str, default="20,50,100")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    kmax = max(ks)

    # 1) 读原始交互 + 排序（同 timestamp 用行号打破平局）
    df = pd.read_csv(args.raw_csv, usecols=[args.user_col, args.item_col, args.ts_col])
    df = df.dropna()
    df[args.ts_col] = pd.to_numeric(df[args.ts_col], errors="coerce")
    df = df.dropna(subset=[args.ts_col])
    df[args.ts_col] = df[args.ts_col].astype("int64")
    df["__row__"] = range(len(df))
    df = df.sort_values([args.user_col, args.ts_col, "__row__"], kind="mergesort")

    # 2) 按“每个用户自己的时间线”切分，只统计 train 段交互的热度
    cnt = Counter()
    for u, g in df.groupby(args.user_col, sort=False):
        items = g[args.item_col].astype(int).tolist()
        n = len(items)
        sp = time_split_indices(n, args.train_ratio, args.val_ratio)
        if sp is None:
            continue
        n_train, n_val = sp
        train_items = items[:n_train]
        cnt.update(train_items)

    popular = [it for it, _ in cnt.most_common()]
    if len(popular) == 0:
        raise RuntimeError("Popularity list is empty. Check raw_csv columns / split settings.")

    # 3) 在 eval_jsonl 上评估（过滤已看过 history）
    rec_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    n_users = 0

    for r in read_jsonl(args.eval_jsonl):
        seen = set(r["history"])
        target = r["target"]

        rank = []
        for it in popular:
            if it not in seen:
                rank.append(it)
            if len(rank) >= kmax:
                break

        n_users += 1
        for k in ks:
            rec_sum[k] += recall_at_k(rank, target, k)
            ndcg_sum[k] += ndcg_at_k(rank, target, k)

    print(f"users={n_users}  popular_size={len(popular)}")
    for k in ks:
        print(f"Recall@{k}: {rec_sum[k]/n_users:.6f}   NDCG@{k}: {ndcg_sum[k]/n_users:.6f}")

if __name__ == "__main__":
    main()