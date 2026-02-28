import argparse
import json
import os
from collections import Counter

import pandas as pd

from .eval_recall_metrics import evaluate_rows, save_metrics


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=str, default="./KuaiRec 2.0/data/small_matrix.csv", help="small_matrix.csv or big_matrix.csv")
    ap.add_argument("--eval_jsonl", type=str, default="./data/processed/small_matrix_sw/test.jsonl", help="processed/.../test.jsonl or val.jsonl")
    ap.add_argument("--metrics_out", type=str, default="", help="output json for metrics")
    ap.add_argument("--user_col", type=str, default="user_id")
    ap.add_argument("--item_col", type=str, default="video_id")
    ap.add_argument("--ts_col", type=str, default="timestamp")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--ks", type=str, default="20,50,100")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    kmax = max(ks)

    df = pd.read_csv(args.raw_csv, usecols=[args.user_col, args.item_col, args.ts_col])
    df = df.dropna()
    df[args.ts_col] = pd.to_numeric(df[args.ts_col], errors="coerce")
    df = df.dropna(subset=[args.ts_col])
    df[args.ts_col] = df[args.ts_col].astype("int64")
    df["__row__"] = range(len(df))
    df = df.sort_values([args.user_col, args.ts_col, "__row__"], kind="mergesort")

    cnt = Counter()
    for _, g in df.groupby(args.user_col, sort=False):
        items = g[args.item_col].astype(int).tolist()
        sp = time_split_indices(len(items), args.train_ratio, args.val_ratio)
        if sp is None:
            continue
        n_train, _ = sp
        cnt.update(items[:n_train])

    popular = [it for it, _ in cnt.most_common()]
    if len(popular) == 0:
        raise RuntimeError("Popularity list is empty. Check raw_csv columns / split settings.")

    rows = []
    for r in read_jsonl(args.eval_jsonl):
        seen = set(r["history"])
        target = int(r["target"])

        rank = []
        for it in popular:
            if it not in seen:
                rank.append(it)
            if len(rank) >= kmax:
                break

        rows.append({"rank": rank, "target": target})

    metrics = evaluate_rows(rows, ks)
    metrics["popular_size"] = len(popular)

    metrics_out = args.metrics_out or os.path.join(os.path.dirname(args.eval_jsonl), "pop_metrics.json")
    save_metrics(metrics_out, metrics)

    print(f"users={metrics['users']}  popular_size={len(popular)}")
    for k in ks:
        print(f"Recall@{k}: {metrics[f'Recall@{k}']:.6f}   NDCG@{k}: {metrics[f'NDCG@{k}']:.6f}")
    print(f"metrics_saved: {metrics_out}")


if __name__ == "__main__":
    main()
