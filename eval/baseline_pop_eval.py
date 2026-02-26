import json
import math
import argparse
from collections import Counter

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def recall_at_k(rank, target, k):
    return 1.0 if target in rank[:k] else 0.0

def ndcg_at_k(rank, target, k):
    # 单个 target 的 DCG：命中位置 p -> 1/log2(p+2)
    if target in rank[:k]:
        p = rank.index(target)
        return 1.0 / math.log2(p + 2)
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--test", type=str, required=True)
    ap.add_argument("--ks", type=str, default="20,50,100")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    # 1) 统计 train 热度（用 target 统计就够了，也可以把 history 全算进去）
    cnt = Counter()
    for r in read_jsonl(args.train):
        cnt[r["target"]] += 1

    # 热门榜（从高到低）
    popular = [item for item, _ in cnt.most_common()]

    # 2) 测试：对每个用户推荐 topK（过滤已看过 history）
    rec_sum = {k: 0.0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    n = 0

    for r in read_jsonl(args.test):
        seen = set(r["history"])
        target = r["target"]

        # 过滤已看过的
        rank = []
        for it in popular:
            if it not in seen:
                rank.append(it)
            if len(rank) >= max(ks):
                break

        n += 1
        for k in ks:
            rec_sum[k] += recall_at_k(rank, target, k)
            ndcg_sum[k] += ndcg_at_k(rank, target, k)

    print(f"users={n}")
    for k in ks:
        print(f"Recall@{k}: {rec_sum[k]/n:.6f}   NDCG@{k}: {ndcg_sum[k]/n:.6f}")

if __name__ == "__main__":
    main()