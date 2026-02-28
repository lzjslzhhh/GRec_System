import argparse
import json
import os
import sys
from collections import Counter

try:
    from baseline.eval_recall_metrics import evaluate_rows, save_metrics
except ModuleNotFoundError:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT_DIR = os.path.dirname(_THIS_DIR)
    if _ROOT_DIR not in sys.path:
        sys.path.insert(0, _ROOT_DIR)
    from baseline.eval_recall_metrics import evaluate_rows, save_metrics


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--eval_jsonl", type=str, required=True)
    ap.add_argument("--metrics_out", type=str, default="")
    ap.add_argument("--ks", type=str, default="20,50,100")
    ap.add_argument("--seen_filter", dest="seen_filter", action="store_true")
    ap.add_argument("--no_seen_filter", dest="seen_filter", action="store_false")
    ap.set_defaults(seen_filter=True)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    kmax = max(ks)

    cnt = Counter()
    for ex in read_jsonl(args.train_jsonl):
        cnt[int(ex["target"])] += 1

    popular = [sid for sid, _ in cnt.most_common()]
    vocab = set(popular)

    rows = []
    target_total = 0
    target_in_vocab = 0
    target_reachable = 0

    for ex in read_jsonl(args.eval_jsonl):
        history = [int(x) for x in ex.get("history", [])]
        target = int(ex["target"])
        seen = set(history) if args.seen_filter else set()

        target_total += 1
        if target in vocab:
            target_in_vocab += 1
        if target in vocab and (not args.seen_filter or target not in seen):
            target_reachable += 1

        rank = []
        for sid in popular:
            if args.seen_filter and sid in seen:
                continue
            rank.append(sid)
            if len(rank) >= kmax:
                break

        rows.append({"rank": rank, "target": target})

    metrics = evaluate_rows(rows, ks)
    metrics["token_pop_size"] = len(popular)
    metrics["target_total"] = target_total
    metrics["target_in_vocab"] = target_in_vocab
    metrics["target_in_vocab_ratio"] = (target_in_vocab / target_total) if target_total else 0.0
    metrics["target_reachable"] = target_reachable
    metrics["target_reachable_ratio"] = (target_reachable / target_total) if target_total else 0.0

    metrics_out = args.metrics_out or os.path.join(os.path.dirname(args.eval_jsonl), "token_pop_metrics.json")
    save_metrics(metrics_out, metrics)

    print(f"users={metrics['users']}  token_pop_size={len(popular)}")
    for k in ks:
        print(f"Recall@{k}: {metrics[f'Recall@{k}']:.6f}   NDCG@{k}: {metrics[f'NDCG@{k}']:.6f}")
    print(
        "coverage: "
        f"in_vocab={target_in_vocab}/{target_total} ({metrics['target_in_vocab_ratio']:.6f})  "
        f"reachable={target_reachable}/{target_total} ({metrics['target_reachable_ratio']:.6f})"
    )
    print(f"metrics_saved: {metrics_out}")


if __name__ == "__main__":
    main()
