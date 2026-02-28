import argparse
import os
import sys

import numpy as np

try:
    from baseline.eval_recall_metrics import evaluate_rows, save_metrics
    from baseline.item2vec.data import read_eval_jsonl
    from baseline.item2vec.recall import build_topk_neighbors, reachable_set, recall_from_neighbors
except ModuleNotFoundError:
    # Allow running this file directly in IDE debugger (run_path) without package context.
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT_DIR = os.path.dirname(_THIS_DIR)
    if _ROOT_DIR not in sys.path:
        sys.path.insert(0, _ROOT_DIR)
    from baseline.eval_recall_metrics import evaluate_rows, save_metrics
    from baseline.item2vec.data import read_eval_jsonl
    from baseline.item2vec.recall import build_topk_neighbors, reachable_set, recall_from_neighbors


def load_item_embeddings_from_npz(path: str):
    z = np.load(path)
    item_ids = z["item_ids"].astype(np.int64)
    emb = z["embeddings"].astype(np.float32)
    if item_ids.ndim != 1 or emb.ndim != 2 or emb.shape[0] != item_ids.shape[0]:
        raise ValueError("Invalid npz format. Expected item_ids[N] and embeddings[N, D].")
    return item_ids, emb


def load_item_embeddings_from_npy(item_ids_npy: str, item_emb_npy: str):
    item_ids = np.load(item_ids_npy).astype(np.int64)
    emb = np.load(item_emb_npy).astype(np.float32)
    if item_ids.ndim != 1 or emb.ndim != 2 or emb.shape[0] != item_ids.shape[0]:
        raise ValueError("Invalid npy format. Expected item_ids[N] and item_emb[N, D].")
    return item_ids, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npz", type=str, default="")
    ap.add_argument("--item_ids_npy", type=str, default="./artifacts/item2vec/item_ids.npy")
    ap.add_argument("--item_emb_npy", type=str, default="./artifacts/item2vec/item_emb.npy")
    ap.add_argument("--eval_jsonl", type=str, default="./data/processed/small_matrix_sw/test.jsonl")
    ap.add_argument("--metrics_out", type=str, default="", help="output json for metrics")

    ap.add_argument("--topk_sim", type=int, default=1000, help="neighbors per item")
    ap.add_argument("--ks", type=str, default="20,50,100")
    ap.add_argument("--recent_n", type=int, default=10)
    ap.add_argument("--pos_decay", type=float, default=0.8)
    ap.add_argument("--device", type=str, default="cuda", help="cpu / cuda / mps")

    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    max_k = max(ks)

    if args.emb_npz:
        item_ids, emb = load_item_embeddings_from_npz(args.emb_npz)
    elif os.path.exists(args.item_ids_npy) and os.path.exists(args.item_emb_npy):
        item_ids, emb = load_item_embeddings_from_npy(args.item_ids_npy, args.item_emb_npy)
    else:
        legacy_npz = "./data/processed/small_matrix_sw/item2vec/item_embeddings.npz"
        item_ids, emb = load_item_embeddings_from_npz(legacy_npz)
    item_set = set(int(x) for x in item_ids.tolist())
    neighbors = build_topk_neighbors(item_ids=item_ids, emb=emb, topk=args.topk_sim, device=args.device)

    eval_data = read_eval_jsonl(args.eval_jsonl)

    rows = []
    target_total = 0
    target_in_emb = 0
    target_reachable = 0

    for ex in eval_data:
        history = [int(x) for x in ex["history"]]
        target = int(ex["target"])

        target_total += 1
        if target in item_set:
            target_in_emb += 1
        if target in reachable_set(
            history=history,
            neighbors=neighbors,
            seen_filter=True,
            recent_n=args.recent_n,
        ):
            target_reachable += 1

        recs = recall_from_neighbors(
            history=history,
            neighbors=neighbors,
            max_k=max_k,
            seen_filter=True,
            recent_n=args.recent_n,
            pos_decay=args.pos_decay,
        )
        rows.append({"rank": recs, "target": target})

    metrics = evaluate_rows(rows, ks)
    metrics["item2vec_size"] = len(neighbors)
    metrics["target_total"] = target_total
    metrics["target_in_emb"] = target_in_emb
    metrics["target_in_emb_ratio"] = (target_in_emb / target_total) if target_total else 0.0
    metrics["target_reachable"] = target_reachable
    metrics["target_reachable_ratio"] = (target_reachable / target_total) if target_total else 0.0

    metrics_out = args.metrics_out or os.path.join(os.path.dirname(args.eval_jsonl), "item2vec_metrics.json")
    save_metrics(metrics_out, metrics)

    print(f"users={metrics['users']}  item2vec_size={len(neighbors)}")
    for k in ks:
        print(f"Recall@{k}: {metrics[f'Recall@{k}']:.6f}   NDCG@{k}: {metrics[f'NDCG@{k}']:.6f}")
    print(
        "coverage: "
        f"in_emb={target_in_emb}/{target_total} ({metrics['target_in_emb_ratio']:.6f})  "
        f"reachable={target_reachable}/{target_total} ({metrics['target_reachable_ratio']:.6f})"
    )
    print(f"metrics_saved: {metrics_out}")


if __name__ == "__main__":
    main()
