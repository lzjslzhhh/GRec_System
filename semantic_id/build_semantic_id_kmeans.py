import argparse
import json
import os
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def cluster_stats(labels: np.ndarray, k: int) -> Dict[str, object]:
    sizes = np.bincount(labels, minlength=k)
    order = np.argsort(-sizes)
    top10: List[Dict[str, int]] = []
    for idx in order[:10]:
        top10.append({"cluster": int(idx), "size": int(sizes[idx])})

    return {
        "num_clusters": int(k),
        "cluster_size_min": int(sizes.min()) if sizes.size else 0,
        "cluster_size_mean": float(sizes.mean()) if sizes.size else 0.0,
        "cluster_size_median": float(np.median(sizes)) if sizes.size else 0.0,
        "cluster_size_p95": float(np.percentile(sizes, 95)) if sizes.size else 0.0,
        "cluster_size_max": int(sizes.max()) if sizes.size else 0,
        "num_empty_clusters": int((sizes == 0).sum()) if sizes.size else 0,
        "top10_largest_clusters": top10,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item_ids_npy", type=str, default="../artifacts/item2vec/item_ids.npy")
    ap.add_argument("--item_emb_npy", type=str, default="../artifacts/item2vec/item_emb.npy")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--disable_l2_norm", action="store_true", help="disable l2 normalization before k-means")
    args = ap.parse_args()

    out_dir = args.out_dir or f"../artifacts/semantic_id/kmeans_k{args.k}"
    os.makedirs(out_dir, exist_ok=True)

    item_ids = np.load(args.item_ids_npy).astype(np.int64)
    emb = np.load(args.item_emb_npy).astype(np.float32)
    if item_ids.ndim != 1 or emb.ndim != 2 or emb.shape[0] != item_ids.shape[0]:
        raise ValueError("Invalid input format. Need item_ids[N] and item_emb[N, D].")

    x = emb
    if not args.disable_l2_norm:
        x = l2_normalize(x)

    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto")
    labels = km.fit_predict(x)

    mapping = {str(int(it)): int(sid) for it, sid in zip(item_ids.tolist(), labels.tolist())}
    with open(os.path.join(out_dir, "item2sid.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(out_dir, "centers.npy"), km.cluster_centers_.astype(np.float32))

    cstats = cluster_stats(labels=labels, k=args.k)
    with open(os.path.join(out_dir, "cluster_stats.json"), "w", encoding="utf-8") as f:
        json.dump(cstats, f, ensure_ascii=False, indent=2)

    coverage = {
        "N_items": int(item_ids.shape[0]),
        "K": int(args.k),
        "D": int(emb.shape[1]),
        "l2_normalized": bool(not args.disable_l2_norm),
    }
    with open(os.path.join(out_dir, "coverage.json"), "w", encoding="utf-8") as f:
        json.dump(coverage, f, ensure_ascii=False, indent=2)

    print(f"items={coverage['N_items']}  dim={coverage['D']}  K={args.k}")
    print(
        "cluster_size: "
        f"min={cstats['cluster_size_min']}  mean={cstats['cluster_size_mean']:.3f}  "
        f"median={cstats['cluster_size_median']:.3f}  p95={cstats['cluster_size_p95']:.3f}  "
        f"max={cstats['cluster_size_max']}  empty={cstats['num_empty_clusters']}"
    )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
