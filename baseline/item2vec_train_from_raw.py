import argparse
import json
import os
import sys

import numpy as np

try:
    from baseline.item2vec.data import build_train_sequences, sequences_to_token_sequences
    from baseline.item2vec.train import train_item2vec_gensim
except ModuleNotFoundError:
    # Allow running this file directly in IDE debugger (run_path) without package context.
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT_DIR = os.path.dirname(_THIS_DIR)
    if _ROOT_DIR not in sys.path:
        sys.path.insert(0, _ROOT_DIR)
    from baseline.item2vec.data import build_train_sequences, sequences_to_token_sequences
    from baseline.item2vec.train import train_item2vec_gensim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=str, default="./KuaiRec 2.0/data/small_matrix.csv")
    ap.add_argument("--out_dir", type=str, default="./artifacts/item2vec")

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--max_seq_len", type=int, default=200, help="cap per-user train sequence length")

    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--window_size", type=int, default=10)
    ap.add_argument("--negative", type=int, default=10)
    ap.add_argument("--sample", type=float, default=1e-5, help="subsampling threshold for frequent items")
    ap.add_argument("--ns_exponent", type=float, default=0.75, help="negative sampling distribution exponent")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--min_count", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sg", type=int, default=1, help="1=skip-gram, 0=cbow")

    args = ap.parse_args()

    user_train = build_train_sequences(
        raw_csv=args.raw_csv,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_seq_len=args.max_seq_len,
    )
    corpus = sequences_to_token_sequences(user_train)

    print(f"users_with_train_seq={len(corpus)}  sequences={len(corpus)}")

    item_ids, emb, stats = train_item2vec_gensim(
        token_sequences=corpus,
        vector_size=args.dim,
        window_size=args.window_size,
        negative=args.negative,
        epochs=args.epochs,
        min_count=args.min_count,
        workers=args.workers,
        seed=args.seed,
        sample=args.sample,
        ns_exponent=args.ns_exponent,
        sg=args.sg,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Standardized exports for semantic-id pipeline.
    item_ids_npy = os.path.join(args.out_dir, "item_ids.npy")
    item_emb_npy = os.path.join(args.out_dir, "item_emb.npy")
    np.save(item_ids_npy, item_ids.astype(np.int64))
    np.save(item_emb_npy, emb.astype(np.float32))

    # Backward-compatible export for existing eval script users.
    emb_npz = os.path.join(args.out_dir, "item_embeddings.npz")
    np.savez_compressed(emb_npz, item_ids=item_ids.astype(np.int64), embeddings=emb.astype(np.float32))

    config = {
        "raw_csv": args.raw_csv,
        "vocab_size": int(stats["vocab_size"]),
        "corpus_count": int(stats["corpus_count"]),
        "corpus_total_words": int(stats["corpus_total_words"]),
        "dim": int(args.dim),
        "window_size": int(args.window_size),
        "negative": int(args.negative),
        "sample": float(args.sample),
        "ns_exponent": float(args.ns_exponent),
        "epochs": int(args.epochs),
        "min_count": int(args.min_count),
        "workers": int(args.workers),
        "sg": int(args.sg),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "max_seq_len": int(args.max_seq_len),
        "users_with_train_seq": int(len(corpus)),
    }
    config_out = os.path.join(args.out_dir, "config.json")
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Keep legacy meta for compatibility.
    legacy_meta_out = os.path.join(args.out_dir, "train_meta.json")
    with open(legacy_meta_out, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"item_ids_saved: {item_ids_npy}")
    print(f"item_emb_saved: {item_emb_npy}")
    print(f"config_saved: {config_out}")
    print(f"legacy_npz_saved: {emb_npz}")


if __name__ == "__main__":
    main()
