import os, json, argparse
import pandas as pd

def dump_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def time_split(seq, train_ratio=0.8, val_ratio=0.1):
    n = len(seq)
    if n < 3:
        return None
    n_train = max(int(n * train_ratio), 1)
    n_val = max(int(n * val_ratio), 1)
    if n - n_train - n_val <= 0:
        n_train = max(n - 2, 1)
        n_val = 1
        if n - n_train - n_val <= 0:
            return None
    return seq[:n_train], seq[n_train:n_train+n_val], seq[n_train+n_val:]

def pack_one(user_id, hist_seq, target_item, target_ts):
    # hist_seq: [(item, ts), ...]
    items = [i for i, _ in hist_seq]
    tss = [t for _, t in hist_seq]
    return {
        "user_id": int(user_id),
        "history": items,
        "history_ts": tss,
        "target": int(target_item),
        "target_ts": int(target_ts),
    }

def gen_sliding_window_samples(user_id, seq, max_seq_len=50, min_hist_len=1, stride=1, max_samples=None):
    """
    seq: [(item, ts), ...] 只在 train 段上做
    对每个位置 i 生成样本：history = seq[max(0, i-max_seq_len):i], target = seq[i]
    i 从 min_hist_len 到 len(seq)-1（保证 history 至少 min_hist_len）
    """
    n = len(seq)
    samples = []
    # target index i 的范围：至少有 min_hist_len 的 history
    for i in range(min_hist_len, n):
        hist = seq[max(0, i - max_seq_len): i]
        tgt_item, tgt_ts = seq[i]
        samples.append(pack_one(user_id, hist, tgt_item, tgt_ts))

    # stride：下采样（比如 stride=2 每隔一个取一个）
    if stride > 1:
        samples = samples[::stride]

    # 限制每个用户的样本数，避免超长序列用户占比过大
    if max_samples is not None and len(samples) > max_samples:
        # 取最近的 max_samples 条（更贴近“近期行为更重要”）
        samples = samples[-max_samples:]

    return samples

def pack_eval(user_id, full_seq):
    # 用 full_seq 的最后一个做 target，前面全部做 history（用于 val/test）
    if len(full_seq) < 2:
        return None
    hist = full_seq[:-1]
    tgt_item, tgt_ts = full_seq[-1]
    return pack_one(user_id, hist, tgt_item, tgt_ts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="KuaiRec 2.0/data")
    ap.add_argument("--matrix", type=str, default="small_matrix.csv")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--min_seq_len", type=int, default=5)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    # sliding window params
    ap.add_argument("--max_seq_len", type=int, default=50)
    ap.add_argument("--min_hist_len", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_train_samples_per_user", type=int, default=1000)

    args = ap.parse_args()

    path = os.path.join(args.data_dir, args.matrix)
    df = pd.read_csv(path)

    user_col, item_col, ts_col = "user_id", "video_id", "timestamp"

    df = df[[user_col, item_col, ts_col]].dropna()
    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df[ts_col] = df[ts_col].astype("int64")

    # 稳定排序：同 timestamp 用行号打破平局
    df["__row__"] = range(len(df))
    df = df.sort_values([user_col, ts_col, "__row__"], kind="mergesort")

    # user -> [(item, ts), ...]
    seqs = df.groupby(user_col, sort=False).apply(
        lambda g: list(zip(g[item_col].astype(int).tolist(), g[ts_col].tolist()))
    )

    train_rows, val_rows, test_rows = [], [], []
    stats = {
        "users_total": int(seqs.shape[0]),
        "users_kept": 0,
        "dropped_too_short": 0,
        "train_samples": 0,
        "avg_train_samples_per_user": 0.0,
    }

    per_user_train_counts = []

    for u, seq in seqs.items():
        if len(seq) < args.min_seq_len:
            stats["dropped_too_short"] += 1
            continue
        sp = time_split(seq, args.train_ratio, args.val_ratio)
        if sp is None:
            stats["dropped_too_short"] += 1
            continue
        tr, va, te = sp
        stats["users_kept"] += 1

        # 1) train：滑窗多样本（只用 tr 段，避免泄漏）
        tr_samples = gen_sliding_window_samples(
            u, tr,
            max_seq_len=args.max_seq_len,
            min_hist_len=args.min_hist_len,
            stride=args.stride,
            max_samples=args.max_train_samples_per_user
        )
        train_rows.extend(tr_samples)
        per_user_train_counts.append(len(tr_samples))

        # 2) val/test：各 1 条样本（更稳定、方便对比）
        v = pack_eval(u, tr + va)
        t = pack_eval(u, tr + va + te)
        if v: val_rows.append(v)
        if t: test_rows.append(t)

    stats["train_samples"] = len(train_rows)
    stats["avg_train_samples_per_user"] = float(sum(per_user_train_counts) / max(len(per_user_train_counts), 1))

    out_base = os.path.join(args.out_dir, os.path.splitext(args.matrix)[0] + "_sw")
    dump_jsonl(os.path.join(out_base, "train.jsonl"), train_rows)
    dump_jsonl(os.path.join(out_base, "val.jsonl"), val_rows)
    dump_jsonl(os.path.join(out_base, "test.jsonl"), test_rows)

    os.makedirs(out_base, exist_ok=True)
    with open(os.path.join(out_base, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Output:", out_base)

if __name__ == "__main__":
    main()