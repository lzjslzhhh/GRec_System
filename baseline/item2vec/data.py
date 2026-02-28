import csv
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def read_eval_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def build_train_sequences(
    raw_csv: str,
    train_ratio: float,
    val_ratio: float,
    max_seq_len: int,
) -> Dict[int, List[int]]:
    seqs = read_raw_sequences(raw_csv)
    user_train = {}
    for u, seq in seqs.items():
        train, _, _ = split_by_ratio(seq, train_ratio, val_ratio)
        if max_seq_len > 0 and len(train) > max_seq_len:
            train = train[-max_seq_len:]
        user_train[u] = train
    return user_train


def build_vocab(user_train_seqs: Dict[int, List[int]], min_count: int) -> Tuple[Dict[int, int], List[int], List[int]]:
    cnt = Counter()
    for seq in user_train_seqs.values():
        cnt.update(seq)

    items = [it for it, c in cnt.items() if c >= min_count]
    items.sort()
    item_to_idx = {it: i for i, it in enumerate(items)}
    freqs = [cnt[it] for it in items]
    return item_to_idx, items, freqs


def sequences_to_indices(
    user_train_seqs: Dict[int, List[int]],
    item_to_idx: Dict[int, int],
) -> List[List[int]]:
    out = []
    for seq in user_train_seqs.values():
        idx_seq = [item_to_idx[it] for it in seq if it in item_to_idx]
        if len(idx_seq) >= 2:
            out.append(idx_seq)
    return out


def sequences_to_token_sequences(user_train_seqs: Dict[int, List[int]]) -> List[List[str]]:
    corpus = []
    for seq in user_train_seqs.values():
        if len(seq) >= 2:
            corpus.append([str(it) for it in seq])
    return corpus
