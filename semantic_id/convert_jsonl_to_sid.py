import argparse
import json
import os
from typing import Dict, List, Tuple


def load_item2sid(path: str) -> Dict[int, int]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def map_items(
    items: List[int],
    item2sid: Dict[int, int],
    oov_strategy: str,
    unk_sid: int,
) -> Tuple[List[int], int, bool]:
    out = []
    unk_count = 0
    has_oov = False
    for it in items:
        if it in item2sid:
            out.append(item2sid[it])
        else:
            has_oov = True
            if oov_strategy == "drop":
                return [], 0, True
            out.append(unk_sid)
            unk_count += 1
    return out, unk_count, has_oov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True)
    ap.add_argument("--item2sid", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--keep_item_fields", type=str, default="true", help="true/false")
    ap.add_argument("--oov_strategy", type=str, choices=["drop", "map_to_unk"], default="drop")
    args = ap.parse_args()

    keep_item_fields = args.keep_item_fields.lower() in ("1", "true", "yes", "y")
    item2sid = load_item2sid(args.item2sid)
    sid_vocab_size = (max(item2sid.values()) + 1) if item2sid else 0
    unk_sid = sid_vocab_size

    total_lines = 0
    kept_lines = 0
    dropped_oov = 0
    unk_count = 0

    unique_items = set()
    unique_items_mapped = set()
    unique_sids = set()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.in_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            ex = json.loads(line)

            history_item = [int(x) for x in ex.get("history", [])]
            target_item = int(ex["target"])

            unique_items.update(history_item)
            unique_items.add(target_item)

            history_sid, h_unk, h_oov = map_items(history_item, item2sid, args.oov_strategy, unk_sid)
            target_sid_list, t_unk, t_oov = map_items([target_item], item2sid, args.oov_strategy, unk_sid)
            if args.oov_strategy == "drop" and (h_oov or t_oov):
                dropped_oov += 1
                continue

            target_sid = target_sid_list[0]
            unk_count += h_unk + t_unk
            kept_lines += 1

            mapped_items = [it for it in history_item if it in item2sid]
            if target_item in item2sid:
                mapped_items.append(target_item)
            unique_items_mapped.update(mapped_items)

            unique_sids.update(history_sid)
            unique_sids.add(target_sid)

            out = dict(ex)
            out["history"] = history_sid
            out["target"] = target_sid
            if keep_item_fields:
                out["history_item"] = history_item
                out["target_item"] = target_item

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    sid_vocab_out = sid_vocab_size + (1 if args.oov_strategy == "map_to_unk" else 0)
    stats = {
        "in_jsonl": args.in_jsonl,
        "item2sid": args.item2sid,
        "out_jsonl": args.out_jsonl,
        "oov_strategy": args.oov_strategy,
        "keep_item_fields": keep_item_fields,
        "total_lines": total_lines,
        "kept_lines": kept_lines,
        "dropped_oov": dropped_oov,
        "unk_count": unk_count,
        "sid_vocab_size": int(sid_vocab_out),
        "sid_vocab_base_k": int(sid_vocab_size),
        "unk_sid": int(unk_sid) if args.oov_strategy == "map_to_unk" else None,
        "unique_items_in_input": int(len(unique_items)),
        "unique_items_mapped": int(len(unique_items_mapped)),
        "item_coverage_ratio_unique": float(len(unique_items_mapped) / len(unique_items)) if unique_items else 0.0,
        "unique_sid_in_output": int(len(unique_sids)),
        "line_keep_ratio": float(kept_lines / total_lines) if total_lines else 0.0,
    }

    stats_path = args.out_jsonl + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"total_lines={total_lines}  kept_lines={kept_lines}  "
        f"dropped_oov={dropped_oov}  unk_count={unk_count}"
    )
    print(
        "coverage: "
        f"unique_items_mapped={stats['unique_items_mapped']}/{stats['unique_items_in_input']} "
        f"({stats['item_coverage_ratio_unique']:.6f})"
    )
    print(f"stats_saved: {stats_path}")


if __name__ == "__main__":
    main()
