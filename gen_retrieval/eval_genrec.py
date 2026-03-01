import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from baseline.eval_recall_metrics import evaluate_rows, save_metrics
except ModuleNotFoundError:
    import sys

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT_DIR = os.path.dirname(_THIS_DIR)
    if _ROOT_DIR not in sys.path:
        sys.path.insert(0, _ROOT_DIR)
    from baseline.eval_recall_metrics import evaluate_rows, save_metrics

from gen_retrieval.expand import (
    build_raw_train_interactions,
    build_sid2items,
    cluster_pop_order,
    expand_by_cluster_embed,
    expand_by_cluster_pop,
    load_item2sid,
    load_item2vec_embeddings,
)
from gen_retrieval.model import GenRecDecoderLM, get_last_logits


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_ks(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


@dataclass
class EvalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_token: torch.Tensor
    target_in_vocab: torch.Tensor
    history_raw: List[List[int]]
    target_raw: List[int]


class TokenEvalDataset(Dataset):
    def __init__(self, mapped_rows: List[Dict[str, object]], max_len: int):
        self.rows = mapped_rows
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = self.rows[idx]
        history = list(ex["history_token"])  # type: ignore[arg-type]
        if len(history) == 0:
            history = [int(ex["target_token"])]
        if len(history) > self.max_len:
            history = history[-self.max_len :]
        return {
            "history_token": history,
            "target_token": int(ex["target_token"]),
            "target_in_vocab": int(ex["target_in_vocab"]),
            "history_raw": list(ex["history_raw"]),  # type: ignore[arg-type]
            "target_raw": int(ex["target_raw"]),
        }


def build_collate(pad_id: int):
    def _fn(batch: List[Dict[str, object]]) -> EvalBatch:
        bsz = len(batch)
        max_len = max(len(x["history_token"]) for x in batch)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((bsz, max_len), dtype=torch.bool)
        target_token = torch.zeros((bsz,), dtype=torch.long)
        target_in_vocab = torch.zeros((bsz,), dtype=torch.long)
        history_raw: List[List[int]] = []
        target_raw: List[int] = []
        for i, ex in enumerate(batch):
            hist: List[int] = ex["history_token"]  # type: ignore[assignment]
            L = len(hist)
            input_ids[i, :L] = torch.tensor(hist, dtype=torch.long)
            attn[i, :L] = True
            target_token[i] = int(ex["target_token"])
            target_in_vocab[i] = int(ex["target_in_vocab"])
            history_raw.append(list(ex["history_raw"]))  # type: ignore[arg-type]
            target_raw.append(int(ex["target_raw"]))
        return EvalBatch(
            input_ids=input_ids,
            attention_mask=attn,
            target_token=target_token,
            target_in_vocab=target_in_vocab,
            history_raw=history_raw,
            target_raw=target_raw,
        )

    return _fn


def write_results_row(path: str, row: Dict[str, object]) -> None:
    fields = [
        "mode",
        "run_dir",
        "eval_split",
        "vocab_size",
        "model_vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "token_Recall@20",
        "token_NDCG@20",
        "token_Recall@50",
        "token_NDCG@50",
        "token_Recall@100",
        "token_NDCG@100",
        "item_Recall@20",
        "item_NDCG@20",
        "item_Recall@50",
        "item_NDCG@50",
        "item_Recall@100",
        "item_NDCG@100",
        "target_total",
        "target_in_vocab_ratio",
        "target_reachable_ratio",
        "notes",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def maybe_map_item_rows(eval_rows: List[dict], cfg: Dict[str, object], run_dir: str) -> List[Dict[str, object]]:
    item2tok_path = os.path.join(run_dir, "item2tok.json")
    if not os.path.exists(item2tok_path):
        raise FileNotFoundError(f"item2tok.json not found in run_dir: {item2tok_path}")
    with open(item2tok_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    item2tok = {int(k): int(v) for k, v in raw.items()}
    oov_strategy = str(cfg.get("oov_strategy", "map_to_unk"))
    unk_id = cfg.get("unk_id", None)
    unk_id = int(unk_id) if unk_id is not None else None
    known_item_size = int(cfg.get("known_item_size", len(item2tok)))

    mapped = []
    for ex in eval_rows:
        hist_item = [int(x) for x in ex.get("history", [])]
        target_item = int(ex["target"])
        hist_tok = []
        drop = False
        for item in hist_item:
            if item in item2tok:
                hist_tok.append(int(item2tok[item]))
            elif oov_strategy == "map_to_unk" and unk_id is not None:
                hist_tok.append(int(unk_id))
            else:
                drop = True
                break
        if drop:
            continue
        if target_item in item2tok:
            target_tok = int(item2tok[target_item])
            target_in_vocab = 1
        elif oov_strategy == "map_to_unk" and unk_id is not None:
            target_tok = int(unk_id)
            target_in_vocab = 0
        else:
            continue
        if len(hist_tok) == 0:
            hist_tok = [target_tok]
        mapped.append(
            {
                "history_token": hist_tok,
                "target_token": int(target_tok),
                "target_in_vocab": int(target_in_vocab),
                "history_raw": hist_item,
                "target_raw": target_item,
            }
        )
    return mapped


def maybe_map_sid_rows(eval_rows: List[dict], cfg: Dict[str, object]) -> List[Dict[str, object]]:
    token_vocab_size = int(cfg["token_vocab_size"])
    mapped = []
    for ex in eval_rows:
        hist_sid = [int(x) for x in ex.get("history", [])]
        target_sid = int(ex["target"])
        if len(hist_sid) == 0:
            hist_sid = [target_sid]
        history_raw = [int(x) for x in ex.get("history_item", [])]
        target_raw = int(ex.get("target_item", ex["target"]))
        target_in_vocab = 1 if 0 <= target_sid < token_vocab_size else 0
        mapped.append(
            {
                "history_token": hist_sid,
                "target_token": target_sid,
                "target_in_vocab": target_in_vocab,
                "history_raw": history_raw,
                "target_raw": target_raw,
            }
        )
    return mapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--eval_jsonl", type=str, required=True)
    ap.add_argument("--mode", type=str, default="", choices=["", "sid", "item"])
    ap.add_argument("--ckpt_path", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ks", type=str, default="20,50,100")
    ap.add_argument("--topk_tokens", type=int, default=100)
    ap.add_argument("--topk_items", type=int, default=100)
    ap.add_argument("--expand_strategy", type=str, default="expand_embed", choices=["expand_pop", "expand_embed"])
    ap.add_argument("--decode_strategy", type=str, default="next_token", choices=["next_token", "beam"])
    ap.add_argument("--beam_size", type=int, default=4)
    ap.add_argument("--beam_steps", type=int, default=2)
    ap.add_argument("--raw_csv", type=str, default="", help="required for sid item-wise expansion")
    ap.add_argument("--item2sid", type=str, default="", help="required for sid item-wise expansion")
    ap.add_argument("--item_ids_npy", type=str, default="artifacts/item2vec/item_ids.npy")
    ap.add_argument("--item_emb_npy", type=str, default="artifacts/item2vec/item_emb.npy")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--metrics_token_out", type=str, default="")
    ap.add_argument("--metrics_item_out", type=str, default="")
    ap.add_argument("--sample_cases_out", type=str, default="")
    ap.add_argument("--results_csv", type=str, default="experiments/results.csv")
    ap.add_argument("--results_tag", type=str, default="")
    args = ap.parse_args()

    cfg_path = os.path.join(args.run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    mode = args.mode or str(cfg.get("mode", "sid"))
    ks = parse_ks(args.ks)
    token_vocab_size = int(cfg["token_vocab_size"])
    model_vocab_size = int(cfg.get("model_vocab_size", token_vocab_size + 1))
    pad_id = int(cfg["pad_id"])

    eval_rows = load_jsonl(args.eval_jsonl)
    if mode == "item":
        mapped_rows = maybe_map_item_rows(eval_rows, cfg=cfg, run_dir=args.run_dir)
    else:
        mapped_rows = maybe_map_sid_rows(eval_rows, cfg=cfg)
    if len(mapped_rows) == 0:
        raise ValueError("No eval rows after mapping/OOV handling.")

    dataset = TokenEvalDataset(mapped_rows=mapped_rows, max_len=int(cfg["max_len"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=build_collate(pad_id=pad_id),
    )

    device = torch.device(args.device)
    model = GenRecDecoderLM(
        vocab_size=model_vocab_size,
        pad_id=pad_id,
        max_len=int(cfg["max_len"]),
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        ff_mult=int(cfg.get("ff_mult", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)

    ckpt_path = args.ckpt_path or os.path.join(args.run_dir, "checkpoint_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, "checkpoint_last.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found in run_dir: {args.run_dir}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if args.decode_strategy == "beam":
        print(
            f"[warn] decode_strategy=beam requested (beam_size={args.beam_size}, beam_steps={args.beam_steps}), "
            "v0 currently falls back to next-token logits for ranking.",
            flush=True,
        )

    kmax = max(max(ks), args.topk_tokens)
    token_rows = []
    token_ranks: List[List[int]] = []
    raw_histories: List[List[int]] = []
    raw_targets: List[int] = []
    sample_cases: List[dict] = []
    target_total = 0
    target_in_vocab = 0
    target_reachable = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch.input_ids.to(device)
            attn = batch.attention_mask.to(device)
            logits = model(input_ids, attn)
            last_logits = get_last_logits(logits, attn)
            last_logits[:, pad_id] = -1e9
            rank = torch.topk(last_logits, k=min(kmax, last_logits.size(-1)), dim=-1).indices.cpu().tolist()
            ys = batch.target_token.tolist()
            iv = batch.target_in_vocab.tolist()
            for r, y, in_vocab, h_raw, t_raw in zip(rank, ys, iv, batch.history_raw, batch.target_raw):
                rank_used = [int(x) for x in r[: args.topk_tokens] if int(x) < token_vocab_size]
                token_rows.append({"rank": rank_used, "target": int(y)})
                token_ranks.append(rank_used)
                raw_histories.append(h_raw)
                raw_targets.append(int(t_raw))
                target_total += 1
                target_in_vocab += int(in_vocab)
                if int(y) in rank_used:
                    target_reachable += 1
                if len(sample_cases) < 10:
                    sample_cases.append(
                        {
                            "history": h_raw,
                            "target": int(t_raw),
                            "target_token": int(y),
                            "top10_token": rank_used[:10],
                        }
                    )

    token_metrics = evaluate_rows(token_rows, ks=ks)
    token_metrics["target_total"] = int(target_total)
    token_metrics["target_in_vocab"] = int(target_in_vocab)
    token_metrics["target_in_vocab_ratio"] = float(target_in_vocab / target_total) if target_total else 0.0
    token_metrics["target_reachable"] = int(target_reachable)
    token_metrics["target_reachable_ratio"] = float(target_reachable / target_total) if target_total else 0.0
    token_path = args.metrics_token_out or os.path.join(args.run_dir, "metrics_token.json")
    save_metrics(token_path, token_metrics)

    item_metrics: Optional[Dict[str, object]] = None
    if mode == "sid" and args.raw_csv and args.item2sid:
        item2sid = load_item2sid(args.item2sid)
        sid2items = build_sid2items(item2sid)
        _, train_item_pop = build_raw_train_interactions(
            raw_csv=args.raw_csv,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        train_items = set(train_item_pop.keys())
        sid2items_pop = cluster_pop_order(sid2items=sid2items, train_item_pop=train_item_pop)
        item_vecs: Dict[int, object] = {}
        if args.expand_strategy == "expand_embed":
            item_vecs = load_item2vec_embeddings(args.item_ids_npy, args.item_emb_npy)

        item_rows = []
        item_reachable = 0
        target_in_train = 0
        target_in_item2sid = 0
        for sid_rank, hist_items, target_item in zip(token_ranks, raw_histories, raw_targets):
            seen = set(int(x) for x in hist_items)
            if int(target_item) in train_items:
                target_in_train += 1
            if int(target_item) in item2sid:
                target_in_item2sid += 1
            if args.expand_strategy == "expand_embed":
                last_item = int(hist_items[-1]) if len(hist_items) > 0 else -1
                item_rank = expand_by_cluster_embed(
                    token_rank=sid_rank,
                    sid2items_pop=sid2items_pop,
                    item_vecs=item_vecs,
                    last_item=last_item,
                    seen_items=seen,
                    topk_items=args.topk_items,
                )
            else:
                item_rank = expand_by_cluster_pop(
                    token_rank=sid_rank,
                    sid2items_pop=sid2items_pop,
                    seen_items=seen,
                    topk_items=args.topk_items,
                )
            item_rows.append({"rank": item_rank, "target": int(target_item)})
            if int(target_item) in item_rank:
                item_reachable += 1

        item_metrics = evaluate_rows(item_rows, ks=ks)
        item_metrics["mode"] = f"sid_item_{args.expand_strategy}"
        item_metrics["expand_strategy"] = args.expand_strategy
        item_metrics["train_item_size"] = int(len(train_items))
        item_metrics["test_target_total"] = int(len(raw_targets))
        item_metrics["test_target_in_train"] = int(target_in_train)
        item_metrics["test_target_in_train_ratio"] = float(target_in_train / len(raw_targets)) if raw_targets else 0.0
        item_metrics["test_target_in_item2sid"] = int(target_in_item2sid)
        item_metrics["test_target_in_item2sid_ratio"] = (
            float(target_in_item2sid / len(raw_targets)) if raw_targets else 0.0
        )
        item_metrics["item_reachable"] = int(item_reachable)
        item_metrics["item_reachable_ratio"] = float(item_reachable / len(raw_targets)) if raw_targets else 0.0
        item_metrics["token_reachable_ratio"] = token_metrics["target_reachable_ratio"]
        item_path = args.metrics_item_out or os.path.join(args.run_dir, "metrics_item.json")
        save_metrics(item_path, item_metrics)
    elif mode == "item":
        # For item mode, token and item are equivalent.
        item_metrics = {
            "Recall@20": token_metrics.get("Recall@20", 0.0),
            "NDCG@20": token_metrics.get("NDCG@20", 0.0),
            "Recall@50": token_metrics.get("Recall@50", 0.0),
            "NDCG@50": token_metrics.get("NDCG@50", 0.0),
            "Recall@100": token_metrics.get("Recall@100", 0.0),
            "NDCG@100": token_metrics.get("NDCG@100", 0.0),
            "item_reachable_ratio": token_metrics.get("target_reachable_ratio", 0.0),
            "test_target_total": token_metrics.get("target_total", 0),
        }
        item_path = args.metrics_item_out or os.path.join(args.run_dir, "metrics_item.json")
        save_metrics(item_path, item_metrics)

    sample_path = args.sample_cases_out or os.path.join(args.run_dir, "sample_cases.json")
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_cases, f, ensure_ascii=False, indent=2)

    result_row = {
        "mode": mode,
        "run_dir": args.run_dir,
        "eval_split": os.path.basename(args.eval_jsonl),
        "vocab_size": token_vocab_size,
        "model_vocab_size": model_vocab_size,
        "d_model": int(cfg["d_model"]),
        "n_layers": int(cfg["n_layers"]),
        "n_heads": int(cfg["n_heads"]),
        "token_Recall@20": token_metrics.get("Recall@20", 0.0),
        "token_NDCG@20": token_metrics.get("NDCG@20", 0.0),
        "token_Recall@50": token_metrics.get("Recall@50", 0.0),
        "token_NDCG@50": token_metrics.get("NDCG@50", 0.0),
        "token_Recall@100": token_metrics.get("Recall@100", 0.0),
        "token_NDCG@100": token_metrics.get("NDCG@100", 0.0),
        "item_Recall@20": (item_metrics or {}).get("Recall@20", 0.0),
        "item_NDCG@20": (item_metrics or {}).get("NDCG@20", 0.0),
        "item_Recall@50": (item_metrics or {}).get("Recall@50", 0.0),
        "item_NDCG@50": (item_metrics or {}).get("NDCG@50", 0.0),
        "item_Recall@100": (item_metrics or {}).get("Recall@100", 0.0),
        "item_NDCG@100": (item_metrics or {}).get("NDCG@100", 0.0),
        "target_total": token_metrics.get("target_total", 0),
        "target_in_vocab_ratio": token_metrics.get("target_in_vocab_ratio", 0.0),
        "target_reachable_ratio": token_metrics.get("target_reachable_ratio", 0.0),
        "notes": args.results_tag,
    }
    if args.results_csv:
        write_results_row(args.results_csv, result_row)

    print(f"users={token_metrics['users']}")
    for k in ks:
        print(f"token Recall@{k}: {token_metrics[f'Recall@{k}']:.6f}   token NDCG@{k}: {token_metrics[f'NDCG@{k}']:.6f}")
    if item_metrics is not None:
        for k in ks:
            print(
                f"item  Recall@{k}: {float(item_metrics.get(f'Recall@{k}', 0.0)):.6f}   "
                f"item  NDCG@{k}: {float(item_metrics.get(f'NDCG@{k}', 0.0)):.6f}"
            )
    print(
        "coverage: "
        f"in_vocab={token_metrics['target_in_vocab']}/{token_metrics['target_total']} "
        f"({token_metrics['target_in_vocab_ratio']:.6f})  "
        f"reachable={token_metrics['target_reachable']}/{token_metrics['target_total']} "
        f"({token_metrics['target_reachable_ratio']:.6f})"
    )
    print(f"metrics_saved: token={token_path} item={(args.metrics_item_out or os.path.join(args.run_dir, 'metrics_item.json'))}")
    print(f"sample_saved: {sample_path}")
    if args.results_csv:
        print(f"results_saved: {args.results_csv}")


if __name__ == "__main__":
    main()
