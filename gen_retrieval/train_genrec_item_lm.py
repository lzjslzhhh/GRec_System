import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
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

from gen_retrieval.model import GenRecDecoderLM, get_last_logits


IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_ks(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def append_jsonl(path: str, payload: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_item_vocab(train_rows: List[dict], map_to_unk: bool) -> Tuple[Dict[int, int], int, int, Optional[int]]:
    item_set = set()
    for ex in train_rows:
        hist = [int(x) for x in ex.get("history", [])]
        target = int(ex["target"])
        item_set.update(hist)
        item_set.add(target)
    items = sorted(item_set)
    item2tok = {int(item): int(i) for i, item in enumerate(items)}
    known_size = len(items)
    unk_id: Optional[int] = None
    token_vocab_size = known_size
    if map_to_unk:
        unk_id = token_vocab_size
        token_vocab_size += 1
    pad_id = token_vocab_size
    return item2tok, token_vocab_size, pad_id, unk_id


def map_item(item: int, item2tok: Dict[int, int], oov_strategy: str, unk_id: Optional[int]) -> Optional[int]:
    if item in item2tok:
        return int(item2tok[item])
    if oov_strategy == "drop":
        return None
    if unk_id is None:
        raise ValueError("unk_id is required when oov_strategy=map_to_unk")
    return int(unk_id)


def map_row(
    ex: dict,
    item2tok: Dict[int, int],
    oov_strategy: str,
    unk_id: Optional[int],
    max_len: int,
) -> Optional[Dict[str, object]]:
    hist_raw = [int(x) for x in ex.get("history", [])]
    target_raw = int(ex["target"])
    hist_tok = []
    for item in hist_raw:
        token = map_item(item, item2tok=item2tok, oov_strategy=oov_strategy, unk_id=unk_id)
        if token is None:
            return None
        hist_tok.append(token)
    target_tok = map_item(target_raw, item2tok=item2tok, oov_strategy=oov_strategy, unk_id=unk_id)
    if target_tok is None:
        return None
    if len(hist_tok) == 0:
        hist_tok = [target_tok]
    if len(hist_tok) > max_len:
        hist_tok = hist_tok[-max_len:]
    return {
        "history_tok": hist_tok,
        "target_tok": int(target_tok),
        "history_item": hist_raw,
        "target_item": target_raw,
    }


@dataclass
class TrainBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass
class EvalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_tok: torch.Tensor
    target_in_vocab: torch.Tensor
    history_item: List[List[int]]
    target_item: List[int]


class ItemLmTrainDataset(Dataset):
    def __init__(self, mapped_rows: List[Dict[str, object]], max_len: int):
        self.rows = mapped_rows
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = self.rows[idx]
        seq = list(ex["history_tok"]) + [int(ex["target_tok"])]  # type: ignore[arg-type]
        if len(seq) > self.max_len:
            seq = seq[-self.max_len :]
        return {"seq": seq}


class ItemLmEvalDataset(Dataset):
    def __init__(self, mapped_rows: List[Dict[str, object]], token_vocab_size_without_unk: int, unk_id: Optional[int]):
        self.rows = mapped_rows
        self.vocab_wo_unk = int(token_vocab_size_without_unk)
        self.unk_id = unk_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = self.rows[idx]
        target_tok = int(ex["target_tok"])
        target_is_known = target_tok < self.vocab_wo_unk
        return {
            "history_tok": list(ex["history_tok"]),  # type: ignore[arg-type]
            "target_tok": target_tok,
            "target_in_vocab": int(target_is_known),
            "history_item": list(ex["history_item"]),  # type: ignore[arg-type]
            "target_item": int(ex["target_item"]),
        }


def build_train_collate(pad_id: int):
    def _fn(batch: List[Dict[str, object]]) -> TrainBatch:
        bsz = len(batch)
        max_len = max(len(x["seq"]) for x in batch)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((bsz, max_len), dtype=torch.bool)
        labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
        for i, ex in enumerate(batch):
            seq: List[int] = ex["seq"]  # type: ignore[assignment]
            seq_t = torch.tensor(seq, dtype=torch.long)
            L = len(seq)
            input_ids[i, :L] = seq_t
            attn[i, :L] = True
            if L > 1:
                labels[i, : L - 1] = seq_t[1:]
        return TrainBatch(input_ids=input_ids, attention_mask=attn, labels=labels)

    return _fn


def build_eval_collate(pad_id: int):
    def _fn(batch: List[Dict[str, object]]) -> EvalBatch:
        bsz = len(batch)
        max_len = max(len(x["history_tok"]) for x in batch)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((bsz, max_len), dtype=torch.bool)
        target_tok = torch.zeros((bsz,), dtype=torch.long)
        target_in_vocab = torch.zeros((bsz,), dtype=torch.long)
        history_item: List[List[int]] = []
        target_item: List[int] = []
        for i, ex in enumerate(batch):
            hist: List[int] = ex["history_tok"]  # type: ignore[assignment]
            L = len(hist)
            input_ids[i, :L] = torch.tensor(hist, dtype=torch.long)
            attn[i, :L] = True
            target_tok[i] = int(ex["target_tok"])
            target_in_vocab[i] = int(ex["target_in_vocab"])
            history_item.append(list(ex["history_item"]))  # type: ignore[arg-type]
            target_item.append(int(ex["target_item"]))
        return EvalBatch(
            input_ids=input_ids,
            attention_mask=attn,
            target_tok=target_tok,
            target_in_vocab=target_in_vocab,
            history_item=history_item,
            target_item=target_item,
        )

    return _fn


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), 1)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
    }
    torch.save(payload, path)


def evaluate_token(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    ks: List[int],
    topk_tokens: int,
    pad_id: int,
    token_vocab_size: int,
) -> Dict[str, float]:
    model.eval()
    rows = []
    target_total = 0
    target_in_vocab = 0
    target_reachable = 0
    kmax = max(max(ks), topk_tokens)
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch.input_ids.to(device)
            attn = batch.attention_mask.to(device)
            target = batch.target_tok.to(device)
            logits = model(input_ids, attn)
            last_logits = get_last_logits(logits, attn)
            last_logits[:, pad_id] = -1e9
            rank = torch.topk(last_logits, k=min(kmax, last_logits.size(-1)), dim=-1).indices.cpu().tolist()
            ys = target.cpu().tolist()
            in_vocab = batch.target_in_vocab.tolist()
            for r, y, iv in zip(rank, ys, in_vocab):
                used = [int(t) for t in r[:topk_tokens] if int(t) < token_vocab_size]
                rows.append({"rank": used, "target": int(y)})
                target_total += 1
                target_in_vocab += int(iv)
                if int(y) in used:
                    target_reachable += 1
    metrics = evaluate_rows(rows, ks)
    metrics["target_total"] = int(target_total)
    metrics["target_in_vocab"] = int(target_in_vocab)
    metrics["target_in_vocab_ratio"] = float(target_in_vocab / target_total) if target_total else 0.0
    metrics["target_reachable"] = int(target_reachable)
    metrics["target_reachable_ratio"] = float(target_reachable / target_total) if target_total else 0.0
    return metrics


def try_auto_batch_size(
    model: nn.Module,
    dataset: Dataset,
    collate_fn,
    args,
    device: torch.device,
    criterion: nn.Module,
    amp_enabled: bool,
) -> int:
    if str(device) == "cpu":
        return int(args.batch_size)
    candidates = [int(args.batch_size)]
    for v in [128, 64]:
        if v not in candidates and v <= int(args.batch_size):
            candidates.append(v)
    for bs in candidates:
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        if len(loader) == 0:
            return bs
        batch = next(iter(loader))
        try:
            model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch.input_ids.to(device), batch.attention_mask.to(device))
                loss = criterion(logits.reshape(-1, logits.size(-1)), batch.labels.to(device).reshape(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return bs
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            continue
    raise RuntimeError("OOM even with batch_size=64, try lower --batch_size manually")


def validate_resume_config(saved: Dict[str, object], current: Dict[str, object]) -> None:
    keys = ["token_vocab_size", "model_vocab_size", "pad_id", "max_len", "d_model", "n_layers", "n_heads", "ff_mult"]
    bad = []
    for k in keys:
        if saved.get(k) != current.get(k):
            bad.append((k, saved.get(k), current.get(k)))
    if bad:
        raise ValueError(
            "Resume config mismatch: "
            + ", ".join([f"{k}: saved={a}, current={b}" for k, a, b in bad])
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--oov_strategy", type=str, default="map_to_unk", choices=["drop", "map_to_unk"])
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--topk_tokens", type=int, default=100)
    ap.add_argument("--ks", type=str, default="20,50,100")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt_path", type=str, default="")
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    set_seed(args.seed)
    ks = parse_ks(args.ks)
    map_to_unk = args.oov_strategy == "map_to_unk"

    train_rows_raw = load_jsonl(args.train_jsonl)
    val_rows_raw = load_jsonl(args.val_jsonl)
    item2tok, token_vocab_size, pad_id, unk_id = build_item_vocab(train_rows_raw, map_to_unk=map_to_unk)
    model_vocab_size = token_vocab_size + 1
    known_item_size = len(item2tok)

    train_mapped = []
    val_mapped = []
    train_drop = 0
    val_drop = 0
    for ex in train_rows_raw:
        mapped = map_row(ex, item2tok=item2tok, oov_strategy=args.oov_strategy, unk_id=unk_id, max_len=args.max_len)
        if mapped is None:
            train_drop += 1
            continue
        train_mapped.append(mapped)
    for ex in val_rows_raw:
        mapped = map_row(ex, item2tok=item2tok, oov_strategy=args.oov_strategy, unk_id=unk_id, max_len=args.max_len)
        if mapped is None:
            val_drop += 1
            continue
        val_mapped.append(mapped)

    train_ds = ItemLmTrainDataset(train_mapped, max_len=args.max_len)
    val_ds = ItemLmEvalDataset(val_mapped, token_vocab_size_without_unk=known_item_size, unk_id=unk_id)
    if len(train_ds) == 0:
        raise ValueError("No train samples after OOV handling.")
    if len(val_ds) == 0:
        raise ValueError("No val samples after OOV handling.")

    item2tok_path = os.path.join(args.run_dir, "item2tok.json")
    with open(item2tok_path, "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in item2tok.items()}, f, ensure_ascii=False, indent=2)

    tok2item = [0] * known_item_size
    for item, tok in item2tok.items():
        tok2item[int(tok)] = int(item)
    tok2item_path = os.path.join(args.run_dir, "tok2item.json")
    with open(tok2item_path, "w", encoding="utf-8") as f:
        json.dump(tok2item, f, ensure_ascii=False, indent=2)

    device = torch.device(args.device)
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    model = GenRecDecoderLM(
        vocab_size=model_vocab_size,
        pad_id=pad_id,
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    train_collate = build_train_collate(pad_id=pad_id)
    chosen_bs = try_auto_batch_size(
        model=model,
        dataset=train_ds,
        collate_fn=train_collate,
        args=args,
        device=device,
        criterion=criterion,
        amp_enabled=amp_enabled,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=chosen_bs,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=chosen_bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=build_eval_collate(pad_id=pad_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, int(args.epochs) * max(1, len(train_loader)))
    scheduler = build_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    config = {
        "mode": "item",
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "oov_strategy": args.oov_strategy,
        "known_item_size": int(known_item_size),
        "token_vocab_size": int(token_vocab_size),
        "model_vocab_size": int(model_vocab_size),
        "unk_id": int(unk_id) if unk_id is not None else None,
        "pad_id": int(pad_id),
        "max_len": int(args.max_len),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "ff_mult": int(args.ff_mult),
        "dropout": float(args.dropout),
        "batch_size": int(chosen_bs),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_steps": int(args.warmup_steps),
        "patience": int(args.patience),
        "grad_clip": float(args.grad_clip),
        "topk_tokens": int(args.topk_tokens),
        "ks": ks,
        "seed": int(args.seed),
        "amp_enabled": bool(amp_enabled),
        "train_rows_raw": int(len(train_rows_raw)),
        "train_rows_used": int(len(train_mapped)),
        "train_rows_dropped_oov": int(train_drop),
        "val_rows_raw": int(len(val_rows_raw)),
        "val_rows_used": int(len(val_mapped)),
        "val_rows_dropped_oov": int(val_drop),
    }
    config_path = os.path.join(args.run_dir, "config.json")

    start_epoch = 1
    global_step = 0
    best_metric = float("-inf")
    best_epoch = 0
    patience_count = 0

    if args.resume:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json for resume: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        validate_resume_config(saved_cfg, config)
        ckpt_path = args.ckpt_path or os.path.join(args.run_dir, "checkpoint_last.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") is not None and amp_enabled:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_metric = float(ckpt.get("best_metric", float("-inf")))
        best_epoch = int(ckpt.get("best_epoch", 0))
        print(
            f"[resume] ckpt={ckpt_path} start_epoch={start_epoch} global_step={global_step} "
            f"best_val_NDCG@20={best_metric:.6f}",
            flush=True,
        )
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(args.run_dir, "train_log.jsonl")
    if not args.resume and os.path.exists(log_path):
        os.remove(log_path)

    print(
        f"[train-start] device={device} amp={amp_enabled} train={len(train_ds)} val={len(val_ds)} "
        f"batch_size={chosen_bs} steps_per_epoch={len(train_loader)} epochs={args.epochs} "
        f"known_items={known_item_size} token_vocab_size={token_vocab_size} model_vocab_size={model_vocab_size}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        loss_sum = 0.0
        seen_rows = 0
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch.input_ids.to(device)
            attn = batch.attention_mask.to(device)
            labels = batch.labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(input_ids, attn)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_rows = int(labels.size(0))
            loss_sum += float(loss.item()) * batch_rows
            seen_rows += batch_rows
            global_step += 1

            if args.log_every > 0 and (step % args.log_every == 0 or step == 1 or step == len(train_loader)):
                avg_loss = loss_sum / max(1, seen_rows)
                lr = optimizer.param_groups[0]["lr"]
                msg = {
                    "event": "train_step",
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "loss_avg": avg_loss,
                    "lr": lr,
                }
                append_jsonl(log_path, msg)
                print(
                    f"[train] epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"global_step={global_step} loss_avg={avg_loss:.6f} lr={lr:.6e}",
                    flush=True,
                )

        val_metrics = evaluate_token(
            model=model,
            data_loader=val_loader,
            device=device,
            ks=ks,
            topk_tokens=args.topk_tokens,
            pad_id=pad_id,
            token_vocab_size=token_vocab_size,
        )
        val_score = float(val_metrics.get("NDCG@20", 0.0))
        improved = val_score > best_metric
        if improved:
            best_metric = val_score
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        ckpt_last = os.path.join(args.run_dir, "checkpoint_last.pt")
        save_checkpoint(
            ckpt_last,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
        if improved:
            ckpt_best = os.path.join(args.run_dir, "checkpoint_best.pt")
            save_checkpoint(
                ckpt_best,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                best_epoch=best_epoch,
            )

        train_loss = loss_sum / max(1, seen_rows)
        epoch_sec = time.time() - epoch_start
        epoch_log = {
            "event": "epoch_end",
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_Recall@20": val_metrics.get("Recall@20", 0.0),
            "val_NDCG@20": val_metrics.get("NDCG@20", 0.0),
            "best_val_NDCG@20": best_metric,
            "best_epoch": best_epoch,
            "epoch_sec": epoch_sec,
        }
        append_jsonl(log_path, epoch_log)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_Recall@20={val_metrics.get('Recall@20', 0.0):.6f} "
            f"val_NDCG@20={val_metrics.get('NDCG@20', 0.0):.6f} best_val_NDCG@20={best_metric:.6f} "
            f"patience={patience_count}/{args.patience} epoch_sec={epoch_sec:.1f}",
            flush=True,
        )
        if patience_count >= args.patience:
            print(f"[early-stop] no improvement for {args.patience} epochs, stop at epoch={epoch}", flush=True)
            break

    best_ckpt_path = os.path.join(args.run_dir, "checkpoint_best.pt")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])

    final_val_metrics = evaluate_token(
        model=model,
        data_loader=val_loader,
        device=device,
        ks=ks,
        topk_tokens=args.topk_tokens,
        pad_id=pad_id,
        token_vocab_size=token_vocab_size,
    )
    val_metrics_path = os.path.join(args.run_dir, "metrics_token_val.json")
    save_metrics(val_metrics_path, final_val_metrics)

    print(f"users={final_val_metrics['users']}")
    for k in ks:
        print(f"Recall@{k}: {final_val_metrics[f'Recall@{k}']:.6f}   NDCG@{k}: {final_val_metrics[f'NDCG@{k}']:.6f}")
    print(
        "coverage: "
        f"in_vocab={final_val_metrics['target_in_vocab']}/{final_val_metrics['target_total']} "
        f"({final_val_metrics['target_in_vocab_ratio']:.6f})  "
        f"reachable={final_val_metrics['target_reachable']}/{final_val_metrics['target_total']} "
        f"({final_val_metrics['target_reachable_ratio']:.6f})"
    )
    print(f"saved: {args.run_dir}")


if __name__ == "__main__":
    main()
