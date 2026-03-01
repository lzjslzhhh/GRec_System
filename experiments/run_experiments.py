import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def abs_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(ROOT_DIR, p))


def run_cmd(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)


def read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_dataset_tag(raw_csv: str, train_jsonl: str, test_jsonl: str) -> str:
    text = " ".join([raw_csv.lower(), train_jsonl.lower(), test_jsonl.lower()])
    if "big_matrix" in text or "big_matrix_sw" in text:
        return "big_matrix"
    if "small_matrix" in text or "small_matrix_sw" in text:
        return "small_matrix"
    return "unknown"


def maybe_train_item2vec(args) -> None:
    item_ids_npy = abs_path(args.item_ids_npy)
    item_emb_npy = abs_path(args.item_emb_npy)
    if os.path.exists(item_ids_npy) and os.path.exists(item_emb_npy):
        return
    if not args.train_item2vec_if_missing:
        raise FileNotFoundError(
            f"Missing item2vec npy files: {item_ids_npy}, {item_emb_npy}. "
            "Provide files or set --train_item2vec_if_missing."
        )
    run_cmd(
        [
            args.python_bin,
            "baseline/item2vec_train_from_raw.py",
            "--raw_csv",
            abs_path(args.raw_csv),
            "--out_dir",
            abs_path(args.item2vec_out_dir),
            "--train_ratio",
            str(args.train_ratio),
            "--val_ratio",
            str(args.val_ratio),
            "--max_seq_len",
            str(args.item2vec_max_seq_len),
            "--dim",
            str(args.item2vec_dim),
            "--window_size",
            str(args.item2vec_window_size),
            "--negative",
            str(args.item2vec_negative),
            "--sample",
            str(args.item2vec_sample),
            "--ns_exponent",
            str(args.item2vec_ns_exponent),
            "--epochs",
            str(args.item2vec_epochs),
            "--min_count",
            str(args.item2vec_min_count),
            "--workers",
            str(args.item2vec_workers),
            "--seed",
            str(args.seed),
            "--sg",
            "1",
        ]
    )


def write_results(path: str, rows: List[Dict[str, object]]) -> None:
    fields = [
        "dataset",
        "mode",
        "exp_name",
        "K",
        "oov_strategy",
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
        "run_dir",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def make_result_row(
    dataset: str,
    mode: str,
    exp_name: str,
    k: int,
    oov_strategy: str,
    cfg: Dict[str, object],
    token_metrics: Dict[str, object],
    item_metrics: Dict[str, object],
    run_dir: str,
    notes: str,
) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "mode": mode,
        "exp_name": exp_name,
        "K": int(k),
        "oov_strategy": oov_strategy,
        "vocab_size": int(cfg.get("token_vocab_size", 0)),
        "model_vocab_size": int(cfg.get("model_vocab_size", 0)),
        "d_model": int(cfg.get("d_model", 0)),
        "n_layers": int(cfg.get("n_layers", 0)),
        "n_heads": int(cfg.get("n_heads", 0)),
        "token_Recall@20": token_metrics.get("Recall@20", 0.0),
        "token_NDCG@20": token_metrics.get("NDCG@20", 0.0),
        "token_Recall@50": token_metrics.get("Recall@50", 0.0),
        "token_NDCG@50": token_metrics.get("NDCG@50", 0.0),
        "token_Recall@100": token_metrics.get("Recall@100", 0.0),
        "token_NDCG@100": token_metrics.get("NDCG@100", 0.0),
        "item_Recall@20": item_metrics.get("Recall@20", 0.0),
        "item_NDCG@20": item_metrics.get("NDCG@20", 0.0),
        "item_Recall@50": item_metrics.get("Recall@50", 0.0),
        "item_NDCG@50": item_metrics.get("NDCG@50", 0.0),
        "item_Recall@100": item_metrics.get("Recall@100", 0.0),
        "item_NDCG@100": item_metrics.get("NDCG@100", 0.0),
        "target_total": token_metrics.get("target_total", 0),
        "target_in_vocab_ratio": token_metrics.get("target_in_vocab_ratio", 0.0),
        "target_reachable_ratio": token_metrics.get("target_reachable_ratio", 0.0),
        "notes": notes,
        "run_dir": run_dir,
    }


def run_sid_pipeline(args, k: int, all_rows: List[Dict[str, object]]) -> None:
    exp_name = f"{args.dataset_tag}_GenRec-SID-LM-K{k}"
    run_dir = abs_path(os.path.join(args.runs_dir, exp_name))
    os.makedirs(run_dir, exist_ok=True)

    run_cmd(
        [
            args.python_bin,
            "semantic_id/build_semantic_id_kmeans.py",
            "--item_ids_npy",
            abs_path(args.item_ids_npy),
            "--item_emb_npy",
            abs_path(args.item_emb_npy),
            "--k",
            str(k),
            "--seed",
            str(args.seed),
            "--out_dir",
            run_dir,
        ]
    )

    item2sid_path = os.path.join(run_dir, "item2sid.json")
    train_sid = os.path.join(run_dir, "train_sid.jsonl")
    val_sid = os.path.join(run_dir, "val_sid.jsonl")
    test_sid = os.path.join(run_dir, "test_sid.jsonl")
    for src, dst in [
        (abs_path(args.train_jsonl), train_sid),
        (abs_path(args.val_jsonl), val_sid),
        (abs_path(args.test_jsonl), test_sid),
    ]:
        run_cmd(
            [
                args.python_bin,
                "semantic_id/convert_jsonl_to_sid.py",
                "--in_jsonl",
                src,
                "--item2sid",
                item2sid_path,
                "--out_jsonl",
                dst,
                "--oov_strategy",
                args.oov_strategy,
                "--keep_item_fields",
                "true",
            ]
        )

    train_cmd = [
        args.python_bin,
        "gen_retrieval/train_genrec_sid_lm.py",
        "--train_jsonl",
        train_sid,
        "--val_jsonl",
        val_sid,
        "--run_dir",
        run_dir,
        "--kmeans_k",
        str(k),
        "--max_len",
        str(args.max_len),
        "--d_model",
        str(args.d_model),
        "--n_layers",
        str(args.n_layers),
        "--n_heads",
        str(args.n_heads),
        "--ff_mult",
        str(args.ff_mult),
        "--dropout",
        str(args.dropout),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--warmup_steps",
        str(args.warmup_steps),
        "--patience",
        str(args.patience),
        "--topk_tokens",
        str(args.topk_tokens),
        "--ks",
        args.ks,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.no_amp:
        train_cmd.append("--no_amp")
    if args.resume:
        train_cmd.append("--resume")
    run_cmd(train_cmd)

    token_metrics_path = os.path.join(run_dir, "metrics_token_test.json")
    item_metrics_path = os.path.join(run_dir, "metrics_item_test.json")
    run_cmd(
        [
            args.python_bin,
            "gen_retrieval/eval_genrec.py",
            "--run_dir",
            run_dir,
            "--eval_jsonl",
            test_sid,
            "--mode",
            "sid",
            "--raw_csv",
            abs_path(args.raw_csv),
            "--item2sid",
            item2sid_path,
            "--train_ratio",
            str(args.train_ratio),
            "--val_ratio",
            str(args.val_ratio),
            "--topk_tokens",
            str(args.topk_tokens),
            "--topk_items",
            str(args.topk_items),
            "--expand_strategy",
            args.expand_strategy,
            "--item_ids_npy",
            abs_path(args.item_ids_npy),
            "--item_emb_npy",
            abs_path(args.item_emb_npy),
            "--ks",
            args.ks,
            "--device",
            args.device,
            "--metrics_token_out",
            token_metrics_path,
            "--metrics_item_out",
            item_metrics_path,
            "--sample_cases_out",
            os.path.join(run_dir, "sample_cases.json"),
            "--results_csv",
            "",
            "--results_tag",
            f"{args.dataset_tag} GenRec-SID-LM K={k}",
        ]
    )

    cfg = read_json(os.path.join(run_dir, "config.json"))
    token_metrics = read_json(token_metrics_path)
    item_metrics = read_json(item_metrics_path)
    all_rows.append(
        make_result_row(
            dataset=args.dataset_tag,
            mode="sid",
            exp_name=exp_name,
            k=k,
            oov_strategy=args.oov_strategy,
            cfg=cfg,
            token_metrics=token_metrics,
            item_metrics=item_metrics,
            run_dir=run_dir,
            notes=f"{args.dataset_tag} GenRec-SID-LM K={k} {args.expand_strategy}",
        )
    )


def run_item_pipeline(args, all_rows: List[Dict[str, object]]) -> None:
    exp_name = f"{args.dataset_tag}_GenRec-ITEM-LM"
    run_dir = abs_path(os.path.join(args.runs_dir, exp_name))
    os.makedirs(run_dir, exist_ok=True)

    train_cmd = [
        args.python_bin,
        "gen_retrieval/train_genrec_item_lm.py",
        "--train_jsonl",
        abs_path(args.train_jsonl),
        "--val_jsonl",
        abs_path(args.val_jsonl),
        "--run_dir",
        run_dir,
        "--oov_strategy",
        args.oov_strategy,
        "--max_len",
        str(args.max_len),
        "--d_model",
        str(args.d_model),
        "--n_layers",
        str(args.n_layers),
        "--n_heads",
        str(args.n_heads),
        "--ff_mult",
        str(args.ff_mult),
        "--dropout",
        str(args.dropout),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--warmup_steps",
        str(args.warmup_steps),
        "--patience",
        str(args.patience),
        "--topk_tokens",
        str(args.topk_tokens),
        "--ks",
        args.ks,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.no_amp:
        train_cmd.append("--no_amp")
    if args.resume:
        train_cmd.append("--resume")
    run_cmd(train_cmd)

    token_metrics_path = os.path.join(run_dir, "metrics_token_test.json")
    item_metrics_path = os.path.join(run_dir, "metrics_item_test.json")
    run_cmd(
        [
            args.python_bin,
            "gen_retrieval/eval_genrec.py",
            "--run_dir",
            run_dir,
            "--eval_jsonl",
            abs_path(args.test_jsonl),
            "--mode",
            "item",
            "--topk_tokens",
            str(args.topk_tokens),
            "--ks",
            args.ks,
            "--device",
            args.device,
            "--metrics_token_out",
            token_metrics_path,
            "--metrics_item_out",
            item_metrics_path,
            "--sample_cases_out",
            os.path.join(run_dir, "sample_cases.json"),
            "--results_csv",
            "",
            "--results_tag",
            f"{args.dataset_tag} GenRec-ITEM-LM",
        ]
    )

    cfg = read_json(os.path.join(run_dir, "config.json"))
    token_metrics = read_json(token_metrics_path)
    item_metrics = read_json(item_metrics_path)
    all_rows.append(
        make_result_row(
            dataset=args.dataset_tag,
            mode="item",
            exp_name=exp_name,
            k=-1,
            oov_strategy=args.oov_strategy,
            cfg=cfg,
            token_metrics=token_metrics,
            item_metrics=item_metrics,
            run_dir=run_dir,
            notes=f"{args.dataset_tag} GenRec-ITEM-LM",
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python_bin", type=str, default=sys.executable)
    ap.add_argument("--raw_csv", type=str, default="KuaiRec 2.0/data/small_matrix.csv")
    ap.add_argument("--train_jsonl", type=str, default="data/processed/small_matrix_sw/train.jsonl")
    ap.add_argument("--val_jsonl", type=str, default="data/processed/small_matrix_sw/val.jsonl")
    ap.add_argument("--test_jsonl", type=str, default="data/processed/small_matrix_sw/test.jsonl")
    ap.add_argument("--runs_dir", type=str, default="experiments/runs")
    ap.add_argument("--results_csv", type=str, default="experiments/results.csv")
    ap.add_argument("--dataset_tag", type=str, default="")
    ap.add_argument("--ks", type=str, default="20,50,100")
    ap.add_argument("--k_list", type=str, default="256,512,1024")
    ap.add_argument("--oov_strategy", type=str, default="map_to_unk", choices=["drop", "map_to_unk"])
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--topk_tokens", type=int, default=100)
    ap.add_argument("--topk_items", type=int, default=100)
    ap.add_argument("--expand_strategy", type=str, default="expand_embed", choices=["expand_pop", "expand_embed"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--run_item", action="store_true", default=True)
    ap.add_argument("--skip_item", action="store_true")
    ap.add_argument("--item_ids_npy", type=str, default="artifacts/item2vec/item_ids.npy")
    ap.add_argument("--item_emb_npy", type=str, default="artifacts/item2vec/item_emb.npy")
    ap.add_argument("--train_item2vec_if_missing", action="store_true")
    ap.add_argument("--item2vec_out_dir", type=str, default="artifacts/item2vec")
    ap.add_argument("--item2vec_max_seq_len", type=int, default=200)
    ap.add_argument("--item2vec_dim", type=int, default=64)
    ap.add_argument("--item2vec_window_size", type=int, default=5)
    ap.add_argument("--item2vec_negative", type=int, default=5)
    ap.add_argument("--item2vec_sample", type=float, default=1e-3)
    ap.add_argument("--item2vec_ns_exponent", type=float, default=0.75)
    ap.add_argument("--item2vec_epochs", type=int, default=5)
    ap.add_argument("--item2vec_min_count", type=int, default=1)
    ap.add_argument("--item2vec_workers", type=int, default=4)
    args = ap.parse_args()
    if not args.dataset_tag:
        args.dataset_tag = infer_dataset_tag(
            raw_csv=args.raw_csv,
            train_jsonl=args.train_jsonl,
            test_jsonl=args.test_jsonl,
        )

    os.makedirs(abs_path(args.runs_dir), exist_ok=True)
    os.makedirs(os.path.dirname(abs_path(args.results_csv)), exist_ok=True)
    maybe_train_item2vec(args)

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    if not k_list:
        raise ValueError("k_list is empty.")

    all_rows: List[Dict[str, object]] = []
    for k in k_list:
        run_sid_pipeline(args=args, k=k, all_rows=all_rows)
    if args.run_item and not args.skip_item:
        run_item_pipeline(args=args, all_rows=all_rows)

    write_results(abs_path(args.results_csv), rows=all_rows)
    print(f"saved results: {abs_path(args.results_csv)}")
    print(f"total_rows={len(all_rows)}")


if __name__ == "__main__":
    main()
