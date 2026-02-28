# GRec_System

一个面向**搜索/推荐实习**的端到端实验项目：  
**语义 ID（离散化表征）+ 生成式召回（Gen Retrieval）+ 多任务精排（MTL Ranker）**

## 当前进度

- ✅ KuaiRec 数据落地（`small_matrix.csv` 跑通全流程）
- ✅ 用户序列构建（稳定排序）
- ✅ 按用户时间线切分 `train/val/test`（严格无泄漏）
- ✅ 仅 `train` 段滑动窗口构造训练样本（next-item）
- ✅ Popularity 召回 baseline（从原始交互统计 train 热度 + seen filter）
- ✅ ItemCF 召回 baseline + 覆盖率诊断
- ✅ Item2Vec baseline（`gensim` 训练 embedding + `torch` 相似度检索）与覆盖率诊断
- ✅ Item2Vec 训练新增长尾相关参数：`sample`、`ns_exponent`
- ✅ 离线指标跑通（Recall/NDCG）

## 1. 目标概述

### 1.1 输入

- 用户历史行为序列（click/cart/buy...）
- item 内容特征（category/title/brand/caption...，后续在语义 ID 阶段引入）

### 1.2 输出

1. item 的**语义 ID**（离散 token 串）
2. **生成式召回**：给定用户序列生成 TopK 候选 item（基于语义 ID 生成）
3. **多任务精排**：对候选进行 CTR/CVR/CTCVR/停留时长等多目标排序

### 1.3 评估指标

- 召回：Recall@K、NDCG@K
- 精排：AUC、GAUC、LogLoss（可选：Calibration）
- 端到端：NDCG@K / HitRate@K（最终列表）

## 2. 数据集

- 数据集：**KuaiRec**
- 当前使用：`small_matrix.csv`（已跑通全流程）
- 原始交互表关键字段（当前项目用到）：`user_id`、`video_id`、`timestamp`

说明：后续语义 ID 阶段将进一步引入 KuaiRec 的 item 侧信息（如类别/文本/caption 等）。

## 3. 数据预处理（已完成）

### 3.1 用户序列构建

从 `small_matrix.csv` 读取 `user_id, video_id, timestamp`，对每个用户按 `timestamp` 升序排序得到序列：

`[(video_1, t_1), (video_2, t_2), ...]`

为避免同一用户同一时间戳下顺序不稳定，使用**行号**作为次级排序键（稳定排序）。

### 3.2 时间切分（严格防泄漏）

对每个用户序列按比例切分（按用户时间线）：

- train：前 80%
- val：中间 10%
- test：最后 10%

> 切分方式为“按用户时间线切分”，保证 test 是未来。训练统计与模型训练均不使用 val/test 信息。

### 3.3 滑动窗口多样本（训练集，仅 train 段）

仅在 `train` 段做滑动窗口生成多条 next-item 样本：

- 对 train 序列 `[(v1,t1),...,(vn,tn)]`
- 对每个位置 `i`（从 `min_hist_len` 到 `n-1`）构造：
  - `history = seq[max(0, i-max_seq_len): i]`
  - `target = seq[i]`

关键参数（当前配置）：

- `max_seq_len = 50`
- `min_hist_len = 3`
- `stride = 1`
- `max_train_samples_per_user = 200`（每个用户最多保留最近 200 条样本）

### 3.4 val/test 构造方式（每个用户 1 条样本）

- val：使用（train + val）历史预测 val 最后一次
- test：使用（train + val + test）历史预测 test 最后一次

> 当前实现以“每个用户最后一次交互”为预测目标，确保评估对齐未来一步。

### 3.5 预处理统计结果

```json
{
  "users_total": 1411,
  "users_kept": 1411,
  "dropped_too_short": 0,
  "train_samples": 282200,
  "avg_train_samples_per_user": 200.0
}
```

### 3.6 输出文件

- `data/processed/small_matrix_sw/train.jsonl`
- `data/processed/small_matrix_sw/val.jsonl`
- `data/processed/small_matrix_sw/test.jsonl`
- `data/processed/small_matrix_sw/stats.json`

## 4. Baseline 召回

### 4.1 Popularity 统计口径（严格无泄漏）

热度统计来自原始交互表的 train 时间段（按用户时间线切分后的 train 段交互）：

- 不使用滑窗样本统计热度（避免滑窗导致 history 重复计数放大）
- 推荐阶段对每个用户：
  - 从热门榜按顺序取 TopK
  - `seen filter`：过滤掉用户 `history` 里已经看过的 item

### 4.2 Popularity 评估结果（test）

```text
users=1411  popular_size=3044
Recall@20 : 0.001417   NDCG@20 : 0.000391
Recall@50 : 0.016300   NDCG@50 : 0.003185
Recall@100: 0.099220   NDCG@100: 0.016360
```

### 4.3 ItemCF 评估结果（test）

```text
users=1411  itemcf_size=1995  train_item_size=3044
Recall@20 : 0.119773   NDCG@20 : 0.055135
Recall@50 : 0.131821   NDCG@50 : 0.057643
Recall@100: 0.133239   NDCG@100: 0.057886
```

### 4.4 ItemCF 覆盖率诊断（为什么 Recall 会被卡住）

在 test 集（1411 个 target）上，统计结果如下：

- `target_in_train = 460 / 1411 = 32.60%`
  - 含义：只有 32.6% 的 test target 在训练交互里出现过（其余属于冷启动/未见过，纯协同过滤天然学不到）
- `target_in_itemcf = 453 / 1411 = 32.10%`
  - 含义：在训练中出现过的 item 里，仍有一部分没有进入 ItemCF 可用节点（无邻居或被过滤）
- `target_reachable = 188 / 1411 = 13.32%`
  - 含义：对当前每个用户的 history，target 能从 ItemCF 图“一跳召回”到的比例只有 13.32%

关键解释：

- `itemcf_size=1995` 表示 ItemCF 图中“有邻居可用”的 item 数（`itemcf_topk` 的 key 数）
- 它小于 `train_item_size=3044`，说明有不少训练 item 没有形成稳定共现邻居（受去重、窗口、`min_co` 等影响）
- 当前配置下，`target_reachable_ratio=13.32%` 与 `Recall@100=13.32%` 基本一致，说明瓶颈主要不是排序，而是候选覆盖（可达性）上限

### 4.5 Item2Vec 评估结果（test）

Item2Vec 用于学习 item embedding，并通过“embedding 近邻”进行召回。该 baseline 的核心诊断点是：候选覆盖（邻居 topk）会直接影响可达性与 Recall 上限。

#### 4.5.1 Item2Vec（邻居 topk = 200）

```text
users=1411  item2vec_size=2307
Recall@20 : 0.057406   NDCG@20 : 0.023999
Recall@50 : 0.067328   NDCG@50 : 0.026043
Recall@100: 0.068037   NDCG@100: 0.026166
coverage: in_emb=459/1411 (0.325301)  reachable=96/1411 (0.068037)
```

#### 4.5.2 Item2Vec（邻居 topk = 1000）

```text
users=1411  item2vec_size=2307
Recall@20 : 0.048901   NDCG@20 : 0.018001
Recall@50 : 0.102764   NDCG@50 : 0.028636
Recall@100: 0.150248   NDCG@100: 0.036401
coverage: in_emb=459/1411 (0.325301)  reachable=222/1411 (0.157335)
```

#### 4.5.3 Item2Vec（邻居 topk = 2000）

```text
users=1411  item2vec_size=2307
Recall@20 : 0.052445   NDCG@20 : 0.020982
Recall@50 : 0.111977   NDCG@50 : 0.032526
Recall@100: 0.212615   NDCG@100: 0.048823
coverage: in_emb=459/1411 (0.325301)  reachable=364/1411 (0.257973)
```

#### 4.5.4 Item2Vec 覆盖率解释（关键结论）

- `in_emb = 459 / 1411 = 32.53%`
  - 含义：只有 32.53% 的 test target 在 item2vec 的词表/embedding 中（严格用 train 训练时，cold item 会导致 hard upper bound）
- `reachable` 会随着邻居 topk 增大显著上升（`6.80% -> 15.73% -> 25.80%`）
  - 含义：Item2Vec baseline 的主要瓶颈首先是候选覆盖（可达性）
- 当候选池变大后，`Recall@100` 提升显著，但 `Recall@20` 提升有限
  - 含义：扩大候选覆盖后，前排排序压力增大，后续可由更强的精排模型（MTL Ranker）承接优化

## 5. 运行方式（当前阶段）

### 5.1 预处理（滑窗多样本）

```bash
python preprocess/preprocess_kuairec.py \
  --data_dir "./KuaiRec 2.0/data" \
  --matrix small_matrix.csv \
  --out_dir ./data/processed \
  --min_seq_len 5 \
  --max_seq_len 50 \
  --min_hist_len 3 \
  --stride 1 \
  --max_train_samples_per_user 200
```

### 5.2 Popularity baseline（从原始交互统计 train 热度）

```bash
python baseline/baseline_pop_eval.py \
  --raw_csv "./KuaiRec 2.0/data/small_matrix.csv" \
  --eval_jsonl "./data/processed/small_matrix_sw/test.jsonl" \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --ks 20,50,100
```

### 5.3 ItemCF baseline（从原始交互构建共现图）

```bash
python -m baseline.baseline_itemcf_from_raw \
  --raw_csv "./KuaiRec 2.0/data/small_matrix.csv" \
  --eval_jsonl "./data/processed/small_matrix_sw/test.jsonl" \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --topk_sim 200 \
  --co_window 50 \
  --min_co 1.0 \
  --recent_n 10 \
  --pos_decay 0.8 \
  --ks 20,50,100
```

### 5.4 Item2Vec 训练（`gensim`）

```bash
python baseline/item2vec_train_from_raw.py \
  --raw_csv "./KuaiRec 2.0/data/small_matrix.csv" \
  --out_dir "./artifacts/item2vec" \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --max_seq_len 200 \
  --dim 64 \
  --window_size 5 \
  --negative 5 \
  --sample 1e-3 \
  --ns_exponent 0.75 \
  --epochs 5 \
  --min_count 1 \
  --workers 4 \
  --seed 42 \
  --sg 1
```

### 5.5 Item2Vec 评估（`torch` 相似度检索）

```bash
python baseline/baseline_item2vec_eval.py \
  --item_ids_npy "./artifacts/item2vec/item_ids.npy" \
  --item_emb_npy "./artifacts/item2vec/item_emb.npy" \
  --eval_jsonl "./data/processed/small_matrix_sw/test.jsonl" \
  --topk_sim 2000 \
  --recent_n 10 \
  --pos_decay 0.8 \
  --ks 20,50,100 \
  --device cpu
```

### 5.6 语义 ID v0（K-Means）构建

```bash
python semantic_id/build_semantic_id_kmeans.py \
  --item_ids_npy "./artifacts/item2vec/item_ids.npy" \
  --item_emb_npy "./artifacts/item2vec/item_emb.npy" \
  --k 512 \
  --seed 42 \
  --out_dir "./artifacts/semantic_id/kmeans_k512"
```

### 5.7 将 `train/val/test.jsonl` 转为 SID 序列

```bash
python semantic_id/convert_jsonl_to_sid.py \
  --in_jsonl "./data/processed/small_matrix_sw/train.jsonl" \
  --item2sid "./artifacts/semantic_id/kmeans_k512/item2sid.json" \
  --out_jsonl "./data/processed/small_matrix_sw_sid/kmeans_k512/train_sid.jsonl" \
  --oov_strategy drop \
  --keep_item_fields true

python semantic_id/convert_jsonl_to_sid.py \
  --in_jsonl "./data/processed/small_matrix_sw/val.jsonl" \
  --item2sid "./artifacts/semantic_id/kmeans_k512/item2sid.json" \
  --out_jsonl "./data/processed/small_matrix_sw_sid/kmeans_k512/val_sid.jsonl" \
  --oov_strategy map_to_unk \
  --keep_item_fields true

python semantic_id/convert_jsonl_to_sid.py \
  --in_jsonl "./data/processed/small_matrix_sw/test.jsonl" \
  --item2sid "./artifacts/semantic_id/kmeans_k512/item2sid.json" \
  --out_jsonl "./data/processed/small_matrix_sw_sid/kmeans_k512/test_sid.jsonl" \
  --oov_strategy map_to_unk \
  --keep_item_fields true
```

### 5.8 SID 空间 token popularity baseline（sanity check）

```bash
python semantic_id/token_pop_baseline.py \
  --train_jsonl "./data/processed/small_matrix_sw_sid/kmeans_k512/train_sid.jsonl" \
  --eval_jsonl "./data/processed/small_matrix_sw_sid/kmeans_k512/test_sid.jsonl" \
  --ks 20,50,100
```

## 6. 目录结构

```text
GRec_System/
├── README.md
├── KuaiRec 2.0/
│   └── data/
│       └── small_matrix.csv
├── preprocess/
│   └── preprocess_kuairec.py
├── semantic_id/
│   ├── build_semantic_id_kmeans.py
│   ├── convert_jsonl_to_sid.py
│   └── token_pop_baseline.py
├── baseline/
│   ├── baseline_pop_eval.py
│   ├── baseline_itemcf_from_raw.py
│   ├── baseline_item2vec_eval.py
│   ├── item2vec_train_from_raw.py
│   ├── eval_recall_metrics.py
│   └── item2vec/
│       ├── __init__.py
│       ├── data.py
│       ├── train.py
│       ├── recall.py
│       └── model.py
├── artifacts/
│   ├── item2vec/
│   │   ├── item_ids.npy
│   │   ├── item_emb.npy
│   │   ├── config.json
│   │   └── item_embeddings.npz
│   └── semantic_id/
│       └── kmeans_k512/
│           ├── item2sid.json
│           ├── centers.npy
│           ├── cluster_stats.json
│           └── coverage.json
└── data/
    └── processed/
        └── small_matrix_sw/
            ├── train.jsonl
            ├── val.jsonl
            ├── test.jsonl
            ├── stats.json
            ├── pop_metrics.json
            ├── itemcf_metrics.json
            └── item2vec_metrics.json
```

## 7. 下一步计划（Next）

### 7.1 召回 baseline 强化（更强对照）

- ItemKNN（共现/相似度）
- Item2Vec（序列 embedding）
- 可选：更强序列召回（GRU4Rec / SASRec）

### 7.2 语义 ID 两阶段实现（已确定）

> 目标：构建可用于 **生成式召回** 的离散语义表征（Semantic ID）。先用低成本方案打通闭环，再用更强表征与离散化提升语义质量与冷启动能力。

---

#### 阶段 A：`Item2Vec（仅交互） + K-Means`（MVP，优先实现）

- **定位**：最小可行闭环（MVP），快速验证“embedding → 离散化 → semantic ID → 下游可用”
- **输入**：仅用户-物品交互序列（train 段，严格无泄漏）
- **方法**：
  1) 用 Item2Vec（skip-gram/CBOW）训练 item embedding  
  2) 对 item embedding 做 K-Means 离散化，得到 **单层 semantic token**（或短 token 串）
- **产出**：
  - `item_emb`：item embedding（N × d）
  - `item2sid`：item → semantic_id（cluster id / token）
  - `*_sid.jsonl`：样本中的 history/target 替换为 semantic token 序列，用于生成式训练
- **评估 / Sanity Check**：
  - 覆盖率：train/test 出现 item 的 sid 覆盖率（尤其 train 需接近 100%）
  - 离散化分布：cluster size 分布（避免极端不均衡）
  - 下游可用性：基于 semantic id 的 next-token/next-item 训练是否收敛；召回 Recall/NDCG 是否可跑通

**阶段 A 当前最小闭环产物（已实现）**：

- Item2Vec 标准导出：`artifacts/item2vec/{item_ids.npy,item_emb.npy,config.json}`
- K-Means 语义 ID：`artifacts/semantic_id/kmeans_k{K}/item2sid.json`
- SID 数据集转换：`train_sid.jsonl / val_sid.jsonl / test_sid.jsonl` 及对应 `*.stats.json`
- 核心统计：
  - `cluster_stats.json`：`min/mean/median/p95/max`、`top10_largest_clusters`
  - `coverage.json`：`N_items/K/D`
  - `*.stats.json`：`total_lines/kept_lines/dropped_oov/unk_count/sid_vocab_size`

---

#### 阶段 B：`Two-Tower（交互 + 内容） + VQ / RQ-VAE`（增强版，阶段 A 稳定后推进）

- **定位**：提升表征语义质量 + 冷启动能力；升级离散化表达能力（多层 token 串）
- **输入**：
  - 交互特征（user 行为序列/统计等）
  - item 内容特征（category/caption/title/brand…，按 KuaiRec 可用字段接入）
- **方法**：
  1) Two-Tower 学习更强的 item 表征（支持内容信息融入，提升新物品泛化）
  2) VQ / RQ-VAE 将连续表征离散化为 **多层 token 串**（更强表达能力）
- **产出**：
  - `item_emb_v2`：增强 item embedding
  - `item2sid_v2`：item → 多层离散 token 串（语义 ID 序列）
  - 下游统一使用 semantic ID：生成式召回与排序阶段复用同一套 token 化接口
- **评估**：
  - 冷启动切片：对“train 未出现但有内容特征”的 item，观察 sid 覆盖与下游召回能力
  - 对比实验：阶段 A vs 阶段 B（召回指标、端到端指标、计算开销）

---

**说明**：Two-Tower 的核心作用是学习更强的 item 表征（尤其内容增强与冷启动），离散化与生成式召回阶段统一消费 **item semantic ID**，从而保证端到端链路一致。

### 7.3 生成式召回（Gen Retrieval）

- 用语义 ID token 序列训练 GPT-style decoder
- 训练目标：next-token / next-item generation
- 推断：生成 TopK 候选 item，与传统召回对比 Recall/NDCG

### 7.4 多任务精排（MTL Ranker）

- 结构：Shared-bottom / MMoE / PLE
- 目标：CTR + 完播/停留等多目标（按数据可得性选择）
- 指标：AUC/GAUC/LogLoss（可选 Calibration）

### 7.5 端到端评估与消融实验

- 无语义 ID vs 有语义 ID
- 无生成式 vs 生成式召回
- 单任务 vs 多任务
- 端到端 NDCG/HitRate 改善与代价（时延/参数量）
