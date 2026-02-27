# GRec_System

一个面向 **搜索/推荐实习** 的端到端实验项目：  
**语义ID（离散化表征） + 生成式召回（Gen Retrieval） + 多任务精排（MTL Ranker）**

当前已完成：
- ✅ KuaiRec 数据落地（`small_matrix.csv` 跑通全流程）
- ✅ 用户序列构建（稳定排序）
- ✅ 按用户时间线切分 train/val/test（严格无泄漏）
- ✅ 仅 train 段滑动窗口构造训练样本（next-item）
- ✅ Popularity 召回 baseline（从原始交互统计 train 热度 + seen filter）
- ✅ 离线指标跑通（Recall/NDCG）

---

## 1. 目标概述

### 1.1 输入
- 用户历史行为序列（click/cart/buy…）
- item 内容特征（category/title/brand/caption… *后续在语义ID阶段引入*）

### 1.2 输出
1) item 的 **语义ID**（离散 token 串）  
2) **生成式召回**：给定用户序列生成 TopK 候选 item（基于语义ID生成）  
3) **多任务精排**：对候选进行 CTR/CVR/CTCVR/停留时长等多目标排序  

### 1.3 评估指标
- 召回：Recall@K、NDCG@K
- 精排：AUC、GAUC、LogLoss（可选：Calibration）
- 端到端：NDCG@K / HitRate@K（最终列表）

---

## 2. 数据集

- 数据集：**KuaiRec**
- 当前使用：`small_matrix.csv`（已跑通全流程）
- 原始交互表关键字段（当前项目用到）：
  - `user_id`
  - `video_id`
  - `timestamp`

说明：
- 后续语义ID阶段将进一步引入 KuaiRec 的 item 侧信息（如类别/文本/caption 等）。

---

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

> 切分方式为「按用户时间线切分」，保证 test 是“未来”。  
> 训练统计与模型训练均不使用 val/test 信息。

### 3.3 滑动窗口多样本（训练集，仅 train 段）
仅在 **train 段** 做滑动窗口生成多条 next-item 样本：

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

> 注：当前实现以“每个用户最后一次交互”为预测目标，确保评估对齐“未来一步”。

### 3.5 预处理统计结果
```json
{
  "users_total": 1411,
  "users_kept": 1411,
  "dropped_too_short": 0,
  "train_samples": 282200,
  "avg_train_samples_per_user": 200.0
}
3.6 输出文件

processed/small_matrix_sw/train.jsonl

processed/small_matrix_sw/val.jsonl

processed/small_matrix_sw/test.jsonl

processed/small_matrix_sw/stats.json

4. Baseline：Popularity 召回（已完成）
4.1 统计口径（严格无泄漏）

热度统计来自原始交互表的 train 时间段（按用户时间线切分后的 train 段交互）：

✅ 不使用滑窗样本统计热度（避免滑窗导致 history 重复计数放大）

✅ 推荐阶段对每个用户：

从热门榜按顺序取 TopK

seen filter：过滤掉用户 history 里已经看过的 item

4.2 评估结果（test）
users=1411  popular_size=3044
Recall@20 : 0.001417   NDCG@20 : 0.000391
Recall@50 : 0.016300   NDCG@50 : 0.003185
Recall@100: 0.099220   NDCG@100: 0.016360
5. 运行方式（当前阶段）
5.1 预处理（滑窗多样本）
python preprocess_kuairec_sw.py \
  --data_dir . --matrix small_matrix.csv \
  --out_dir processed \
  --min_seq_len 5 \
  --max_seq_len 50 --min_hist_len 3 \
  --stride 1 --max_train_samples_per_user 200
5.2 Popularity baseline（从原始交互统计 train 热度）
python baseline_pop_from_raw.py \
  --raw_csv small_matrix.csv \
  --eval_jsonl processed/small_matrix_sw/test.jsonl \
  --train_ratio 0.8 --val_ratio 0.1 \
  --ks 20,50,100
6. 目录结构（建议）
GRec_System/
  data/
    small_matrix.csv
  processed/
    small_matrix_sw/
      train.jsonl
      val.jsonl
      test.jsonl
      stats.json
  preprocess_kuairec_sw.py
  baseline_pop_from_raw.py
  README.md
7. 下一步计划（Next）
7.1 召回 baseline 强化（更强对照）

ItemKNN（共现/相似度）

Item2Vec（序列 embedding）
-（可选）更强序列召回：GRU4Rec / SASRec（用于形成“非生成式强对照”）

7.2 语义ID v0（离散化表征）

引入 item 侧信息（caption/category 等）训练/构建 item embedding

离散化方式：

K-means（简单强基线）

VQ / RQ-VAE（更贴近 semantic ID 路线）

输出：每个 item 的 token 串（Semantic ID）

7.3 生成式召回（Gen Retrieval）

用语义ID token 序列训练 GPT-style decoder

训练目标：next-token / next-item generation

推断：生成 TopK 候选 item，与传统召回对比 Recall/NDCG

7.4 多任务精排（MTL Ranker）

结构：Shared-bottom / MMoE / PLE

目标：CTR + 完播/停留等多目标（按数据可得性选择）

指标：AUC/GAUC/LogLoss +（可选）Calibration

7.5 端到端评估与消融实验

无语义ID vs 有语义ID

无生成式 vs 生成式召回

单任务 vs 多任务

端到端 NDCG/HitRate 改善与代价（时延/参数量）