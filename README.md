# GRec_System
- **输入**：用户历史行为序列（click/cart/buy…）+ item 内容特征（category/title/brand…可选）
- **输出**：
  1) item 的 **语义ID**（离散 token 串）
  2) **生成式召回**：给定用户序列生成 TopK 候选 item（基于语义ID生成）
  3) **多任务精排**：对候选进行 CTR/CVR/CTCVR/停留时长等多目标排序
- **评估指标**：
  - 召回：Recall@K、NDCG@K、Coverage@K（可选：Novelty）
  - 精排：AUC、GAUC、LogLoss（可选：Calibration）
  - 端到端：NDCG@K / HitRate@K（最终列表）