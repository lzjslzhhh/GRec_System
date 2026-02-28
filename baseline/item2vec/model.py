from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchCosineIndex(nn.Module):
    """Thin PyTorch API wrapper for embedding normalization and similarity search."""

    def __init__(self, embeddings: torch.Tensor):
        super().__init__()
        self.register_buffer("emb", F.normalize(embeddings, p=2, dim=1))

    @torch.no_grad()
    def topk(self, query: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(query, p=2, dim=1)
        sim = query @ self.emb.T
        return torch.topk(sim, k=k, dim=1, largest=True, sorted=True)
