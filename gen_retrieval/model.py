import math
from typing import Optional

import torch
import torch.nn as nn


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # True means "masked out" for nn.MultiheadAttention.
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


def get_last_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V], attention_mask: [B, T] (True for valid tokens)
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    return logits[torch.arange(logits.size(0), device=logits.device), lengths]


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)
        x = x + self.ff(self.ln2(x))
        return x


class GenRecDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.max_len = int(max_len)

        self.token_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(self.max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model=d_model, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab_size, bias=False)

        self.register_buffer("_causal_mask_cache", torch.empty(0, 0, dtype=torch.bool), persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask_cache.size(0) < seq_len or self._causal_mask_cache.device != device:
            self._causal_mask_cache = build_causal_mask(seq_len, device=device)
        return self._causal_mask_cache[:seq_len, :seq_len]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T], attention_mask: [B, T] bool
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"seq_len {seq_len} exceeds model.max_len {self.max_len}")

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        causal_mask = self._get_causal_mask(seq_len=seq_len, device=input_ids.device)
        key_padding_mask = ~attention_mask.bool()
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.final_ln(x)
        logits = self.head(x)
        return logits
