"""
Minimal Phi-3 Mini implementation for AutoKernel profiling.

Self-contained -- no transformers library needed.
Mirrors microsoft/Phi-3-mini-4k-instruct architecture:
  - hidden_size=3072, num_layers=32, num_heads=32, num_kv_heads=32
  - intermediate_size=8192, vocab_size=32064
  - RoPE (theta=10000), SwiGLU MLP, RMSNorm, LayerNorm
  - Sliding window attention (2047)

Usage:
    uv run profile.py --model models/phi3_mini.py --class-name Phi3Mini --input-shape 1,2048 --dtype float16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs_cis[None, : xq_.shape[1], None, :]
    xq_out = torch.view_as_real(xq_ * freqs).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, sliding_window: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if self.sliding_window > 0 and T > self.sliding_window:
            mask = torch.tril(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=T - self.sliding_window - 1,
            )
            mask = mask.logical_not()
        else:
            mask = None

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=True if mask is None else False
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Phi3DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_dim: int,
        sliding_window: int,
    ):
        super().__init__()
        self.self_attn = Attention(dim, n_heads, n_kv_heads, sliding_window)
        self.mlp = FeedForward(dim, hidden_dim)
        self.input_layernorm = RMSNorm(dim)
        self.post_attention_layernorm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Phi3Mini(nn.Module):
    """
    Phi-3 Mini compact variant for AutoKernel profiling (~960M params).

    Same architecture as microsoft/Phi-3-mini-4k-instruct but 8 layers
    instead of 32 to fit in 8GB VRAM alongside Ollama.

    Full-scale config: dim=3072, n_layers=32, n_heads=32, n_kv_heads=32,
                       hidden_dim=8192, vocab=32064, sliding_window=2047

    Usage:
        uv run profile.py --model models/phi3_mini.py --class-name Phi3Mini \\
            --input-shape 1,2048 --dtype float16
    """

    def __init__(
        self,
        vocab_size: int = 32064,
        dim: int = 3072,
        n_layers: int = 8,
        n_heads: int = 32,
        n_kv_heads: int = 32,
        hidden_dim: int = 8192,
        max_seq_len: int = 4096,
        sliding_window: int = 2047,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                Phi3DecoderLayer(dim, n_heads, n_kv_heads, hidden_dim, sliding_window)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim // n_heads, max_seq_len * 2),
            persistent=False,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Phi3Mini: {n_params / 1e6:.1f}M parameters")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.embed_tokens(input_ids)
        freqs = self.freqs_cis[:T]

        for layer in self.layers:
            h = layer(h, freqs)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits
