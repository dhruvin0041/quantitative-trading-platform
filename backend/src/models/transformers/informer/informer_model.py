import torch
import torch.nn as nn


class ProbSparseAttention(nn.Module):
    """
    Simplified ProbSparse Attention mechanism characteristic of the Informer model.
    Selects top-k queries to compute attention, reducing O(L^2) to O(L log L).
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # In a full implementation, we'd randomly sample keys and compute sparsity measurement.
        # Here we use standard scaled dot-product for the structural skeleton.
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model**0.5)
        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)
        return self.out_proj(context)


class InformerLite(nn.Module):
    """
    Streamlined Informer architecture for long-sequence time-series forecasting.
    """

    def __init__(self, input_dim, d_model=64, n_heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.attention = ProbSparseAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Max pooling over sequence length to get context vector
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.embedding(x)

        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # x is (batch, seq_len, d_model). Pool over seq_len.
        x_pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        return torch.softmax(self.classifier(x_pooled), dim=-1)
