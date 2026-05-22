import torch
import torch.nn as pd
import torch.nn as nn

class TemporalFusionTransformerLite(nn.Module):
    """
    A streamlined implementation of the Temporal Fusion Transformer (TFT).
    Combines variable selection networks and temporal self-attention.
    """
    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Variable Selection (simplified)
        self.grn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Temporal Self-Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Position-wise Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, 3) # 3 classes: Sell, Hold, Buy

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x_proj = self.input_projection(x)
        
        # Variable Selection Gating
        gated = self.grn(x_proj)
        
        # Attention
        attn_out, _ = self.attention(gated, gated, gated)
        x_attn = self.norm1(gated + attn_out)
        
        # Feed-Forward
        ffn_out = self.ffn(x_attn)
        x_out = self.norm2(x_attn + ffn_out)
        
        # Take the last sequence step for classification
        last_step = x_out[:, -1, :]
        return torch.softmax(self.output_layer(last_step), dim=-1)
