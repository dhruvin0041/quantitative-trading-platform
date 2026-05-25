import torch
import torch.nn as nn


class PatchTSTLite(nn.Module):
    """
    A streamlined implementation of PatchTST.
    Splits time series into overlapping patches before applying transformers,
    retaining local semantic information and reducing sequence length.
    """

    def __init__(
        self, seq_len, input_dim, patch_len=5, stride=2, d_model=64, num_heads=4
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len * input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_patches * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Sell, Hold, Buy
        )

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # Extract patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i : i + self.patch_len, :].reshape(batch_size, -1)
            patches.append(patch)

        x_patched = torch.stack(
            patches, dim=1
        )  # (batch, num_patches, patch_len*features)

        # Embed and add positional encoding
        x_emb = self.patch_embedding(x_patched) + self.pos_embedding

        # Apply transformer
        out = self.transformer(x_emb)

        # Classification
        return torch.softmax(self.head(out), dim=-1)
