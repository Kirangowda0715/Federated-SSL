"""
src/models/decoder.py
---------------------
Lightweight Transformer-based MAE decoder.

Input  : Latent embeddings of visible patches + learnable mask tokens
Output : Reconstructed pixel values for ALL patches (masked + visible)
Loss   : MSE computed ONLY on masked patches (standard MAE formulation)
"""

import torch
import torch.nn as nn
import math


class MAEDecoder(nn.Module):
    """
    Lightweight Transformer decoder for Masked Autoencoder.

    Architecture:
        1. Linear projection: embed_dim → decoder_embed_dim
        2. Prepend learnable mask tokens at masked positions
        3. Add positional embeddings
        4. N Transformer blocks (decoder_depth layers)
        5. Linear head: decoder_embed_dim → patch_size² × channels

    Args:
        embed_dim       : Encoder output dimensionality (e.g., 512)
        num_patches     : Total number of patches in image (e.g., 196 for 224/16)
        patch_size      : Spatial size of each patch in pixels (e.g., 16)
        in_channels     : Number of image channels (default 3)
        decoder_embed_dim : Internal decoder width (default 256)
        decoder_depth   : Number of Transformer blocks (default 4)
        decoder_num_heads : Number of attention heads (default 8)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_patches: int = 196,
        patch_size: int = 16,
        in_channels: int = 3,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.decoder_embed_dim = decoder_embed_dim

        # 1. Project encoder embeddings to decoder dimensionality
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # 2. Learnable mask token (filled in at masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # 3. Fixed sinusoidal positional embedding (for all patches)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim),
            requires_grad=False,  # Fixed, not learned
        )
        self._init_pos_embed()

        # 4. Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=4.0,
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # 5. Prediction head: decoder dims → patch pixel values
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size * in_channels,
            bias=True,
        )

    def _init_pos_embed(self) -> None:
        """Initialize sinusoidal positional embeddings."""
        pos_embed = _get_sinusoidal_pos_embed(self.decoder_embed_dim, self.num_patches)
        self.decoder_pos_embed.data.copy_(pos_embed.unsqueeze(0))

    def forward(
        self,
        visible_embeddings: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct all patch pixel values.

        Args:
            visible_embeddings : (B, num_visible, embed_dim) — encoder output
            ids_restore        : (B, num_patches) — indices to restore original order

        Returns:
            pred : (B, num_patches, patch_size²×C) — reconstructed pixel values
        """
        B = visible_embeddings.shape[0]
        num_visible = visible_embeddings.shape[1]
        num_masked = self.num_patches - num_visible

        # Project to decoder dimensionality
        x = self.decoder_embed(visible_embeddings)   # (B, num_visible, D)

        # Expand mask tokens for all masked positions
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)  # (B, num_masked, D)

        # Concatenate visible + mask tokens, then restore original patch order
        x_full = torch.cat([x, mask_tokens], dim=1)             # (B, num_patches, D)
        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim),
        )                                                        # (B, num_patches, D)

        # Add positional embeddings
        x_full = x_full + self.decoder_pos_embed                 # (B, num_patches, D)

        # Transformer blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)
        x_full = self.decoder_norm(x_full)

        # Predict pixel values
        pred = self.decoder_pred(x_full)                         # (B, num_patches, P²×C)
        return pred


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Standard Transformer block: LN → MHA → LN → MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


# ─── Positional Embedding ────────────────────────────────────────────────────

def _get_sinusoidal_pos_embed(embed_dim: int, num_patches: int) -> torch.Tensor:
    """Generate fixed sinusoidal positional embeddings, shape (num_patches, embed_dim)."""
    pos = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32)
        * -(math.log(10000.0) / embed_dim)
    )
    emb = torch.zeros(num_patches, embed_dim)
    emb[:, 0::2] = torch.sin(pos * div)
    emb[:, 1::2] = torch.cos(pos * div[: embed_dim // 2])
    return emb
