"""Content-adaptive watermark embedding in mel spectrograms."""

import torch
import torch.nn as nn


class MelWatermarkEncoder(nn.Module):
    def __init__(
        self,
        mel_dim: int = 80,
        watermark_dim: int = 16,
        embed_dim: int = 80,
        num_heads: int = 4,
        strength_init: float = 1.0,
        max_strength: float = 1.0,
        target_snr_db: float = 25.0,
        **kwargs,
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.watermark_dim = watermark_dim
        self.embed_dim = embed_dim
        self.max_strength = max_strength
        self.target_snr_db = target_snr_db

        self.watermark_embedding = nn.Embedding(2, embed_dim)
        self.position_embedding = nn.Embedding(watermark_dim, embed_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.mask_predictor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=(1, 1)),
            nn.Sigmoid(),
        )

        self.strength = nn.Parameter(torch.tensor(strength_init))

    def forward(self, mel: torch.Tensor, watermark_bits: torch.Tensor) -> torch.Tensor:
        B, C, T = mel.shape

        watermark_idx = watermark_bits.long()
        watermark_embed = self.watermark_embedding(watermark_idx)
        positions = torch.arange(self.watermark_dim, device=watermark_bits.device)
        position_embed = self.position_embedding(positions.unsqueeze(0).expand(B, -1))
        watermark_embed = watermark_embed + position_embed

        mel_reshaped = mel.transpose(1, 2)
        attn_output, _ = self.cross_attention(
            query=mel_reshaped,
            key=watermark_embed,
            value=watermark_embed,
        )
        raw_signal = attn_output.transpose(1, 2)

        effective_strength = torch.sigmoid(self.strength) * self.max_strength
        raw_signal = raw_signal * effective_strength

        learned_mask = self.mask_predictor(mel.unsqueeze(1)).squeeze(1)
        learned_mask = torch.clamp(learned_mask, min=0.1)

        raw_signal = raw_signal * learned_mask

        return mel + raw_signal
