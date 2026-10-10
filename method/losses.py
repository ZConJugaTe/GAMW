"""Focal BCE, dynamic per-bit weighting and mel-domain fidelity loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelDomainWatermarkLoss(nn.Module):
    """Focal BCE + unnormalized per-bit batch weighting + mel-domain MSE."""

    def __init__(
        self,
        lambda_decode: float = 3.0,
        lambda_audio: float = 5.0,
        focal_gamma: float = 2.0,
        bit_weight_scale: float = 10.0,
    ):
        super().__init__()
        self.lambda_decode = lambda_decode
        self.lambda_mel = lambda_audio
        self.focal_gamma = focal_gamma
        self.bit_weight_scale = bit_weight_scale

    def forward(
        self,
        detected_bits: torch.Tensor,
        target_bits: torch.Tensor,
        watermarked_mel: torch.Tensor,
        clean_mel: torch.Tensor,
    ):
        target_float = target_bits.float()

        bce = F.binary_cross_entropy(detected_bits, target_float, reduction='none')
        pt = torch.where(target_float == 1, detected_bits, 1 - detected_bits)
        focal_weight = (1 - pt) ** self.focal_gamma

        with torch.no_grad():
            predicted_bits = (detected_bits >= 0.5).float()
            bit_accuracy = (predicted_bits == target_float).float().mean(dim=0)
            bit_weight = 1.0 + (1.0 - bit_accuracy) * self.bit_weight_scale

        decode_loss = (focal_weight * bce * bit_weight.unsqueeze(0)).mean()

        mel_loss = F.mse_loss(watermarked_mel, clean_mel)
        total_loss = self.lambda_decode * decode_loss + self.lambda_mel * mel_loss

        ber = (predicted_bits != target_float).float().mean().item()

        return {
            'total_loss': total_loss,
            'decode_loss': decode_loss,
            'audio_loss': mel_loss,
            'ber': ber,
            'bit_accuracy': bit_accuracy,
            'bit_weight': bit_weight,
        }
