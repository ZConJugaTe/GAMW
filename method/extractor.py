"""Extract watermark bits from one-second, 24 kHz mono waveforms."""

import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn


# License for the mel preprocessing helpers (Matcha-TTS):
#
# MIT License
#
# Copyright (c) 2023 Shivam Mehta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class Detector(nn.Module):
    """Recover bit probabilities with a shared-query Transformer decoder."""

    def __init__(
        self,
        device='cuda',
        watermark_length=16,
        feat_dim=256,
        num_decoder_layers=5,
        nhead=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.device = device
        self.watermark_length = watermark_length
        self.expected_length_24k = 24000
        self.feat_dim = feat_dim

        self.backbone_2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.feature_proj = nn.Sequential(
            nn.Conv1d(128 * 10, feat_dim, kernel_size=1),
            nn.BatchNorm1d(feat_dim),
            nn.GELU(),
        )

        self.pos_enc = nn.Parameter(torch.randn(1, 50, feat_dim) * 0.02)
        self.shared_query = nn.Parameter(torch.randn(1, feat_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.output_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, watermark_length),
        )

        self.to(device)

    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        return mel_spectrogram(
            audio,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000,
            center=False,
        )

    def extract_features(self, audio_24k: torch.Tensor) -> torch.Tensor:
        mel = self.audio_to_mel(audio_24k)
        mel_2d = mel.unsqueeze(1)
        out_2d = self.backbone_2d(mel_2d)

        bsz, channels, freq_dim, time_dim = out_2d.shape
        out_1d = out_2d.reshape(bsz, channels * freq_dim, time_dim)

        features = self.feature_proj(out_1d)
        features = features.transpose(1, 2)
        features = features + self.pos_enc

        return features

    def forward(self, audio_24k: torch.Tensor) -> torch.Tensor:
        bsz, _ = audio_24k.shape
        features = self.extract_features(audio_24k)

        query = self.shared_query.unsqueeze(0).expand(bsz, 1, -1)
        decoded = self.transformer_decoder(
            tgt=query,
            memory=features,
        )

        decoded = decoded.squeeze(1)
        logits = self.output_head(decoded)
        return torch.sigmoid(logits)

    def detect_from_audio(self, audio_24k: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return (self.forward(audio_24k) >= 0.5).float()
