"""
批量化训练水印编码器 - 直接使用预保存Mel版本 + DDP 并行 + 数据增强

修复说明：
  1. 原版 Pipeline 把 detector 放在 nn.Module 内，导致 DDP 报错。已移出单独包装。
  2. 修复 Encodec 批处理时因长度一致导致 processor 缺失 `padding_mask` 触发 KeyError 的隐患。
  3. 修复 Augmenter 内静默跳过报错的问题，加入强显报警。
  4. 植入 STE (直通估计器)，打通 Encodec/MP3 等不可导攻击的梯度回传。

结构：
  self.pipeline (DDP)          : WatermarkPipeline(encoder + hift)
  self.watermark_detector (DDP): Detector
"""

import argparse
import csv
import logging
import os

# torchrun 会在启动每个子进程时设置 LOCAL_RANK。必须在导入 torch、
# onnxruntime 等可能初始化 CUDA 的库之前，将该进程隔离到对应物理卡，
# 否则所有 ONNX Runtime 会话都会默认占用第一张可见卡。
_watermark_gpu_ids = [
    gpu_id.strip()
    for gpu_id in os.environ.get('WATERMARK_GPU_IDS', '').split(',')
    if gpu_id.strip()
]
_local_rank = os.environ.get('LOCAL_RANK')
if _watermark_gpu_ids and _local_rank is not None:
    _local_rank_index = int(_local_rank)
    if _local_rank_index >= len(_watermark_gpu_ids):
        raise RuntimeError(
            f'LOCAL_RANK={_local_rank_index} exceeds WATERMARK_GPU_IDS='
            f'{_watermark_gpu_ids}'
        )
    os.environ['CUDA_VISIBLE_DEVICES'] = _watermark_gpu_ids[_local_rank_index]

# ====== 新增：强制优先本地缓存，避免联网超时 ======
os.environ.pop("HF_ENDPOINT", None)  # 去掉 hf-mirror，避免继续请求镜像
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 你的 EnCodec 本地缓存目录
ENCODEC_LOCAL_DIR = "/home/zhangyihang/.cache/huggingface/hub/models--facebook--encodec_24khz"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
import json
from torch.utils.data import Dataset
from pathlib import Path
import random
import gc


import sys
from cosyvoice.watermark.audio2mel_detector_32_new4_shared import Detector
from cosyvoice.watermark.new_losses_2 import MelDomainWatermarkLoss
from cosyvoice.watermark.mel_watermark_encoder_new7 import MelWatermarkEncoder

from hyperpyyaml import load_hyperpyyaml
import torch.nn.functional as F

sys.path.append('third_party/Matcha-TTS')
sys.path.append('/home/zhangyihang/workspace/CosyVoice/third_party/Matcha-TTS')

from attacks import (
    pert_gaussian_noise,
    pert_quantization,
    pert_lowpass,
    pert_highpass,
    pert_echo,
    pert_mp3,
    pert_smooth,
    pert_background_noise,
)

import builtins

_original_print = builtins.print

def _filtered_print(*args, **kwargs):
    # 将要打印的内容拼成字符串
    text = " ".join(str(arg) for arg in args)
    # 如果包含这些烦人的关键字，直接默默丢弃，不打印！
    if "max value is" in text or "min value is" in text:
        return
    # 否则，正常打印
    _original_print(*args, **kwargs)

# 替换全局的 print 函数
builtins.print = _filtered_print

# ============================================================
# ✅ Pipeline：只含 encoder + hift，不含 detector
# ============================================================
class WatermarkPipeline(nn.Module):
    def __init__(self, watermark_encoder, hift):
        super().__init__()
        self.watermark_encoder = watermark_encoder
        self.hift = hift

    def forward(self, mel_feat, watermark_bits, use_bit_inverse=True):
        device = mel_feat.device

        if use_bit_inverse:
            mel_feat       = torch.cat([mel_feat,       mel_feat          ], dim=0)
            watermark_bits = torch.cat([watermark_bits, 1 - watermark_bits], dim=0)

        watermarked_mel = self.watermark_encoder(mel_feat, watermark_bits)

        batch_dict_wm = {'speech_feat': watermarked_mel.transpose(1, 2)}
        audio_wm, _   = self.hift.forward(batch_dict_wm, device)

        return audio_wm, watermarked_mel, mel_feat, watermark_bits


# # ============================================================
# # EnCodec 模型懒加载（新增）
# # ============================================================
# _encodec_model     = None
# _encodec_processor = None

# def _get_encodec(device='cuda'):
#     """懒加载并缓存 EnCodec 模型与处理器，模型放在指定 device 上。"""
#     global _encodec_model, _encodec_processor
#     if _encodec_model is None:
#         import warnings
#         from transformers import EncodecModel, AutoProcessor
#         warnings.filterwarnings(
#             "ignore",
#             message=".*Could not find image processor class.*feature_extractor_type.*"
#         )
#         _encodec_model     = EncodecModel.from_pretrained("facebook/encodec_24khz")
#         _encodec_model     = _encodec_model.to(device).eval()
#         _encodec_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
#         print(f"✓ EnCodec model loaded on {device}")
#     return _encodec_model, _encodec_processor

# ============================================================
# EnCodec 模型懒加载
# 修改：优先从本地 Hugging Face 缓存离线加载，并支持 device 切换
# ============================================================
_encodec_model = None
_encodec_processor = None
_encodec_device = None


def _resolve_encodec_local_path() -> str:
    """
    返回 Hugging Face 缓存中的可直接 from_pretrained 的本地目录。
    优先使用 snapshots 下的具体快照目录；如果没有，再退回最外层目录。
    """
    base = Path(ENCODEC_LOCAL_DIR)
    if not base.exists():
        raise FileNotFoundError(
            f"EnCodec local cache not found: {ENCODEC_LOCAL_DIR}"
        )

    snapshots_dir = base / "snapshots"
    if snapshots_dir.exists():
        snapshot_subdirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if snapshot_subdirs:
            snapshot_subdirs = sorted(
                snapshot_subdirs,
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            return str(snapshot_subdirs[0])

    return str(base)


def _get_encodec(device='cuda'):
    """懒加载并缓存 EnCodec 模型与处理器，强制从本地缓存加载。"""
    global _encodec_model, _encodec_processor, _encodec_device

    if _encodec_model is None:
        import warnings
        from transformers import EncodecModel, AutoProcessor
        warnings.filterwarnings(
            "ignore",
            message=".*Could not find image processor class.*feature_extractor_type.*"
        )

        local_model_dir = _resolve_encodec_local_path()
        print(f"Loading EnCodec from local cache: {local_model_dir}")

        _encodec_model = EncodecModel.from_pretrained(
            local_model_dir,
            local_files_only=True,
        )
        _encodec_model = _encodec_model.to(device).eval()

        _encodec_processor = AutoProcessor.from_pretrained(
            local_model_dir,
            local_files_only=True,
        )

        _encodec_device = device
        print(f"✓ EnCodec model loaded from local cache on {device}")

    elif _encodec_device != device:
        _encodec_model = _encodec_model.to(device).eval()
        _encodec_device = device

    return _encodec_model, _encodec_processor


# ============================================================
# ✅ 音频增强器（GPU 批量版）
# ============================================================
class AudioAugmenter:
    def __init__(
        self,
        sample_rate: int = 24000,
        aug_prob: float = 0.8,
        max_attacks: int = 2,
        warmup_epochs: int = 5,
        use_mp3: bool = False,
        use_echo: bool = True,
        noise_path: str = None,
        allowed_attacks: list = None,
        rank: int = 0,
    ):
        self.sample_rate   = sample_rate
        self.aug_prob      = aug_prob
        self.max_attacks   = max_attacks
        self.warmup_epochs = warmup_epochs

        # 预加载背景噪声
        self._bg_noise = None
        if noise_path is not None:
            try:
                import torchaudio
                noise, sr = torchaudio.load(noise_path)
                if sr != sample_rate:
                    noise = torchaudio.functional.resample(noise, sr, sample_rate)
                self._bg_noise = noise.mean(0)
                if rank == 0:
                    print(f"  背景噪声预加载: {noise_path}")
            except Exception as e:
                if rank == 0:
                    print(f"  ⚠️ 背景噪声加载失败: {e}")

        # 攻击池
        self.attack_pool = []

        self.attack_pool.append((
            'gaussian_noise',
            lambda a, d, snr_db: self._gaussian_noise(a, snr_db),
            lambda: {'snr_db': random.choice([10, 20, 30])},
        ))
        self.attack_pool.append((
            'lowpass',
            lambda a, d, cutoff_ratio: self._lowpass(a, cutoff_ratio, d),
            lambda: {'cutoff_ratio': 0.2},
        ))
        self.attack_pool.append((
            'highpass',
            lambda a, d, cutoff_ratio: self._highpass(a, cutoff_ratio, d),
            lambda: {'cutoff_ratio': 0.05},
        ))
        self.attack_pool.append((
            'quantization',
            lambda a, d, quantization_levels: self._quantization(a, quantization_levels),
            lambda: {'quantization_levels': random.choice([64, 128, 256, 512])},
        ))
        self.attack_pool.append((
            'smooth',
            lambda a, d, window_size: self._smooth(a, window_size, d),
            lambda: {'window_size': random.choice([6, 10, 14, 18])},
        ))
        if use_echo:
            self.attack_pool.append((
                'echo',
                lambda a, d, volume, duration, sample_rate: self._echo(a, volume, duration, d),
                lambda: {
                    'volume': random.uniform(0.1, 0.5),
                    'duration': random.uniform(0.1, 0.5),
                    'sample_rate': self.sample_rate,
                },
            ))
        if self._bg_noise is not None:
            self.attack_pool.append((
                'background_noise',
                lambda a, d, snr_db: self._background_noise(a, snr_db, d),
                lambda: {'snr_db': random.choice([10, 20, 30])},
            ))
        if use_mp3:
            self.attack_pool.append((
                'mp3',
                lambda a, d, bitrate, sample_rate: self._mp3_cpu(a, bitrate),
                lambda: {
                    'bitrate': random.choice([32, 48, 64, 96, 128]),
                    'sample_rate': self.sample_rate,
                },
            ))

        self.attack_pool.append((
            'encodec',
            None,
            lambda: {'bandwidth': 24.0},
        ))
        self.attack_pool.append((
            'time_stretch',
            lambda a, d, speed_factor: self._time_stretch(a, speed_factor),
            lambda: {'speed_factor': random.choice([0.9, 1.1])},
        ))

        # 按白名单过滤
        if allowed_attacks is not None:
            allowed_set = set(allowed_attacks)
            self.attack_pool = [(n, f, p) for n, f, p in self.attack_pool if n in allowed_set]

        if rank == 0:
            print(f"✅ AudioAugmenter (GPU加速版): {[n for n, _, _ in self.attack_pool]}")
            print(f"   aug_prob={aug_prob}, max_attacks={max_attacks}, warmup_epochs={warmup_epochs}")

    # ---- GPU 攻击实现 ------------------------------------------------

    @staticmethod
    def _gaussian_noise(audio: torch.Tensor, snr_db: float) -> torch.Tensor:
        sig_pow = audio.pow(2).mean(dim=-1, keepdim=True)
        snr_lin = 10 ** (snr_db / 10)
        noise   = torch.randn_like(audio) * (sig_pow / snr_lin).sqrt()
        return audio + noise

    @staticmethod
    def _quantization(audio: torch.Tensor, quantization_levels: int) -> torch.Tensor:
        lo  = audio.min(dim=-1, keepdim=True).values
        hi  = audio.max(dim=-1, keepdim=True).values
        rng = (hi - lo).clamp(min=1e-8)
        q   = torch.round((audio - lo) / rng * (quantization_levels - 1)) / (quantization_levels - 1)
        return q * rng + lo

    @staticmethod
    def _sinc_kernel(cutoff: float, size: int, device: torch.device) -> torch.Tensor:
        half = size // 2
        t    = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
        h    = torch.where(t == 0,
                           torch.tensor(2.0 * cutoff, device=device),
                           torch.sin(2 * torch.pi * cutoff * t) / (torch.pi * t))
        win  = 0.5 - 0.5 * torch.cos(
                   2 * torch.pi * torch.arange(size + 1, device=device) / size)[:size]
        h = h * win
        return (h / h.sum()).view(1, 1, -1)

    def _lowpass(self, audio: torch.Tensor, cutoff_ratio: float, device: torch.device) -> torch.Tensor:
        k   = self._sinc_kernel(cutoff_ratio, 101, device)
        out = F.conv1d(audio.unsqueeze(1), k, padding=50).squeeze(1)
        return out[..., :audio.shape[-1]]

    def _highpass(self, audio: torch.Tensor, cutoff_ratio: float, device: torch.device) -> torch.Tensor:
        return audio - self._lowpass(audio, cutoff_ratio, device)

    def _smooth(self, audio: torch.Tensor, window_size: int, device: torch.device) -> torch.Tensor:
        k   = torch.ones(1, 1, window_size, device=device) / window_size
        out = F.conv1d(audio.unsqueeze(1), k, padding=window_size // 2).squeeze(1)
        T   = audio.shape[-1]
        if out.shape[-1] > T:   out = out[..., :T]
        elif out.shape[-1] < T: out = F.pad(out, (0, T - out.shape[-1]))
        return out

    def _echo(self, audio: torch.Tensor, volume: float, duration: float,
              device: torch.device) -> torch.Tensor:
        delay      = int(self.sample_rate * duration)
        echo_audio = torch.zeros_like(audio)
        if delay < audio.shape[-1]:
            echo_audio[..., delay:] = audio[..., :-delay] * volume
        out    = audio + echo_audio
        mx_in  = audio.abs().max(dim=-1, keepdim=True).values.clamp(1e-8)
        mx_out = out.abs().max(dim=-1, keepdim=True).values.clamp(1e-8)
        return out / mx_out * mx_in

    def _background_noise(self, audio: torch.Tensor, snr_db: float,
                           device: torch.device) -> torch.Tensor:
        noise = self._bg_noise.to(device)
        T     = audio.shape[-1]
        if noise.shape[0] < T:
            noise = noise.repeat(T // noise.shape[0] + 1)
        noise   = noise[:T].unsqueeze(0).expand_as(audio)
        sig_pow = audio.pow(2).mean(-1, keepdim=True)
        noi_pow = noise.pow(2).mean(-1, keepdim=True).clamp(1e-10)
        scale   = (sig_pow / (10 ** (snr_db / 10) * noi_pow)).sqrt()
        return audio + noise * scale

    def _mp3_cpu(self, audio: torch.Tensor, bitrate: int) -> torch.Tensor:
        import tempfile, os, uuid
        try:
            import pydub, librosa
            from audiomentations.core.utils import convert_float_samples_to_int16
            device  = audio.device
            cpu     = audio.cpu()
            results = []
            for i in range(cpu.shape[0]):
                # 💡 修复：加上 .detach()，显式断开计算图
                w   = cpu[i].detach().numpy() 
                s   = convert_float_samples_to_int16(w)
                seg = pydub.AudioSegment(s.tobytes(), frame_rate=self.sample_rate,
                                         sample_width=2, channels=1)
                tmp = os.path.join(tempfile.gettempdir(),
                                   f"_mp3_{uuid.uuid4().hex[:8]}.mp3")
                seg.export(tmp, bitrate=f"{bitrate}k").close()
                d, _ = librosa.load(tmp, sr=self.sample_rate, mono=True)
                os.unlink(tmp)
                d = torch.from_numpy(d).float()
                T = cpu.shape[-1]
                if d.shape[0] < T:   d = F.pad(d, (0, T - d.shape[0]))
                elif d.shape[0] > T: d = d[:T]
                results.append(d)
            return torch.stack(results).to(device)
        except Exception as e:
            logging.debug(f"[Augmenter] mp3 failed: {e}")
            return audio

    # ← 新增：encodec 批处理实现
    def _encodec(self, audio: torch.Tensor, device: torch.device,
                 bandwidth: float) -> torch.Tensor:
        """
        整个 batch [B, T] 一次 encode/decode，返回 [B, T] 在原 device 上。
        """
        model_ec, processor_ec = _get_encodec(device=str(device))
        orig_len   = audio.shape[-1]
        batch_size = audio.shape[0]

        audio_cpu      = audio.cpu().float()
        
        # 💡 修复：加上 .detach()，安全地转为 numpy
        raw_audio_list = [audio_cpu[i].detach().numpy() for i in range(batch_size)]

        inputs = processor_ec(
            raw_audio=raw_audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).to(device)

        # 安全提取 padding_mask
        pad_mask = inputs.get("padding_mask", None)

        with torch.no_grad():
            encoder_outputs = model_ec.encode(
                inputs["input_values"],
                pad_mask, 
                bandwidth,
            )
            audio_values = model_ec.decode(
                encoder_outputs.audio_codes,
                encoder_outputs.audio_scales,
                pad_mask, 
            )[0]  # [B, 1, T']

        out = audio_values.squeeze(1)  # [B, T']

        # 对齐长度
        if out.shape[-1] < orig_len:
            out = F.pad(out, (0, orig_len - out.shape[-1]))
        else:
            out = out[:, :orig_len]

        return out.to(device)

    def _time_stretch(self, audio: torch.Tensor, speed_factor: float) -> torch.Tensor:
        import librosa
        device = audio.device
        results = []
        for i in range(audio.shape[0]):
            wav_np = audio[i].detach().cpu().numpy()
            stretched = librosa.effects.time_stretch(wav_np, rate=speed_factor)
            stretched = torch.from_numpy(stretched).float()
            T = audio.shape[-1]
            if stretched.shape[-1] < T:
                stretched = F.pad(stretched, (0, T - stretched.shape[-1]))
            else:
                stretched = stretched[:T]
            results.append(stretched)
        return torch.stack(results).to(device)

    # ---- 对外接口 ----------------------------------------------------

    def __call__(self, audio_batch: torch.Tensor, epoch: int = 999) -> torch.Tensor:
        if epoch <= self.warmup_epochs:
            return audio_batch

        device = audio_batch.device
        B      = audio_batch.shape[0]

        mask = (torch.rand(B, device=device) <= self.aug_prob)
        if not mask.any():
            return audio_batch

        n        = random.randint(1, min(self.max_attacks, len(self.attack_pool)))
        selected = random.sample(self.attack_pool, n)

        result = audio_batch.clone()
        for name, batch_fn, param_fn in selected:
            try:
                kwargs = param_fn()
                if name == 'encodec':
                    # encodec 单独走批处理分支，不用 batch_fn
                    augmented = self._encodec(result, device, **kwargs)
                else:
                    augmented = batch_fn(result, device, **kwargs)

                # =======================================================
                # 💡 核心修复 2：STE (直通估计器) 
                # =======================================================
                if name in ['encodec', 'mp3', 'quantization','time_stretch'] or not augmented.requires_grad:
                    augmented = result + (augmented - result).detach()

                result = torch.where(mask.unsqueeze(-1), augmented, result)
            except Exception as e:
                # 💡 核心修复 3：拆掉静音报错器，强制打印所有异常
                print(f"\n🚨🚨🚨 [Augmenter 致命报错] 攻击 '{name}' 失败！原因: {e} 🚨🚨🚨\n")
                import logging
                logging.error(f"[Augmenter] {name} failed: {e}, skipping")

        return result

    def describe(self) -> str:
        return f"攻击池({len(self.attack_pool)}种, GPU加速): {', '.join(n for n, _, _ in self.attack_pool)}"


# ============================================================
# 工具函数
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def generate_random_watermark(batch_size, watermark_length=32):
    return torch.randint(0, 2, (batch_size, watermark_length))


# ============================================================
# 参数解析
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(description='Train Watermark (DDP + Aug, detector separated)')
    parser.add_argument('--cosyvoice_model_dir', required=True)
    parser.add_argument('--config', default='conf/train_watermark.yaml')
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--cv_data', '--cvdata', dest='cv_data', required=True)
    parser.add_argument('--mel_dir', default='/data/zhangyihang/LJSpeech-1.1/watermark_data/mel')
    parser.add_argument(
        '--cv_mel_dir', '--cvmeldir', dest='cv_mel_dir', default=None,
        help='CV集的mel目录；未设置时沿用--mel_dir。')
    parser.add_argument(
        '--validate_step', '--validatestep', dest='validate_step', type=int, default=0,
        help='第一个epoch内每隔多少训练step验证一次ACC；0表示禁用step验证。')
    parser.add_argument(
        '--disable_validation', action='store_true',
        help='完全禁用验证，包括step验证和每个epoch结束后的验证。')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--tensorboard_dir', default='tensorboard')
    parser.add_argument('--batch_size_per_gpu', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_samples', type=int, default=None)
    parser.add_argument('--val_samples', type=int, default=None)
    parser.add_argument('--lambda_decode', type=float, default=1.0)
    parser.add_argument('--lambda_audio', type=float, default=0.5)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--watermark_dim', type=int, default=16)
    parser.add_argument('--audio_output_dir', default='output_audio')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', type=str, default=None)

    # 在 --resume_path 那行后面加
    parser.add_argument('--resume_lr', type=float, default=None)
    parser.add_argument('--resume_epoch', type=int, default=0)

    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    parser.add_argument('--master_port', type=str, default='29500')
    # 数据增强
    parser.add_argument('--aug_prob', type=float, default=0.8)
    parser.add_argument('--aug_max_attacks', type=int, default=2)
    parser.add_argument('--aug_warmup_epochs', type=int, default=5)
    parser.add_argument('--aug_sample_rate', type=int, default=24000)
    parser.add_argument('--disable_mp3_aug', action='store_true')
    parser.add_argument('--disable_echo_aug', action='store_true')
    parser.add_argument('--noise_dir', type=str, default=None)
    parser.add_argument('--aug_attacks', type=str, default=None,
                        help='逗号分隔的攻击名列表，如 gaussian_noise,lowpass,encodec。留空则使用全部攻击。')
    return parser.parse_args()


# ============================================================
# Dataset / Collate
# ============================================================
class MelWatermarkDataset(Dataset):
    def __init__(self, data_list_file, mel_dir, rank=0):
        self.samples = []
        self.mel_dir = Path(mel_dir)
        if rank == 0:
            print(f"Scanning dataset from {data_list_file}...")
        with open(data_list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        iterator = tqdm(lines, desc="Loading mel samples") if rank == 0 else lines
        for line in iterator:
            data = json.loads(line.strip())
            if 'utt_id' not in data:
                continue
            mel_path = self.mel_dir / f"{data['utt_id']}.pt"
            if not mel_path.exists():
                continue
            self.samples.append({
                'utt_id': data['utt_id'],
                'mel_path': str(mel_path),
                'text': data.get('text', ''),
                'spk_id': data.get('spk_id', 0)
            })
        if rank == 0:
            print(f"✓ Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'utt_id': s['utt_id'],
            'mel_feat': torch.load(s['mel_path']),
            'text': s['text'],
            'spk_id': s['spk_id'],
        }


def collate_fn_mel_batch(batch):
    min_T = min(item['mel_feat'].shape[-1] for item in batch)
    mel_list = []
    for item in batch:
        mel = item['mel_feat']
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel_list.append(mel[:, :, :min_T])
    return {
        'utt_ids': [item['utt_id'] for item in batch],
        'mel_feat': torch.cat(mel_list, dim=0),
        'text': [item['text'] for item in batch],
        'spk_ids': [item['spk_id'] for item in batch]
    }


# ============================================================
# Trainer（与原版完全一致，无改动）
# ============================================================
class WatermarkTrainer:
    def __init__(self, args, rank, world_size):
        self.args       = args
        self.rank       = rank
        self.world_size = world_size
        # worker_fn为每个进程仅暴露一张物理GPU，因此进程内统一使用cuda:0。
        self.device     = torch.device('cuda:0')
        torch.cuda.set_device(0)

        if rank == 0:
            print(f"🖥️  DDP: {world_size} GPUs, "
                  f"batch_per_gpu={args.batch_size_per_gpu}, "
                  f"total_batch={args.batch_size_per_gpu * world_size}")
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.tensorboard_dir, exist_ok=True)

        self.config = {}
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                self.config = load_hyperpyyaml(f)

        self._init_models()
        self._init_losses()
        self._init_optimizer()
        self._init_augmenter()

        self.writer = None
        if rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(args.tensorboard_dir)

        self.global_step = 0
        self.audio_output_dir = args.audio_output_dir
        if rank == 0:
            os.makedirs(self.audio_output_dir, exist_ok=True)

    # ----------------------------------------------------------
    def _init_augmenter(self):
        allowed = [x.strip() for x in self.args.aug_attacks.split(',')] \
                  if self.args.aug_attacks else None
        self.augmenter = AudioAugmenter(
            sample_rate=self.args.aug_sample_rate,
            aug_prob=self.args.aug_prob,
            max_attacks=self.args.aug_max_attacks,
            warmup_epochs=self.args.aug_warmup_epochs,
            use_mp3=not self.args.disable_mp3_aug,
            use_echo=not self.args.disable_echo_aug,
            noise_path=self.args.noise_dir,
            allowed_attacks=allowed,
            rank=self.rank,
        )
        if self.rank == 0:
            logging.info(f"✅ {self.augmenter.describe()}")

    # ----------------------------------------------------------
    def _init_models(self):
        from cosyvoice.cli.cosyvoice import CosyVoice2

        print(f"[Rank {self.rank}] Loading CosyVoice2 "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

        cosyvoice = CosyVoice2(
            model_dir=self.args.cosyvoice_model_dir,
            load_jit=False, load_trt=False, load_vllm=False, fp16=self.args.fp16
        )
        hift = cosyvoice.model.hift
        assert hasattr(hift, 'trainable_inference'), "hift.trainable_inference not found!"

        cosyvoice.model.hift = None
        for attr in ('llm', 'flow', 'frontend'):
            if hasattr(cosyvoice.model, attr):
                delattr(cosyvoice.model, attr)
        del cosyvoice
        gc.collect()
        torch.cuda.empty_cache()

        for param in hift.parameters():
            param.requires_grad = False
        hift.eval()

        self.watermark_encoder = MelWatermarkEncoder(
            mel_dim=80, watermark_dim=self.args.watermark_dim,
            embed_dim=80, strength_init=1.0, max_strength=1.0
        ).to(self.device)
        for p in self.watermark_encoder.parameters():
            p.requires_grad = True
        self.watermark_encoder.train()

        self.watermark_detector = Detector(
            device=self.device, watermark_length=self.args.watermark_dim,
        ).to(self.device)
        for p in self.watermark_detector.parameters():
            p.requires_grad = True
        self.watermark_detector.train()

        if self.args.resume and self.args.resume_path:
            ckpt = torch.load(self.args.resume_path, map_location=self.device)
            self.watermark_encoder.load_state_dict(ckpt['encoder_state_dict'])
            self.watermark_detector.load_state_dict(ckpt['detector_state_dict'])
            if self.rank == 0:
                logging.info(f"✓ Loaded from epoch {ckpt.get('epoch', '?')}")

        self.pipeline = WatermarkPipeline(
            watermark_encoder=self.watermark_encoder,
            hift=hift,
        ).to(self.device)
        self.pipeline = DDP(
            self.pipeline,
            device_ids=[0], output_device=0,
            find_unused_parameters=False,
        )

        self.watermark_detector = DDP(
            self.watermark_detector,
            device_ids=[0], output_device=0,
            find_unused_parameters=False,
        )

        if self.rank == 0:
            self._print_trainable_parameters()
        dist.barrier()

    # ----------------------------------------------------------
    def _print_trainable_parameters(self):
        enc  = self.pipeline.module.watermark_encoder
        det  = self.watermark_detector.module
        hift = self.pipeline.module.hift

        print("\n" + "=" * 80)
        print("📊 MODEL PARAMETERS SUMMARY")
        print("=" * 80)
        total_p = total_t = 0
        for name, model in [('Watermark Encoder', enc),
                             ('Watermark Detector', det),
                             ('HiFiGAN (frozen)', hift)]:
            p = sum(x.numel() for x in model.parameters())
            t = sum(x.numel() for x in model.parameters() if x.requires_grad)
            total_p += p; total_t += t
            print(f"\n📦 {name}:")
            print(f"  Total: {p:,} ({p/1e6:.2f}M)  Trainable: {t:,} ({t/1e6:.2f}M)  "
                  f"Ratio: {t/p*100:.1f}%")
        print(f"\n🎯 ALL: {total_p:,}  Trainable: {total_t:,}")
        print("=" * 80 + "\n")

    # ----------------------------------------------------------
    def _init_losses(self):
        self.criterion = MelDomainWatermarkLoss(
            lambda_decode=self.args.lambda_decode,
            lambda_audio=self.args.lambda_audio,
            focal_gamma=2.0,
        )

    # ----------------------------------------------------------
    def _init_optimizer(self):
        self.optimizer = optim.AdamW(
            list(self.pipeline.module.watermark_encoder.parameters()) +
            list(self.watermark_detector.module.parameters()),
            lr=self.args.lr,
            weight_decay=1e-4,
        )

        # ↓↓↓ 新增：resume时恢复optimizer状态并冻结LR ↓↓↓
        if self.args.resume and self.args.resume_path:
            # 预实验
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.num_epochs, eta_min=1e-6
            )
            # ckpt = torch.load(self.args.resume_path, map_location=self.device)
            # if 'optimizer_state_dict' in ckpt:
            #     self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            #     if self.rank == 0:
            #         logging.info("✓ Optimizer state restored")
            # resume_lr = self.args.resume_lr or self.args.lr
            # for pg in self.optimizer.param_groups:
            #     pg['lr'] = resume_lr
            # if self.rank == 0:
            #     logging.info(f"✓ LR locked at {resume_lr}")
            # # 冻结LR，不再衰减
            # self.scheduler = optim.lr_scheduler.ConstantLR(
            #     self.optimizer, factor=1.0, total_iters=99999
            # )
        else:
            # 预实验
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.num_epochs, eta_min=1e-6
                # 之前一直是1e-6,transformerdecoder 4e-6,cross 5e-6
            )

    # ----------------------------------------------------------
    def save_comparison_audio(self, batch, watermark_bits, epoch, batch_idx):
        if self.rank != 0:
            return
        import torchaudio
        compare_dir = os.path.join(self.audio_output_dir, f"epoch_{epoch}")
        os.makedirs(compare_dir, exist_ok=True)

        mel_feat = batch['mel_feat'][0:1].to(self.device)
        utt_id   = batch['utt_ids'][0]
        wm_bits  = watermark_bits[0:1].to(self.device)
        enc      = self.pipeline.module.watermark_encoder
        hift     = self.pipeline.module.hift

        with torch.no_grad():
            audio_orig, _ = hift.trainable_inference(
                speech_feat=mel_feat,
                cache_source=torch.zeros(1, 1, 0).to(self.device)
            )
            audio_orig = audio_orig.squeeze()
            wm_mel     = enc(mel_feat, wm_bits)
            audio_wm, _ = hift.trainable_inference(
                speech_feat=wm_mel,
                cache_source=torch.zeros(1, 1, 0).to(self.device)
            )
            audio_wm = audio_wm.squeeze()

            mel_diff_mean = (wm_mel - mel_feat).abs().mean().item()
            mel_diff_max  = (wm_mel - mel_feat).abs().max().item()
            min_len = min(audio_orig.shape[0], audio_wm.shape[0])
            audio_diff = (audio_wm[:min_len] - audio_orig[:min_len]).abs()

            sr = self.args.aug_sample_rate
            torchaudio.save(os.path.join(compare_dir, f"{utt_id}_original.wav"),
                            audio_orig.unsqueeze(0).cpu(), sr)
            torchaudio.save(os.path.join(compare_dir, f"{utt_id}_watermarked.wav"),
                            audio_wm.unsqueeze(0).cpu(), sr)
            torchaudio.save(os.path.join(compare_dir, f"{utt_id}_diff_x10.wav"),
                            ((audio_wm[:min_len] - audio_orig[:min_len]) * 10).unsqueeze(0).cpu(), sr)

            info = {'utt_id': utt_id, 'watermark_bits': wm_bits.cpu().tolist(),
                    'mel_diff_mean': mel_diff_mean, 'mel_diff_max': mel_diff_max,
                    'audio_diff_mean': audio_diff.mean().item(),
                    'audio_diff_max': audio_diff.max().item(), 'audio_length': min_len}
            with open(os.path.join(compare_dir, f"{utt_id}_info.json"), 'w') as fp:
                json.dump(info, fp, indent=2)

            print(f"\n[对比诊断 Epoch {epoch}] {utt_id} | "
                  f"mel_diff={mel_diff_mean:.4f} | audio_diff={audio_diff.mean().item():.4f}")

    # ----------------------------------------------------------
    def train_one_epoch(self, dataloader, epoch, val_loader=None):
        self.pipeline.train()
        self.pipeline.module.hift.eval()
        self.watermark_detector.train()

        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        total_loss = total_ber = n_batches = 0
        per_bit_correct = torch.zeros(self.args.watermark_dim, device=self.device)
        per_bit_total   = torch.zeros(self.args.watermark_dim, device=self.device)

        if self.rank == 0:
            if epoch <= self.args.aug_warmup_epochs:
                print(f"\n[Epoch {epoch}] ⏳ Warmup，本 epoch 不施加攻击")
            else:
                print(f"\n[Epoch {epoch}] 🔥 增强激活 aug_prob={self.args.aug_prob}")

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(self.rank != 0))

        for batch_idx, batch in enumerate(pbar):
            batch_size     = len(batch['utt_ids'])
            watermark_bits = generate_random_watermark(batch_size, self.args.watermark_dim).to(self.device)
            mel_feat       = batch['mel_feat'].to(self.device)

            if batch_idx == 0:
                self.save_comparison_audio(batch, watermark_bits, epoch, batch_idx)

            # ---- Step 1: mel → encoder → hift → audio_wm ----
            audio_wm, mel_wm, mel_clean, wm_bits_doubled = self.pipeline(
                mel_feat, watermark_bits, use_bit_inverse=True
            )

            # ---- Step 2: 随机音频攻击（encodec 逻辑已在 augmenter 内）----
            audio_wm_aug = self.augmenter(audio_wm, epoch=epoch)

            # ---- Step 3: 增强音频 → detector ----
            detected_bits = self.watermark_detector(audio_wm_aug)

            # ---- 逐位精度统计 ----
            with torch.no_grad():
                pred = (detected_bits >= 0.5).float()
                per_bit_correct += (pred == wm_bits_doubled.float()).float().sum(dim=0)
                per_bit_total   += wm_bits_doubled.shape[0]

            # ---- 损失 ----
            loss_dict = self.criterion(
                detected_bits=detected_bits,
                target_bits=wm_bits_doubled,
                watermarked_mel=mel_wm,
                clean_mel=mel_clean,
            )
            loss = (loss_dict['total_loss'].mean()
                    if loss_dict['total_loss'].dim() > 0
                    else loss_dict['total_loss'])

            if batch_idx == 1 and self.rank == 0:
                print(f"Loss:{loss.item():.4f} BER:{loss_dict['ber']:.4f} "
                      f"d:{loss_dict['decode_loss']:.4f} a:{loss_dict['audio_loss']:.4f}")

            # ---- 反向传播 ----
            self.optimizer.zero_grad()
            loss.backward()

            if self.rank == 0:
                enc_grads = [p.grad.abs().mean().item()
                             for p in self.pipeline.module.watermark_encoder.parameters()
                             if p.grad is not None and p.grad.abs().sum() > 0]
                det_grads = [p.grad.abs().mean().item()
                             for p in self.watermark_detector.module.parameters()
                             if p.grad is not None and p.grad.abs().sum() > 0]
                enc_score = sum(enc_grads) / len(enc_grads) if enc_grads else 0
                det_score = sum(det_grads) / len(det_grads) if det_grads else 0
                if batch_idx % 10 == 0:
                    print(f"\n[Step {batch_idx}] {'✅' if enc_score > 0 else '❌'} "
                          f"Enc:{enc_score:.2e} Det:{det_score:.2e}")
            else:
                enc_score = 0

            torch.nn.utils.clip_grad_norm_(
                self.pipeline.module.watermark_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(
                self.watermark_detector.module.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_ber  += loss_dict['ber']
            n_batches  += 1

            if self.rank == 0:
                pbar.set_postfix({
                    'avg_loss': f'{total_loss/n_batches:.4f}',
                    'avg_ber':  f'{total_ber/n_batches:.4f}',
                    'loss': f'{loss.item():.4f}',
                    'ber':  f'{loss_dict["ber"]:.4f}',
                    'd':    f'{loss_dict["decode_loss"].item():.4f}',
                    'a':    f'{loss_dict["audio_loss"].item():.4f}',
                    'enc_g': f'{enc_score:.2e}',
                })

            if self.writer and self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss',         loss.item(),       self.global_step)
                self.writer.add_scalar('train/ber',          loss_dict['ber'],  self.global_step)
                self.writer.add_scalar('train/encoder_grad', enc_score,         self.global_step)

            self.global_step += 1

            # 第一个epoch内按固定间隔评估CV集，只统计跨卡汇总的ACC。
            step_in_epoch = batch_idx + 1
            if (epoch == 1 and val_loader is not None and
                    self.args.validate_step > 0 and
                    step_in_epoch % self.args.validate_step == 0):
                val_metrics = self.validate(val_loader)
                if self.rank == 0:
                    val_acc = val_metrics['acc']
                    logging.info(
                        f"Step Validation | epoch={epoch} "
                        f"step={step_in_epoch} global_step={self.global_step} "
                        f"ACC={val_acc:.6f} "
                        f"MinBitACC={val_metrics['min_per_bit_acc']:.6f} "
                        f"MinBit={val_metrics['min_bit']}")
                    if self.writer:
                        self.writer.add_scalar(
                            'validation/acc', val_acc, self.global_step)
                        self.writer.add_scalar(
                            'validation/min_per_bit_acc',
                            val_metrics['min_per_bit_acc'], self.global_step)
                        for bit_idx, bit_acc in enumerate(
                                val_metrics['per_bit_acc']):
                            self.writer.add_scalar(
                                f'validation/bit_{bit_idx}_acc',
                                bit_acc, self.global_step)
                    self._append_step_validation(
                        epoch, step_in_epoch, self.global_step, val_metrics)

        # ---- 跨卡汇总 ----
        dist.all_reduce(per_bit_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(per_bit_total,   op=dist.ReduceOp.SUM)

        if self.rank == 0:
            per_bit_acc = per_bit_correct / per_bit_total
            print(f"\n📊 Epoch {epoch} Per-Bit Accuracy:")
            for i, acc in enumerate(per_bit_acc.cpu().tolist()):
                bar  = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                flag = "⚠️" if acc < 0.85 else "✅"
                print(f"  Bit {i:2d}: {bar} {acc:.4f} {flag}")
            print(f"  Mean:{per_bit_acc.mean():.4f}  Min:{per_bit_acc.min():.4f}"
                  f" (bit {per_bit_acc.argmin()})")

        avg_loss_t = torch.tensor([total_loss / n_batches], device=self.device)
        avg_ber_t  = torch.tensor([total_ber  / n_batches], device=self.device)
        dist.all_reduce(avg_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_ber_t,  op=dist.ReduceOp.SUM)
        avg_loss_t /= self.world_size
        avg_ber_t  /= self.world_size

        return {'total_loss': avg_loss_t.item(), 'avg_ber': avg_ber_t.item()}

    # ----------------------------------------------------------
    @torch.no_grad()
    def validate(self, dataloader):
        pipeline_was_training = self.pipeline.training
        detector_was_training = self.watermark_detector.training

        # 验证不应消耗训练使用的随机数；HiFT可能包含随机相位/噪声生成。
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device)
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()

        # 每次step验证都从相同RNG状态开始，使HiFT的随机生成可比较。
        validation_seed = 20260801 + self.rank
        torch.manual_seed(validation_seed)
        torch.cuda.manual_seed(validation_seed)

        self.pipeline.eval()
        self.watermark_detector.eval()
        per_bit_correct = torch.zeros(
            self.args.watermark_dim, dtype=torch.float64, device=self.device)
        per_bit_total = torch.zeros(
            self.args.watermark_dim, dtype=torch.float64, device=self.device)

        # 独立generator保证每次step验证及各消融版本使用相同的验证水印，
        # 同时不改变训练的全局CPU RNG状态。
        watermark_generator = torch.Generator(device='cpu')
        watermark_generator.manual_seed(validation_seed)

        try:
            for batch in tqdm(dataloader, desc="Val ACC", disable=(self.rank != 0)):
                mel_feat = batch['mel_feat'].to(self.device)
                watermark_bits = torch.randint(
                    0, 2,
                    (len(batch['utt_ids']), self.args.watermark_dim),
                    generator=watermark_generator,
                    device='cpu',
                ).to(self.device)

                audio_wm, _, _, target_bits = self.pipeline(
                    mel_feat, watermark_bits, use_bit_inverse=False)
                detected_bits = self.watermark_detector(audio_wm)
                predicted_bits = (detected_bits >= 0.5)
                target_bits = target_bits.bool()
                bit_matches = (predicted_bits == target_bits)
                per_bit_correct += bit_matches.sum(
                    dim=0, dtype=torch.float64)
                per_bit_total += bit_matches.shape[0]

            dist.all_reduce(per_bit_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(per_bit_total, op=dist.ReduceOp.SUM)
            per_bit_acc_tensor = (
                per_bit_correct / per_bit_total.clamp_min(1.0))
            acc = (
                per_bit_correct.sum() /
                per_bit_total.sum().clamp_min(1.0)
            ).item()
            min_bit = int(per_bit_acc_tensor.argmin().item())
            min_per_bit_acc = per_bit_acc_tensor[min_bit].item()
            per_bit_acc = per_bit_acc_tensor.cpu().tolist()

            if self.rank == 0:
                print("\n📊 Validation Per-Bit Accuracy:")
                for bit_idx, bit_acc in enumerate(per_bit_acc):
                    bar = "█" * int(bit_acc * 20) + "░" * (
                        20 - int(bit_acc * 20))
                    flag = "⚠️" if bit_acc < 0.85 else "✅"
                    print(
                        f"  Bit {bit_idx:2d}: {bar} "
                        f"{bit_acc:.4f} {flag}")
                print(
                    f"  Mean:{per_bit_acc_tensor.mean().item():.4f} "
                    f"Min:{min_per_bit_acc:.4f} (bit {min_bit}) "
                    f"Overall:{acc:.4f}")
        finally:
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)

            if pipeline_was_training:
                self.pipeline.train()
                self.pipeline.module.hift.eval()
            if detector_was_training:
                self.watermark_detector.train()

        return {
            'acc': acc,
            'mean_per_bit_acc': float(np.mean(per_bit_acc)),
            'min_per_bit_acc': min_per_bit_acc,
            'min_bit': min_bit,
            'per_bit_acc': per_bit_acc,
        }

    # ----------------------------------------------------------
    def _append_step_validation(
            self, epoch, step_in_epoch, global_step, metrics):
        """在rank 0上增量保存step验证ACC，便于后续直接绘图。"""
        if self.rank != 0:
            return
        output_path = os.path.join(
            self.args.output_dir, 'step_validation_acc.csv')
        needs_header = not os.path.exists(output_path)
        with open(output_path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            if needs_header:
                writer.writerow([
                    'epoch', 'step_in_epoch', 'global_step', 'overall_acc',
                    'mean_per_bit_acc', 'min_per_bit_acc', 'min_bit',
                    *[
                        f'bit_{bit_idx}_acc'
                        for bit_idx in range(self.args.watermark_dim)
                    ],
                ])
            writer.writerow([
                epoch,
                step_in_epoch,
                global_step,
                f"{metrics['acc']:.8f}",
                f"{metrics['mean_per_bit_acc']:.8f}",
                f"{metrics['min_per_bit_acc']:.8f}",
                metrics['min_bit'],
                *[f'{bit_acc:.8f}' for bit_acc in metrics['per_bit_acc']],
            ])

    # ----------------------------------------------------------
    def save_checkpoint(self, epoch, metrics):
        if self.rank != 0:
            return
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict':   self.pipeline.module.watermark_encoder.state_dict(),
            'detector_state_dict':  self.watermark_detector.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        save_path = os.path.join(self.args.output_dir, f'model_epoch_{epoch}.pt')
        torch.save(checkpoint, save_path)
        logging.info(f"Saved: {save_path}")

    # ----------------------------------------------------------
    def train(self, train_loader, val_loader):
        if self.rank == 0:
            logging.info("🚀 Training started (DDP + Aug, detector separated)...")

        start_epoch = getattr(self.args, 'resume_epoch', 0) + 1
        for epoch in range(start_epoch, start_epoch + self.args.num_epochs):
            train_metrics = self.train_one_epoch(train_loader, epoch, val_loader)

            if self.rank == 0:
                logging.info(f"Epoch {epoch} Loss:{train_metrics['total_loss']:.4f} "
                             f"BER:{train_metrics['avg_ber']:.4f}")

            if val_loader:
                val_metrics = self.validate(val_loader)
                if self.rank == 0:
                    logging.info(f"Val ACC (clean): {val_metrics['acc']:.6f}")
                    # 若epoch末尾不是validate_step的整数倍，额外保存终点结果。
                    if (epoch == 1 and self.args.validate_step > 0 and
                            len(train_loader) % self.args.validate_step != 0):
                        self._append_step_validation(
                            epoch, len(train_loader), self.global_step,
                            val_metrics)
            else:
                val_metrics = train_metrics

            self.scheduler.step()
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(epoch, val_metrics)

            dist.barrier()


# ============================================================
# DDP Worker 入口
# ============================================================
def worker_fn(args):
    # torchrun 自动设置 LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    rank       = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    physical_gpu_ids = [
        gpu_id.strip()
        for gpu_id in os.environ.get('WATERMARK_GPU_IDS', '').split(',')
        if gpu_id.strip()
    ]
    if physical_gpu_ids:
        if len(physical_gpu_ids) != world_size:
            raise RuntimeError(
                f'WATERMARK_GPU_IDS has {len(physical_gpu_ids)} entries, '
                f'but WORLD_SIZE={world_size}: {physical_gpu_ids}'
            )
        os.environ['CUDA_VISIBLE_DEVICES'] = physical_gpu_ids[rank]
    dist.init_process_group(backend='nccl')
    seed_everything(42)
    # seed_everything(42+rank)

    if rank == 0:
        print("=" * 60)
        print("🚀 WATERMARK TRAINING - DDP + Aug (detector separated) [torchrun]")
        print(f"   world_size={world_size}")
        print("=" * 60)

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s %(message)s'
    )

    trainer = WatermarkTrainer(args, rank=rank, world_size=world_size)

    train_dataset = MelWatermarkDataset(args.train_data, args.mel_dir, rank=rank)
    if args.train_samples:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset,
                               list(range(min(args.train_samples, len(train_dataset)))))

    val_dataset = None
    if (not args.disable_validation and
            args.cv_data and os.path.exists(args.cv_data)):
        cv_mel_dir = args.cv_mel_dir or args.mel_dir
        val_dataset = MelWatermarkDataset(args.cv_data, cv_mel_dir, rank=rank)
        if args.val_samples:
            from torch.utils.data import Subset
            val_dataset = Subset(val_dataset,
                                 list(range(min(args.val_samples, len(val_dataset)))))

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size_per_gpu, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn_mel_batch,
        pin_memory=True, drop_last=True)

    val_loader = None
    if val_dataset:
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size_per_gpu, sampler=val_sampler,
            num_workers=args.num_workers, collate_fn=collate_fn_mel_batch)

    if rank == 0:
        print(f"Train: {len(train_dataset)} samples, "
              f"{len(train_loader)} batches/gpu × {world_size} gpus")
        if val_dataset:
            print(f"Val: {len(val_dataset)} samples, mel_dir={args.cv_mel_dir or args.mel_dir}")
        elif args.disable_validation:
            print("Val: disabled (--disable_validation)")

    trainer.train(train_loader, val_loader)
    dist.destroy_process_group()


# ============================================================
# 主入口
# ============================================================
def main():
    args = get_args()
    worker_fn(args)
    print("\n🎉 Training finished!")


if __name__ == '__main__':
    main()
