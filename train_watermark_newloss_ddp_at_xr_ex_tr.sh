#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/home/zhangyihang/CosyVoice/audiowatermark:$PYTHONPATH

# ====== 1. Paths ======
COSYVOICE_MODEL="/home/zhangyihang/workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B"
TRAIN_DATA="/data/zhangyihang/watermark_data/train/data.list"
CV_DATA="/data/zhangyihang/LJSpeech-1.1/watermark_data/cv/data.list"
MEL_DIR="/data/zhangyihang/watermark_data/train/mel"

# ====== 2. Experiment and outputs ======
EXP_NAME="10bits_5"
OUTPUT_DIR="shared/ex/exp/$EXP_NAME"
TB_DIR="shared/ex/tensorboard/tensorboard_$EXP_NAME"
AUDIO_OUT_DIR="shared/ex/output_audio/$EXP_NAME"

mkdir -p $OUTPUT_DIR

# ====== 3. Hyperparameters ======
NUM_EPOCHS=50
LR=5e-4
LAMBDA_DECODE=1.0
LAMBDA_AUDIO=0.2
WATERMARK_DIM=10
BATCH_SIZE_PER_GPU=32
GPU_IDS="0,1,2,3,4,5,6,7"
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ID_ARRAY[@]}
MASTER_PORT=29501
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export WATERMARK_GPU_IDS="$GPU_IDS"

# ====== 4. Data augmentation ======
AUG_PROB=0.7
AUG_MAX_ATTACKS=1
AUG_WARMUP_EPOCHS=20
AUG_SAMPLE_RATE=24000

NOISE_DIR="/home/zhangyihang/.cache/torch/hub/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav"

# ====== 5. Start training ======
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    cosyvoice/bin/train_watermark_mel_newloss_ddp_at_e_tr.py \
    --train_samples 450475 \
    --cosyvoice_model_dir $COSYVOICE_MODEL \
    --train_data $TRAIN_DATA \
    --cv_data $CV_DATA \
    --mel_dir $MEL_DIR \
    --output_dir $OUTPUT_DIR \
    --tensorboard_dir $TB_DIR \
    --audio_output_dir $AUDIO_OUT_DIR \
    --batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --lambda_decode $LAMBDA_DECODE \
    --lambda_audio $LAMBDA_AUDIO \
    --num_workers 4 \
    --save_interval 1 \
    --watermark_dim $WATERMARK_DIM \
    --master_port "$MASTER_PORT" \
    --aug_prob $AUG_PROB \
    --aug_max_attacks $AUG_MAX_ATTACKS \
    --aug_warmup_epochs $AUG_WARMUP_EPOCHS \
    --aug_sample_rate $AUG_SAMPLE_RATE \
    --aug_attacks gaussian_noise,background_noise,lowpass,highpass,smooth,echo,encodec,time_stretch \
    --noise_dir $NOISE_DIR \
    --disable_mp3_aug \
    --fp16

    # --aug_attacks gaussian_noise,background_noise,lowpass,highpass,smooth,echo,encodec \
