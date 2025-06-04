#!/bin/bash


export TORCH_DISTRIBUTED_DEBUG=DETAIL

# GPUS="0,1,2,3"
GPUS="0"
MODEL="spike_mar_base"
BATCH_SIZE=64
IMG_H=256
IMG_W=256
KERNEL_SIZE=40
BLUR_INTENSITY=40
TARGET_COVERAGE=0.1
GAMMA=2
SMOOTH_SIGMA=1
DATA_ROOT="./imagenet1k/train"
RGB_VAE_PATH="ostris/vae-kl-f8-d16"
DIFFUSION_CKPT="./pretrained_weights/mar/checkpoint-last.pth"
OUTPUT_DIR="./output/spikegen_pretrained"
EPOCHS=100
EVAL_FREQ=100
SAVE_FREQ=10
LR=1e-4
# RESUME="./output/spike_mar_imagenet_full_v1/checkpoint-last.pth"

export CUDA_VISIBLE_DEVICES=$GPUS


IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "use $NUM_GPUS GPUs: $GPUS"
echo "batch size: $BATCH_SIZE (per GPU)"
echo "image size: $IMG_SIZE"
echo "epochs: $EPOCHS"


mkdir -p "$OUTPUT_DIR"


/home/daigaole/conda/envs/GBA/bin/torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$(( RANDOM % 1000 + 29000 )) \
    train_spikegen.py \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --img_h $IMG_H \
    --img_w $IMG_W \
    --kernel_size $KERNEL_SIZE \
    --blur_intensity $BLUR_INTENSITY \
    --target_coverage $TARGET_COVERAGE \
    --gamma $GAMMA \
    --smooth_sigma $SMOOTH_SIGMA \
    --data_root $DATA_ROOT \
    --rgb_vae_path $RGB_VAE_PATH \
    --diffusion_ckpt $DIFFUSION_CKPT \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --eval_freq $EVAL_FREQ \
    --save_last_freq $SAVE_FREQ \
    --online_eval \
    --lr $LR \
    # --resume $RESUME

echo "train done!"