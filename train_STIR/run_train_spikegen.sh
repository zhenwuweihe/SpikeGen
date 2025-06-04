#!/bin/bash

# Set DEBUG environment variable for detailed info
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export PATH=$PATH:/path/to/code/spikegen/mar
export PYTHONPATH=$PYTHONPATH:/path/to/code/spikegen/mar

# Set default parameters
GPUS="1,2,3,4,5,6,7"
# GPUS="1"
MODEL="spike_mar_base"
BATCH_SIZE=16
IMG_H=360
IMG_W=640
KERNEL_SIZE=40
BLUR_INTENSITY=40
TARGET_COVERAGE=0.5
GAMMA=1
SMOOTH_SIGMA=1
PHOTON_SAMPLES=16
DATA_ROOT="./SSIR/train/"
RGB_VAE_PATH="vae-kl-f8-d16"
DIFFUSION_CKPT="./pretrained_weights/imagenet1k/checkpoint-last.pth"
OUTPUT_DIR="./output/spike_STIR"
EPOCHS=100
EVAL_FREQ=100
SAVE_FREQ=1
LR=1e-4
# Set CUDA_VISIBLE_DEVICES using GPU IDs
export CUDA_VISIBLE_DEVICES=$GPUS

# Calculate number of GPUs to use
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Using $NUM_GPUS GPUs: $GPUS"
echo "Batch size: $BATCH_SIZE (per GPU)"
echo "Image size: $IMG_SIZE"
echo "Number of training epochs: $EPOCHS"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$(( RANDOM % 1000 + 29000 )) \
    train_spikegen.py \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --img_h $IMG_H \
    --img_w $IMG_W \
    --kernel_size $KERNEL_SIZE \
    --blur_intensity $BLUR_INTENSITY \
    --smooth_sigma $SMOOTH_SIGMA \
    --gamma $GAMMA \
    --target_coverage $TARGET_COVERAGE \
    --photon_samples $PHOTON_SAMPLES \
    --data_root $DATA_ROOT \
    --rgb_vae_path $RGB_VAE_PATH \
    --diffusion_ckpt $DIFFUSION_CKPT \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --eval_freq $EVAL_FREQ \
    --save_last_freq $SAVE_FREQ \
    --online_eval \
    --lr $LR 
    # --resume $RESUME

echo "Training completed!"
