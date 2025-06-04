#!/bin/bash

# Set DEBUG environment variable for detailed output
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PATH=$PATH:/path/to/code/spikegen/mar
export PYTHONPATH=$PYTHONPATH:/path/to/code/spikegen/mar

# Set default parameters
# GPUS="0,1,2,3,4,5,6,7"
GPUS="0,1,2,3"
MODEL="spike_mar_base"
BATCH_SIZE=12
IMG_H=400   
IMG_W=600  
KERNEL_SIZE=25
BLUR_INTENSITY=25
TARGET_COVERAGE=0.25
DATA_ROOT="./SpikeGS/train/"
RGB_VAE_PATH="vae-kl-f8-d16"
DIFFUSION_CKPT="./pretrained_weights/imagenet1k/checkpoint-last.pth"
OUTPUT_DIR="./output/spike_SpikeGS"
EPOCHS=500
EVAL_FREQ=100
SAVE_FREQ=25
LR=1e-4

# Set CUDA_VISIBLE_DEVICES based on GPU ID
export CUDA_VISIBLE_DEVICES=$GPUS

# Calculate number of GPUs used
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Using $NUM_GPUS GPU(s): $GPUS"
echo "Batch size: $BATCH_SIZE (per GPU)"
echo "Image size: $IMG_SIZE"
echo "Number of epochs: $EPOCHS"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start training
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
    --target_coverage $TARGET_COVERAGE \
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

echo "Training complete!"
