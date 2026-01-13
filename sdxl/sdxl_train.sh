#!/bin/bash

# SDXL Identity-Preserving LoRA Training Script
# This script trains a LoRA with character identity conditioning using reference images

# ====================
# CONFIGURATION - EDIT THESE PATHS
# ====================

# Model and output paths
PRETRAINED_MODEL="/workspace/runpod-slim/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors"  # Path to SDXL base model
OUTPUT_DIR="output/character_lora"
OUTPUT_NAME="character_lora"

# Dataset paths (expects this structure):
# dataset/
# ├── target_images/      # Training images (desired outputs)
# │   ├── 001.jpg
# │   ├── 002.jpg
# ├── control_images/   # Character reference images (identity source)
# │   ├── 001.jpg        # Must match target image names
# │   ├── 002.jpg
# └── captions/
#     ├── 001.txt
#     ├── 002.txt

TRAIN_DATA_DIR="dataset"
CONDITIONING_DATA_DIR="dataset/control_images"
TEST_CONDITIONING_DATA_DIR="dataset/test/control_images"
CAPTION_EXTENSION=".txt"

# Training parameters
RESOLUTION="512,512"
BATCH_SIZE=1
LEARNING_RATE="1e-4"
MAX_TRAIN_EPOCHS=10
SAVE_EVERY_N_EPOCHS=2

# LoRA parameters
NETWORK_DIM=32          # LoRA rank (higher = more capacity, more VRAM)
NETWORK_ALPHA=16        # LoRA alpha (typically half of dim)

# Identity conditioning parameters
USE_IDENTITY=true       # Set to false to train regular LoRA without identity conditioning
IDENTITY_STRENGTH=1.0   # 0.0-1.0, higher = stronger identity preservation

# Mixed precision (bf16 recommended for 30xx/40xx GPUs, fp16 for older)
MIXED_PRECISION="bf16"

# ====================
# TRAINING COMMAND
# ====================

accelerate launch --num_cpu_threads_per_process=8 ./sdxl_train_network.py \
  --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
  --train_data_dir="${TRAIN_DATA_DIR}" \
  --conditioning_data_dir="${CONDITIONING_DATA_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --output_name="${OUTPUT_NAME}" \
  --caption_extension="${CAPTION_EXTENSION}" \
  --resolution="${RESOLUTION}" \
  --train_batch_size=${BATCH_SIZE} \
  --learning_rate="${LEARNING_RATE}" \
  --max_train_epochs=${MAX_TRAIN_EPOCHS} \
  --save_every_n_epochs=${SAVE_EVERY_N_EPOCHS} \
  --network_module=networks.lora \
  --network_dim=${NETWORK_DIM} \
  --network_alpha=${NETWORK_ALPHA} \
  --save_model_as=safetensors \
  --mixed_precision="${MIXED_PRECISION}" \
  --save_precision="fp16" \
  --cache_latents \
  --gradient_checkpointing \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --use_identity_conditioning \
  --identity_conditioning_strength=${IDENTITY_STRENGTH} \
  --enable_bucket \
  --min_bucket_reso=256 \
  --max_bucket_reso=1024 \
  --bucket_reso_steps=64 \
  --caption_dropout_rate=0.05 \
  --shuffle_caption \
  --keep_tokens=1 \
  --seed=42 \
  --max_data_loader_n_workers=2 \
  --persistent_data_loader_workers \
  --log_with="tensorboard" \
  --logging_dir="logs" \
  --test_character_images_dir="${TEST_CONDITIONING_DATA_DIR}"

# ====================
# NOTES
# ====================
# - First run will download CLIP vision model (~4GB)
# - Training with identity conditioning uses ~2GB more VRAM than regular LoRA
# - If you run out of memory:
#   - Reduce NETWORK_DIM to 16 or 8
#   - Use MIXED_PRECISION="fp16"
#   - Remove --gradient_checkpointing (uses more VRAM but faster)
# - Reference images should show the same character as target images
# - Images are matched by filename (001.jpg pairs with 001.jpg)
