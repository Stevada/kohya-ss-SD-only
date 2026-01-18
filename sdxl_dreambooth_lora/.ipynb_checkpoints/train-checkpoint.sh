#!/bin/bash
# SDXL LoRA DreamBooth Training Script
# Works for single or multiple characters - just modify config.toml
# Edit the variables below before running

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the parent directory (kohya-ss-SD-only)
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

# === REQUIRED: EDIT THESE PATHS ===
# MODEL_PATH="/workspace/runpod-slim/ComfyUI/models/checkpoints/ponyRealism_V22.safetensors"
MODEL_PATH="/workspace/runpod-slim/ComfyUI/models/checkpoints/ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
DATASET_CONFIG="${REPO_DIR}/config.toml"           # Your dataset configuration
OUTPUT_NAME="ava_lora"                     # Name for output files

# === OPTIONAL: TRAINING PARAMETERS ===
OUTPUT_DIR="./output"
EPOCHS=20
LEARNING_RATE="1e-5"                      # 1e-4 to 5e-4 recommended
NETWORK_DIM=32                            # 16 or 8 for lower VRAM
NETWORK_ALPHA=16

# === OPTIONAL: SAMPLE GENERATION ===
SAMPLE_PROMPTS="ava1 girl, front view, leaning on table, cafe setting, casual pose, high quality"                         # Path to prompts file (optional)
SAMPLE_EVERY_N_EPOCHS=5                   # 0 to disable

# === EXECUTION ===
echo "Starting SDXL LoRA DreamBooth training..."
echo "Model: ${MODEL_PATH}"
echo "Config: ${DATASET_CONFIG}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors"
echo ""

accelerate launch --num_cpu_threads_per_process=8 "${REPO_DIR}/sdxl_train_network.py" \
  --pretrained_model_name_or_path="${MODEL_PATH}" \
  --dataset_config="${DATASET_CONFIG}" \
  --output_dir="${OUTPUT_DIR}" \
  --output_name="${OUTPUT_NAME}" \
  --save_model_as=safetensors \
  --prior_loss_weight=1.0 \
  --max_train_epochs=${EPOCHS} \
  --learning_rate=${LEARNING_RATE} \
  --network_module=networks.lora \
  --network_dim=${NETWORK_DIM} \
  --network_alpha=${NETWORK_ALPHA} \
  --network_train_unet_only \
  --cache_latents \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --save_precision="fp16" \
  --optimizer_type="AdamW" \
  --bucket_reso_steps=32 \
  --bucket_no_upscale \
  --no_half_vae \
  --save_every_n_epochs="${SAMPLE_EVERY_N_EPOCHS}" \
  --logging_dir="${REPO_DIR}/logs" \
  --log_prefix="${OUTPUT_NAME}" \
  ${SAMPLE_PROMPTS:+--sample_prompts="${SAMPLE_PROMPTS}"} \
  ${SAMPLE_PROMPTS:+--sample_sampler="euler_a"}

echo ""
echo "Training complete! LoRA saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors"
