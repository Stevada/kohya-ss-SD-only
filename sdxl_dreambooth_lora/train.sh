#!/bin/bash
# SDXL LoRA DreamBooth Training Script
# Works for single or multiple characters - just modify config.toml
# Edit the variables below before running

set -e  # Exit on error

# === REQUIRED: EDIT THESE PATHS ===
MODEL_PATH="/path/to/sd_xl_base_1.0.safetensors"
DATASET_CONFIG="./config.toml"           # Your dataset configuration
OUTPUT_NAME="my_lora"                     # Name for output files

# === OPTIONAL: TRAINING PARAMETERS ===
OUTPUT_DIR="./output"
EPOCHS=10
LEARNING_RATE="1e-4"
NETWORK_DIM=32                            # 16 or 8 for lower VRAM
NETWORK_ALPHA=16

# === OPTIONAL: SAMPLE GENERATION ===
SAMPLE_PROMPTS=""                         # Path to prompts file (optional)
SAMPLE_EVERY_N_EPOCHS=1                   # 0 to disable

# === EXECUTION ===
echo "Starting SDXL LoRA DreamBooth training..."
echo "Model: ${MODEL_PATH}"
echo "Config: ${DATASET_CONFIG}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors"
echo ""

accelerate launch --num_cpu_threads_per_process=8 ../sdxl_train_network.py \
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
  --cache_text_encoder_outputs \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --save_precision="fp16" \
  --optimizer_type="AdamW8bit" \
  --bucket_reso_steps=32 \
  --bucket_no_upscale \
  --xformers \
  --no_half_vae \
  --save_every_n_epochs=1 \
  --logging_dir="${OUTPUT_DIR}/logs" \
  --log_prefix="${OUTPUT_NAME}" \
  ${SAMPLE_PROMPTS:+--sample_every_n_epochs=${SAMPLE_EVERY_N_EPOCHS}} \
  ${SAMPLE_PROMPTS:+--sample_prompts="${SAMPLE_PROMPTS}"} \
  ${SAMPLE_PROMPTS:+--sample_sampler="euler_a"}

echo ""
echo "Training complete! LoRA saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors"
