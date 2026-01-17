#!/bin/bash
# SDXL LoRA Testing Script
# Generate images with your trained LoRA

set -e  # Exit on error

# === REQUIRED: EDIT THESE PATHS ===
MODEL_PATH="/path/to/sd_xl_base_1.0.safetensors"
LORA_PATH="./output/my_lora.safetensors"
PROMPT="ohwx person, portrait, high quality"
NEGATIVE_PROMPT="low quality, blurry"

# === OPTIONAL: GENERATION SETTINGS ===
OUTPUT_DIR="./generated"
LORA_WEIGHT=1.0        # 0.0-1.5, adjust strength
WIDTH=1024
HEIGHT=1024
STEPS=30
SEED=-1                # -1 for random

# === EXECUTION ===
echo "Generating images with SDXL LoRA..."
echo "Model: ${MODEL_PATH}"
echo "LoRA: ${LORA_PATH}"
echo "Prompt: ${PROMPT}"
echo ""

mkdir -p "${OUTPUT_DIR}"

python ../sdxl_gen_img.py \
  --ckpt="${MODEL_PATH}" \
  --network_module=networks.lora \
  --network_weights="${LORA_PATH}" \
  --network_mul=${LORA_WEIGHT} \
  --prompt="${PROMPT}" \
  --negative_prompt="${NEGATIVE_PROMPT}" \
  --W=${WIDTH} \
  --H=${HEIGHT} \
  --steps=${STEPS} \
  --sampler=euler_a \
  --outdir="${OUTPUT_DIR}" \
  ${SEED:+--seed=${SEED}}

echo ""
echo "Images generated in: ${OUTPUT_DIR}"
