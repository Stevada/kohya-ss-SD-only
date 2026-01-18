#!/bin/bash
# SDXL LoRA Testing Script
# Generate images with your trained LoRA

set -e  # Exit on error

# === REQUIRED: EDIT THESE PATHS ===
REPO_DIR="/root/kohya-ss-SD-only"
MODEL_PATH="${REPO_DIR}/models/ponyRealism_V22.safetensors"
TRIGGER_WORD="ava1037"
OUTPUT_NAME="${TRIGGER_WORD}_lora_ponyRealism_V22" # Name for output files
OUTPUT_DIR="${REPO_DIR}/output/${OUTPUT_NAME}"
LORA_PATH="${OUTPUT_DIR}/${TRIGGER_WORD}_lora.safetensors"

# === OPTIONAL: GENERATION SETTINGS ===
PROMPT="${TRIGGER_WORD}, blow job, the female is sucking the male's dick, 8K, realistic, high quality --n low quality, bad anatomy, extra fingers, missing fingers, extra limbs, missing limbs, blurry, noise, artifacts"
RESULTS_DIR="${OUTPUT_DIR}/tested"
LORA_WEIGHT=1.0        # 0.0-1.5, adjust strength
WIDTH=1024
HEIGHT=1024
STEPS=20
SEED=-1                # -1 for random

# === EXECUTION ===
echo "Generating images with SDXL LoRA..."
echo "Model: ${MODEL_PATH}"
echo "LoRA: ${LORA_PATH}"
echo "Prompt: ${PROMPT}"
echo ""

mkdir -p "${RESULTS_DIR}"

python "${REPO_DIR}/sdxl_gen_img.py" \
  --ckpt="${MODEL_PATH}" \
  --network_module=networks.lora \
  --network_weights="${LORA_PATH}" \
  --network_mul=${LORA_WEIGHT} \
  --prompt="${PROMPT}" \
  --W=${WIDTH} \
  --H=${HEIGHT} \
  --steps=${STEPS} \
  --sampler=euler_a \
  --outdir="${RESULTS_DIR}" \
  --fp16 \
  ${SEED:+--seed=${SEED}}

echo ""
echo "Images generated in: ${RESULTS_DIR}"
