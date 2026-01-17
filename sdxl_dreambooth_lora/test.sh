#!/bin/bash
# SDXL LoRA Testing Script
# Generate images with your trained LoRA

set -e  # Exit on error

# === REQUIRED: EDIT THESE PATHS ===
REPO_DIR="/root/kohya-ss-SD-only"
MODEL_PATH="/workspace/runpod-slim/ComfyUI/models/checkpoints/ponyRealism_V22.safetensors"
LOCAL_PATH="${REPO_DIR}/sdxl_dreambooth_lora"

LORA_PATH="${LOCAL_PATH}/output/ava_lora.safetensors"
PROMPT="ava1 girl, blowjob, sucking a male's dick. 8k, hyperrealistic. --n low quality, bad anatomy, extra fingers, missing fingers, extra limbs, missing limbs, blurry, noise, artifacts"

# === OPTIONAL: GENERATION SETTINGS ===
OUTPUT_DIR="${LOCAL_PATH}/generated"
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

mkdir -p "${OUTPUT_DIR}"

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
  --outdir="${OUTPUT_DIR}" \
  --fp16 \
  ${SEED:+--seed=${SEED}}

echo ""
echo "Images generated in: ${OUTPUT_DIR}"
