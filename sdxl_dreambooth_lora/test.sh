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
REALISM_POSITIVE_PROMPT_PREFIX="score_9, score_8_up, score_7_up, BREAK, rating_explicit"
REALISM_POSITIVE_PROMPT_SUFFIX="8K, realistic, high quality, nsfw, detailed skin"
REALISM_NEGATIVE_PROMPT_TAGS="score_4, score_5, score_6"
REALISM_NEGATIVE_PROMPT="${REALISM_NEGATIVE_PROMPT_TAGS}, ai-generated, artifact, artifacts, bad quality, bad scan, blurred, blurry, compressed, compression artifacts, corrupted, dirty art scan, dirty scan, dithering, downsampling, faded lines, frameborder, grainy, heavily compressed, heavily pixelated, high noise, image noise, low dpi, low fidelity, low resolution, lowres, moire pattern, moiré pattern, motion blur, muddy colors, noise, noisy background, overcompressed, pixelation, pixels, poor quality, poor lineart, scanned with errors, scan artifact, scan errors, very low quality, visible pixels"
ACTION_PROMPT="male POV close flat view: beautiful nude female kneeling straddling male's thighs performing paizuri titjob, male lying flat legs wrapped around her waist hips on her lap, massive soft breasts pressed together by hands from underside clamping vertical erect penis in cleavage head slightly protruding top, breasts moving up and down stroking shaft, hands only on breasts no grip on penis, average realistic penis size natural texture,female leaning forward looking at viewer aroused moaning expression, natural hair and breast movement"
PROMPT="${REALISM_POSITIVE_PROMPT_PREFIX}, ${TRIGGER_WORD}, ${ACTION_PROMPT}, ${REALISM_POSITIVE_PROMPT_SUFFIX} --n ${REALISM_NEGATIVE_PROMPT}"

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
