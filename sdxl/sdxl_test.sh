#!/bin/bash

# SDXL Identity-Preserving LoRA Testing Script
# Generate images with your trained LoRA using character reference images

# ====================
# CONFIGURATION - EDIT THESE PATHS
# ====================

# Model paths
SDXL_MODEL="sd_xl_base_1.0.safetensors"
LORA_WEIGHTS="output/character_lora/character_lora.safetensors"

# Character reference image for identity conditioning
REFERENCE_IMAGE="dataset/reference_images/001.jpg"

# Output settings
OUTPUT_DIR="output/generated_images"
BATCH_SIZE=1

# Generation parameters
PROMPT="1girl, standing at beach, sunset, beautiful smile, long hair flowing in wind"
NEGATIVE_PROMPT="low quality, worst quality, bad anatomy, blurry, watermark, text"

WIDTH=1024
HEIGHT=1024
CFG_SCALE=7.0
STEPS=40
SAMPLER="k_euler_a"

# Identity conditioning strength (0.0-1.0)
# Higher = stronger identity preservation from reference image
CLIP_VISION_STRENGTH=1.0

# Number of images to generate
NUM_IMAGES=4

# ====================
# GENERATION COMMAND
# ====================

mkdir -p "${OUTPUT_DIR}"

echo "Generating ${NUM_IMAGES} images with identity conditioning..."
echo "Reference image: ${REFERENCE_IMAGE}"
echo "Prompt: ${PROMPT}"
echo ""

python ../sdxl_gen_img.py \
  --ckpt="${SDXL_MODEL}" \
  --network_module=networks.lora \
  --network_weights="${LORA_WEIGHTS}" \
  --network_mul=1.0 \
  --clip_vision_strength=${CLIP_VISION_STRENGTH} \
  --init_image="${REFERENCE_IMAGE}" \
  --prompt="${PROMPT}" \
  --negative_prompt="${NEGATIVE_PROMPT}" \
  --W=${WIDTH} \
  --H=${HEIGHT} \
  --scale=${CFG_SCALE} \
  --steps=${STEPS} \
  --sampler="${SAMPLER}" \
  --outdir="${OUTPUT_DIR}" \
  --images_per_prompt=${NUM_IMAGES} \
  --xformers \
  --bf16

echo ""
echo "Generation complete! Images saved to: ${OUTPUT_DIR}"

# ====================
# TESTING WITHOUT IDENTITY (COMPARISON)
# ====================
# Uncomment below to generate comparison images WITHOUT reference image
# This helps you see the difference identity conditioning makes

# echo ""
# echo "Generating comparison images WITHOUT identity conditioning..."
# python ../sdxl_gen_img.py \
#   --ckpt="${SDXL_MODEL}" \
#   --network_module=networks.lora \
#   --network_weights="${LORA_WEIGHTS}" \
#   --network_mul=1.0 \
#   --prompt="${PROMPT}" \
#   --negative_prompt="${NEGATIVE_PROMPT}" \
#   --W=${WIDTH} \
#   --H=${HEIGHT} \
#   --scale=${CFG_SCALE} \
#   --steps=${STEPS} \
#   --sampler="${SAMPLER}" \
#   --outdir="${OUTPUT_DIR}/no_identity" \
#   --images_per_prompt=${NUM_IMAGES} \
#   --xformers \
#   --bf16

# ====================
# NOTES
# ====================
# - MUST provide --init_image and --clip_vision_strength when using LoRA trained with identity conditioning
# - Try different CLIP_VISION_STRENGTH values:
#   - 0.5: Subtle identity hints
#   - 1.0: Strong identity preservation (recommended)
#   - 1.5: Very strong (may reduce flexibility)
# - Reference image should clearly show the character's face/features
# - Different reference images from same character will produce similar identity
