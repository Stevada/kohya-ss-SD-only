#!/bin/bash
# SDXL LoRA DreamBooth Training Script
# Works for single or multiple characters - just modify config.toml
# Edit the variables below before running

set -e  # Exit on error

# === REQUIRED: EDIT THESE PATHS ===
REPO_DIR="/root/kohya-ss-SD-only"
# MODEL_PATH="${REPO_DIR}/models/ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
MODEL_PATH="${REPO_DIR}/models/ponyRealism_V22.safetensors"
DATASET_CONFIG="${REPO_DIR}/config.toml" # Your dataset configuration
TRIGGER_WORD="ava1037"
OUTPUT_NAME="${TRIGGER_WORD}_lora_ponyRealism_V22" # Name for output files
OUTPUT_DIR="${REPO_DIR}/output/${OUTPUT_NAME}"

# === OPTIONAL: TRAINING PARAMETERS ===
EPOCHS=20
LEARNING_RATE="5e-5"                      # 1e-4 to 5e-4 recommended
NETWORK_DIM=32                            # 16 or 8 for lower VRAM
NETWORK_ALPHA=16

# === OPTIONAL: SAMPLE GENERATION ===
SAMPLE_EVERY_N_EPOCHS=2                     # 0 to disable
SAVE_N_EPOCH_RATIO=5                        # Save 5 models throughout the training

# Auto-generate sample prompts file with trigger word
SAMPLE_SIZE=1024
SAMPLE_STEPS=20
SAMPLE_CFG_SCALE=6.0
SAMPLE_PROMPTS_FILE="${REPO_DIR}/sdxl_dreambooth_lora/sample_prompts.txt"
REALISM_POSITIVE_PROMPT_TAGS="score_9, score_8_up, score_7_up, BREAK, 8K, realistic, high quality, nsfw"
REALISM_POSITIVE_PROMPT="${TRIGGER_WORD}, ${REALISM_POSITIVE_PROMPT_TAGS}"
REALISM_NEGATIVE_PROMPT_TAGS="score_4, score_5, score_6"
REALISM_NEGATIVE_PROMPT="${REALISM_NEGATIVE_PROMPT_TAGS}, ai-generated, artifact, artifacts, bad quality, bad scan, blurred, blurry, compressed, compression artifacts, corrupted, dirty art scan, dirty scan, dithering, downsampling, faded lines, frameborder, grainy, heavily compressed, heavily pixelated, high noise, image noise, low dpi, low fidelity, low resolution, lowres, moire pattern, moiré pattern, motion blur, muddy colors, noise, noisy background, overcompressed, pixelation, pixels, poor quality, poor lineart, scanned with errors, scan artifact, scan errors, very low quality, visible pixels"

ANIME_POSITIVE_PROMPT_TAGS="score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, source_anime, BREAK"
ANIME_POSITIVE_PROMPT="${TRIGGER_WORD}, ${ANIME_POSITIVE_PROMPT_TAGS}"
ANIME_NEGATIVE_PROMPT_TAGS="3D"
ANIME_NEGATIVE_PROMPT="${ANIME_NEGATIVE_PROMPT_TAGS}, ai-generated, artifact, artifacts, bad quality, bad scan, blurred, blurry, compressed, compression artifacts, corrupted, dirty art scan, dirty scan, dithering, downsampling, faded lines, frameborder, grainy, heavily compressed, heavily pixelated, high noise, image noise, low dpi, low fidelity, low resolution, lowres, moire pattern, moiré pattern, motion blur, muddy colors, noise, noisy background, overcompressed, pixelation, pixels, poor quality, poor lineart, scanned with errors, scan artifact, scan errors, very low quality, visible pixels"

cat > "${SAMPLE_PROMPTS_FILE}" <<EOF
# Sample prompts for ${TRIGGER_WORD} LoRA training
# Lines starting with # are ignored

${REALISM_POSITIVE_PROMPT}, blow job, the female is sucking the male's dick, front view --w ${SAMPLE_SIZE} --h ${SAMPLE_SIZE} --s ${SAMPLE_STEPS} --l ${SAMPLE_CFG_SCALE} --n ${REALISM_NEGATIVE_PROMPT}
${REALISM_POSITIVE_PROMPT}, blow job, the female is sucking the male's dick, side view --w ${SAMPLE_SIZE} --h ${SAMPLE_SIZE} --s ${SAMPLE_STEPS} --l ${SAMPLE_CFG_SCALE} --n ${REALISM_NEGATIVE_PROMPT}
${REALISM_POSITIVE_PROMPT}, blow job, the female is sucking the male's dick, close-up --w ${SAMPLE_SIZE} --h ${SAMPLE_SIZE} --s ${SAMPLE_STEPS} --l ${SAMPLE_CFG_SCALE} --n ${REALISM_NEGATIVE_PROMPT}
EOF


# === EXECUTION ===
echo "Starting SDXL LoRA DreamBooth training..."
echo "Model: ${MODEL_PATH}"
echo "Config: ${DATASET_CONFIG}"
mkdir -p "${OUTPUT_DIR}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors"
echo "Sample prompts: ${SAMPLE_PROMPTS_FILE}"
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
  --sample_every_n_epochs="${SAMPLE_EVERY_N_EPOCHS}" \
  --save_n_epoch_ratio="${SAVE_N_EPOCH_RATIO}" \
  --log_with="wandb" \
  --log_tracker_name="SpicyChat-CharacterConsistency" \
  --log_config \
  --wandb_run_name="${OUTPUT_NAME}" \
  --logging_dir="${OUTPUT_DIR}/logs" \
  --log_prefix="${OUTPUT_NAME}" \
  ${SAMPLE_PROMPTS_FILE:+--sample_prompts="${SAMPLE_PROMPTS_FILE}"} \
  ${SAMPLE_PROMPTS_FILE:+--sample_sampler="euler_a"}

echo ""
echo "Training complete! LoRA saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors"
