#!/bin/bash

# SDXL Cache Script (Optional)
# Pre-cache latents and text encoder outputs for faster training
# This is especially useful if you'll train multiple times on the same dataset

# ====================
# CONFIGURATION - EDIT THESE PATHS
# ====================

SDXL_MODEL="sd_xl_base_1.0.safetensors"
TRAIN_DATA_DIR="dataset/target_images"
CAPTION_EXTENSION=".txt"

# Cache settings
BATCH_SIZE=4  # Can be higher than training batch size for faster caching
RESOLUTION="1024,1024"

# ====================
# CACHE LATENTS
# ====================

echo "Caching latents for faster training..."
echo "This may take several minutes depending on dataset size..."
echo ""

python ../tools/cache_latents.py \
  --pretrained_model_name_or_path="${SDXL_MODEL}" \
  --train_data_dir="${TRAIN_DATA_DIR}" \
  --batch_size=${BATCH_SIZE} \
  --max_resolution="${RESOLUTION}" \
  --mixed_precision="bf16"

echo ""
echo "✓ Latents cached successfully!"

# ====================
# CACHE TEXT ENCODER OUTPUTS
# ====================

echo ""
echo "Caching text encoder outputs for faster training..."
echo ""

python ../tools/cache_text_encoder_outputs.py \
  --pretrained_model_name_or_path="${SDXL_MODEL}" \
  --train_data_dir="${TRAIN_DATA_DIR}" \
  --caption_extension="${CAPTION_EXTENSION}" \
  --batch_size=${BATCH_SIZE} \
  --max_token_length=225 \
  --max_resolution="${RESOLUTION}" \
  --mixed_precision="bf16"

echo ""
echo "✓ Text encoder outputs cached successfully!"
echo ""
echo "You can now run sdxl_train.sh for faster training."
echo "The training script already includes --cache_latents and --cache_text_encoder_outputs flags."

# ====================
# NOTES
# ====================
# - Cached files are stored as .npz files alongside your images
# - If you modify captions, you'll need to re-run this script
# - If you add/remove images, you'll need to re-run this script
# - Cache files can be large (several GB for large datasets)
# - Caching saves ~30-50% training time by pre-computing VAE latents and text embeddings
# - CLIP vision embeddings are computed on-the-fly during training (very fast)
