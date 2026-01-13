# SDXL Identity-Preserving LoRA Training Scripts

This folder contains scripts for training and testing SDXL LoRA models with character identity conditioning.

## Quick Start

### 1. Prepare Your Dataset

Create a dataset folder with this structure:

```
dataset/
├── target_images/          # Training images (what you want to generate)
│   ├── 001.jpg            # Character in pose 1
│   ├── 002.jpg            # Character in pose 2
│   └── 003.jpg            # Character in pose 3
├── reference_images/       # Character reference images (for identity)
│   ├── 001.jpg            # Same character, pose/angle for image 001
│   ├── 002.jpg            # Same character, pose/angle for image 002
│   └── 003.jpg            # Same character, pose/angle for image 003
└── captions/
    ├── 001.txt            # "1girl, standing, outdoor"
    ├── 002.txt            # "1girl, sitting, indoor"
    └── 003.txt            # "1girl, jumping, studio"
```

**Important**: Reference images must have the same filename as target images (they're matched by name).

### 2. Edit Configuration

Edit the paths in `sdxl_train.sh`:

```bash
PRETRAINED_MODEL="path/to/sd_xl_base_1.0.safetensors"
TRAIN_DATA_DIR="dataset/target_images"
CONDITIONING_DATA_DIR="dataset/reference_images"
```

### 3. Train

```bash
cd sdxl
./sdxl_train.sh
```

First run will download the CLIP vision model (~4GB). Training progress is logged to TensorBoard.

### 4. Test/Generate Images

After training, edit `sdxl_test.sh` with your paths and run:

```bash
./sdxl_test.sh
```

This generates images using a reference image to preserve character identity.

## Scripts Overview

### `sdxl_train.sh`
Main training script with identity conditioning enabled. Key parameters:

- `USE_IDENTITY=true` - Enable identity conditioning (set to `false` for regular LoRA)
- `IDENTITY_STRENGTH=1.0` - How strongly to preserve identity (0.0-1.0)
- `NETWORK_DIM=32` - LoRA rank (higher = more capacity)
- `LEARNING_RATE="1e-4"` - Training speed

### `sdxl_test.sh`
Image generation script. Key parameters:

- `REFERENCE_IMAGE` - Character reference for identity preservation
- `CLIP_VISION_STRENGTH=1.0` - Identity conditioning strength at inference
- `PROMPT` - What to generate
- `NUM_IMAGES=4` - How many images to generate

### `sdxl_cache.sh` (Optional)
Pre-caches latents and text encoder outputs for 30-50% faster training. Run before training if you plan to train multiple times on the same dataset.

## What is Identity Conditioning?

Traditional LoRA training learns character features from multiple images but can't guarantee consistent identity across different poses/expressions.

**Identity conditioning** uses a reference image during both training AND inference to preserve character appearance:

- **Training**: Each target image is paired with a reference image showing the same character
- **Inference**: You provide a reference image, and the model preserves that character's identity

This is similar to IP-Adapter but implemented as pure LoRA without extra modules.

## Tips & Troubleshooting

### Memory Issues

If you run out of VRAM:

1. Reduce `NETWORK_DIM` to 16 or 8
2. Use `MIXED_PRECISION="fp16"` instead of "bf16"
3. Reduce `BATCH_SIZE` to 1 (if not already)
4. Remove `--gradient_checkpointing` (uses more VRAM but faster)

### Identity Not Preserved

If generated images don't match the reference character:

1. Increase `IDENTITY_STRENGTH` to 1.5 or 2.0 (in both training and testing)
2. Train for more epochs (20-30 instead of 10)
3. Use higher quality reference images (clear face, good lighting)
4. Increase `NETWORK_DIM` to 64 or 128

### Training Too Slow

1. Run `./sdxl_cache.sh` first to pre-cache latents
2. Reduce `MAX_TRAIN_EPOCHS`
3. Increase `BATCH_SIZE` if you have VRAM
4. Remove `--shuffle_caption` (minor speedup)

### Comparison Test

To see the difference identity conditioning makes, generate images with and without reference:

```bash
# With identity (in sdxl_test.sh)
--clip_vision_strength=1.0 \
--init_image="reference.jpg"

# Without identity (uncomment comparison section in sdxl_test.sh)
# (no --clip_vision_strength or --init_image flags)
```

## Example Workflow

```bash
# 1. Prepare dataset (see structure above)

# 2. Optional: Pre-cache for faster training
./sdxl_cache.sh

# 3. Train LoRA
./sdxl_train.sh

# 4. Generate test images
./sdxl_test.sh

# 5. Compare results with different identity strengths
# Edit CLIP_VISION_STRENGTH in sdxl_test.sh: try 0.5, 1.0, 1.5
./sdxl_test.sh
```

## Advanced: Training Config File (Alternative)

Instead of command-line arguments, you can use a `.toml` config file:

```toml
[general]
resolution = 1024
enable_bucket = true

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "dataset/target_images"
  conditioning_data_dir = "dataset/reference_images"
  caption_extension = ".txt"
  num_repeats = 10
  use_identity_conditioning = true
```

Then train with:

```bash
accelerate launch ../sdxl_train_network.py \
  --dataset_config=config.toml \
  --use_identity_conditioning \
  --identity_conditioning_strength=1.0 \
  ... (other args)
```

## Technical Details

- **CLIP Vision Model**: `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` (auto-downloaded)
- **Embedding Size**: 1280 dimensions
- **Memory Overhead**: ~2GB VRAM for CLIP vision model
- **Training Format**: Standard LoRA `.safetensors` (compatible with WebUI, ComfyUI)
- **Inference**: Requires `--clip_vision_strength` and `--init_image` flags

## Support

For issues or questions:
1. Check training logs in `output/character_lora/logs/`
2. Try reducing complexity (lower NETWORK_DIM, simpler dataset)
3. Verify dataset structure matches expected format
4. Check that reference images match target image filenames exactly
