# SDXL DreamBooth LoRA Training Guide

This directory contains production-ready scripts for training SDXL LoRA models using the DreamBooth method. These scripts work for both single and multi-character scenarios and are optimized for 8-12GB VRAM.

## Table of Contents

- [Quick Start](#quick-start)
- [Dataset Structure](#dataset-structure)
- [File Naming Conventions](#file-naming-conventions)
- [Caption Format](#caption-format)
- [Preparing Training Images](#preparing-training-images)
- [Preparing Regularization Images](#preparing-regularization-images)
- [Configuration Examples](#configuration-examples)
- [Script Usage](#script-usage)
- [Memory Optimization Tips](#memory-optimization-tips)
- [Training Parameters Guide](#training-parameters-guide)
- [Testing Your LoRA](#testing-your-lora)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Prepare Your Dataset

Create the following folder structure:

```
sdxl_dreambooth_lora/
├── dataset/
│   ├── my_character/        # 20-50 training images of your character
│   │   ├── 001.jpg
│   │   ├── 001.txt          # Optional caption file
│   │   ├── 002.jpg
│   │   └── ...
│   └── regularization/      # 100-200 generic person images
│       ├── reg_001.jpg
│       ├── reg_002.jpg
│       └── ...
├── config.toml              # Your configuration (copy from template)
├── train.sh
├── test.sh
└── README.md
```

### 2. Configure Training

```bash
# Copy the template configuration
cp config.toml my_config.toml

# Edit my_config.toml:
# - Set image_dir paths to your dataset folders
# - Set class_tokens with your trigger word (e.g., "ohwx person")
# - Adjust num_repeats if needed (10-20 is typical)
```

### 3. Edit Training Script

Open `train.sh` and edit the variables at the top:

```bash
MODEL_PATH="/path/to/sd_xl_base_1.0.safetensors"  # Your SDXL base model
DATASET_CONFIG="./my_config.toml"                  # Your config file
OUTPUT_NAME="my_character_lora"                    # Output filename
```

Optional adjustments:
- `NETWORK_DIM=32` - Lower to 16 or 8 for less VRAM
- `EPOCHS=10` - Increase for more training
- `LEARNING_RATE="1e-4"` - Lower (5e-5) for more stability

### 4. Run Training

```bash
cd sdxl_dreambooth_lora
chmod +x train.sh test.sh
./train.sh
```

Training will create:
- `output/my_character_lora.safetensors` - Your trained LoRA
- `output/logs/` - Training logs (viewable with tensorboard)

### 5. Test Your LoRA

Edit `test.sh` and set:

```bash
MODEL_PATH="/path/to/sd_xl_base_1.0.safetensors"
LORA_PATH="./output/my_character_lora.safetensors"
PROMPT="ohwx person, portrait, high quality"  # Use your trigger word!
```

Then run:

```bash
./test.sh
```

Generated images will be saved to `generated/`.

---

## Dataset Structure

### Complete Example

```
dataset/
├── my_character/           # Training images
│   ├── 001.jpg            # Image file (any name)
│   ├── 001.txt            # Optional caption file (same name as image)
│   ├── 002.jpg
│   ├── 002.txt
│   ├── portrait_01.png    # Different naming is fine
│   ├── portrait_01.txt
│   └── ...                # 20-50 total images recommended
│
└── regularization/         # Regularization images
    ├── reg_001.jpg        # Image file (any name)
    ├── reg_002.jpg
    ├── person_001.png
    └── ...                # 100-200 images recommended
```

### Multi-Character Example

```
dataset/
├── alice/                 # Character 1
│   ├── alice_01.jpg
│   ├── alice_01.txt
│   └── ...
├── bob/                   # Character 2
│   ├── bob_01.jpg
│   ├── bob_01.txt
│   └── ...
└── regularization/         # Shared regularization
    ├── reg_001.jpg
    └── ...
```

---

## File Naming Conventions

### Image Files
- **Any filename works**: `001.jpg`, `alice_portrait.png`, `IMG_1234.jpg`
- **Supported formats**: `.jpg`, `.png`, `.webp`
- **Aspect ratios**: Any ratio works (bucketing handles this automatically)

### Caption Files
- **Same name as image** with `.txt` extension
- `001.jpg` → `001.txt`
- `alice_portrait.png` → `alice_portrait.txt`
- Optional: If no caption file exists, `class_tokens` from config.toml is used

### Regularization Images
- **No caption files needed**
- Any naming scheme works
- Just place images in the regularization folder

---

## Caption Format

### Basic Caption (No File)
If you don't create `.txt` files, the `class_tokens` from `config.toml` will be used:

```toml
class_tokens = "ohwx person"
```

### Caption File Examples

**Simple caption:**
```
ohwx person, wearing red dress, smiling
```

**Detailed caption:**
```
ohwx person, 1girl, red dress, smile, standing outdoors, garden background, high quality, masterpiece
```

**With keep_tokens_separator (|||):**
```
ohwx person ||| wearing red dress, smiling, outdoors, high quality
```
- Text before `|||` is never shuffled/dropped (trigger word preserved)
- Requires `keep_tokens_separator = "|||"` in config.toml

**Tag-based (booru style):**
```
ohwx person, 1girl, red dress, blue eyes, long hair, smile, outdoors, masterpiece, best quality
```
- Use with `shuffle_caption = true` to randomize tag order
- Set `keep_tokens = 1` to preserve trigger word

### Caption Best Practices

1. **Always include trigger word** (e.g., "ohwx person") at the start
2. **Be specific**: Describe pose, clothing, expression, background
3. **Avoid overfitting**: Vary descriptions across images
4. **Quality tags**: Add "high quality", "masterpiece" for better results
5. **Keep it natural**: Write like you'd prompt SDXL normally

---

## Preparing Training Images

### Quantity
- **Minimum**: 10-15 images
- **Recommended**: 20-50 images
- **Maximum**: 100+ images (diminishing returns, longer training)

### Quality Guidelines

**Good training images:**
- High resolution (1024x1024 or higher)
- Sharp focus on subject
- Clear, well-lit images
- Variety in poses, expressions, angles
- Variety in clothing/hairstyles (if applicable)
- Different backgrounds

**Avoid:**
- Blurry or low-resolution images
- Heavy filters or artistic effects
- Multiple people in frame (unless training multiple characters)
- Extreme crops where subject is tiny
- Heavily compressed JPEGs

### Variety Recommendations

| Aspect | Target Variety |
|--------|---------------|
| **Poses** | 5+ different poses (standing, sitting, profile, etc.) |
| **Expressions** | 3+ expressions (neutral, smiling, serious) |
| **Angles** | 3+ camera angles (front, 3/4 view, side) |
| **Backgrounds** | 3+ different settings (indoor, outdoor, studio) |
| **Clothing** | 3+ different outfits (if applicable) |
| **Lighting** | 2+ lighting conditions (natural, studio) |

### Cropping and Composition

- **Face visibility**: Face should be clearly visible in most images
- **Subject size**: Subject should fill 30-70% of frame
- **Aspect ratios**: Any aspect ratio works (bucketing handles this)
- **No manual cropping needed**: SDXL's bucketing automatically handles different sizes

---

## Preparing Regularization Images

### Purpose
Regularization images prevent **catastrophic forgetting** - they teach the model to maintain general knowledge about the concept class (e.g., "person") while learning your specific subject.

### Quantity
- **Recommended ratio**: 10:1 (regularization:training)
- If you have 20 training images → use 200 regularization images
- If you have 50 training images → use 500 regularization images
- **Minimum**: 100 images
- **Practical range**: 100-500 images

### Content Guidelines

**For person/character training:**
- Generic photos of people
- Diverse ages, genders, ethnicities
- Various poses, clothing, backgrounds
- Similar quality to training images
- NO images of your specific character

### Sources for Regularization Images

**Option 1: Generate with SDXL**
```bash
# Use sdxl_gen_img.py to generate 200+ images
python ../sdxl_gen_img.py \
  --ckpt="sd_xl_base_1.0.safetensors" \
  --prompt="person, portrait, high quality" \
  --batch_size=4 \
  --images_per_prompt=50 \
  --outdir="dataset/regularization"
```

**Option 2: Download from datasets**
- LAION subsets
- CelebA (for faces)
- FFHQ (Flickr-Faces-HQ)
- Any stock photo collection

**Option 3: Use existing collections**
- Personal photo library (excluding your character)
- Stock photos
- Creative commons images

### Regularization Best Practices

1. **Match the concept**: If training "person", use person images. If training "dog", use dog images.
2. **Diversity matters**: More variety = better generalization
3. **Quality consistency**: Similar quality/resolution to training images
4. **No duplicates**: Avoid identical or near-identical images
5. **No captions needed**: Regularization uses only `class_tokens` from config

---

## Configuration Examples

### Example 1: Single Character

**File: `config_single.toml`**

```toml
[general]
enable_bucket = true
bucket_reso_steps = 32
bucket_no_upscale = true
caption_extension = ".txt"
shuffle_caption = false
keep_tokens = 1

# Training images - your character
[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "dataset/my_character"
  class_tokens = "ohwx person"
  num_repeats = 10

# Regularization images
[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "dataset/regularization"
  class_tokens = "person"
  num_repeats = 1
  is_reg = true
```

**Usage:**
```bash
# In train.sh, set:
DATASET_CONFIG="./config_single.toml"
```

---

### Example 2: Multiple Characters

**File: `config_multi.toml`**

```toml
[general]
enable_bucket = true
bucket_reso_steps = 32
bucket_no_upscale = true
caption_extension = ".txt"
shuffle_caption = false
keep_tokens = 1

# Character 1: Alice
[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "dataset/alice"
  class_tokens = "alice1 woman"
  num_repeats = 10

# Character 2: Bob
[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "dataset/bob"
  class_tokens = "bob1 man"
  num_repeats = 10

# Shared regularization
[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "dataset/regularization"
  class_tokens = "person"
  num_repeats = 1
  is_reg = true
```

**Usage:**
```bash
# In train.sh, set:
DATASET_CONFIG="./config_multi.toml"

# When generating:
PROMPT="alice1 woman and bob1 man, standing together"
```

---

### Example 3: Tag-Based Captions

**File: `config_tags.toml`**

```toml
[general]
enable_bucket = true
bucket_reso_steps = 32
bucket_no_upscale = true
caption_extension = ".txt"
shuffle_caption = true        # Shuffle tags
keep_tokens = 1               # Keep trigger word first
keep_tokens_separator = "|||"

[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "dataset/my_character"
  class_tokens = "sks person"
  num_repeats = 15

[[datasets]]
resolution = 1024
batch_size = 1

  [[datasets.subsets]]
  image_dir = "dataset/regularization"
  class_tokens = "person"
  num_repeats = 1
  is_reg = true
```

**Caption file example (`001.txt`):**
```
sks person ||| 1girl, red dress, blue eyes, long hair, smile, outdoors, masterpiece, best quality
```

---

### Example 4: High VRAM (24GB+)

**File: `config_highvram.toml`**

```toml
[general]
enable_bucket = true
bucket_reso_steps = 32
bucket_no_upscale = true
caption_extension = ".txt"

[[datasets]]
resolution = 1024
batch_size = 4              # Larger batch size

  [[datasets.subsets]]
  image_dir = "dataset/my_character"
  class_tokens = "ohwx person"
  num_repeats = 10

[[datasets]]
resolution = 1024
batch_size = 4              # Larger batch size

  [[datasets.subsets]]
  image_dir = "dataset/regularization"
  class_tokens = "person"
  num_repeats = 1
  is_reg = true
```

**In `train.sh`, also increase:**
```bash
NETWORK_DIM=64              # Higher rank
NETWORK_ALPHA=32
LEARNING_RATE="5e-5"        # Lower LR for larger batch
```

---

## Script Usage

### Training Script (`train.sh`)

**Basic usage:**
```bash
# Edit paths in train.sh first
./train.sh
```

**Using environment variables:**
```bash
# Override specific variables
MODEL_PATH="/models/sd_xl_base_1.0.safetensors" \
DATASET_CONFIG="./my_custom_config.toml" \
OUTPUT_NAME="alice_v1" \
./train.sh
```

**With sample generation:**
```bash
# Create prompts.txt with one prompt per line:
echo "ohwx person, portrait" > prompts.txt
echo "ohwx person, full body" >> prompts.txt

# In train.sh, set:
SAMPLE_PROMPTS="./prompts.txt"
SAMPLE_EVERY_N_EPOCHS=2

# Then run
./train.sh
```

**For low VRAM (8GB):**
```bash
# In train.sh, edit:
NETWORK_DIM=16              # Lower rank
NETWORK_ALPHA=8
EPOCHS=15                   # More epochs to compensate

./train.sh
```

---

### Testing Script (`test.sh`)

**Basic usage:**
```bash
# Edit paths and prompt in test.sh first
./test.sh
```

**Quick test with different prompt:**
```bash
PROMPT="ohwx person wearing a red hat, portrait" ./test.sh
```

**Adjust LoRA strength:**
```bash
LORA_WEIGHT=0.7 \
PROMPT="ohwx person, casual outfit" \
./test.sh
```

**Generate multiple images:**
```bash
# Edit test.sh and change:
SEED=-1  # Random seed each time

# Run multiple times
for i in {1..10}; do
  ./test.sh
done
```

**Different resolutions:**
```bash
WIDTH=768 HEIGHT=1344 ./test.sh  # Portrait
WIDTH=1344 HEIGHT=768 ./test.sh  # Landscape
```

---

## Memory Optimization Tips

### For 8GB VRAM

```bash
# In train.sh, use these settings:
NETWORK_DIM=8               # Very low rank
NETWORK_ALPHA=4
EPOCHS=20                   # More epochs to compensate

# Already enabled by default:
# --cache_latents
# --cache_text_encoder_outputs
# --gradient_checkpointing
# --optimizer_type="AdamW8bit"
```

### For 10-12GB VRAM (Recommended)

```bash
# Default settings work well:
NETWORK_DIM=32
NETWORK_ALPHA=16
EPOCHS=10
```

### For 16GB+ VRAM

```bash
NETWORK_DIM=64              # Higher rank
NETWORK_ALPHA=32
# Optionally increase batch_size in config.toml to 2
```

### For 24GB+ VRAM

```bash
NETWORK_DIM=128             # Very high rank
NETWORK_ALPHA=64
# Set batch_size=4 in config.toml
# Can disable some caching for faster training:
# Remove --cache_text_encoder_outputs if using caption shuffling
```

### Additional Optimizations

**Cache latents beforehand** (faster training startup):
```bash
accelerate launch --num_cpu_threads_per_process=1 \
  ../tools/cache_latents.py \
  --pretrained_model_name_or_path="/path/to/sd_xl_base_1.0.safetensors" \
  --dataset_config="./config.toml"
```

**Monitor memory usage:**
```bash
# In another terminal while training:
watch -n 1 nvidia-smi
```

---

## Training Parameters Guide

### Core Parameters

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| `resolution` | 1024 | 1024 | SDXL native resolution |
| `batch_size` | 1 | 1-4 | Higher needs more VRAM |
| `max_train_epochs` | 10-20 | 5-50 | Monitor for overtraining |
| `learning_rate` | 1e-4 | 5e-5 to 5e-4 | Lower = more stable |
| `network_dim` | 32 | 8-128 | LoRA rank, higher = more capacity |
| `network_alpha` | 16 | 4-64 | Often half of dim |
| `num_repeats` (training) | 10-20 | 5-50 | Balance dataset size |
| `num_repeats` (reg) | 1 | 1 | Always 1 for regularization |

### Network Dim (Rank) Guide

| VRAM | Recommended Dim | Notes |
|------|----------------|-------|
| 8GB | 8-16 | Minimal capacity, needs more epochs |
| 10GB | 16-32 | Good balance |
| 12GB | 32-64 | Recommended sweet spot |
| 16GB+ | 64-128 | High capacity, may overfit easily |

### Learning Rate Guide

| Scenario | Learning Rate |
|----------|--------------|
| **First time training** | 1e-4 |
| **More stability needed** | 5e-5 |
| **Faster convergence** | 2e-4 |
| **Large dataset (100+ images)** | 5e-5 |
| **Small dataset (10-20 images)** | 1e-4 to 2e-4 |

### Epoch Guidelines

**How many epochs?**
- Start with 10 epochs
- Check results after 5, 10, 15 epochs
- Stop if you see overtraining (loss stops improving, outputs too similar)

**Signs of undertraining:**
- LoRA doesn't capture character likeness
- Trigger word has minimal effect
- → Increase epochs or learning rate

**Signs of overtraining:**
- All outputs look nearly identical
- Loss increases or plateaus
- Poor generalization to new prompts
- → Reduce epochs, lower learning rate, or add more variety to dataset

---

## Testing Your LoRA

### Finding the Right LoRA Weight

```bash
# Test different weights:
LORA_WEIGHT=0.5 ./test.sh   # Subtle effect
LORA_WEIGHT=0.7 ./test.sh   # Moderate
LORA_WEIGHT=1.0 ./test.sh   # Full strength (default)
LORA_WEIGHT=1.2 ./test.sh   # Stronger
```

**Weight guide:**
- **0.3-0.5**: Subtle character features
- **0.7-0.9**: Balanced (recommended)
- **1.0-1.2**: Strong character likeness
- **1.5+**: May cause artifacts

### Prompt Engineering with Your LoRA

**Basic prompts:**
```
ohwx person, portrait
ohwx person, full body, standing
ohwx person, sitting, casual outfit
```

**Detailed prompts:**
```
ohwx person, wearing red dress, smiling, outdoors, garden background, high quality, masterpiece
ohwx person, professional headshot, business suit, studio lighting, 4k
```

**Testing generalization:**
```
ohwx person as a cyberpunk character
ohwx person in anime style
ohwx person wearing medieval armor
```

### Quality Comparison

Generate the same prompt with and without LoRA:

```bash
# Without LoRA:
python ../sdxl_gen_img.py \
  --ckpt="model.safetensors" \
  --prompt="person, portrait" \
  --outdir="test_without"

# With LoRA:
PROMPT="ohwx person, portrait" ./test.sh
```

Compare results to verify your trigger word creates distinct character.

---

## Troubleshooting

### Training Issues

**Problem: Out of memory (OOM) error**

Solution:
```bash
# Reduce network_dim:
NETWORK_DIM=16              # or even 8
NETWORK_ALPHA=8             # or 4

# Ensure caching is enabled (should be by default):
# --cache_latents
# --cache_text_encoder_outputs
# --gradient_checkpointing

# Reduce batch_size in config.toml to 1
```

**Problem: NaN loss or loss becomes NaN**

Solution:
```bash
# Already enabled by default, but verify:
# --no_half_vae

# Try lower learning rate:
LEARNING_RATE="5e-5"

# Check dataset for corrupted images:
find dataset/ -name "*.jpg" -o -name "*.png" | while read f; do
  identify "$f" > /dev/null 2>&1 || echo "Corrupt: $f"
done
```

**Problem: Training is very slow**

Solution:
```bash
# Pre-cache latents:
accelerate launch --num_cpu_threads_per_process=1 \
  ../tools/cache_latents.py \
  --pretrained_model_name_or_path="${MODEL_PATH}" \
  --dataset_config="./config.toml"

# Verify xformers is working:
python -c "import xformers; print(xformers.__version__)"

# Reduce num_repeats in config.toml if dataset is too large
```

**Problem: Loss doesn't decrease**

Solution:
- Check learning rate (try 1e-4 or 2e-4)
- Verify images are loading (check console output)
- Ensure `class_tokens` is set correctly
- Try higher `num_repeats` (15-20)

---

### Generation Issues

**Problem: Trigger word doesn't work**

Solution:
- Verify you're using the exact trigger word from training
- Check LoRA weight (try 1.0 or higher)
- Ensure LoRA file path is correct
- Verify training completed without errors

**Problem: Poor character likeness**

Solution:
- Train for more epochs (15-20)
- Use higher `num_repeats` (15-20)
- Increase `network_dim` (64 or 128)
- Add more training images (30-50)
- Verify training images are high quality
- Check that captions describe character accurately

**Problem: LoRA works but outputs are too similar**

Solution (overfitting):
- Reduce epochs (try 5-10)
- Add more variety to training images
- Lower `num_repeats` (5-10)
- Add more regularization images
- Use lower `network_dim` (16-32)

**Problem: Artifacts or distorted images**

Solution:
- Lower LoRA weight (0.7-0.9)
- Reduce `network_dim` in training
- Check for corrupted training images
- Verify `--no_half_vae` was used during training

---

### Dataset Issues

**Problem: "No images found" error**

Solution:
```bash
# Verify paths in config.toml are correct:
ls dataset/my_character/
ls dataset/regularization/

# Check that image formats are supported (.jpg, .png, .webp)
```

**Problem: Bucketing errors**

Solution:
```bash
# Verify bucket_reso_steps is 32 (not 64):
# In config.toml:
bucket_reso_steps = 32

# Check that images aren't too small (min ~512px)
```

**Problem: Caption files not loading**

Solution:
```bash
# Verify caption_extension matches your files:
# In config.toml:
caption_extension = ".txt"

# Check file naming:
# image: 001.jpg
# caption: 001.txt (exact same name)

# Verify UTF-8 encoding:
file -i dataset/my_character/*.txt
```

---

### Configuration Issues

**Problem: TOML syntax error**

Solution:
```bash
# Validate TOML syntax:
python -c "import toml; print(toml.load('config.toml'))"

# Common issues:
# - Missing quotes around strings
# - Incorrect [[datasets]] vs [[datasets.subsets]]
# - Typos in parameter names
```

**Problem: Regularization images not used**

Solution:
```toml
# Verify regularization subset has is_reg = true:
[[datasets.subsets]]
image_dir = "dataset/regularization"
class_tokens = "person"
num_repeats = 1
is_reg = true              # Must be present!
```

---

### Performance Tuning

**Goal: Faster training**

```bash
# Use caching (already enabled by default)
# Reduce resolution if acceptable:
resolution = 768           # In config.toml

# Use larger batch_size if VRAM allows:
batch_size = 2             # In config.toml
```

**Goal: Better quality**

```bash
# Use higher network_dim:
NETWORK_DIM=64

# More epochs:
EPOCHS=20

# Lower learning rate for stability:
LEARNING_RATE="5e-5"

# More training images (30-50)
# Better quality training images
```

**Goal: Smaller file size**

```bash
# Use lower network_dim:
NETWORK_DIM=16

# File size ≈ dim × 2MB for SDXL LoRA
# dim=16 → ~32MB
# dim=32 → ~64MB
# dim=64 → ~128MB
```

---

## Advanced Topics

### Using Different Optimizers

Edit `train.sh` to try different optimizers:

```bash
# AdamW8bit (default, good balance):
--optimizer_type="AdamW8bit"

# Adafactor (lowest memory):
--optimizer_type="adafactor" \
--optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"

# Lion8bit (sometimes better quality):
--optimizer_type="Lion8bit"

# Prodigy (adaptive learning rate):
--optimizer_type="Prodigy"
```

### Block-Wise Learning Rates

Train different U-Net blocks with different learning rates:

```bash
# In train.sh, add:
--block_lr "0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002,0.0002"

# 23 comma-separated values for SDXL blocks
# Higher values for output blocks (later in list)
```

### Learning Rate Scheduling

```bash
# Cosine with restarts:
--lr_scheduler="cosine_with_restarts" \
--lr_scheduler_num_cycles=3

# Polynomial decay:
--lr_scheduler="polynomial" \
--lr_scheduler_power=1.0

# With warmup:
--lr_warmup_steps=100
```

### Resume Training

```bash
# In train.sh, add:
--resume="./output/my_lora-000010.safetensors"

# This resumes from epoch 10
```

---

## Additional Resources

### Viewing Training Progress

```bash
# Install tensorboard:
pip install tensorboard

# View logs:
tensorboard --logdir=output/logs

# Open browser to: http://localhost:6006
```

### Merging Multiple LoRAs

```bash
python ../networks/sdxl_merge_lora.py \
  --save_to="merged.safetensors" \
  --models lora1.safetensors lora2.safetensors \
  --ratios 0.7 0.3
```

### Converting to Different Formats

```bash
# LoRA is already in safetensors format (recommended)
# Most SDXL tools support .safetensors directly

# If needed, convert to diffusers format:
# (See main repository documentation)
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check training logs**: `output/logs/` directory
2. **Verify configuration**: Run `python -c "import toml; print(toml.load('config.toml'))"`
3. **Test with minimal config**: Use 5 images, 5 epochs, dim=16
4. **Check main repository issues**: [https://github.com/kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)

---

## Summary

This guide provides everything needed to train SDXL DreamBooth LoRA models:

- ✅ Production-ready scripts (`train.sh`, `test.sh`)
- ✅ Configuration templates and examples
- ✅ Dataset preparation guidelines
- ✅ Memory optimization for 8-12GB VRAM
- ✅ Troubleshooting for common issues
- ✅ Advanced techniques for experienced users

**Quick reminder:**
1. Prepare 20-50 training images + 100-200 regularization images
2. Edit `config.toml` with paths and trigger word
3. Edit `train.sh` with model path
4. Run `./train.sh`
5. Test with `./test.sh`

Happy training!
