# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is kohya-ss's training scripts repository for Stable Diffusion models. It provides training scripts for fine-tuning, DreamBooth, LoRA, Textual Inversion, and SDXL models. This fork focuses exclusively on SDXL training with identity-preserving features using CLIP vision conditioning.

## Essential Commands

### Setup
```bash
# Initial setup (installs AWS CLI, huggingface-cli, and dependencies)
./setup.sh

# Manual installation alternative
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

# Configure accelerate (required before first training)
accelerate config
# Select: This machine, No distributed training, NO, NO, NO, all, fp16 (or bf16)
```

### Training Commands

**SDXL LoRA Training (with identity conditioning):**
```bash
cd sdxl
./sdxl_train.sh  # Edit paths in script first
```

**Standard LoRA Training (SD1.5/2.x):**
```bash
accelerate launch --num_cpu_threads_per_process=8 train_network.py \
  --pretrained_model_name_or_path="model.safetensors" \
  --train_data_dir="dataset/images" \
  --output_dir="output" \
  --network_module=networks.lora \
  --network_dim=32 \
  --resolution=512 \
  --learning_rate=1e-4 \
  --max_train_epochs=10
```

**SDXL Fine-tuning:**
```bash
accelerate launch sdxl_train.py \
  --pretrained_model_name_or_path="sd_xl_base_1.0.safetensors" \
  --train_data_dir="dataset" \
  --resolution=1024 \
  --cache_text_encoder_outputs \
  --cache_latents
```

### Image Generation

**SDXL with LoRA (identity conditioning):**
```bash
cd sdxl
./sdxl_test.sh  # Edit paths and prompts in script
```

**Standard generation:**
```bash
python sdxl_gen_img.py \
  --ckpt="model.safetensors" \
  --network_module=networks.lora \
  --network_weights="lora.safetensors" \
  --prompt="your prompt here" \
  --W=1024 --H=1024
```

### Utility Commands

**Cache latents for faster training:**
```bash
accelerate launch --num_cpu_threads_per_process=1 tools/cache_latents.py \
  --pretrained_model_name_or_path="model.safetensors" \
  --train_data_dir="dataset"
```

**Cache text encoder outputs:**
```bash
accelerate launch tools/cache_text_encoder_outputs.py \
  --pretrained_model_name_or_path="model.safetensors" \
  --train_data_dir="dataset"
```

**Merge LoRA to model:**
```bash
python networks/sdxl_merge_lora.py \
  --sd_model="base_model.safetensors" \
  --save_to="merged_model.safetensors" \
  --models lora1.safetensors lora2.safetensors \
  --ratios 0.8 0.5
```

## Architecture Overview

### Core Training Pipeline

The training system follows a hierarchical structure:

1. **Training Scripts (Entry Points)**
   - `train_network.py` - LoRA/network training for SD1.5/2.x
   - `sdxl_train_network.py` - LoRA training for SDXL (extends `train_network.NetworkTrainer`)
   - `sdxl_train.py` - Full fine-tuning for SDXL
   - `train_db.py` - DreamBooth training
   - `fine_tune.py` - Standard fine-tuning

2. **Library Layer** (`library/`)
   - `train_util.py` - Core training utilities, dataset handling, bucket management
   - `sdxl_train_util.py` - SDXL-specific training utilities
   - `model_util.py` - Model loading/saving, conversion utilities
   - `sdxl_model_util.py` - SDXL model utilities
   - `config_util.py` - Dataset configuration parsing (`.toml` files)
   - `custom_train_functions.py` - Loss functions, SNR weighting, v-parameterization

3. **Network Modules** (`networks/`)
   - `lora.py` - Standard LoRA implementation
   - `dylora.py` - Dynamic LoRA (rank can change during training)
   - `oft.py` - Orthogonal Finetuning
   - `lora_fa.py` - LoRA with full attention
   - Network modules are dynamically loaded via `--network_module=networks.lora`

### Key Architectural Patterns

**Training Class Hierarchy:**
```
NetworkTrainer (train_network.py)
└── SdxlNetworkTrainer (sdxl_train_network.py)
    - Overrides: load_target_model(), load_tokenizer()
    - Adds: CLIP vision model loading for identity conditioning
```

**Dataset System:**
- `BaseSubset` → dataset configuration from `.toml` or command-line
- `DreamBoothDataset` → manages image loading, bucketing, caching
- `BucketManager` → resolution bucketing (groups images by similar aspect ratios)
- Images cached as latents (`.npz` files) for faster training
- Text encoder outputs can be cached (SDXL only with `--cache_text_encoder_outputs`)

**Model Loading Flow:**
1. Detect format: Diffusers vs original checkpoint (`.ckpt`/`.safetensors`)
2. Load U-Net, VAE, Text Encoder(s)
3. Apply memory optimizations (gradient checkpointing, xformers)
4. Optionally cache outputs to reduce memory

**LoRA Training Mechanism:**
- Networks injected into U-Net via `network.prepare_grad_etc()`
- Only network weights require gradients (base model frozen)
- Supports block-wise learning rates (different LR per U-Net block)
- LoRA+ supported (`loraplus_lr_ratio` for asymmetric learning rates)

### Identity-Preserving Feature (Custom Addition)

Located in `sdxl_train_network.py` and training scripts:

**Training Phase:**
- Loads CLIP vision model: `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`
- Processes reference images (`conditioning_data_dir`) alongside target images
- Embeds reference images into 1280-dim vectors
- Conditions U-Net on these embeddings during denoising
- Enable with `--use_identity_conditioning --identity_conditioning_strength=1.0`

**Inference Phase:**
- Pass `--init_image` (reference) and `--clip_vision_strength` to generation script
- CLIP vision model encodes reference image
- U-Net conditioned on this embedding to preserve character identity

**Dataset Structure:**
```
dataset/
├── target_images/      # Training images (matched by filename)
├── control_images/     # Reference images for identity
└── captions/           # Text captions (.txt files)
```

### Important Configuration Concepts

**Bucket Resolution:**
- Training images grouped by similar aspect ratios
- Default step: 64 pixels (use 32 for SDXL with `--bucket_reso_steps=32`)
- Prevents distortion by not forcing all images to same dimensions
- Managed by `BucketManager` in `train_util.py`

**Caption Handling:**
- `keep_tokens_separator` (e.g., `|||`) - tokens before separator never shuffled/dropped
- `secondary_separator` (e.g., `;;;`) - groups tags together during shuffle
- `caption_dropout_rate` - randomly drops caption for unconditional training
- `shuffle_caption` - randomizes tag order (standard danbooru practice)

**Memory Optimization Techniques:**
1. `--cache_latents` - Pre-encode images to latent space
2. `--cache_text_encoder_outputs` - Pre-encode captions (SDXL only, disables caption augmentation)
3. `--gradient_checkpointing` - Trade compute for memory
4. `--fused_backward_pass` - Fuse backward/optimizer step (SDXL only, requires AdaFactor)
5. `--fused_optimizer_groups` - Split optimizer into groups (SDXL only)

**Loss Functions:**
- Standard: MSE (L2) loss
- `--loss_type=huber` or `smooth_l1` - More robust to outliers
- `--huber_schedule=snr` - Schedule Huber parameter by noise level
- `--masked_loss` - Use alpha channel as loss mask
- `--min_snr_gamma` - SNR weighting for better training stability

## Working with This Codebase

### Adding New Network Types

1. Create `networks/my_network.py`
2. Implement class with methods: `__init__`, `apply_to`, `prepare_optimizer_params`, `save_weights`, `load_weights`
3. Use with `--network_module=networks.my_network`

### Modifying Training Logic

- Loss computation: `library/custom_train_functions.py`
- Dataset loading: `library/train_util.py` (see `BaseDataset`, `DreamBoothDataset`)
- SDXL-specific: `library/sdxl_train_util.py`
- Main training loop: `train_network.py` → `NetworkTrainer.train()`

### Dataset Configuration Files

Prefer `.toml` over command-line for complex setups:

```toml
[general]
enable_bucket = true
resolution = 512

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "path/to/images"
  caption_extension = ".txt"
  num_repeats = 10
  keep_tokens = 1  # Don't shuffle first token
```

Use with `--dataset_config=config.toml`

### Testing Changes

- No formal test suite exists
- Test training: use small dataset (5-10 images), low epochs (2-3)
- Monitor loss curve in tensorboard: `tensorboard --logdir=logs`
- Generate sample images during training: `--sample_every_n_steps=100 --sample_prompts=prompts.txt`

## Common Issues

**Out of Memory (OOM):**
- Enable all caching options: `--cache_latents --cache_text_encoder_outputs`
- Use `--gradient_checkpointing`
- Reduce `--train_batch_size` to 1
- For SDXL: use `--network_train_unet_only` (don't train text encoders)
- Use 8-bit optimizer: `--optimizer_type=AdamW8bit`

**SDXL NaN Loss:**
- Add `--no_half_vae` (VAE produces NaNs in fp16)
- Check `--max_data_loader_n_workers` (reduce if too high)

**Slow Training:**
1. Cache latents first: `tools/cache_latents.py`
2. Cache text encoder: `tools/cache_text_encoder_outputs.py`
3. Increase `--train_batch_size` if memory allows
4. Use `--persistent_data_loader_workers`

**Caption Issues:**
- Cannot use `--shuffle_caption` with `--cache_text_encoder_outputs`
- `keep_tokens_separator` requires exact format in captions
- Wildcards require `enable_wildcard=true` in dataset config

## Version-Specific Notes

**PyTorch Version:**
- Tested with PyTorch 2.1.2
- PyTorch 2.2+ should work but untested
- Match xformers version to PyTorch version

**SDXL Requirements:**
- 24GB VRAM minimum for fine-tuning (with aggressive optimizations)
- 10GB VRAM for LoRA training
- `--bucket_reso_steps` must be ≥32 (not 64 like SD1.5)

**DeepSpeed Support:**
- Available but experimental
- Configure with `accelerate config` → choose DeepSpeed
- See PR #1101 for details

## File Paths Referenced

- Training scripts: `train_network.py`, `sdxl_train_network.py`, `sdxl_train.py`
- Core library: `library/train_util.py`, `library/sdxl_train_util.py`, `library/model_util.py`
- Networks: `networks/lora.py`, `networks/dylora.py`, `networks/oft.py`
- Generation: `sdxl_gen_img.py`, `gen_img.py`
- Utilities: `tools/cache_latents.py`, `tools/cache_text_encoder_outputs.py`
- SDXL scripts: `sdxl/sdxl_train.sh`, `sdxl/sdxl_test.sh`, `sdxl/README.md`
