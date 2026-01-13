import argparse

import torch
from library.device_utils import init_ipex, clean_memory_on_device
init_ipex()

from library import sdxl_model_util, sdxl_train_util, train_util
import train_network
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

# Import for CLIP vision model (identity conditioning)
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(self, args, train_dataset_group):
        sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        # Load CLIP vision model for identity conditioning if requested
        if hasattr(args, 'use_identity_conditioning') and args.use_identity_conditioning:
            CLIP_VISION_MODEL = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            logger.info(f"Loading CLIP vision model for identity conditioning: {CLIP_VISION_MODEL}")

            try:
                clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
                    CLIP_VISION_MODEL,
                    projection_dim=1280
                ).to(accelerator.device, dtype=weight_dtype)
                clip_vision_processor = CLIPImageProcessor.from_pretrained(CLIP_VISION_MODEL)

                # Freeze CLIP vision model - only LoRA weights will be trained
                clip_vision_model.requires_grad_(False)
                clip_vision_model.eval()

                self.clip_vision_model = clip_vision_model
                self.clip_vision_processor = clip_vision_processor
                logger.info("CLIP vision model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CLIP vision model: {e}")
                logger.error("Please check network connection or download the model manually")
                raise
        else:
            self.clip_vision_model = None
            self.clip_vision_processor = None

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def load_tokenizer(self, args):
        tokenizer = sdxl_train_util.load_tokenizers(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            with accelerator.autocast():
                dataset.cache_text_encoder_outputs(
                    tokenizers,
                    text_encoders,
                    accelerator.device,
                    weight_dtype,
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )

            text_encoders[0].to("cpu", dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

        # Pass CLIP vision model to ControlNetDataset if identity conditioning is enabled
        if hasattr(args, 'use_identity_conditioning') and args.use_identity_conditioning:
            if hasattr(self, 'clip_vision_model') and self.clip_vision_model is not None:
                logger.info("Passing CLIP vision model to ControlNetDataset for identity conditioning")
                for dataset in dataset.datasets if hasattr(dataset, 'datasets') else [dataset]:
                    if hasattr(dataset, 'clip_vision_model'):
                        dataset.clip_vision_model = self.clip_vision_model
                        dataset.clip_vision_processor = self.clip_vision_processor
                        logger.info(f"CLIP vision model assigned to dataset: {type(dataset).__name__}")

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                # Get the text embedding for conditioning
                # TODO support weighted captions
                # if args.weighted_captions:
                #     encoder_hidden_states = get_weighted_text_embeddings(
                #         tokenizer,
                #         text_encoder,
                #         batch["captions"],
                #         accelerator.device,
                #         args.max_token_length // 75 if args.max_token_length else 1,
                #         clip_skip=args.clip_skip,
                #     )
                # else:
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                    accelerator=accelerator,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

            # # verify that the text encoder outputs are correct
            # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
            #     args.max_token_length,
            #     batch["input_ids"].to(text_encoders[0].device),
            #     batch["input_ids2"].to(text_encoders[0].device),
            #     tokenizers[0],
            #     tokenizers[1],
            #     text_encoders[0],
            #     text_encoders[1],
            #     None if not args.full_fp16 else weight_dtype,
            # )
            # b_size = encoder_hidden_states1.shape[0]
            # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # logger.info("text encoder outputs verified")

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds

        # Inject CLIP vision embeddings for identity conditioning if available
        if "clip_vision_embeddings" in batch and hasattr(args, 'use_identity_conditioning') and args.use_identity_conditioning:
            clip_vision_embeds = batch["clip_vision_embeddings"].to(accelerator.device, dtype=weight_dtype)

            # Apply strength scaling if specified
            if hasattr(args, 'identity_conditioning_strength'):
                clip_vision_embeds = clip_vision_embeds * args.identity_conditioning_strength

            # Concatenate CLIP vision embeddings with text pool
            # pool2: [B, 1280], clip_vision_embeds: [B, 1280] -> combined_pool: [B, 2560]
            combined_pool = torch.cat([pool2, clip_vision_embeds], dim=1)

            # For now, test if U-Net accepts [B, 2816] vector_embedding
            # If it fails, we'll need to add a projection layer to reduce [B, 2560] -> [B, 1280]
            vector_embedding = torch.cat([combined_pool, embs], dim=1).to(weight_dtype)
        else:
            # Standard path without identity conditioning
            vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)

        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)

    # Add identity conditioning arguments
    parser.add_argument(
        "--use_identity_conditioning",
        action="store_true",
        help="Enable identity-preserving image conditioning using CLIP vision embeddings from conditioning images. Requires conditioning_data_dir in dataset config. / conditioning画像のCLIP vision埋め込みを使用した同一性保持画像条件付けを有効化。データセット設定にconditioning_data_dirが必要。"
    )
    parser.add_argument(
        "--identity_conditioning_strength",
        type=float,
        default=1.0,
        help="Strength multiplier for identity conditioning embeddings (0.0-1.0). Higher values strengthen identity preservation. Default: 1.0 / 同一性条件付け埋め込みの強度倍率 (0.0-1.0)。値が大きいほど同一性保持が強くなる。デフォルト: 1.0"
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlNetworkTrainer()
    trainer.train(args)
