# -*- coding: utf-8 -*-
"""
MedSAM wrapper for ReaSeg.

This wrapper keeps the MedSAM/SAM image encoder, prompt encoder, and mask decoder
in one clean module.

Design choice:
    ReaSeg does NOT modify PromptEncoder.forward(text_embeds=...).
    Instead, ReaSegProjector directly generates sparse prompt embeddings and
    feeds them into MedSAM mask_decoder.

Main flow:
    medsam_images -> image_encoder -> image_embeddings
    seg_prompt_embeddings -> sparse_prompt_embeddings
    no_mask_embed -> dense_prompt_embeddings
    image_embeddings + sparse/dense prompts -> mask_decoder -> masks
"""

import os
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .segment_anything import build_sam_vit_b
except Exception:
    from model.segment_anything import build_sam_vit_b


def _as_hw_tuple(size_like: Any) -> Tuple[int, int]:
    """
    Convert tuple/list/tensor to (H, W).
    """
    if isinstance(size_like, torch.Tensor):
        size_like = size_like.detach().cpu().tolist()

    if isinstance(size_like, (list, tuple)):
        if len(size_like) < 2:
            raise ValueError(f"Invalid size object: {size_like}")
        return int(size_like[0]), int(size_like[1])

    raise TypeError(f"Unsupported size type: {type(size_like)}")


def load_medsam_checkpoint(
    sam_model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> Dict[str, List[str]]:
    """
    Load MedSAM/SAM checkpoint into build_sam_vit_b() model.

    Supports checkpoint formats:
        - raw state_dict
        - {"model": state_dict}
        - {"state_dict": state_dict}

    Returns:
        {
            "missing_keys": [...],
            "unexpected_keys": [...]
        }
    """
    if not checkpoint_path:
        logging.warning("No MedSAM checkpoint path provided.")
        return {"missing_keys": [], "unexpected_keys": []}

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    load_msg = sam_model.load_state_dict(state_dict, strict=strict)

    missing_keys = list(getattr(load_msg, "missing_keys", []))
    unexpected_keys = list(getattr(load_msg, "unexpected_keys", []))

    logging.info(
        f"Loaded MedSAM/SAM checkpoint from {checkpoint_path}. "
        f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
    )

    if missing_keys:
        logging.warning(f"MedSAM missing keys example: {missing_keys[:30]}")

    if unexpected_keys:
        logging.warning(f"MedSAM unexpected keys example: {unexpected_keys[:30]}")

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


class MedSAMWrapper(nn.Module):
    """
    Wrapper around MedSAM/SAM ViT-B.

    Args:
        checkpoint_path:
            Path to medsam_vit_b.pth.
        image_size:
            Padded SAM input size, usually 1024.
        freeze_image_encoder:
            Whether to freeze image encoder.
        freeze_prompt_encoder:
            Whether to freeze prompt encoder.
        freeze_mask_decoder:
            Whether to freeze mask decoder.
        keep_fp32_decoder:
            Keep prompt encoder and mask decoder in fp32.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        image_size: int = 1024,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = True,
        keep_fp32_decoder: bool = True,
        strict_load: bool = False,
    ) -> None:
        super().__init__()

        self.image_size = int(image_size)
        self.checkpoint_path = checkpoint_path
        self.strict_load = bool(strict_load)

        self.sam = build_sam_vit_b()

        if checkpoint_path is not None:
            self.load_info = load_medsam_checkpoint(
                self.sam,
                checkpoint_path=checkpoint_path,
                strict=strict_load,
            )
        else:
            self.load_info = {"missing_keys": [], "unexpected_keys": []}
            logging.warning("Using randomly initialized MedSAM/SAM.")

        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder
        self.mask_decoder = self.sam.mask_decoder

        if freeze_image_encoder:
            self.freeze_image_encoder()

        if freeze_prompt_encoder:
            self.freeze_prompt_encoder()

        if freeze_mask_decoder:
            self.freeze_mask_decoder()

        if keep_fp32_decoder:
            self.keep_fp32_decoder()

    # --------------------------------------------------------
    # dtype / trainability helpers
    # --------------------------------------------------------

    def keep_fp32_decoder(self) -> None:
        """
        Keep the full MedSAM branch in fp32.
        """
        self.force_fp32_all()


    def keep_fp32_all(self) -> None:
        """
        Explicit alias for keeping the full MedSAM branch in fp32.
        """
        self.force_fp32_all()

    def force_fp32_all(self) -> None:
        """
        Force the whole MedSAM branch to fp32.

        This is required for partial mask-decoder training under bf16 DeepSpeed,
        because mask_decoder internals may otherwise become mixed fp32/bf16.
        """
        self.sam.float()
        self.image_encoder.float()
        self.prompt_encoder.float()
        self.mask_decoder.float()

    def reload_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Reload MedSAM/SAM checkpoint.

        This is important when ReaSeg is created with HuggingFace from_pretrained().
        HF first constructs the whole ReaSeg model, then loads the Qwen checkpoint.
        Since medsam.* keys are missing from the Qwen checkpoint, custom MedSAM
        modules may be treated as newly initialized. Therefore, reload MedSAM after
        from_pretrained() in the training script.
        """
        ckpt = checkpoint_path or self.checkpoint_path

        if ckpt is None or ckpt == "":
            raise ValueError("No MedSAM checkpoint path is available for reload.")

        self.load_info = load_medsam_checkpoint(
            self.sam,
            checkpoint_path=ckpt,
            strict=self.strict_load,
        )

        self.keep_fp32_all()

        bad = self.find_nonfinite_medsam_tensors(max_show=20)
        if bad:
            for item in bad:
                logging.error(
                    "[MedSAM Finite Check] "
                    f"name={item['name']}, kind={item['kind']}, "
                    f"shape={item['shape']}, dtype={item['dtype']}, "
                    f"has_nan={item['has_nan']}, has_inf={item['has_inf']}"
                )
            raise RuntimeError(
                "Non-finite tensors found in MedSAM after checkpoint reload."
            )

        logging.info(f"Reloaded MedSAM checkpoint after HF from_pretrained: {ckpt}")


    def find_nonfinite_medsam_tensors(self, max_show: int = 50):
        """
        Check all MedSAM parameters and buffers, including positional encoding buffers.
        """
        bad = []

        for name, p in self.sam.named_parameters():
            with torch.no_grad():
                x = p.detach()
                if not torch.isfinite(x).all():
                    bad.append(
                        {
                            "kind": "parameter",
                            "name": name,
                            "shape": tuple(x.shape),
                            "dtype": str(x.dtype),
                            "has_nan": torch.isnan(x).any().item(),
                            "has_inf": torch.isinf(x).any().item(),
                        }
                    )
                    if len(bad) >= max_show:
                        return bad

        for name, b in self.sam.named_buffers():
            with torch.no_grad():
                x = b.detach()
                if not torch.isfinite(x).all():
                    bad.append(
                        {
                            "kind": "buffer",
                            "name": name,
                            "shape": tuple(x.shape),
                            "dtype": str(x.dtype),
                            "has_nan": torch.isnan(x).any().item(),
                            "has_inf": torch.isinf(x).any().item(),
                        }
                    )
                    if len(bad) >= max_show:
                        return bad

        return bad

    def keep_fp32_all(self) -> None:
        """
        Explicit alias for keeping the full MedSAM branch in fp32.
        """
        self.keep_fp32_decoder()

    def freeze_image_encoder(self) -> None:
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        self.image_encoder.eval()

    def freeze_prompt_encoder(self) -> None:
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False
        self.prompt_encoder.eval()

    def freeze_mask_decoder(self) -> None:
        for p in self.mask_decoder.parameters():
            p.requires_grad = False
        self.mask_decoder.eval()

    def enable_image_encoder_training(self, enabled: bool = True) -> None:
        for p in self.image_encoder.parameters():
            p.requires_grad = bool(enabled)
        self.image_encoder.train(enabled)

    def enable_prompt_encoder_training(self, enabled: bool = True) -> None:
        for p in self.prompt_encoder.parameters():
            p.requires_grad = bool(enabled)
        self.prompt_encoder.train(enabled)

    def set_mask_decoder_train_mode(self, mode: str = "none") -> None:
        """
        Configure trainable mask decoder parameters.

        Args:
            mode:
                none:
                    Freeze all mask decoder parameters.

                partial:
                    Train:
                        - iou_token
                        - mask_tokens
                        - output_hypernetworks_mlps
                        - iou_prediction_head

                head_plus_upscaling:
                    partial + output_upscaling

                full:
                    Train all mask decoder parameters.
        """
        mode = str(mode).lower()

        for p in self.mask_decoder.parameters():
            p.requires_grad = False

        if mode == "none":
            self.mask_decoder.eval()
            return

        if mode == "partial":
            keys = [
                "iou_token",
                "mask_tokens",
                "output_hypernetworks_mlps",
                "iou_prediction_head",
            ]
        elif mode == "head_plus_upscaling":
            keys = [
                "iou_token",
                "mask_tokens",
                "output_hypernetworks_mlps",
                "iou_prediction_head",
                "output_upscaling",
            ]
        elif mode == "full":
            keys = [""]
        else:
            raise ValueError(
                f"Unsupported mask decoder train mode: {mode}. "
                "Expected one of: none, partial, head_plus_upscaling, full."
            )

        for name, p in self.mask_decoder.named_parameters():
            if any(k in name for k in keys):
                p.requires_grad = True

        self.mask_decoder.train()

    # --------------------------------------------------------
    # core MedSAM operations
    # --------------------------------------------------------

    def encode_image(self, medsam_images: torch.Tensor) -> torch.Tensor:
        """
        Encode MedSAM-preprocessed images.

        Args:
            medsam_images:
                Tensor with shape [B, 3, image_size, image_size].
                It should already be normalized and padded by the dataset.

        Returns:
            image_embeddings:
                Tensor from MedSAM/SAM image encoder.
        """
        if medsam_images.dim() == 3:
            medsam_images = medsam_images.unsqueeze(0)

        if medsam_images.dim() != 4:
            raise ValueError(
                f"medsam_images should be [B, 3, H, W], got {tuple(medsam_images.shape)}"
            )

        first_param = next(self.image_encoder.parameters())
        device = first_param.device
        dtype = first_param.dtype

        medsam_images = medsam_images.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )

        has_trainable_params = any(p.requires_grad for p in self.image_encoder.parameters())

        if has_trainable_params:
            return self.image_encoder(medsam_images)

        with torch.no_grad():
            return self.image_encoder(medsam_images)

    def get_image_pe(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Return dense positional encoding for mask decoder.
        """
        return self.prompt_encoder.get_dense_pe().to(
            device=device,
            dtype=dtype,
        )

    def build_no_mask_dense_prompt(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Build dense prompt embedding corresponding to "no mask input".

        Returns:
            Tensor with shape [B, 256, H_embed, W_embed].
        """
        image_embedding_size = self.prompt_encoder.image_embedding_size

        dense_prompt = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        dense_prompt = dense_prompt.expand(
            batch_size,
            -1,
            image_embedding_size[0],
            image_embedding_size[1],
        )

        return dense_prompt.to(
            device=device,
            dtype=dtype,
        )

    def decode_masks(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        check_numerics: bool = True,
    ):
        """
        Decode masks using MedSAM mask decoder.

        Args:
            image_embeddings:
                Tensor with shape [B, C, H, W].
            sparse_prompt_embeddings:
                Tensor with shape [B, N_prompt, 256].
            dense_prompt_embeddings:
                Optional dense prompt tensor. If None, no-mask prompt is used.
            multimask_output:
                Whether to output multiple masks.
            check_numerics:
                If True, raise RuntimeError when decoder output has NaN/Inf.

        Returns:
            low_res_masks, iou_predictions
        """
        if image_embeddings.dim() != 4:
            raise ValueError(
                f"image_embeddings should be [B, C, H, W], got {tuple(image_embeddings.shape)}"
            )

        if sparse_prompt_embeddings.dim() != 3:
            raise ValueError(
                "sparse_prompt_embeddings should be [B, N_prompt, C], "
                f"got {tuple(sparse_prompt_embeddings.shape)}"
            )

        batch_size = image_embeddings.shape[0]
        device = image_embeddings.device

        if dense_prompt_embeddings is None:
            dense_prompt_embeddings = self.build_no_mask_dense_prompt(
                batch_size=batch_size,
                device=device,
                dtype=torch.float32,
            )

        # Force MedSAM decoder path to fp32 before every decoder forward.
        # This avoids mixed dtype inside mask_decoder under bf16 DeepSpeed.
        self.force_fp32_all()

        decoder_param = next(self.mask_decoder.parameters())
        decoder_device = decoder_param.device
        decoder_dtype = torch.float32

        with torch.amp.autocast(device_type="cuda", enabled=False):
            image_embeddings = image_embeddings.to(
                device=decoder_device,
                dtype=decoder_dtype,
                non_blocking=True,
            )

            sparse_prompt_embeddings = sparse_prompt_embeddings.to(
                device=decoder_device,
                dtype=decoder_dtype,
                non_blocking=True,
            )

            dense_prompt_embeddings = dense_prompt_embeddings.to(
                device=decoder_device,
                dtype=decoder_dtype,
                non_blocking=True,
            )

            image_pe = self.get_image_pe(
                device=decoder_device,
                dtype=decoder_dtype,
            )
            # if check_numerics:
            #     def _stat(name, x):
            #         x_float = x.detach().float()
            #         logging.info(
            #             f"[MedSAM Decode Input] {name}: "
            #             f"shape={tuple(x.shape)}, dtype={x.dtype}, "
            #             f"nan={torch.isnan(x_float).any().item()}, "
            #             f"inf={torch.isinf(x_float).any().item()}, "
            #             f"min={x_float.min().item():.6f}, "
            #             f"max={x_float.max().item():.6f}, "
            #             f"mean={x_float.mean().item():.6f}, "
            #             f"absmax={x_float.abs().max().item():.6f}"
            #         )

            #     _stat("image_embeddings", image_embeddings)
            #     _stat("image_pe", image_pe)
            #     _stat("sparse_prompt_embeddings", sparse_prompt_embeddings)
            #     _stat("dense_prompt_embeddings", dense_prompt_embeddings)
            
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=multimask_output,
            )

        if check_numerics:
            # low_float = low_res_masks.detach().float()
            # iou_float = iou_predictions.detach().float()

            # logging.info(
            #     "[MedSAM Decode Output] "
            #     f"low_res_masks: shape={tuple(low_res_masks.shape)}, dtype={low_res_masks.dtype}, "
            #     f"nan={torch.isnan(low_float).any().item()}, "
            #     f"inf={torch.isinf(low_float).any().item()}, "
            #     f"min={low_float.nan_to_num().min().item():.6f}, "
            #     f"max={low_float.nan_to_num().max().item():.6f}, "
            #     f"mean={low_float.nan_to_num().mean().item():.6f}, "
            #     f"absmax={low_float.nan_to_num().abs().max().item():.6f}; "
            #     f"iou_predictions: shape={tuple(iou_predictions.shape)}, dtype={iou_predictions.dtype}, "
            #     f"nan={torch.isnan(iou_float).any().item()}, "
            #     f"inf={torch.isinf(iou_float).any().item()}"
            # )

            if not torch.isfinite(low_res_masks).all():
                raise RuntimeError(
                    "MedSAM mask decoder produced non-finite mask logits. "
                    "Check the [MedSAM Decode Input] logs above to locate whether "
                    "image_embeddings, image_pe, sparse_prompt_embeddings, or "
                    "dense_prompt_embeddings is abnormal. "
                    f"nan={torch.isnan(low_res_masks).any().item()}, "
                    f"inf={torch.isinf(low_res_masks).any().item()}"
                )

        return low_res_masks, iou_predictions

    def postprocess_masks(
        self,
        low_res_masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Postprocess low-res masks to original image size.

        Args:
            low_res_masks:
                Raw low-res masks from mask decoder.
            input_size:
                Resized image size before padding, as (H, W).
            original_size:
                Original image size, as (H, W).

        Returns:
            Masks resized to original image size.
        """
        input_size = _as_hw_tuple(input_size)
        original_size = _as_hw_tuple(original_size)

        low_res_masks = low_res_masks.float().clamp(-20.0, 20.0)

        return self.sam.postprocess_masks(
            low_res_masks,
            input_size=input_size,
            original_size=original_size,
        )

    def decode_and_postprocess(
        self,
        image_embedding: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        original_size: Tuple[int, int],
        resize_shape: Tuple[int, int],
        multimask_output: bool = False,
        check_numerics: bool = True,
    ) -> torch.Tensor:
        """
        Convenience function for one sample.

        Args:
            image_embedding:
                Tensor with shape [1, C, H, W].
            sparse_prompt_embeddings:
                Tensor with shape [1, N_prompt, 256].
            original_size:
                Original image size (H, W).
            resize_shape:
                Resized image size before padding (H, W).

        Returns:
            postprocessed mask tensor.
            Usually [1, N_mask, H_original, W_original].
        """
        low_res_masks, _ = self.decode_masks(
            image_embeddings=image_embedding,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=None,
            multimask_output=multimask_output,
            check_numerics=check_numerics,
        )

        return self.postprocess_masks(
            low_res_masks,
            input_size=resize_shape,
            original_size=original_size,
        )