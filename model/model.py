# -*- coding: utf-8 -*-
"""
ReaSeg main model.

Clean data flow:
    Qwen2.5-VL
        -> [SEG] token hidden state
        -> ReaSegProjector
        -> sparse prompt embeddings
        -> MedSAM image encoder + MedSAM mask decoder
        -> segmentation masks

This file intentionally avoids legacy QWSA naming and keeps only the
core ReaSeg training/inference logic.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration

try:
    # Package import: reaseg.model.model
    from .seg_projector import ReaSegProjector
    from .medsam_wrapper import MedSAMWrapper
    from ..losses.mask_losses import compute_mask_losses, stable_lm_ce_loss
except Exception:
    # Script-style import from project root.
    from model.seg_projector import ReaSegProjector
    from model.medsam_wrapper import MedSAMWrapper
    from losses.mask_losses import compute_mask_losses, stable_lm_ce_loss


def _get_hidden_size_from_config(config: Any) -> int:
    """
    Infer Qwen hidden size from HuggingFace config.
    """
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)

    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)

    raise AttributeError(
        "Cannot infer hidden size from config. Expected config.hidden_size "
        "or config.text_config.hidden_size."
    )


def _as_hw_tuple(size_like: Any) -> Tuple[int, int]:
    """
    Convert tuple/list/tensor-like object to (H, W).
    """
    if isinstance(size_like, torch.Tensor):
        size_like = size_like.detach().cpu().tolist()

    if isinstance(size_like, (list, tuple)):
        # Some old dataset versions may store resize_shape as [(H, W)].
        if len(size_like) == 1 and isinstance(size_like[0], (list, tuple, torch.Tensor)):
            return _as_hw_tuple(size_like[0])

        if len(size_like) < 2:
            raise ValueError(f"Invalid size object: {size_like}")

        return int(size_like[0]), int(size_like[1])

    raise TypeError(f"Unsupported size type: {type(size_like)}")


def _get_batch_item(container: Any, index: int) -> Any:
    """
    Get one item from list/tuple/tensor container.
    """
    if isinstance(container, torch.Tensor):
        return container[index]
    return container[index]


class ReaSegForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    ReaSeg-VL model.

    Expected custom training batch fields:

        Preferred new names:
            medsam_images:
                Tensor[B, 3, image_size, image_size].
                Preprocessed for MedSAM/SAM image encoder.

            gt_masks:
                list[Tensor[N_i, H_i, W_i]] or Tensor[B, N, H, W].

            original_sizes:
                list[(H, W)].

            resize_shapes:
                list[(new_h, new_w)].

        Backward-compatible old names:
            images -> medsam_images
            masks_list -> gt_masks
            original_size_list -> original_sizes
            resize_list -> resize_shapes

    Standard Qwen2.5-VL fields:
        input_ids
        attention_mask
        labels
        pixel_values
        image_grid_thw
    """

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(config)

        self.seg_token_idx = kwargs.get("seg_token_idx", None)
        if self.seg_token_idx is None:
            raise ValueError("seg_token_idx must be provided when initializing ReaSeg.")

        self.image_size = int(kwargs.get("image_size", 1024))
        self.prompt_dim = int(kwargs.get("prompt_dim", kwargs.get("out_dim", 256)))

        self.ce_loss_weight = float(kwargs.get("ce_loss_weight", 1.0))
        self.dice_loss_weight = float(kwargs.get("dice_loss_weight", 0.5))
        self.bce_loss_weight = float(kwargs.get("bce_loss_weight", 2.0))

        self.detach_seg_hidden_for_mask = bool(
            kwargs.get("detach_seg_hidden_for_mask", True)
        )

        self.check_mask_numerics = bool(kwargs.get("check_mask_numerics", True))

        hidden_size = _get_hidden_size_from_config(config)

        self.seg_projector = ReaSegProjector(
            hidden_size=hidden_size,
            prompt_dim=self.prompt_dim,
            init_std=float(kwargs.get("projector_init_std", 1e-3)),
            output_scale=float(kwargs.get("projector_output_scale", 1.0)),
            clamp_value=float(kwargs.get("projector_clamp_value", 10.0)),
            use_layernorm=bool(kwargs.get("projector_use_layernorm", True)),
        )

        medsam_checkpoint = kwargs.get(
            "medsam_checkpoint",
            kwargs.get("vision_pretrained", None),
        )

        self.medsam = MedSAMWrapper(
            checkpoint_path=medsam_checkpoint,
            image_size=self.image_size,
            freeze_image_encoder=bool(kwargs.get("freeze_medsam_image_encoder", True)),
            freeze_prompt_encoder=bool(kwargs.get("freeze_medsam_prompt_encoder", True)),
            freeze_mask_decoder=bool(kwargs.get("freeze_medsam_mask_decoder", True)),
            keep_fp32_decoder=bool(kwargs.get("keep_medsam_decoder_fp32", True)),
            strict_load=bool(kwargs.get("strict_medsam_load", False)),
        )

        # Compatibility alias for older training scripts that expect visual_model.
        self.visual_model = self.medsam.sam

        # Default: train script decides what to unfreeze.
        self.enable_seg_projector_training(False)

    # ------------------------------------------------------------------
    # Training-control helpers
    # ------------------------------------------------------------------

    def keep_fp32_modules(self) -> None:
        """
        Keep fragile projection and MedSAM modules in fp32.
        """
        self.seg_projector.float()

        if hasattr(self.medsam, "force_fp32_all"):
            self.medsam.force_fp32_all()
        elif hasattr(self.medsam, "keep_fp32_all"):
            self.medsam.keep_fp32_all()
        else:
            self.medsam.keep_fp32_decoder()
    
    def reload_medsam_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """
        Reload MedSAM checkpoint after HuggingFace from_pretrained().
        """
        if not hasattr(self, "medsam"):
            raise AttributeError("ReaSeg model has no medsam module.")

        self.medsam.reload_checkpoint(checkpoint_path)
        self.visual_model = self.medsam.sam
        self.keep_fp32_modules()

    def freeze_medsam_image_encoder(self) -> None:
        self.medsam.freeze_image_encoder()

    def freeze_medsam_prompt_encoder(self) -> None:
        self.medsam.freeze_prompt_encoder()

    def freeze_medsam_mask_decoder(self) -> None:
        self.medsam.freeze_mask_decoder()

    def enable_seg_projector_training(self, enabled: bool = True) -> None:
        for p in self.seg_projector.parameters():
            p.requires_grad = bool(enabled)

        self.seg_projector.train(bool(enabled))

    def set_mask_decoder_train_mode(self, mode: str = "none") -> None:
        """
        Proxy to MedSAMWrapper.set_mask_decoder_train_mode.

        mode:
            none
            partial
            head_plus_upscaling
            full
        """
        self.medsam.set_mask_decoder_train_mode(mode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_seg_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract hidden states corresponding to [SEG] token positions.

        Returns:
            seg_hidden_states:
                Tensor[N_total_seg, hidden_size]
            seg_token_counts:
                Tensor[B], number of [SEG] tokens in each sample
        """
        seg_token_mask = input_ids.eq(self.seg_token_idx)

        # Align sequence length if Qwen hidden states and input_ids differ.
        if hidden_states.shape[1] > seg_token_mask.shape[1]:
            seg_token_mask = F.pad(
                seg_token_mask,
                (0, hidden_states.shape[1] - seg_token_mask.shape[1]),
                value=False,
            )
        elif hidden_states.shape[1] < seg_token_mask.shape[1]:
            seg_token_mask = seg_token_mask[:, : hidden_states.shape[1]]

        seg_token_counts = seg_token_mask.int().sum(dim=-1)
        seg_hidden_states = hidden_states[seg_token_mask]

        return seg_hidden_states, seg_token_counts

    def _build_seg_offsets(self, seg_token_counts: torch.Tensor) -> torch.Tensor:
        """
        Build offsets for flattened [SEG] embeddings.
        """
        return torch.cat(
            [
                torch.zeros(1, device=seg_token_counts.device, dtype=torch.long),
                seg_token_counts.cumsum(dim=0).long(),
            ],
            dim=0,
        )

    def _resolve_resize_shape(
        self,
        original_size: Tuple[int, int],
        resize_shape: Optional[Any],
    ) -> Tuple[int, int]:
        """
        Resolve resized shape before padding to SAM image_size.
        """
        if resize_shape is not None:
            return _as_hw_tuple(resize_shape)

        scale = float(self.image_size) / max(original_size[0], original_size[1])
        return (
            int(original_size[0] * scale + 0.5),
            int(original_size[1] * scale + 0.5),
        )

    def _decode_sample_masks(
        self,
        image_embedding: torch.Tensor,
        sample_prompts: torch.Tensor,
        original_size: Tuple[int, int],
        resize_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Decode masks for one sample.

        Args:
            image_embedding:
                Tensor[1, C, H, W].
            sample_prompts:
                Tensor[N_seg, prompt_dim].
            original_size:
                Original image size.
            resize_shape:
                Resized image size before padding.

        Returns:
            pred_masks:
                Tensor[N_seg, H_original, W_original] in raw logits.
        """
        if sample_prompts.dim() != 2:
            raise ValueError(
                f"sample_prompts should be [N_seg, C], got {tuple(sample_prompts.shape)}"
            )

        num_seg = int(sample_prompts.shape[0])

        if num_seg == 0:
            raise ValueError("sample_prompts is empty.")

        # SAM MaskDecoder supports multiple prompt batches by using:
        #   image_embeddings: [1, C, H, W]
        #   sparse_prompt_embeddings: [N_seg, 1, 256]
        # Internally, standard SAM repeats the image embedding for prompt batch size.
        sparse_prompt_embeddings = sample_prompts.unsqueeze(1).float()

        dense_prompt_embeddings = self.medsam.build_no_mask_dense_prompt(
            batch_size=num_seg,
            device=image_embedding.device,
            dtype=torch.float32,
        )

        low_res_masks, _ = self.medsam.decode_masks(
            image_embeddings=image_embedding.float(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
            check_numerics=self.check_mask_numerics,
        )

        low_res_masks = low_res_masks.float().clamp(-20.0, 20.0)

        pred_masks = self.medsam.postprocess_masks(
            low_res_masks=low_res_masks,
            input_size=resize_shape,
            original_size=original_size,
        )

        pred_masks = pred_masks.float().clamp(-20.0, 20.0)

        # Standard SAM output: [B_prompt, 1, H, W] when multimask_output=False.
        if pred_masks.dim() == 4:
            if pred_masks.shape[1] == 1:
                pred_masks = pred_masks[:, 0]
            elif pred_masks.shape[0] == 1:
                pred_masks = pred_masks[0]
            else:
                raise ValueError(f"Unexpected pred_masks shape: {tuple(pred_masks.shape)}")
        elif pred_masks.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected pred_masks shape: {tuple(pred_masks.shape)}")

        return pred_masks

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs: Any) -> Union[Dict[str, torch.Tensor], Any]:
        """
        Forward for SFT-style training.

        Returns a dict during training:
            loss
            ce_loss
            mask_loss
            raw_mask_bce_loss
            raw_mask_dice_loss
        """

        # Custom ReaSeg fields.
        medsam_images = kwargs.pop("medsam_images", None)
        if medsam_images is None:
            medsam_images = kwargs.pop("images", None)

        gt_masks = kwargs.pop("gt_masks", None)
        if gt_masks is None:
            gt_masks = kwargs.pop("masks_list", None)

        original_sizes = kwargs.pop("original_sizes", None)
        if original_sizes is None:
            original_sizes = kwargs.pop("original_size_list", None)

        resize_shapes = kwargs.pop("resize_shapes", None)
        if resize_shapes is None:
            resize_shapes = kwargs.pop("resize_list", None)

        # Optional metadata fields, ignored by model.
        kwargs.pop("image_paths", None)
        kwargs.pop("offset", None)
        kwargs.pop("questions_list", None)
        kwargs.pop("sampled_classes_list", None)
        kwargs.pop("inference", None)

        # In eval/generation, behave like Qwen.
        if not self.training:
            return super().forward(**kwargs)

        labels = kwargs.get("labels", None)

        # Qwen forward.
        # Do not pass labels into HF internal CE; compute stable CE manually.
        lm_kwargs = dict(kwargs)
        lm_kwargs.pop("labels", None)
        lm_kwargs["output_hidden_states"] = True
        lm_kwargs["return_dict"] = True

        outputs = super().forward(**lm_kwargs)

        ce_loss = stable_lm_ce_loss(
            logits=outputs.logits,
            labels=labels,
            ignore_index=-100,
        )

        zero_loss = torch.zeros_like(ce_loss)

        # CE-only branch.
        if medsam_images is None or gt_masks is None or original_sizes is None:
            total_loss = self.ce_loss_weight * ce_loss
            return {
                "loss": total_loss,
                "ce_loss": ce_loss.detach(),
                "mask_loss": zero_loss.detach(),
                "raw_mask_bce_loss": zero_loss.detach(),
                "raw_mask_dice_loss": zero_loss.detach(),
            }

        input_ids = kwargs["input_ids"]
        hidden_states = outputs.hidden_states[-1]

        seg_hidden_states, seg_token_counts = self._extract_seg_hidden_states(
            hidden_states=hidden_states,
            input_ids=input_ids,
        )

        # No [SEG] in this batch.
        if seg_token_counts.sum().item() == 0:
            total_loss = self.ce_loss_weight * ce_loss
            return {
                "loss": total_loss,
                "ce_loss": ce_loss.detach(),
                "mask_loss": zero_loss.detach(),
                "raw_mask_bce_loss": zero_loss.detach(),
                "raw_mask_dice_loss": zero_loss.detach(),
            }

        if self.detach_seg_hidden_for_mask:
            seg_hidden_states = seg_hidden_states.detach()

        # [N_total_seg, hidden_size] -> [N_total_seg, prompt_dim]
        seg_prompt_embeddings = self.seg_projector(seg_hidden_states)

        seg_offsets = self._build_seg_offsets(seg_token_counts)

        # MedSAM image embeddings.
        image_embeddings = self.medsam.encode_image(medsam_images)

        pred_masks_for_loss: List[torch.Tensor] = []
        gt_masks_for_loss: List[torch.Tensor] = []

        batch_size = int(image_embeddings.shape[0])

        for batch_idx in range(batch_size):
            start = int(seg_offsets[batch_idx].item())
            end = int(seg_offsets[batch_idx + 1].item())

            if start >= end:
                continue

            sample_prompts = seg_prompt_embeddings[start:end]

            original_size = _as_hw_tuple(
                _get_batch_item(original_sizes, batch_idx)
            )

            if resize_shapes is not None:
                resize_shape_item = _get_batch_item(resize_shapes, batch_idx)
            else:
                resize_shape_item = None

            resize_shape = self._resolve_resize_shape(
                original_size=original_size,
                resize_shape=resize_shape_item,
            )

            image_embedding = image_embeddings[batch_idx : batch_idx + 1]

            pred_masks = self._decode_sample_masks(
                image_embedding=image_embedding,
                sample_prompts=sample_prompts,
                original_size=original_size,
                resize_shape=resize_shape,
            )

            pred_masks_for_loss.append(pred_masks)
            gt_masks_for_loss.append(_get_batch_item(gt_masks, batch_idx))

        if len(pred_masks_for_loss) == 0:
            total_loss = self.ce_loss_weight * ce_loss
            return {
                "loss": total_loss,
                "ce_loss": ce_loss.detach(),
                "mask_loss": zero_loss.detach(),
                "raw_mask_bce_loss": zero_loss.detach(),
                "raw_mask_dice_loss": zero_loss.detach(),
            }

        mask_loss_dict = compute_mask_losses(
            pred_masks=pred_masks_for_loss,
            gt_masks=gt_masks_for_loss,
            bce_loss_weight=self.bce_loss_weight,
            dice_loss_weight=self.dice_loss_weight,
        )

        mask_loss = mask_loss_dict["mask_loss"]
        raw_mask_bce_loss = mask_loss_dict["raw_mask_bce_loss"]
        raw_mask_dice_loss = mask_loss_dict["raw_mask_dice_loss"]

        total_loss = self.ce_loss_weight * ce_loss + mask_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "mask_loss": mask_loss.detach(),
            "raw_mask_bce_loss": raw_mask_bce_loss.detach(),
            "raw_mask_dice_loss": raw_mask_dice_loss.detach(),
        }


__all__ = ["ReaSegForConditionalGeneration"]