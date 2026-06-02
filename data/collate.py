# -*- coding: utf-8 -*-
"""
Collate function for ReaSeg SFT.

Input sample fields from ReaSegReasonSegDataset:

Preferred names:
    input_ids
    attention_mask
    labels
    pixel_values
    image_grid_thw

    medsam_image
    gt_masks
    original_size
    resize_shape
    image_path

Backward-compatible aliases:
    images
    masks_list
    original_size_list
    resize_list

Output batch fields:

Preferred names:
    input_ids
    attention_mask
    labels
    pixel_values
    image_grid_thw

    medsam_images
    gt_masks
    original_sizes
    resize_shapes
    image_paths

Compatibility names:
    images
    masks_list
    original_size_list
    resize_list
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def _to_hw_tuple(size_like: Any) -> Tuple[int, int]:
    """
    Convert list/tuple/tensor-like object to (H, W).
    """
    if isinstance(size_like, torch.Tensor):
        size_like = size_like.detach().cpu().tolist()

    if isinstance(size_like, (list, tuple)):
        # Some legacy samples store resize_list as [(H, W)].
        if len(size_like) == 1 and isinstance(size_like[0], (list, tuple, torch.Tensor)):
            return _to_hw_tuple(size_like[0])

        if len(size_like) < 2:
            raise ValueError(f"Invalid size object: {size_like}")

        return int(size_like[0]), int(size_like[1])

    raise TypeError(f"Unsupported size type: {type(size_like)}")


def _get_sample_value(sample: Dict[str, Any], preferred_key: str, fallback_key: Optional[str] = None):
    """
    Fetch value from preferred key or fallback key.
    """
    if preferred_key in sample:
        return sample[preferred_key]

    if fallback_key is not None and fallback_key in sample:
        return sample[fallback_key]

    raise KeyError(
        f"Sample is missing required key: {preferred_key}"
        + (f" or fallback key: {fallback_key}" if fallback_key else "")
    )


def _get_optional_sample_value(
    sample: Dict[str, Any],
    preferred_key: str,
    fallback_key: Optional[str] = None,
    default: Any = None,
):
    if preferred_key in sample:
        return sample[preferred_key]

    if fallback_key is not None and fallback_key in sample:
        return sample[fallback_key]

    return default


def _normalize_mask_tensor(mask: Any) -> torch.Tensor:
    """
    Convert mask to float binary tensor.

    Expected:
        [H, W] or [N, H, W]
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)

    mask = mask.float()

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    if mask.dim() != 3:
        raise ValueError(f"Expected mask shape [N,H,W] or [H,W], got {tuple(mask.shape)}")

    mask = (mask > 0).float()

    return mask


def reaseg_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer,
    precision: str = "bf16",
    return_legacy_keys: bool = True,
) -> Dict[str, Any]:
    """
    Collate function for ReaSeg SFT.

    Args:
        batch:
            List of dataset samples.
        tokenizer:
            Qwen tokenizer. Needed for pad_token_id.
        precision:
            pixel_values dtype. One of: bf16, fp16, fp32.
        return_legacy_keys:
            If True, also return images/masks_list/original_size_list/resize_list
            for compatibility with old code paths.

    Returns:
        Batch dict.
    """
    if tokenizer is None:
        raise ValueError("tokenizer must be provided to reaseg_collate_fn.")

    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        raise ValueError("Empty batch after filtering None samples.")

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    if precision not in dtype_map:
        raise ValueError(f"Unsupported precision: {precision}. Expected one of {list(dtype_map.keys())}.")

    pixel_dtype = dtype_map[precision]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("tokenizer.pad_token_id and tokenizer.eos_token_id are both None.")

    # ------------------------------------------------------------
    # Text branch
    # ------------------------------------------------------------
    input_ids = pad_sequence(
        [sample["input_ids"] for sample in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )

    attention_mask = pad_sequence(
        [sample["attention_mask"] for sample in batch],
        batch_first=True,
        padding_value=0,
    )

    labels = pad_sequence(
        [sample["labels"] for sample in batch],
        batch_first=True,
        padding_value=-100,
    )

    # ------------------------------------------------------------
    # Qwen2.5-VL visual branch
    # ------------------------------------------------------------
    pixel_values_list = []
    image_grid_list = []

    for sample in batch:
        pixel_values = sample["pixel_values"]

        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.as_tensor(pixel_values)

        # Qwen2.5-VL processor usually returns [N_img_tokens, feature_dim]
        # or [1, N_img_tokens, feature_dim]. Flatten leading dims.
        pixel_values = pixel_values.view(-1, pixel_values.shape[-1])
        pixel_values_list.append(pixel_values)

        image_grid_thw = sample.get("image_grid_thw", None)
        if image_grid_thw is None:
            raise KeyError("Sample is missing image_grid_thw.")

        if not isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = torch.as_tensor(image_grid_thw)

        image_grid_list.append(image_grid_thw.view(-1, 3))

    pixel_values = torch.cat(pixel_values_list, dim=0).to(pixel_dtype)
    image_grid_thw = torch.cat(image_grid_list, dim=0)

    # ------------------------------------------------------------
    # MedSAM branch
    # ------------------------------------------------------------
    medsam_images = []

    for sample in batch:
        medsam_image = _get_sample_value(
            sample,
            preferred_key="medsam_image",
            fallback_key="images",
        )

        if not isinstance(medsam_image, torch.Tensor):
            medsam_image = torch.as_tensor(medsam_image)

        if medsam_image.dim() != 3:
            raise ValueError(
                f"medsam_image should be [3,H,W], got {tuple(medsam_image.shape)}"
            )

        medsam_images.append(medsam_image.float())

    medsam_images = torch.stack(medsam_images, dim=0)

    gt_masks = []
    original_sizes = []
    resize_shapes = []
    image_paths = []

    for sample in batch:
        mask = _get_sample_value(
            sample,
            preferred_key="gt_masks",
            fallback_key="masks_list",
        )
        gt_masks.append(_normalize_mask_tensor(mask))

        original_size = _get_sample_value(
            sample,
            preferred_key="original_size",
            fallback_key="original_size_list",
        )
        original_sizes.append(_to_hw_tuple(original_size))

        resize_shape = _get_sample_value(
            sample,
            preferred_key="resize_shape",
            fallback_key="resize_list",
        )
        resize_shapes.append(_to_hw_tuple(resize_shape))

        image_paths.append(str(sample.get("image_path", "")))

    output = {
        # Qwen2.5-VL branch
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,

        # Preferred ReaSeg names
        "medsam_images": medsam_images,
        "gt_masks": gt_masks,
        "original_sizes": original_sizes,
        "resize_shapes": resize_shapes,
        "image_paths": image_paths,
    }

    if return_legacy_keys:
        output.update(
            {
                "images": medsam_images,
                "masks_list": gt_masks,
                "original_size_list": original_sizes,
                "resize_list": resize_shapes,
            }
        )

    return output


# Backward-compatible alias.
collate_fn = reaseg_collate_fn


__all__ = [
    "reaseg_collate_fn",
    "collate_fn",
]