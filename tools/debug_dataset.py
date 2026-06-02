#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug ReaSeg dataset and collate function.

This script checks:
    1. tokenizer / processor loading
    2. [SEG] token id
    3. one dataset sample
    4. one collated batch
    5. Qwen2.5-VL image token alignment
    6. MedSAM image / mask shapes

Example:

cd /data/ReaSeg/pathchatseg-r1-main/reaseg

python tools/debug_dataset.py \
  --data_path ./dataset \
  --model_name_or_path ./checkpoints/Qwen/Qwen2.5-VL-3B-Instruct \
  --split train \
  --sample_index 0 \
  --batch_size 2 \
  --precision bf16
"""

import os
import sys
import argparse
import logging
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader


# tools/debug_dataset.py -> reaseg/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.reason_seg_dataset import ReaSegReasonSegDataset
from data.collate import reaseg_collate_fn


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def shape_str(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    if isinstance(x, list):
        return f"list(len={len(x)})"
    if isinstance(x, tuple):
        return f"tuple({x})"
    return str(type(x))


def tensor_stats(x: torch.Tensor, name: str) -> None:
    x_float = x.detach().float()

    print(
        f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, "
        f"min={x_float.min().item():.6f}, "
        f"max={x_float.max().item():.6f}, "
        f"mean={x_float.mean().item():.6f}, "
        f"has_nan={torch.isnan(x_float).any().item()}, "
        f"has_inf={torch.isinf(x_float).any().item()}"
    )


def load_tokenizer_and_processor(model_name_or_path: str, seg_token: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab = tokenizer.get_vocab()

    if seg_token not in vocab:
        logging.info(f"Adding segmentation token: {seg_token}")
        tokenizer.add_tokens([seg_token], special_tokens=True)
    else:
        logging.info(f"Segmentation token already exists: {seg_token}")

    seg_token_idx = tokenizer.convert_tokens_to_ids(seg_token)

    if seg_token_idx is None or seg_token_idx < 0:
        raise RuntimeError(f"Failed to get token id for {seg_token}")

    logging.info(f"{seg_token} token id: {seg_token_idx}")
    logging.info(f"Tokenizer length: {len(tokenizer)}")

    return tokenizer, processor, int(seg_token_idx)


def check_vl_alignment(sample_or_batch: Dict[str, Any], tokenizer, prefix: str = "") -> None:
    """
    Check Qwen2.5-VL image token and pixel feature alignment.
    """
    if "input_ids" not in sample_or_batch:
        print(f"{prefix}No input_ids; skip VL alignment check.")
        return

    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    if image_token_id is None or image_token_id < 0:
        print(f"{prefix}Cannot find <|image_pad|> token id; skip VL alignment check.")
        return

    input_ids = sample_or_batch["input_ids"]

    if input_ids.dim() == 1:
        num_image_tokens = (input_ids == image_token_id).sum().item()
    else:
        num_image_tokens = (input_ids == image_token_id).sum().item()

    pixel_values = sample_or_batch.get("pixel_values", None)

    if pixel_values is None:
        num_image_features = 0
        pixel_shape = None
    else:
        num_image_features = pixel_values.shape[0]
        pixel_shape = tuple(pixel_values.shape)

    image_grid_thw = sample_or_batch.get("image_grid_thw", None)

    print(
        f"{prefix}VL alignment: "
        f"image_tokens={num_image_tokens}, "
        f"pixel_values_shape={pixel_shape}, "
        f"num_pixel_features={num_image_features}, "
        f"image_grid_thw={image_grid_thw}"
    )

    if num_image_features > 0 and num_image_tokens == 0:
        raise RuntimeError(
            "Qwen2.5-VL image-token mismatch: pixel_values exist but "
            "<|image_pad|> tokens are missing."
        )


def print_sample_debug(sample: Dict[str, Any], tokenizer, seg_token_idx: int) -> None:
    print("\n========== Dataset sample sanity check ==========")

    print(f"sample keys: {list(sample.keys())}")

    for key in [
        "input_ids",
        "labels",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "medsam_image",
        "gt_masks",
        "original_size",
        "resize_shape",
        "images",
        "masks_list",
        "original_size_list",
        "resize_list",
        "image_path",
    ]:
        if key in sample:
            value = sample[key]
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
            else:
                print(f"{key}: {value}")

    valid_label_tokens = (sample["labels"] != -100).sum().item()
    seg_token_count = (sample["input_ids"] == seg_token_idx).sum().item()

    print(f"valid label tokens: {valid_label_tokens}")
    print(f"SEG token count in input_ids: {seg_token_count}")

    decoded_preview = tokenizer.decode(
        sample["input_ids"],
        skip_special_tokens=False,
    )
    print("\nDecoded sample preview:")
    print(decoded_preview[:1500])

    print("\nTensor statistics:")
    tensor_stats(sample["pixel_values"], "pixel_values")
    tensor_stats(sample["medsam_image"], "medsam_image")
    tensor_stats(sample["gt_masks"], "gt_masks")

    gt_masks = sample["gt_masks"]
    if gt_masks.dim() == 3:
        mask_area = gt_masks.flatten(1).sum(dim=1)
        print(f"gt mask areas: {[float(x) for x in mask_area]}")

    check_vl_alignment(sample, tokenizer, prefix="[sample] ")


def print_batch_debug(batch: Dict[str, Any], tokenizer, seg_token_idx: int) -> None:
    print("\n========== Collate batch sanity check ==========")

    print(f"batch keys: {list(batch.keys())}")

    for key in [
        "input_ids",
        "labels",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "medsam_images",
        "images",
        "gt_masks",
        "masks_list",
        "original_sizes",
        "original_size_list",
        "resize_shapes",
        "resize_list",
        "image_paths",
    ]:
        if key in batch:
            value = batch[key]

            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
            elif isinstance(value, list):
                print(f"{key}: list length={len(value)}")
                if len(value) > 0:
                    first = value[0]
                    if isinstance(first, torch.Tensor):
                        print(f"{key}[0]: shape={tuple(first.shape)}, dtype={first.dtype}")
                    else:
                        print(f"{key}[0]: {first}")
            else:
                print(f"{key}: {value}")

    valid_label_tokens = (batch["labels"] != -100).sum().item()
    seg_token_count = (batch["input_ids"] == seg_token_idx).sum().item()

    print(f"valid label tokens in batch: {valid_label_tokens}")
    print(f"SEG token count in batch input_ids: {seg_token_count}")

    tensor_stats(batch["pixel_values"], "batch.pixel_values")
    tensor_stats(batch["medsam_images"], "batch.medsam_images")

    if "gt_masks" in batch and len(batch["gt_masks"]) > 0:
        tensor_stats(batch["gt_masks"][0], "batch.gt_masks[0]")

    check_vl_alignment(batch, tokenizer, prefix="[batch] ")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Debug ReaSeg reason-seg dataset and collate function."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Directory containing train.json/val.json/test.json, or a JSON file.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./checkpoints/Qwen/Qwen2.5-VL-3B-Instruct",
        help="Path to Qwen2.5-VL model.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )

    parser.add_argument(
        "--seg_token",
        type=str,
        default="[SEG]",
    )

    parser.add_argument(
        "--no_force_seg_token",
        action="store_true",
        help="Disable auto-appending [SEG] to training answers.",
    )

    parser.add_argument(
        "--no_merge_masks_for_single_seg",
        action="store_true",
        help="Disable merging multiple masks when there is only one [SEG].",
    )

    return parser


def main():
    setup_logging()

    parser = create_parser()
    args = parser.parse_args()

    tokenizer, processor, seg_token_idx = load_tokenizer_and_processor(
        model_name_or_path=args.model_name_or_path,
        seg_token=args.seg_token,
    )

    dataset = ReaSegReasonSegDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        image_size=args.image_size,
        max_seq_length=args.max_seq_length,
        split=args.split,
        seg_token=args.seg_token,
        seg_token_idx=seg_token_idx,
        force_seg_token=not args.no_force_seg_token,
        merge_masks_for_single_seg=not args.no_merge_masks_for_single_seg,
    )

    print("\n========== Dataset info ==========")
    print(f"dataset length: {len(dataset)}")
    print(f"data_path: {args.data_path}")
    print(f"split: {args.split}")
    print(f"sample_index: {args.sample_index}")

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={args.sample_index} out of range for dataset length={len(dataset)}"
        )

    sample = dataset[args.sample_index]
    print_sample_debug(sample, tokenizer, seg_token_idx)

    # Collate debug.
    batch_samples = []
    for i in range(args.batch_size):
        idx = (args.sample_index + i) % len(dataset)
        batch_samples.append(dataset[idx])

    batch = reaseg_collate_fn(
        batch_samples,
        tokenizer=tokenizer,
        precision=args.precision,
        return_legacy_keys=True,
    )

    print_batch_debug(batch, tokenizer, seg_token_idx)

    print("\n✅ Dataset and collate sanity check passed.")


if __name__ == "__main__":
    main()