#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import argparse
import logging
from typing import Any, Dict, List, Tuple, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model import ReaSegForConditionalGeneration
from data.reason_seg_dataset import ReaSegReasonSegDataset
from data.collate import reaseg_collate_fn


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_torch_dtype(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def load_tokenizer_processor_and_seg_id(model_name_or_path: str, seg_token: str):
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

    if seg_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([seg_token], special_tokens=True)

    seg_token_idx = tokenizer.convert_tokens_to_ids(seg_token)
    logging.info(f"{seg_token} token id: {seg_token_idx}")
    logging.info(f"Tokenizer length: {len(tokenizer)}")

    return tokenizer, processor, int(seg_token_idx)


def build_lora_target_modules(lora_scope: str):
    llm_regex = (
        r".*language_model\.layers\.[0-9]+\.self_attn\."
        r"(q_proj|k_proj|v_proj|o_proj)$"
    )
    qwen_visual_regex = (
        r".*visual\.blocks\.[0-9]+\.attn\."
        r"(qkv|proj)$"
    )

    if lora_scope == "llm":
        return llm_regex
    if lora_scope == "qwen_visual":
        return qwen_visual_regex
    if lora_scope == "llm_qwen_visual":
        return rf"({llm_regex})|({qwen_visual_regex})"

    raise ValueError(f"Unsupported lora_scope for test script: {lora_scope}")


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}

    for k, v in state_dict.items():
        nk = k

        if nk.startswith("module."):
            nk = nk[len("module."):]

        cleaned[nk] = v

    return cleaned


def load_hf_sharded_state_dict(index_json_path: str) -> Dict[str, torch.Tensor]:
    """
    Load HuggingFace sharded checkpoint from pytorch_model.bin.index.json.

    Expected files:
        pytorch_model.bin.index.json
        pytorch_model-00001-of-00002.bin
        pytorch_model-00002-of-00002.bin
        ...
    """
    with open(index_json_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"No weight_map found in index file: {index_json_path}")

    base_dir = os.path.dirname(index_json_path)
    shard_files = sorted(set(weight_map.values()))

    merged_state = {}

    for shard_name in shard_files:
        shard_path = os.path.join(base_dir, shard_name)

        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Checkpoint shard not found: {shard_path}")

        logging.info(f"Loading checkpoint shard: {shard_path}")
        shard_state = torch.load(shard_path, map_location="cpu")

        if isinstance(shard_state, dict) and "state_dict" in shard_state:
            shard_state = shard_state["state_dict"]

        merged_state.update(shard_state)

    logging.info(
        f"Loaded HF sharded checkpoint: {index_json_path}, "
        f"num_shards={len(shard_files)}, num_tensors={len(merged_state)}"
    )

    return merged_state

def load_zero_fp32_state_if_given(model, checkpoint_state_dict: str):
    """
    Load merged fp32 checkpoint.

    Supports:
        1. Single file:
            pytorch_model.bin

        2. HuggingFace sharded checkpoint:
            pytorch_model.bin.index.json
            pytorch_model-00001-of-00002.bin
            ...

        3. Directory containing either of the above.
    """
    if not checkpoint_state_dict:
        return model

    if os.path.isdir(checkpoint_state_dict):
        single_file = os.path.join(checkpoint_state_dict, "pytorch_model.bin")
        index_file = os.path.join(checkpoint_state_dict, "pytorch_model.bin.index.json")

        if os.path.exists(index_file):
            checkpoint_state_dict = index_file
        elif os.path.exists(single_file):
            checkpoint_state_dict = single_file
        else:
            raise FileNotFoundError(
                "No pytorch_model.bin or pytorch_model.bin.index.json found in "
                f"directory: {checkpoint_state_dict}"
            )

    logging.info(f"Loading checkpoint state dict: {checkpoint_state_dict}")

    if checkpoint_state_dict.endswith(".index.json"):
        state = load_hf_sharded_state_dict(checkpoint_state_dict)
    else:
        state = torch.load(checkpoint_state_dict, map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

    state = clean_state_dict_keys(state)

    msg = model.load_state_dict(state, strict=False)

    missing = list(getattr(msg, "missing_keys", []))
    unexpected = list(getattr(msg, "unexpected_keys", []))

    logging.info(
        f"Loaded checkpoint state dict. "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    if missing:
        logging.warning(f"Missing keys example: {missing[:30]}")

    if unexpected:
        logging.warning(f"Unexpected keys example: {unexpected[:30]}")

    return model

def unwrap_reaseg_model(model):
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass

    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model

    return model


def build_model(args, tokenizer, seg_token_idx):
    dtype = get_torch_dtype(args.precision)

    model = ReaSegForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        medsam_checkpoint=args.medsam_checkpoint,
        image_size=args.image_size,
        prompt_dim=args.prompt_dim,
        seg_token_idx=seg_token_idx,
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        detach_seg_hidden_for_mask=True,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if hasattr(model, "reload_medsam_checkpoint"):
        model.reload_medsam_checkpoint(args.medsam_checkpoint)

    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model, "keep_fp32_modules"):
        model.keep_fp32_modules()

    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required when --use_lora is enabled.")

        target_modules = build_lora_target_modules(args.lora_scope)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model = load_zero_fp32_state_if_given(model, args.checkpoint_state_dict)

    core = unwrap_reaseg_model(model)
    if hasattr(core, "keep_fp32_modules"):
        core.keep_fp32_modules()

    return model


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def to_hw_tuple(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        if len(x) == 1 and isinstance(x[0], (list, tuple)):
            return to_hw_tuple(x[0])
        return int(x[0]), int(x[1])
    raise TypeError(type(x))


@torch.no_grad()
def predict_batch_teacher_forced(model, batch, device, seg_token_idx: int):
    """
    Teacher-forced segmentation:
    input_ids already contain assistant output and [SEG].
    """
    model.eval()
    core = unwrap_reaseg_model(model)

    batch = move_to_device(batch, device)

    lm_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values": batch["pixel_values"],
        "image_grid_thw": batch["image_grid_thw"],
        "output_hidden_states": True,
        "return_dict": True,
    }

    outputs = model(**lm_kwargs)
    hidden_states = outputs.hidden_states[-1]

    seg_hidden_states, seg_token_counts = core._extract_seg_hidden_states(
        hidden_states=hidden_states,
        input_ids=batch["input_ids"],
    )

    if seg_token_counts.sum().item() == 0:
        return []

    seg_prompt_embeddings = core.seg_projector(seg_hidden_states)
    seg_offsets = core._build_seg_offsets(seg_token_counts)

    image_embeddings = core.medsam.encode_image(batch["medsam_images"])

    pred_masks_all = []

    for batch_idx in range(image_embeddings.shape[0]):
        start = int(seg_offsets[batch_idx].item())
        end = int(seg_offsets[batch_idx + 1].item())

        if start >= end:
            pred_masks_all.append(None)
            continue

        sample_prompts = seg_prompt_embeddings[start:end]

        original_size = to_hw_tuple(batch["original_sizes"][batch_idx])
        resize_shape = to_hw_tuple(batch["resize_shapes"][batch_idx])

        pred_masks = core._decode_sample_masks(
            image_embedding=image_embeddings[batch_idx:batch_idx + 1],
            sample_prompts=sample_prompts,
            original_size=original_size,
            resize_shape=resize_shape,
        )

        pred_masks_all.append(pred_masks.detach().cpu())

    return pred_masks_all


def align_pred_gt(pred: torch.Tensor, gt: torch.Tensor):
    if gt.dim() == 2:
        gt = gt.unsqueeze(0)
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)

    if pred.shape[0] != gt.shape[0]:
        if pred.shape[0] == 1 and gt.shape[0] > 1:
            gt = gt.max(dim=0, keepdim=True).values
        elif gt.shape[0] == 1 and pred.shape[0] > 1:
            gt = gt.expand(pred.shape[0], -1, -1)
        else:
            pred = pred[:1]
            gt = gt.max(dim=0, keepdim=True).values

    return pred, gt


def binary_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7):
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    dice = (2 * tp + eps) / (pred_sum + gt_sum + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pred_area": int(pred_sum),
        "gt_area": int(gt_sum),
    }


def hd95_metric(pred_bin: np.ndarray, gt_bin: np.ndarray):
    if not SCIPY_AVAILABLE:
        return None

    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0

    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    pred_surface = np.logical_xor(pred, binary_erosion(pred))
    gt_surface = np.logical_xor(gt, binary_erosion(gt))

    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]

    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    if distances.size == 0:
        return float("inf")

    return float(np.percentile(distances, 95))


def main():
    setup_logging()

    parser = argparse.ArgumentParser("Test ReaSeg segmentation metrics")

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--medsam_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint_state_dict", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seg_token", type=str, default="[SEG]")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--prompt_dim", type=int, default=256)
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--logit_threshold", type=float, default=0.0)

    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_scope", type=str, default="llm")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--limit", type=int, default=-1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, processor, seg_token_idx = load_tokenizer_processor_and_seg_id(
        args.model_name_or_path,
        args.seg_token,
    )

    model = build_model(args, tokenizer, seg_token_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Teacher-forcing eval:
    # Load split JSON as train-style sample so answer/output + [SEG] is included.
    if os.path.isdir(args.data_path):
        eval_json = os.path.join(args.data_path, f"{args.split}.json")
    else:
        eval_json = args.data_path

    dataset = ReaSegReasonSegDataset(
        data_path=eval_json,
        tokenizer=tokenizer,
        processor=processor,
        image_size=args.image_size,
        split="train",
        seg_token=args.seg_token,
        seg_token_idx=seg_token_idx,
        force_seg_token=True,
        merge_masks_for_single_seg=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: reaseg_collate_fn(
            b,
            tokenizer=tokenizer,
            precision=args.precision,
            return_legacy_keys=True,
        ),
    )

    rows = []
    total = 0

    for batch_idx, batch in enumerate(loader):
        pred_list = predict_batch_teacher_forced(
            model=model,
            batch=batch,
            device=device,
            seg_token_idx=seg_token_idx,
        )

        for i, pred_logits in enumerate(pred_list):
            if pred_logits is None:
                continue

            gt = batch["gt_masks"][i].cpu().float()
            pred_logits, gt = align_pred_gt(pred_logits.float(), gt)

            pred_bin = (pred_logits > args.logit_threshold).numpy().astype(np.uint8)
            gt_bin = (gt > 0.5).numpy().astype(np.uint8)

            # If multiple masks, evaluate union.
            pred_union = pred_bin.max(axis=0)
            gt_union = gt_bin.max(axis=0)

            m = binary_metrics(pred_union, gt_union)
            hd95 = hd95_metric(pred_union, gt_union)

            image_path = batch["image_paths"][i] if "image_paths" in batch else ""

            row = {
                "index": total,
                "image_path": image_path,
                **m,
                "hd95": hd95,
            }
            rows.append(row)
            total += 1

            if args.limit > 0 and total >= args.limit:
                break

        if args.limit > 0 and total >= args.limit:
            break

    csv_path = os.path.join(args.output_dir, "test_metrics.csv")
    json_path = os.path.join(args.output_dir, "test_summary.json")

    fieldnames = [
        "index",
        "image_path",
        "dice",
        "iou",
        "precision",
        "recall",
        "pred_area",
        "gt_area",
        "hd95",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    def mean_finite(key):
        vals = []
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            if isinstance(v, float) and not np.isfinite(v):
                continue
            vals.append(float(v))
        return float(np.mean(vals)) if vals else None

    summary = {
        "num_samples": len(rows),
        "mean_dice": mean_finite("dice"),
        "mean_iou": mean_finite("iou"),
        "mean_precision": mean_finite("precision"),
        "mean_recall": mean_finite("recall"),
        "mean_hd95": mean_finite("hd95"),
        "logit_threshold": args.logit_threshold,
        "checkpoint_state_dict": args.checkpoint_state_dict,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved CSV: {csv_path}")
    logging.info(f"Saved summary: {json_path}")
    logging.info(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()