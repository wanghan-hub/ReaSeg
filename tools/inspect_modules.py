#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect ReaSeg modules and trainable parameters.

Usage example:

python tools/inspect_modules.py \
  --model_name_or_path ./checkpoints/Qwen/Qwen2.5-VL-3B-Instruct \
  --medsam_checkpoint ./checkpoints/medsam_vit_b.pth \
  --precision bf16 \
  --use_lora \
  --lora_scope llm \
  --train_seg_projector \
  --mask_decoder_train_mode none

This script does NOT start training.
It only:
  1. loads ReaSegForConditionalGeneration
  2. adds [SEG] token if needed
  3. optionally applies LoRA
  4. freezes all params
  5. enables selected trainable modules
  6. prints trainable summary, dtype summary, and optional parameter name matches
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


# Add reaseg project root to sys.path.
# tools/inspect_modules.py -> reaseg/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model import ReaSegForConditionalGeneration


# ============================================================
# Basic utilities
# ============================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_torch_dtype(precision: str):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    if precision not in dtype_map:
        raise ValueError(
            f"Unsupported precision: {precision}. "
            f"Expected one of {list(dtype_map.keys())}."
        )

    return dtype_map[precision]


def load_tokenizer_and_seg_id(
    model_name_or_path: str,
    seg_token: str = "[SEG]",
) -> Tuple[AutoTokenizer, int]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
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

    return tokenizer, int(seg_token_idx)


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False


def unwrap_reaseg_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Unwrap PEFT / DDP / DeepSpeed-like wrappers and return the underlying ReaSeg model.
    """
    m = model.module if hasattr(model, "module") else model

    if hasattr(m, "get_base_model"):
        try:
            return m.get_base_model()
        except Exception:
            pass

    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        return m.base_model.model

    if hasattr(m, "model") and isinstance(m.model, torch.nn.Module):
        return m.model

    return m


# ============================================================
# LoRA
# ============================================================

def build_lora_target_modules(args):
    """
    Build LoRA target modules according to lora_scope.

    lora_scope:
      - llm:
          Qwen language model self-attention q/k/v/o projection.
      - qwen_visual:
          Qwen visual encoder attention qkv/proj.
      - llm_qwen_visual:
          both language model and Qwen visual encoder.
      - custom:
          use --lora_target_modules directly.
    """

    llm_regex = (
        r".*language_model\.layers\.[0-9]+\.self_attn\."
        r"(q_proj|k_proj|v_proj|o_proj)$"
    )

    qwen_visual_regex = (
        r".*visual\.blocks\.[0-9]+\.attn\."
        r"(qkv|proj)$"
    )

    if args.lora_scope == "llm":
        return llm_regex

    if args.lora_scope == "qwen_visual":
        return qwen_visual_regex

    if args.lora_scope == "llm_qwen_visual":
        return rf"({llm_regex})|({qwen_visual_regex})"

    if args.lora_scope == "custom":
        if not args.lora_target_modules:
            raise ValueError("--lora_scope custom requires --lora_target_modules.")

        target_modules = args.lora_target_modules

        if isinstance(target_modules, str) and target_modules.startswith("regex:"):
            return target_modules[len("regex:"):]

        if isinstance(target_modules, str) and "," in target_modules:
            return [x.strip() for x in target_modules.split(",") if x.strip()]

        return target_modules

    raise ValueError(f"Unsupported lora_scope: {args.lora_scope}")


def apply_lora_if_needed(model: torch.nn.Module, args):
    if not args.use_lora:
        logging.info("LoRA disabled.")
        return model

    if LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "peft is not installed or cannot be imported. "
            "Please install peft or run without --use_lora."
        )

    target_modules = build_lora_target_modules(args)

    logging.info(f"Using LoRA scope: {args.lora_scope}")
    logging.info(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    return model


# ============================================================
# Trainable selection
# ============================================================

def enable_qwen_visual_projector_training(model: torch.nn.Module) -> int:
    """
    Enable Qwen native visual merger/projector parameters.

    Different Qwen-VL versions use slightly different names.
    """
    keywords = [
        "visual.merger",
        "visual_merger",
        "multi_modal_projector",
        "mm_projector",
    ]

    count = 0
    for name, p in model.named_parameters():
        if any(k in name for k in keywords):
            p.requires_grad = True
            count += p.numel()
            logging.info(f"Enabled Qwen visual projector param: {name}")

    if count == 0:
        logging.info("No Qwen visual projector/merger parameters found.")

    return count


def enable_reaseg_trainables(model: torch.nn.Module, args) -> None:
    """
    Enable selected trainable modules for ReaSeg inspection.
    """
    core = unwrap_reaseg_model(model)

    if args.train_seg_projector:
        if hasattr(core, "enable_seg_projector_training"):
            core.enable_seg_projector_training(True)
            logging.info("Enabled ReaSeg SEG projector training.")
        else:
            found = False
            for name, p in core.named_parameters():
                if "seg_projector" in name:
                    p.requires_grad = True
                    found = True
            if found:
                logging.info("Enabled ReaSeg SEG projector parameters by name.")
            else:
                logging.warning("seg_projector not found.")

    if hasattr(core, "set_mask_decoder_train_mode"):
        core.set_mask_decoder_train_mode(args.mask_decoder_train_mode)
        logging.info(f"Set mask decoder train mode: {args.mask_decoder_train_mode}")
    else:
        logging.warning("Model has no set_mask_decoder_train_mode method.")

    if args.train_qwen_visual_projector:
        enable_qwen_visual_projector_training(model)

    if hasattr(core, "keep_fp32_modules"):
        core.keep_fp32_modules()
        logging.info("Called keep_fp32_modules().")


# ============================================================
# Summaries
# ============================================================

def classify_param_group(name: str) -> str:
    lname = name.lower()

    if "lora_" in lname and "language_model" in lname:
        return "llm_lora"

    if "lora_" in lname and ".visual." in lname:
        return "qwen_visual_lora"

    if any(k in name for k in ["visual.merger", "visual_merger", "multi_modal_projector", "mm_projector"]):
        return "qwen_visual_projector"

    if "seg_projector" in lname:
        return "seg_projector"
    
    if "prompt_encoder" in lname and ("medsam" in lname or "visual_model" in lname):
        return "medsam_prompt_encoder"
    
    if "image_encoder" in lname and ("medsam" in lname or "visual_model" in lname):
        return "medsam_image_encoder"

    if "mask_decoder" in lname and ("medsam" in lname or "visual_model" in lname):
        return "medsam_mask_decoder"

    return "other_trainable"


def print_trainable_summary(
    model: torch.nn.Module,
    max_trainable_names: int = 300,
) -> None:
    total = 0
    trainable = 0

    groups: Dict[str, int] = {
        "llm_lora": 0,
        "qwen_visual_lora": 0,
        "qwen_visual_projector": 0,
        "seg_projector": 0,
        "medsam_image_encoder": 0,
        "medsam_prompt_encoder": 0,
        "medsam_mask_decoder": 0,
        "other_trainable": 0,
    }

    for name, p in model.named_parameters():
        n = p.numel()
        total += n

        if not p.requires_grad:
            continue

        trainable += n
        group = classify_param_group(name)

        if group not in groups:
            group = "other_trainable"

        groups[group] += n

    pct = 100.0 * trainable / max(total, 1)

    logging.info("=" * 100)
    logging.info(f"Trainable params: {trainable:,} / {total:,} ({pct:.6f}%)")
    logging.info("-" * 100)

    for key, value in groups.items():
        logging.info(f"{key:30s}: {value:,}")

    logging.info("=" * 100)

    shown = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            logging.info(
                f"[trainable] {name} | "
                f"shape={tuple(p.shape)} | "
                f"numel={p.numel():,} | "
                f"dtype={p.dtype}"
            )
            shown += 1

            if shown >= max_trainable_names:
                logging.info(
                    f"Only showing first {max_trainable_names} trainable parameters."
                )
                break


def print_dtype_summary(model: torch.nn.Module) -> None:
    dtype_total: Dict[str, int] = {}
    dtype_trainable: Dict[str, int] = {}

    for _, p in model.named_parameters():
        dtype_name = str(p.dtype)
        dtype_total[dtype_name] = dtype_total.get(dtype_name, 0) + p.numel()

        if p.requires_grad:
            dtype_trainable[dtype_name] = dtype_trainable.get(dtype_name, 0) + p.numel()

    logging.info("=" * 100)
    logging.info("Parameter dtype summary")
    logging.info("-" * 100)

    logging.info("All parameters:")
    for dtype_name, count in sorted(dtype_total.items()):
        logging.info(f"  {dtype_name:20s}: {count:,}")

    logging.info("Trainable parameters:")
    for dtype_name, count in sorted(dtype_trainable.items()):
        logging.info(f"  {dtype_name:20s}: {count:,}")

    logging.info("=" * 100)


def print_module_overview(model: torch.nn.Module) -> None:
    core = unwrap_reaseg_model(model)

    logging.info("=" * 100)
    logging.info("Core module overview")
    logging.info("-" * 100)

    logging.info(f"Wrapped model type: {type(model)}")
    logging.info(f"Core model type:    {type(core)}")

    for attr in [
        "seg_projector",
        "medsam",
        "visual_model",
        "model",
        "visual",
        "language_model",
    ]:
        logging.info(f"hasattr(core, {attr!r}): {hasattr(core, attr)}")

    if hasattr(core, "seg_projector"):
        logging.info(f"seg_projector: {type(core.seg_projector)}")

    if hasattr(core, "medsam"):
        logging.info(f"medsam: {type(core.medsam)}")
        logging.info(f"medsam.image_encoder:  {type(core.medsam.image_encoder)}")
        logging.info(f"medsam.prompt_encoder: {type(core.medsam.prompt_encoder)}")
        logging.info(f"medsam.mask_decoder:   {type(core.medsam.mask_decoder)}")

    logging.info("=" * 100)


def print_matching_parameters(
    model: torch.nn.Module,
    pattern: Optional[str],
    max_names: int = 300,
) -> None:
    if not pattern:
        return

    pattern_lower = pattern.lower()
    matched = []

    for name, p in model.named_parameters():
        if pattern_lower in name.lower():
            matched.append((name, p))

    logging.info("=" * 100)
    logging.info(f"Parameters matching pattern: {pattern!r}, count={len(matched)}")
    logging.info("-" * 100)

    for idx, (name, p) in enumerate(matched):
        logging.info(
            f"[match] {name} | "
            f"shape={tuple(p.shape)} | "
            f"numel={p.numel():,} | "
            f"dtype={p.dtype} | "
            f"requires_grad={p.requires_grad}"
        )

        if idx + 1 >= max_names:
            logging.info(f"Only showing first {max_names} matched parameters.")
            break

    logging.info("=" * 100)


def find_nonfinite_parameters(
    model: torch.nn.Module,
    only_trainable: bool = True,
    max_show: int = 50,
) -> List[dict]:
    bad = []

    for name, p in model.named_parameters():
        if only_trainable and not p.requires_grad:
            continue

        with torch.no_grad():
            data = p.detach()

            if torch.isfinite(data).all():
                continue

            has_nan = torch.isnan(data).any().item()
            has_inf = torch.isinf(data).any().item()
            finite_mask = torch.isfinite(data)

            if finite_mask.any():
                finite_data = data[finite_mask].float()
                finite_min = finite_data.min().item()
                finite_max = finite_data.max().item()
                abs_max = finite_data.abs().max().item()
                finite_ratio = finite_mask.float().mean().item()
            else:
                finite_min = float("nan")
                finite_max = float("nan")
                abs_max = float("nan")
                finite_ratio = 0.0

            bad.append(
                {
                    "name": name,
                    "shape": tuple(p.shape),
                    "dtype": str(p.dtype),
                    "requires_grad": p.requires_grad,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "finite_ratio": finite_ratio,
                    "finite_min": finite_min,
                    "finite_max": finite_max,
                    "abs_max": abs_max,
                }
            )

            if len(bad) >= max_show:
                break

    return bad


def print_nonfinite_check(model: torch.nn.Module) -> None:
    bad = find_nonfinite_parameters(model, only_trainable=True)

    logging.info("=" * 100)
    logging.info("Non-finite trainable parameter check")
    logging.info("-" * 100)

    if not bad:
        logging.info("No non-finite trainable parameters found.")
    else:
        logging.error(f"Found {len(bad)} non-finite trainable parameters.")
        for item in bad:
            logging.error(
                f"name={item['name']}, "
                f"shape={item['shape']}, "
                f"dtype={item['dtype']}, "
                f"requires_grad={item['requires_grad']}, "
                f"has_nan={item['has_nan']}, "
                f"has_inf={item['has_inf']}, "
                f"finite_ratio={item['finite_ratio']}, "
                f"finite_min={item['finite_min']}, "
                f"finite_max={item['finite_max']}, "
                f"abs_max={item['abs_max']}"
            )

    logging.info("=" * 100)


# ============================================================
# Model loading
# ============================================================

def build_model(args):
    tokenizer, seg_token_idx = load_tokenizer_and_seg_id(
        model_name_or_path=args.model_name_or_path,
        seg_token=args.seg_token,
    )

    torch_dtype = get_torch_dtype(args.precision)

    logging.info(f"Loading ReaSeg model from: {args.model_name_or_path}")
    logging.info(f"Using torch_dtype={torch_dtype}")
    logging.info(f"Using MedSAM checkpoint: {args.medsam_checkpoint}")

    model = ReaSegForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        medsam_checkpoint=args.medsam_checkpoint,
        image_size=args.image_size,
        prompt_dim=args.prompt_dim,
        seg_token_idx=seg_token_idx,
        ce_loss_weight=args.ce_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight,
        detach_seg_hidden_for_mask=args.detach_seg_hidden_for_mask,
        projector_init_std=args.projector_init_std,
        projector_output_scale=args.projector_output_scale,
        projector_clamp_value=args.projector_clamp_value,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    if hasattr(model, "reload_medsam_checkpoint"):
        model.reload_medsam_checkpoint(args.medsam_checkpoint)
        logging.info("Reloaded MedSAM checkpoint after ReaSeg from_pretrained().")

    model.resize_token_embeddings(len(tokenizer))
    model.keep_fp32_modules()

    logging.info("Freezing all parameters first.")
    freeze_all_parameters(model)

    model = apply_lora_if_needed(model, args)

    enable_reaseg_trainables(model, args)

    return model, tokenizer


# ============================================================
# CLI
# ============================================================

def create_parser():
    parser = argparse.ArgumentParser(
        description="Inspect ReaSeg model modules and trainable parameters."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./checkpoints/Qwen/Qwen2.5-VL-3B-Instruct",
        help="Path to Qwen2.5-VL model.",
    )

    parser.add_argument(
        "--medsam_checkpoint",
        type=str,
        default="./checkpoints/medsam_vit_b.pth",
        help="Path to MedSAM/SAM ViT-B checkpoint.",
    )

    parser.add_argument(
        "--seg_token",
        type=str,
        default="[SEG]",
        help="Segmentation token.",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--prompt_dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )

    parser.add_argument(
        "--ce_loss_weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dice_loss_weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--bce_loss_weight",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--detach_seg_hidden_for_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--projector_init_std",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--projector_output_scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--projector_clamp_value",
        type=float,
        default=10.0,
    )

    # LoRA
    parser.add_argument(
        "--use_lora",
        action="store_true",
    )

    parser.add_argument(
        "--lora_scope",
        type=str,
        default="llm",
        choices=["llm", "qwen_visual", "llm_qwen_visual", "custom"],
    )

    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    # Trainable module switches.
    parser.add_argument(
        "--train_seg_projector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ReaSeg SEG projector parameters.",
    )

    parser.add_argument(
        "--mask_decoder_train_mode",
        type=str,
        default="none",
        choices=["none", "partial", "head_plus_upscaling", "full"],
        help="MedSAM mask decoder train mode.",
    )

    parser.add_argument(
        "--train_qwen_visual_projector",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen native visual merger/projector parameters.",
    )

    # Inspection options.
    parser.add_argument(
        "--match",
        type=str,
        default=None,
        help="Print parameters whose names contain this substring.",
    )

    parser.add_argument(
        "--max_trainable_names",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--max_match_names",
        type=int,
        default=300,
    )

    return parser


def main():
    setup_logging()

    parser = create_parser()
    args = parser.parse_args()

    logging.info("=" * 100)
    logging.info("Starting ReaSeg module inspection")
    logging.info(f"Args: {args}")
    logging.info("=" * 100)

    model, tokenizer = build_model(args)

    print_module_overview(model)
    print_trainable_summary(model, max_trainable_names=args.max_trainable_names)
    print_dtype_summary(model)
    print_matching_parameters(model, args.match, max_names=args.max_match_names)
    print_nonfinite_check(model)

    logging.info("Inspection completed.")


if __name__ == "__main__":
    main()