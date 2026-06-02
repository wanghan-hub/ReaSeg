# -*- coding: utf-8 -*-
"""
Shared training utilities for ReaSeg.

This file contains reusable utilities for:
    - argument parser construction
    - logging
    - tokenizer / processor loading
    - [SEG] token handling
    - DeepSpeed config creation
    - LoRA target construction
    - trainable module selection
    - optimizer parameter groups
    - checkpoint saving
    - validation
    - numerical checks

It is intentionally clean and does not include legacy QWSA / PathChat branches
such as RuiPath, GRPO, stain augmentation, etc.
"""

import os
import json
import random
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoProcessor

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


# ============================================================
# Logging / seed / process helpers
# ============================================================

def is_main_process(local_rank: int) -> bool:
    return local_rank in (-1, 0)


def normalize_local_rank(args):
    """
    Normalize local_rank from DeepSpeed launcher environment.
    """
    env_local_rank = os.environ.get("LOCAL_RANK", None)
    if env_local_rank is not None:
        args.local_rank = int(env_local_rank)
    return args


def setup_logging(output_dir: str, local_rank: int = 0, log_name: str = "train.log") -> None:
    """
    Setup console + file logging.

    Only rank 0 writes the file log.
    """
    os.makedirs(output_dir, exist_ok=True)

    log_level = logging.INFO if is_main_process(local_rank) else logging.WARNING
    handlers = [logging.StreamHandler()]

    if is_main_process(local_rank):
        handlers.append(
            logging.FileHandler(
                os.path.join(output_dir, log_name),
                mode="a",
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_torch_dtype(precision: str):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    if precision not in dtype_map:
        raise ValueError(
            f"Unsupported precision={precision}. "
            f"Expected one of {list(dtype_map.keys())}."
        )

    return dtype_map[precision]


def move_to_device(obj: Any, device: torch.device):
    """
    Recursively move tensors inside dict/list/tuple to target device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)

    return obj


def get_output_value(outputs: Any, key: str, default=None):
    """
    Safely extract key from dict / ModelOutput / tuple-like outputs.
    """
    if isinstance(outputs, dict):
        return outputs.get(key, default)

    if hasattr(outputs, key):
        return getattr(outputs, key)

    if hasattr(outputs, "get"):
        try:
            return outputs.get(key, default)
        except Exception:
            pass

    return default


# ============================================================
# Tokenizer / processor / [SEG]
# ============================================================

def load_tokenizer_and_processor(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    padding_side: str = "right",
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
        use_fast=False,
    )

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, processor


def ensure_seg_token(tokenizer, seg_token: str = "[SEG]") -> int:
    """
    Ensure [SEG] exists in tokenizer and return its token id.
    """
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

    return int(seg_token_idx)


def load_tokenizer_processor_and_seg_id(
    model_name_or_path: str,
    seg_token: str = "[SEG]",
    trust_remote_code: bool = True,
):
    tokenizer, processor = load_tokenizer_and_processor(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    seg_token_idx = ensure_seg_token(tokenizer, seg_token=seg_token)

    return tokenizer, processor, seg_token_idx


# ============================================================
# Argument parser
# ============================================================

def add_reaseg_base_arguments(parser: argparse.ArgumentParser):
    # Paths
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--medsam_checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Model / data
    parser.add_argument("--seg_token", type=str, default="[SEG]")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--prompt_dim", type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # Loss
    parser.add_argument("--ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--dice_loss_weight", type=float, default=0.5)
    parser.add_argument("--bce_loss_weight", type=float, default=2.0)

    # Projector
    parser.add_argument("--projector_init_std", type=float, default=1e-3)
    parser.add_argument("--projector_output_scale", type=float, default=1.0)
    parser.add_argument("--projector_clamp_value", type=float, default=10.0)

    # Trainable controls
    parser.add_argument(
        "--detach_seg_hidden_for_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detach [SEG] hidden states before mask projection.",
    )

    parser.add_argument(
        "--train_seg_projector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ReaSeg [SEG] projector training.",
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
        help="Train Qwen native visual merger/projector.",
    )

    # LoRA
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--lora_scope",
        type=str,
        default="llm",
        choices=["llm", "qwen_visual", "llm_qwen_visual", "custom"],
    )
    parser.add_argument("--lora_target_modules", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seg_projector_lr", type=float, default=1e-5)
    parser.add_argument("--mask_decoder_lr", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--warmup_steps", type=int, default=0)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Precision / DeepSpeed
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="",
        help="Optional external DeepSpeed JSON config. If set, it overrides generated config.",
    )

    # Logging / eval / save
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=20)

    # Debug
    parser.add_argument("--debug_vl_alignment", action="store_true")
    parser.add_argument("--debug_numerics", action="store_true")
    parser.add_argument("--debug_steps", type=int, default=5)
    parser.add_argument("--skip_nonfinite_loss", action="store_true")

    # Launcher
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", type=str, default="", help="DeepSpeed checkpoint directory from previous stage.",)
    parser.add_argument("--resume_optimizer_states", action=argparse.BooleanOptionalAction,default=False, help="Whether to resume optimizer/lr scheduler states. For a new stage, usually False.",)

    parser.set_defaults(training_mode="sft")

    return parser


def create_reaseg_sft_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ReaSeg SFT Training")
    parser = add_reaseg_base_arguments(parser)
    return parser


def validate_reaseg_args(args):
    args = normalize_local_rank(args)

    if not os.path.exists(args.model_name_or_path):
        raise FileNotFoundError(f"model_name_or_path not found: {args.model_name_or_path}")

    if not os.path.exists(args.medsam_checkpoint):
        raise FileNotFoundError(f"medsam_checkpoint not found: {args.medsam_checkpoint}")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"data_path not found: {args.data_path}")

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be > 0.")

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")

    if args.use_lora and LoraConfig is None:
        raise ImportError("peft is required when --use_lora is enabled.")

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def get_default_reaseg_sft_config() -> Dict[str, Any]:
    return {
        "seg_token": "[SEG]",
        "image_size": 1024,
        "prompt_dim": 256,
        "max_seq_length": 2048,
        "ce_loss_weight": 1.0,
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "projector_init_std": 1e-3,
        "projector_output_scale": 1.0,
        "projector_clamp_value": 10.0,
        "detach_seg_hidden_for_mask": True,
        "train_seg_projector": True,
        "mask_decoder_train_mode": "none",
        "train_qwen_visual_projector": False,
        "use_lora": True,
        "lora_scope": "llm",
        "lora_r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "learning_rate": 1e-5,
        "seg_projector_lr": 1e-5,
        "mask_decoder_lr": 1e-8,
        "weight_decay": 0.0,
        "adam_eps": 1e-6,
        "max_grad_norm": 0.3,
        "warmup_steps": 0,
        "epochs": 10,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "workers": 4,
        "seed": 42,
        "precision": "bf16",
        "zero_stage": 2,
        "log_interval": 10,
        "eval_interval": 0,
        "save_interval": 0,
    }


# ============================================================
# LoRA
# ============================================================

def build_lora_target_modules(args):
    """
    Build LoRA target modules according to lora_scope.
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

        if target_modules.startswith("regex:"):
            return target_modules[len("regex:"):]

        if "," in target_modules:
            return [x.strip() for x in target_modules.split(",") if x.strip()]

        return target_modules

    raise ValueError(f"Unsupported lora_scope: {args.lora_scope}")


def apply_lora_if_needed(model: torch.nn.Module, args):
    if not getattr(args, "use_lora", False):
        logging.info("LoRA disabled.")
        return model

    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required when --use_lora is enabled.")

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
# Model unwrap / freeze / trainable modules
# ============================================================

def unwrap_reaseg_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Unwrap DeepSpeed / DDP / PEFT wrappers and return the underlying ReaSeg model.
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


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False


def enable_qwen_visual_projector_training(model: torch.nn.Module) -> int:
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
    Enable selected ReaSeg trainable modules.
    """
    core = unwrap_reaseg_model(model)

    if getattr(args, "train_seg_projector", True):
        if hasattr(core, "enable_seg_projector_training"):
            core.enable_seg_projector_training(True)
            logging.info("Enabled ReaSeg SEG projector training.")
        else:
            found = False
            for name, p in model.named_parameters():
                if "seg_projector" in name:
                    p.requires_grad = True
                    found = True

            if found:
                logging.info("Enabled seg_projector parameters by name.")
            else:
                logging.warning("seg_projector not found.")

    if hasattr(core, "set_mask_decoder_train_mode"):
        core.set_mask_decoder_train_mode(getattr(args, "mask_decoder_train_mode", "none"))
        logging.info(f"Set mask decoder train mode: {args.mask_decoder_train_mode}")
    else:
        logging.warning("Model has no set_mask_decoder_train_mode method.")

    if getattr(args, "train_qwen_visual_projector", False):
        enable_qwen_visual_projector_training(model)

    if hasattr(core, "keep_fp32_modules"):
        core.keep_fp32_modules()
        logging.info("Called keep_fp32_modules().")


# ============================================================
# Parameter summaries / optimizer groups
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

    if "mask_decoder" in lname and ("medsam" in lname or "visual_model" in lname):
        return "medsam_mask_decoder"

    if "prompt_encoder" in lname and ("medsam" in lname or "visual_model" in lname):
        return "medsam_prompt_encoder"

    if "image_encoder" in lname and ("medsam" in lname or "visual_model" in lname):
        return "medsam_image_encoder"

    return "other_trainable"


def print_trainable_summary(model: torch.nn.Module, max_names: int = 200) -> None:
    total = 0
    trainable = 0

    groups = {
        "llm_lora": 0,
        "qwen_visual_lora": 0,
        "qwen_visual_projector": 0,
        "seg_projector": 0,
        "medsam_mask_decoder": 0,
        "medsam_prompt_encoder": 0,
        "medsam_image_encoder": 0,
        "other_trainable": 0,
    }

    for name, p in model.named_parameters():
        n = p.numel()
        total += n

        if not p.requires_grad:
            continue

        trainable += n
        group = classify_param_group(name)
        groups[group] = groups.get(group, 0) + n

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
            if shown >= max_names:
                logging.info(f"Only showing first {max_names} trainable parameters.")
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


def build_optimizer_param_groups(model: torch.nn.Module, args):
    base_lr = args.learning_rate
    seg_projector_lr = args.seg_projector_lr
    mask_decoder_lr = args.mask_decoder_lr

    other_params = []
    seg_projector_params = []
    mask_decoder_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lname = name.lower()

        if "seg_projector" in lname:
            seg_projector_params.append(p)
        elif "mask_decoder" in lname and ("medsam" in lname or "visual_model" in lname):
            mask_decoder_params.append(p)
        else:
            other_params.append(p)

    param_groups = []

    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": base_lr,
                "weight_decay": args.weight_decay,
            }
        )

    if seg_projector_params:
        param_groups.append(
            {
                "params": seg_projector_params,
                "lr": seg_projector_lr,
                "weight_decay": 0.0,
            }
        )

    if mask_decoder_params:
        param_groups.append(
            {
                "params": mask_decoder_params,
                "lr": mask_decoder_lr,
                "weight_decay": 0.0,
            }
        )

    logging.info(
        f"Optimizer groups: "
        f"other={len(other_params)}, "
        f"seg_projector={len(seg_projector_params)}, "
        f"mask_decoder={len(mask_decoder_params)}, "
        f"base_lr={base_lr}, "
        f"seg_projector_lr={seg_projector_lr}, "
        f"mask_decoder_lr={mask_decoder_lr}"
    )

    return param_groups


# ============================================================
# DeepSpeed config
# ============================================================

def load_json_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_deepspeed_config(args, steps_per_epoch: int = 1000) -> Dict[str, Any]:
    """
    Create a clean DeepSpeed config.

    If args.deepspeed_config is set, load that file and patch basic runtime values.
    """
    if getattr(args, "deepspeed_config", ""):
        if not os.path.exists(args.deepspeed_config):
            raise FileNotFoundError(f"DeepSpeed config not found: {args.deepspeed_config}")

        ds_config = load_json_config(args.deepspeed_config)
        ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        ds_config["gradient_clipping"] = args.max_grad_norm
        return ds_config

    total_steps = max(1, int(args.epochs) * max(1, int(steps_per_epoch)))

    if args.precision == "bf16":
        fp16_config = {"enabled": False}
        bf16_config = {"enabled": True}
    elif args.precision == "fp16":
        fp16_config = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
        bf16_config = {"enabled": False}
    else:
        fp16_config = {"enabled": False}
        bf16_config = {"enabled": False}

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.max_grad_norm,

        "fp16": fp16_config,
        "bf16": bf16_config,

        "zero_optimization": {
            "stage": args.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.95],
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
            },
        },

        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": total_steps,
                "warmup_num_steps": args.warmup_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
            },
        },

        "wall_clock_breakdown": False,
    }

    if args.zero_stage == 0:
        ds_config["zero_optimization"] = {"stage": 0}

    if args.zero_stage == 3:
        ds_config["zero_optimization"].update(
            {
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_max_live_parameters": 1e8,
                "stage3_max_reuse_distance": 1e8,
            }
        )

    return ds_config


# ============================================================
# Validation / saving
# ============================================================

@torch.no_grad()
def validate_model_loss(
    model_engine,
    val_loader,
    args,
    global_step: int = 0,
) -> Optional[float]:
    if val_loader is None:
        return None

    model_engine.train()

    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        batch = move_to_device(batch, model_engine.device)

        outputs = model_engine(**batch)
        loss = get_output_value(outputs, "loss", None)

        if loss is None:
            continue

        if torch.isfinite(loss):
            total_loss += float(loss.item())
            num_batches += 1

        if args.max_val_batches > 0 and num_batches >= args.max_val_batches:
            break

    if num_batches == 0:
        logging.warning("Validation produced no valid loss batches.")
        return None

    avg_loss = total_loss / num_batches
    logging.info(f"[Validation] global_step={global_step}, val_loss={avg_loss:.6f}")

    return avg_loss


def save_training_checkpoint(
    model_engine,
    tokenizer,
    processor,
    args,
    tag: str,
) -> str:
    save_dir = os.path.join(args.output_dir, tag)
    os.makedirs(save_dir, exist_ok=True)

    model_engine.save_checkpoint(save_dir)

    if is_main_process(args.local_rank):
        tokenizer.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

        args_path = os.path.join(save_dir, "training_args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

        logging.info(f"Saved checkpoint to: {save_dir}")

    return save_dir


# ============================================================
# Numerical checks
# ============================================================

def _finite_stats_tensor(x: torch.Tensor) -> Dict[str, Any]:
    with torch.no_grad():
        x_detached = x.detach()

        has_nan = torch.isnan(x_detached).any().item()
        has_inf = torch.isinf(x_detached).any().item()
        finite_mask = torch.isfinite(x_detached)

        if finite_mask.any():
            finite_x = x_detached[finite_mask].float()
            return {
                "has_nan": has_nan,
                "has_inf": has_inf,
                "finite_ratio": finite_mask.float().mean().item(),
                "finite_min": finite_x.min().item(),
                "finite_max": finite_x.max().item(),
                "finite_mean": finite_x.mean().item(),
                "abs_max": finite_x.abs().max().item(),
            }

        return {
            "has_nan": has_nan,
            "has_inf": has_inf,
            "finite_ratio": 0.0,
            "finite_min": float("nan"),
            "finite_max": float("nan"),
            "finite_mean": float("nan"),
            "abs_max": float("nan"),
        }


def find_nonfinite_trainable_params(
    model: torch.nn.Module,
    max_show: int = 30,
) -> List[dict]:
    module = model.module if hasattr(model, "module") else model
    bad = []

    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue

        try:
            data = p.detach()
            if not torch.isfinite(data).all():
                bad.append(
                    {
                        "name": name,
                        "shape": tuple(data.shape),
                        "dtype": str(data.dtype),
                        **_finite_stats_tensor(data),
                    }
                )
        except Exception as e:
            bad.append(
                {
                    "name": name,
                    "shape": tuple(p.shape),
                    "dtype": str(p.dtype),
                    "error": str(e),
                }
            )

        if len(bad) >= max_show:
            break

    return bad


def find_nonfinite_trainable_grads(
    model: torch.nn.Module,
    max_show: int = 30,
) -> List[dict]:
    module = model.module if hasattr(model, "module") else model
    bad = []

    for name, p in module.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue

        try:
            grad = p.grad.detach()
            if not torch.isfinite(grad).all():
                bad.append(
                    {
                        "name": name,
                        "shape": tuple(grad.shape),
                        "dtype": str(grad.dtype),
                        **_finite_stats_tensor(grad),
                    }
                )
        except Exception as e:
            bad.append(
                {
                    "name": name,
                    "shape": tuple(p.shape),
                    "dtype": str(p.dtype),
                    "error": str(e),
                }
            )

        if len(bad) >= max_show:
            break

    return bad


def log_bad_tensors(prefix: str, bad_items: List[dict]) -> None:
    if not bad_items:
        logging.info(f"{prefix}: no non-finite tensors found.")
        return

    logging.error(f"{prefix}: found {len(bad_items)} non-finite tensors.")

    for item in bad_items:
        logging.error(
            f"{prefix}: "
            f"name={item.get('name')}, "
            f"shape={item.get('shape')}, "
            f"dtype={item.get('dtype')}, "
            f"has_nan={item.get('has_nan')}, "
            f"has_inf={item.get('has_inf')}, "
            f"finite_ratio={item.get('finite_ratio')}, "
            f"finite_min={item.get('finite_min')}, "
            f"finite_max={item.get('finite_max')}, "
            f"abs_max={item.get('abs_max')}, "
            f"error={item.get('error')}"
        )


__all__ = [
    "setup_logging",
    "set_seed",
    "is_main_process",
    "normalize_local_rank",
    "get_torch_dtype",
    "move_to_device",
    "get_output_value",
    "load_tokenizer_and_processor",
    "ensure_seg_token",
    "load_tokenizer_processor_and_seg_id",
    "create_reaseg_sft_parser",
    "validate_reaseg_args",
    "get_default_reaseg_sft_config",
    "build_lora_target_modules",
    "apply_lora_if_needed",
    "unwrap_reaseg_model",
    "freeze_all_parameters",
    "enable_qwen_visual_projector_training",
    "enable_reaseg_trainables",
    "print_trainable_summary",
    "print_dtype_summary",
    "build_optimizer_param_groups",
    "create_deepspeed_config",
    "validate_model_loss",
    "save_training_checkpoint",
    "find_nonfinite_trainable_params",
    "find_nonfinite_trainable_grads",
    "log_bad_tensors",
]