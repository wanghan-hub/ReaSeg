#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ReaSeg SFT training script.

This script is intentionally thin. Most reusable utilities are placed in
train/train_utils.py.

Training flow:
    1. Parse and validate args
    2. Load tokenizer / processor / [SEG] token id
    3. Build ReaSeg model
    4. Freeze all params
    5. Apply LoRA if enabled
    6. Enable selected ReaSeg trainable modules
    7. Build dataset / dataloader through DeepSpeed
    8. Train with CE + Dice + BCE loss
"""

import os
import sys
import json
import math
import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

import deepspeed


# train/train_sft.py -> reaseg/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from model.model import ReaSegForConditionalGeneration
from data.reason_seg_dataset import ReaSegReasonSegDataset
from data.collate import reaseg_collate_fn

from train.train_utils import (
    setup_logging,
    set_seed,
    is_main_process,
    get_torch_dtype,
    move_to_device,
    get_output_value,
    load_tokenizer_processor_and_seg_id,
    create_reaseg_sft_parser,
    validate_reaseg_args,
    apply_lora_if_needed,
    freeze_all_parameters,
    enable_reaseg_trainables,
    print_trainable_summary,
    print_dtype_summary,
    build_optimizer_param_groups,
    create_deepspeed_config,
    validate_model_loss,
    save_training_checkpoint,
    find_nonfinite_trainable_params,
    find_nonfinite_trainable_grads,
    log_bad_tensors,
    unwrap_reaseg_model,
)


# ============================================================
# Model / dataset builders
# ============================================================

def build_reaseg_model(args, tokenizer, seg_token_idx: int):
    """
    Build ReaSeg model and configure trainable modules.
    """
    torch_dtype = get_torch_dtype(args.precision)

    logging.info(f"Loading ReaSeg model from: {args.model_name_or_path}")
    logging.info(f"Using MedSAM checkpoint: {args.medsam_checkpoint}")
    logging.info(f"torch_dtype={torch_dtype}")

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
    
    # New [SEG] token may change tokenizer size.
    model.resize_token_embeddings(len(tokenizer))

    # Disable cache during training to avoid unnecessary memory usage.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Keep projector and MedSAM decoder/prompt modules in fp32.
    if hasattr(model, "keep_fp32_modules"):
        model.keep_fp32_modules()

    logging.info("Freezing all parameters first.")
    freeze_all_parameters(model)

    # PEFT will enable LoRA params.
    model = apply_lora_if_needed(model, args)

    # Enable ReaSeg-specific modules.
    enable_reaseg_trainables(model, args)

    print_trainable_summary(model)
    print_dtype_summary(model)

    return model


def build_datasets(args, tokenizer, processor, seg_token_idx: int):
    """
    Build train and optional validation datasets.

    If data_path is a directory:
        expects train.json and optionally val.json.

    If data_path is a single JSON file:
        it is used as train set only.
    """
    train_dataset = ReaSegReasonSegDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        processor=processor,
        image_size=args.image_size,
        max_seq_length=args.max_seq_length,
        split="train",
        seg_token=args.seg_token,
        seg_token_idx=seg_token_idx,
        force_seg_token=True,
        merge_masks_for_single_seg=True,
    )

    val_dataset = None

    if os.path.isdir(args.data_path):
        val_json = os.path.join(args.data_path, "val.json")

        if os.path.exists(val_json):
            val_dataset = ReaSegReasonSegDataset(
                data_path=args.data_path,
                tokenizer=tokenizer,
                processor=processor,
                image_size=args.image_size,
                max_seq_length=args.max_seq_length,
                split="val",
                seg_token=args.seg_token,
                seg_token_idx=seg_token_idx,
                force_seg_token=False,
                merge_masks_for_single_seg=True,
            )
            logging.info(f"Validation dataset enabled: {len(val_dataset)} samples.")
        else:
            logging.info("No val.json found; validation disabled.")
    else:
        logging.info("data_path is a JSON file; validation disabled.")

    logging.info(f"Train dataset size: {len(train_dataset)}")

    return train_dataset, val_dataset


def build_val_loader(args, val_dataset, tokenizer):
    if val_dataset is None:
        return None

    return DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda batch: reaseg_collate_fn(
            batch,
            tokenizer=tokenizer,
            precision=args.precision,
            return_legacy_keys=True,
        ),
    )


# ============================================================
# Debug helpers
# ============================================================

def debug_vl_alignment(batch, tokenizer, args, global_step: int) -> None:
    """
    Check Qwen2.5-VL image token / pixel feature alignment.
    """
    if not args.debug_vl_alignment:
        return

    if global_step >= args.debug_steps:
        return

    if not is_main_process(args.local_rank):
        return

    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    if image_token_id is None or image_token_id < 0:
        logging.warning("Cannot find <|image_pad|> token id; skip VL alignment debug.")
        return

    num_image_tokens = (batch["input_ids"] == image_token_id).sum().item()

    if "pixel_values" in batch and batch["pixel_values"] is not None:
        num_image_features = batch["pixel_values"].shape[0]
        pixel_shape = tuple(batch["pixel_values"].shape)
    else:
        num_image_features = 0
        pixel_shape = None

    logging.info(
        f"[VL Debug] image_tokens={num_image_tokens}, "
        f"pixel_values_shape={pixel_shape}, "
        f"num_image_features={num_image_features}, "
        f"image_grid_thw={batch.get('image_grid_thw', None)}"
    )

    if num_image_features > 0 and num_image_tokens == 0:
        decoded = tokenizer.decode(
            batch["input_ids"][0],
            skip_special_tokens=False,
        )

        logging.error("Decoded sample preview:")
        logging.error(decoded[:1500])

        raise RuntimeError(
            "Qwen2.5-VL image-token mismatch: pixel_values exist but "
            "<|image_pad|> tokens are missing."
        )


def debug_nonfinite_params_before_forward(model_engine, args, forward_step: int, global_step: int) -> None:
    if not args.debug_numerics:
        return

    if forward_step > args.debug_steps:
        return

    bad_params = find_nonfinite_trainable_params(model_engine)

    if bad_params:
        log_bad_tensors(
            prefix=f"[Param Debug BEFORE forward_step={forward_step}, global_step={global_step}]",
            bad_items=bad_params,
        )
        raise RuntimeError("Non-finite trainable parameters found before forward.")


def debug_nonfinite_grads_after_backward(model_engine, args, forward_step: int, global_step: int) -> None:
    if not args.debug_numerics:
        return

    if forward_step > args.debug_steps:
        return

    bad_grads = find_nonfinite_trainable_grads(model_engine)

    if bad_grads:
        log_bad_tensors(
            prefix=f"[Grad Debug AFTER backward forward_step={forward_step}, global_step={global_step}]",
            bad_items=bad_grads,
        )
        raise RuntimeError("Non-finite trainable gradients found after backward.")


def debug_nonfinite_params_after_step(model_engine, args, forward_step: int, global_step: int) -> None:
    if not args.debug_numerics:
        return

    if forward_step > args.debug_steps:
        return

    bad_params = find_nonfinite_trainable_params(model_engine)

    if bad_params:
        log_bad_tensors(
            prefix=f"[Param Debug AFTER optimizer step forward_step={forward_step}, global_step={global_step}]",
            bad_items=bad_params,
        )
        raise RuntimeError("Optimizer step produced non-finite trainable parameters.")


def log_first_output_structure(outputs, batch, model_engine, args, global_step: int) -> None:
    if global_step >= 3:
        return

    if not is_main_process(args.local_rank):
        return

    logging.info(f"[Debug] outputs type: {type(outputs)}")

    if isinstance(outputs, dict):
        logging.info(f"[Debug] outputs keys: {list(outputs.keys())}")
    elif hasattr(outputs, "keys"):
        try:
            logging.info(f"[Debug] outputs keys: {list(outputs.keys())}")
        except Exception:
            pass

    logging.info(
        f"[Debug] batch has medsam_images: {'medsam_images' in batch}, "
        f"gt_masks: {'gt_masks' in batch}, "
        f"original_sizes: {'original_sizes' in batch}, "
        f"resize_shapes: {'resize_shapes' in batch}"
    )

    try:
        logging.info(
            f"[Debug] model_engine.training={model_engine.training}, "
            f"module.training={model_engine.module.training}"
        )
    except Exception:
        pass


# ============================================================
# Training
# ============================================================

def train(args) -> None:
    set_seed(args.seed)
    setup_logging(args.output_dir, args.local_rank)

    if is_main_process(args.local_rank):
        logging.info("=" * 100)
        logging.info("Starting ReaSeg SFT training")
        logging.info(f"Args: {args}")
        logging.info("=" * 100)

    tokenizer, processor, seg_token_idx = load_tokenizer_processor_and_seg_id(
        model_name_or_path=args.model_name_or_path,
        seg_token=args.seg_token,
    )

    model = build_reaseg_model(args, tokenizer, seg_token_idx)
    train_dataset, val_dataset = build_datasets(args, tokenizer, processor, seg_token_idx)

    steps_per_epoch = max(
        1,
        math.ceil(
            len(train_dataset)
            / max(1, args.batch_size * args.gradient_accumulation_steps)
        ),
    )

    ds_config = create_deepspeed_config(args, steps_per_epoch=steps_per_epoch)

    if is_main_process(args.local_rank):
        logging.info("DeepSpeed config:")
        logging.info(json.dumps(ds_config, indent=2, ensure_ascii=False))

    model_parameters = build_optimizer_param_groups(model, args)

    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        training_data=train_dataset,
        config=ds_config,
        collate_fn=lambda batch: reaseg_collate_fn(
            batch,
            tokenizer=tokenizer,
            precision=args.precision,
            return_legacy_keys=True,
        ),
    )
    if args.resume_from_checkpoint:
        logging.info("=" * 100)
        logging.info(f"Loading DeepSpeed checkpoint from previous stage: {args.resume_from_checkpoint}")
        logging.info(f"resume_optimizer_states={args.resume_optimizer_states}")
        logging.info("=" * 100)

        load_path, client_state = model_engine.load_checkpoint(
            args.resume_from_checkpoint,
            load_module_strict=False,
            load_optimizer_states=args.resume_optimizer_states,
            load_lr_scheduler_states=args.resume_optimizer_states,
        )

        if load_path is None:
            raise RuntimeError(
                f"Failed to load DeepSpeed checkpoint from: {args.resume_from_checkpoint}"
            )

        logging.info("=" * 100)
        logging.info(f"Successfully loaded previous-stage checkpoint from: {load_path}")
        logging.info(f"client_state keys: {list(client_state.keys()) if isinstance(client_state, dict) else type(client_state)}")
        logging.info("=" * 100)
    else:
        logging.info("No resume_from_checkpoint is provided. Training from base Qwen + MedSAM initialization.")

    # DeepSpeed / PEFT may recast modules. Restore fp32 islands after wrapping.
    core_model = unwrap_reaseg_model(model_engine)
    if hasattr(core_model, "keep_fp32_modules"):
        core_model.keep_fp32_modules()
        logging.info("Post-DeepSpeed: called keep_fp32_modules().")

    # Extra robust fp32 enforcement for PEFT/DeepSpeed wrapped models.
    # This catches cases where DeepSpeed recasts partial trainable decoder layers to bf16.
    for name, module in model_engine.module.named_modules():
        lname = name.lower()
        if "seg_projector" in lname or "medsam" in lname or "visual_model" in lname:
            module.float()

    logging.info("Post-DeepSpeed: force-casted seg_projector and MedSAM-related modules to fp32.")

    if is_main_process(args.local_rank):
        print_trainable_summary(model_engine)
        print_dtype_summary(model_engine)

    val_loader = build_val_loader(args, val_dataset, tokenizer)

    global_step = 0
    forward_step = 0
    best_val_loss = float("inf")

    logging.info("Starting training loop.")

    for epoch in range(args.epochs):
        model_engine.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for loader_step, batch in enumerate(train_loader):
            forward_step += 1

            batch = move_to_device(batch, model_engine.device)

            debug_vl_alignment(batch, tokenizer, args, global_step)
            debug_nonfinite_params_before_forward(
                model_engine=model_engine,
                args=args,
                forward_step=forward_step,
                global_step=global_step,
            )

            outputs = model_engine(**batch)

            # log_first_output_structure(
            #     outputs=outputs,
            #     batch=batch,
            #     model_engine=model_engine,
            #     args=args,
            #     global_step=global_step,
            # )

            loss = get_output_value(outputs, "loss", None)
            ce_loss = get_output_value(outputs, "ce_loss", None)
            mask_loss = get_output_value(outputs, "mask_loss", None)
            raw_bce = get_output_value(outputs, "raw_mask_bce_loss", None)
            raw_dice = get_output_value(outputs, "raw_mask_dice_loss", None)

            if loss is None:
                if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    loss = outputs[0]
                else:
                    raise RuntimeError(
                        f"Cannot extract loss from outputs. "
                        f"type(outputs)={type(outputs)}, outputs={outputs}"
                    )

            if not torch.isfinite(loss):
                logging.error(
                    f"Non-finite loss detected at "
                    f"epoch={epoch}, loader_step={loader_step}, "
                    f"forward_step={forward_step}, global_step={global_step}, "
                    f"loss={loss}"
                )

                logging.error(
                    f"ce_loss={ce_loss}, mask_loss={mask_loss}, "
                    f"raw_bce={raw_bce}, raw_dice={raw_dice}"
                )

                bad_params = find_nonfinite_trainable_params(model_engine)
                if bad_params:
                    log_bad_tensors("[Param Debug NONFINITE loss]", bad_params)

                try:
                    model_engine.zero_grad()
                except Exception:
                    pass

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if args.skip_nonfinite_loss:
                    continue

                raise RuntimeError("Non-finite loss detected.")

            model_engine.backward(loss)

            debug_nonfinite_grads_after_backward(
                model_engine=model_engine,
                args=args,
                forward_step=forward_step,
                global_step=global_step,
            )

            model_engine.step()

            debug_nonfinite_params_after_step(
                model_engine=model_engine,
                args=args,
                forward_step=forward_step,
                global_step=global_step,
            )

            epoch_loss += float(loss.item())
            epoch_steps += 1
            global_step += 1

            if global_step % args.log_interval == 0 and is_main_process(args.local_rank):
                msg = (
                    f"Epoch {epoch}, Step {global_step}, "
                    f"Loss: {loss.item():.6f}"
                )

                if ce_loss is not None:
                    msg += f", CE: {ce_loss.item():.6f}"

                if mask_loss is not None:
                    msg += f", Mask: {mask_loss.item():.6f}"

                if raw_bce is not None:
                    msg += f", RawBCE: {raw_bce.item():.6f}"

                if raw_dice is not None:
                    msg += f", RawDice: {raw_dice.item():.6f}"

                logging.info(msg)

            if (
                args.eval_interval > 0
                and val_loader is not None
                and global_step % args.eval_interval == 0
            ):
                val_loss = validate_model_loss(
                    model_engine=model_engine,
                    val_loader=val_loader,
                    args=args,
                    global_step=global_step,
                )

                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss

                    save_training_checkpoint(
                        model_engine=model_engine,
                        tokenizer=tokenizer,
                        processor=processor,
                        args=args,
                        tag="best_model",
                    )

                model_engine.train()

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                save_training_checkpoint(
                    model_engine=model_engine,
                    tokenizer=tokenizer,
                    processor=processor,
                    args=args,
                    tag=f"checkpoint-{global_step}",
                )

        avg_epoch_loss = epoch_loss / max(1, epoch_steps)

        if is_main_process(args.local_rank):
            logging.info(
                f"Epoch {epoch} completed. "
                f"Average loss: {avg_epoch_loss:.6f}"
            )

    save_training_checkpoint(
        model_engine=model_engine,
        tokenizer=tokenizer,
        processor=processor,
        args=args,
        tag="final_model",
    )

    if is_main_process(args.local_rank):
        logging.info("ReaSeg SFT training completed.")


def main() -> None:
    parser = create_reaseg_sft_parser()
    args = parser.parse_args()
    args = validate_reaseg_args(args)

    train(args)


if __name__ == "__main__":
    main()