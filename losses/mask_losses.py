# -*- coding: utf-8 -*-
"""
Mask and language-model losses for ReaSeg.

This file is intentionally independent from the model definition.
It contains:
    - Stable Dice loss for binary masks
    - Stable BCE-with-logits loss for binary masks
    - Memory-efficient causal language modeling CE loss
"""

from typing import Optional

import torch
import torch.nn.functional as F


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Stable Dice loss for binary mask supervision.

    Args:
        inputs:
            Raw mask logits with shape [N, H, W].
        targets:
            Binary mask targets with shape [N, H, W].
        num_masks:
            Normalization factor. Usually the number of valid masks.
        eps:
            Numerical stability term.

    Returns:
        Scalar Dice loss.
    """
    inputs = torch.nan_to_num(
        inputs.float(),
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    ).clamp(-20.0, 20.0)

    targets = torch.nan_to_num(
        targets.float(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)

    probs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)

    numerator = 2.0 * (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)

    loss = 1.0 - (numerator + eps) / (denominator + eps)
    loss = torch.nan_to_num(
        loss,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return loss.sum() / (num_masks + 1e-8)


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
) -> torch.Tensor:
    """
    Stable BCE-with-logits loss for binary mask supervision.

    Args:
        inputs:
            Raw mask logits with shape [N, H, W].
        targets:
            Binary mask targets with shape [N, H, W].
        num_masks:
            Normalization factor. Usually the number of valid masks.

    Returns:
        Scalar BCE loss.
    """
    inputs = torch.nan_to_num(
        inputs.float(),
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    ).clamp(-20.0, 20.0)

    targets = torch.nan_to_num(
        targets.float(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)

    loss = F.binary_cross_entropy_with_logits(
        inputs,
        targets,
        reduction="none",
    )

    loss = torch.nan_to_num(
        loss,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return loss.flatten(1).mean(dim=1).sum() / (num_masks + 1e-8)


def stable_lm_ce_loss(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    ignore_index: int = -100,
    clamp_value: float = 30.0,
) -> torch.Tensor:
    """
    Memory-efficient stable causal language modeling CE loss.

    This function only computes CE over valid supervised label positions,
    instead of converting the full [B, L, V] logits tensor to fp32.

    Args:
        logits:
            Language model logits with shape [B, L, V].
        labels:
            Labels with shape [B, L]. Positions with ignore_index are ignored.
        ignore_index:
            Label value to ignore.
        clamp_value:
            Clamp value for fp32 logits.

    Returns:
        Scalar CE loss.
    """
    if labels is None:
        return logits.sum() * 0.0

    labels = labels.to(device=logits.device)

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    valid_mask = shift_labels.ne(ignore_index)

    if valid_mask.sum() == 0:
        return logits.sum() * 0.0

    valid_logits = shift_logits[valid_mask]
    valid_labels = shift_labels[valid_mask].long()

    valid_logits = torch.nan_to_num(
        valid_logits.float(),
        nan=0.0,
        posinf=clamp_value,
        neginf=-clamp_value,
    ).clamp(-clamp_value, clamp_value)

    loss = F.cross_entropy(
        valid_logits,
        valid_labels,
        reduction="mean",
    )

    return torch.nan_to_num(
        loss,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def compute_mask_losses(
    pred_masks,
    gt_masks,
    bce_loss_weight: float = 2.0,
    dice_loss_weight: float = 0.5,
):
    """
    Compute weighted BCE + Dice mask loss for a batch.

    Args:
        pred_masks:
            List of tensors. Each tensor has shape [N_i, H_i, W_i].
        gt_masks:
            List of tensors. Each tensor has shape [M_i, H_i, W_i] or [H_i, W_i].
        bce_loss_weight:
            Weight for BCE loss.
        dice_loss_weight:
            Weight for Dice loss.

    Returns:
        dict with:
            mask_loss
            raw_mask_bce_loss
            raw_mask_dice_loss
            num_valid_masks
    """
    if len(pred_masks) == 0:
        raise ValueError("pred_masks is empty; cannot compute mask loss.")

    device = pred_masks[0].device

    total_bce = torch.tensor(0.0, device=device)
    total_dice = torch.tensor(0.0, device=device)
    num_valid_masks = 0

    for i, pred_mask in enumerate(pred_masks):
        gt_mask = gt_masks[i]

        if not isinstance(gt_mask, torch.Tensor):
            gt_mask = torch.as_tensor(gt_mask)

        gt_mask = gt_mask.to(
            device=pred_mask.device,
            dtype=pred_mask.dtype,
            non_blocking=True,
        )

        if pred_mask.dim() == 2:
            pred_mask = pred_mask.unsqueeze(0)

        if gt_mask.dim() == 2:
            gt_mask = gt_mask.unsqueeze(0)

        if pred_mask.numel() == 0 or gt_mask.numel() == 0:
            continue

        if pred_mask.shape[0] != gt_mask.shape[0]:
            if pred_mask.shape[0] == 1 and gt_mask.shape[0] > 1:
                gt_mask = gt_mask.max(dim=0, keepdim=True).values
            elif gt_mask.shape[0] == 1 and pred_mask.shape[0] > 1:
                gt_mask = gt_mask.expand(pred_mask.shape[0], -1, -1)
            else:
                continue

        n_masks = int(pred_mask.shape[0])
        num_valid_masks += n_masks

        total_bce = total_bce + sigmoid_ce_loss(
            pred_mask,
            gt_mask,
            num_masks=1,
        ) * n_masks

        total_dice = total_dice + dice_loss(
            pred_mask,
            gt_mask,
            num_masks=1,
        ) * n_masks

    if num_valid_masks == 0:
        raw_bce = torch.tensor(0.0, device=device)
        raw_dice = torch.tensor(0.0, device=device)
    else:
        raw_bce = total_bce / (num_valid_masks + 1e-8)
        raw_dice = total_dice / (num_valid_masks + 1e-8)

    mask_loss = bce_loss_weight * raw_bce + dice_loss_weight * raw_dice

    return {
        "mask_loss": mask_loss,
        "raw_mask_bce_loss": raw_bce,
        "raw_mask_dice_loss": raw_dice,
        "num_valid_masks": num_valid_masks,
    }