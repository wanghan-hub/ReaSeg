# -*- coding: utf-8 -*-
"""
SEG projector for ReaSeg.

This module maps Qwen2.5-VL [SEG] token hidden states into the
MedSAM/SAM mask decoder prompt embedding space.

Default mapping:
    Qwen hidden_size -> LayerNorm -> Linear -> prompt_dim

The forward pass disables autocast internally and casts inputs to the
projector parameter dtype to avoid fp32/bf16 mismatch under DeepSpeed.
"""

from typing import Optional

import torch
import torch.nn as nn


class ReaSegProjector(nn.Module):
    """
    Project [SEG] hidden states to MedSAM prompt embeddings.

    Args:
        hidden_size:
            Hidden size of Qwen2.5-VL language model.
        prompt_dim:
            MedSAM/SAM prompt embedding dimension. Usually 256.
        init_std:
            Initialization std for the projection layer.
        output_scale:
            Multiplicative scale applied to projector output.
        clamp_value:
            Clamp range for projector output.
        use_layernorm:
            Whether to apply LayerNorm before Linear.
    """

    def __init__(
        self,
        hidden_size: int,
        prompt_dim: int = 256,
        init_std: float = 1e-3,
        output_scale: float = 1.0,
        clamp_value: float = 10.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.prompt_dim = int(prompt_dim)
        self.output_scale = float(output_scale)
        self.clamp_value = float(clamp_value)
        self.use_layernorm = bool(use_layernorm)

        if self.use_layernorm:
            self.norm = nn.LayerNorm(self.hidden_size)
        else:
            self.norm = nn.Identity()

        self.proj = nn.Linear(self.hidden_size, self.prompt_dim)

        self.reset_parameters(init_std=init_std)

    def reset_parameters(self, init_std: float = 1e-3) -> None:
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        if isinstance(self.norm, nn.LayerNorm):
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states:
                Tensor with shape [N, hidden_size] or [B, N, hidden_size].

        Returns:
            Tensor with shape [N, prompt_dim] or [B, N, prompt_dim].
        """
        target_param = self.proj.weight
        target_device = target_param.device
        target_dtype = target_param.dtype

        if hidden_states.is_cuda:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                x = hidden_states.to(
                    device=target_device,
                    dtype=target_dtype,
                    non_blocking=True,
                )
                x = self.norm(x)
                x = self.proj(x)

                x = torch.nan_to_num(
                    x,
                    nan=0.0,
                    posinf=self.clamp_value,
                    neginf=-self.clamp_value,
                ).clamp(-self.clamp_value, self.clamp_value)

                return x * self.output_scale

        x = hidden_states.to(device=target_device, dtype=target_dtype)
        x = self.norm(x)
        x = self.proj(x)

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=self.clamp_value,
            neginf=-self.clamp_value,
        ).clamp(-self.clamp_value, self.clamp_value)

        return x * self.output_scale


class ReaSegMLPProjector(nn.Module):
    """
    Optional stronger MLP projector.

    This is not recommended as the default first-stage projector,
    but it can be used after the clean framework is stable.

    Structure:
        hidden_size -> LayerNorm -> Linear -> GELU -> Linear -> prompt_dim
    """

    def __init__(
        self,
        hidden_size: int,
        prompt_dim: int = 256,
        mlp_dim: Optional[int] = None,
        init_std: float = 1e-3,
        output_scale: float = 1.0,
        clamp_value: float = 10.0,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.prompt_dim = int(prompt_dim)
        self.mlp_dim = int(mlp_dim) if mlp_dim is not None else int(hidden_size)
        self.output_scale = float(output_scale)
        self.clamp_value = float(clamp_value)

        self.norm = nn.LayerNorm(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.mlp_dim, self.prompt_dim)

        self.reset_parameters(init_std=init_std)

    def reset_parameters(self, init_std: float = 1e-3) -> None:
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.fc1.bias)

        nn.init.normal_(self.fc2.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_param = self.fc2.weight
        target_device = target_param.device
        target_dtype = target_param.dtype

        if hidden_states.is_cuda:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                x = hidden_states.to(
                    device=target_device,
                    dtype=target_dtype,
                    non_blocking=True,
                )
                x = self.norm(x)
                x = self.fc1(x)
                x = self.act(x)
                x = self.fc2(x)

                x = torch.nan_to_num(
                    x,
                    nan=0.0,
                    posinf=self.clamp_value,
                    neginf=-self.clamp_value,
                ).clamp(-self.clamp_value, self.clamp_value)

                return x * self.output_scale

        x = hidden_states.to(device=target_device, dtype=target_dtype)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=self.clamp_value,
            neginf=-self.clamp_value,
        ).clamp(-self.clamp_value, self.clamp_value)

        return x * self.output_scale