"""DPSGD-Global-Adapt Optimizer and Privacy Engine.

Implements the adaptive gradient scaling mitigation from Section 5 of:
"Disparate Impact in Differential Privacy from Gradient Misalignment" (arXiv:2206.07737)
"""

import math
from typing import Callable, List, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from opacus import PrivacyEngine
from opacus.optimizers.optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _mark_as_processed,
)


class GlobalAdaptiveOptimizer(DPOptimizer):
    """Adaptive DP-SGD Optimizer implementing DPSGD-Global-Adapt.

    Scales unclipped gradients with norm <= Z by (C / Z).
    Clips outlier gradients with norm > Z to C.
    Dynamically updates Z using a differentially private noisy count mechanism.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int,
        strict_max_grad_norm: float = 50.0,
        bits_noise_multiplier: float = 10.0,
        lr_Z: float = 0.1,
        threshold: float = 0.7,
        sample_rate: float = 1.0,
        accountant=None,
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )
        self.strict_max_grad_norm = strict_max_grad_norm  # Dynamic bound Z
        self.bits_noise_multiplier = bits_noise_multiplier  # sigma_2
        self.lr_Z = lr_Z  # eta_Z
        self.threshold = threshold  # tau
        self.sample_rate = sample_rate  # q = batch_size / N
        self.accountant = accountant
        self.current_per_sample_norms = None

    def clip_and_accumulate(self):
        """Performs dual-regime clipping:
        - If norm <= Z: scales by (C / Z)
        - If norm > Z: clips to C (scales by C / norm)
        """
        if len(self.grad_samples[0]) == 0:
            per_sample_clip_factor = torch.zeros((0,), device=self.grad_samples[0].device)
            self.current_per_sample_norms = torch.zeros((0,), device=self.grad_samples[0].device)
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            if per_param_norms:
                target_device = per_param_norms[0].device
                per_param_norms = [norm.to(target_device) for norm in per_param_norms]

            self.current_per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            standard_clip_factor = (
                self.max_grad_norm / (self.current_per_sample_norms + 1e-6)
            ).clamp(max=1.0)

            # C = max_grad_norm, Z = strict_max_grad_norm
            # Condition standard_clip_factor >= C / Z is equivalent to norm <= Z
            c_over_z = self.max_grad_norm / (self.strict_max_grad_norm + 1e-6)

            # Global-Adapt factor: scale by C/Z if norm <= Z, else clip to C
            per_sample_clip_factor = torch.where(
                standard_clip_factor >= c_over_z,
                torch.full_like(standard_clip_factor, c_over_z),
                standard_clip_factor,
            )

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad_sample = grad_sample.to(p.dtype)
            clip_factor_on_device = per_sample_clip_factor.to(grad_sample.device).to(p.dtype)
            grad = torch.einsum("i,i...", clip_factor_on_device, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def update_Z(self) -> float:
        """Adapts Z dynamically using private noisy count and records accountant step."""
        if self.current_per_sample_norms is None or len(self.current_per_sample_norms) == 0:
            return self.strict_max_grad_norm

        batch_size = len(self.current_per_sample_norms)
        # Fraction exceeding tau * Z
        cutoff = self.threshold * self.strict_max_grad_norm
        exceeding_count = (self.current_per_sample_norms > cutoff).float().sum().item()
        d_t = exceeding_count / batch_size

        # Add Gaussian noise with multiplier sigma_2
        noise = torch.normal(0, self.bits_noise_multiplier, (1,)).item() / batch_size
        noisy_d_t = d_t + noise

        # Update Z: Z_{t+1} = Z_t * exp(-eta_Z + noisy_d_t)
        factor = math.exp(-self.lr_Z + noisy_d_t)
        self.strict_max_grad_norm = self.strict_max_grad_norm * factor

        # Account for privacy loss of releasing the noisy count
        if self.accountant is not None:
            self.accountant.step(
                noise_multiplier=self.bits_noise_multiplier,
                sample_rate=self.sample_rate,
            )

        return self.strict_max_grad_norm

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Custom step performing clipping, noising, Z adaptation, and parameter update."""
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            loss = self.original_optimizer.step()
            # Dynamically update Z after gradient step
            self.update_Z()
            return loss
        else:
            return None


class GlobalAdaptivePrivacyEngine(PrivacyEngine):
    """Customized PrivacyEngine that wraps optimizer with GlobalAdaptiveOptimizer."""

    def __init__(
        self,
        strict_max_grad_norm: float = 50.0,
        bits_noise_multiplier: float = 10.0,
        lr_Z: float = 0.1,
        threshold: float = 0.7,
        sample_rate: float = 1.0,
        accountant: str = "rdp",
        **kwargs,
    ):
        try:
            super().__init__(accountant=accountant, **kwargs)
        except TypeError:
            super().__init__(**kwargs)

        self.strict_max_grad_norm = strict_max_grad_norm
        self.bits_noise_multiplier = bits_noise_multiplier
        self.lr_Z = lr_Z
        self.threshold = threshold
        self.sample_rate = sample_rate

    def _prepare_optimizer(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optimizer = GlobalAdaptiveOptimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            strict_max_grad_norm=self.strict_max_grad_norm,
            bits_noise_multiplier=self.bits_noise_multiplier,
            lr_Z=self.lr_Z,
            threshold=self.threshold,
            sample_rate=self.sample_rate,
            accountant=self.accountant,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
        )
        return optimizer
