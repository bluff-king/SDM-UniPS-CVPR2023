"""
Lighting Variation Confidence (LVC) module.

Estimates per-pixel reconstruction reliability R(p) from the normalised
intensity variance (CV2) across K input images. Requires no learned
parameters.

Reference: thesis Chapter 3, Section 3.3.
Ported from LINO_UniPS_LVCmodule (unchanged math).
"""

import torch
import torch.nn as nn


class LVCModule(nn.Module):
    """
    Parameter-free reliability estimator based on the coefficient of variation
    squared (CV2) of pixel intensities across input images.

    Args:
        alpha       : sensitivity of the exponential mapping (default 10).
        w_min       : minimum training weight for low-reliability pixels (default 0.1).
        eps         : numerical stability constant (default 1e-6).
        per_channel : if True, compute CV2 per channel and take the max over
                      channels (preferred for specular scenes); otherwise use
                      grayscale average.
    """

    def __init__(
        self,
        alpha: float = 10.0,
        w_min: float = 0.1,
        eps: float = 1e-6,
        per_channel: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.w_min = w_min
        self.eps = eps
        self.per_channel = per_channel

    # ------------------------------------------------------------------
    def _cv2_along_last(self, x: torch.Tensor) -> torch.Tensor:
        """CV2 = Var / (mean^2 + eps) along the last axis."""
        mu = x.mean(dim=-1)
        var = x.var(dim=-1, unbiased=False)
        return var / (mu ** 2 + self.eps)

    # ------------------------------------------------------------------
    def reliability_map(self, images: torch.Tensor) -> torch.Tensor:
        """Global per-pixel reliability R(p) in [0, 1].

        images : [B, C, H, W, N]
        Returns: [B, H, W]
        """
        if self.per_channel:
            cv2 = self._cv2_along_last(images)   # [B, C, H, W]
            cv2 = cv2.amax(dim=1)                # [B, H, W]
        else:
            gray = images.mean(dim=1)            # [B, H, W, N]
            cv2 = self._cv2_along_last(gray)
        return 1.0 - torch.exp(-self.alpha * cv2)

    def loss_weights(self, reliability: torch.Tensor) -> torch.Tensor:
        """w(p) = w_min + (1 - w_min) * R(p), shape preserved."""
        return self.w_min + (1.0 - self.w_min) * reliability

    def per_image_reliability(self, images: torch.Tensor) -> torch.Tensor:
        """Per-image per-pixel reliability R_k(p) in [0, 1].

        images : [B, C, H, W, N]
        Returns: [B, N, H, W]
        """
        mu = images.mean(dim=-1, keepdim=True)
        diff_sq = (images - mu) ** 2
        if self.per_channel:
            cv2_k = diff_sq / (mu ** 2 + self.eps)
            cv2_k = cv2_k.amax(dim=1)
        else:
            diff_sq = diff_sq.mean(dim=1)
            mu_g = mu.mean(dim=1)
            cv2_k = diff_sq / (mu_g ** 2 + self.eps)
        r_k = 1.0 - torch.exp(-self.alpha * cv2_k)
        return r_k.permute(0, 3, 1, 2)

    def forward(self, images: torch.Tensor):
        return self.reliability_map(images), self.per_image_reliability(images)
