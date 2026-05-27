"""
LVC-augmented SDM-UniPS Lightning modules.

Three ablation variants:
  * SDM_LVC_Loss_Module   — Version A: LVC reweights MSE loss only.
  * SDM_LVC_Feat_Module   — Version B: LVC scales encoder features only.
  * SDM_LVC_Full_Module   — Version C: both (proposed method).

All variants share the same TrainableSDMNet backbone (no architectural
changes — LVC inserts via the hooks `R_k_enc` / `pixel_weight_map`).
"""

from __future__ import annotations
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric

from .net_train import (
    TrainableSDMNet,
    normal_mse_loss,
    angular_mae,
)
from .module.lvc import LVCModule


# ============================================================================
# Shared base
# ============================================================================

class _LVCBase(pl.LightningModule):
    """Common machinery for all LVC variants."""

    use_lvc_loss = False
    use_lvc_feat = False

    def __init__(
        self,
        pixel_samples: int = 1024,
        canonical_resolution: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        step_size: int = 10,
        gamma: float = 0.8,
        max_epochs: int = 100,
        save_dir: str = "checkpoints",
        # LVC hyperparams
        lvc_alpha: float = 10.0,
        lvc_w_min: float = 0.1,
        lvc_per_channel: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.pixel_samples = pixel_samples
        self.canonical_resolution = canonical_resolution
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.save_dir = save_dir

        self.net = TrainableSDMNet(pixel_samples=pixel_samples, device=torch.device("cpu"))
        self.lvc = LVCModule(
            alpha=lvc_alpha,
            w_min=lvc_w_min,
            per_channel=lvc_per_channel,
        )

        # metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_mae = MeanMetric()
        self.val_mae = MeanMetric()

    # ------------------------------------------------------------------
    def _unpack_batch(self, batch: Dict[str, torch.Tensor]):
        I = batch["img"]
        N_gt = batch["nml"][:, :, :, :, 0]
        M = batch["mask"][:, :, :, :, 0]
        nImgArray = batch["numberOfImages"].reshape(-1, 1).cpu()
        return I, N_gt, M, nImgArray

    def _compute_lvc(self, I: torch.Tensor):
        """Compute (R_global, R_k) from input images.

        I : [B, 3, H, W, K]
        Returns:
          R_global : [B, H, W]
          R_k      : [B, K, H, W]
        """
        # LVC works in float32 for numerical stability
        I32 = I.float()
        with torch.no_grad():
            R_global, R_k = self.lvc(I32)
        return R_global, R_k

    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        I, N_gt, M, nImgArray = self._unpack_batch(batch)
        H = I.shape[2]

        # LVC reliability maps
        R_global = R_k = None
        if self.use_lvc_loss or self.use_lvc_feat:
            R_global, R_k = self._compute_lvc(I)

        # Convert to loss weights w(p) ∈ [w_min, 1] for loss reweighting
        if self.use_lvc_loss:
            w_map = self.lvc.loss_weights(R_global)        # [B, H, W]
        else:
            w_map = None

        # Feature scaling also uses the floored weight (not raw R_k) so that
        # features are never multiplied by ~0 → avoids numerical instability
        # (NaN observed at random init with raw R_k * R interaction).
        R_k_for_feat = self.lvc.loss_weights(R_k) if self.use_lvc_feat else None

        out = self.net.forward_train(
            I=I, M=M, N_gt=N_gt, nImgArray=nImgArray,
            decoder_resolution=H,
            canonical_resolution=self.canonical_resolution,
            R_k_enc=R_k_for_feat,
            pixel_weight_map=w_map,
        )
        loss = normal_mse_loss(out["n_pred"], out["n_true"], out["w_pix"])
        # MAE measured on unweighted mask for fair comparison
        mae = angular_mae(out["n_pred"], out["n_true"], torch.ones_like(out["w_pix"]))
        return {"loss": loss, "mae": mae}

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        d = self._step(batch)
        self.train_loss(d["loss"])
        self.train_mae(d["mae"])
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae",  self.train_mae,  on_step=False, on_epoch=True, prog_bar=True)
        # Underscore alias for ModelCheckpoint filename templating
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=False, on_epoch=True)
        return d["loss"]

    def validation_step(self, batch, batch_idx):
        d = self._step(batch)
        self.val_loss(d["loss"])
        self.val_mae(d["mae"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae",  self.val_mae,  on_step=False, on_epoch=True, prog_bar=True)
        # Underscore alias for ModelCheckpoint filename templating
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_mae",  self.val_mae,  on_step=False, on_epoch=True, prog_bar=False)
        return d["loss"]

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=self.step_size, gamma=self.gamma)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ============================================================================
# Variants
# ============================================================================

class SDM_LVC_Loss_Module(_LVCBase):
    """Version A: LVC reweights pixel-wise MSE loss only."""
    use_lvc_loss = True
    use_lvc_feat = False


class SDM_LVC_Feat_Module(_LVCBase):
    """Version B: LVC scales encoder features only."""
    use_lvc_loss = False
    use_lvc_feat = True


class SDM_LVC_Full_Module(_LVCBase):
    """Version C: full LVC — both loss reweighting and feature scaling."""
    use_lvc_loss = True
    use_lvc_feat = True
