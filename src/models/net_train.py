"""
Trainable SDM-UniPS network + PyTorch-Lightning module.

Why a subclass instead of editing sdm_unips/modules/model/model.py:
  * Original `Net.forward()` has `.detach()` on the predicted normals — fine
    for inference but breaks gradients during training.
  * Original is built around an inference-time pixel batching scheme; for
    training we want a fixed random pixel subset per batch item.
  * Keeps `sdm_unips/` intact so existing pretrained checkpoints still load.

LVC hook points (used by net_lvc.py):
  (A) feature scaling : `glc = glc * R_k_enc`  right after image_encoder.
  (B) loss weighting  : per-pixel weight `w(p)` from LVCModule.loss_weights().
"""

from __future__ import annotations
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric

# Make sdm_unips importable when train.py runs from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sdm_unips.modules.model.model import Net as _SDMNetBase
from sdm_unips.modules.model.model_utils import make_index_list
from sdm_unips.modules.utils import gauss_filter
from sdm_unips.modules.utils.ind2sub import ind2coords


# ============================================================================
# Trainable subclass of SDM Net
# ============================================================================

class TrainableSDMNet(_SDMNetBase):
    """SDM-UniPS Net adapted for training.

    Differences from base Net.forward():
      * No `.detach()` on predicted normals.
      * Accepts ground-truth normals + masks + (optional) LVC reliability maps.
      * Returns sampled predicted/target normals + per-pixel weights so the
        loss can be computed externally.
      * Always uses target='normal' (BRDF heads disabled for training scope).

    Args (forwarded to parent):
      pixel_samples : int  — number of foreground pixels sampled per batch item.
      device        : torch.device.
    """

    def __init__(self, pixel_samples: int, device: torch.device):
        super().__init__(pixel_samples=pixel_samples, output="normal", device=device)
        # Note: BRDF heads in parent are not instantiated when output='normal'.

    # ------------------------------------------------------------------
    # Encoder pass — exposed separately so LVC variants can hook in.
    # ------------------------------------------------------------------

    def encode(
        self,
        I: torch.Tensor,           # [B, 3, H, W, Nmax]
        M: torch.Tensor,           # [B, 1, H, W]
        nImgArray: torch.Tensor,   # [B] int
        canonical_resolution: int,
    ) -> torch.Tensor:
        """Run encoder; return GLC features [sumN, 256, H/4, W/4]."""
        B, C, H, W, Nmax = I.shape

        I_enc = I.permute(0, 4, 1, 2, 3).reshape(-1, C, H, W)   # [B*Nmax, 3, H, W]
        M_enc = M.unsqueeze(1).expand(-1, Nmax, -1, -1, -1).reshape(-1, 1, H, W)
        img_index = make_index_list(Nmax, nImgArray)             # numpy int array
        data = torch.cat([I_enc * M_enc, M_enc], dim=1)          # [B*Nmax, 4, H, W]
        data = data[img_index == 1, :, :, :]
        glc = self.image_encoder(data, nImgArray, canonical_resolution)
        return glc                                               # [sumN, 256, H/4, W/4]

    # ------------------------------------------------------------------
    # Decoder pass — samples pixels, predicts normals.
    # ------------------------------------------------------------------

    def decode_predict(
        self,
        glc: torch.Tensor,           # [sumN, 256, H_enc, W_enc]
        I: torch.Tensor,             # [B, 3, H, W, Nmax]
        M: torch.Tensor,             # [B, 1, H, W]
        N_gt: torch.Tensor,          # [B, 3, H, W]
        nImgArray: torch.Tensor,     # [B] int
        decoder_resolution: int,
        canonical_resolution: int,
        pixel_weight_map: Optional[torch.Tensor] = None,   # [B, H_dec, W_dec] in [0, 1]
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W, Nmax = I.shape

        # Resize images, mask, normals to decoder resolution
        img = I.permute(0, 4, 1, 2, 3).reshape(-1, C, H, W)
        img_index = make_index_list(Nmax, nImgArray)
        img = img[img_index == 1, :, :, :]

        decoder_imgsize = (decoder_resolution, decoder_resolution)
        I_dec = F.interpolate(img.float(), size=decoder_imgsize, mode="bilinear", align_corners=False)
        M_dec = F.interpolate(M.float(), size=decoder_imgsize, mode="nearest")
        N_dec = F.interpolate(N_gt.float(), size=decoder_imgsize, mode="bilinear", align_corners=False)

        if pixel_weight_map is not None:
            # [B, H, W] -> [B, 1, H, W] for interpolation
            w_dec = F.interpolate(
                pixel_weight_map.unsqueeze(1).float(),
                size=decoder_imgsize, mode="bilinear", align_corners=False,
            )                                  # [B, 1, Hd, Wd]
        else:
            w_dec = None

        # Smooth glc to canonical_resolution match (mirrors original)
        if self.glc_smoothing:
            f_scale = max(decoder_resolution // canonical_resolution, 1)
            smoothing = gauss_filter.gauss_filter(
                glc.shape[1], 10 * f_scale + 1, 1
            ).to(glc.device)
            glc = smoothing(glc)

        Hd = decoder_resolution
        Wd = decoder_resolution
        n_pred_list = []
        n_true_list = []
        w_pix_list = []

        # Per-batch-item pixel sampling
        ptr = 0
        for b in range(B):
            n_imgs_b = int(nImgArray[b])
            target = range(ptr, ptr + n_imgs_b)
            ptr += n_imgs_b

            m_flat = M_dec[b, 0].reshape(-1)                 # [Hd*Wd]
            fg_ids = torch.nonzero(m_flat > 0.5, as_tuple=False).squeeze(-1)
            if fg_ids.numel() == 0:
                # No foreground pixels — produce zero loss entry
                n_pred_list.append(torch.zeros(self.pixel_samples, 3, device=glc.device, dtype=glc.dtype))
                n_true_list.append(torch.zeros(self.pixel_samples, 3, device=glc.device, dtype=glc.dtype))
                w_pix_list.append(torch.zeros(self.pixel_samples, device=glc.device, dtype=glc.dtype))
                continue

            # Random sample with replacement if too few foreground pixels
            if fg_ids.numel() >= self.pixel_samples:
                perm = torch.randperm(fg_ids.numel(), device=fg_ids.device)[: self.pixel_samples]
            else:
                perm = torch.randint(0, fg_ids.numel(), (self.pixel_samples,), device=fg_ids.device)
            ids = fg_ids[perm]                                # [S]
            S = ids.numel()

            # Sample input images at these pixels: [S, N_b, 3]
            o_ = I_dec[target].reshape(n_imgs_b, C, Hd * Wd).permute(2, 0, 1)
            o_ids = o_[ids, :, :]

            # Sample glc features via grid_sample (sub-pixel from coarse grid)
            # ind2coords uses torch.div internally → must pass torch tensor, not numpy.
            coords = ind2coords(np.array((Hd, Wd)), ids.cpu())   # [1, 1, S, 2]
            coords = coords.expand(n_imgs_b, -1, -1, -1).to(glc.device)
            glc_ids = F.grid_sample(
                glc[target], coords, mode="bilinear", align_corners=False
            ).reshape(n_imgs_b, -1, S).permute(2, 0, 1)        # [S, N_b, F]

            # Decoder: cat input pixel RGB with glc -> upsample -> aggregate -> regress
            x = torch.cat([o_ids, glc_ids], dim=2)             # [S, N_b, 256+3]
            glc_ids2 = self.glc_upsample(x)
            x = torch.cat([o_ids, glc_ids2], dim=2)
            x = self.glc_aggregation(x)                        # [S, 384]
            x_n, _ = self.regressor(x, S)                      # [S, 3]
            x_n = F.normalize(x_n, dim=1, p=2)                 # unit normal

            # Target normal at same pixels (re-normalised after interpolation)
            n_true_flat = N_dec[b].reshape(3, Hd * Wd).permute(1, 0)   # [Hd*Wd, 3]
            n_true = n_true_flat[ids, :]
            n_true = F.normalize(n_true, p=2, dim=-1)

            # Per-pixel LVC weight (1.0 if disabled)
            if w_dec is not None:
                w_flat = w_dec[b, 0].reshape(-1)
                w_pix = w_flat[ids]
            else:
                w_pix = torch.ones(S, device=glc.device, dtype=x_n.dtype)

            n_pred_list.append(x_n)
            n_true_list.append(n_true)
            w_pix_list.append(w_pix.to(x_n.dtype))

        n_pred = torch.stack(n_pred_list, dim=0)               # [B, S, 3]
        n_true = torch.stack(n_true_list, dim=0)               # [B, S, 3]
        w_pix = torch.stack(w_pix_list, dim=0)                 # [B, S]
        return {"n_pred": n_pred, "n_true": n_true, "w_pix": w_pix}

    # ------------------------------------------------------------------
    # Full-image inference (for eval/demo) — fixed numpy/GPU indexing bug
    # of original SDM Net.forward()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_full(
        self,
        I: torch.Tensor,
        M: torch.Tensor,
        nImgArray: torch.Tensor,
        decoder_resolution,
        canonical_resolution,
    ) -> torch.Tensor:
        """Full-image normal prediction for ALL foreground pixels.

        Equivalent to SDM Net.forward() but:
          * Replaces numpy nonzero/perm with torch ops (PyTorch 2.x compat)
          * Returns single normal tensor [B, 3, H, W] (no BRDF)
          * `@no_grad()` decorator since this is inference-only.

        Args (all on same device, dtype-compatible):
            I: [B, 3, H, W, Nmax]
            M: [B, 1, H, W]
            nImgArray: tensor [B] or [B, 1] of actual image counts
            decoder_resolution, canonical_resolution: int OR tensor [B, 1]
        """
        # Coerce scalars
        if torch.is_tensor(decoder_resolution):
            decoder_resolution = int(decoder_resolution.reshape(-1)[0].item())
        if torch.is_tensor(canonical_resolution):
            canonical_resolution = int(canonical_resolution.reshape(-1)[0].item())
        # nImgArray for make_index_list needs to be iterable on CPU
        if torch.is_tensor(nImgArray):
            nImgArray_cpu = nImgArray.reshape(-1, 1).cpu()
        else:
            nImgArray_cpu = torch.tensor(nImgArray).reshape(-1, 1)

        B, C_in, H_in, W_in, Nmax = I.shape
        device = I.device

        # ---- Encoder ----
        I_enc = I.permute(0, 4, 1, 2, 3).reshape(-1, C_in, H_in, W_in)
        M_enc = M.unsqueeze(1).expand(-1, Nmax, -1, -1, -1).reshape(-1, 1, H_in, W_in)
        img_index = make_index_list(Nmax, nImgArray_cpu)
        data = torch.cat([I_enc * M_enc, M_enc], dim=1)
        data = data[img_index == 1, :, :, :]
        glc = self.image_encoder(data, nImgArray_cpu, canonical_resolution)

        # ---- Decoder full-image ----
        img = I.permute(0, 4, 1, 2, 3).reshape(-1, C_in, H_in, W_in)
        img = img[img_index == 1, :, :, :]
        decoder_imgsize = (decoder_resolution, decoder_resolution)
        I_dec = F.interpolate(img.float(), size=decoder_imgsize,
                              mode="bilinear", align_corners=False)
        M_dec = F.interpolate(M.float(), size=decoder_imgsize, mode="nearest")

        H = W = decoder_resolution
        C = I_dec.shape[1]
        nout = torch.zeros(B, H * W, 3, device=device, dtype=I_dec.dtype)

        if self.glc_smoothing:
            f_scale = max(decoder_resolution // canonical_resolution, 1)
            smoothing = gauss_filter.gauss_filter(
                glc.shape[1], 10 * f_scale + 1, 1
            ).to(glc.device)
            glc = smoothing(glc)

        ptr = 0
        for b in range(B):
            n_imgs_b = int(nImgArray_cpu[b].item())
            target = range(ptr, ptr + n_imgs_b)
            ptr += n_imgs_b

            m_flat = M_dec[b, 0].reshape(-1)                              # [H*W] on device
            ids_full = torch.nonzero(m_flat > 0.5, as_tuple=False).squeeze(-1)  # on device
            if ids_full.numel() == 0:
                continue
            # Permute (random order for fair chunk split)
            perm = torch.randperm(ids_full.numel(), device=device)
            ids_full = ids_full[perm]

            # Split into chunks of pixel_samples
            if ids_full.numel() > self.pixel_samples:
                num_split = (ids_full.numel() + self.pixel_samples - 1) // self.pixel_samples
                idset = torch.tensor_split(ids_full, num_split)
            else:
                idset = [ids_full]

            o_ = I_dec[target].reshape(n_imgs_b, C, H * W).permute(2, 0, 1)   # [H*W, N, C]
            for ids in idset:
                S = ids.numel()
                o_ids = o_[ids, :, :]
                # ind2coords expects CPU tensor of indices
                coords = ind2coords(np.array((H, W)), ids.cpu())
                coords = coords.expand(n_imgs_b, -1, -1, -1).to(device)
                glc_ids = F.grid_sample(
                    glc[target], coords, mode="bilinear", align_corners=False
                ).reshape(n_imgs_b, -1, S).permute(2, 0, 1)

                x = torch.cat([o_ids, glc_ids], dim=2)
                glc_ids2 = self.glc_upsample(x)
                x = torch.cat([o_ids, glc_ids2], dim=2)
                x = self.glc_aggregation(x)
                x_n, _ = self.regressor(x, S)
                X_n = F.normalize(x_n, dim=1, p=2)
                nout[b, ids, :] = X_n.to(nout.dtype)

        nout = nout.permute(0, 2, 1).reshape(B, 3, H, W)
        return nout

    # ------------------------------------------------------------------
    # Public training forward
    # ------------------------------------------------------------------

    def forward_train(
        self,
        I: torch.Tensor,
        M: torch.Tensor,
        N_gt: torch.Tensor,
        nImgArray: torch.Tensor,
        decoder_resolution: int,
        canonical_resolution: int,
        R_k_enc: Optional[torch.Tensor] = None,        # [B, Nmax, H_enc, W_enc] LVC per-image
        pixel_weight_map: Optional[torch.Tensor] = None,  # [B, H_dec, W_dec] LVC global
    ) -> Dict[str, torch.Tensor]:
        """End-to-end train forward."""
        glc = self.encode(I, M, nImgArray, canonical_resolution)

        # LVC feature scaling (Version B / Full): scale glc per-image by R_k.
        # glc shape: [sumN, 256, H_enc, W_enc]; R_k_enc: [B, Nmax, H_enc, W_enc].
        if R_k_enc is not None:
            B, Nmax = R_k_enc.shape[:2]
            img_index = make_index_list(Nmax, nImgArray)
            R_flat = R_k_enc.reshape(B * Nmax, R_k_enc.shape[2], R_k_enc.shape[3])
            R_flat = R_flat[img_index == 1]                       # [sumN, H, W]
            # Resize R to glc spatial size if needed
            if R_flat.shape[-2:] != glc.shape[-2:]:
                R_flat = F.interpolate(
                    R_flat.unsqueeze(1), size=glc.shape[-2:],
                    mode="bilinear", align_corners=False
                ).squeeze(1)
            glc = glc * R_flat.unsqueeze(1).to(glc.dtype)         # broadcast over channel

        return self.decode_predict(
            glc, I, M, N_gt, nImgArray,
            decoder_resolution, canonical_resolution,
            pixel_weight_map=pixel_weight_map,
        )


# ============================================================================
# Loss helpers
# ============================================================================

def normal_mse_loss(n_pred: torch.Tensor, n_true: torch.Tensor, w_pix: torch.Tensor) -> torch.Tensor:
    """Weighted MSE between unit-normalised normals.

    n_pred, n_true : [B, S, 3]
    w_pix          : [B, S]
    """
    # Guard against NaN/Inf in predictions (can happen at random init with
    # extreme LVC feature scaling — gradient clip handles training)
    n_pred = torch.nan_to_num(n_pred, nan=0.0, posinf=1.0, neginf=-1.0)
    sq = (n_pred - n_true) ** 2                                 # [B, S, 3]
    w = w_pix.unsqueeze(-1).expand_as(sq)
    denom = w.sum().clamp(min=1e-8)
    return (sq * w).sum() / denom


def angular_mae(n_pred: torch.Tensor, n_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean angular error in degrees (metric, not for backprop).

    n_pred, n_true : [B, S, 3] (already unit-normalised)
    mask           : [B, S] (foreground / valid)
    """
    cos = (n_pred * n_true).sum(-1).clamp(-1.0, 1.0)             # [B, S]
    err_rad = torch.acos(cos)
    err_deg = err_rad * (180.0 / torch.pi)
    m = mask.float()
    denom = m.sum().clamp(min=1e-8)
    return (err_deg * m).sum() / denom


# ============================================================================
# PyTorch Lightning module — BASELINE (no LVC)
# ============================================================================

class SDM_UniPSModule(pl.LightningModule):
    """Baseline trainer wrapping TrainableSDMNet — no LVC, MSE-only loss."""

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

        # Net built at .setup() so we know device
        self.net = TrainableSDMNet(pixel_samples=pixel_samples, device=torch.device("cpu"))

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_mae = MeanMetric()
        self.val_mae = MeanMetric()

    # --------------------------------------------------------------
    # Forward / step
    # --------------------------------------------------------------

    def _unpack_batch(self, batch: Dict[str, torch.Tensor]):
        I = batch["img"]                              # [B, 3, H, W, K]
        N_gt = batch["nml"][:, :, :, :, 0]            # [B, 3, H, W]
        M = batch["mask"][:, :, :, :, 0]              # [B, 1, H, W]
        # SDM convention: nImgArray shape [B, 1] on CPU (used as numpy index inside)
        nImgArray = batch["numberOfImages"].reshape(-1, 1).cpu()
        return I, N_gt, M, nImgArray

    def _step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        I, N_gt, M, nImgArray = self._unpack_batch(batch)
        H = I.shape[2]
        out = self.net.forward_train(
            I=I, M=M, N_gt=N_gt, nImgArray=nImgArray,
            decoder_resolution=H,
            canonical_resolution=self.canonical_resolution,
            R_k_enc=None,
            pixel_weight_map=None,
        )
        loss = normal_mse_loss(out["n_pred"], out["n_true"], out["w_pix"])
        mae = angular_mae(out["n_pred"], out["n_true"], out["w_pix"])
        return {"loss": loss, "mae": mae}

    def training_step(self, batch, batch_idx):
        d = self._step(batch)
        self.train_loss(d["loss"])
        self.train_mae(d["mae"])
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae",  self.train_mae,  on_step=False, on_epoch=True, prog_bar=True)
        # Underscore alias for ModelCheckpoint filename templating (Lightning
        # doesn't auto-translate "train/loss" -> "train_loss" in filename)
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

    # --------------------------------------------------------------
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
