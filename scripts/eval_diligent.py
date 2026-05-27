"""
Evaluate SDM-LVC checkpoint on DiLiGenT benchmark.

DiLiGenT dataset structure (10 objects):
    DiLiGenT/
    ├── bear/
    │   ├── 001.png ... 096.png    (96 directional-light images)
    │   ├── mask.png               (object mask)
    │   ├── Normal_gt.png          (encoded GT normal, optional)
    │   └── normal.txt             (raw GT normal as N×3 ASCII)
    ├── buddha/
    ├── cat/
    ├── ...

Usage:
    python scripts/eval_diligent.py \
        --hf_repo HUST-CVLab-PS/sdm-lvc-checkpoints \
        --hf_path lvc_full/lvc_full_paper_e200/last.ckpt \
        --variant lvc_full \
        --diligent_root ~/DiLiGenT \
        --num_input_images 10 \
        --image_size 256

Download DiLiGenT:
    https://sites.google.com/site/photometricstereodata/single
    (or use 'mvs' folder reorganised — same per-object structure)
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.net_train import SDM_UniPSModule
from src.models.net_lvc import (
    SDM_LVC_Loss_Module,
    SDM_LVC_Feat_Module,
    SDM_LVC_Full_Module,
)

VARIANT_MAP = {
    "baseline":  SDM_UniPSModule,
    "lvc_loss":  SDM_LVC_Loss_Module,
    "lvc_feat":  SDM_LVC_Feat_Module,
    "lvc_full":  SDM_LVC_Full_Module,
}


# ============================================================================
# DiLiGenT loader
# ============================================================================

def load_diligent_scene(scene_dir: str, num_imgs: int, image_size: int) -> dict:
    """Load 1 DiLiGenT object: K images + mask + GT normal.

    Returns dict:
        I    : [1, 3, H, W, K]  float32 tensor
        M    : [1, 1, H, W]     float32 tensor
        N_gt : [H, W, 3]        float32 numpy (camera-space normal in [-1, 1])
        valid: [H, W]           bool numpy (mask after resize)
        name : str
    """
    name = os.path.basename(os.path.normpath(scene_dir))

    # --- Find K image files ---
    candidates = sorted(
        glob.glob(os.path.join(scene_dir, "*.png"))
        + glob.glob(os.path.join(scene_dir, "*.jpg"))
        + glob.glob(os.path.join(scene_dir, "*.bmp"))
    )
    # Skip non-light files
    candidates = [p for p in candidates if not any(
        kw in os.path.basename(p).lower()
        for kw in ("mask", "normal", "albedo", "depth", "rough", "metal")
    )]
    if len(candidates) == 0:
        raise RuntimeError(f"No image files in {scene_dir}")

    # Random subset
    if len(candidates) > num_imgs:
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(len(candidates), num_imgs, replace=False))
        paths = [candidates[i] for i in idx]
    else:
        paths = candidates

    # --- Mask ---
    mask_path = None
    for cand in ["mask.png", "Mask.png", "mask.jpg"]:
        p = os.path.join(scene_dir, cand)
        if os.path.isfile(p):
            mask_path = p
            break
    if mask_path is None:
        raise RuntimeError(f"No mask.png found in {scene_dir}")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"failed to read mask: {mask_path}")
    H_orig, W_orig = mask.shape

    # --- GT normal: prefer normal.txt (raw float), fallback to Normal_gt.png ---
    N_gt = None
    txt_path = os.path.join(scene_dir, "normal.txt")
    if os.path.isfile(txt_path):
        # normal.txt: H*W lines of "nx ny nz" (only foreground pixels)
        # Need to map back to (H, W, 3) using mask
        try:
            vals = np.loadtxt(txt_path).astype(np.float32)  # [Nfg, 3]
            N_gt = np.zeros((H_orig, W_orig, 3), dtype=np.float32)
            fg_idx = np.where(mask > 127)
            if vals.shape[0] == len(fg_idx[0]):
                N_gt[fg_idx[0], fg_idx[1], :] = vals
            else:
                # Try full-image format (H*W rows)
                N_gt = vals.reshape(H_orig, W_orig, 3)
        except Exception as e:
            print(f"  [{name}] normal.txt parse failed: {e}, falling back to PNG")
            N_gt = None
    if N_gt is None:
        for cand in ["Normal_gt.png", "normal_gt.png", "gt_normal.png"]:
            p = os.path.join(scene_dir, cand)
            if os.path.isfile(p):
                raw = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
                if raw.ndim == 3 and raw.shape[-1] == 3:
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                # Most encoded as (n+1)/2 in [0,1] (uint8) → decode
                raw = raw / 255.0 if raw.max() > 1.5 else raw
                N_gt = raw * 2.0 - 1.0
                break
    if N_gt is None:
        raise RuntimeError(f"No GT normal found in {scene_dir} "
                           f"(tried normal.txt and Normal_gt.png)")

    # --- Read images, resize to image_size ---
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    imgs = np.stack(imgs, axis=-1)  # [H, W, 3, K]

    mask_r = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    M = (mask_r > 127).astype(np.float32)  # [H, W]

    # Resize GT (keep at image_size for fair MAE comparison)
    if N_gt.shape[:2] != (image_size, image_size):
        N_gt = cv2.resize(N_gt, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    # Re-normalise GT to unit (may have introduced small drift after resize)
    norm = np.linalg.norm(N_gt, axis=-1, keepdims=True)
    N_gt = N_gt / (norm + 1e-8)
    N_gt[M < 0.5] = 0  # zero outside mask

    # Mask inputs
    imgs = imgs * M[..., None, None]

    # To tensors (SDM expects [B, 3, H, W, K] and [B, 1, H, W])
    I = torch.from_numpy(imgs).permute(2, 0, 1, 3).unsqueeze(0).float()
    M_t = torch.from_numpy(M).unsqueeze(0).unsqueeze(0).float()

    return {
        "I": I, "M": M_t, "N_gt": N_gt, "valid": M > 0.5,
        "name": name, "K": len(paths),
    }


# ============================================================================
# Inference using inherited SDM Net.forward (full image, all foreground pixels)
# ============================================================================

@torch.no_grad()
def predict_normal_full(net, I, M, K, canonical_resolution, device):
    """Full-image normal prediction using TrainableSDMNet.predict_full.

    Uses our fixed predict_full() instead of parent's Net.forward() which has
    a numpy/GPU indexing bug on newer PyTorch.

    Returns: [H, W, 3] numpy unit normal
    """
    H = I.shape[2]
    nImgArray = torch.tensor([[K]], dtype=torch.int32)

    I_d = I.to(device)
    M_d = M.to(device)

    # Our trainable subclass exposes predict_full() — fixed indexing.
    nout = net.predict_full(I_d, M_d, nImgArray,
                            decoder_resolution=H,
                            canonical_resolution=canonical_resolution)
    # nout: [1, 3, H, W]
    N = nout[0].permute(1, 2, 0).cpu().numpy()
    norm = np.linalg.norm(N, axis=-1, keepdims=True)
    N = N / (norm + 1e-8)
    return N


# ============================================================================
# Metric
# ============================================================================

def angular_error_deg(N_pred: np.ndarray, N_gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-pixel angular error in degrees, only on valid pixels."""
    cos = (N_pred * N_gt).sum(axis=-1).clip(-1.0, 1.0)
    err_rad = np.arccos(cos)
    err_deg = np.degrees(err_rad)
    return err_deg[valid]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", type=str, default=None)
    parser.add_argument("--hf_path", type=str, default=None)
    parser.add_argument("--ckpt_local", type=str, default=None)
    parser.add_argument("--download_dir", type=str, default="~/hf_ckpts")
    parser.add_argument("--variant", required=True, choices=list(VARIANT_MAP.keys()))
    parser.add_argument("--diligent_root", required=True,
                        help="Path to DiLiGenT folder (containing bear/, buddha/, ...)")
    parser.add_argument("--num_input_images", type=int, default=10,
                        help="K = number of light images to feed (paper recommends 10-16)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--canonical_resolution", type=int, default=128)
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="Optional dir to save predicted normal maps (PNG)")
    args = parser.parse_args()

    # Resolve ckpt
    if args.ckpt_local:
        ckpt_path = os.path.expanduser(args.ckpt_local)
    else:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id=args.hf_repo,
            filename=args.hf_path,
            repo_type="model",
            local_dir=os.path.expanduser(args.download_dir),
        )
    print(f"[ckpt] {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModuleCls = VARIANT_MAP[args.variant]
    module = ModuleCls.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
    module.eval().to(device)
    net = module.net   # TrainableSDMNet

    # Find DiLiGenT scenes
    root = os.path.expanduser(args.diligent_root)
    scenes = sorted([d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d))])
    # Filter: must have at least 10 .png/.bmp + mask
    valid_scenes = []
    for s in scenes:
        sd = os.path.join(root, s)
        n_imgs = len(glob.glob(os.path.join(sd, "*.png")) + glob.glob(os.path.join(sd, "*.bmp")))
        has_mask = any(os.path.isfile(os.path.join(sd, c)) for c in ["mask.png", "Mask.png"])
        if n_imgs >= 5 and has_mask:
            valid_scenes.append(s)
    print(f"[diligent] found {len(valid_scenes)} scenes: {valid_scenes}")

    if args.save_predictions:
        os.makedirs(os.path.expanduser(args.save_predictions), exist_ok=True)

    results = []
    for scene_name in valid_scenes:
        scene_dir = os.path.join(root, scene_name)
        try:
            t0 = time.time()
            data = load_diligent_scene(scene_dir, args.num_input_images, args.image_size)
            N_pred = predict_normal_full(
                net, data["I"], data["M"], data["K"],
                args.canonical_resolution, device,
            )
            errs = angular_error_deg(N_pred, data["N_gt"], data["valid"])
            mae = float(errs.mean())
            results.append((scene_name, mae, errs.shape[0]))
            print(f"  {scene_name:<15s}  MAE = {mae:6.2f}°  "
                  f"(K={data['K']}, fg={errs.shape[0]} px, {time.time()-t0:.1f}s)")

            # Save predicted normal as visualization
            if args.save_predictions:
                vis = ((N_pred + 1.0) * 127.5).astype(np.uint8)
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                vis[~data["valid"]] = 0
                out_path = os.path.join(
                    os.path.expanduser(args.save_predictions),
                    f"{scene_name}_pred.png",
                )
                cv2.imwrite(out_path, vis)
        except Exception as e:
            print(f"  {scene_name:<15s}  FAILED: {e}")
            results.append((scene_name, float("nan"), 0))

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Scene':<15s}  {'MAE (deg)':>10s}")
    print("-" * 60)
    valid_maes = []
    for name, mae, _ in results:
        marker = "" if not np.isnan(mae) else " (fail)"
        print(f"{name:<15s}  {mae:>10.2f}{marker}")
        if not np.isnan(mae):
            valid_maes.append(mae)
    print("-" * 60)
    if valid_maes:
        print(f"{'Mean':<15s}  {np.mean(valid_maes):>10.2f}")
        print(f"{'Median':<15s}  {np.median(valid_maes):>10.2f}")
        print(f"{'Std':<15s}  {np.std(valid_maes):>10.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
