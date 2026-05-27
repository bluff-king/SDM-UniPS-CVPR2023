"""
Download a trained SDM-LVC checkpoint from HuggingFace + run evaluation.

Two modes:
  --mode val          : evaluate MAE on HDLong val split (needs GT)
  --mode predict      : predict normal map for 1 folder of input images (no GT)

Usage:
    # Download + eval on HDLong val (50 obj split)
    python scripts/download_and_eval.py \
        --hf_repo HUST-CVLab-PS/sdm-lvc-checkpoints \
        --hf_path lvc_full/ckpts_smoke/last.ckpt \
        --variant lvc_full \
        --mode val \
        --data_root $DATA_ROOT \
        --val_size 50

    # Predict normal from a folder of input EXR/PNG (no GT needed)
    python scripts/download_and_eval.py \
        --ckpt_local ~/ckpts/lvc_full/last.ckpt \
        --variant lvc_full \
        --mode predict \
        --input_dir ~/data/some_object/cam_00001 \
        --output_dir ~/predictions/some_object
"""

from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.net_train import SDM_UniPSModule, angular_mae
from src.models.net_lvc import (
    SDM_LVC_Loss_Module,
    SDM_LVC_Feat_Module,
    SDM_LVC_Full_Module,
)

VARIANT_MAP = {
    "baseline": SDM_UniPSModule,
    "lvc_loss": SDM_LVC_Loss_Module,
    "lvc_feat": SDM_LVC_Feat_Module,
    "lvc_full": SDM_LVC_Full_Module,
}


# ============================================================================
# Download checkpoint from HF
# ============================================================================

def download_ckpt(hf_repo: str, hf_path: str, local_dir: str) -> str:
    """Download a single .ckpt file from HF model repo. Returns local path."""
    from huggingface_hub import hf_hub_download
    print(f"[download] {hf_repo}:{hf_path}  ->  {local_dir}")
    local_path = hf_hub_download(
        repo_id=hf_repo,
        filename=hf_path,
        repo_type="model",
        local_dir=local_dir,
    )
    print(f"[download] saved {local_path}  ({os.path.getsize(local_path)/1e6:.1f} MB)")
    return local_path


# ============================================================================
# Load Lightning checkpoint -> Module
# ============================================================================

def load_module(ckpt_path: str, variant: str, device: torch.device):
    """Load Lightning ckpt into the correct variant module."""
    print(f"[load] ckpt = {ckpt_path}")
    print(f"[load] variant = {variant}")
    ModuleCls = VARIANT_MAP[variant]
    # Lightning's load_from_checkpoint reconstructs the module + loads state
    module = ModuleCls.load_from_checkpoint(ckpt_path, map_location=device, strict=False)
    module.eval()
    module.to(device)
    return module


# ============================================================================
# Mode: VAL — evaluate on HDLong val split
# ============================================================================

def eval_on_val(module, args):
    from src.data.data_hdlong import HDLongDataset
    from torch.utils.data import DataLoader

    print(f"\n[eval] building HDLong val split (val_size={args.val_size})")
    ds = HDLongDataset(
        mode="Val",
        data_root=args.data_root,
        numImages=args.num_input_images,
        image_size=args.image_size,
        val_size=args.val_size,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    device = next(module.parameters()).device
    all_mae = []
    print(f"[eval] running on {len(ds)} val samples...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            d = module._step(batch)
            mae_val = d["mae"].item()
            all_mae.append(mae_val)
            if (i + 1) % 10 == 0 or i == len(ds) - 1:
                print(f"  [{i+1}/{len(ds)}] mae={mae_val:.2f}deg  running avg={np.mean(all_mae):.2f}deg")

    print(f"\n[eval] ====================")
    print(f"[eval] Mean MAE = {np.mean(all_mae):.3f} deg")
    print(f"[eval] Median   = {np.median(all_mae):.3f} deg")
    print(f"[eval] Std      = {np.std(all_mae):.3f}")
    print(f"[eval] Min/Max  = {np.min(all_mae):.2f} / {np.max(all_mae):.2f}")
    print(f"[eval] ====================")
    return all_mae


# ============================================================================
# Mode: PREDICT — predict normal for a single object folder
# ============================================================================

def predict_one(module, args):
    """Predict normal map from one folder of input images (no GT)."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    print(f"\n[predict] input_dir  = {args.input_dir}")
    print(f"[predict] output_dir = {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect input images (EXR or PNG)
    paths = sorted(
        glob.glob(os.path.join(args.input_dir, "*.exr"))
        + glob.glob(os.path.join(args.input_dir, "*.png"))
        + glob.glob(os.path.join(args.input_dir, "*.jpg"))
    )
    # Skip non-light files (mask, normal, depth)
    paths = [p for p in paths if not any(
        kw in os.path.basename(p).lower()
        for kw in ("mask", "normal", "depth", "albedo", "roughness", "metallic")
    )]
    if not paths:
        raise RuntimeError(f"No input images found in {args.input_dir}")
    if len(paths) > args.num_input_images:
        # Random subset
        np.random.seed(42)
        idx = np.random.choice(len(paths), args.num_input_images, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    print(f"[predict] using {len(paths)} input images:")
    for p in paths:
        print(f"  {os.path.basename(p)}")

    # Read mask if exists
    mask_paths = glob.glob(os.path.join(args.input_dir, "*mask*.exr")) + \
                 glob.glob(os.path.join(args.input_dir, "*mask*.png"))
    if mask_paths:
        m = cv2.imread(mask_paths[0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if m.ndim == 3:
            m = m[..., 0]
        M = (m > 0.5).astype(np.float32)
        print(f"[predict] using mask {mask_paths[0]}")
    else:
        # No mask = use entire image as foreground
        M = None
        print(f"[predict] no mask found, treating full image as foreground")

    # Read and stack inputs
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"failed to read {p}")
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (args.image_size, args.image_size))
        imgs.append(img)
    imgs = np.stack(imgs, axis=-1)  # [H, W, 3, K]
    H, W = imgs.shape[:2]

    if M is None:
        M = np.ones((H, W), dtype=np.float32)
    else:
        M = cv2.resize(M, (W, H), interpolation=cv2.INTER_NEAREST)

    # Mask inputs
    imgs = imgs * M[..., None, None]

    device = next(module.parameters()).device
    I = torch.from_numpy(imgs).permute(2, 0, 1, 3).unsqueeze(0).to(device).float()   # [1, 3, H, W, K]
    M_t = torch.from_numpy(M).unsqueeze(0).unsqueeze(0).to(device).float()           # [1, 1, H, W]
    # Build fake N_gt (zeros — only used for shape; doesn't affect prediction)
    N_gt = torch.zeros(1, 3, H, W, device=device)
    nImg = torch.tensor([[len(paths)]], dtype=torch.int32)

    print(f"[predict] running inference...")
    with torch.no_grad():
        # Use full image grid as "sample" — re-implement decode loop without random sample
        out = module.net.forward_train(
            I=I, M=M_t, N_gt=N_gt, nImgArray=nImg,
            decoder_resolution=H,
            canonical_resolution=args.canonical_resolution,
        )
    # out["n_pred"] is [1, S, 3] where S = pixel_samples — but we want full image
    # → workaround: temporarily set pixel_samples = H*W, but that's huge.
    # Better: directly call full-image forward path
    print(f"[predict] WARNING: full-image prediction not implemented yet.")
    print(f"[predict] Use --mode val on a val sample, or implement full-image forward in net_train.py")
    print(f"[predict] Sampled prediction at {out['n_pred'].shape[1]} pixels saved as preview.")

    # Save the sampled prediction as a visualization
    n_pred = out["n_pred"][0].cpu().numpy()    # [S, 3]
    preview = ((n_pred + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    out_path = os.path.join(args.output_dir, "predicted_normals_sampled.npy")
    np.save(out_path, n_pred)
    print(f"[predict] saved {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    # ckpt source
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="e.g. HUST-CVLab-PS/sdm-lvc-checkpoints")
    parser.add_argument("--hf_path", type=str, default=None,
                        help="path inside repo, e.g. lvc_full/ckpts_smoke/last.ckpt")
    parser.add_argument("--ckpt_local", type=str, default=None,
                        help="if set, skip HF download and use this local path")
    parser.add_argument("--download_dir", type=str, default="~/hf_ckpts")

    # Variant must match training
    parser.add_argument("--variant", required=True,
                        choices=list(VARIANT_MAP.keys()))

    # Mode
    parser.add_argument("--mode", required=True, choices=["val", "predict"])

    # Val mode params
    parser.add_argument("--data_root", type=str, default=None,
                        help="HDLong DATA_ROOT (for --mode val)")
    parser.add_argument("--val_size", type=int, default=50)

    # Predict mode params
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./predictions")

    # Common
    parser.add_argument("--num_input_images", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--canonical_resolution", type=int, default=128)

    args = parser.parse_args()

    # Resolve ckpt path
    if args.ckpt_local:
        ckpt_path = os.path.expanduser(args.ckpt_local)
    else:
        if not (args.hf_repo and args.hf_path):
            raise ValueError("Either --ckpt_local OR (--hf_repo + --hf_path) required")
        ckpt_path = download_ckpt(
            args.hf_repo, args.hf_path,
            os.path.expanduser(args.download_dir),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device = {device}")
    module = load_module(ckpt_path, args.variant, device)

    if args.mode == "val":
        if not args.data_root:
            raise ValueError("--mode val requires --data_root")
        eval_on_val(module, args)
    else:  # predict
        if not args.input_dir:
            raise ValueError("--mode predict requires --input_dir")
        predict_one(module, args)


if __name__ == "__main__":
    main()
