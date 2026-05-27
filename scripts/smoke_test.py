"""
Smoke test for SDM-UniPS training pipeline.

Verifies (without doing real training):
  1. HDLong dataloader can load 1 sample from local data path.
  2. TrainableSDMNet builds + forward pass works on tiny input.
  3. Loss is finite and gradient flows.
  4. All 4 variants (baseline + 3 LVC) instantiate.

Run from repo root:
    python scripts/smoke_test.py --data_root D:/Ky8/hdlong-complexv1
"""

import argparse
import os
import sys

# Add repo root to path
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

from src.data.data_hdlong import HDLongDataset
from src.models.net_train import TrainableSDMNet, SDM_UniPSModule, normal_mse_loss, angular_mae
from src.models.net_lvc import (
    SDM_LVC_Loss_Module,
    SDM_LVC_Feat_Module,
    SDM_LVC_Full_Module,
)
from src.models.module.lvc import LVCModule


def test_dataloader(args):
    print("\n=== TEST 1: HDLong dataloader ===")
    ds = HDLongDataset(
        mode="Train",
        data_root=args.data_root,
        numImages=4,
        image_size=128,        # small for speed
        val_size=1,
        num_train_per_epoch=1,
    )
    sample = ds[0]
    print(f"  img.shape:   {sample['img'].shape}  expected [3, 128, 128, 4]")
    print(f"  nml.shape:   {sample['nml'].shape}  expected [3, 128, 128, 1]")
    print(f"  mask.shape:  {sample['mask'].shape}  expected [1, 128, 128, 1]")
    print(f"  nImg:        {sample['numberOfImages']}")
    print(f"  img range:   [{sample['img'].min():.3f}, {sample['img'].max():.3f}]")
    print(f"  nml range:   [{sample['nml'].min():.3f}, {sample['nml'].max():.3f}]")
    print(f"  mask fg %:   {sample['mask'].float().mean():.3f}")
    print("  OK")
    return sample


def test_lvc():
    print("\n=== TEST 2: LVC module ===")
    lvc = LVCModule(alpha=10.0, w_min=0.1)
    I = torch.randn(1, 3, 64, 64, 4).abs()
    R, R_k = lvc(I)
    print(f"  R.shape:    {R.shape}   expected [1, 64, 64]")
    print(f"  R_k.shape:  {R_k.shape} expected [1, 4, 64, 64]")
    print(f"  R range:    [{R.min():.3f}, {R.max():.3f}]   (should be in [0,1])")
    print(f"  R_k range:  [{R_k.min():.3f}, {R_k.max():.3f}]")
    w = lvc.loss_weights(R)
    print(f"  w range:    [{w.min():.3f}, {w.max():.3f}]   (should be in [0.1, 1])")
    print("  OK")


def test_model_forward(sample, args):
    print("\n=== TEST 3: TrainableSDMNet forward + backward ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # Build batch (B=1)
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    I = batch["img"]
    N_gt = batch["nml"][:, :, :, :, 0]
    M = batch["mask"][:, :, :, :, 0]
    nImg = batch["numberOfImages"].reshape(-1, 1).cpu()

    net = TrainableSDMNet(pixel_samples=256, device=device).to(device)
    net.train()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  #params: {n_params/1e6:.2f} M")

    out = net.forward_train(
        I=I, M=M, N_gt=N_gt, nImgArray=nImg,
        decoder_resolution=I.shape[2],
        canonical_resolution=128,
    )
    print(f"  n_pred.shape: {out['n_pred'].shape}  (B, S, 3)")
    print(f"  n_true.shape: {out['n_true'].shape}")
    print(f"  w_pix.shape:  {out['w_pix'].shape}")

    loss = normal_mse_loss(out["n_pred"], out["n_true"], out["w_pix"])
    mae = angular_mae(out["n_pred"], out["n_true"], torch.ones_like(out["w_pix"]))
    print(f"  loss: {loss.item():.4f}  (finite? {torch.isfinite(loss).item()})")
    print(f"  mae:  {mae.item():.2f} deg")

    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    print(f"  gradient flowed: {has_grad}")
    assert has_grad, "no gradient — backward broken"
    print("  OK")


def test_lvc_variants(sample, args):
    print("\n=== TEST 4: All variants instantiate + 1 forward ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

    for name, cls in [
        ("baseline", SDM_UniPSModule),
        ("lvc_loss", SDM_LVC_Loss_Module),
        ("lvc_feat", SDM_LVC_Feat_Module),
        ("lvc_full", SDM_LVC_Full_Module),
    ]:
        print(f"  -- {name}")
        mod = cls(pixel_samples=256, canonical_resolution=128).to(device)
        mod.train()
        # Lightning's _step needs `self.log` — bypass by calling internal _step
        try:
            d = mod._step(batch)
            print(f"     loss={d['loss'].item():.4f}  mae={d['mae'].item():.2f}deg  finite={torch.isfinite(d['loss']).item()}")
        except Exception as e:
            print(f"     FAILED: {e}")
            raise
    print("  OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="HDLong dataset root")
    args = parser.parse_args()

    sample = test_dataloader(args)
    test_lvc()
    test_model_forward(sample, args)
    test_lvc_variants(sample, args)

    print("\n[OK] All smoke tests passed.")


if __name__ == "__main__":
    main()
