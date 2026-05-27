"""
HDLong synthetic photometric stereo dataset — SDM-UniPS port.

Adapted from LINO_UniPS_LVCmodule/src/data/data_hdlong.py.
Differences:
  * No light GT outputs (env_light / point_lights / area_light) — SDM-UniPS
    does not have Light Alignment Loss; only the normal regression loss is
    used, so light metadata is irrelevant.
  * `light_means.config` still used for intensity normalisation (input mix).
  * Output dict keys match what SDM training expects:
        img  : [3, H, W, K]
        nml  : [3, H, W, 1]
        mask : [1, H, W, 1]
        numberOfImages : [1] (int32)

Mixing protocol (unchanged from thesis advisor's spec):
    For each of K input slots:
        hdl_type ∈ {hdl1=point+env, hdl2=dir+env}
        alpha    ~ Uniform(alpha_lo, alpha_hi)
        I = alpha * (I_pri / pri_mean) + (1 - alpha) * (I_env / env_mean)
"""

import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# OpenCV needs this flag to decode EXR.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


# ----------------------------------------------------------------------------
# EXR helpers
# ----------------------------------------------------------------------------

def _read_exr_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cv2 failed to read EXR: {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img, dtype=np.float32)


def _read_exr_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if m is None:
        raise IOError(f"cv2 failed to read mask EXR: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0.5).astype(np.float32)


def _parse_kv_config(path: str) -> dict:
    out = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return out


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class HDLongDataset(Dataset):
    """Synthetic mixed-lighting PS dataset (point/dir + env) for SDM-UniPS."""

    def __init__(
        self,
        mode: str,
        data_root: str,
        numImages: int = 6,
        image_size: int = 256,
        hdl_types=("hdl1", "hdl2"),
        alpha_range=(0.1, 0.9),
        val_size: int = 200,
        num_train_per_epoch: int = None,
        ratio_train: float = 0.9,
        ratio_val: float = 0.05,
        repeat: int = 1,
        seed: int = 42,
        k_min: int = None,   # if provided -> random K per sample in [k_min, numImages]
        **kwargs,
    ):
        self.mode = mode
        self.data_root = data_root
        self.K = int(numImages)
        self.k_min = int(k_min) if k_min is not None else None
        self.S = int(image_size)
        self.hdl_types = tuple(hdl_types)
        self.alpha_lo, self.alpha_hi = alpha_range
        self.repeat = max(1, int(repeat))
        self.num_train_per_epoch = num_train_per_epoch

        # ---- scan objects ----
        print(f"[HDLongDataset/{mode}] scanning {data_root} ...")
        all_objs = []
        n_no_cam = 0
        for d in tqdm(sorted(os.listdir(data_root)), desc="scanning objects"):
            if d.startswith("."):
                continue
            p = os.path.join(data_root, d)
            if not os.path.isdir(p):
                continue
            try:
                has_cam = any(
                    sub.startswith("cam_") and os.path.isdir(os.path.join(p, sub))
                    for sub in os.listdir(p)
                )
            except Exception:
                has_cam = False
            if not has_cam:
                n_no_cam += 1
                continue
            # Need light_means.config for intensity normalisation
            if not os.path.isfile(os.path.join(p, "light_means.config")):
                n_no_cam += 1
                continue
            all_objs.append(p)
        if n_no_cam:
            print(f"[HDLongDataset/{mode}] WARNING: skipped {n_no_cam} dir without cam_* or light_means.config")
        n = len(all_objs)
        if n == 0:
            raise RuntimeError(f"No valid object folders under {data_root}")

        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        all_objs = [all_objs[i] for i in idx]

        # ---- split: val (fixed) + train_pool ----
        if n >= val_size + 1:
            val_objs = all_objs[:val_size]
            train_pool = all_objs[val_size:]
        else:
            n_va = max(1, int(n * ratio_val))
            n_tr = max(1, n - n_va)
            val_objs = all_objs[n_tr:n_tr + n_va] or all_objs[:1]
            train_pool = all_objs[:n_tr]
            print(f"[HDLongDataset/{mode}] WARNING: dataset has only {n} objects "
                  f"< val_size+1={val_size+1}, using ratio fallback")

        # ---- mode selection ----
        if mode == "Train":
            self.objs = train_pool
            if self.num_train_per_epoch is None:
                self.num_train_per_epoch = len(train_pool)
            self.random_subsample = True
            virtual_len = self.num_train_per_epoch * self.repeat
        elif mode in ("Val", "Validation"):
            self.objs = val_objs
            self.random_subsample = False
            virtual_len = len(val_objs)
        else:  # Test
            self.objs = val_objs
            self.random_subsample = False
            virtual_len = len(val_objs)

        self._virtual_len = int(virtual_len)
        print(f"[HDLongDataset/{mode}] pool={len(self.objs)} obj, "
              f"epoch_len={self._virtual_len}, random_subsample={self.random_subsample}")

    def __len__(self):
        return self._virtual_len

    def _resize(self, arr: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
        if arr.shape[0] == self.S and arr.shape[1] == self.S:
            return arr
        return cv2.resize(arr, (self.S, self.S), interpolation=interp)

    def __getitem__(self, index):
        if self.random_subsample:
            obj_idx = random.randrange(len(self.objs))
        else:
            obj_idx = index % len(self.objs)
        last_err = None
        for _ in range(5):
            try:
                return self._get_one(obj_idx)
            except Exception as e:
                last_err = e
                obj_idx = (obj_idx + 1) % len(self.objs)
        raise RuntimeError(f"HDLongDataset: 5 consecutive failures; last: {last_err}")

    def _get_one(self, obj_idx: int) -> dict:
        obj_path = self.objs[obj_idx]
        means = _parse_kv_config(os.path.join(obj_path, "light_means.config"))
        dir_mean = float(means.get("dir_mean", 1.0)) or 1.0
        pt_mean = float(means.get("point_mean", 1.0)) or 1.0
        env_mean = float(means.get("env_mean", 1.0)) or 1.0

        cams = sorted([d for d in os.listdir(obj_path)
                       if d.startswith("cam_") and os.path.isdir(os.path.join(obj_path, d))])
        if not cams:
            raise RuntimeError(f"No cam_* folders in {obj_path}")
        cam = random.choice(cams)
        cam_path = os.path.join(obj_path, cam)

        # ---- GT normal + mask ----
        # local_normal.exr stored as (n+1)/2 -> decode to [-1, 1]
        N_raw = _read_exr_rgb(os.path.join(cam_path, "local_normal.exr"))[..., :3]
        N = N_raw * 2.0 - 1.0
        M = _read_exr_mask(os.path.join(cam_path, "binary_mask.exr"))
        N = self._resize(N, cv2.INTER_NEAREST)
        M = self._resize(M, cv2.INTER_NEAREST)
        M = M[..., None]    # [H, W, 1]
        N = N * M

        # ---- candidate light images ----
        env_paths = sorted(glob.glob(os.path.join(cam_path, "env_light_*.exr")))
        dir_paths = sorted(glob.glob(os.path.join(cam_path, "dir_light_*.exr")))
        pt_paths = sorted(glob.glob(os.path.join(cam_path, "point_light_*.exr")))
        if not env_paths or (not dir_paths and not pt_paths):
            raise RuntimeError(f"Missing light EXRs in {cam_path}")

        # ---- random K per sample (if k_min set) ----
        if self.k_min is not None:
            K = random.randint(self.k_min, self.K)
        else:
            K = self.K

        # ---- assemble K mixed inputs ----
        imgs = np.zeros((self.S, self.S, 3, K), dtype=np.float32)
        for i in range(K):
            avail = [t for t in self.hdl_types
                     if (t == "hdl1" and pt_paths) or (t == "hdl2" and dir_paths)]
            hdl = random.choice(avail) if avail else "hdl2"
            alpha = random.uniform(self.alpha_lo, self.alpha_hi)

            I_env = _read_exr_rgb(random.choice(env_paths)) / (env_mean + 1e-8)

            if hdl == "hdl1":
                I_pri = _read_exr_rgb(random.choice(pt_paths)) / (pt_mean + 1e-8)
            else:
                I_pri = _read_exr_rgb(random.choice(dir_paths)) / (dir_mean + 1e-8)

            I_mix = alpha * I_pri + (1.0 - alpha) * I_env
            I_mix = self._resize(I_mix, cv2.INTER_LINEAR)
            I_mix = np.nan_to_num(I_mix, nan=0.0, posinf=0.0, neginf=0.0)
            imgs[..., i] = I_mix

        imgs = imgs * M[..., None]   # mask inputs

        # ---- tensors (SDM-compatible layout) ----
        img_t = torch.from_numpy(imgs).permute(2, 0, 1, 3).contiguous()   # [3, H, W, K]
        nml_t = torch.from_numpy(N).permute(2, 0, 1).unsqueeze(-1).contiguous()  # [3, H, W, 1]
        mask_t = torch.from_numpy(M).permute(2, 0, 1).unsqueeze(-1).contiguous() # [1, H, W, 1]

        return {
            "img": img_t,
            "nml": nml_t,
            "mask": mask_t,
            "numberOfImages": torch.tensor([K], dtype=torch.int32),
        }
