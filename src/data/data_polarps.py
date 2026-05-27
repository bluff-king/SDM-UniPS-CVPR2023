"""
PolarPS dataset loader — SDM-UniPS port.

Adapted from LINO_UniPS_LVCmodule/src/data/data_polarps.py.
PolarPS provides:
  * normal.exr at view level
  * light-XX/S0.exr (direction-light renderings, 32 or 64 per material)
  * No env / point / area / HDRI info — no Light Alignment data
PolarPS is therefore a "pure normal supervision" source, complementing HDLong
(which contributes lighting diversity via point/dir + env mixing).

Per-material intensity normalisation is applied because raw S0.exr values
are ~100-1000x smaller than HDLong's mean-normalised inputs. We divide each
light by the material's mean intensity (sampled at scan time on a few lights
to avoid full preload).
"""

import os
import glob
import random
from typing import List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


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


def _decode_normal_and_mask(normal_path: str, mask_eps: float = 1e-3):
    raw = _read_exr_rgb(normal_path)
    M = (np.abs(raw - 0.5).max(axis=-1) > mask_eps).astype(np.float32)
    N = raw * 2.0 - 1.0
    N = N * M[..., None]
    return N, M


Sample = Tuple[str, str, Tuple[str, ...], float]   # (normal, material_dir, light_dirs, mat_mean)


def _scan_polarps(data_root: str, sample_for_mean: int = 3) -> List[Sample]:
    """Recursively find every (normal, material, lights) triple."""
    samples: List[Sample] = []
    normals = glob.glob(os.path.join(data_root, "**", "normal.exr"), recursive=True)
    normals.sort()

    n_no_mat = 0
    n_no_light = 0
    for npath in tqdm(normals, desc="scanning PolarPS"):
        view_dir = os.path.dirname(npath)
        try:
            siblings = os.listdir(view_dir)
        except OSError:
            continue
        mat_dirs = []
        for s in siblings:
            sp = os.path.join(view_dir, s)
            if os.path.isdir(sp):
                mat_dirs.append(sp)
        if not mat_dirs:
            n_no_mat += 1
            continue

        for mdir in mat_dirs:
            try:
                lights = sorted(
                    os.path.join(mdir, l)
                    for l in os.listdir(mdir)
                    if l.startswith("light-")
                    and os.path.isdir(os.path.join(mdir, l))
                    and os.path.exists(os.path.join(mdir, l, "S0.exr"))
                )
            except OSError:
                lights = []
            if not lights:
                n_no_light += 1
                continue

            # Estimate per-material mean intensity (subsample to avoid full preload)
            sample_lights = random.sample(lights, min(sample_for_mean, len(lights)))
            means = []
            try:
                for lp in sample_lights:
                    img = _read_exr_rgb(os.path.join(lp, "S0.exr"))
                    means.append(float(img.mean()))
                mat_mean = float(np.mean(means)) + 1e-8
            except Exception:
                mat_mean = 1.0   # fallback

            samples.append((npath, mdir, tuple(lights), mat_mean))

    if n_no_mat:
        print(f"[PolarPS] WARNING: {n_no_mat} views have no material subfolders")
    if n_no_light:
        print(f"[PolarPS] WARNING: {n_no_light} materials have no valid light-XX/S0.exr")
    return samples


class PolarPSDataset(Dataset):
    """PolarPS direction-light photometric stereo dataset (SDM-compatible)."""

    def __init__(
        self,
        mode: str,
        data_root: str,
        numImages: int = 6,
        image_size: int = 256,
        val_size: int = 50,
        num_train_per_epoch: int = None,
        ratio_train: float = 0.9,
        ratio_val: float = 0.05,
        seed: int = 42,
        mask_eps: float = 1e-3,
        k_min: int = None,
        intensity_norm: bool = True,
        **kwargs,
    ):
        self.mode = mode
        self.data_root = data_root
        self.K = int(numImages)
        self.k_min = int(k_min) if k_min is not None else None
        self.S = int(image_size)
        self.num_train_per_epoch = num_train_per_epoch
        self.mask_eps = float(mask_eps)
        self.intensity_norm = bool(intensity_norm)

        print(f"[PolarPSDataset/{mode}] scanning {data_root} ...")
        all_samples = _scan_polarps(data_root)
        n = len(all_samples)
        if n == 0:
            raise RuntimeError(f"No PolarPS samples found under {data_root}")

        def _scene_of(sample):
            rel = os.path.relpath(sample[0], data_root)
            return rel.split(os.sep)[0]

        scenes = sorted({_scene_of(s) for s in all_samples})
        rng = np.random.RandomState(seed)
        scene_idx = np.arange(len(scenes))
        rng.shuffle(scene_idx)
        scenes_shuffled = [scenes[i] for i in scene_idx]

        if len(scenes) >= val_size + 1:
            val_scenes_set = set(scenes_shuffled[:val_size])
            train_scenes_set = set(scenes_shuffled[val_size:])
        else:
            n_va = max(1, int(len(scenes) * ratio_val))
            n_tr = max(1, len(scenes) - n_va)
            val_scenes_set = set(scenes_shuffled[n_tr:n_tr + n_va] or scenes_shuffled[:1])
            train_scenes_set = set(scenes_shuffled[:n_tr])

        val_samples = [s for s in all_samples if _scene_of(s) in val_scenes_set]
        train_pool = [s for s in all_samples if _scene_of(s) in train_scenes_set]
        print(f"[PolarPSDataset/{mode}] split by SCENE: "
              f"{len(train_scenes_set)} train / {len(val_scenes_set)} val scenes "
              f"-> {len(train_pool)} train / {len(val_samples)} val samples")

        if mode.lower() == "train":
            self.full_pool = train_pool
            virtual_len = num_train_per_epoch if num_train_per_epoch is not None else len(train_pool)
            self.random_subsample = True
        elif mode.lower() == "val":
            self.full_pool = val_samples
            virtual_len = len(val_samples)
            self.random_subsample = False
        else:
            self.full_pool = train_pool
            virtual_len = len(train_pool)
            self.random_subsample = False
        self.virtual_len = int(virtual_len)
        print(f"[PolarPSDataset/{mode}] total={n}, epoch_len={self.virtual_len}")

    def __len__(self):
        return self.virtual_len

    def _resize(self, img: np.ndarray) -> np.ndarray:
        if img.shape[0] == self.S and img.shape[1] == self.S:
            return img
        interp = cv2.INTER_AREA if img.shape[0] > self.S else cv2.INTER_LINEAR
        out = cv2.resize(img, (self.S, self.S), interpolation=interp)
        if out.ndim == 2:
            out = out[..., None]
        return out

    def _load_one(self, sample: Sample) -> dict:
        normal_path, mat_dir, lights, mat_mean = sample

        N, M = _decode_normal_and_mask(normal_path, mask_eps=self.mask_eps)
        N = self._resize(N)
        M = self._resize(M.astype(np.float32))
        if M.ndim == 3:
            M = M[..., 0]
        M = (M > 0.5).astype(np.float32)
        N = N * M[..., None]

        # K random per sample (if k_min set)
        if self.k_min is not None:
            K = random.randint(self.k_min, self.K)
        else:
            K = self.K

        n_avail = len(lights)
        if n_avail >= K:
            picks = random.sample(range(n_avail), K)
        else:
            picks = [random.randrange(n_avail) for _ in range(K)]
        light_paths = [os.path.join(lights[i], "S0.exr") for i in picks]

        imgs = np.zeros((self.S, self.S, 3, K), dtype=np.float32)
        scale = (1.0 / mat_mean) if self.intensity_norm else 1.0
        for k, lp in enumerate(light_paths):
            I = _read_exr_rgb(lp) * scale
            I = self._resize(I)
            I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
            imgs[..., k] = I

        imgs = imgs * M[..., None, None]

        img_t = torch.from_numpy(imgs).permute(2, 0, 1, 3).contiguous()       # [3,H,W,K]
        nml_t = torch.from_numpy(N).permute(2, 0, 1).unsqueeze(-1).contiguous()  # [3,H,W,1]
        mask_t = torch.from_numpy(M).unsqueeze(0).unsqueeze(-1).contiguous()    # [1,H,W,1]

        return {
            "img": img_t,
            "nml": nml_t,
            "mask": mask_t,
            "numberOfImages": torch.tensor([K], dtype=torch.int32),
        }

    def __getitem__(self, idx: int) -> dict:
        if self.random_subsample:
            sample = random.choice(self.full_pool)
        else:
            sample = self.full_pool[idx % len(self.full_pool)]

        last_err = None
        for attempt in range(5):
            try:
                return self._load_one(sample)
            except Exception as e:
                last_err = e
                if self.random_subsample:
                    sample = random.choice(self.full_pool)
                else:
                    sample = self.full_pool[(idx + attempt + 1) % len(self.full_pool)]
        raise RuntimeError(f"PolarPSDataset: 5 consecutive failures; last: {last_err}")
