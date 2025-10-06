import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from pyproj import Proj

def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    proj_enu = Proj(proj='aeqd', lat_0=lat0, lon_0=lon0, datum='WGS84')
    x, y = proj_enu(lon, lat)
    z = alt - alt0
    return x, y, z

def degrade_gnss(xyz, noise_std=1.0, bias_rw_std=0.05, dropout_p=0.0, rng=None):
    if rng is None:
        rng = np.random
    T = xyz.shape[0]
    noisy = xyz.astype(np.float32).copy()

    # Add zero-mean Gaussian noise
    if noise_std > 0:
        noisy += rng.normal(scale=noise_std, size=noisy.shape)

    # Add bias random-walk drift
    if bias_rw_std > 0:
        bias = np.cumsum(rng.normal(scale=bias_rw_std, size=(T, 3)), axis=0)
        noisy += bias

    # Apply dropout (simulate GNSS outage)
    mask = (rng.random(T) > dropout_p).astype(np.float32).reshape(T, 1)
    noisy *= mask
    return noisy, mask

class MultiKITTIOxtsDataset(Dataset):
    """
    Returns (imu_in, gnss_in, target_gt, scale_m)
      imu_in:     [L,6] normalized accelerations + angular rates
      gnss_in:    [L,3] degraded & normalized GNSS positions
      target_gt:  [L,3] clean & normalized GNSS positions
      scale_m:    scalar (per window, in meters)
    """
    def __init__(self,
                 root_dir,
                 seq_len=100,
                 stride=10,
                 split="train",
                 normalize=True,
                 gnss_noise_std=10.0,
                 gnss_bias_rw_std=0.5,
                 gnss_dropout_p=0.05,
                 return_mask=False,
                 verbose=False,
                 seed=42):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.normalize = normalize
        self.return_mask = return_mask

        self.gnss_noise_std = gnss_noise_std
        self.gnss_bias_rw_std = gnss_bias_rw_std
        self.gnss_dropout_p = gnss_dropout_p

        rng = np.random.default_rng(seed)
        all_segments = []

        all_drives = sorted(glob(os.path.join(root_dir, "*", "*_sync")))
        if verbose:
            print(f"Found {len(all_drives)} drives under {root_dir}")

        for drive_path in all_drives:
            oxts_dir = os.path.join(drive_path, "oxts", "data")
            files = sorted(glob(os.path.join(oxts_dir, "*.txt")))
            if not files:
                continue

            data = np.stack([np.loadtxt(f) for f in files])
            lat, lon, alt = data[:, 0], data[:, 1], data[:, 2]
            ax, ay, az = data[:, 11], data[:, 12], data[:, 13]
            wx, wy, wz = data[:, 14], data[:, 15], data[:, 16]

            x, y, z = latlon_to_enu(lat, lon, alt, lat[0], lon[0], alt[0])
            gnss_m = np.stack([x, y, z], axis=1).astype(np.float32)

            imu = np.stack([ax, ay, az, wx, wy, wz], axis=1).astype(np.float32)
            if normalize:
                imu = (imu - imu.mean(0)) / (imu.std(0) + 1e-8)

            N = len(imu)
            for start in range(0, N - seq_len + 1, stride):
                imu_win = imu[start:start + seq_len]
                gnss_win = gnss_m[start:start + seq_len]
                all_segments.append((imu_win, gnss_win))

        total = len(all_segments)
        indices = np.arange(total)
        rng.shuffle(indices)
        split_idx = int(0.8 * total)

        if split == "train":
            sel_idx = indices[:split_idx]
        else:
            sel_idx = indices[split_idx:]

        self.imu_segments = [all_segments[i][0] for i in sel_idx]
        self.gnss_segments = [all_segments[i][1] for i in sel_idx]

        if verbose:
            print(f"[{split.upper()}] using {len(self.imu_segments)} / {total} windows "
                  f"({len(self.imu_segments) / total * 100:.1f}%)")

    def __len__(self):
        return len(self.imu_segments)

    def __getitem__(self, idx):
        imu_in = self.imu_segments[idx]
        gnss_m = self.gnss_segments[idx]

        gt_rel_m = gnss_m - gnss_m[0]

        gnss_noisy_m, mask = degrade_gnss(
            gt_rel_m,
            noise_std=self.gnss_noise_std,
            bias_rw_std=self.gnss_bias_rw_std,
            dropout_p=self.gnss_dropout_p,
        )
        
        if self.normalize:
            scale_m = max(np.max(np.abs(gt_rel_m)), 1.0)
            gt_out = gt_rel_m / scale_m
            gnss_in = gnss_noisy_m / scale_m
        else:
            scale_m = 1.0
            gt_out = gt_rel_m
            gnss_in = gnss_noisy_m

        imu_in = torch.tensor(imu_in, dtype=torch.float32)
        gnss_in = torch.tensor(gnss_in, dtype=torch.float32)
        target = torch.tensor(gt_out, dtype=torch.float32)
        scale_t = torch.tensor(scale_m, dtype=torch.float32)

        if self.return_mask:
            mask_t = torch.tensor(mask, dtype=torch.float32)
            return imu_in, gnss_in, target, scale_t, mask_t
        else:
            return imu_in, gnss_in, target, scale_t
