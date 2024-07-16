# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob
from scipy.signal import resample, butter, filtfilt
from scipy.ndimage import gaussian_filter

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))


    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = np.load(X_path)

        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = np.load(subject_idx_path)

        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = np.load(y_path)

            # データ前処理
            X = self.preprocess(X)
            # データ拡張（トレーニング時のみ）
            if self.split == "train":
                X = self.augment_data(X)

            # デバッグプリント
            #print(f"Index: {i}, X shape: {X.shape}, y shape: {y.shape}, subject_idx shape: {subject_idx.shape}")

            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), torch.tensor(subject_idx, dtype=torch.long)
        else:
            # データ前処理
            X = self.preprocess(X)

            # デバッグプリント
            #print(f"Index: {i}, X shape: {X.shape}, subject_idx shape: {subject_idx.shape}")

            return torch.tensor(X, dtype=torch.float32), torch.tensor(subject_idx, dtype=torch.long)

    def preprocess(self, X):
        target_rate = 100  # 目標のサンプリングレート
        original_rate = 200  # 元のサンプリングレート
        X = resample(X, int(X.shape[1] * target_rate / original_rate))

        # ナイキスト周波数
        nyquist = 0.5 * target_rate
        low = 0.5 / nyquist
        high = 30 / nyquist

        # フィルタリング
        b, a = butter(4, [low, high], btype='band')
        X = filtfilt(b, a, X, axis=1)

        # スケーリング
        X = (X - X.mean()) / X.std()

        # ベースライン補正
        X = X - X[:, :50].mean(axis=1, keepdims=True)

        # パディングを加える
        if X.shape[1] < 281:
            pad_size = 281 - X.shape[1]
            X = np.pad(X, ((0, 0), (0, pad_size)), mode='constant')
        elif X.shape[1] > 281:
            X = X[:, :281]

        return X

    def augment_data(self, X):
        X = self.add_noise(X)
        X = self.shift_data(X)
        X = self.scale_data(X)
        X = self.smooth_data(X)
        return X

    def add_noise(self, data, noise_level=0.01):
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    def shift_data(self, data, shift_max=5):
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(data, shift, axis=1)

    def scale_data(self, data, scale_range=(0.8, 1.2)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale

    def smooth_data(self, data, sigma_range=(0.5, 1.5)):
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        return gaussian_filter(data, sigma=sigma)

    @property
    def num_channels(self) -> int:
        return 140

    @property
    def seq_len(self) -> int:
        return 281