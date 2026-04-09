"""
src/datasets/loader.py
----------------------
PyTorch Dataset classes for NIH ChestX-ray14, Shenzhen TB, and Montgomery TB.

NIH     → unlabeled (SSL pretraining) with two-view augmentation
Shenzhen → binary TB/Normal labels (few-shot fine-tuning)
Montgomery → binary TB/Normal labels (held-out test set)
"""

import os
import glob
from pathlib import Path
from typing import Optional, Callable, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ─── Standard transforms ─────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_base_transform(image_size: int = 224) -> T.Compose:
    """Standard chest X-ray augmentation for supervised/fine-tuning."""
    return T.Compose([
        T.Resize((image_size + 32, image_size + 32)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(),
        T.Grayscale(num_output_channels=3),   # X-rays are grayscale → 3-ch
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = 224) -> T.Compose:
    """Deterministic transform for evaluation (no random ops)."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TwoViewTransform:
    """
    Wraps any transform and applies it twice to the same image,
    returning (view1, view2) for SSL contrastive / MAE pre-training.
    """
    def __init__(self, base_transform: Callable):
        self.transform = base_transform

    def __call__(self, img) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(img), self.transform(img)


# ─── NIH ChestX-ray14 Dataset ────────────────────────────────────────────────

class NIHDataset(Dataset):
    """
    Loads chest X-ray images from the NIH ChestX-ray14 dataset.
    Labels are NOT loaded — used only for unlabeled SSL pretraining.

    Expected directory layout:
        root_dir/
            images/
                *.png  (or *.jpg)

    Args:
        root_dir   : Path to the NIH dataset root (e.g., data/raw/NIH/)
        transform  : Optional callable transform. If None, uses TwoViewTransform
                     with the base augmentation (for SSL two-view crops).
        image_size : Image resize target.
        two_view   : If True, returns (view1, view2) tuple instead of single tensor.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        two_view: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.two_view = two_view

        # Collect all image files
        self.image_paths: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            self.image_paths.extend(sorted(self.root_dir.glob(f"**/{ext}")))

        if len(self.image_paths) == 0:
            # Allow empty dataset (for dry-run / testing)
            print(f"[WARNING] NIHDataset: no images found in {root_dir}")

        if transform is not None:
            self.transform = transform
        elif two_view:
            self.transform = TwoViewTransform(get_base_transform(image_size))
        else:
            self.transform = get_base_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)


# ─── Shenzhen TB Dataset ─────────────────────────────────────────────────────

class ShenzhenDataset(Dataset):
    """
    Loads images + binary labels from the Shenzhen TB dataset.
    Label: 0 = Normal, 1 = TB (Tuberculosis-positive)

    Expected directory layout (two sub-folders):
        root_dir/
            TB/       ← TB positive images
            Normal/   ← Normal images

    If sub-folders don't exist, falls back to a flat directory
    and reads labels from filenames (CXR_*_1.png → TB, CXR_*_0.png → Normal).
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        split: str = "all",   # "all", "train", "val"
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or get_base_transform(image_size)
        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        tb_dir = self.root_dir / "TB"
        normal_dir = self.root_dir / "Normal"

        if tb_dir.exists() and normal_dir.exists():
            # Sub-folder layout
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(tb_dir.glob(f"**/{ext}")):
                    self.image_paths.append(p)
                    self.labels.append(1)
                for p in sorted(normal_dir.glob(f"**/{ext}")):
                    self.image_paths.append(p)
                    self.labels.append(0)
        else:
            # Flat layout: infer label from filename suffix (_1 = TB, _0 = Normal)
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(self.root_dir.glob(f"**/{ext}")):
                    name = p.stem
                    if name.endswith("_1") or "_1" in name[-3:]:
                        label = 1
                    elif name.endswith("_0") or "_0" in name[-3:]:
                        label = 0
                    else:
                        continue   # Skip unrecognized files
                    self.image_paths.append(p)
                    self.labels.append(label)

        if len(self.image_paths) == 0:
            print(f"[WARNING] ShenzhenDataset: no images found in {root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        return self.transform(img), label

    def get_labels(self) -> List[int]:
        """Return the full label list (useful for k-shot sampling)."""
        return self.labels


# ─── Montgomery TB Dataset ────────────────────────────────────────────────────

class MontgomeryDataset(Dataset):
    """
    Loads images + binary labels from the Montgomery TB dataset.
    Label: 0 = Normal, 1 = TB

    Expected directory layout:
        root_dir/
            TB/       ← TB positive
            Normal/   ← Normal

    Falls back to flat-dir filename inference (_1 / _0) if sub-folders absent.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or get_eval_transform(image_size)
        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        tb_dir = self.root_dir / "TB"
        normal_dir = self.root_dir / "Normal"

        if tb_dir.exists() and normal_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(tb_dir.glob(f"**/{ext}")):
                    self.image_paths.append(p)
                    self.labels.append(1)
                for p in sorted(normal_dir.glob(f"**/{ext}")):
                    self.image_paths.append(p)
                    self.labels.append(0)
        else:
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(self.root_dir.glob(f"**/{ext}")):
                    name = p.stem
                    if name.endswith("_1") or "_1" in name[-3:]:
                        label = 1
                    elif name.endswith("_0") or "_0" in name[-3:]:
                        label = 0
                    else:
                        continue
                    self.image_paths.append(p)
                    self.labels.append(label)

        if len(self.image_paths) == 0:
            print(f"[WARNING] MontgomeryDataset: no images found in {root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

    def get_labels(self) -> List[int]:
        return self.labels
