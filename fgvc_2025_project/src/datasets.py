import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T


@dataclass
class Sample:
    path: str
    label_name: Optional[str]


def read_annotations_csv(csv_path: Path) -> List[Sample]:
    rows: List[Sample] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        if not {"path", "label"}.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"annotations.csv must have columns path,label; got {reader.fieldnames}")
        for r in reader:
            p = r["path"].strip()
            lbl = r["label"].strip() if r["label"] is not None else None
            lbl = lbl if lbl not in ("", "None", "null") else None
            rows.append(Sample(path=p, label_name=lbl))
    return rows


def build_label_index(samples: List[Sample]) -> Tuple[Dict[str, int], List[str]]:
    names = sorted(list({s.label_name for s in samples if s.label_name is not None}))
    label2idx = {n: i for i, n in enumerate(names)}
    idx2label = names
    return label2idx, idx2label


class ROIDataset(Dataset):
    def __init__(self, csv_path: Path, label2idx: Optional[Dict[str, int]] = None, image_size: int = 224):
        self.samples = read_annotations_csv(csv_path)
        # if not provided, build from CSV (train/val)
        self.label2idx = label2idx if label2idx is not None else build_label_index(self.samples)[0]
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img)
        y_name = s.label_name
        if y_name is None:
            y = -1
        else:
            y = self.label2idx[y_name]
        return x, y, s.path, y_name


def stratified_split(samples: List[Sample], val_split: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    # Group by label
    by_label: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        if s.label_name is None:
            continue
        by_label.setdefault(s.label_name, []).append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for lbl, idxs in by_label.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_split))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())
    return train_idx, val_idx


def make_dataloaders(csv_path: Path, val_split: float, batch_size: int, num_workers: int, image_size: int = 224, seed: int = 42):
    all_samples = read_annotations_csv(csv_path)
    label2idx, idx2label = build_label_index(all_samples)

    ds = ROIDataset(csv_path, label2idx=label2idx, image_size=image_size)
    train_idx, val_idx = stratified_split(ds.samples, val_split=val_split, seed=seed)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, label2idx, idx2label


def make_loader_for_infer(csv_path: Path, label2idx: Dict[str, int], batch_size: int, num_workers: int, image_size: int = 224):
    ds = ROIDataset(csv_path, label2idx=label2idx, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    idx2label = [None] * len(label2idx)
    for name, idx in label2idx.items():
        idx2label[idx] = name
    return loader, idx2label
