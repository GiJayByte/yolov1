"""PASCAL VOC data pipeline.

Wraps ``torchvision.datasets.VOCDetection`` (which downloads and parses the XML
annotations for us), offers direct-resize or letterbox, applies the paper's
augmentation (random scale/translate <=20%, HSV exposure/saturation <=1.5x) on
the training split, and encodes every label into the 7x7x30 target tensor.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCDetection

from .config import CLASS_TO_IDX
from .utils import encode_targets

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _affine_sample(img: torch.Tensor, scale: float, tx: float, ty: float) -> torch.Tensor:
    """Scale about the centre and translate, in normalised coordinates."""
    _, H, W = img.shape
    ys = (torch.arange(H, dtype=img.dtype) + 0.5) / H
    xs = (torch.arange(W, dtype=img.dtype) + 0.5) / W
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    src_x = (grid_x - tx) / scale
    src_y = (grid_y - ty) / scale
    grid = torch.stack([src_x * 2 - 1, src_y * 2 - 1], dim=-1).unsqueeze(0)
    out = F.grid_sample(
        img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return out.squeeze(0)


def augment_sample(
    img: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, cfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Geometric + photometric augmentation for a single training sample."""
    if cfg.hflip and random.random() < 0.5:
        img = torch.flip(img, dims=[2])
        if boxes.numel():
            flipped = boxes.clone()
            flipped[:, 0] = 1.0 - boxes[:, 2]
            flipped[:, 2] = 1.0 - boxes[:, 0]
            boxes = flipped

    scale = 1.0 + random.uniform(-cfg.scale, cfg.scale)
    tx = random.uniform(-cfg.translate, cfg.translate)
    ty = random.uniform(-cfg.translate, cfg.translate)
    img = _affine_sample(img, scale, tx, ty)

    if boxes.numel():
        shifted = boxes.clone() * scale + torch.tensor([tx, ty, tx, ty], dtype=boxes.dtype)
        shifted[:, 0::2] = shifted[:, 0::2].clamp(0.0, 1.0)
        shifted[:, 1::2] = shifted[:, 1::2].clamp(0.0, 1.0)
        keep = (shifted[:, 2] - shifted[:, 0] > 1e-4) & (shifted[:, 3] - shifted[:, 1] > 1e-4)
        boxes = shifted[keep]
        labels = labels[keep]

    factor = cfg.hsv
    saturation = max(random.uniform(1.0 - factor, 1.0 + factor), 0.05)
    exposure = max(random.uniform(1.0 - factor, 1.0 + factor), 0.05)
    img = (img * exposure).clamp(0.0, 1.0)
    img = F.adjust_saturation(img, saturation).clamp(0.0, 1.0)
    return img, boxes, labels


def letterbox(
    img: torch.Tensor, boxes: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize preserving aspect ratio and pad to ``size`` (boxes stay normalised)."""
    _, h, w = img.shape
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = F.interpolate(
        img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    out = torch.zeros(3, size, size, dtype=img.dtype)
    out[:, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    if boxes.numel():
        shifted = boxes.clone()
        shifted[:, 0::2] = (shifted[:, 0::2] * w * scale + pad_x) / size
        shifted[:, 1::2] = (shifted[:, 1::2] * h * scale + pad_y) / size
        boxes = shifted
    return out, boxes


def _resize(img: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(
        img.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str,
        year: str = "2007",
        image_set: str = "trainval",
        extra_year: str | None = None,
        extra_image_set: str = "trainval",
        image_size: int = 448,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        letterbox_enabled: bool = False,
        train: bool = False,
        augment=None,
        return_meta: bool = False,
        normalize: bool = True,
        download: bool = False,
    ) -> None:
        self.image_size = image_size
        self.S, self.B, self.C = S, B, C
        self.letterbox_enabled = letterbox_enabled
        self.train = train
        self.augment = augment
        self.return_meta = return_meta
        self.normalize = normalize

        combos = [(year, image_set)]
        if extra_year:
            combos.append((extra_year, extra_image_set))
        self.sources = [
            VOCDetection(root=root, year=y, image_set=s, download=download) for y, s in combos
        ]
        self.index: list[tuple[int, int, str]] = []
        for si, ds in enumerate(self.sources):
            for ii in range(len(ds)):
                self.index.append((si, ii, combos[si][0]))

    def __len__(self) -> int:
        return len(self.index)

    def _parse(self, target: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ann = target["annotation"]
        objects = ann.get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        width = float(ann["size"]["width"])
        height = float(ann["size"]["height"])
        boxes, labels, difficult = [], [], []
        for obj in objects:
            name = obj["name"]
            if name not in CLASS_TO_IDX:
                continue
            bb = obj["bndbox"]
            x1 = max(float(bb["xmin"]), 0.0)
            y1 = max(float(bb["ymin"]), 0.0)
            x2 = min(float(bb["xmax"]), width)
            y2 = min(float(bb["ymax"]), height)
            if x2 - x1 <= 1e-3 or y2 - y1 <= 1e-3:
                continue
            boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
            labels.append(CLASS_TO_IDX[name])
            difficult.append(int(obj.get("difficult", 0)))

        box_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_t = torch.tensor(labels, dtype=torch.long)
        diff_t = torch.tensor(difficult, dtype=torch.long)
        return box_t, label_t, diff_t

    def __getitem__(self, idx: int):
        si, ii, _year = self.index[idx]
        img, target = self.sources[si][ii]
        boxes, labels, difficult = self._parse(target)
        filename = target["annotation"].get("filename", f"{si}_{ii}")

        tensor = F.to_tensor(img)  # (3, H, W) in [0, 1]
        if self.letterbox_enabled:
            tensor, boxes = letterbox(tensor, boxes, self.image_size)
        else:
            tensor = _resize(tensor, self.image_size)

        if self.train and self.augment is not None and self.augment.enabled:
            tensor, boxes, labels = augment_sample(tensor, boxes, labels, self.augment)

        target_t = encode_targets(boxes, labels, self.S, self.B, self.C)

        if self.normalize:
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

        if self.return_meta:
            meta = {
                "boxes": boxes,
                "labels": labels,
                "difficult": difficult,
                "filename": filename,
                "index": idx,
            }
            return tensor, target_t, meta
        return tensor, target_t


def denormalize(img: torch.Tensor) -> torch.Tensor:
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)


def build_dataloaders(cfg, download: bool = True):
    common = dict(
        root=cfg.data.root,
        image_size=cfg.data.image_size,
        S=cfg.model.S,
        B=cfg.model.B,
        C=cfg.model.C,
        letterbox_enabled=cfg.data.letterbox,
    )
    train_ds = VOCDataset(
        year=cfg.data.year,
        image_set=cfg.data.image_set,
        extra_year=cfg.data.extra_year,
        extra_image_set=cfg.data.extra_image_set,
        train=True,
        augment=cfg.augment,
        download=download,
        **common,
    )
    val_ds = VOCDataset(
        year=cfg.data.year,
        image_set="val",
        extra_year=None,
        train=False,
        download=False,
        **common,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
