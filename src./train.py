"""Training entry point.

Examples
--------
Sanity gate (loss should fall toward ~0)::

    python -m src.train --config configs/voc.yaml --overfit-batch 4 --epochs 200

Real training (reduce batch size / epochs on CPU)::

    python -m src.train --config configs/voc.yaml --variant fast --epochs 30

Resume::

    python -m src.train --config configs/voc.yaml --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import build_lr_schedule, load_config
from .dataset import build_dataloaders
from .loss import build_loss
from .model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv1 on PASCAL VOC")
    p.add_argument("--config", default="configs/voc.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--device", default=None, help="auto | cpu | cuda")
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--variant", choices=["full", "fast"])
    p.add_argument("--schedule", choices=["paper", "reduced"])
    p.add_argument("--pretrained", action="store_true", help="use GoogLeNet conv trunk")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--overfit-batch", type=int, default=0, help="overfit the first N samples")
    p.add_argument("--limit-batches", type=int, default=0, help="cap batches per epoch (debug)")
    p.add_argument("--seed", type=int)
    return p.parse_args()


def apply_overrides(cfg, args) -> None:
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.device is not None:
        cfg.train.device = args.device
    if args.variant is not None:
        cfg.model.variant = args.variant
    if args.schedule is not None:
        cfg.train.schedule = args.schedule
    if args.pretrained:
        cfg.model.pretrained = True
    if args.no_amp:
        cfg.train.amp = False
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.resume is not None:
        cfg.train.resume = args.resume


def select_device(cfg) -> torch.device:
    if cfg.train.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.train.device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, criterion, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        total += float(criterion(outputs, targets)["loss"].item())
        count += 1
    return total / max(count, 1)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, limit=0):
    model.train()
    running = 0.0
    parts = {"coord": 0.0, "conf": 0.0, "class": 0.0}
    seen = 0
    amp = cfg.train.amp and device.type == "cuda"
    bar = tqdm(loader, desc="train", leave=False)
    for step, (images, targets) in enumerate(bar):
        if limit and step >= limit:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        if scaler is not None and amp:
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        running += float(loss.item())
        seen += 1
        for key in parts:
            parts[key] += float(loss_dict[key].item())
        bar.set_postfix(loss=f"{running / seen:.3f}")
    n = max(seen, 1)
    return running / n, {k: v / n for k, v in parts.items()}


def save_checkpoint(path: Path, model, optimizer, epoch, best_val, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "config": asdict(cfg),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)

    device = select_device(cfg)
    seed_everything(cfg.train.seed)
    print(f"[train] device={device} variant={cfg.model.variant} schedule={cfg.train.schedule}")

    train_loader, val_loader = build_dataloaders(cfg, download=True)

    if args.overfit_batch > 0:
        n = min(args.overfit_batch, len(train_loader.dataset))
        subset = Subset(train_loader.dataset, list(range(n)))
        train_loader = DataLoader(
            subset, batch_size=min(n, cfg.train.batch_size), shuffle=False, num_workers=0
        )
        print(f"[train] overfit mode on {n} samples -> loss should approach 0")

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    schedule = build_lr_schedule(cfg.train.schedule, cfg.train.lr)
    if args.overfit_batch > 0:
        schedule = lambda _epoch: cfg.train.lr  # noqa: E731

    start_epoch, best_val = 0, math.inf
    if cfg.train.resume:
        ckpt = torch.load(cfg.train.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", math.inf))
        print(f"[train] resumed from {cfg.train.resume} at epoch {start_epoch}")

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(cfg.train.log_dir)
    except Exception:  # noqa: BLE001
        writer = None
        print("[train] tensorboard not available, logging to stdout only")

    for epoch in range(start_epoch, cfg.train.epochs):
        lr = schedule(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        t0 = time.time()
        train_loss, parts = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, args.limit_batches
        )
        val_loss = validate(model, val_loader, criterion, device) if val_loader else float("nan")
        dt = time.time() - t0

        print(
            f"[epoch {epoch + 1:03d}/{cfg.train.epochs}] lr={lr:.2e} "
            f"train={train_loss:.4f} (coord={parts['coord']:.3f} conf={parts['conf']:.3f} "
            f"cls={parts['class']:.3f}) val={val_loss:.4f} {dt:.1f}s"
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("lr", lr, epoch)
            for key, value in parts.items():
                writer.add_scalar(f"loss/{key}", value, epoch)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, best_val, cfg)
        if val_loss < best_val or math.isnan(best_val):
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best_val, cfg)
            print(f"[train]   new best val loss {best_val:.4f} -> checkpoints/best.pt")

    if writer is not None:
        writer.close()
    print(f"[train] done. best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
