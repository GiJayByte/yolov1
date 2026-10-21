"""mAP@0.5 evaluation on PASCAL VOC 2007 test (11-point interpolation).

CLI::

    python -m src.metrics --config configs/voc.yaml --checkpoint checkpoints/best.pt

Writes a per-class AP table to stdout, a JSON report and a bar chart under
``assets/figures/``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import VOC_CLASSES, load_config
from .dataset import VOCDataset
from .model import build_model
from .predict import load_model
from .utils import box_iou, decode_predictions

IOU_THRESHOLD = 0.5


def voc_ap_11point(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """VOC2007 11-point interpolated average precision."""
    if n_gt == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = 0.0
    for t in np.arange(0.0, 1.01, 0.1):
        mask = recall >= t
        ap += (precision[mask].max() if mask.any() else 0.0) / 11.0
    return float(ap)


def _ap_for_class(detections, gt_by_image, n_gt) -> float:
    """detections: list of (image_idx, score, box). gt_by_image: {idx: [(box, difficult)]}."""
    detections = sorted(detections, key=lambda d: -d[1])
    matched: dict[int, set[int]] = defaultdict(set)
    tp, fp = [], []

    for image_idx, _score, box in detections:
        candidates = gt_by_image.get(image_idx, [])
        best_iou, best_j = 0.0, -1
        for j, (gt_box, _) in enumerate(candidates):
            iou = float(box_iou(box.unsqueeze(0), gt_box.unsqueeze(0))[0, 0])
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou > IOU_THRESHOLD and best_j >= 0:
            difficult = candidates[best_j][1]
            if difficult:
                continue  # ignored by the VOC protocol
            if best_j not in matched[image_idx]:
                matched[image_idx].add(best_j)
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    return voc_ap_11point(np.array(tp), np.array(fp), n_gt)


@torch.no_grad()
def evaluate(model, dataset, cfg, device, conf_threshold=None, nms_iou=None) -> dict:
    conf_threshold = cfg.inference.conf_threshold if conf_threshold is None else conf_threshold
    nms_iou = cfg.inference.nms_iou if nms_iou is None else nms_iou

    per_class_dets: dict[int, list] = defaultdict(list)
    per_class_gts: dict[int, dict] = defaultdict(dict)

    model.eval()
    start = time.time()
    for idx in tqdm(range(len(dataset)), desc="eval"):
        image, _target, meta = dataset[idx]
        pred = model(image.unsqueeze(0).to(device))
        res = decode_predictions(
            pred, cfg.model.S, cfg.model.B, cfg.model.C, conf_threshold, nms_iou
        )[0]

        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            per_class_dets[int(label)].append((idx, float(score), box.cpu()))
        for box, label, difficult in zip(meta["boxes"], meta["labels"], meta["difficult"]):
            per_class_gts[int(label)].setdefault(idx, []).append((box, int(difficult)))

    aps = {}
    for c in range(cfg.model.C):
        n_gt = sum(
            1
            for items in per_class_gts[c].values()
            for (_box, difficult) in items
            if not difficult
        )
        aps[VOC_CLASSES[c]] = _ap_for_class(per_class_dets[c], per_class_gts[c], n_gt)

    elapsed = time.time() - start
    return {
        "per_class_ap": aps,
        "mAP@0.5": float(np.mean(list(aps.values()))),
        "num_images": len(dataset),
        "seconds": elapsed,
        "fps": len(dataset) / max(elapsed, 1e-9),
    }


def save_report(report: dict, output_dir: str = "assets/figures") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "voc_eval.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(report["per_class_ap"].keys())
        values = list(report["per_class_ap"].values())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(names, values, color="#c33")
        ax.axhline(report["mAP@0.5"], color="black", ls="--", label=f"mAP {report['mAP@0.5']:.3f}")
        ax.set_ylabel("AP@0.5")
        ax.set_title("YOLOv1 per-class AP on VOC2007 test")
        plt.xticks(rotation=60, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "per_class_ap.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"[metrics] skipped chart: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLOv1 mAP@0.5 on VOC2007 test")
    p.add_argument("--config", default="configs/voc.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--nms", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="assets/figures")
    p.add_argument("--limit", type=int, default=0, help="cap images (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = VOCDataset(
        root=cfg.data.root,
        year=cfg.data.year,
        image_set="test",
        extra_year=None,
        image_size=cfg.data.image_size,
        S=cfg.model.S,
        B=cfg.model.B,
        C=cfg.model.C,
        letterbox_enabled=cfg.data.letterbox,
        train=False,
        return_meta=True,
        download=True,
    )
    if args.limit:
        dataset.index = dataset.index[: args.limit]

    model = load_model(cfg, args.checkpoint, device)
    report = evaluate(model, dataset, cfg, device, args.conf, args.nms)

    print("\nPer-class AP@0.5")
    for name, ap in report["per_class_ap"].items():
        print(f"  {name:<14} {ap:.3f}")
    print(f"\nmAP@0.5 = {report['mAP@0.5']:.4f}")
    print(f"images={report['num_images']}  time={report['seconds']:.1f}s  fps={report['fps']:.1f}")

    save_report(report, args.output_dir)
    print(f"[metrics] report written to {args.output_dir}/voc_eval.json")


if __name__ == "__main__":
    main()
