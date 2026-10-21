"""Single-image inference: decode 7x7x30 -> boxes, NMS, draw.

CLI::

    python -m src.predict --config configs/voc.yaml --checkpoint checkpoints/best.pt \
        --image path/to.jpg --output out.jpg --conf 0.25 --nms 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .config import VOC_CLASSES, load_config
from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .model import build_model
from .utils import decode_predictions

_PALETTE = [
    (220, 20, 60), (0, 128, 255), (34, 139, 34), (255, 140, 0), (148, 0, 211),
    (0, 206, 209), (255, 20, 147), (139, 69, 19), (70, 130, 180), (154, 205, 50),
    (255, 215, 0), (0, 100, 0), (199, 21, 133), (47, 79, 79), (255, 99, 71),
    (0, 0, 205), (188, 143, 143), (46, 139, 87), (210, 105, 30), (105, 105, 105),
]


def load_model(cfg, checkpoint: str | None, device: torch.device):
    model = build_model(cfg).to(device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    rgb = image.convert("RGB")
    tensor = F.to_tensor(rgb)
    tensor = F.interpolate(
        tensor.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
    ).squeeze(0)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


@torch.no_grad()
def detect(
    model,
    image: Image.Image,
    cfg,
    device: torch.device,
    conf_threshold: float | None = None,
    nms_iou: float | None = None,
) -> list[dict]:
    """Return detections in original-image pixel coordinates."""
    conf_threshold = cfg.inference.conf_threshold if conf_threshold is None else conf_threshold
    nms_iou = cfg.inference.nms_iou if nms_iou is None else nms_iou

    tensor = preprocess(image, cfg.data.image_size).unsqueeze(0).to(device)
    pred = model(tensor)
    res = decode_predictions(
        pred, cfg.model.S, cfg.model.B, cfg.model.C, conf_threshold, nms_iou
    )[0]

    width, height = image.size
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    boxes = res["boxes"].cpu() * scale
    return [
        {
            "box": boxes[i].tolist(),
            "score": float(res["scores"][i]),
            "label": int(res["labels"][i]),
            "name": VOC_CLASSES[int(res["labels"][i])],
        }
        for i in range(boxes.shape[0])
    ]


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = _PALETTE[det["label"] % len(_PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        caption = f"{det['name']} {det['score']:.2f}"
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        draw.rectangle(
            [text_bbox[0], text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color
        )
        draw.text((x1, text_bbox[1] - 2), caption, fill="white", font=font)
    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv1 single-image inference")
    p.add_argument("--config", default="configs/voc.yaml")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--image", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--nms", type=float, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(cfg, args.checkpoint, device)

    image = Image.open(args.image)
    detections = detect(model, image, cfg, device, args.conf, args.nms)
    print(f"[predict] {len(detections)} detections in {args.image}")
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        print(f"  {det['name']:<12} {det['score']:.3f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

    output = args.output or str(Path(args.image).with_name(Path(args.image).stem + "_pred.jpg"))
    draw_detections(image, detections).save(output)
    print(f"[predict] wrote {output}")


if __name__ == "__main__":
    main()
