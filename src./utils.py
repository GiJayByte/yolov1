"""Geometry and decoding helpers shared by the whole pipeline.

Box conventions used everywhere:
  * "xyxy"  -> (x1, y1, x2, y2) with x1 < x2, y1 < y2
  * "xywh"  -> (cx, cy, w, h) center form
  * all coordinates are normalised to the image, i.e. in [0, 1]
  * inside a grid cell the network predicts (x, y) *relative to the cell*
    (offset in [0, 1]) and (w, h) as a fraction of the whole image.
"""

from __future__ import annotations

import torch

BOX_SLOTS = ("x", "y", "w", "h", "conf")


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    half_w, half_h = w / 2, h / 2
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    area_a = box_area(a).unsqueeze(1)
    area_b = box_area(b).unsqueeze(0)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-9)


def non_max_suppression(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """Greedy NMS. Returns indices (into ``boxes``) of the boxes to keep."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current = order[0]
        keep.append(int(current))
        if order.numel() == 1:
            break
        ious = box_iou(boxes[current : current + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def cell_reference_grid(S: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (grid_x, grid_y), each (S, S), holding the cell column / row index."""
    rows = torch.arange(S, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(rows, rows, indexing="ij")
    return grid_x, grid_y


def encode_targets(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    S: int,
    B: int,
    C: int,
) -> torch.Tensor:
    """Encode one image's GT into a ``(S, S, B*5 + C)`` target tensor.

    ``boxes`` are normalised xyxy, ``labels`` are class indices. Each object is
    assigned to the cell containing its centre and written into box slot 0; the
    responsible slot is resolved later in the loss. If two objects fall in the
    same cell (rare at S=7) the larger one wins - a documented YOLOv1 limit.
    """
    target = torch.zeros(S, S, B * 5 + C, dtype=boxes.dtype, device=boxes.device)
    if boxes.numel() == 0:
        return target

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    col = (cx * S).floor().clamp(0, S - 1).long()
    row = (cy * S).floor().clamp(0, S - 1).long()
    tx = cx * S - col.to(boxes.dtype)
    ty = cy * S - row.to(boxes.dtype)

    for k in range(boxes.shape[0]):
        i, j = int(row[k]), int(col[k])
        if target[i, j, 4] > 0:
            existing_area = (target[i, j, 2] * target[i, j, 3]).item()
            if existing_area >= (w[k] * h[k]).item():
                continue
        target[i, j, 0] = tx[k]
        target[i, j, 1] = ty[k]
        target[i, j, 2] = w[k]
        target[i, j, 3] = h[k]
        target[i, j, 4] = 1.0
        target[i, j, B * 5 :] = 0
        target[i, j, B * 5 + int(labels[k])] = 1.0
    return target


@torch.no_grad()
def decode_predictions(
    pred: torch.Tensor,
    S: int,
    B: int,
    C: int,
    conf_threshold: float = 0.25,
    nms_iou: float = 0.5,
) -> list[dict[str, torch.Tensor]]:
    """Turn raw ``(*, S, S, B*5+C)`` output into per-image detections.

    Returns a list (one entry per image) of dicts with keys
    ``boxes`` (xyxy, normalised), ``scores`` and ``labels``. NMS is applied
    per class, matching the VOC evaluation protocol.
    """
    batched = pred.dim() == 4
    if not batched:
        pred = pred.unsqueeze(0)

    N = pred.shape[0]
    device = pred.device
    results: list[dict[str, torch.Tensor]] = []

    for n in range(N):
        p = pred[n]
        box_raw = p[..., : B * 5].reshape(S, S, B, 5)
        cls_prob = p[..., B * 5 :]  # (S, S, C)

        grid_x, grid_y = cell_reference_grid(S, device, p.dtype)
        tx, ty, tw, th = box_raw[..., 0], box_raw[..., 1], box_raw[..., 2], box_raw[..., 3]
        cx = (grid_x.unsqueeze(-1) + tx) / S
        cy = (grid_y.unsqueeze(-1) + ty) / S
        w = tw.clamp(min=0)
        h = th.clamp(min=0)
        boxes = torch.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
        ).reshape(-1, 4)

        conf = box_raw[..., 4]  # (S, S, B)
        scores = conf.unsqueeze(-1) * cls_prob.unsqueeze(2)  # (S, S, B, C)
        scores = scores.reshape(-1, C)

        best_score, best_class = scores.max(dim=1)
        boxes = boxes.clamp(0.0, 1.0)

        keep_boxes: list[torch.Tensor] = []
        keep_scores: list[torch.Tensor] = []
        keep_labels: list[torch.Tensor] = []

        for c in range(C):
            idx = torch.nonzero((best_class == c) & (best_score > conf_threshold)).flatten()
            if idx.numel() == 0:
                continue
            kept = non_max_suppression(boxes[idx], best_score[idx], nms_iou)
            sel = idx[kept]
            keep_boxes.append(boxes[sel])
            keep_scores.append(best_score[sel])
            keep_labels.append(torch.full_like(best_class[sel], c))

        if keep_boxes:
            results.append(
                {
                    "boxes": torch.cat(keep_boxes),
                    "scores": torch.cat(keep_scores),
                    "labels": torch.cat(keep_labels),
                }
            )
        else:
            results.append(
                {
                    "boxes": torch.zeros(0, 4, device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                }
            )
    return results
