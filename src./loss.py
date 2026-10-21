"""YOLOv1 loss.

The paper's sum-squared-error objective with ``lambda_coord`` / ``lambda_noobj``
weighting and ``sqrt`` on width/height:

    L = lcoord * SUM_obj   SUM_boxes [(x-x^)2 + (y-y^)2]
      + lcoord * SUM_obj   SUM_boxes [(sqrt(w)-sqrt(w^))2 + (sqrt(h)-sqrt(h^))2]
      +          SUM_obj   SUM_boxes (C - C^)2
      + lnoobj * SUM_noobj SUM_boxes (C - C^)2
      +          SUM_obj   SUM_classes (p(c) - p^(c))2

Coordinate loss is applied only to the *responsible* predictor in each object
cell (the one with the highest IoU against the ground truth). Confidence target
is 1 for the responsible box and 0 otherwise.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils import cell_reference_grid


def _to_xyxy(box_xywh: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor, S: int):
    """(N, S, S, 4) cell-relative xywh -> (N, S, S, 4) image xyxy."""
    tx, ty, tw, th = box_xywh.unbind(-1)
    cx = (grid_x.unsqueeze(0) + tx) / S
    cy = (grid_y.unsqueeze(0) + ty) / S
    w = tw.clamp(min=0)
    h = th.clamp(min=0)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class YOLOv1Loss(nn.Module):
    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ) -> None:
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        S, B, C = self.S, self.B, self.C
        D = B * 5 + C
        N = pred.shape[0]
        pred = pred.reshape(N, S, S, D)
        target = target.reshape(N, S, S, D)

        pred_box = pred[..., : B * 5].reshape(N, S, S, B, 5)
        tgt_box = target[..., : B * 5].reshape(N, S, S, B, 5)
        pred_cls = pred[..., B * 5 :]
        tgt_cls = target[..., B * 5 :]

        obj_mask = target[..., 4] > 0.5  # (N, S, S) - slot 0 confidence
        device, dtype = pred.device, pred.dtype
        grid_x, grid_y = cell_reference_grid(S, device, dtype)

        if obj_mask.any():
            gt_xyxy = _to_xyxy(target[..., :4], grid_x, grid_y, S)  # (N, S, S, 4)
            pred_xyxy = torch.stack(
                [_to_xyxy(pred_box[..., b, :4], grid_x, grid_y, S) for b in range(B)],
                dim=3,
            )  # (N, S, S, B, 4)

            gt = gt_xyxy.unsqueeze(3)  # (N, S, S, 1, 4)
            ix1 = torch.maximum(gt[..., 0], pred_xyxy[..., 0])
            iy1 = torch.maximum(gt[..., 1], pred_xyxy[..., 1])
            ix2 = torch.minimum(gt[..., 2], pred_xyxy[..., 2])
            iy2 = torch.minimum(gt[..., 3], pred_xyxy[..., 3])
            inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
            area_gt = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (gt[..., 3] - gt[..., 1]).clamp(min=0)
            area_p = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0) * (
                pred_xyxy[..., 3] - pred_xyxy[..., 1]
            ).clamp(min=0)
            iou = inter / (area_gt + area_p - inter + 1e-9)  # (N, S, S, B)
            responsible = iou.argmax(dim=3)  # (N, S, S)
        else:
            responsible = torch.zeros(N, S, S, dtype=torch.long, device=device)

        resp_mask = torch.zeros(N, S, S, B, device=device, dtype=dtype)
        resp_mask.scatter_(3, responsible.unsqueeze(3), 1.0)
        resp_mask = resp_mask * obj_mask.unsqueeze(3).to(dtype)

        # ---- coordinate loss (responsible box only) ----
        xy_loss = ((pred_box[..., :2] - tgt_box[..., :2]) ** 2).sum(-1)
        pred_wh = pred_box[..., 2:4].clamp(min=0)
        tgt_wh = tgt_box[..., 2:4].clamp(min=0)
        wh_loss = ((pred_wh.sqrt() - tgt_wh.sqrt()) ** 2).sum(-1)
        coord_loss = self.lambda_coord * ((xy_loss + wh_loss) * resp_mask).sum()

        # ---- confidence loss ----
        obj_scale = obj_mask.unsqueeze(3).to(dtype)
        noobj_scale = (~obj_mask).unsqueeze(3).to(dtype)
        pred_conf = pred_box[..., 4]
        tgt_conf = tgt_box[..., 4]
        obj_conf_loss = ((pred_conf - tgt_conf) ** 2 * obj_scale).sum()
        noobj_conf_loss = (pred_conf**2 * noobj_scale).sum()
        conf_loss = obj_conf_loss + self.lambda_noobj * noobj_conf_loss

        # ---- class loss ----
        class_loss = ((pred_cls - tgt_cls) ** 2 * obj_mask.unsqueeze(-1).to(dtype)).sum()

        total = (coord_loss + conf_loss + class_loss) / N
        return {
            "loss": total,
            "coord": coord_loss / N,
            "conf": conf_loss / N,
            "class": class_loss / N,
            "obj_conf": obj_conf_loss / N,
            "noobj_conf": noobj_conf_loss / N,
        }


def build_loss(cfg) -> YOLOv1Loss:
    return YOLOv1Loss(
        S=cfg.model.S,
        B=cfg.model.B,
        C=cfg.model.C,
        lambda_coord=cfg.loss.lambda_coord,
        lambda_noobj=cfg.loss.lambda_noobj,
    )
