import torch

from src.loss import YOLOv1Loss
from src.utils import encode_targets


def _perfect_case(S=7, B=2, C=20):
    boxes = torch.tensor([[0.2, 0.3, 0.5, 0.8], [0.6, 0.6, 0.9, 0.95]])
    labels = torch.tensor([3, 14])
    target = encode_targets(boxes, labels, S, B, C).unsqueeze(0)
    return target


def test_perfect_prediction_has_zero_loss():
    S, B, C = 7, 2, 20
    criterion = YOLOv1Loss(S, B, C)
    target = _perfect_case(S, B, C)
    out = criterion(target.clone(), target)
    assert out["loss"].item() < 1e-5
    assert out["coord"].item() < 1e-5
    assert out["conf"].item() < 1e-5
    assert out["class"].item() < 1e-5


def test_noobj_weight_scales_confidence_loss():
    S, B, C = 7, 2, 20
    target = torch.zeros(1, S, S, B * 5 + C)
    pred = torch.zeros(1, S, S, B * 5 + C)
    pred[..., 4] = 0.5  # every box predicts confidence with no objects present

    low = YOLOv1Loss(S, B, C, lambda_noobj=0.5)(pred, target)["conf"].item()
    high = YOLOv1Loss(S, B, C, lambda_noobj=2.0)(pred, target)["conf"].item()
    assert abs(high / low - 4.0) < 1e-4


def test_lambda_coord_scales_coord_loss():
    S, B, C = 7, 2, 20
    target = _perfect_case(S, B, C)
    pred = target.clone()
    pred[..., 0] += 0.1  # perturb x of the (responsible) box

    low = YOLOv1Loss(S, B, C, lambda_coord=1.0)(pred, target)["coord"].item()
    high = YOLOv1Loss(S, B, C, lambda_coord=5.0)(pred, target)["coord"].item()
    assert abs(high / low - 5.0) < 1e-4


def test_coord_loss_only_on_responsible_box():
    S, B, C = 7, 2, 20
    target = _perfect_case(S, B, C)
    # perturb the *second* (non-responsible) box in every cell; coord loss must stay 0
    pred = target.clone()
    pred[..., B * 5 - 4 : B * 5] += 0.2
    out = YOLOv1Loss(S, B, C)(pred, target)
    assert out["coord"].item() < 1e-5
