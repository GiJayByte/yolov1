import torch

from src.utils import (
    box_iou,
    decode_predictions,
    encode_targets,
    non_max_suppression,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_xywh_xyxy_round_trip():
    boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.2, 0.2]])
    assert torch.allclose(xyxy_to_xywh(xywh_to_xyxy(boxes)), boxes, atol=1e-6)


def test_box_iou_identical_and_disjoint():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert box_iou(a, a)[0, 0].item() == 1.0

    b = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
    assert box_iou(a, b)[0, 0].item() == 0.0


def test_box_iou_half_overlap():
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[1.0, 0.0, 3.0, 2.0]])
    expected = 2.0 / 6.0
    assert abs(box_iou(a, b)[0, 0].item() - expected) < 1e-6


def test_nms_suppresses_overlap_keeps_distinct():
    boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.05, 0.05, 1.0, 1.0], [0.0, 0.0, 0.2, 0.2]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
    assert 0 in keep.tolist()
    assert 1 not in keep.tolist()
    assert 2 in keep.tolist()


def test_encode_decode_round_trip():
    S, B, C = 7, 2, 20
    boxes = torch.tensor([[0.30, 0.40, 0.50, 0.70]])
    labels = torch.tensor([5])
    target = encode_targets(boxes, labels, S, B, C)

    dets = decode_predictions(target, S, B, C, conf_threshold=0.05, nms_iou=0.5)[0]
    assert dets["boxes"].shape[0] == 1
    assert dets["labels"][0].item() == 5
    assert torch.allclose(dets["boxes"][0], boxes[0], atol=1e-4)


def test_decode_thresholds_low_confidence():
    S, B, C = 7, 2, 20
    pred = torch.zeros(S, S, B * 5 + C)
    pred[0, 0, 4] = 0.1  # low confidence
    pred[0, 0, B * 5 + 3] = 1.0
    dets = decode_predictions(pred, S, B, C, conf_threshold=0.5)[0]
    assert dets["boxes"].shape[0] == 0
