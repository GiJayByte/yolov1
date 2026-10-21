import torch

from src.utils import encode_targets


def test_object_written_to_centre_cell():
    S, B, C = 7, 2, 20
    # centre of the box at (0.5, 0.5) -> cell (3, 3)
    boxes = torch.tensor([[0.40, 0.40, 0.60, 0.60]])
    labels = torch.tensor([7])
    target = encode_targets(boxes, labels, S, B, C)

    cell = target[3, 3]
    assert cell[4].item() == 1.0  # objectness slot 0
    assert cell[B * 5 + 7].item() == 1.0  # class 7 one-hot
    assert cell[B * 5 :].sum().item() == 1.0  # exactly one class

    # cell-relative centre should be ~0.5 in both axes
    assert abs(cell[0].item() - 0.5) < 1e-6
    assert abs(cell[1].item() - 0.5) < 1e-6


def test_offcenter_box_cell_and_offsets():
    S, B, C = 7, 2, 20
    # centre at (0.5/7 * 4.5) -> cell col 4? build explicitly: centre x = 0.643 -> *7 = 4.5
    boxes = torch.tensor([[0.5, 0.1, 0.5 + 2 * (4.5 / 7 - 0.5), 0.3]])
    cx = (boxes[0, 0] + boxes[0, 2]) / 2
    col = int(cx * S)
    target = encode_targets(boxes, torch.tensor([0]), S, B, C)
    assert target[0, col, 4].item() == 1.0
    assert abs(target[0, col, 0].item() - (cx * S - col)) < 1e-5


def test_empty_cells_stay_zero():
    S, B, C = 7, 2, 20
    boxes = torch.tensor([[0.10, 0.10, 0.20, 0.20]])
    target = encode_targets(boxes, torch.tensor([2]), S, B, C)
    assert target.sum().item() > 0
    zero_cells = target[target[:, :, 4] == 0]
    assert torch.all(zero_cells == 0)


def test_no_boxes_gives_all_zero_target():
    S, B, C = 7, 2, 20
    target = encode_targets(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), S, B, C)
    assert target.shape == (S, S, B * 5 + C)
    assert target.sum().item() == 0


def test_two_objects_one_cell_keeps_larger():
    S, B, C = 7, 2, 20
    small = [0.45, 0.45, 0.50, 0.50]
    large = [0.44, 0.44, 0.56, 0.56]
    boxes = torch.tensor([small, large])
    labels = torch.tensor([1, 9])
    target = encode_targets(boxes, labels, S, B, C)
    cell = target[3, 3]
    assert cell[4].item() == 1.0
    assert cell[B * 5 + 9].item() == 1.0  # the larger box (class 9) won
    assert cell[B * 5 :].sum().item() == 1.0
