"""Typed configuration loaded from YAML.

Every hyperparameter lives in ``configs/voc.yaml``; nothing is hard-coded in the
modules. Unknown keys in the YAML are ignored, missing keys fall back to the
dataclass defaults below.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, get_type_hints

import yaml

VOC_CLASSES: tuple[str, ...] = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(VOC_CLASSES)}


@dataclass
class DataConfig:
    root: str = "data"
    year: str = "2007"
    image_set: str = "trainval"
    extra_year: str | None = "2012"
    extra_image_set: str = "trainval"
    image_size: int = 448
    letterbox: bool = False
    num_workers: int = 4


@dataclass
class ModelConfig:
    S: int = 7
    B: int = 2
    C: int = 20
    variant: str = "full"
    pretrained: bool = False
    dropout: float = 0.5
    hidden_dim: int = 4096
    batchnorm: bool = True


@dataclass
class LossConfig:
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 135
    schedule: str = "paper"
    batch_size: int = 16
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    amp: bool = True
    grad_clip: float = 10.0
    eval_every: int = 5
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    resume: str | None = None
    seed: int = 42
    device: str = "auto"


@dataclass
class AugmentConfig:
    enabled: bool = True
    scale: float = 0.2
    translate: float = 0.2
    hsv: float = 0.5
    hflip: bool = False


@dataclass
class InferenceConfig:
    conf_threshold: float = 0.25
    nms_iou: float = 0.5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _populate(cls: type, data: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in hints:
        if name not in data:
            continue
        value = data[name]
        target = hints[name]
        if is_dataclass(target) and isinstance(value, dict):
            kwargs[name] = _populate(target, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(
        data=_populate(DataConfig, raw.get("data", {})),
        model=_populate(ModelConfig, raw.get("model", {})),
        loss=_populate(LossConfig, raw.get("loss", {})),
        train=_populate(TrainConfig, raw.get("train", {})),
        augment=_populate(AugmentConfig, raw.get("augment", {})),
        inference=_populate(InferenceConfig, raw.get("inference", {})),
    )


# --------------------------------------------------------------------------- #
# Learning-rate schedule (paper Section 2.2)
# --------------------------------------------------------------------------- #
SCHEDULE_EPOCHS: dict[str, list[int]] = {
    "paper": [75, 30, 30],
    "reduced": [10, 10, 5, 5],
}
_LR_MULTIPLIERS = [1.0, 0.1, 0.01, 0.001, 0.0001]


def build_lr_schedule(
    schedule: str, base_lr: float, warmup_epochs: int = 1
) -> Callable[[int], float]:
    """Return ``epoch -> learning rate``.

    During epoch 0 the LR rises linearly from ``0.1 * base_lr`` to ``base_lr``
    (the paper ramps to 1e-2 over the first epoch), then it is held at each
    segment value in turn.
    """
    segments = SCHEDULE_EPOCHS.get(schedule, SCHEDULE_EPOCHS["paper"])

    def lr_at(epoch: int) -> float:
        if epoch < warmup_epochs:
            frac = (epoch + 1) / max(warmup_epochs, 1)
            return base_lr * (0.1 + 0.9 * frac)
        e = epoch - warmup_epochs
        acc = 0
        for k, length in enumerate(segments):
            acc += length
            if e < acc:
                return base_lr * _LR_MULTIPLIERS[min(k, len(_LR_MULTIPLIERS) - 1)]
        return base_lr * _LR_MULTIPLIERS[min(len(segments), len(_LR_MULTIPLIERS)) - 1]

    return lr_at


def total_schedule_epochs(schedule: str, warmup_epochs: int = 1) -> int:
    return sum(SCHEDULE_EPOCHS.get(schedule, SCHEDULE_EPOCHS["paper"])) + warmup_epochs


def fields_of(cls: type) -> list[str]:
    return [f.name for f in fields(cls)]
