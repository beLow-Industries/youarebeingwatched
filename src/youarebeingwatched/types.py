from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

import numpy as np

Frame: TypeAlias = np.ndarray


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


class CropMask(Protocol):
    def apply(self, crop: Frame, box: Box, source_shape: tuple[int, ...]) -> Frame:
        ...


@dataclass(frozen=True)
class UnmaskedCrop:
    def apply(self, crop: Frame, box: Box, source_shape: tuple[int, ...]) -> Frame:
        return crop


@dataclass(frozen=True)
class BinaryCropMask:
    data: np.ndarray = field(compare=False, repr=False)

    def apply(self, crop: Frame, box: Box, source_shape: tuple[int, ...]) -> Frame:
        import cv2

        crop_height, crop_width = crop.shape[:2]
        if crop_height <= 0 or crop_width <= 0:
            return crop

        source_height, source_width = source_shape[:2]
        mask = self.data
        if mask.shape[:2] != (source_height, source_width):
            mask = cv2.resize(mask.astype(np.uint8), (source_width, source_height), interpolation=cv2.INTER_NEAREST)

        mask_crop = mask[box.y1 : box.y2, box.x1 : box.x2]
        if mask_crop.shape[:2] != (crop_height, crop_width):
            mask_crop = cv2.resize(mask_crop.astype(np.uint8), (crop_width, crop_height), interpolation=cv2.INTER_NEAREST)

        keep = mask_crop.astype(bool)
        masked = np.zeros_like(crop)
        masked[keep] = crop[keep]
        return masked


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: Box
    mask: CropMask = field(default_factory=UnmaskedCrop, compare=False, repr=False)
