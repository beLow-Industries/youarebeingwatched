from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

Frame: TypeAlias = np.ndarray


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: Box

