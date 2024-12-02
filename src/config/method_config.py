from dataclasses import dataclass, field
from typing import List

from logging import getLogger

logger = getLogger()

@dataclass
class MethodConfig:
    weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )