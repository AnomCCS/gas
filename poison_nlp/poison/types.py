__all__ = [
    "CutoffTailPoint",
]

import dataclasses
from typing import Optional


@dataclasses.dataclass
class CutoffTailPoint:
    r""" Tail cutoff point for the points under consideration """
    pred_lbl_only: Optional[float] = None
