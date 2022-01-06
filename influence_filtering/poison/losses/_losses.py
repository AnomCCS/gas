__all__ = [
    "Loss",
    "RiskEstimatorBase",
    "TORCH_DEVICE",
    "ce_loss",
]

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn

TORCH_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# _log_ce_module = nn.CrossEntropyLoss(reduction="none")
log_bce_module = nn.BCEWithLogitsLoss(reduction="none")
log_ce_module = nn.CrossEntropyLoss(reduction="none")


def ce_loss(inputs: Tensor, targets: Tensor, use_bce: bool = False) -> Tensor:
    r""" Cross-entropy loss that takes two arguments instead of default one """
    # clamp_val = 1e7
    global log_ce_module, log_bce_module
    if use_bce:
        loss_module = log_bce_module
        targets = targets.float() if inputs.dtype == torch.float else targets.double()
    else:
        loss_module = log_ce_module

    return loss_module.forward(inputs, targets)
    # return _log_ce_module.forward(inputs.clamp(-clamp_val, clamp_val), targets)


class RiskEstimatorBase(ABC):
    def __init__(self, train_loss: Callable, valid_loss: Optional[Callable] = None,
                 name_suffix: str = ""):
        if valid_loss is not None:
            valid_loss = train_loss

        self.tr_loss = train_loss
        self.val_loss = valid_loss

        self._name_suffix = name_suffix

    @abstractmethod
    def name(self) -> str:
        r""" Name of the risk estimator """

    def calc_train_loss(self, dec_scores: Tensor, labels: Tensor, **kwargs) -> Tensor:
        r""" Calculates the risk using the TRAINING specific loss function """
        return self._loss(dec_scores=dec_scores, lbls=labels, f_loss=self.tr_loss, **kwargs)

    def calc_validation_loss(self, dec_scores: Tensor, labels: Tensor, **kwargs) -> Tensor:
        r""" Calculates the risk using the VALIDATION specific loss function """
        return self._loss(dec_scores=dec_scores, lbls=labels, f_loss=self.val_loss, **kwargs)

    @staticmethod
    def has_any(mask: Tensor) -> bool:
        r""" Checks if the mask has any set to \p True """
        assert mask.dtype == torch.bool, "Mask should be a Boolean Tensor"
        return bool(mask.any().item())

    @abstractmethod
    def _loss(self, dec_scores: Tensor, lbls: Tensor, f_loss: Callable, **kwargs) -> Tensor:
        r""" Single function for calculating the loss """


class Loss(RiskEstimatorBase):
    RETURN_MEAN_KEY = "return_mean"

    def __init__(self, train_loss: Callable, valid_loss: Optional[Callable] = None,
                 name_suffix: str = "", use_bce: bool = True):
        super().__init__(train_loss=train_loss, valid_loss=valid_loss, name_suffix=name_suffix)
        self._use_bce = use_bce

    def name(self) -> str:
        return self._name_suffix

    def _loss(self, dec_scores: Tensor, lbls: Tensor, f_loss: Callable, **kwargs) -> Tensor:
        r""" Straight forward PN loss -- No weighting by prior & label """
        assert self._use_bce or len(dec_scores.shape) == 2, "Bizarre input shape"
        assert dec_scores.shape[0] == lbls.shape[0], "Vector shape loss mismatch"

        lst_loss = f_loss(dec_scores, lbls, use_bce=self._use_bce)
        if self.RETURN_MEAN_KEY in kwargs and not kwargs[self.RETURN_MEAN_KEY]:
            return lst_loss
        return lst_loss.mean()
