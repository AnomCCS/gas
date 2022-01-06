__all__ = [
    "InfluenceMethod",
    "MIN_LOSS",
    "build_log_start_flds",
    "calc_pearsonr",
    "calc_spearmanr",
    "construct_ep_str",
    "derivative_of_loss",
    "log_time",
]

import enum
import logging
import time
from typing import Callable, NoReturn, Optional

import scipy.stats

import torch
from torch import LongTensor, Tensor

MIN_LOSS = 1E-12


class InfluenceMethod(enum.Enum):
    r""" Influence method of interest """
    BASELINE = "Baseline No Filtering"
    RANDOM = "Random Filtering"
    RANDOM_TARG_CLASS = "Random Target Class Filtering"

    INF_FUNC = "Influence Function"
    INF_FUNC_SIM = f"{INF_FUNC} Similarity"
    INF_FUNC_SIM_L = f"{INF_FUNC_SIM} Layerwise"

    TRACINCP = f"TracInCP"

    GAS = "GAS"
    GAS_L = f"{GAS} Layerwise"

    TRACIN = "TracIn"
    TRACIN_NORMED = f"{TRACIN} Normalized"
    TRACIN_SIM = f"{TRACIN} Similarity"


def build_log_start_flds(block, res_type: Optional[InfluenceMethod],
                         ep: Optional[int] = None, subepoch: Optional[int] = None,
                         ex_id: Optional[int] = None) -> str:
    r""" Creates the log starter fields """
    flds = []
    if block is not None:
        flds.append(block.name())
    if ex_id is not None:
        flds.append(f"Ex={ex_id}")
    flds.append(construct_ep_str(ep=ep, subepoch=subepoch))
    if res_type is not None:
        flds.append(res_type.value)
    return " ".join(flds)


def log_time(res_type: InfluenceMethod):
    r""" Logs the running time of each influence method """
    def decorator(func):
        r""" Need to nest the decorator since decorator takes an argument (\p res_type) """
        def wrapper(*args, **kwargs) -> NoReturn:
            start = time.time()

            rets = func(*args, **kwargs)

            total = time.time() - start
            logging.info(f"{res_type.value} Execution Time: {total:,.6f} seconds")

            return rets
        return wrapper
    return decorator


def construct_ep_str(ep: Optional[int], subepoch: Optional[int]) -> str:
    r""" Helper method to standardize constructing the epoch strings """
    if ep is None:
        assert subepoch is None, "Subepoch is specified without an epoch"
        ep_str = "Final"
    else:
        ep_str = f"Ep {ep}"
        if subepoch is not None:
            ep_str = f"{ep_str}.{subepoch:03}"
    return ep_str


def _error_check_correlation_tensors(x: Tensor, y: Tensor) -> NoReturn:
    r""" Standardizes checks for correlation variables """
    assert x.numel() == y.numel(), "Mismatch in number of elements"
    assert len(x.shape) <= 2 and x.shape[0] == x.numel(), "x tensor has a bizarre shape"
    assert len(y.shape) <= 2 and y.shape[0] == y.numel(), "y tensor has a bizarre shape"


def calc_pearsonr(x: Tensor, y: Tensor) -> float:
    r""" Calculates Pearson coefficient between the \p x and \p y tensors """
    _error_check_correlation_tensors(x=x, y=y)
    x, y = x.numpy(), y.numpy()
    r, _ = scipy.stats.pearsonr(x=x, y=y)
    return r


def calc_spearmanr(x: Tensor, y: Tensor) -> float:
    r""" Calculates Pearson coefficient between the \p x and \p y tensors """
    _error_check_correlation_tensors(x=x, y=y)
    x, y = x.numpy(), y.numpy()
    r, _ = scipy.stats.spearmanr(a=x, b=y)
    return r


def derivative_of_loss(acts: Tensor, lbls: LongTensor, f_loss: Callable) -> Tensor:
    r"""
    Calculates the dervice of loss function \p f_loss w.r.t. output activation \p outs
    and labels \p lbls
    """
    for tensor, name in ((acts, "outs"), (lbls, "lbls")):
        assert len(tensor.shape) <= 2 and tensor.numel() == 1, f"Unexpected shape for {name}"
    # Need to require gradient to calculate derive
    acts = acts.detach().clone()
    acts.requires_grad = True
    # Calculates the loss
    loss = f_loss(acts, lbls).view([1])
    ones = torch.ones(acts.shape, dtype=acts.dtype, device=acts.device)
    # Back propagate the gradients
    loss.backward(ones)
    return acts.grad.clone().detach().type(acts.dtype)  # type: Tensor
