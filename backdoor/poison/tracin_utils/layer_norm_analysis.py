__all__ = [
    "log_stats",
]

import logging
from typing import NoReturn, Optional, Tuple

from torch import Tensor

from . import utils
from .. import influence_utils
from ..utils import ClassifierBlock


def log_stats(block: Optional[ClassifierBlock],
              ep: Optional[int], subep: Optional[int], ex_id: Optional[int],
              grad_x: Tuple[Tensor, ...]) -> NoReturn:
    r"""
    Log the layer parameter information
    :param block: Block under analysis
    :param ep: Epoch under analysis
    :param subep: Subepoch under analysis
    :param grad_x: Representative example gradient
    :param ex_id: Optional target example ID number
    """
    assert isinstance(grad_x, list), "Gradient is expected to be a list"

    header = influence_utils.build_log_start_flds(block=block, res_type=None, ep=ep,
                                                  subepoch=subep, ex_id=ex_id)
    # Layer count information.  Some layers may none
    n_layers = len(grad_x)
    n_none = len([vec for vec in grad_x if vec is None])
    logging.info(f"{header} Total # Layerwise-Norm Layers: {n_layers}")
    logging.info(f"{header} # Unused Layerwise-Norm Layers: {n_none}")

    # Each element in the vector is a parameter
    param_cnts = [vec.numel() if vec is not None else 0 for vec in grad_x]
    tot_num_params = sum(param_cnts)
    logging.info(f"{header} Total # Params: {tot_num_params:.4E}")

    norm = utils.flatten_grad([grad_x]).norm()  # noqa
    logging.info(f"{header} Total Norm: {norm:.3E}")
    # Log parameter and norm information
    for i, vec in enumerate(grad_x):
        if vec is None:
            n_params, norm = 0, 0
        else:
            n_params, norm = vec.numel(), vec.norm()
        logging.info(f"{header} Layer #{i + 1} -- {n_params} Params: {norm:.3E}")
