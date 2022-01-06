__all__ = [
    "CLEAN_LABEL",
    "InfluenceMethod",
    "MIN_LOSS",
    "POISON_LABEL",
    "build_layer_norm",
    "build_log_start_flds",
    "calc_cutoff_detect_rates",
    "calc_poison_auprc",
    "count_poison",
    "flatten",
    "get_full_ids",
    "is_pois",
    "label_ids",
    "log_time",
    "reset_layer_norm_structs",
]

import enum
import logging
import time
from typing import Collection, NoReturn, Optional

import sklearn.metrics

import torch
from torch import BoolTensor, LongTensor, Tensor

from .. import _config as config

POISON_LABEL = +1
CLEAN_LABEL = -1
MIN_LOSS = 1E-12
MIN_NORM = 1E-10


class InfluenceMethod(enum.Enum):
    r""" Influence method of interest """
    REP_POINT = "Representer Point"
    REP_POINT_SIM = f"{REP_POINT} Similarity"

    INF_FUNC = "Influence Function"
    INF_FUNC_SIM = f"{INF_FUNC} Similarity"
    INF_FUNC_SIM_L = f"{INF_FUNC_SIM}-Layerwise"

    TRACINCP = "TracInCP"

    GAS = "GAS"
    GAS_L = f"{GAS} Layer Norm"

    TRACIN = "TracIn"
    TRACIN_SIM = f"{TRACIN} Similarity"
    TRACIN_SIM_L = f"{TRACIN_SIM}-Layerwise"


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


def label_ids(ids: Tensor) -> Tensor:
    r"""
    NLP poison always appears at the end of the dataset.
    :param ids: List of training set IDs.  Not required to be ordered.
    :return: One-to-one mapping of the training set IDs to either the poison or clean classes
    """
    labels = torch.full(ids.shape, fill_value=CLEAN_LABEL, dtype=torch.long)

    threshold = config.N_TRAIN
    labels[ids >= threshold] = POISON_LABEL
    return labels


def is_pois(ids: LongTensor) -> BoolTensor:
    r""" Returns whether each ID is a backdoor """
    lbls = label_ids(ids=ids)
    return lbls == POISON_LABEL


def count_poison(ids: LongTensor) -> int:
    r""" Counts the number of poison examples """
    is_pois_mask = is_pois(ids=ids)
    return torch.sum(is_pois_mask).item()


# def calc_poison_auprc(block, res_type: InfluenceMethod,
def calc_poison_auprc(res_type: InfluenceMethod,
                      ids: Tensor, inf: Tensor,  # file_path: Optional[Path] = None,
                      ep: Optional[int] = None, n_updates: Optional[int] = None,
                      ex_id: Optional[int] = None) -> float:
    r""" Calculate the block's AUPRC """
    return _base_roc_calc(res_type=res_type, ids=ids, inf=inf,
                          ep=ep, n_updates=n_updates, ex_id=ex_id)


def _base_roc_calc(res_type: InfluenceMethod,
                   ids: Tensor, inf: Tensor,
                   ep: Optional[int] = None, n_updates: Optional[int] = None,
                   ex_id: Optional[int] = None) -> float:
    r"""
    Calculate and log the ROC

    :param res_type: Result type to be stored
    :param ids: Training example IDs
    :param inf: Corresponding influence values for the list of training ids
    :param ep: If specified, AUPRC is reported for a specific epoch
    :param n_updates: If specified, n_updates value
    :param ex_id: Example ID number
    :return: AUC value
    """
    labels = label_ids(ids=ids)

    # noinspection PyUnresolvedReferences
    prec, recall, _ = sklearn.metrics.precision_recall_curve(y_true=labels, probas_pred=inf)
    # noinspection PyUnresolvedReferences
    roc_val = sklearn.metrics.average_precision_score(labels, inf)
    name = "AUPRC"

    header = build_log_start_flds(res_type=res_type, ep=ep, n_updates=n_updates, ex_id=ex_id)
    flds = [header, f"{name}:", f"{roc_val:.6f}"]
    msg = " ".join(flds)
    logging.info(msg)

    return roc_val


def build_log_start_flds(res_type: Optional[InfluenceMethod],
                         ep: Optional[int] = None, n_updates: Optional[int] = None,
                         ex_id: Optional[int] = None) -> str:
    r""" Creates the log starter fields """
    flds = []
    if ex_id is not None:
        flds.append(f"Ex={ex_id}")
    flds.append(_construct_ep_str(ep=ep, n_updates=n_updates))
    if res_type is not None:
        flds.append(res_type.value)
    return " ".join(flds)


def _construct_ep_str(ep: Optional[int], n_updates: Optional[int]) -> str:
    r""" Helper method to standardize constructing the epoch strings """
    if ep is None:
        assert n_updates is None, "n_updates is specified without an epoch"
        ep_str = "Final"
    else:
        ep_str = f"Ep {ep}"
        if n_updates is not None:
            ep_str = f"{ep_str}.{n_updates:06}"
    return ep_str


def get_full_ids() -> LongTensor:
    r""" Constructs the full ID list """
    # Does not include end
    return torch.arange(0, config.N_TRAIN + config.POISON_CNT, dtype=torch.long)  # noqa


def calc_cutoff_detect_rates(res_type: InfluenceMethod, helpful_ids: LongTensor,
                             ep: Optional[int] = None,
                             n_updates: Optional[int] = None) -> NoReturn:
    r"""
    Logs how much poison is detected at various cutoffs

    :param res_type:
    :param helpful_ids: List of IDs ordered from most helpful to least helpful
    :param ep: Epoch number
    :param n_updates: Optional number of model updates so far
    """
    flds_str = build_log_start_flds(res_type=res_type, ep=ep, n_updates=n_updates)

    pois_mask = is_pois(helpful_ids)
    n_pois = torch.sum(pois_mask).item()  # Total number of poison
    for percent in (0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 10, 20):
        count = int(percent / 100. * helpful_ids.numel())
        p_bd = torch.sum(pois_mask[:count]).item()  # Number of poison at specified percent
        denom = min(n_pois, count)
        rate = p_bd / denom  # Fraction backdoor detected

        msg = f"{flds_str} Poison {percent}% Data: {p_bd} / {denom} ({rate:.1%})"
        logging.info(msg)


def flatten(vec: Collection[Tensor]) -> Tensor:
    r""" Flatten the gradient into a vector """
    return torch.cat([flat.detach().view([-1]) for flat in vec], dim=0)


# Use global variable to prevent reinitializing memory
gbl_layer_norm = None


def reset_layer_norm_structs() -> NoReturn:
    r""" Result all datastructures related to layer normalization """
    global gbl_layer_norm
    gbl_layer_norm = None


def build_layer_norm(grad) -> Tensor:
    r""" Construct a layer norm vector """
    assert len(grad) > 1, "Flatten vector is not supported"

    global gbl_layer_norm
    if gbl_layer_norm is None:
        gbl_layer_norm = [vec.clone().detach() for vec in grad]

    assert len(gbl_layer_norm) == len(grad), "Unexpected length mismatch"
    for layer, vec in zip(gbl_layer_norm, grad):  # type: Tensor, Tensor
        norm = vec.detach().norm().item()
        if norm < MIN_NORM:
            norm = MIN_NORM
        layer.fill_(norm)
    return flatten(gbl_layer_norm)
