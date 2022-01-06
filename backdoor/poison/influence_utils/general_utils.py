__all__ = [
    "BACKDOOR_LABEL",
    "CLEAN_LABEL",
    "InfluenceMethod",
    "check_is_bd",
    "construct_ep_str",
    "build_log_start_flds",
    "calc_backdoor_auprc",
    "calc_identified_backdoor_frac",
    "check_bd_ids_contents",
    "check_duplicate_ds_ids",
    "count_backdoor",
    "derivative_of_loss",
    "get_holdout_mask",
    "is_bd",
    "label_holdout",
    "label_ids",
    "log_all_results",
    "log_time",
]

import enum
import logging
import time
from typing import Callable, NoReturn, Optional

import sklearn.metrics

import torch
from torch import BoolTensor, LongTensor, Tensor

from .. import _config as config

BACKDOOR_LABEL = +1
CLEAN_LABEL = -1
MIN_LOSS = 1E-12

BD_CIFAR_OFFSET = 10000
SP_MAX_BD_ID = 300


class InfluenceMethod(enum.Enum):
    r""" Influence method of interest """
    INF_FUNC = "Influence Function"
    INF_FUNC_SIM = f"{INF_FUNC} Similarity"
    INF_FUNC_LAYER = f"{INF_FUNC_SIM}-Layerwise"

    REP_POINT = "Representer Point"
    REP_POINT_SIM = f"{REP_POINT} Similarity"

    TRACIN = "TracIn"
    TRACIN_SIM = f"{TRACIN}-Similarity"
    TRACINCP = "TracInCP"

    GAS = "GAS"
    GAS_L = f"{GAS}-Layerwise"

    LOSS_SPOT = "Spot Loss"
    LOSS_ADV_SPOT = "Spot Adv. Loss"
    LOSS_CLEAN_SPOT = "Spot Clean Loss"

    GRAD_NORM_MAG_RATIO = "Spot Gradient Norm Magnitude Ratio"


def _get_num_backdoor_examples(block) -> int:
    r""" Gets the number of backdoor examples considered for the specified block """
    num_bd_ex = block.loss.n_bd
    if num_bd_ex == 0:
        num_bd_ex = config.BACKDOOR_CNT
    return num_bd_ex


def check_duplicate_ds_ids(ds_ids: Tensor) -> NoReturn:
    r""" Ensure there are no duplicate dataset IDs """
    assert ds_ids.dtype == torch.long, "Dataset IDs does not appear to be longs"
    uniq = torch.unique(ds_ids)
    assert uniq.shape[0] == ds_ids.shape[0], "Duplicate dataset IDs should not occur"


def check_bd_ids_contents(bd_ids: Tensor) -> NoReturn:
    r""" Ensure not too many backdoor IDs """
    assert bd_ids.dtype == torch.long, "Backdoor IDs does not appear to be longs"
    uniq = torch.unique(bd_ids)
    # Add 1 since one possible additional value for non-backdoor examples
    assert uniq.shape[0] <= config.BACKDOOR_CNT + 1


def calc_identified_backdoor_frac(block, res_type: InfluenceMethod, helpful_bd_ids: Tensor,
                                  ep: Optional[int] = None, subepoch: Optional[int] = None,
                                  holdout_v_rest: bool = False,
                                  ex_id: Optional[int] = None) -> float:
    r""" Calculates and logs the number of examples from poisoned blocks """
    assert helpful_bd_ids.dtype == torch.long, "helpful_ids does not appear to be a list of IDs"
    check_bd_ids_contents(bd_ids=helpful_bd_ids)

    # Extract the most influential samples:
    if not holdout_v_rest:
        heldout_mask = get_holdout_mask(bd_ids=helpful_bd_ids)
        n_ex = config.BACKDOOR_CNT - config.BACKDOOR_HOLDOUT
        # Exclude heldout examples
        labels = label_ids(bd_ids=helpful_bd_ids[~heldout_mask][:n_ex])
        bd_only = labels[labels == BACKDOOR_LABEL]
    else:
        n_ex = config.BACKDOOR_HOLDOUT
        bd_only = helpful_bd_ids[:n_ex]
        bd_only = bd_only[get_holdout_mask(bd_ids=bd_only)]

    # Extract the subset of IDs that are poison
    frac_pois = bd_only.shape[0] / n_ex

    # Store the poison detect rate only if the final results
    if ep is None:
        block.backdoor_detect_rate.set_value(res_type, frac_pois)

    flds_val = build_log_start_flds(block=block, res_type=res_type,
                                    ep=ep, subepoch=subepoch, ex_id=ex_id)
    data_name = 'Holdout' if holdout_v_rest else 'Backdoor'
    logging.info(f"{flds_val} {data_name} Detected: {frac_pois:.1%}")
    return frac_pois


def build_log_start_flds(block, res_type: Optional[InfluenceMethod],
                         ep: Optional[int] = None, subepoch: Optional[int] = None,
                         ex_id: Optional[int] = None) -> str:
    r""" Creates the log starter fields """
    flds = []
    if ex_id is not None:
        flds.append(f"Ex={ex_id}")
        bd_str = "Adv" if check_is_bd(ex_id) else "Cl"
        flds.append(f"({bd_str})")
    flds.append(construct_ep_str(ep=ep, subepoch=subepoch))
    if res_type is not None:
        flds.append(res_type.value)
    return " ".join(flds)


def check_is_bd(id_val: int) -> bool:
    r""" Checks if the specified example is backdoored example """
    if config.DATASET.is_cifar():
        return id_val >= BD_CIFAR_OFFSET
    elif config.DATASET.is_speech():
        return id_val < SP_MAX_BD_ID
    raise ValueError("Cannot check whether {id_val} is backdoor for this dataset")


def log_all_results(block, res_type: InfluenceMethod,
                    helpful_bd_ids: Tensor, helpful_ds_ids: Tensor, helpful_inf: Tensor,
                    ep: Optional[int] = None, subepoch: Optional[int] = None,
                    holdout_v_rest: bool = False, ex_id: Optional[int] = None) -> NoReturn:
    calc_backdoor_auprc(block=block, res_type=res_type,
                        bd_ids=helpful_bd_ids, ds_ids=helpful_ds_ids, inf=helpful_inf,
                        ep=ep, subepoch=subepoch, ex_id=ex_id,
                        holdout_v_rest=holdout_v_rest)


def label_ids(bd_ids: Tensor, n_bd: Optional[int] = None, inc_holdout: bool = False) -> Tensor:
    r"""
    Poison is treated as the positive class with label "+1".  Clean data is treated as the negative
    class with label "-1".
    :param bd_ids: List of training set IDs.  Not required to be ordered.
    :param n_bd: number of backdoor examples
    :param inc_holdout: Exclude the holdout data
    :return: One-to-one mapping of the training set IDs to either the poison or clean classes
    """
    check_bd_ids_contents(bd_ids=bd_ids)

    # assert n_bd > 0, "At least one backdoor example is expected"
    if n_bd is None or n_bd == 0:
        n_bd = config.BACKDOOR_CNT

    labels = torch.full(bd_ids.shape, fill_value=CLEAN_LABEL, dtype=torch.long)
    mask = bd_ids < n_bd
    if not inc_holdout:
        mask &= ~get_holdout_mask(bd_ids=bd_ids)

    labels[mask] = BACKDOOR_LABEL
    return labels


def is_bd(bd_ids: Tensor, n_bd: Optional[int] = None, inc_holdout: bool = False) -> BoolTensor:
    r""" Returns whether each ID is a backdoor """
    lbls = label_ids(bd_ids=bd_ids, n_bd=n_bd, inc_holdout=inc_holdout)
    return lbls == BACKDOOR_LABEL


def count_backdoor(bd_ids: Tensor, inc_holdout: bool = False) -> int:
    r""" Counts the number of poison examples """
    labels = label_ids(bd_ids=bd_ids, inc_holdout=inc_holdout)
    return torch.sum(labels == BACKDOOR_LABEL).item()


def label_holdout(bd_ids: Tensor) -> Tensor:
    r""" Label specifically the examples held out """
    labels = torch.full(bd_ids.shape, fill_value=CLEAN_LABEL, dtype=torch.long)
    labels[get_holdout_mask(bd_ids=bd_ids)] = BACKDOOR_LABEL
    return labels


def get_holdout_mask(bd_ids: Tensor) -> Tensor:
    r""" Labels only the held out examples """
    mask = bd_ids >= config.get_first_heldout_id()
    mask &= bd_ids < config.BACKDOOR_CNT
    return mask


def calc_backdoor_auprc(block, res_type: InfluenceMethod,
                        bd_ids: Tensor, ds_ids: Tensor, inf: Tensor,
                        ep: Optional[int] = None, subepoch: Optional[int] = None,
                        holdout_v_rest: bool = False, ex_id: Optional[int] = None) -> float:
    r""" Calculate the block's AUPRC """
    return _base_roc_calc(is_auroc=False, block=block, res_type=res_type,
                          bd_ids=bd_ids, ds_ids=ds_ids, inf=inf,
                          ep=ep, subepoch=subepoch, holdout_v_rest=holdout_v_rest, ex_id=ex_id)


def _base_roc_calc(is_auroc: bool, block, res_type: InfluenceMethod,
                   bd_ids: Tensor, ds_ids: Tensor, inf: Tensor, holdout_v_rest: bool,
                   ep: Optional[int] = None, subepoch: Optional[int] = None,
                   ex_id: Optional[int] = None) -> float:
    r"""
    Calculate and log the ROC

    :param is_auroc: If \p True, return the AUROC
    :param block: Block of interest
    :param res_type: Result type to be stored
    :param bd_ids: Backdoor IDs used to determine number of backdoor samples
    :param ds_ids: Training example IDs
    :param inf: Corresponding influence values for the list of training ids
    :param ep: If specified, AUROC is reported for a specific epoch
    :param subepoch: If specified, subepoch value
    :param holdout_v_rest: If \p True, test holdout versus the rest of objects
    :return: AUC value
    """
    check_duplicate_ds_ids(ds_ids=ds_ids)
    check_bd_ids_contents(bd_ids=bd_ids)
    assert bd_ids.shape == ds_ids.shape == inf.shape, "IDs and influence values do not match"

    if holdout_v_rest:
        labels = label_holdout(bd_ids=bd_ids)
    else:
        # Filter out the heldout examples
        not_holdout = ~get_holdout_mask(bd_ids=bd_ids)
        bd_ids, ds, inf = bd_ids[not_holdout], ds_ids[not_holdout], inf[not_holdout]
        labels = label_ids(bd_ids=bd_ids)

    if is_auroc:
        # noinspection PyUnresolvedReferences
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, inf)
        # noinspection PyUnresolvedReferences
        roc_val = sklearn.metrics.auc(fpr, tpr)
    else:
        # noinspection PyUnresolvedReferences
        prec, recall, _ = sklearn.metrics.precision_recall_curve(y_true=labels, probas_pred=inf)
        # noinspection PyUnresolvedReferences
        roc_val = sklearn.metrics.average_precision_score(labels, inf)

    header = build_log_start_flds(block=block, ep=ep, subepoch=subepoch, ex_id=ex_id,
                                  res_type=res_type)
    data_name = 'Holdout' if holdout_v_rest else 'Backdoor'
    roc_name = "AUROC" if is_auroc else "AUPRC"
    flds = [header, data_name, f"{roc_name}:", f"{roc_val:.3f}"]
    msg = " ".join(flds)
    logging.info(msg)

    return roc_val


def log_time(res_type: InfluenceMethod):
    r""" Logs the running time of each influence method """
    def decorator(func):
        r""" Need to nest the decorator since decorator takes an argument (\p res_type) """
        def wrapper(*args, **kwargs) -> NoReturn:
            start = time.time()

            rets = func(*args, **kwargs)

            total = time.time() - start
            logging.info(f"{res_type.value} Execution Time: {total:,.3f} seconds")

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


def derivative_of_loss(acts: Tensor, lbls: LongTensor, f_loss: Callable) -> Tensor:
    r"""
    Calculates the dervice of loss function \p f_loss w.r.t. output activation \p outs
    and labels \p lbls
    """
    for tensor, name in ((acts, "acts"), (lbls, "lbls")):
        assert len(tensor.shape) <= 2, f"Unexpected shape for {name}"
    # Need to require gradient to calculate derive
    acts = acts.detach().clone()
    acts.requires_grad = True
    # Calculates the loss
    loss = f_loss(acts, lbls).view([1])
    ones = torch.ones(acts.shape[:1], dtype=acts.dtype, device=acts.device)
    # Back propagate the gradients
    loss.backward(ones)
    return acts.grad.clone().detach().type(acts.dtype)  # type: Tensor
