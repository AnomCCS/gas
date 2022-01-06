__all__ = [
    "calculate_results",
]

from dataclasses import dataclass
import io
import logging
import sys
from typing import NoReturn, Tuple

import numpy as np
import pycm

from fastai.basic_data import DeviceDataLoader
import torch

from . import _config as config
from . import influence_utils
from .types import CustomTensorDataset, TensorGroup
from . import utils

BATCH_SIZE_MULTIPLIER = 5  # Since not using gradients can have much larger batches

TE_CLN_DS = "test"
TE_ADV_DS = "test-adv"
TE_ADV_ONLY = "test-only-bd"


@dataclass(init=True, order=True)
class DatasetResult:
    r""" Encapsulates results of a model on a SINGLE dataset """
    ds_size: int
    acc: float = None


@dataclass(order=True)
class LearnerResults:
    r""" Encapsulates ALL results for a single NLP learner model """
    loss_name = None
    valid_loss = None
    targ_loss = None


def calculate_results(tg: TensorGroup, block: utils.ClassifierBlock, is_pretrain: bool = False,
                      log_targ: bool = True) -> LearnerResults:
    r"""
    Calculates and writes to disk the model's results

    :param tg: Tensor group containing the test conditions
    :param block: Trained block
    :param is_pretrain: If \p True, use the pretrain data
    :param log_targ: Log the target's information
    :return: Dictionary containing results of all experiments
    """
    block.eval()
    assert not is_pretrain or not log_targ, "Cannot log target for pretrain model"

    ds_flds = _build_ds_fields(tg=tg, is_pretrain=is_pretrain, log_targ=log_targ)

    res = LearnerResults()
    res.loss_name = block.name()
    res.valid_loss = block.best_loss

    if log_targ:
        _get_target_losses(res=res, tg=tg, block=block)

    for ds_name, tensors in ds_flds:
        ds = CustomTensorDataset(tensors, config.get_test_tfms())
        # noinspection PyTypeChecker
        dl = DeviceDataLoader.create(ds, shuffle=False, drop_last=False,
                                     bs=BATCH_SIZE_MULTIPLIER * config.BATCH_SIZE,
                                     num_workers=0, device=utils.TORCH_DEVICE)
        all_y, all_yhat = [], []
        for xs, ys in dl:
            all_y.append(ys.cpu())
            with torch.no_grad():
                all_yhat.append(block.module.predict(xs).cpu())

        # Iterator transforms label so transform it back
        y = torch.cat(all_y, dim=0).view([-1]).numpy()
        y_hat = torch.cat(all_yhat, dim=0).view([-1]).numpy()
        # Store for name "unlabel" or "test"
        single_res = _single_ds_results(block.name(), ds_name, y, y_hat)
        res.__setattr__(ds_name, single_res)

    # Append the result
    _log_validation_loss(block=block, res=res, is_pretrain=is_pretrain)
    return res


def _single_ds_results(block_name: str, ds_name: str, y: np.ndarray,
                       y_hat: np.ndarray) -> DatasetResult:
    r""" Logs and returns the results on a single dataset """
    res = DatasetResult(y.shape[0])

    str_prefix = f"{block_name} {ds_name}:"

    logging.debug(f"{str_prefix} Dataset Size: {res.ds_size:,}")
    # Pre-calculate fields needed in other calculations
    conf_matrix = pycm.ConfusionMatrix(actual_vector=y, predict_vector=y_hat)

    # noinspection PyUnresolvedReferences
    res.acc = conf_matrix.Overall_ACC
    logging.debug(f"{str_prefix} Accuracy: {100. * res.acc:.3}%")

    # Write confusion matrix to a string
    sys.stdout = cm_out = io.StringIO()
    conf_matrix.print_matrix()
    sys.stdout = sys.__stdout__
    # Log the confusion matrix
    cm_str = cm_out.getvalue()
    logging.debug(f"{str_prefix} Confusion Matrix: \n{cm_str}")

    return res


def _build_ds_fields(tg: TensorGroup, is_pretrain: bool, log_targ: bool):
    r""" Builds dataset fields.  Poisoned data my not be included if it has not been created yet """
    if is_pretrain:
        return [("pretr", (tg.pretr_x, tg.pretr_y)), ("preval", (tg.preval_x, tg.preval_y)),
                ("pretest", (tg.pretest_x, tg.pretest_y))]

    ds_flds = [("tr", (tg.tr_x, tg.tr_y)),
               ("val", (tg.val_x, tg.val_y))]
    if log_targ:
        ds_flds.append(("targ", (tg.targ_x, tg.targ_y)))
    ds_flds.append(("test", (tg.test_x, tg.test_y)))

    mask = tg.test_y == config.TARG_CLS
    ds_flds.append(("te_targ_cls", (tg.test_x[mask], tg.test_y[mask])))

    return tuple(ds_flds)


def _get_target_losses(res: LearnerResults, tg: TensorGroup,
                       block: utils.ClassifierBlock) -> NoReturn:
    r""" Logs and stores the target loss values """
    x = config.get_test_tfms()(tg.targ_x).to(utils.TORCH_DEVICE)
    targ_y = tg.targ_y.to(utils.TORCH_DEVICE)

    # Final weights determine final losses
    block.restore_best()
    block.eval()
    with torch.no_grad():
        targ_scores = block.forward(x)

    res.targ_loss = block.loss.calc_validation_loss(dec_scores=targ_scores, labels=targ_y).item()
    logging.info(f"{block.name()} ID {config.TARG_IDX} Loss: {res.targ_loss:.3E}")


def _extract_targ_pois_idx(conf_matrix: pycm.ConfusionMatrix) -> Tuple[int, int]:
    r"""
    Extracts the index of the target and poison classes respectively from the confusion matrix's
    classes.  If the class number does not appear in the class list, -1 is returned.
    """
    classes = conf_matrix.classes
    idx_lst = []
    for cls_id in (config.TARG_CLS, config.POIS_CLS):
        for i, val in enumerate(classes):
            try:
                val = int(val)
            except ValueError:  # Cannot convert val to string
                continue
            if int(val) == cls_id:
                idx_lst.append(i)
                break
        else:
            idx_lst.append(-1)
    return tuple(idx_lst)  # noqa


def _log_validation_loss(block: utils.ClassifierBlock, res: LearnerResults,
                         is_pretrain: bool) -> NoReturn:
    r""" Log the validation loss for subsequent analysis """
    header = influence_utils.build_log_start_flds(block=block, ep=None, subepoch=None,
                                                  res_type=None, ex_id=None)
    pretrain_desc = " Pretrain" if is_pretrain else ""
    logging.info(f"{header}{pretrain_desc} Validation Loss: {res.valid_loss:.6E}")
