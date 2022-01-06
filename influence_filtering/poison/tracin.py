__all__ = [
    "calc",
    "log_final",
]

import copy
import logging
import sys
from typing import List, NoReturn, Optional, Tuple, Union

import tqdm

from fastai.basic_data import DeviceDataLoader
import torch
import torch.autograd
from torch import LongTensor, Tensor

from . import _config as config
from . import influence_utils
from .influence_utils import InfluenceMethod
from . import tracin_utils
from . import utils as parent_utils
from .types import CustomTensorDataset

FloatInt = Union[float, int]
OptTensor = Optional[Tensor]


@influence_utils.log_time(res_type=InfluenceMethod.GAS)
def calc(block: parent_utils.ClassifierBlock,
         train_dl: DeviceDataLoader, n_epoch: int,
         wd: Optional[Union[float, List[float]]], bs: Union[int, List[int]],
         x_targ: Tensor, y_targ: LongTensor,
         ex_id: Optional[int] = None) -> tracin_utils.TracInTensors:
    r"""
    Calculate the TracIn influence values.

    :param block: Block being studied
    :param train_dl: Training dataloader
    :param n_epoch: Number of training epochs
    :param wd: Weight decay value(s)
    :param bs: Batch size value(s)
    :param x_targ: Target X value
    :param y_targ: Target y value
    :param ex_id: Optional target example ID number
    :return: Tuple of the influence values, backdoor IDs, and dataset IDs
    """
    # Clean and error check inputs
    assert tracin_utils.START_EPOCH in (0, 1), "Only start epoch of 0 or 1 supported"
    assert x_targ.shape[0] == 1, "Only a single test example can be specified"
    assert y_targ.numel() == 1, "Multiple y labels specified"

    assert not tracin_utils.SPEEDUP or block.best_epoch > 0, \
        "best_epoch = 0 Training only made the loss worse so skipping TracIn"

    if tracin_utils.SPEEDUP and n_epoch != block.best_epoch:
        # Always capture the best epoch epoch even if START_EPOCH == 0 in case we want to
        # post-process that data.
        n_epoch = block.best_epoch
        logging.info(f"# TracIn Epochs: {n_epoch}")

    wd = _clean_hyperparam(wd, n_epoch)
    bs = _clean_hyperparam(bs, n_epoch)
    ep_lst = list(range(tracin_utils.START_EPOCH, n_epoch + 1))
    assert len(ep_lst) == len(wd), "Mismatch in parameter lengths"

    train_dl = tracin_utils.configure_train_dataloader(train_dl=train_dl)
    full_ids, _, id_map, dataset = _get_train_ids(block=block, train_dl=train_dl)

    # Create 32-bit and 64 bit versions of target tensors
    x_targ32, y_targ = _config_targ_tensors(x_targ=x_targ, y_targ=y_targ)
    x_targ64 = x_targ.double().to(parent_utils.TORCH_DEVICE)
    # 32 and 64 bit versions of the devices
    block32, block64 = block, copy.deepcopy(block)
    block64.to(parent_utils.TORCH_DEVICE)

    ep_iters = zip(ep_lst, wd, bs)
    desc = f"{block.name()} Ep"
    ep_tqdm = tqdm.tqdm(ep_iters, total=n_epoch + 1 - tracin_utils.START_EPOCH, file=sys.stdout,
                        desc=desc, disable=config.QUIET)
    # Initialize the results arrays
    all_tensors = tracin_utils.TracInTensors(full_ids=full_ids, inf_numel=config.N_FULL_TR)

    with ep_tqdm as pbar:
        for ep, ep_wd, ep_bs in pbar:
            tracin_utils.all_in.process_epoch(block32=block32, block64=block64,
                                              dataset=dataset, id_map=id_map,
                                              ep=ep, ep_wd=ep_wd, ep_bs=ep_bs,
                                              x_targ32=x_targ32, x_targ64=x_targ64, y_targ=y_targ,
                                              tensors=all_tensors, ex_id=ex_id)

    _log_final_params(block=block, targ_x=x_targ32, targ_y=y_targ, tensors=all_tensors,
                      ex_id=ex_id)
    # # Log final is placed outside of the method in the filter_end_to_end run method to allow
    # # for more accurate estimation of the TracIn function's run time.
    # log_final(block=block, tensors=all_tensors, train_dl=train_dl, ex_id=ex_id)

    # Restore the best model configuration
    block.restore_best()
    block.eval()
    return all_tensors


def _get_train_ids(block: parent_utils.ClassifierBlock, train_dl: DeviceDataLoader) \
        -> Tuple[LongTensor, LongTensor, dict, CustomTensorDataset]:
    r"""
    Gets the training set IDs (backdoor and original)
    :returns: Tuple of the backdoor IDs, dataset ID, and ID map that maps dataset ID number
              to the index in the dataset
    """
    all_ids, all_lbls, id_map = [], [], dict()
    # noinspection PyUnresolvedReferences
    ds = CustomTensorDataset(tensors=train_dl.dl.dataset.tensors, transform=config.get_test_tfms())
    for cnt in range(len(ds)):
        batch_tensors = ds[[cnt]]
        batch = block.organize_batch(batch_tensors)
        if batch.skip():
            continue

        all_ids.append(batch.ids)
        all_lbls.append(batch.lbls)
        id_map[batch.ids.item()] = cnt

    ids, lbls = torch.cat(all_ids, dim=0), torch.cat(all_lbls, dim=0)

    return ids, lbls, id_map, ds  # noqa


def _config_targ_tensors(x_targ: Tensor, y_targ: LongTensor) -> Tuple[Tensor, LongTensor]:
    r""" Configures the X/y target tensors including applying any needed transforms """
    assert torch.min(x_targ).item() >= 0, "Target tensor already transformed"
    test_tfms = config.get_test_tfms()
    if test_tfms is not None:
        x_targ = test_tfms(x_targ)
    return x_targ.to(parent_utils.TORCH_DEVICE), y_targ.to(parent_utils.TORCH_DEVICE)


def _clean_hyperparam(_param: Union[FloatInt, List[FloatInt]],
                      n_epoch: int) -> List[Optional[FloatInt]]:
    r""" Cleans the hyperparameter setting information for standard interface"""
    if isinstance(_param, (float, int)) or _param is None:
        return [_param] * (n_epoch + 1 - tracin_utils.START_EPOCH)
    # assert len(_param) == n_epoch + 1, "No epoch 0 data in hyperparameter list"
    # Shave off as n_epoch can be set to best_epoch
    return _param[tracin_utils.START_EPOCH:n_epoch + 1]


def log_final(block: parent_utils.ClassifierBlock, train_dl: DeviceDataLoader,
              tensors: tracin_utils.TracInTensors, ex_id: Optional[int]) -> NoReturn:
    r""" Log the final results """
    # Log number of underflows in total and for specifically adversarial backdoors
    header = influence_utils.build_log_start_flds(block=block, res_type=None, ep=None,
                                                  subepoch=None, ex_id=ex_id)
    cumul_tot = torch.sum(tensors.tot_zeros)
    logging.info(f"{header} Total Underflow Cumulative Count: {cumul_tot}")


def _log_targ_label_info(block, tensors: tracin_utils.TracInTensors,
                         ex_id: Optional[int]) -> NoReturn:
    r""" Log final parameters, e.g. target loss """
    header = influence_utils.build_log_start_flds(block=block, ep=None, subepoch=None,
                                                  res_type=None, ex_id=ex_id)
    logging.info(f"{header} {InfluenceMethod.LAST_TARG_TOGGLE.value}: {tensors.last_targ_toggle}")
    logging.info(f"{header} {InfluenceMethod.NUM_TARG_TOGGLE.value}: {tensors.targ_lbl_flips}")
    p_toggle = _calc_toggle_prob(block=block, n_toggle=tensors.targ_lbl_flips)
    logging.info(f"{header} {InfluenceMethod.PR_TARG_TOGGLE.value}: {p_toggle}")


def _calc_toggle_prob(block, n_toggle: Union[int, LongTensor]) -> Union[float, Tensor]:
    r""" Normalizes number of toggles by number of train epochs since early stopping used"""
    denom = config.NUM_SUBEPOCH * block.best_epoch - 1  # Sub 1 since toggle in first (sub)epoch
    if isinstance(n_toggle, int):
        return n_toggle / denom
    # LongTensor case
    n_toggle = n_toggle.type(tracin_utils.DTYPE).div_(denom)  # noqa
    return n_toggle


def _log_final_params(block, targ_x: Tensor, targ_y: LongTensor,
                      tensors: tracin_utils.TracInTensors, ex_id: Optional[int]) -> NoReturn:
    r""" Log final parameters, e.g. target loss """
    tracin_utils.all_in.log_target_stats(block=block, ep=None, subepoch=None,
                                         x_targ=targ_x, y_targ=targ_y, tensors=tensors,
                                         ex_id=ex_id, grad_targ=None)


def _log_magnitude_ratio(block: parent_utils.ClassifierBlock, tensors: tracin_utils.TracInTensors,
                         ex_id: Optional[int]) -> NoReturn:
    r""" Log statistics of the magnitude ratio information """
    # Group magnitude data into a single tensor for statistics extraction
    all_magnitude_ratio = torch.cat(tensors.magnitude_ratio, dim=0)

    tracin_utils.log_vals_stats(block=block, ep=None, subep=None,
                                res_type=InfluenceMethod.GRAD_NORM_MAG_RATIO,
                                norms=all_magnitude_ratio, ex_id=ex_id)
