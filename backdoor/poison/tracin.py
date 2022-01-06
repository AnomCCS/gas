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
         ex_ids: List[int]) -> tracin_utils.TracInTensors:
    r"""
    Calculate the TracIn influence values.

    :param block: Block being studied
    :param train_dl: Training dataloader
    :param n_epoch: Number of training epochs
    :param wd: Weight decay value(s)
    :param bs: Batch size value(s)
    :param x_targ: Target X value
    :param y_targ: Target y value
    :param ex_ids: Optional target example ID number
    :return: Tuple of the influence values, backdoor IDs, and dataset IDs
    """
    # Clean and error check inputs
    assert tracin_utils.START_EPOCH in (0, 1), "Only start epoch of 0 or 1 supported"
    # assert x_targ.shape[0] == 1, "Only a single test example can be specified"
    # assert y_targ.numel() == 1, "Multiple y labels specified"

    assert not tracin_utils.SPEEDUP or block.best_epoch > 0, \
        "best_epoch = 0 Training only made the loss worse so skipping TracIn"

    if tracin_utils.SPEEDUP and n_epoch != block.best_epoch:
        # Always capture the best epoch even if START_EPOCH == 0 in case we want to
        # post-process that data.
        n_epoch = block.best_epoch
        logging.info(f"# TracIn Epochs: {n_epoch}")

    wd = _clean_hyperparam(wd, n_epoch)
    bs = _clean_hyperparam(bs, n_epoch)
    ep_lst = list(range(tracin_utils.START_EPOCH, n_epoch + 1))
    assert len(ep_lst) == len(wd), "Mismatch in parameter lengths"

    train_dl = tracin_utils.configure_train_dataloader(train_dl=train_dl)
    full_bd_ids, full_ds_ids, _, id_map, dataset = _get_train_ids(block=block, train_dl=train_dl)

    # Create 32-bit and 64 bit versions of target tensors
    x_targ32, y_targ = _config_targ_tensors(x_targ=x_targ, y_targ=y_targ)
    x_targ64 = x_targ32.double().to(parent_utils.TORCH_DEVICE)
    # 32 and 64 bit versions of the devices
    block32, block64 = block, copy.deepcopy(block)
    block64.to(parent_utils.TORCH_DEVICE)

    ep_iters = zip(ep_lst, wd, bs)
    desc = f"{block.name()} Ep"
    ep_tqdm = tqdm.tqdm(ep_iters, total=n_epoch + 1 - tracin_utils.START_EPOCH, file=sys.stdout,
                        desc=desc, disable=config.QUIET)
    # Initialize the results arrays
    all_tensors = tracin_utils.TracInTensors(full_ds_ids=full_ds_ids, full_bd_ids=full_bd_ids,
                                             id_numel=len(ex_ids), inf_numel=config.N_FULL_TR)

    with ep_tqdm as pbar:
        for ep, ep_wd, ep_bs in pbar:
            tracin_utils.all_in.process_epoch(block32=block32,
                                              block64=block64,
                                              dataset=dataset, id_map=id_map,
                                              ep=ep, ep_wd=ep_wd, ep_bs=ep_bs,
                                              x_targ32=x_targ32,
                                              x_targ64=x_targ64,
                                              y_targ=y_targ,
                                              tensors=all_tensors, ex_ids=ex_ids)

    # Restore the best model configuration
    block.restore_best()
    block.eval()
    return all_tensors


def _get_train_ids(block: parent_utils.ClassifierBlock, train_dl: DeviceDataLoader) \
        -> Tuple[LongTensor, LongTensor, LongTensor, dict, CustomTensorDataset]:
    r"""
    Gets the training set IDs (backdoor and original)
    :returns: Tuple of the backdoor IDs, dataset ID, and ID map that maps dataset ID number
              to the index in the dataset
    """
    all_ds_ids, all_bd_ids, all_lbls, id_map = [], [], [], dict()
    # noinspection PyUnresolvedReferences
    ds = CustomTensorDataset(tensors=train_dl.dl.dataset.tensors, transform=config.get_test_tfms())
    for cnt in range(len(ds)):
        batch_tensors = ds[[cnt]]
        batch = block.organize_batch(batch_tensors, process_mask=True, include_holdout=True)
        if batch.skip():
            continue

        all_bd_ids.append(batch.bd_ids)
        all_ds_ids.append(batch.ds_ids)
        all_lbls.append(batch.lbls)
        id_map[batch.ds_ids.item()] = cnt

    bd_ids, ds_ids = torch.cat(all_bd_ids, dim=0), torch.cat(all_ds_ids, dim=0)
    lbls = torch.cat(all_lbls, dim=0)
    influence_utils.check_bd_ids_contents(bd_ids=bd_ids)
    influence_utils.check_duplicate_ds_ids(ds_ids=ds_ids)
    assert bd_ids.shape == ds_ids.shape

    return bd_ids, ds_ids, lbls, id_map, ds  # noqa


def _config_targ_tensors(x_targ: Tensor, y_targ: LongTensor) -> Tuple[Tensor, LongTensor]:
    r""" Configures the X/y target tensors including applying any needed transforms """
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


def log_final(block: parent_utils.ClassifierBlock,
              tensors: tracin_utils.TracInTensors, ex_ids: List[int]) -> NoReturn:
    r""" Log the final results """
    full_ds_ids, full_bd_ids = tensors.full_ds_ids, tensors.full_bd_ids

    for idx in range(len(ex_ids)):
        tmp_ex_id = ex_ids[idx]

        def _log_result(inf_vals: Tensor, method: InfluenceMethod,
                        log_stats: bool = False) -> NoReturn:
            if isinstance(inf_vals, LongTensor):
                inf_vals = inf_vals.type(tracin_utils.DTYPE)  # noqa
            _log_result_master(block=block, log_stats=log_stats,
                               inf_vals=inf_vals, full_ds_ids=full_ds_ids, full_bd_ids=full_bd_ids,
                               method=method, ex_id=tmp_ex_id)

        log_func_list = (
            tracin_utils.get_gas_log_flds,  # GAS Results
            tracin_utils.get_tracin_log_flds,
            tracin_utils.get_tracincp_log_flds,
        )
        for log_flds_func in log_func_list:
            for inf, _method, _name in log_flds_func(tensors=tensors):
                _log_result(inf[idx], method=_method)

    logging.info("")  # Dummy line to improve readability


def _log_result_master(block: parent_utils.ClassifierBlock,
                       full_ds_ids: LongTensor, full_bd_ids: LongTensor,
                       inf_vals: Tensor, method: InfluenceMethod,
                       log_stats: bool, ex_id: Optional) -> NoReturn:
    tracin_utils.results.generate_epoch_stats(ep=None, subepoch=None,
                                              block=block, method=method,
                                              inf_vals=inf_vals[full_ds_ids],
                                              ds_ids=full_ds_ids, bd_ids=full_bd_ids,
                                              ex_id=ex_id, log_cutoff=True)
    if log_stats:
        tracin_utils.log_vals_stats(block=block, res_type=method, ep=None, subep=None,
                                    norms=inf_vals[full_ds_ids], ex_id=ex_id)
