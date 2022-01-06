__all__ = [
    "process_epoch",
]

import logging
import sys
from typing import List, NoReturn, Optional

import torch
import tqdm
from torch import LongTensor, Tensor

from . import _settings as settings
from . import layer_norm_analysis
from . import results
from . import utils
from .. import _config as config
from .. import influence_utils
from ..influence_utils import InfluenceMethod
from .. import types as parent_types
from .. import utils as parent_utils

LOG_LAYER_NORM_STATS = True


def process_epoch(block32: parent_utils.ClassifierBlock,
                  block64: parent_utils.ClassifierBlock,
                  dataset: parent_types.CustomTensorDataset,
                  id_map: dict, ep: int, ep_wd: float, ep_bs: int,
                  x_targ32: Tensor,
                  x_targ64: Tensor,
                  y_targ: LongTensor,
                  tensors: utils.TracInTensors, ex_ids: List[int]) -> NoReturn:
    r"""
    Performs TracIn on a single epoch (including subepochs if applicable) for the specified
    \p block

    :param block32: Block for use with floats (i.e., float32)
    :param block64: Block for use with doubles (i.e., float64)
    :param dataset: Dataset object of interest
    :param id_map: Maps example ID to dataset index
    :param ep: Active epoch number
    :param ep_wd: Epoch weight decay value
    :param ep_bs: Epoch batch_size value
    :param x_targ32: Already transformed X-target
    :param y_targ: y of target example
    :param tensors: All results tensors
    :param ex_ids: Optional target example ID number
    :return:
    """
    assert isinstance(dataset, parent_types.CustomTensorDataset), "Dataset class is not supported"
    influence_utils.check_duplicate_ds_ids(ds_ids=tensors.full_ds_ids)
    influence_utils.check_bd_ids_contents(bd_ids=tensors.full_bd_ids)

    def _load_blocks(_ep: int, _subep: Optional[int]):
        r""" Standardizes loading the block parameters """
        for block in block32, block64:
            block.restore_epoch_params(ep=_ep, subepoch=_subep)
            block.eval()
        block64.double()

    cur_subep = 0
    _load_blocks(_ep=ep - 1, _subep=None)
    # Continue learning rate from end of the last epoch
    lr = block32.get_prev_lr_val(ep=ep - 1, subepoch=None)

    # Epoch dataset IDs ordered by batches
    ep_ds_ids = block32.get_epoch_dataset_ids(ep=ep)
    subep_ends = block32.get_subepoch_ends(ep=ep)
    n = len(id_map)

    # Iterate through the subepochs
    n_subep = len(block32.get_subepoch_ends(ep=ep))
    tqdm_desc = f"Ep {ep} Subep %d"
    subep_tqdm = tqdm.tqdm(range(n_subep + 1), desc=tqdm_desc % cur_subep, file=sys.stdout,
                           total=n_subep + 1, disable=config.QUIET)
    for cur_subep in subep_tqdm:
        subep_tqdm.set_description(tqdm_desc % cur_subep)

        # Get subepoch IDs for TracIn
        start_rng = subep_ends[cur_subep - 1] if cur_subep > 0 else 0
        end_rng = subep_ends[cur_subep] if cur_subep < len(subep_ends) - 1 else n
        subep_ids = ep_ds_ids[start_rng:end_rng]

        # Subepoch used to load stored data
        subep_load = cur_subep if cur_subep < n_subep else None

        # Get the loss gradient for the test (target) example
        # Calculate 32-bit first to ensure more efficient memory
        _, _, grad_targ32 = utils.compute_grad(block32, ep_wd, x_targ32, y_targ)
        num_layers = len(grad_targ32[0])
        grad32_layer = utils.build_layer_norm(grad_targ32)
        grad_targ32 = utils.flatten_grad(grad_targ32)  # noqa
        # Calculate 64bit if underflow
        _, _, grad_targ64 = utils.compute_grad(block64, ep_wd, x_targ64, y_targ)
        grad64_layer = utils.build_layer_norm(grad_targ64)
        for idx, ex_id in enumerate(ex_ids):
            if not influence_utils.check_is_bd(id_val=ex_id):
                continue
            layer_norm_analysis.log_stats(block=block64, ep=ep, subep=cur_subep,
                                          grad_x=grad_targ64[idx], ex_id=ex_id)
        grad_targ64 = utils.flatten_grad(grad_targ64)  # noqa

        # Skip iter if target has zero gradient even at double precision
        ex_desc = f"{block32.name()} Ep {ep} Sub: {cur_subep} Train Ex."
        ex_tqdm = tqdm.tqdm(tensors.full_ds_ids, total=tensors.full_ds_ids.shape[0],
                            file=sys.stdout, disable=config.QUIET, desc=ex_desc)
        with ex_tqdm as ex_bar:
            tensors.subep.reset()
            for cnt, id_val in enumerate(ex_bar):
                utils.tracin_dot_product(block32=block32, block64=block64,
                                         grad_targ32=grad_targ32, grad_targ64=grad_targ64,
                                         grad32_layer=grad32_layer,
                                         grad64_layer=grad64_layer,
                                         subep_tensors=tensors.subep,
                                         ds=dataset, id_val=id_val, id_map=id_map,
                                         ep_wd=ep_wd)

        # Perform normalization based on learning rate and batch size as specified by TracIn
        # Done all at once externally to allow for faster CUDA parallelism
        tensors.subep.dot_vals *= lr / ep_bs
        # tensors.subep.grain_vals *= lr / ep_bs
        tensors.subep.gas_vals *= lr / ep_bs
        # Perform the sqrt externally to use CUDA
        tensors.subep.grad_norms.sqrt_()

        _combine_and_log_results(block=block64, ep=ep, subepoch=cur_subep, lr=lr, bs=ep_bs,
                                 grad_targ=grad_targ64, num_layers=num_layers,
                                 subep_ids=subep_ids, tensors=tensors, ex_ids=ex_ids)

        # Load parameters and learning rate for the next (sub)epoch
        _load_blocks(_ep=ep, _subep=subep_load)
        lr = block32.get_prev_lr_val(ep=ep, subepoch=subep_load)

    subep_tqdm.close()


def _combine_and_log_results(block: parent_utils.ClassifierBlock, ep: int, subepoch: int,
                             lr: float, bs: int,
                             grad_targ: Tensor,
                             subep_ids: LongTensor,
                             num_layers: int,
                             tensors: utils.TracInTensors,
                             ex_ids: List[int]) -> NoReturn:
    r""" Combines and logs all results """
    full_ds_ids, full_bd_ids = tensors.full_ds_ids, tensors.full_bd_ids
    tensors.subep.dot_normed = tensors.subep.dot_vals / tensors.subep.grad_norms

    for idx in range(len(ex_ids)):
        tmp_ex_id = ex_ids[idx]  # Actual ID value

        # Log test results
        def _log_result(inf_vals: Tensor, method: InfluenceMethod,
                        log_cutoff: bool = False) -> NoReturn:
            results.generate_epoch_stats(ep=ep, subepoch=subepoch, block=block, method=method,
                                         inf_vals=inf_vals[full_ds_ids],
                                         ds_ids=full_ds_ids, bd_ids=full_bd_ids,
                                         ex_id=tmp_ex_id, log_cutoff=log_cutoff)

        # Equivalent of TracInCp
        tensors.tracincp_inf[idx, full_ds_ids] += tensors.subep.dot_vals[idx, full_ds_ids]
        for _inf, _method, _ in utils.get_tracincp_log_flds(tensors=tensors):
            _log_result(_inf[idx], method=_method)

        targ_grad_norm = grad_targ[idx].norm()
        targ_grad_norm[targ_grad_norm <= settings.MIN_NORM] = settings.MIN_NORM
        # GAS Results
        gas_sim_base = tensors.subep.dot_normed[idx] / targ_grad_norm.cpu()
        tensors.gas_sim[idx, full_ds_ids] += gas_sim_base[full_ds_ids]
        # GAS with layerwise norm
        tensors.gas_l_sim[idx, full_ds_ids] += tensors.subep.gas_vals[idx, full_ds_ids]
        for _inf, _method, _ in utils.get_gas_log_flds(tensors=tensors):
            _log_result(_inf[idx], method=_method)

        # TracIn Results
        tensors.tracin_inf[idx, subep_ids] += tensors.subep.dot_vals[idx, subep_ids]
        # TracIn normalized by L2 gas norm
        tensors.tracin_sim[idx, subep_ids] += gas_sim_base[subep_ids]
        for _inf, _method, _ in utils.get_tracin_log_flds(tensors=tensors):
            _log_result(_inf[idx], method=_method)

        # Logs and accessors the adversarial and clean magnitude ratios
        _log_ratio_stats(block=block, ep=ep, subep=subepoch,
                         full_ds_ids=full_ds_ids, full_bd_ids=full_bd_ids,
                         vals=tensors.subep.loss_vals[idx], ex_id=tmp_ex_id, is_grad_norm=False)
        # Logs and accessors the adversarial and clean magnitude ratios
        _log_ratio_stats(block=block, ep=ep, subep=subepoch,
                         full_ds_ids=full_ds_ids, full_bd_ids=full_bd_ids,
                         vals=tensors.subep.grad_norms[idx], ex_id=tmp_ex_id, is_grad_norm=True)

    logging.info("")  # Dummy line to improve readability


def _log_ratio_stats(block: parent_utils.ClassifierBlock, ep: int, subep: int,
                     vals: Tensor, full_ds_ids: LongTensor,
                     full_bd_ids: LongTensor, ex_id: Optional[int],
                     is_grad_norm: bool) -> NoReturn:
    r""" Calculates and returns the adversarial and clean mean norms respectively """
    assert full_bd_ids.numel() == full_ds_ids.numel(), "Backdoor/dataset length mismatch"
    # Extract only the relevant cumulative IDs
    assert vals.numel() > torch.max(full_ds_ids).item(), "Some dataset ID not found"
    vals = vals[full_ds_ids]

    # Label whether each example is a backdoor or not
    is_bd = influence_utils.is_bd(bd_ids=full_bd_ids)

    adv_vals, cl_vals = vals[is_bd], vals[~is_bd]
    if not is_grad_norm:
        res_types = (InfluenceMethod.LOSS_CLEAN_SPOT, InfluenceMethod.LOSS_ADV_SPOT)
        for vals, r_type in zip((cl_vals, adv_vals), res_types):
            utils.log_vals_stats(block=block, ep=ep, subep=subep, res_type=r_type, norms=vals,
                                 ex_id=ex_id)
    else:
        ratio_res_type = InfluenceMethod.GRAD_NORM_MAG_RATIO
        # Log mean and median ratios for clear documentation
        header = influence_utils.build_log_start_flds(block=block, ep=ep, subepoch=subep,
                                                      res_type=ratio_res_type, ex_id=ex_id)
        median_mag_ratio = (adv_vals.median() / cl_vals.median()).view([1])
        logging.info(f"{header} Median: {median_mag_ratio.item():.3E}")
