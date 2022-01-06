__all__ = [
    "log_target_stats",
    "process_epoch",
]

import logging
import sys
from typing import NoReturn, Optional, Tuple

import torch
import tqdm
from torch import DoubleTensor, LongTensor, Tensor

from . import _settings as settings
from . import utils
from .. import _config as config
from .. import dirs
from .. import influence_utils
from .. import types as parent_types
from .. import utils as parent_utils


def process_epoch(block32: parent_utils.ClassifierBlock, block64: parent_utils.ClassifierBlock,
                  dataset: parent_types.CustomTensorDataset,
                  id_map: dict, ep: int, ep_wd: float, ep_bs: int,
                  x_targ32: Tensor, x_targ64: Tensor, y_targ: LongTensor,
                  tensors: utils.TracInTensors, ex_id: Optional[int]) -> NoReturn:
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
    :param x_targ64: Already transformed X-target
    :param y_targ: y of target example
    :param tensors: All results tensors
    :param ex_id: Optional target example ID number
    :return:
    """
    assert isinstance(dataset, parent_types.CustomTensorDataset), "Dataset class is not supported"

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
    ep_ds_ids = block32.get_epoch_ids(ep=ep)
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

        # Initialize the tensors storing the results from the subepoch
        tensors.subep.reset()

        # Get the loss gradient for the test (target) example
        _, _, grad_targ32 = utils.compute_grad(block32, ep_wd, x_targ32, y_targ, flatten=False)
        grad32_layer = utils.build_layer_norm(grad_targ32)
        loss_targ64, acts_targ64, grad_targ64 = utils.compute_grad(block64, ep_wd,
                                                                   x_targ64, y_targ,
                                                                   flatten=False)
        grad64_layer = utils.build_layer_norm(grad_targ64)
        grad_targ32 = utils.flatten_grad(grad_targ32)
        grad_targ64 = utils.flatten_grad(grad_targ64)

        always_use_dbl = utils.is_grad_zero(grad=grad_targ32)
        if always_use_dbl:
            header = influence_utils.build_log_start_flds(block=block32, ep=ep, subepoch=cur_subep,
                                                          res_type=None, ex_id=ex_id)
            logging.info(f"{header}: Target underflow at 32bit")
        # Skip iter if target has zero gradient even at double precision
        skip_iter = always_use_dbl and utils.is_grad_zero(grad=grad_targ64)

        ex_desc = f"{block32.name()} Ep {ep} Sub: {cur_subep} Train Ex."
        ex_tqdm = tqdm.tqdm(tensors.full_ids, total=tensors.full_ids.shape[0],
                            file=sys.stdout, disable=config.QUIET, desc=ex_desc)
        with ex_tqdm as ex_bar:
            if not skip_iter:  # skip if targ grad is zero
                for cnt, id_val in enumerate(ex_bar):
                    utils.tracin_dot_product(block32=block32,
                                             grad_targ32=grad_targ32,
                                             grad32_layer=grad32_layer,
                                             block64=block64,
                                             grad_targ64=grad_targ64,
                                             grad64_layer=grad64_layer,
                                             subep_tensors=tensors.subep,
                                             ds=dataset, id_val=id_val, id_map=id_map,
                                             ep_wd=ep_wd, always_use_dbl=always_use_dbl)
            else:
                loss_targ64.fill_(settings.MIN_LOSS)
                acts_targ64.fill_(settings.MIN_LOSS)
                # Prevent divide by zero errors in later calculations
                tensors.subep.grad_norms.fill_(settings.MIN_NORM)
                logging.info(f"{header}: Target underflow at 64bit")  # noqa

        # Perform normalization based on learning rate and batch size as specified by TracIn
        # Perform externally to make faster with CUDA
        tensors.subep.dot_vals *= lr / ep_bs
        tensors.subep.gas_vals *= lr / ep_bs
        # Perform the sqrt externally to use CUDA
        tensors.subep.grad_norms.sqrt_()

        targ_pred, _ = log_target_stats(block=block64, ep=ep, subepoch=cur_subep,
                                        grad_targ=grad_targ64, x_targ=x_targ64, y_targ=y_targ,
                                        tensors=tensors, ex_id=ex_id)
        _log_target_grad_norm(block=block64, ep=ep, subepoch=cur_subep,
                              grad_targ=grad_targ64, ex_id=ex_id)
        _log_learning_rate(block=block64, ep=ep, subepoch=cur_subep, lr=lr, ex_id=ex_id)

        _combine_and_log_results(grad_targ=grad_targ64, subep_ids=subep_ids, tensors=tensors)

        # Load parameters and learning rate for the next (sub)epoch
        _load_blocks(_ep=ep, _subep=subep_load)
        lr = block64.get_prev_lr_val(ep=ep, subepoch=subep_load)

    subep_tqdm.close()


def _log_target_grad_norm(block, ep: int, subepoch: int, grad_targ: Tensor,
                          ex_id: Optional[int] = None) -> NoReturn:
    r""" Log the target's gradient norm """
    targ_norm = torch.norm(grad_targ)
    _log_val_base(block=block, ep=ep, subepoch=subepoch, ex_id=ex_id, name_val="Targ Grad Norm",
                  val_str=f"{targ_norm:.8E}")


def _log_learning_rate(block, ep: int, subepoch: int, lr: float,
                       ex_id: Optional[int] = None) -> NoReturn:
    r""" Log the target's gradient norm """
    _log_val_base(block=block, ep=ep, subepoch=subepoch, ex_id=ex_id, name_val="Learning Rate",
                  val_str=f"{lr:.4E}")


def log_target_stats(block, ep: Optional[int], subepoch: Optional[int],
                     grad_targ: Optional[DoubleTensor], x_targ: Tensor, y_targ: LongTensor,
                     tensors: utils.TracInTensors, ex_id: Optional[int]) -> Tuple[int, Tensor]:
    r""" Log the target's gradient norm """
    with torch.no_grad():
        # Use predict function as it has special checks for classes that do not use generic
        # argmax (e.g., for a single output binary classification)
        pred = block.module.predict(x_targ).cpu().item()
    _log_val_base(block=block, ep=ep, subepoch=subepoch, ex_id=ex_id, name_val="Targ Pred",
                  val_str=str(pred))

    with torch.no_grad():
        dec_score = block.forward(x_targ)
        loss = block.loss.calc_train_loss(dec_score, y_targ).cpu().item()
    _log_val_base(block=block, ep=ep, subepoch=subepoch, ex_id=ex_id, name_val="Targ Loss",
                  val_str=f"{loss:.8E}")

    if config.PLOT and grad_targ is not None:
        _plot_targ_grad_hist(grad_targ=grad_targ, ep=ep, subep=subepoch, ex_id=ex_id)
    return pred, loss


def _log_val_base(block, ep: int, subepoch: int, ex_id: Optional[int], name_val: str,
                  val_str: str) -> NoReturn:
    r""" Log the values from the execution """
    header = influence_utils.build_log_start_flds(block=block, ep=ep, subepoch=subepoch,
                                                  res_type=None, ex_id=ex_id)
    logging.info(f"{header} {name_val}: {val_str}")


def _combine_and_log_results(grad_targ: Tensor, subep_ids: LongTensor,
                             tensors: utils.TracInTensors) -> NoReturn:
    r""" Combines and logs all results """
    full_ids = tensors.full_ids
    tensors.subep.dot_normed = tensors.subep.dot_vals / tensors.subep.grad_norms

    tensors.tracincp[full_ids] += tensors.subep.dot_vals[full_ids]

    # GAS Results
    targ_grad_norm = grad_targ.norm().item()
    if targ_grad_norm == 0:
        targ_grad_norm = settings.MIN_NORM
    gas_sim_base = tensors.subep.dot_normed / targ_grad_norm
    subep_gas = gas_sim_base[full_ids]
    tensors.gas_sim[full_ids] += subep_gas
    # GAS with layerwise norm
    tensors.gas_l[full_ids] += tensors.subep.gas_vals[full_ids]

    # TracIn Results
    tensors.tracin_inf[subep_ids] += tensors.subep.dot_vals[subep_ids]
    # TracIn normalized by L2 gas norm
    tensors.tracin_sim[subep_ids] += gas_sim_base[subep_ids]


def _plot_targ_grad_hist(grad_targ: DoubleTensor, ep: int, subep: Optional[int],
                         ex_id: Optional[int], n_bins: int = 500) -> NoReturn:
    grad_targ = grad_targ.abs()
    # Construct the epoch string
    ep_str = influence_utils.construct_ep_str(ep=ep, subepoch=subep)

    out_dir = dirs.PLOTS_DIR / config.DATASET.value.name.lower() / "targ-grad-hist"
    file_ep_str = ep_str.replace(" ", "=").lower()
    path = parent_utils.construct_filename(f"targ-grad-hist_ep={file_ep_str}", ex_id=ex_id,
                                           out_dir=out_dir, file_ext="png", add_ds_to_path=False)
    norm = grad_targ.norm().item()
    title = f"{ep_str}: Target Gradient Parameter Magnitude Histogram (Norm={norm:.1E})"
    plot_utils.plot_histogram(path=path, vals=grad_targ, xlabel=r'$\vert g_t \vert$',
                              n_bins=n_bins, title=title, log_y=True, weighted_cumul=True)
