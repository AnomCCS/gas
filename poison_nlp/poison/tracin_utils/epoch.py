__all__ = [
    "process",
]

import logging
from typing import List, NoReturn

import torch
from torch import LongTensor, Tensor

from . import _settings as settings
from . import results
from . import utils
from .. import _config as config
from .. import influence_utils
from ..influence_utils import InfluenceMethod
from .. import utils as parent_utils

LOG_LAYER_NORM_STATS = True


def process(subepoch_info, trainer, targ_ds, ex_ids: List[int], tensors: utils.TracInTensors,
            toggle_targ_lbl: bool, full_pass: bool, ex_lbls: List[int]) -> NoReturn:
    r"""
    Performs TracIn on a single epoch (including subepochs if applicable) for the specified
    \p block

    :param trainer: \p fairseq \p Trainer object
    :param subepoch_info: Subepoch information
    :param targ_ds: Target dataset
    :param tensors: All results tensors
    :param ex_ids: Target IDs
    :param toggle_targ_lbl: If \p True, toggle the target label.  Generally \p False.  Used for
                            testing misclassified examples
    :param full_pass: If \p True, performing second pass and considering all models not just
                      the final one
    :param ex_lbls: If specified, forces label values for the example IDs
    """
    assert not ex_lbls or len(ex_lbls) == len(ex_ids), "Mismatch in label and example lengths"

    n_updates = subepoch_info.num_updates
    # Get the subepoch stats
    decoder_only = config.TRACIN_DECODER_ONLY

    wd = None
    if full_pass:
        assert ex_lbls, "Second pass requires the example labels to be specified"
        lr = subepoch_info.lr
        trainer.load_checkpoint(filename=subepoch_info.checkpoint)
        trainer.get_model().eval()
    else:
        lr = 1

    # In all-in mode, iterate through all IDs. In normal mode, consider only those IDs used
    id_itr = tensors.full_ids.tolist()
    subep_ids = torch.tensor(subepoch_info.ids.tolist(), dtype=torch.long).long()
    train_ds = parent_utils.get_train_ds(trainer=trainer)
    ep = subepoch_info.epoch

    # # Log initial only results
    if lr == 0 and n_updates == 0:
        return

    # Get the loss gradient for the test (target) example
    # assert len(targ_ds) == 1, "Only a single example at a time is supported"
    targ_grads, grad_targ_layer = [], []
    for idx, id_val in enumerate(ex_ids):
        lbl_val = ex_lbls[idx] if ex_lbls else None
        targ_grad, targ_layer = utils.get_grads_and_layer(id_val=id_val, lbl_val=lbl_val,
                                                          ds=targ_ds, wd=wd, trainer=trainer,
                                                          decoder_only=decoder_only,
                                                          toggle_sample_label=toggle_targ_lbl)
        targ_grads.append(targ_grad)
        # Normalize the gradient norm by the layer norm
        grad_targ_layer.append(targ_layer)
    # Get the target grad values
    grad_targ, grad_targ_layer = torch.cat(targ_grads, dim=0), torch.cat(grad_targ_layer, dim=0)

    tensors.subep.reset()
    for id_val in id_itr:
        if lr == 0.: return  # noqa
        utils.tracin_dot_product(trainer=trainer, train_ds=train_ds, id_val=id_val,
                                 subep_tensors=tensors.subep,
                                 grad_targ=grad_targ, grad_targ_layer=grad_targ_layer,
                                 ep_wd=None, decoder_only=decoder_only)
    # Perform normalization based on learning rate and batch size as specified by TracIn
    # Perform externally to make faster with CUDA
    tensors.subep.dot_vals *= lr / config.BATCH_SIZE
    tensors.subep.gas_vals *= lr / config.BATCH_SIZE
    # Perform the sqrt externally to use CUDA
    tensors.subep.grad_norms.sqrt_()

    _combine_and_log_results(trainer=trainer, detect_ds=targ_ds, ep=ep, n_updates=n_updates,
                             subep_ids=subep_ids, grad_targ=grad_targ, tensors=tensors,
                             full_pass=full_pass, ex_ids=ex_ids)


def _log_val_base(ep: int, n_updates: int, name_val: str,
                  val_str: str) -> NoReturn:
    r""" Log the values from the execution """
    header = influence_utils.build_log_start_flds(ep=ep, n_updates=n_updates, res_type=None)
    logging.info(f"{header} {name_val}: {val_str}")


def _combine_and_log_results(trainer, detect_ds, ep: int, n_updates: int,
                             subep_ids: LongTensor, grad_targ: Tensor, full_pass: bool,
                             tensors: utils.TracInTensors, ex_ids: List[int]) -> NoReturn:
    r""" Combines and logs all results """
    full_ids = tensors.full_ids
    tensors.subep.dot_normed = tensors.subep.dot_vals / tensors.subep.grad_norms

    # Log test results
    def _log_result(inf_vals: Tensor, method: InfluenceMethod) -> NoReturn:
        results.generate_epoch_stats(ep=ep, n_updates=n_updates, method=method,
                                     inf_vals=inf_vals[full_ids], ids=full_ids,
                                     ex_id=ex_id)

    # GAS Results
    targ_grad_norm = grad_targ.norm(dim=1)
    assert len(ex_ids) == targ_grad_norm.numel(), "Mismatch in number of targ grad norms"
    targ_grad_norm[targ_grad_norm == 0] = settings.MIN_NORM
    gas_sim_base = tensors.subep.dot_normed / targ_grad_norm.unsqueeze(dim=1).cpu()
    tensors.gas_inf[:, full_ids] += gas_sim_base[:, full_ids]
    # GAS-L
    tensors.gas_layer[:, full_ids] += tensors.subep.gas_vals[:, full_ids]
    # TracInCP
    tensors.tracincp[:, full_ids] += tensors.subep.dot_vals[:, full_ids]
    # TracIn and variants
    tensors.tracin_inf[:, subep_ids] += tensors.subep.dot_vals[:, subep_ids]
    tensors.tracin_sim[:, subep_ids] += gas_sim_base[:, subep_ids]
    tensors.tracin_sim_l[:, subep_ids] += tensors.subep.gas_vals[:, subep_ids]

    for row, ex_id in enumerate(ex_ids):
        log_methods = (utils.get_gas_log_flds,
                       utils.get_tracincp_log_flds,
                       utils.get_tracin_log_flds,
                       )
        for log_func in log_methods:
            for inf, _method, _ in log_func(tensors=tensors):
                _log_result(inf[row], method=_method)
