__all__ = [
    "calc",
    "log_final_results",
]

import dill as pk
import logging
import sys
from typing import List, NoReturn

import tqdm

import torch
from torch import Tensor

from . import _config as config
from . import dirs
from . import influence_utils
from .influence_utils import InfluenceMethod
from . import tracin_utils
from . import utils


@influence_utils.log_time(res_type=InfluenceMethod.GAS)
def calc(trainer, tracin_hist, targ_ds, ex_ids: List[int], i_repeat: int, toggle_targ_lbl: bool,
         full_pass: bool) -> tracin_utils.TracInTensors:
    r"""
    Calculate the TracIn influence values.

    :param trainer: trainer object under study
    :param tracin_hist: Stores the trace of the training
    :param targ_ds: Target dataset
    :param ex_ids: Target IDs
    :param i_repeat: Repeat iteration
    :param toggle_targ_lbl: If \p True, toggle the target label.  Generally \p False.  Used for
                            testing misclassified examples
    :param full_pass: If \p True, second pass where
    :return: Tuple of the influence values and dataset IDs
    """
    assert len(targ_ds) > max(ex_ids), "Example ID exceeds the dataset length"
    tracin_hist.validate_checkpoints()
    trainer.model.eval()

    # Clean and error check inputs
    assert tracin_utils.START_EPOCH in (0, 1), "Only start epoch of 0 or 1 supported"

    hist = tracin_hist.get_subepoch_info()
    ep_tqdm = tqdm.tqdm(hist, total=len(hist), file=sys.stdout, desc="GrAIN Ep", disable=True)

    ex_lbls = []
    if full_pass:
        ex_lbls = _get_example_predictions(trainer=trainer, tracin_hist=tracin_hist,
                                           targ_ds=targ_ds, ex_ids=ex_ids)

    full_ids = influence_utils.general_utils.get_full_ids()
    # Initialize the results arrays
    all_tensors = tracin_utils.TracInTensors(full_ids=full_ids,
                                             inf_numel=config.N_TRAIN + config.POISON_CNT,
                                             id_numel=len(ex_ids))
    # Do not perform a deepcopy of the dataset to reduce RAM overhead
    with ep_tqdm as pbar:
        for subepoch_info in pbar:
            tracin_utils.epoch.process(subepoch_info=subepoch_info, trainer=trainer,
                                       targ_ds=targ_ds, ex_ids=ex_ids, tensors=all_tensors,
                                       toggle_targ_lbl=toggle_targ_lbl, full_pass=full_pass,
                                       ex_lbls=ex_lbls)

            if not full_pass:
                break

            prefix = ["various-ids",
                      "full-pass" if full_pass else "single-itr",
                      f"n_repeat={i_repeat}"
                      f"n-updates={subepoch_info.num_updates:03d}"]
            path = utils.construct_filename("_".join(prefix), out_dir=dirs.RES_DIR, file_ext="pk")
            with open(str(path), "wb+") as f_out:
                pk.dump((ex_ids, all_tensors), f_out)

    # Final influence and IDs
    return all_tensors


def _get_example_predictions(trainer, tracin_hist, targ_ds, ex_ids: List[int]) -> List[int]:
    r""" Get the prediction associated with the target example """
    reload_best_checkpoint(trainer=trainer, tracin_hist=tracin_hist)

    ex_lbls = []
    for id_val in ex_ids:
        sample = utils.get_ds_sample(idx=id_val, ds=targ_ds, trainer=trainer)
        with torch.no_grad():
            _, logits = influence_utils.get_loss_with_weight_decay(sample=sample, trainer=trainer,
                                                                   weight_decay=None,
                                                                   weight_decay_ignores=None)

        ex_lbls.append(torch.argmax(logits, dim=1).item())
    return ex_lbls


def log_final_results(trainer, tensors: tracin_utils.TracInTensors, ex_ids: List[int]) -> NoReturn:
    r""" Log the final results """
    full_ids = tensors.full_ids

    def _log_result(inf_vals: Tensor, method: InfluenceMethod) -> NoReturn:
        tracin_utils.results.generate_epoch_stats(ep=None, n_updates=None, method=method,
                                                  inf_vals=inf_vals[full_ids],
                                                  ids=full_ids, ex_id=ex_id)

    for row, ex_id in enumerate(ex_ids):
        log_methods = (tracin_utils.get_gas_log_flds,
                       tracin_utils.get_tracin_log_flds,
                       tracin_utils.get_tracincp_log_flds,
                       )
        for log_func in log_methods:
            for inf, _method, _ in log_func(tensors=tensors):
                _log_result(inf[row], method=_method)


def reload_best_checkpoint(trainer, tracin_hist) -> NoReturn:
    r"""
    Reloads the states for the specified \p Trainer

    :param trainer: Trainer under analysis
    :param tracin_hist: Checkpoints of training history
    """
    best_checkpoint = tracin_hist.get_best_checkpoint()
    logging.info(f"Reloading best checkpoint \"{best_checkpoint}\"")
    trainer.load_checkpoint(best_checkpoint)
    logging.info("Best checkpoint reload complete")
