__all__ = [
    "test_inf_est",
    "train_and_print_stats",
]

import collections
import logging
from typing import List, NoReturn, Optional, Sequence, Union

import torch

from . import _config as config
from .datasets.types import LearnerModule
from . import influence_utils
from .influence_utils import InfluenceMethod
from . import generate_results
from . import learner
from .learner import CombinedLearner
from . import tracin_utils
from .types import InfStruct, TensorGroup
from . import utils


def test_inf_est(block: utils.ClassifierBlock,
                 inf_setups: Union[InfStruct, Sequence[InfStruct]],
                 tg: TensorGroup, init_module: Optional[LearnerModule],
                 ex_id: Optional[int]) -> NoReturn:
    r""" Calculates and logs each of the filtering probabilities """
    if isinstance(inf_setups, InfStruct):
        inf_setups = [inf_setups]

    if config.USE_WANDB:
        train_dl, _ = learner.create_fit_dataloader(tg=tg, is_pretrain=False)
        for setup in inf_setups:
            tracin_utils.generate_wandb_results(block=block, inf_vals=setup.inf,
                                                method=setup.method, ids=setup.ids,
                                                train_dl=train_dl, ex_id=ex_id)

    filt_fracs = sorted(config.FILT_FRACS, reverse=True)
    for frac_filt in filt_fracs:
        for setup in inf_setups:
            setup.select_filter_ids(frac_filt=frac_filt)

        # include filter fracture in the model name
        train_and_print_stats(tg=tg, inf_setups=inf_setups, frac_filt=frac_filt,
                              init_module=init_module, ex_id=ex_id)


def train_and_print_stats(tg: TensorGroup, inf_setups: Sequence[InfStruct],
                          frac_filt: float, init_module: Optional[LearnerModule],
                          ex_id: Optional[int]) -> NoReturn:
    r"""
    Training specified number of models and report the results statistics.

    :param inf_setups:
    :param tg: \p TensorGroup of the tensors used in training
    :param frac_filt: Fraction of examples to remove.  Used only for logging
    :param init_module:
    :param ex_id: Example ID
    """
    assert 0 <= frac_filt < 1, "Filter fraction must be in range [0,1)"

    res_hist = collections.defaultdict(list)
    for itr in range(1, config.NUM_RETRAIN + 1):
        # Create a new module from scratch
        if init_module is None:
            module = utils.build_init_module(tg=tg)
        else:
            module = init_module

        learners = CombinedLearner.build_filt_learner(base_module=module, itr=itr,
                                                      inf_setups=inf_setups, frac_filt=frac_filt)
        learners.fit(tg=tg)
        for name, block in learners.blocks():
            res = generate_results.calculate_results(block=block, tg=tg)
            res_hist[name].append(res)

        learners.clear_models()  # Prevent explosion of model data
        del learners
        torch.cuda.empty_cache()

    for setup in inf_setups:
        _print_train_stats(method=setup.method, res_hist=res_hist[setup.name],
                           frac_filt=frac_filt, ex_id=ex_id)


def _print_train_stats(method: InfluenceMethod, res_hist: List[generate_results.LearnerResults],
                       frac_filt: float, ex_id: Optional[int]) -> NoReturn:
    r"""
    Writes the statistics of the retraining to the log.

    :param method: Influence analysis method
    :param res_hist: Training model results stats
    :param frac_filt: Fraction of examples filtered
    """
    assert len(res_hist) == config.NUM_RETRAIN, "Unexpected number of history results"

    # Log the basic setup information
    header = influence_utils.build_log_start_flds(res_type=method, block=None, ex_id=ex_id)
    logging.info(f"{header}: Filter Percentage: {frac_filt:.1%}")
    header = f"{header} p_rem={frac_filt:.1%}:"
    flds = (
            ("targ", "Targ Accuracy"),
    )
    for ds_name, metric_name in flds:
        acc = [res.__getattribute__(ds_name).acc for res in res_hist]
        _print_stat(header=header, vals=acc, metric_name=metric_name)


def _print_stat(header: str, vals: List[float], metric_name: str) -> NoReturn:
    r""" Print the statistic information """
    header = f"{header} {metric_name}"
    vals = torch.tensor(vals, dtype=tracin_utils.DTYPE)
    _, mean = torch.std_mean(vals, unbiased=True)
    logging.info(f"{header} Mean: {mean.item():.0%}")
