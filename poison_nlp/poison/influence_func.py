__all__ = [
    "calc",
]

import json
import logging
from pathlib import Path
import pickle as pk
from typing import NoReturn

import torch
from torch import Tensor

from . import _config as config
from . import dirs
from . import influence_utils
from .influence_utils import InfluenceMethod
from .influence_utils.nn_influence_utils import InfFuncTensors
from . import utils

DEFAULT_R = 10
INF_RES_DIR = None


@influence_utils.log_time(res_type=InfluenceMethod.INF_FUNC)
def calc(trainer, targ_ds):
    r""" Performs the influence function calculations """
    global INF_RES_DIR
    INF_RES_DIR = dirs.RES_DIR / "inf_func"
    INF_RES_DIR.mkdir(exist_ok=True, parents=True)

    # Disable dropout
    trainer.get_model().eval()

    # Each learner will have different influence function values
    _calc_block_inf(trainer=trainer, targ_ds=targ_ds)


def _calc_block_inf(trainer, targ_ds) -> InfFuncTensors:
    r""" Encapsulates calculating the influence of blocks """
    wd = config.WEIGHT_DECAY

    full_ids = influence_utils.get_full_ids()
    filename = INF_RES_DIR / f"inf_results.pkl"  # noqa
    if True:
        # Order of res: influences, train_inputs_collections, s_test
        res = influence_utils.compute_influences(trainer=trainer,
                                                 full_ids=full_ids,
                                                 test_ds=targ_ds,
                                                 weight_decay=wd,
                                                 hvp_batch_size=config.HVP_BATCH_SIZE,
                                                 s_test_damp=config.DAMP,
                                                 s_test_scale=config.SCALE,
                                                 s_test_num_samples=config.R_DEPTH,
                                                 s_test_iterations=config.T_REPEATS)

        with open(filename, "wb+") as f_out:
            pk.dump(res, f_out)

    with open(filename, "rb") as f_in:
        res = pk.load(f_in)  # type: InfFuncTensors

    fields = ((res.inf_base, InfluenceMethod.INF_FUNC),
              (res.inf_sim, InfluenceMethod.INF_FUNC_SIM),
              (res.inf_sim_l, InfluenceMethod.INF_FUNC_SIM_L),
              )
    for influences, method in fields:
        # Convert the estimated loss
        est_loss = -1 / full_ids.numel() * influences
        influence_utils.calc_poison_auprc(res_type=method, ids=full_ids, inf=est_loss)
    return res


def _log_influence_results(est_loss: Tensor, helpful: Tensor,
                           ids: Tensor, y: Tensor) -> NoReturn:
    r"""
    :param est_loss: Estimated change in loss if training example is removed
    :param helpful: Training examples numbered from 0 to (# training examples - 1)
    :param ids: ID numbers for the training examples used by the learner
    :param y: Labels for the training examples used by the block
    """
    harmful = torch.flip(helpful, dims=[0])
    for i in range(2):
        if i == 0:
            name = "helpful"
        else:
            name = "harmful"
        top_ord = locals()[name][:5]
        top_ids = ids[top_ord]
        logging.info(f"Top {name} IDs: {top_ids.tolist()}")
        logging.info(f"Top {name} Est. Change Loss: {est_loss[top_ord].tolist()}")
        logging.info(f"Top {name} Labels: {y[top_ord].tolist()}")


def _build_inf_results_file(helpful: Tensor, influence_vals: Tensor,
                            est_loss_vals: Tensor, ids: Tensor, y: Tensor) -> NoReturn:
    r"""
    Constructs the influence results file

    :param helpful:
    :param influence_vals:
    :param est_loss_vals:
    :param ids:
    :param y:
    :return:
    """
    assert ids.shape == helpful.shape, "Helpful tensor shape does not match the ID tensor"
    assert y.shape == helpful.shape, "Helpful tensor shape does not match the y tensor"

    ord_ids, ord_y = ids[helpful], y[helpful]
    ord_inf, ord_est_loss = influence_vals[helpful], est_loss_vals[helpful]
    # noinspection PyDictCreation
    inf_res = {"hvp_batch_size": config.HVP_BATCH_SIZE,
               "damp": config.DAMP,
               "scale": config.SCALE,
               "helpful-ids": ord_ids.tolist(),
               "helpful-ord-loss": ord_est_loss.tolist(),
               "helpful-y": ord_y.tolist(),
               "helpful-influence": ord_inf.tolist()}

    res_path = _build_inf_res_filename()
    with open(res_path, "w+") as f_out:
        json.dump(inf_res, f_out)


def _build_inf_res_filename() -> Path:
    r""" Construct the filename for the results """
    prefix = f"inf"
    return utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="json",
                                    add_timestamp=True)


def _build_y_tensor(trainer, full_ids: Tensor) -> Tensor:
    r""" Constructs the set of Y-values for the learner """
    train_ds = utils.get_train_ds(trainer=trainer)
    full_y = torch.zeros(full_ids.shape, dtype=torch.long)
    for id_val in full_ids:
        sample = utils.get_ds_sample(idx=id_val, ds=train_ds, trainer=trainer)
        full_y[id_val] = utils.get_sample_label(sample=sample)
    return full_y
