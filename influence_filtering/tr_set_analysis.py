__all__ = [
    "influence_methods",
]

from typing import NoReturn, Optional, Tuple

import dill as pk
import torch
from torch import LongTensor, Tensor

from poison import config, learner, tracin_utils
import poison.filter_end_to_end
import poison.influence_func
from poison.datasets.types import LearnerModule
import poison.dirs
from poison.filter_end_to_end import test_inf_est
from poison.influence_utils import InfluenceMethod
import poison.learner
import poison.tracin
import poison.tracin_utils
from poison.types import InfStruct, TensorGroup
import poison.utils
from poison.utils import ClassifierBlock


def _construct_rand_inf_setups(tg: TensorGroup) -> InfStruct:
    r"""
    Construct the random influence learners
    :return: Tuple of the random all classes and random target class only influence setups
    """
    # Filter uniformly at random from both classes
    inf_rand = torch.rand(tg.tr_ids.shape)
    train_dl, _ = poison.learner.create_fit_dataloader(tg=tg, is_pretrain=False)
    setup_rand_all = InfStruct(name="random", method=InfluenceMethod.RANDOM, ids=tg.tr_ids,
                               inf=inf_rand.clone())
    return setup_rand_all


def collect_baselines(tg: TensorGroup, ex_id: Optional[int],
                      init_module: Optional[LearnerModule]) -> NoReturn:
    r""" Collects the baseline performance without filtering and with random filtering """
    # No filtering at all
    inf_setup = InfStruct(name="baseline", method=InfluenceMethod.BASELINE, ids=tg.tr_ids,
                          inf=torch.zeros(tg.tr_ids.shape))
    # include filter fracture in the model name
    poison.filter_end_to_end.train_and_print_stats(tg=tg, inf_setups=[inf_setup], frac_filt=0,
                                                   init_module=init_module, ex_id=ex_id)


def influence_methods(block: ClassifierBlock, tg: TensorGroup, ex_id: int,
                      targ_x: Tensor, targ_y: LongTensor,
                      init_module: Optional[LearnerModule] = None) -> NoReturn:
    r""" Executes the various influence estimation methods and impact on influence """
    _run_tracin(block=block, tg=tg, targ_x=targ_x, targ_y=targ_y,
                ex_id=ex_id, init_module=init_module)

    _run_influence_func(block=block, tg=tg, targ_x=targ_x, targ_y=targ_y, ex_id=ex_id,
                        init_module=init_module)


def _run_influence_func(block: ClassifierBlock, tg: TensorGroup,
                        targ_x: Tensor, targ_y: LongTensor, ex_id: int,
                        init_module: Optional[LearnerModule]) -> NoReturn:
    r""" Performs the influence function experiments """
    train_dl, _ = poison.learner.create_fit_dataloader(tg=tg, is_pretrain=False)

    # Influence functions and variants
    block.to(poison.utils.TORCH_DEVICE)  # Move back and forth from GPU to save memory
    if_res = poison.influence_func.calc(block=block, tr_dl=train_dl, te_x=targ_x, te_y=targ_y,
                                        ex_id=ex_id)
    block.cpu()

    flds = (
            (if_res.inf_base, InfluenceMethod.INF_FUNC, "if-base"),
            (if_res.inf_sim, InfluenceMethod.INF_FUNC_SIM, "if-normed"),
            (if_res.inf_sim_l, InfluenceMethod.INF_FUNC_SIM_L, "if-layer-norm"),
            )
    ids = if_res.ids
    inf_setups = [InfStruct(ids=ids, method=method, inf=inf, name=name)
                  for inf, method, name in flds]

    # Test multiple random retrainings and log statistics of the resulting m
    test_inf_est(block=block, tg=tg, init_module=init_module, inf_setups=inf_setups, ex_id=ex_id)


def _run_tracin(block: ClassifierBlock, tg: TensorGroup, ex_id: int,
                targ_x: Tensor, targ_y: LongTensor, init_module: Optional[LearnerModule],
                wd: Optional[float] = None) -> NoReturn:
    r""" Performs end-to-end filtering using epoch information and batch size """
    train_dl, _ = learner.create_fit_dataloader(tg=tg, is_pretrain=False)
    train_dl = tracin_utils.configure_train_dataloader(train_dl)

    tracin_res_dir = poison.dirs.RES_DIR / "tracin"
    inf_path = poison.utils.construct_filename("inf-vals", out_dir=tracin_res_dir, file_ext="pk")

    block.to(poison.utils.TORCH_DEVICE)  # Move back and forth from GPU to save memory
    # Select the poisoned IDs to remove
    all_inf = poison.tracin.calc(block=block, train_dl=train_dl, wd=wd,
                                 n_epoch=config.NUM_EPOCH, bs=config.BATCH_SIZE,
                                 x_targ=targ_x, y_targ=targ_y, ex_id=ex_id)
    with open(str(inf_path), "wb+") as f_out:
        pk.dump(all_inf, f_out)
    block.cpu()
    with open(str(inf_path), "rb") as f_in:
        all_inf = pk.load(f_in)
    poison.tracin.log_final(block=block, ex_id=ex_id, train_dl=train_dl, tensors=all_inf)

    flds = (
            (all_inf.gas_sim, InfluenceMethod.GAS, "gas"),
            (all_inf.gas_l, InfluenceMethod.GAS_L, "gas-layer"),
            (all_inf.tracincp, InfluenceMethod.TRACINCP, "tracincp"),
            (all_inf.tracin_inf, InfluenceMethod.TRACIN, "tracin"),
            # (all_inf.tracin_sim, InfluenceMethod.TRACIN_SIM, "tracin-sim"),
            )

    ids = all_inf.full_ids
    inf_setups = [InfStruct(ids=ids, method=method, inf=inf[ids], name=name)
                  for inf, method, name in flds]
    inf_setups.append(_construct_rand_inf_setups(tg=tg))
    # Test multiple random retrainings and log statistics of the resulting models
    test_inf_est(block=block, tg=tg, init_module=init_module, inf_setups=inf_setups, ex_id=ex_id)
