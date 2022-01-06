__all__ = [
    "calc",
]

import copy
import logging
from pathlib import Path
import pickle as pk
from typing import Optional, Tuple

import numpy as np

from fastai.basic_data import DeviceDataLoader
import torch
from torch import LongTensor, Tensor

from torch.utils.data import DataLoader

from . import _config as config
from . import dirs
from . import influence_utils
from .influence_utils.nn_influence_utils import InfFuncTensors, compute_influences
from .influence_utils import InfluenceMethod
from .losses import TORCH_DEVICE
from . import tracin_utils
from .types import CustomTensorDataset, LearnerParams
from . import utils

DEFAULT_R = 10
INF_RES_DIR = None


@influence_utils.log_time(res_type=InfluenceMethod.INF_FUNC)
def calc(block: utils.ClassifierBlock, tr_dl: DeviceDataLoader, te_x: Tensor,
         te_y: Tensor, ex_id: Optional[int], use_precompute: bool = False) -> InfFuncTensors:
    r"""
    :param block: Block of interest
    :param tr_dl: \p DataLoader used to train the learners
    :param te_x:
    :param te_y:
    :param ex_id:
    :param use_precompute:
    :return:
    """
    global INF_RES_DIR
    INF_RES_DIR = dirs.RES_DIR / config.DATASET.name.lower() / "inf_func" / block.name().lower()
    INF_RES_DIR.mkdir(exist_ok=True, parents=True)

    assert torch.min(te_x).item() >= 0, "Test tensor already transformed"
    # Use the tensors to create a supervised training set. Supports optional test transform
    kw_args = {}
    if config.has_tfms():
        kw_args = {"transform": config.get_test_tfms()}
    ds = CustomTensorDataset([te_x, te_y], **kw_args)
    te_dl = DataLoader(ds, drop_last=False, shuffle=False, num_workers=utils.NUM_WORKERS,
                       batch_size=1)

    tr_dl = tracin_utils.configure_train_dataloader(tr_dl)

    # Do not use the original training dataloader since it may require transforms or drop part of
    # the dataset.  Create temporary Dataloader that overcomes those issues
    assert isinstance(tr_dl.dl.dataset, CustomTensorDataset), "Code supports certain dataset type"

    # Each learner will have different influence function values
    res = _calc_block_inf(block=block, tr_dl=tr_dl, te_dl=te_dl, ex_id=ex_id,
                          use_precompute=use_precompute)
    return res


def _build_auc_path_name(block: utils.ClassifierBlock, res_name: str) -> Path:
    r""" Constructs a file path to which the AUROC plot is stored"""
    ds_name = config.DATASET.name.lower()
    out_dir = dirs.PLOTS_DIR / ds_name / "inf-func" / block.name().lower() / res_name.lower()
    out_dir.mkdir(exist_ok=True, parents=True)

    file_prefix = [block.name(), "inf-func", res_name.lower(), "fin"]
    return utils.construct_filename(prefix="-".join(file_prefix).lower(), file_ext="png",
                                    out_dir=out_dir, add_ds_to_path=False, add_timestamp=True)


def _calc_block_inf(block: utils.ClassifierBlock, tr_dl: DeviceDataLoader, te_dl: DataLoader,
                    ex_id: Optional[int], use_precompute: bool) -> InfFuncTensors:
    r"""
    Encapsulates calculating the influence of blocks

    :param block: Block for which influence is calculated
    :param tr_dl: Train \p DeviceDataLoader
    :param te_dl: Test \p DataLoader
    :return: Tuple of the sorted influence, backdoor IDs, and dataset IDs
    """
    block.eval()
    wd = config.get_learner_val(block.name(), LearnerParams.Attribute.WEIGHT_DECAY)

    bl_x, bl_y, ids = _build_learner_tensors(block=block, train_dl=tr_dl)
    n_tr = bl_y.shape[0]
    batch_tr_dl, instance_tr_dl = _build_block_dataloaders(bl_x=bl_x, bl_y=bl_y)

    flds = [block.name(), "inf-fin"]
    prefix = "-".join(flds).lower()
    # noinspection PyTypeChecker
    filename = utils.construct_filename(prefix=prefix, out_dir=INF_RES_DIR, ex_id=ex_id,
                                        file_ext="pkl", add_ds_to_path=False)
    # if not filename.exists():
    if True:
        if use_precompute:
            logging.debug("Reusing precomputed s_test")
            assert filename.exists(), "Precompute specified but no precomputed s_test file"
            with open(filename, "rb") as f_in:
                precomputed_s_test = pk.load(f_in).s_test
        else:
            precomputed_s_test = None
        # Order of res: influences, train_inputs_collections, s_test
        res = compute_influences(model=block,
                                 n_gpu=1 if torch.cuda.is_available() else 0,
                                 ids=ids,
                                 device=TORCH_DEVICE,
                                 f_loss=block.loss.calc_train_loss,
                                 test_dl=te_dl,
                                 batch_train_data_loader=batch_tr_dl,
                                 instance_train_data_loader=instance_tr_dl,
                                 weight_decay=wd,
                                 s_test_damp=config.DAMP,
                                 s_test_scale=config.SCALE,
                                 s_test_num_samples=config.R_DEPTH,
                                 s_test_iterations=config.T_REPEATS,
                                 precomputed_s_test=precomputed_s_test)
        # Convert the tensors to loss values
        for inf in (res.inf_base, res.inf_sim, res.inf_sim_l):
            inf *= -1 / n_tr

        # Extract the result fields
        with open(filename, "wb+") as f_out:
            pk.dump(res, f_out)

    with open(filename, "rb") as f_in:
        inf_tensors = pk.load(f_in)
    return inf_tensors


def _build_inf_res_filename(block: utils.ClassifierBlock) -> Path:
    r""" Construct the filename for the results """
    prefix = f"inf-{block.name()}-t-id={config.TARG_IDX}"
    return utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="json",
                                    add_timestamp=True)


def _build_learner_tensors(block: utils.ClassifierBlock,
                           train_dl: DeviceDataLoader) -> Tuple[Tensor, Tensor, LongTensor]:
    r"""
    Construct the X/y/IDs tensors based on any example filtering by the block
    :param block:
    :param train_dl:
    :return: Tuple of X, y, backdoor IDs, and dataset IDs tensors respectively
    """
    tmp_ds = copy.copy(train_dl.dl.dataset)
    tmp_ds.transform = None
    cp_tr_dl = DataLoader(tmp_ds, batch_size=config.BATCH_SIZE,
                          drop_last=False, shuffle=False, num_workers=utils.NUM_WORKERS)

    all_x, all_y, all_ids = [], [], []
    for batch_tensors in cp_tr_dl:
        batch = block.organize_batch(batch_tensors, verify_contents=True)
        if batch.skip():
            continue

        all_x.append(batch.xs.cpu())
        all_y.append(batch.lbls.cpu())
        all_ids.append(batch.ids.cpu())

    all_ids, id_ordering = torch.sort(torch.cat(all_ids, dim=0), dim=0)
    # Consolidate into tensors and sort by the image IDs
    n_tr = id_ordering.shape[0]  # Number of training examples
    # Strange count for number of training examples
    assert n_tr == id_ordering.numel(), "Weird size mismatch"

    filename = "_".join([block.name().lower(), "inf-func", block.start_time])
    # noinspection PyUnresolvedReferences
    np.savetxt(INF_RES_DIR / (filename + "_ord-id.csv"), all_ids.numpy(), fmt='%d',
               delimiter=',')

    tr_x, tr_y = torch.cat(all_x, dim=0)[id_ordering], torch.cat(all_y, dim=0)[id_ordering]
    return tr_x, tr_y, all_ids  # noqa


def _build_block_dataloaders(bl_x: Tensor, bl_y: Tensor) -> Tuple[DataLoader, DataLoader]:
    r"""
    Constructs two separate dataloaders.  You may want different properties dataloader properties
    when estimating the Hessian vector product (HVP) and when estimating influence.  By specifying
    separate \p DataLoaders, those two roles are separated.

    :param bl_x: Block's X tensor
    :param bl_y: Blocks y (i.e. label) tensor
    :return: Tuple of the batch \p DataLoader (used for generating the HVP) and the instance
             \p DataLoader used when estimating influence.
    """
    # ToDo Determine whether to use transforms
    ds = CustomTensorDataset((bl_x, bl_y), transform=config.get_train_tfms())
    batch_tr_dl = DataLoader(ds, batch_size=config.HVP_BATCH_SIZE,
                             shuffle=True, drop_last=True, num_workers=utils.NUM_WORKERS)

    ds = CustomTensorDataset((bl_x, bl_y), transform=config.get_test_tfms())
    instance_tr_dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False,
                                num_workers=utils.NUM_WORKERS)
    return batch_tr_dl, instance_tr_dl
