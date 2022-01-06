__all__ = [
    "run",
]

import logging
from typing import List, NoReturn, Union

from fastai.basic_data import DeviceDataLoader
import torch
from torch import Tensor

from . import _config as config
from . import filter_end_to_end
from . import influence_utils
from .types import TensorGroup
from . import utils


def run(block: utils.ClassifierBlock, tg: TensorGroup, train_dl: DeviceDataLoader,
        ex_ids: List[int], toggle_lbl: bool = False) -> NoReturn:
    run_rand_baseline(ex_ids=ex_ids)
    run_loss_baseline(block=block, ex_ids=ex_ids, tg=tg, toggle_lbl=toggle_lbl)
    run_knn_baseline(block=block, tg=tg, train_dl=train_dl, ex_ids=ex_ids)

    logging.info("")  # Blank line for better readability


def _log_baseline(id_val: int, desc: str, val: Union[float, Tensor]) -> NoReturn:
    r""" Helper method for logging baseline values """
    if isinstance(val, Tensor):
        val = val.item()
    header = influence_utils.build_log_start_flds(block=None, res_type=None, ep=None,
                                                  subepoch=None, ex_id=id_val)
    logging.info(f"{header} {desc} Baseline: {val:.3E}")


def _get_tfms_x_and_y(id_val: int, tg: TensorGroup):
    r""" Standardizes getting the X and y tensors for the specified \p id_val """
    is_bd, id_val = filter_end_to_end.check_is_bd(id_val, tg=tg)
    mask = tg.test_ids == id_val
    # Extract the example with the specified value
    test_x, test_y = tg.test_x[mask], tg.test_y[mask]
    assert test_y.numel(), f"Only a single element is expected. Got {test_y.numel()}"
    if is_bd:
        if config.DATASET.is_cifar():
            test_x = tg.test_d[mask]
        test_y = tg.test_adv_y[mask]
    # Perform transforms (if any)
    test_x = config.get_test_tfms()(test_x).to(utils.TORCH_DEVICE)
    return test_x, test_y


def run_rand_baseline(ex_ids: List[int]) -> NoReturn:
    r""" Execute the baseline where random value generated for each example """
    n_ex_ids = len(ex_ids)
    rand_vals = torch.rand((n_ex_ids,)).cpu()
    for id_val, val in zip(ex_ids, rand_vals):
        _log_baseline(id_val=id_val, desc="Random", val=val)


def run_loss_baseline(block: utils.ClassifierBlock, ex_ids: List[int],
                      tg: TensorGroup, toggle_lbl: bool) -> NoReturn:
    r""" Calculates and logs the loss as the baseline """
    block.eval()
    for id_val in ex_ids:
        test_x, test_y = _get_tfms_x_and_y(id_val=id_val, tg=tg)
        with torch.no_grad():
            acts = block.forward(test_x)
        if toggle_lbl:
            test_y = block.module.predict(test_x).view([-1])
        # Calculate and log the loss
        loss = block.loss.calc_validation_loss(dec_scores=acts, labels=test_y)
        _log_baseline(id_val=id_val, desc="Loss", val=loss)


def run_knn_baseline(block: utils.ClassifierBlock, train_dl: DeviceDataLoader,
                     tg: TensorGroup, ex_ids: List[int]) -> NoReturn:
    r""" K-Nearest neighbors baseline """
    block.eval()

    train_dl = train_dl.new(batch_size=config.BATCH_SIZE)
    # Module supports the penu command for getting the penultimate features
    feats, module = [], block.get_module()
    with torch.no_grad():
        for tensors in train_dl:
            batch = block.organize_batch(tensors, process_mask=True)
            vecs = module.forward(batch.xs, penu=True)
            feats.append(vecs)
    # Construct the feature vector
    feats = torch.cat(feats)
    for id_val in ex_ids:
        test_x, _ = _get_tfms_x_and_y(id_val=id_val, tg=tg)
        feat_te = module.forward(test_x, penu=True)
        assert len(feat_te.shape) == 2, "Single dimension features not expected"
        assert feat_te.shape[0] == 1 and feat_te.shape[1] == feats.shape[1], "Bizarre shape"
        # Subtract the features
        diff_feats = feats - feat_te
        norms = torch.norm(diff_feats, dim=1,)
        assert norms.numel() == feats.shape[0], "Math error leads to unexpected shapes"
        # Sort in ascending order
        norms, _ = torch.sort(norms)

        log_cnt = config.ANOM_CUTOFF
        _log_baseline(id_val=id_val, desc=f"L2 KNN Dist", val=norms[log_cnt - 1])
