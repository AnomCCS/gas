__all__ = [
    "run",
]

import dill as pk
import logging
from pathlib import Path
from typing import List, NoReturn, Union

import torch
from torch import Tensor

from . import _config as config
from . import dirs
from . import influence_utils
from . import utils

KNN_LOG_COUNTS = config.ANOM_CUTOFF


def run(trainer, detect_ds, ex_ids: List[int], toggle_lbl: bool = False) -> NoReturn:
    run_rand_baseline(ex_ids=ex_ids)
    run_loss_baseline(trainer=trainer, ex_ids=ex_ids, detect_ds=detect_ds, toggle_lbl=toggle_lbl)
    run_knn_baseline(trainer=trainer, detect_ds=detect_ds, ex_ids=ex_ids)


def _log_baseline(id_val: int, desc: str, val: Union[float, Tensor]) -> NoReturn:
    r""" Helper method for logging baseline values """
    if isinstance(val, Tensor):
        val = val.item()
    logging.info(f"Ex ID {id_val}: {desc} Baseline: {val:.15f}")


def _get_sample(id_val: int, ds, trainer):
    r""" Standardizes getting the X and y tensors for the specified \p id_val """
    return utils.get_ds_sample(idx=id_val, ds=ds, trainer=trainer)


def run_rand_baseline(ex_ids: List[int]) -> NoReturn:
    r""" Execute the baseline where random value generated for each example """
    n_ex_ids = len(ex_ids)
    rand_vals = torch.rand((n_ex_ids,)).cpu()
    for id_val, val in zip(ex_ids, rand_vals):
        _log_baseline(id_val=id_val, desc="Random", val=val)


def run_loss_baseline(trainer, ex_ids: List[int], detect_ds, toggle_lbl: bool) -> NoReturn:
    r""" Calculates and logs the loss as the baseline """
    trainer.model.eval()
    for id_val in ex_ids:
        sample = _get_sample(id_val=id_val, trainer=trainer, ds=detect_ds)
        if toggle_lbl:
            label = sample["target"].item()
            sample["target"].fill_(label ^ 1)
        with torch.no_grad():
            loss, logits = influence_utils.get_loss_with_weight_decay(sample=sample,
                                                                      trainer=trainer,
                                                                      weight_decay=None,
                                                                      weight_decay_ignores=None)
        # Calculate and log the loss
        _log_baseline(id_val=id_val, desc="Loss", val=loss)


def run_knn_baseline(trainer, detect_ds, ex_ids: List[int]) -> NoReturn:
    r""" K-Nearest neighbors baseline """
    # Module supports the penu command for getting the penultimate features
    wrapped_model = utils.WrapRoBERTa(model=trainer.get_model())
    wrapped_model.eval()

    knn_feats_path = _build_knn_feats_filename()
    if not knn_feats_path.exists():
        train_ds = utils.get_train_ds(trainer=trainer)
        full_ids = influence_utils.general_utils.get_full_ids()
        feats = []
        with torch.no_grad():
            for id_val in full_ids:
                sample = _get_sample(id_val=id_val, ds=train_ds, trainer=trainer)
                vecs, _ = wrapped_model.forward(sample=sample, penu=True)
                feats.append(vecs)
        # Construct the feature vector
        feats = torch.cat(feats)
        # Serialize the features
        with open(str(knn_feats_path), "wb+") as f_out:
            pk.dump(feats.cpu(), f_out)
    # Load the KNN features from disk
    with open(str(knn_feats_path), "rb") as f_in:
        feats = pk.load(f_in)
    feats = feats.to(utils.TORCH_DEVICE)

    for id_val in ex_ids:
        sample = _get_sample(id_val=id_val, ds=detect_ds, trainer=trainer)
        feat_te, _ = wrapped_model.forward(sample=sample, penu=True)
        assert len(feat_te.shape) == 2, "Single dimension features not expected"
        assert feat_te.shape[0] == 1 and feat_te.shape[1] == feats.shape[1], "Bizarre shape"
        # Subtract the features
        diff_feats = feats - feat_te
        norms = torch.norm(diff_feats, dim=1,)
        assert norms.numel() == feats.shape[0], "Math error leads to unexpected shapes"
        # Sort in ascending order
        norms, _ = torch.sort(norms)

        log_cnt = config.ANOM_CUTOFF
        _log_baseline(id_val=id_val, desc=f"L2 Min Dist ({log_cnt})", val=norms[log_cnt - 1])


def _build_knn_feats_filename() -> Path:
    r""" Influence filename for first pass without flipped labels """
    prefix = f"nlp-targ-detect-knn-baselines-train-feats"
    return utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="pk")
