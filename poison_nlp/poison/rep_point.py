__all__ = [
    "calc_representer_vals",
    "log_representer_scores",
]

import logging
import time
from typing import NoReturn, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F  # noqa

from . import _config as config
from . import dirs
from . import influence_utils
from .influence_utils import InfluenceMethod
from . import utils


def calc_representer_vals(trainer, targ_ds,
                          method: InfluenceMethod) -> Tuple[utils.WrapRoBERTa, Tensor]:
    r"""
    :param trainer: Training module
    :param targ_ds: Target dataset under study
    :param method: Exact version of representer point to run
    :return: Dictionary containing the representer values for each test example in \p test_x.
             Dimension of the representer tensor is #Test x #Classes x #Examples.
    """
    start = time.time()

    ret = _calc_representer_vals(trainer=trainer, targ_ds=targ_ds, method=method)

    total = time.time() - start
    logging.info(f"{method.value} Execution Time: {total:.6f} seconds")
    return ret


def _calc_representer_vals(trainer, targ_ds,
                           method: InfluenceMethod) -> Tuple[utils.WrapRoBERTa, Tensor]:
    r"""
    Encapsulated version of method to allow for calculating execution time

    :param trainer: Training module
    :param targ_ds: Target dataset under study
    :param method: Exact version of representer point to run
    :return: Dictionary containing the representer values for each test example in \p test_x.
             Dimension of the representer tensor is #Test x #Classes x #Examples.
    """
    inc_tr_loss = method in (InfluenceMethod.REP_POINT,)

    wrapped_model = utils.WrapRoBERTa(model=trainer.get_model())
    wrapped_model.eval()

    train_ds = utils.get_train_ds(trainer=trainer)

    _calc_alpha_i(trainer=trainer, wrapped_model=wrapped_model, train_ds=train_ds,
                  inc_tr_loss=inc_tr_loss)

    # noinspection PyPep8Naming
    w_dot_T = _calc_feature_dot_product(train_ds=train_ds, targ_ds=targ_ds, trainer=trainer,
                                        wrapped_model=wrapped_model)

    # w_dot is dimension  #Test x 1 x #TrainExamples.  Feature dot product f_i^T f_t is
    # independent of the class label.
    w_dot = torch.transpose(w_dot_T, 0, 1).unsqueeze(dim=1)

    # Matrix for \alpha_i in the paper
    alpha_i_raw = wrapped_model.alpha_i
    # Initial dimension of alpha_i is 1 x #Classes x #TrainExamples since the alpha_i values
    # are independent of the test examples.
    alpha_i = torch.transpose(alpha_i_raw, 0, 1).unsqueeze(dim=0)

    # elewise_pro dimensions:
    #    Dim 0: Target example ID (assumed 1 for this code)
    #    Dim 1: Label
    #    Dim 2: Influence value
    elewise_pro = w_dot.to(utils.TORCH_DEVICE) * alpha_i.to(utils.TORCH_DEVICE)
    elewise_pro = elewise_pro.cpu()

    log_representer_scores(model=wrapped_model, rep_vals=elewise_pro, targ_ds=targ_ds,
                           method=method)

    return wrapped_model, elewise_pro


def _calc_alpha_i(trainer, wrapped_model: utils.WrapRoBERTa, train_ds,
                  inc_tr_loss: bool) -> NoReturn:
    r"""
    Calculates the set of :math:`\alpha_i` values used in representer point calculations
    """
    # Initialize the alpha values
    alpha_i = torch.zeros([config.N_TRAIN + config.POISON_CNT, config.N_CLASSES], dtype=torch.float)
    all_ids = []

    lmbd = config.WEIGHT_DECAY
    alpha_scale = -1. / (2 * lmbd * (config.N_TRAIN + config.POISON_CNT))

    for tr_idx in range(config.N_TRAIN + config.POISON_CNT):
        sample = utils.get_ds_sample(idx=tr_idx, ds=train_ds, trainer=trainer)
        ids = sample["id"]
        all_ids.append(ids)
        # Do not weight by the training loss's gradient
        with torch.no_grad():
            xs, _ = wrapped_model.forward(sample=sample, penu=True)

        with torch.no_grad():
            logits = wrapped_model.linear.forward(xs)
        logits.requires_grad_(True)

        # Use autograd to calculate the gradient
        loss = utils.sentiment_criterion(logits=logits, sample=sample, reduction="none")
        ones = torch.ones(logits.shape[:1], dtype=torch.float, device=utils.TORCH_DEVICE)
        loss.backward(gradient=ones)
        # # Error check code for cross entropy loss checking
        # man_grad = _error_check_grad(dec_scores.shape[0], dec_scores, lbls)
        # Calculate the alpha values
        dec_grad = logits.grad.clone().detach().cpu()  # type: Tensor

        dec_grad *= alpha_scale
        if not inc_tr_loss:
            dec_grad.sign_()
        alpha_i[ids] = dec_grad
    # Store the alpha IDs and alpha_i values
    all_ids = torch.tensor(all_ids).view([-1]).long()
    wrapped_model.alpha_ids, _ = torch.sort(all_ids)
    wrapped_model.alpha_i = alpha_i[wrapped_model.alpha_ids]


def _calc_feature_dot_product(train_ds, trainer, wrapped_model, targ_ds) -> Tensor:
    r"""
    As detailed in representer theorem 3.1, representer point calculation requires calculating
    :math:`f_i^T f_t`.

    :param train_ds: Training dataset
    :param trainer:
    :param wrapped_model: Function used to encode the feature vectors, i.e., transform :math:`x`
                          into :math:`f`.
    :param targ_ds: Dataset of target examples

    :return: Weight dot product.  Dimension is [# training & poison samples vs. # test samples]
    """
    assert len(targ_ds) == 1, "Code assumes only a single dimension"
    n_targ = len(targ_ds)
    n_ids = config.N_TRAIN + config.POISON_CNT  # Total number of all training points (inc. poison)
    wrapped_model.eval()

    targ_sample = utils.get_ds_sample(idx=0, ds=targ_ds, trainer=trainer)
    with torch.no_grad():
        targ_x, _ = wrapped_model.forward(sample=targ_sample, penu=True)
    targ_x_T = torch.transpose(targ_x, 0, 1).to(utils.TORCH_DEVICE)  # noqa

    w_dot = torch.zeros([n_ids, n_targ], dtype=torch.float)
    for tr_idx in range(config.N_TRAIN + config.POISON_CNT):
        sample = utils.get_ds_sample(idx=tr_idx, ds=train_ds, trainer=trainer)

        with torch.no_grad():
            xs, _ = wrapped_model.forward(sample=sample, penu=True)
        dot_prod = xs @ targ_x_T

        ids = sample["id"]
        w_dot[ids] = dot_prod.cpu()

    w_dot = w_dot[wrapped_model.alpha_ids]
    return w_dot


def log_representer_scores(model: utils.WrapRoBERTa, rep_vals: Tensor, targ_ds,
                           method: InfluenceMethod) -> NoReturn:
    r""" Calculates the representer scores """
    ids = model.alpha_ids.cpu()

    assert len(targ_ds) == 1, "Only singleton target is supported"
    sample = targ_ds[0]
    lbl = utils.get_sample_label(sample=sample)  # Target dataset has the true label so negate it

    # Only a single target example so index is 0
    rep_vals = rep_vals[0, lbl].view([-1])  # type: Tensor

    argsort = torch.argsort(rep_vals, dim=0, descending=True)

    # Filter rows
    if len(ids.shape) == 1:
        ids = ids.unsqueeze(dim=1)

    ids, rep_vals = ids[argsort], rep_vals[argsort]

    influence_utils.calc_poison_auprc(res_type=method, inf=rep_vals, ids=ids)

    dirs.RES_DIR.mkdir(exist_ok=True, parents=True)
    path = dirs.RES_DIR / "rep_point_vals.csv"
    # Print a formatted file of representer point values
    with open(str(path), "w+") as f_out:
        f_out.write(f"helpful-ids,rep_point_inf\n")
        for i in range(0, ids.shape[0]):
            f_out.write(f"{ids[i].item()},{rep_vals[i].item():.6E}\n")
