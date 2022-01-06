__all__ = [
    "InfFuncTensors",
    "compute_gradients",
    "compute_influences",
    "derivative_of_loss",
    "get_loss_with_weight_decay",
]

import dataclasses
import itertools
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import torch
from torch import LongTensor, Tensor
import torch.nn.functional as F  # noqa

from . import general_utils as utils
from .. import _config as config
from .. import utils as parent_utils


@dataclasses.dataclass
class InfFuncTensors:
    inf_base: Tensor = None
    inf_sim: Tensor = None
    inf_sim_l: Tensor = None

    s_test: List[Tensor] = None

    ids: LongTensor = None


def compute_gradients(trainer, sample: Union[dict, List[dict]],
                      params_filter: Optional[List[str]], weight_decay: Optional[float],
                      weight_decay_ignores: Optional[List[str]],
                      create_graph: bool = True, use_decoder_only: bool = False,
                      return_loss: bool = False, return_acts: bool = False):
    r"""
    :param trainer: \p Trainer under investigation
    :param sample: Sample batch over which loss will be calculated
    :param params_filter:
    :param weight_decay:
    :param weight_decay_ignores:
    :param create_graph: If \p True, enables construction of derivative graph. This allows
                         computing higher order derivative products.
    :param use_decoder_only: If \p True, only use the decoder for computing the gradients
    :param return_loss: If \p True, also return the loss value
    :param return_acts: If \p True, also return the logits vector
    :return:
    """
    if params_filter is None:
        params_filter = []

    trainer.get_model().zero_grad()
    if isinstance(sample, dict):
        # Single loss value
        loss, logits = get_loss_with_weight_decay(trainer=trainer, sample=sample,
                                                  weight_decay=weight_decay,
                                                  weight_decay_ignores=weight_decay_ignores)
    elif isinstance(sample, list):
        assert not return_acts, "Returning logits is not supported if multiple instances"
        # Take the average of multiple
        loss = []
        for single_sample in sample:
            single_loss, _ = get_loss_with_weight_decay(trainer=trainer, sample=single_sample,
                                                        weight_decay=weight_decay,
                                                        weight_decay_ignores=weight_decay_ignores)
            loss.append(single_loss)
        loss = torch.cat(loss).mean()
    else:
        raise ValueError("Unknown type for sample")

    # inputs: Inputs w.r.t. which the gradient is taken.  Simply the parameters since
    #         the gradient is w.r.t. \theta, i.e., $\nabla_{\theta}$.
    if not params_filter:
        inputs = get_model_params(trainer=trainer, use_decoder_only=use_decoder_only)
    else:
        inputs = [param for name, param in trainer.get_model().named_parameters()
                  if name not in params_filter]

    # create_graph=True: Enables construction of derivative graph. This allows computing higher
    # order derivative products.
    grad = torch.autograd.grad(outputs=loss, inputs=inputs, create_graph=create_graph,
                               allow_unused=True)
    if not return_acts and not return_loss:
        return grad

    rets = []
    if return_loss:
        rets.append(loss.detach())
    if return_acts:
        rets.append(logits.detach())  # noqa
    rets.append(grad)
    return tuple(rets)


def get_model_params(trainer, use_decoder_only: bool) -> List[Tensor]:
    r"""
    :param trainer: Trainer whose parameters will be returned
    :param use_decoder_only: If \p True, consider on the decoder parameters
    :return: List of the parameters
    """
    model = trainer.model
    if use_decoder_only:
        model = model.decoder

    return list(
        filter(
            lambda p: p.requires_grad,
            itertools.chain(model.parameters(), trainer.criterion.parameters()),
        )
    )


def get_loss_with_weight_decay(trainer, sample: dict, weight_decay: Optional[float],
                               weight_decay_ignores: Optional[List[str]]) -> Tuple[Tensor, Tensor]:
    r""" Returns Tuple of loss and logits respectively """
    loss, logits = get_loss_with_logits(trainer=trainer, sample=sample)

    # # model outputs are always tuple in transformers (see doc)
    # loss = outputs[0]
    # loss = loss.mean()

    # In PyTorch, weight-decay loss and gradients are calculated in
    # optimizers rather in nn.Module, so we have to manually specify
    # this for the loss here.
    if weight_decay is not None:
        no_decay = (weight_decay_ignores if weight_decay_ignores is not None else [])

        # noinspection PyUnresolvedReferences
        weight_decay_loss = torch.cat([
            p.square().view(-1)
            for n, p in trainer.get_model().named_parameters()
            if not any(nd in n for nd in no_decay)
        ]).sum() * weight_decay
        loss = loss + weight_decay_loss

    return loss, logits


def get_loss_without_wd(trainer, sample) -> Tensor:
    r""" Calclates the loss without weight decay """
    # logits = parent_utils.trainer_forward(trainer=trainer, sample=sample)
    # loss = parent_utils.sentiment_criterion(logits=logits, sample=sample, reduction="mean")
    loss, _, _ = trainer.criterion(trainer.get_model(), sample)
    return loss


def get_loss_with_logits(trainer, sample) -> Tuple[Tensor, Tensor]:
    r""" Calculates the loss without weight decay and also returns the logits """
    logits = parent_utils.trainer_forward(trainer=trainer, sample=sample)
    loss = parent_utils.sentiment_criterion(logits=logits, sample=sample, reduction="mean")
    # loss, _, _ = trainer.criterion(trainer.get_model(), sample)
    return loss, logits


def derivative_of_loss(acts: Tensor, sample: dict) -> Tensor:
    r"""
    Calculates the derivative of loss function \p f_loss w.r.t. output activation \p outs
    and labels \p lbls
    """
    assert len(acts.shape) <= 2, f"Unexpected shape for outs"
    # Need to require gradient to calculate derive
    acts = acts.detach().clone()
    acts.requires_grad = True

    loss = parent_utils.sentiment_criterion(logits=acts, sample=sample, reduction="mean")
    # Calculates the loss
    ones = torch.ones([], dtype=acts.dtype, device=acts.device)
    # Back propagate the gradients
    loss.backward(ones)
    return acts.grad.clone().detach().type(acts.dtype)  # type: Tensor


def compute_influences(trainer, full_ids: Tensor, test_ds,
                       params_filter: Optional[List[str]] = None,
                       weight_decay: Optional[float] = None,
                       weight_decay_ignores: Optional[List[str]] = None,
                       hvp_batch_size: int = 1,
                       s_test_damp: float = 3e-5, s_test_scale: float = 1e4,
                       s_test_num_samples: Optional[int] = None, s_test_iterations: int = 1,
                       precomputed_s_test: Optional[List[torch.Tensor]] = None) \
        -> InfFuncTensors:
    r"""
    :param trainer: Trainer for the model under investigation
    :param full_ids: All IDs considered by the trainer
    :param test_ds: Test dataset
    :param params_filter:
    :param weight_decay:
    :param weight_decay_ignores:
    :param hvp_batch_size: Hessian vector product batch size
    :param s_test_damp:
    :param s_test_scale:
    :param s_test_num_samples:
    :param s_test_iterations:
    :param precomputed_s_test:
    :return:
    """

    if s_test_iterations < 1:
        raise ValueError("`s_test_iterations` must be >= 1")
    if hvp_batch_size < 1:
        raise ValueError("HVP batch size must be at >=1")
    assert len(test_ds) == 1, "Only singleton target dataset is supported"

    if weight_decay_ignores is None:
        # https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/trainer.py#L325
        weight_decay_ignores = [
            "bias",
            "LayerNorm.weight"]

    # Filter the unused parameters in the RoBERTa model
    add_filter = ["decoder.lm_head.bias",
                  "decoder.lm_head.dense.bias",
                  "decoder.lm_head.layer_norm.bias"]
    if params_filter is None:
        params_filter = add_filter
    else:
        params_filter += add_filter

    if precomputed_s_test is not None:
        logging.info("Using precomputed s_test")
        s_test = precomputed_s_test
    else:
        s_test = None
        for _ in range(s_test_iterations):
            _s_test = compute_s_test(trainer=trainer, full_ids=full_ids, test_ds=test_ds,
                                     params_filter=params_filter, weight_decay=weight_decay,
                                     weight_decay_ignores=weight_decay_ignores,
                                     hvp_batch_size=hvp_batch_size, damp=s_test_damp,
                                     scale=s_test_scale, num_samples=s_test_num_samples)

            # Sum the values across runs
            if s_test is None:
                s_test = _s_test
            else:
                s_test = [a + b for a, b in zip(s_test, _s_test)]
        # Do the averaging across multiple random repeats of HVP (i.e., hyperparameter r)
        s_test = [a / s_test_iterations for a in s_test]
    # Flatten s_test for computational efficiency
    s_test_flat = utils.flatten(vec=s_test)

    utils.reset_layer_norm_structs()
    # Compute the final influence values
    train_ds = parent_utils.get_train_ds(trainer=trainer)
    inf_unnormed = torch.zeros([config.N_TRAIN + config.POISON_CNT], dtype=torch.float)
    inf_lderiv_vec, inf_lderiv_scalar = inf_unnormed.clone(), inf_unnormed.clone()
    inf_normed, inf_layer_norm = inf_unnormed.clone(), inf_unnormed.clone()
    inf_loss_norm = inf_unnormed.clone()
    itr = tqdm(full_ids, disable=config.QUIET)
    with itr as ex_tqdm:
        for idx in ex_tqdm:
            sample = parent_utils.get_ds_sample(idx=idx, ds=train_ds, trainer=trainer)
            grad_z = compute_gradients(trainer=trainer, sample=sample, params_filter=params_filter,
                                       weight_decay=weight_decay,
                                       weight_decay_ignores=weight_decay_ignores)
            with torch.no_grad():
                loss, acts = get_loss_with_logits(trainer=trainer, sample=sample)

            # Must call before flatten to keep layer structure
            layer_norm = utils.build_layer_norm(grad_z)
            grad_z = utils.flatten(grad_z)
            dot_val = -torch.dot(grad_z, s_test_flat)
            inf_unnormed[idx] = dot_val

            # Loss Derivative vector norm
            lderiv = derivative_of_loss(acts=acts, sample=sample)
            ld_norm = lderiv.norm()
            if ld_norm.item() <= 0:
                ld_norm.fill_(utils.MIN_LOSS)
            inf_lderiv_vec[idx] = dot_val / ld_norm
            # Loss derivative scalar norm
            lderiv = lderiv[0, parent_utils.get_sample_label(sample=sample)].abs_()
            if lderiv.item() == 0:
                lderiv.fill_(utils.MIN_LOSS)
            inf_lderiv_scalar[idx] = dot_val / lderiv

            # Normalize by just the loss
            if loss.item() <= 0:
                loss.fill_(utils.MIN_LOSS)
            inf_loss_norm[idx] = dot_val / loss

            # Normalize the gradient on each dimension individually to reduce underflows
            inf_normed[idx] = -torch.dot(grad_z / grad_z.norm(), s_test_flat)
            # Normalize by the layerwise norm
            inf_layer_norm[idx] = -torch.dot(grad_z / layer_norm, s_test_flat)

    # Accumulate all the results into a single tensor
    res = InfFuncTensors()
    res.ids = full_ids
    res.s_test = s_test
    res.inf_base = inf_unnormed[full_ids]
    res.inf_sim = inf_normed[full_ids]
    res.inf_sim_l = inf_layer_norm[full_ids]
    res.inf_loss_norm = inf_loss_norm[full_ids]
    res.inf_lderiv_scalar = inf_lderiv_scalar[full_ids]
    res.inf_lderiv_vec = inf_lderiv_vec[full_ids]
    return res


def compute_s_test(trainer, test_ds, full_ids: Tensor,
                   # train_data_loader: torch.utils.data.DataLoader,
                   params_filter: Optional[List[str]], hvp_batch_size: int,
                   weight_decay: Optional[float], weight_decay_ignores: Optional[List[str]],
                   damp: float, scale: float, num_samples: Optional[int] = None,
                   verbose: bool = True) -> List[torch.Tensor]:
    assert len(test_ds) == 1, "Only a single target example supported at a time"
    # all_x, all_y = [], []
    # for xs, ys in test_dl:
    #     all_x.append(xs)
    #     all_y.append(ys)

    grad_sample = parent_utils.get_ds_sample(idx=0, ds=test_ds, trainer=trainer)
    v = compute_gradients(trainer=trainer, sample=grad_sample,
                          params_filter=params_filter,
                          weight_decay=weight_decay, weight_decay_ignores=weight_decay_ignores)

    # Technically, it's hv^-1
    train_ds = parent_utils.get_train_ds(trainer=trainer)
    last_estimate = list(v).copy()
    cumulative_num_samples = 0
    # with tqdm(total=num_samples) as pbar:
    #     for data_loader in train_data_loaders:
    #         for i, inputs in enumerate(data_loader):
    with tqdm(total=num_samples, disable=config.QUIET) as pbar:
        i, samples = 0, []
        perm = torch.randperm(full_ids.shape[0])
        for idx in full_ids[perm]:
            # Extract the sample(s).
            single_sample = parent_utils.get_ds_sample(idx=idx, ds=train_ds, trainer=trainer)
            if hvp_batch_size == 1:
                samples = single_sample
            else:
                samples.append(single_sample)
                if len(samples) < hvp_batch_size:
                    continue
            this_estim = compute_hessian_vector_products(trainer=trainer, sample=samples,
                                                         vectors=last_estimate,
                                                         params_filter=params_filter,
                                                         weight_decay=weight_decay,
                                                         weight_decay_ignores=weight_decay_ignores)
            # Recursively calculate h_estimate
            # https://github.com/dedeswim/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_functions/hvp_grad.py#L118
            with torch.no_grad():
                new_estimate = [a + (1 - damp) * b - c / scale
                                for a, b, c in zip(v, last_estimate, this_estim)]

            pbar.update(1)
            if verbose:
                new_estimate_norm = new_estimate[0].norm().item()
                last_estimate_norm = last_estimate[0].norm().item()
                assert not np.isnan(new_estimate_norm), "HVP norm estimate cannot be NaN"
                estimate_norm_diff = new_estimate_norm - last_estimate_norm
                pbar.set_description(f"{new_estimate_norm:.2f} | {estimate_norm_diff:.2f}")

            cumulative_num_samples += 1
            last_estimate = new_estimate
            if num_samples is not None and i > num_samples:
                break
            # Reset the loop variables
            i, samples = i + 1, []

    # References:
    # https://github.com/kohpangwei/influence-release/blob/master/influence/genericNeuralNet.py#L475
    # Do this for each iteration of estimation
    # Since we use one estimation, we put this at the end
    inverse_hvp = [X / scale for X in last_estimate]

    # Sanity check
    # Note that in parallel settings, we should have `num_samples`
    # whereas in sequential settings we would have `num_samples + 2`.
    # This is caused by some loose stop condition. In parallel settings,
    # We only allocate `num_samples` data to reduce communication overhead.
    # Should probably make this more consistent sometime.
    if cumulative_num_samples not in [num_samples, num_samples + 2]:
        raise ValueError(f"cumulative_num_samples={cumulative_num_samples} f"
                         f"but num_samples={num_samples}: Untested Territory")

    return inverse_hvp


def compute_hessian_vector_products(trainer, sample: Union[dict, List[dict]],
                                    vectors: Tuple[torch.Tensor, ...],
                                    params_filter: Optional[List[str]],
                                    weight_decay: Optional[float],
                                    weight_decay_ignores: Optional[List[str]]) \
        -> Tuple[Tensor, ...]:
    r"""
    :param trainer: Trainer for the model under investigation
    :param sample: Training example(s) used to create the product
    :param vectors: Gradient vectors used for evaluation
    :param params_filter: Name of any neural network parameters not to be considered in the
                          Hessian vector product gradient.
    :param weight_decay: Weight decay hyperparameter to ensure accurate calculation of the loss
                         since \p torch's weight decay is handled by the optimizer.  Essentially
                         L2 regularization.
    :param weight_decay_ignores: Any parameters (e.g., bias) not considered in the weight decay
                                 (L2) regularization calculation.
    :return:
    """
    if params_filter is None:
        params_filter = []
    # Output is the parameterized gradient \nabla_{\theta} L(x, y)
    grad_tuple = compute_gradients(trainer=trainer, sample=sample, params_filter=params_filter,
                                   weight_decay=weight_decay,
                                   weight_decay_ignores=weight_decay_ignores)

    trainer.get_model().zero_grad()
    grad_grad_inputs = [param for name, param in trainer.get_model().named_parameters()
                        if name not in params_filter]

    # inputs: Inputs w.r.t. which the gradient is taken.  Simply the parameters since
    #         Hessian is w.r.t. \theta, i.e., $\nabla^{2}_{\theta}$
    # outputs: Outputs of the function.  This function is being differentiated.  Here we are
    #          differentiating the \nabla_{\theta} L(x,y).  This yields the Hessian.
    # create_graph=False (unlike in method compute_gradients) since no need to create graph as only
    #                    taking the Hessian and not higher order terms.
    grad_grad_tuple = torch.autograd.grad(outputs=grad_tuple, inputs=grad_grad_inputs,
                                          grad_outputs=vectors, only_inputs=True)

    return grad_grad_tuple
