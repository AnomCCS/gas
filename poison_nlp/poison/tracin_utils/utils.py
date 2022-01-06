__all__ = [
    "TracInTensors",
    "export_tracin_epoch_inf",
    "get_gas_log_flds",
    # "get_init_only_log_flds",
    "get_tracin_log_flds",
    "get_tracincp_log_flds",
    "grad_dot_prod",
    "log_vals_stats",
    "sort_ids_and_inf",
]

import dataclasses
import dill as pk
import logging
from typing import List, NoReturn, Optional, Tuple, Union

import torch
from torch import BoolTensor, LongTensor, Tensor

from . import _settings as settings
from .. import dirs
from .. import influence_utils
from ..influence_utils import InfluenceMethod
from .. import utils as parent_utils

DTYPE = torch.float32


@dataclasses.dataclass
class SubepochTensors:
    # Dot product values
    dot_vals: Tensor
    # GAS layer norm vals
    gas_vals: Tensor
    # Gradient norms
    grad_norms: Tensor
    # Dot product normalized by L2 norm
    dot_normed: Tensor

    def __init__(self, id_numel: int, inf_numel: int):
        # Initialize all the fields to the same tensor shape
        for f in dataclasses.fields(self):
            if f.type == BoolTensor:
                dtype = torch.bool
            elif f.type == LongTensor:
                dtype = torch.long
            elif f.type == Tensor:
                dtype = DTYPE
            else:
                raise ValueError("Unknown type to copy")
            tensor = torch.zeros([id_numel, inf_numel], dtype=dtype, requires_grad=False)
            setattr(self, f.name, tensor)

    def reset(self) -> NoReturn:
        r""" Reset the tensor at the start of an epoch """
        for f in dataclasses.fields(self):
            val = self.__getattribute__(f.name)
            if val is not None and f.type == Tensor:
                clone = torch.zeros_like(val)
                setattr(self, f.name, clone)
        self.dot_normed = self.dot_loss_normed = None  # noqa


@dataclasses.dataclass
class TracInTensors:
    # full_ids: Full set of IDs used
    full_ids: LongTensor
    # tracincp: TracInCP modified in place
    tracincp: Tensor
    # gas_inf: Modified in place tensor storing the GAS values
    gas_inf: Tensor
    # gas_layer: GAS using layer norm instead of normal norms
    gas_layer: Tensor
    # tracin_inf: Modified in place tensor storing the TracIn influence values
    tracin_inf: Tensor
    # tracin_sim: Modified in place tensor storing the TracIn similarity values
    tracin_sim: Tensor
    # tracin_inf: Modified in place tensor storing the TracIn similarity layerwise values
    tracin_sim_l: Tensor
    # Subepoch Tensors
    subep: SubepochTensors

    def __init__(self, full_ids: LongTensor, id_numel: int, inf_numel: int):
        inf_base = torch.zeros([id_numel, inf_numel], dtype=DTYPE, requires_grad=False)
        # Initialize all the fields to the same tensor shape
        for f in dataclasses.fields(self):
            if f.type == Tensor:
                setattr(self, f.name, inf_base.clone())

        # Store the IDs
        self.full_ids = full_ids
        # Subepoch tensors
        self.subep = SubepochTensors(inf_numel=inf_numel, id_numel=id_numel)


def tracin_dot_product(trainer, subep_tensors: SubepochTensors,
                       id_val: LongTensor, grad_targ: Tensor, grad_targ_layer: Tensor,
                       train_ds, ep_wd: Optional[float], decoder_only: bool) -> NoReturn:
    r"""
    Computes the TracIn dot product

    :param trainer: Block of interest
    :param id_val:
    :param grad_targ:
    :param train_ds:
    :param ep_wd: Weight decay for the epoch (if applicable)
    :param decoder_only: If \p True, only consider gradients across the decoder
    :param subep_tensors: Subepoch tensors
    :param grad_targ_layer: Target gradient layer norm
    """
    grad_x, x_layer = get_grads_and_layer(id_val=id_val, ds=train_ds, trainer=trainer,
                                          decoder_only=decoder_only, wd=ep_wd)
    subep_tensors.dot_vals[:, id_val] = grad_dot_prod(grad_x, grad_targ).cpu()

    l2_dot = grad_dot_prod(grad_x, grad_x)  # Sqrt done for all examples at once
    l2_dot[l2_dot == 0] = settings.MIN_NORM
    # if l2_dot.item() <= 0:
    #     l2_dot.fill_(settings.MIN_NORM)
    #     subep_tensors.tot_zeros[id_val] += 1
    #     if influence_utils.is_pois(ids=id_val).item():
    #         subep_tensors.adv_zeros[id_val] += 1
    subep_tensors.grad_norms[:, id_val] = l2_dot.cpu()

    # GAS Layer Norm also normalizes by target norm
    subep_tensors.gas_vals[:, id_val] = grad_dot_prod(grad_targ_layer, x_layer).cpu()


def compute_grad(trainer, sample: dict, wd: Optional[float], decoder_only: bool,
                 flatten: bool = True) -> Tuple[Tensor, Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
    r""" Helper method to standardize gradient computation """
    grad = influence_utils.compute_gradients(trainer=trainer, sample=sample,
                                             weight_decay=wd,
                                             weight_decay_ignores=None,
                                             params_filter=None, create_graph=False,
                                             use_decoder_only=decoder_only)
    if flatten:
        grad = flatten_grad(grad).detach()
    else:
        grad = [vec.detach() for vec in grad if vec is not None]
    return grad  # noqa


def flatten_grad(grad: Tuple[Tensor, ...]) -> Tensor:
    r""" Flattens gradients into a single contiguous vector """
    return torch.cat([vec.view([-1]) for vec in grad if vec is not None], dim=0)


# def grad_dot_prod(grad_1: Tensor, grad_2: Tensor) -> Tensor:
#     r""" Gradient dot product """
#     return torch.dot(grad_1, grad_2)


def grad_dot_prod(grad_1: Tensor, grad_2: Tensor) -> Tensor:
    r""" Gradient dot product """
    # assert grad_1.shape == grad_2.shape, "Shape mismatch"
    # assert prod.shape == grad_1.shape, "Weird shape after product"
    # return torch.dot(grad_1, grad_2)
    prod = grad_1 * grad_2
    # assert grad_1.shape == prod.shape, "Shape mismatch.  Shape should not change"
    return torch.sum(prod, dim=1)


def sort_ids_and_inf(inf_arr: Tensor, ids_arr: LongTensor) -> Tuple[Tensor, LongTensor]:
    r""" Helper method for sorting the IDs and in """
    assert inf_arr.dtype == torch.float, "Influence array is not floats"
    assert ids_arr.shape[0] == inf_arr.shape[0], "Num ele mismatch"
    assert ids_arr.dtype == torch.long, "ID arrays is not integers"

    ord_ids = torch.argsort(inf_arr, dim=0, descending=True)

    ord_inf = inf_arr.clone()[ord_ids]
    ord_ids = ids_arr.clone()[ord_ids]
    return ord_inf, ord_ids


def export_tracin_epoch_inf(all_in: bool,
                            ep_inf: List[Tuple[Tensor, Tensor, Tensor]]) -> NoReturn:
    r""" Backup-up the TracIn data for later post-processing """
    outdir = dirs.RES_DIR / "tracin" / "ep-inf"
    outdir.mkdir(parents=True, exist_ok=True)

    desc = "all-in" if all_in else "sep"
    path = parent_utils.construct_filename(prefix=f"ep-inf-{desc}", out_dir=outdir, file_ext="pk",
                                           add_timestamp=True)
    with open(str(path), "wb+") as f_out:
        pk.dump(ep_inf, f_out)


def log_vals_stats(res_type: InfluenceMethod, ep: Optional[int], n_updates: Optional[int],
                   norms: Tensor, ex_id: Optional[int] = None) -> NoReturn:
    r""" Standardizing method for logging norm mean and standard deviation """
    header = influence_utils.build_log_start_flds(ep=ep, n_updates=n_updates,
                                                  res_type=res_type, ex_id=ex_id)

    # Calculate quantiles
    quantiles = torch.tensor([0., 0.25, 0.5, 0.75, 1.], dtype=DTYPE)
    names = ["Min", "25%-Quartile", "Median", "75%-Quartile", "Max"]
    quant_vals = torch.quantile(norms, q=quantiles)
    for name, val in zip(names, quant_vals.tolist()):
        logging.info(f"{header} {name}: {val:.6E}")
    # Interquartile range
    val = quant_vals[-2] - quant_vals[1]
    logging.info(f"{header} IQR: {val.item():.6E}")

    std, mean = torch.std_mean(norms, unbiased=True)
    for val, val_name in zip((mean, std), ("Mean", "Stdev")):
        logging.info(f"{header} {val_name}: {val.item():.6E}")


def get_tracin_log_flds(tensors: TracInTensors) -> Tuple[Tuple[Tensor, InfluenceMethod, str], ...]:
    r""" Construct the TracIn result fields for logging """
    return (
        (tensors.tracin_inf, InfluenceMethod.TRACIN, "tracin"),
        (tensors.tracin_sim, InfluenceMethod.TRACIN_SIM, "tracin-sim"),
        (tensors.tracin_sim_l, InfluenceMethod.TRACIN_SIM_L, "tracin-sim-l"),
    )


def get_gas_log_flds(tensors: TracInTensors) -> Tuple[Tuple[Tensor, InfluenceMethod, str], ...]:
    r""" Construct the GAS result fields for logging """
    return (
        (tensors.gas_inf, InfluenceMethod.GAS, "gas"),
        (tensors.gas_layer, InfluenceMethod.GAS_L, "gas-layer"),
    )


def get_tracincp_log_flds(tensors: TracInTensors) \
        -> Tuple[Tuple[Tensor, InfluenceMethod, str], ...]:
    r""" Construct the TracInCP result fields for logging """
    return (
        (tensors.tracincp, InfluenceMethod.TRACINCP, "tracincp"),
    )


def get_grads_and_layer(id_val: Union[int, LongTensor], trainer, ds, wd: Optional[float],
                        decoder_only: bool, toggle_sample_label: bool = False,
                        lbl_val: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    r""" Gets the gradient and layer norm information """
    sample = parent_utils.get_ds_sample(idx=id_val, ds=ds, trainer=trainer)
    if toggle_sample_label:
        assert lbl_val is None, "Label value cannot be specified with label flip"
        label = sample["target"].item()
        assert label in {0, 1}, "Only binary labels supported"
        sample["target"].fill_(label ^ 1)
    elif lbl_val is not None:
        sample["target"].fill_(lbl_val)
    grad = compute_grad(trainer=trainer, sample=sample, wd=wd,
                        decoder_only=decoder_only, flatten=False)
    layer_norm = influence_utils.build_layer_norm(grad)
    grad = flatten_grad(grad)
    layer_norm = grad / layer_norm
    return grad.unsqueeze(dim=0), layer_norm.unsqueeze(dim=0)
