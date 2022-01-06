__all__ = [
    "DTYPE",
    "TracInTensors",
    "build_layer_norm",
    "calc_percentile",
    "compute_grad",
    "configure_train_dataloader",
    "export_tracin_epoch_inf",
    "flatten_grad",
    "generate_wandb_results",
    "get_gas_log_flds",
    "get_tracin_log_flds",
    "get_tracincp_log_flds",
    "get_topk_indices",
    "log_vals_stats",
    "sort_ids_and_inf",
    "tracin_dot_product",
]

import copy
import dataclasses
import logging
import pickle as pk
from typing import List, NoReturn, Optional, Tuple, Union

from fastai.basic_data import DeviceDataLoader
import torch
from torch import BoolTensor, DoubleTensor, FloatTensor, LongTensor, Tensor
import torchvision.transforms as transforms
import wandb

from . import _settings as settings
from .. import _config as config
from .. import _wandb as wandb_utils
from .. import dirs
from .. import influence_utils
from ..influence_utils import InfluenceMethod
from ..influence_utils import nn_influence_utils
from .. import utils as parent_utils

DTYPE = torch.double


@dataclasses.dataclass
class SubepochTensors:
    # Dot product values
    dot_vals: Tensor
    # Loss values
    loss_vals: Tensor
    # GAS layerwise norm vals
    gas_vals: Tensor
    # Gradient norms
    grad_norms: Tensor
    # Dot producted normalized by norm
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
            if f.type == Tensor:
                clone = torch.zeros_like(self.__getattribute__(f.name))
                setattr(self, f.name, clone)
        self.dot_normed = self.dot_loss_normed = None  # noqa


@dataclasses.dataclass
class TracInTensors:
    # full_ds_ids: Full set of dataset IDs used
    full_ds_ids: LongTensor
    # full_bd_ids: Full set of backdoor IDs used
    full_bd_ids: LongTensor
    # tracincp: Modified in place tensor storing the TracInCP values
    tracincp_inf: Tensor
    # tracin_inf: Modified in place tensor storing the TracIn values
    tracin_inf: Tensor
    # tracin_sim: Modified in place tensor storing the TracIn similarity values
    tracin_sim: Tensor
    # gas_inf: Modified in place tensor storing the GAS values
    gas_sim: Tensor
    # gas_layer: GAS using layerwise norm instead of L2 norms
    gas_l_sim: Tensor
    # Subepoch Tensors
    subep: SubepochTensors

    def __init__(self, full_ds_ids: LongTensor, full_bd_ids: LongTensor,
                 id_numel: int, inf_numel: int):
        inf_base = torch.zeros([id_numel, inf_numel], dtype=DTYPE, requires_grad=False)
        lng_base = torch.zeros_like(inf_base, dtype=torch.long)
        # Initialize all the fields to the same tensor shape
        for f in dataclasses.fields(self):
            if f.type == LongTensor:
                setattr(self, f.name, lng_base.clone())
            elif f.type == Tensor:
                setattr(self, f.name, inf_base.clone())

        # Stores the adv/clean magnitude ratio
        self.magnitude_ratio = []
        # Store the IDs
        self.full_ds_ids = full_ds_ids
        self.full_bd_ids = full_bd_ids
        # # Store the number of zero gradients for each element
        # self.adv_zeros = torch.zeros([id_numel, inf_numel], dtype=torch.long).long()
        # self.tot_zeros = self.adv_zeros.clone()
        # Subepoch tensors
        self.subep = SubepochTensors(id_numel=id_numel, inf_numel=inf_numel)


def configure_train_dataloader(train_dl: DeviceDataLoader) -> DeviceDataLoader:
    r"""" Configure the DeviceDataLoader for use in TracIn """
    # Switch to the test transform and update the train dataloader to not drop points/shuffle
    ds = copy.copy(train_dl.dl.dataset)
    ds.set_transform(config.get_test_tfms())  # noqa
    # Cannot use the new method since torch does not let you change the dataset of an initialized
    # dataloader
    new_tr_dl = DeviceDataLoader.create(ds, bs=1, device=train_dl.device,
                                        num_workers=train_dl.num_workers,
                                        drop_last=False, shuffle=False, tfms=train_dl.tfms)
    return new_tr_dl


def sort_ids_and_inf(inf_arr: Tensor, bd_ids_arr: Tensor,
                     ds_ids_arr: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    r""" Helper method for sorting the IDs and in """
    assert inf_arr.dtype == DTYPE, "Influence array is not the correct datatype"
    influence_utils.check_bd_ids_contents(bd_ids=bd_ids_arr)
    influence_utils.check_duplicate_ds_ids(ds_ids=ds_ids_arr)
    assert bd_ids_arr.shape[0] == ds_ids_arr.shape[0] == inf_arr.shape[0], "Num ele mismatch"

    ord_ids = torch.argsort(inf_arr, dim=0, descending=True)

    ord_inf = inf_arr.clone()[ord_ids]
    ord_bd_ids, ord_ds_ids = bd_ids_arr.clone()[ord_ids], ds_ids_arr.clone()[ord_ids]
    return ord_inf, ord_bd_ids, ord_ds_ids


def export_tracin_epoch_inf(all_in: bool, block: parent_utils.ClassifierBlock,
                            ep_inf: List[Tuple[Tensor, Tensor, Tensor]]) -> NoReturn:
    r""" Backup-up the TracIn data for later post-processing """
    outdir = dirs.RES_DIR / config.DATASET.name.lower() / "tracin" / block.name().lower() / "ep-inf"
    outdir.mkdir(parents=True, exist_ok=True)

    desc = "all-in" if all_in else "sep"
    path = parent_utils.construct_filename(prefix=f"ep-inf-{desc}", out_dir=outdir, file_ext="pk",
                                           add_ds_to_path=False, add_timestamp=True)
    with open(str(path), "wb+") as f_out:
        pk.dump(ep_inf, f_out)


def compute_grad(block, ep_wd, _x: Tensor, _y: Tensor, is_dbl: bool = False,
                 flatten: bool = False) \
        -> Tuple[Tensor, Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
    r"""
    Helper method to standardize gradient computation
    :return: Tuple of the loss, output activations, and gradient respectively
    """
    # assert _x.shape[0] == 1 and _y.shape[0] == 1, "Only single example supported"
    if is_dbl:
        _x = _x.double()
        # Need to cast to double when using BCE loss
        if config.DATASET.is_mnist() or config.DATASET.is_cifar():
            _y = _y.double()
    all_loss, all_acts, all_grads = [], [], []
    for i in range(_x.shape[0]):
        tmp_x, tmp_y = _x[i:i + 1], _y[i:i + 1]
        loss, acts, grad = nn_influence_utils.compute_gradients(device=parent_utils.TORCH_DEVICE,
                                                                model=block, n_gpu=0,
                                                                f_loss=block.loss.calc_train_loss,
                                                                x=tmp_x, y=tmp_y,
                                                                weight_decay=ep_wd,
                                                                params_filter=None,
                                                                weight_decay_ignores=None,
                                                                create_graph=False,
                                                                conv_only=False,
                                                                return_loss=True,
                                                                return_acts=True)
        all_loss.append(loss.unsqueeze(dim=0).detach())
        all_acts.append(acts.unsqueeze(dim=0).detach())
        if flatten:
            grad = flatten_grad(grad).detach()
        else:
            grad = [vec.detach() for vec in grad]
        all_grads.append(grad)
    all_loss, all_acts = torch.cat(all_loss, dim=0), torch.cat(all_acts, dim=0)
    return all_loss, all_acts, all_grads  # noqa


def build_layer_norm(grad) -> Union[FloatTensor, DoubleTensor]:
    r""" Construct a layerwise norm vector """
    layer = copy.deepcopy(grad)
    for ex_vec in layer:
        for i, layer_vec in enumerate(ex_vec):
            if layer_vec is None:
                continue
            layer_vec = layer_vec.detach()
            norm = layer_vec.norm()
            if norm.item() <= 0:
                norm = settings.MIN_NORM
            layer_vec.fill_(norm)
            ex_vec[i] = layer_vec
    return flatten_grad(layer)


def tracin_dot_product(block32: parent_utils.ClassifierBlock,
                       grad_targ32: FloatTensor, grad32_layer: Tensor,
                       block64: parent_utils.ClassifierBlock,
                       grad_targ64: DoubleTensor,
                       grad64_layer: DoubleTensor,
                       id_val: Tensor,
                       subep_tensors: SubepochTensors,
                       ds, id_map: dict, ep_wd: Optional[float]) -> NoReturn:
    r"""
    Computes the TracIn dot product

    :param block32: Block for use with floats (i.e., float32)
    :param id_val:
    :param grad_targ32: 32-bit version of target gradient vector
    :param grad32_layer: 32-bit gradient layerwise norm
    :param block64: Block for use with floats (i.e., float64)
    :param grad_targ64: 64-bit version of target gradient vector
    :param grad64_layer: 64-bit gradient layerwise norm
    :param ds:
    :param id_map:
    :param ep_wd: Weight decay for the epoch (if applicable)
    :param subep_tensors: Subepoch tensors
    """
    batch_tensors = ds[[id_map[id_val.item()]]]
    batch = block32.organize_batch(batch_tensors, process_mask=True, include_holdout=True)
    if config.DATASET.is_speech(): batch.filter_by_lbl(config.POIS_CLS)
    assert len(batch) == 1, "Only singleton batches supported"

    def _calc_layer_norm() -> Tuple[Tensor, Tensor]:
        return flatten_grad(grad_x), build_layer_norm(grad_x)  # noqa

    def _calc_prods() -> Tuple[Tensor, Tensor]:
        return (grad_dot_prod(grad_x, grad_targ).double().cpu(),
                grad_dot_prod(grad_x, grad_x).double().cpu())

    if batch.skip():
        return

    # Calculate 32-bit
    loss, _, grad_x = compute_grad(block32, ep_wd, batch.xs, batch.lbls, flatten=False)
    grad_x, x_layer = _calc_layer_norm()
    grad_targ = grad_targ32
    fin_dot, dot_prod = _calc_prods()
    gas_vec = _calculate_gas_layer_norm(grad_x=grad_x, layer_x=x_layer,
                                        grad_targ=grad_targ32, layer_targ=grad32_layer)

    mask = (fin_dot == 0).logical_or(dot_prod <= 0).to(parent_utils.TORCH_DEVICE)
    if torch.any(mask).item():
        loss, _, grad_x = compute_grad(block64, ep_wd, batch.xs, batch.lbls,
                                         is_dbl=True, flatten=False)
        grad_x, x_layer = _calc_layer_norm()
        grad_targ = grad_targ64
        fin_dot, dot_prod = _calc_prods()

        gas_vec[mask] = _calculate_gas_layer_norm(grad_x=grad_x, layer_x=x_layer,
                                                  grad_targ=grad_targ[mask],
                                                  layer_targ=grad64_layer[mask])

    # Update the influence arrays using the 32 bit values if training example value is not all
    # zeros and the dot product is not zero
    dot_prod[dot_prod == 0] = settings.MIN_NORM
    subep_tensors.grad_norms[:, id_val] = dot_prod

    subep_tensors.dot_vals[:, id_val] = fin_dot.double().cpu()
    loss[loss <= 0] = settings.MIN_LOSS
    subep_tensors.loss_vals[:, id_val] = loss.double().cpu()
    subep_tensors.gas_vals[:, id_val] = gas_vec


def _calculate_gas_layer_norm(grad_x: Tensor, layer_x: Tensor, grad_targ: Tensor,
                              layer_targ: Tensor) -> DoubleTensor:
    # GrAIn layerwise norm only by train x layerwise norm
    grain_vec = grad_targ * (grad_x / layer_x)  # noqa
    # GAS layerwise norm also normalizes by target norm
    gas_vec = grain_vec / layer_targ  # noqa
    assert gas_vec.numel() == grad_targ.numel(), "Unexpected size of GAS layerwise norm"
    return torch.sum(gas_vec, dim=1).double().cpu()


def grad_dot_prod(grad_1: Tensor, grad_2: Tensor) -> Tensor:
    r""" Gradient dot product """
    # assert grad_1.shape == grad_2.shape, "Shape mismatch"
    # assert prod.shape == grad_1.shape, "Weird shape after product"
    # return torch.dot(grad_1, grad_2)
    prod = grad_1 * grad_2
    # assert grad_1.shape == prod.shape, "Shape mismatch.  Shape should not change"
    return torch.sum(prod, dim=1)


def flatten_grad(grad: List[List[Tensor]]) -> Union[FloatTensor, DoubleTensor]:
    r""" Flattens gradients into a single contiguous vector """
    # noinspection PyTypeChecker
    flt = [torch.cat([v.detach().view([-1]) for v in vec if v is not None], dim=0)
           for vec in grad]
    flt = [vec.unsqueeze(dim=0) for vec in flt]
    return torch.cat(flt, dim=0).detach()  # noqa


def get_topk_indices(grad: Tensor, frac: float) -> Tensor:
    assert 0 < frac < 1, "Fraction of indices to keep"
    k = int(frac * grad.numel())
    _, idx = torch.topk(grad.abs(), k=k)
    mask = torch.zeros_like(grad, dtype=torch.bool)
    mask[idx] = True
    return mask


def generate_wandb_results(block: parent_utils.ClassifierBlock,
                           method: influence_utils.InfluenceMethod,
                           inf_vals: Tensor, ds_ids: LongTensor, bd_ids: LongTensor,
                           train_dl: DeviceDataLoader,
                           ex_id: Optional[int] = None) -> NoReturn:
    r"""
    Generate a summary of the results to using W&B

    :param block: Block being analyzed
    :param method: Influence estimation method
    :param inf_vals: Influence values
    :param ds_ids:
    :param bd_ids:
    :param train_dl:
    :param ex_id:
    """
    if not config.USE_WANDB:
        return

    logging.debug(f"Generating W&B {method.value} influence results table")

    # Sort all the tensors to be safe
    influence_utils.check_bd_ids_contents(bd_ids=bd_ids)
    influence_utils.check_duplicate_ds_ids(ds_ids=ds_ids)
    inf_vals, bd_ids, ds_ids = sort_ids_and_inf(inf_arr=inf_vals, bd_ids_arr=bd_ids,
                                                ds_ids_arr=ds_ids)

    # Get rank of each training example
    ramp = torch.arange(1, ds_ids.numel() + 1, dtype=torch.long)
    ds_rank = torch.full([torch.max(ds_ids) + 1], fill_value=-1, dtype=torch.long)
    ds_inf = torch.zeros_like(ds_rank, dtype=DTYPE)
    ds_rank[ds_ids], ds_inf[ds_ids] = ramp, inf_vals

    # create a wandb.Table() with corresponding columns
    columns = ["id", "image", "inf", "rank", "label", "is_bd"]
    #  Construct images of all results
    to_pil = transforms.ToPILImage()
    all_res = []
    train_dl = configure_train_dataloader(train_dl=train_dl)
    for batch_tensors in train_dl:
        batch = block.organize_batch(batch_tensors, process_mask=True, include_holdout=True)
        if batch.skip():
            continue

        id_val = batch.ds_ids.item()  # Id number of the example
        is_bd = influence_utils.label_ids(bd_ids=batch.bd_ids) == influence_utils.BACKDOOR_LABEL
        # Construct the results
        tr_ex = [id_val, wandb.Image(to_pil(batch.xs[0].clamp(0, 1))),
                 ds_inf[id_val].item(), ds_rank[id_val].item(),
                 batch.lbls.item(), is_bd.item()]
        all_res.append(tr_ex)

    # Generate the table
    run = wandb.init(project=wandb_utils.get_proj_name())
    inf_table = wandb.Table(data=all_res, columns=columns)
    flds = [method.value]
    if ex_id is not None:
        flds.append(f"ex={ex_id}")
    flds.append("inf-sum")
    run.log({"_".join(flds).lower(): inf_table})


def log_vals_stats(block: parent_utils.ClassifierBlock, res_type: InfluenceMethod,
                   ep: Optional[int], subep: Optional[int], norms: Tensor,
                   ex_id: Optional[int]) -> NoReturn:
    r""" Standardizing method for logging norm mean and standard deviation """
    header = influence_utils.build_log_start_flds(block=block, ep=ep, subepoch=subep,
                                                  res_type=res_type, ex_id=ex_id)
    if norms.dtype != DTYPE:
        norms = norms.type(DTYPE)  # noqa

    # Calculate quantiles
    quantiles = torch.tensor([0.25, 0.5, 0.75], dtype=DTYPE)
    names = ["25%-Quartile", "Median", "75%-Quartile"]  # noqa
    quant_vals = torch.quantile(norms, q=quantiles)
    for name, val in zip(names, quant_vals.tolist()):
        logging.info(f"{header} {name}: {val:.3E}")
    # # Interquartile range
    # val = quant_vals[-2] - quant_vals[1]
    # logging.info(f"{header} IQR: {val.item():.6E}")
    #
    # std, mean = torch.std_mean(norms, unbiased=True)
    # for val, val_name in zip((mean, std), ("Mean", "Stdev")):
    #     logging.info(f"{header} {val_name}: {val.item():.6E}")


def is_grad_zero(grad: Tensor) -> BoolTensor:
    r""" Returns \p True if the gradient dot product is zero """
    dot_prod = grad_dot_prod(grad, grad)
    return dot_prod == 0


def calc_percentile(ref: Union[float, Tensor], vals: Tensor) -> float:
    r""" Calculates percentile of \p ref w.r.t. \p vals """
    if isinstance(ref, Tensor):
        assert ref.numel() == 1, "Only single reference value supported"
        ref = ref.item()

    assert len(vals.shape) == 1, "Only 1-D tensors supported"
    vals, _ = torch.sort(vals)
    # Handle base cases
    if ref <= vals[0].item():
        return 0.
    elif ref >= vals[-1].item():
        return 1.
    ref = torch.tensor([ref])
    idx = torch.searchsorted(vals, ref, right=False).item()
    # Verify values in range
    ref, left, right = ref.item(), vals[idx - 1].item(), vals[idx].item()
    assert left <= ref <= right, "Unexpected range vals"
    return (idx + (ref - left) / (right - left)) / vals.numel()


def get_gas_log_flds(tensors: TracInTensors) -> Tuple[Tuple[Tensor, InfluenceMethod, str], ...]:
    r""" Construct the GAS result fields for logging """
    return (
        (tensors.gas_sim, InfluenceMethod.GAS, "gas"),
        (tensors.gas_l_sim, InfluenceMethod.GAS_L, "gas-layer"),
    )


def get_tracin_log_flds(tensors: TracInTensors) -> Tuple[Tuple[Tensor, InfluenceMethod, str], ...]:
    r""" Construct the TracIn result fields for logging """
    return (
        (tensors.tracin_inf, InfluenceMethod.TRACIN, "tracin"),
        (tensors.tracin_sim, InfluenceMethod.TRACIN_SIM, "tracin-sim"),
    )


def get_tracincp_log_flds(tensors: TracInTensors) \
        -> Tuple[Tuple[Tensor, InfluenceMethod, str], ...]:
    r""" Construct the GrAIn-based TracIn result fields for logging """
    return (
        (tensors.tracincp_inf, InfluenceMethod.TRACINCP, "tracincp"),
    )
