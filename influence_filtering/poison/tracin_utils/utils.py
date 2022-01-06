__all__ = [
    "DTYPE",
    "TracInTensors",
    "build_layer_norm",
    "compute_grad",
    "configure_train_dataloader",
    "export_tracin_epoch_inf",
    "flatten_grad",
    "generate_wandb_results",
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
from .. import dirs
from .. import influence_utils
from ..influence_utils import InfluenceMethod, nn_influence_utils
from .. import utils as parent_utils

DTYPE = torch.double


@dataclasses.dataclass
class SubepochTensors:
    # Dot product values
    dot_vals: Tensor
    # GAS layerwise norm vals
    gas_vals: Tensor
    # Gradient norms
    grad_norms: Tensor
    # Dot producted normalized by norm
    dot_normed: Tensor
    # Last (sub)epoch predict
    pred: LongTensor

    def __init__(self, inf_numel: int):
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
            tensor = torch.zeros([inf_numel], dtype=dtype, requires_grad=False)
            setattr(self, f.name, tensor)

    def reset(self) -> NoReturn:
        r""" Reset the tensor at the start of an epoch """
        for f in dataclasses.fields(self):
            if f.type == Tensor:
                clone = torch.zeros_like(self.__getattribute__(f.name))
                setattr(self, f.name, clone)
        self.dot_normed = None  # noqa


@dataclasses.dataclass
class TracInTensors:
    # full_ids: Full set of dataset IDs used
    full_ids: LongTensor
    # TracInCP: Modified in place tensor simulating TracInCP
    tracincp: Tensor
    # gas_inf: Modified in place tensor storing the GAS values
    gas_sim: Tensor
    # gas_layer: GAS using layer norm instead of normal norms
    gas_l: Tensor
    # tracin_inf: Modified in place tensor storing the TracIn values
    tracin_inf: Tensor
    # tracin_sim: Modified in place tensor storing the TracIn similarity values
    tracin_sim: Tensor
    # Subepoch Tensors
    subep: SubepochTensors

    def __init__(self, full_ids: LongTensor, inf_numel: int):
        inf_base = torch.zeros([inf_numel], dtype=DTYPE, requires_grad=False)
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
        self.full_ids = full_ids
        # Store the number of zero gradients for each element
        self.tot_zeros = torch.zeros([inf_numel], dtype=torch.long,  # type: LongTensor # noqa
                                     requires_grad=False)
        # Subepoch tensors
        self.subep = SubepochTensors(inf_numel=inf_numel)


def configure_train_dataloader(train_dl: DeviceDataLoader,
                               use_img_tfms: bool = False) -> DeviceDataLoader:
    r"""" Configure the DeviceDataLoader for use in TracIn """
    # Switch to the test transform and update the train dataloader to not drop points/shuffle
    ds = copy.copy(train_dl.dl.dataset)
    tfms = config.get_img_tfms() if use_img_tfms else config.get_test_tfms()
    ds.set_transform(tfms)  # noqa
    # Cannot use the new method since torch does not let you change the dataset of an initialized
    # dataloader
    new_tr_dl = DeviceDataLoader.create(ds, bs=1, device=train_dl.device,
                                        num_workers=train_dl.num_workers,
                                        drop_last=False, shuffle=False, tfms=train_dl.tfms)
    return new_tr_dl


def sort_ids_and_inf(inf_arr: Tensor, ids_arr: LongTensor) -> Tuple[Tensor, LongTensor]:
    r""" Helper method for sorting the IDs and in """
    assert inf_arr.dtype == DTYPE, "Influence array is not the correct datatype"
    assert ids_arr.shape[0] == inf_arr.shape[0], "Num ele mismatch"

    ord_ids = torch.argsort(inf_arr, dim=0, descending=True)

    ord_inf = inf_arr.clone()[ord_ids]
    ids = ids_arr.clone()[ord_ids]
    return ord_inf, ids


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


def compute_grad(block, ep_wd, _x: Tensor, _y: Tensor, is_dbl: bool = False, flatten: bool = True) \
        -> Tuple[Tensor, Tensor, Union[Tensor, Tuple[Tensor, ...]]]:
    r"""
    Helper method to standardize gradient computation
    :return: Tuple of the loss, output activations, and gradient respectively
    """
    assert _x.shape[0] == 1 and _y.shape[0] == 1, "Only single example supported"
    if is_dbl:
        _x = _x.double()
        # Need to cast to double when using BCE loss
        if config.DATASET.is_cifar():
            _y = _y.double()
    loss, acts, grad = nn_influence_utils.compute_gradients(device=parent_utils.TORCH_DEVICE,
                                                            model=block, n_gpu=0,
                                                            f_loss=block.loss.calc_train_loss,
                                                            x=_x, y=_y, weight_decay=ep_wd,
                                                            params_filter=None,
                                                            weight_decay_ignores=None,
                                                            create_graph=False,
                                                            conv_only=False,
                                                            return_loss=True, return_acts=True)
    if flatten:
        grad = flatten_grad(grad).detach()
    else:
        grad = [vec.detach() for vec in grad]
    return loss.detach(), acts.detach(), grad  # noqa


layer_norm32 = layer_norm64 = None


def build_layer_norm(grad) -> Tensor:
    r""" Construct a layer norm vector """
    global layer_norm32, layer_norm64
    if grad[0].dtype == torch.float:
        if layer_norm32 is None:
            layer_norm32 = [vec.clone().detach() for vec in grad]
        layer_norm = layer_norm32
    else:
        if layer_norm64 is None:
            layer_norm64 = [vec.clone().detach() for vec in grad]
        layer_norm = layer_norm64

    assert len(layer_norm) == len(grad), "Unexpected length mismatch"
    for layer, vec in zip(layer_norm, grad):  # type: Tensor, Tensor
        norm = vec.norm()
        if norm == 0:
            norm = settings.MIN_NORM
        layer.fill_(norm)
    return flatten_grad(layer_norm)  # noqa


def tracin_dot_product(block32: parent_utils.ClassifierBlock,
                       grad_targ32: FloatTensor, grad32_layer: Tensor,
                       block64: parent_utils.ClassifierBlock,
                       grad_targ64: DoubleTensor, grad64_layer: Tensor,
                       id_val: Tensor,
                       subep_tensors: SubepochTensors,
                       ds, id_map: dict, ep_wd: Optional[float],
                       always_use_dbl: bool) -> NoReturn:
    r"""
    Computes the TracIn dot product

    :param block32: Block for use with floats (i.e., float32)
    :param block64: Block for use with doubles (i.e., float64)
    :param id_val:
    :param grad_targ32: 32-bit version of target gradient vector
    :param grad32_layer: 32-bit gradient layerwise norm
    :param grad_targ64: 64-bit version of target gradient vector
    :param grad64_layer: 64-bit gradient layerwise norm
    :param ds:
    :param id_map:
    :param ep_wd: Weight decay for the epoch (if applicable)
    :param always_use_dbl: Always use the double version of the model
    :param subep_tensors: Subepoch tensors
    """
    batch_tensors = ds[[id_map[id_val.item()]]]
    batch = block32.organize_batch(batch_tensors)
    assert len(batch) == 1, "Only singleton batches supported"

    def _calc_layer_norm() -> Tuple[Tensor, Tensor]:
        return flatten_grad(grad_x), build_layer_norm(grad_x)  # noqa

    def _calc_prods() -> Tuple[Tensor, Tensor]:
        return grad_dot_prod(grad_x, grad_targ), grad_dot_prod(grad_x, grad_x)

    if not always_use_dbl:
        loss, acts, grad_x = compute_grad(block32, ep_wd, batch.xs, batch.lbls, flatten=False)
        grad_x, x_layer = _calc_layer_norm()
        grad_targ, layer_targ = grad_targ32, grad32_layer
        fin_dot, dot_prod = _calc_prods()
    # Handle the case where one of the results is 0
    if always_use_dbl or 0. in (fin_dot, dot_prod):  # noqa
        loss, acts, grad_x = compute_grad(block64, ep_wd, batch.xs, batch.lbls, is_dbl=True,
                                          flatten=False)
        grad_x, x_layer = _calc_layer_norm()
        grad_targ, layer_targ = grad_targ64, grad64_layer
        fin_dot, dot_prod = _calc_prods()

    if dot_prod <= 0:
        dot_prod = settings.MIN_NORM

    # Add the gradient dot product
    subep_tensors.dot_vals[id_val] = fin_dot
    subep_tensors.grad_norms[id_val] = dot_prod

    # GrAIn layerwise norm only by train x layerwise norm
    grain_vec = grad_targ * (grad_x / x_layer)  # noqa
    assert grain_vec.numel() == grad_targ.numel(), "Unexpected size of GrAIn layerwise norm"
    # GAS layerwise norm also normalizes by target layerwise norm
    gas_vec = grain_vec / layer_targ  # noqa
    assert gas_vec.numel() == grad_targ.numel(), "Unexpected size of GAS layerwise norm"
    subep_tensors.gas_vals[id_val] = torch.sum(gas_vec)


def grad_dot_prod(grad_1: Tensor, grad_2: Tensor) -> Tensor:
    r""" Gradient dot product """
    # assert grad_1.shape == grad_2.shape, "Shape mismatch"
    # assert prod.shape == grad_1.shape, "Weird shape after product"
    return torch.dot(grad_1, grad_2)


def flatten_grad(grad: Tuple[Tensor, ...]) -> Union[FloatTensor, DoubleTensor]:
    r""" Flattens gradients into a single continguous vector """
    return torch.cat([vec.view([-1]) for vec in grad if vec is not None], dim=0)  # noqa


def get_topk_indices(grad: Tensor, frac: float) -> Tensor:
    assert 0 < frac < 1, "Fraction of indices to keep"
    k = int(frac * grad.numel())
    _, idx = torch.topk(grad.abs(), k=k)
    mask = torch.zeros_like(grad, dtype=torch.bool)
    mask[idx] = True
    return mask


def generate_wandb_results(block: parent_utils.ClassifierBlock,
                           method: InfluenceMethod,
                           inf_vals: Tensor, ids: LongTensor,
                           train_dl: DeviceDataLoader,
                           ex_id: Optional[int] = None) -> NoReturn:
    r"""
    Generate a summary of the results to using W&B

    :param block: Block being analyzed
    :param method: Influence estimation method
    :param inf_vals: Influence values
    :param ids:
    :param train_dl:
    :param ex_id:
    """
    if not config.USE_WANDB:
        return

    logging.debug(f"Generating W&B {method.value} influence results table")

    # Configure and clone the DataLoader to prevent conflicts external and ensure the correct
    # settings
    train_dl = configure_train_dataloader(train_dl=train_dl)

    # Sort all the tensors to be safe
    inf_vals, ids = sort_ids_and_inf(inf_arr=inf_vals, ids_arr=ids)

    # Get rank of each training example
    ramp = torch.arange(1, ids.numel() + 1, dtype=torch.long)
    ds_rank = torch.full([torch.max(ids) + 1], fill_value=-1, dtype=torch.long)
    ds_inf = torch.zeros_like(ds_rank, dtype=DTYPE)
    ds_rank[ids], ds_inf[ids] = ramp, inf_vals

    # create a wandb.Table() with corresponding columns
    columns = ["id", "image", "inf", "rank", "label"]
    #  Construct images of all results
    to_pil = transforms.ToPILImage()
    all_res = []
    for batch_tensors in train_dl:
        batch = block.organize_batch(batch_tensors)
        if batch.skip():
            continue

        id_val = batch.ids.item()  # Id number of the example
        # Construct the results
        tr_ex = [id_val, wandb.Image(to_pil(batch.xs[0].clamp(0, 1))),
                 ds_inf[id_val].item(), ds_rank[id_val].item(), batch.lbls.item()]
        all_res.append(tr_ex)

    # Generate the table
    run = wandb.init(project=parent_utils.get_proj_name())
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
    quantiles = torch.tensor([0., 0.25, 0.5, 0.75, 1.], dtype=DTYPE)
    names = ["Min", "25%-Quartile", "Median", "75%-Quartile", "Max"]  # noqa
    quant_vals = torch.quantile(norms, q=quantiles)
    for name, val in zip(names, quant_vals.tolist()):
        logging.info(f"{header} {name}: {val:.6E}")
    # Interquartile range
    val = quant_vals[-2] - quant_vals[1]
    logging.info(f"{header} IQR: {val.item():.6E}")

    std, mean = torch.std_mean(norms, unbiased=True)
    for val, val_name in zip((mean, std), ("Mean", "Stdev")):
        logging.info(f"{header} {val_name}: {val.item():.6E}")


def is_grad_zero(grad: Tensor) -> bool:
    r""" Returns \p True if the gradient dot product is zero """
    dot_prod = grad_dot_prod(grad, grad)
    return dot_prod.item() == 0
