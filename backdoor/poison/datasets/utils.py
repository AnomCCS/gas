__all__ = [
    "binarize_labels",
    "binom_sample",
    "build_backdoor",
    "calc_bd_perturb_dist",
    "calc_perturb_dist",
    "download_file",
    "extract_d_tensor",
    "filter_classes",
    "get_class_mask",
    "prune_datasets",
    "shuffle_tensorgroup",
    "std_blending_backdoor",
    "std_four_pixel_backdoor",
    "std_one_pixel_backdoor",
]

import logging
from pathlib import Path

import numpy as np
import requests
from typing import Callable, List, NoReturn, Optional, Tuple, Union

import torch
from torch import Tensor
from torch import nn
import torch.distributions as distributions

from .types import BackdoorAttack, NEG_LABEL, POS_LABEL
from .. import _config as config
from .. import influence_utils
from .. import learner
from ..types import TensorGroup

BACKDOOR_BLENDING_TENSOR = None


def binom_sample(prior: float, n: int) -> int:
    r""" Binomial distribution sample """
    binom = distributions.Binomial(n, torch.tensor([prior]))
    return int(binom.sample())


def multinomial_sample(n: int, p_vec: Tensor) -> Tensor:
    r""" Multinomial distribution sample """
    assert p_vec.shape[0] > 0, "Multinomial size doesn't make sense"

    n_per_category = distributions.Multinomial(n, p_vec).sample().int()

    assert p_vec.shape == n_per_category.shape, "Dimension mismatch"
    assert int(n_per_category.sum().item()) == n, "Number of elements mismatch"
    return n_per_category


def download_file(url: str, file_path: Path) -> None:
    r""" Downloads the specified file """
    CHUNK_SIZE = 128 * 1024  # BYTES  # noqa

    if file_path.exists():
        logging.info(f"File \"{file_path}\" already downloaded. Skipping...")
        return

    # Store the download file to a temporary directory
    tmp_file = file_path.parent / f"tmp_{file_path.stem}.download"

    msg = f"Downloading file at \"{url}\""
    logging.info(f"Starting: {msg}...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(str(tmp_file), 'wb+') as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    logging.info(f"COMPLETED: {msg}")

    msg = f"Renaming temporary file \"{tmp_file}\" to \"{file_path}\""
    logging.info(f"Starting: {msg}...")
    tmp_file.rename(file_path)
    logging.info(f"COMPLETED: {msg}")

    assert file_path.exists(), "Specified file path does not exist"


def build_model(model, x: Tensor, hidden_dim: Optional[int] = None) -> nn.Module:
    r""" Constructs the linear layer """
    model.eval()
    # if not torch.cuda.is_available():
    #     logging.warning("No CUDA in the model.  Skipping building the model.  Cannot train")
    #     return model

    x_tfm = config.get_test_tfms()(x[:1])
    with torch.no_grad():
        model.build_fc(x=x_tfm, hidden_dim=hidden_dim)
    return model


def get_class_mask(y: Tensor, cls_lst: Optional[Union[List[int], int]] = None) -> Tensor:
    r""" Removes all classes except the target and poison classes """
    if cls_lst is None:
        cls_lst = [config.TARG_CLS, config.POIS_CLS]
    if isinstance(cls_lst, int):
        cls_lst = [cls_lst]

    cls_mask = torch.full(y.shape, fill_value=False, dtype=torch.bool)
    for i in cls_lst:
        cls_mask |= (y == i)
    return cls_mask


def extract_d_tensor(orig_x: Tensor, perturb_x: Tensor, max_val: float,
                     min_val: Optional[float] = None) -> Tensor:
    r"""
    Extracts the perturbation (including with clipping) from the perturbed tensor.  If \p min_val
    is not specified, no min clipping is performed.
    """
    assert orig_x.shape == perturb_x.shape, "Feature and perturb shape should match"

    kwargs = {"max": max_val}
    if min_val is not None:
        kwargs["min"] = min_val
    return torch.clamp(perturb_x, **kwargs) - orig_x


def filter_classes(x: Tensor, y: Tensor, ids: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    r""" Returns objects based on the """
    cls_mask = get_class_mask(y=y)
    return x[cls_mask], y[cls_mask], ids[cls_mask]


def std_four_pixel_backdoor(x: Tensor, center_row: int, center_col: int, max_val: float,
                            perturb: Optional[float] = None) -> Tensor:
    r""" Standardizes the four pixel attack """
    if perturb is None:
        n_pixels = 4
        n_channels = x.shape[1]  # Number of color channels
        assert n_channels == config.DATASET.value.dim[0], "Unexpected dimension"
        perturb = config.BACKDOOR_DELTA / np.sqrt(n_pixels * n_channels)

    mod = x.clone()

    pixels = ((center_row, center_col), (center_row + 1, center_col + 1),
              (center_row - 1, center_col + 1), (center_row + 1, center_col - 1))
    # Perturb the four pixel
    for row, col in pixels:
        mod[:, :, row, col] = mod[:, :, row, col] + perturb
    return extract_d_tensor(orig_x=x, perturb_x=mod, max_val=max_val)


def std_one_pixel_backdoor(x: Tensor, center_row: int, center_col: int,
                           max_val: float, perturb: Optional[float] = None) -> Tensor:
    r""" Performs the single pixel attack """
    if perturb is None:
        n_pixels = 1
        n_channels = x.shape[1]  # Number of color channels
        assert n_channels == config.DATASET.value.dim[0], "Unexpected dimension"
        perturb = config.BACKDOOR_DELTA / np.sqrt(n_pixels * n_channels)

    mod = x.clone()
    mod[:, :, center_row, center_col] = mod[:, :, center_row, center_col] + perturb
    return extract_d_tensor(orig_x=x, perturb_x=mod, max_val=max_val)


def construct_blending_tensor(x: Tensor, delta: Optional[float] = None) -> NoReturn:
    r"""
    Separates construction of the blending tensor. Raises runtime error if called more than
    once.
    :param x: Features tensor used to extract the data dimensions
    :param delta: Total perturbation norm.  If not specified, uses the value in the configuration.
    """
    global BACKDOOR_BLENDING_TENSOR
    assert BACKDOOR_BLENDING_TENSOR is None, "Cannot reset backdoor tensor"

    if delta is None:
        delta = config.BACKDOOR_DELTA

    assert config.DATASET is not None, "No dataset set"
    x_shape = x.shape[1:]  # Exclude the batch size dimension
    perturb = torch.randn(x_shape).unsqueeze(dim=0)  # Add the batch dimension back with unsqueeze
    BACKDOOR_BLENDING_TENSOR = delta * perturb / perturb.norm()


def std_blending_backdoor(x: Tensor, min_val: float, max_val: float) -> Tensor:
    r""" Performs the blending attack """
    assert BACKDOOR_BLENDING_TENSOR is not None, "Blending tensor not yet constructed"
    # noinspection PyUnresolvedReferences
    assert BACKDOOR_BLENDING_TENSOR.shape[0] == 1, "More than one element in batch dimension"
    # noinspection PyUnresolvedReferences
    assert x.shape[1:] == BACKDOOR_BLENDING_TENSOR.shape[1:], "Blending shape mismatch"

    # noinspection PyUnresolvedReferences
    mod = x + BACKDOOR_BLENDING_TENSOR
    mod = extract_d_tensor(orig_x=x, perturb_x=mod, min_val=min_val, max_val=max_val)
    return mod


def prune_datasets(base_dir: Path, data_dir: Path) -> List[Path]:
    r""" Reduce the training and test sets based on a fixed divider of the ordering """
    # Location to store the pruned data
    prune_dir = base_dir / "pruned"
    prune_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    # div = int(round(1 / config.VALIDATION_SPLIT_RATIO))
    for i in range(3):
        is_train, is_val = i == 0, i == 1
        # Build the path for the data to store
        if is_train:
            dest_fname = f"training_div={config.val_div()}.pth"
        elif is_val:
            dest_fname = f"val_div={config.val_div()}.pth"
        else:
            dest_fname = f"test.pth"
        # Construct the complete filename and if it exists,
        paths.append(prune_dir / dest_fname)
        if paths[-1].exists():
            continue

        # Load the complete source data
        base_fname = "training" if is_train or is_val else "test"
        # Support two different file extensions
        for file_ext in (".pth", ".pt"):
            path = data_dir / (base_fname + file_ext)
            if not path.exists():
                continue
            with open(path, "rb") as f_in:
                x, y = torch.load(f_in)
            y_np = y.numpy()
            ord_ids = torch.from_numpy(np.argsort(y_np, kind="stable")).long()
            x, y = x[ord_ids], y[ord_ids]
            break
        else:
            raise ValueError("Unable to find processed tensor")

        # Add an ID vector that details the original ID number
        ids = torch.arange(x.shape[0], dtype=torch.long)  # type: Tensor # noqa
        if is_train or is_val:
            mask = (ids % config.val_div()) == 0
            if is_train:
                mask = ~mask
        else:
            mask = torch.full(y.shape, True, dtype=torch.bool)  # noqa
        x, y, ids = x[mask], y[mask], ids[mask]

        with open(str(paths[-1]), "wb+") as f_out:
            torch.save((x, y, ids), f_out)

    return paths


def shuffle_tensorgroup(tg: TensorGroup) -> NoReturn:
    r""" Shuffle the tensor group elements """
    for ds in ("tr", "val", "test"):
        n_el = tg.__getattribute__(f"{ds}_ids").numel()
        # Randomize the indices
        shuffled_idx = torch.randperm(n_el)
        assert n_el == shuffled_idx.numel()
        for suffix in ("x", "d", "y", "adv_y", "ids"):
            name = f"{ds}_{suffix}"
            tensor = tg.__getattribute__(name)
            if tensor is None:
                continue

            shuffled = tensor[shuffled_idx]
            tg.__setattr__(name, shuffled)


def build_backdoor(tg: TensorGroup,
                   one_pixel_atk: Callable, four_pixel_atk: Callable, blending_atk: Callable,
                   min_val: Optional[float] = None, max_val: Optional[float] = None) -> NoReturn:
    r""" Constructs the backdoor attack vector d for all vectors """
    # Select the function used to perform the attack
    if config.BACKDOOR_ATTACK == BackdoorAttack.ONE_PIXEL:
        f_perturb = one_pixel_atk
    elif config.BACKDOOR_ATTACK == BackdoorAttack.FOUR_PIXEL:
        f_perturb = four_pixel_atk
    elif config.BACKDOOR_ATTACK == BackdoorAttack.BLENDING:
        construct_blending_tensor(x=tg.tr_x, delta=config.BACKDOOR_DELTA)
        f_perturb = blending_atk
    else:
        raise ValueError(f"Attack {config.BACKDOOR_ATTACK.name} is not supported")

    shuffle_tensorgroup(tg=tg)

    # Construct the backdoor vector
    for ds_name in ("tr", "val", "test"):
        x_name, d_name = f"{ds_name}_x", f"{ds_name}_d"
        x = tg.__getattribute__(x_name)
        # Verify that the x tensor has the expected bounds
        x_min, x_max = torch.min(x).item(), torch.max(x).item()
        assert x_min == min_val, f"Minimum X value should be {min_val}"
        assert x_max == max_val, f"Maximum X value should be {max_val}"
        # Find the backdoor perturbation vector
        d = f_perturb(x=x)
        tg.__setattr__(d_name, d)

    if config.has_min_perturb():
        perturb_threshold = config.MIN_PERTURB_RATIO * config.BACKDOOR_DELTA
        _organize_train_by_perturb_threshold(tg=tg, min_perturb=perturb_threshold)


def calc_perturb_dist(ds: Tensor) -> Tensor:
    r"""
    Calculates the (l2) perturbation distance of each training instance

    :param ds: Perturbation tensor
    :return: Vector of perturbation distances
    """
    # Torch does not support batch-aware multi-dimension norms by default so flatten to enable
    # such a norm first
    flat = ds.view([ds.shape[0], -1])
    perturb_dist = torch.norm(flat, dim=1)

    assert perturb_dist.numel() == ds.shape[0], "Unexpected num elements in perturb arr"
    return perturb_dist


def calc_bd_perturb_dist(tg: TensorGroup) -> Tensor:
    r""" Calculates the perturbation distance specifically for the backdoor set """
    # Select only the backdoor training examples
    bd_ids = learner.select_backdoor_ids(y=tg.tr_y, adv_y=tg.tr_adv_y, ds_ids=tg.tr_ids)
    mask = influence_utils.is_bd(bd_ids=bd_ids)

    return calc_perturb_dist(ds=tg.tr_d[mask])


def binarize_labels(tg: TensorGroup) -> NoReturn:
    r""" Binarize the labels """
    # Extract all the labels
    all_lbls = [tg.__getattribute__(fld) for fld in dir(tg) if fld.endswith("_y")]
    all_lbls = torch.cat(all_lbls)
    min_lbl = torch.min(all_lbls)  # Extract the min label
    for fld in dir(tg):
        if not fld.endswith("_y"):
            continue
        lbls = tg.__getattribute__(fld)
        mask = lbls == min_lbl
        lbls[mask] = NEG_LABEL
        lbls[~mask] = POS_LABEL

    if config.TARG_CLS < config.POIS_CLS:
        targ_cls, pois_cls = NEG_LABEL, POS_LABEL
    else:
        targ_cls, pois_cls = POS_LABEL, NEG_LABEL
    config.update_labels(targ_cls=targ_cls, pois_cls=pois_cls)


def _organize_train_by_perturb_threshold(tg: TensorGroup, min_perturb: float) -> NoReturn:
    r"""
    Reorganizes the training set so only training elements with perturbation at least
    as large as \p min_perturb can be considered for backdoors.

    :param tg: \p TensorGroup of dataset tensors
    :param min_perturb: Minimum perturbation value to be considered for backdoors.  These are
                        placed at the beginning of the training tensor so they are candidates for
                        being a backdoor
    """
    assert tg.tr_d is not None, "Training perturbation not set"

    perturb_dist = calc_perturb_dist(ds=tg.tr_d)
    # Mark the valid backdoors that meet the perturb threshold
    mask = (perturb_dist >= min_perturb).logical_and(tg.tr_y == config.TARG_CLS)

    n_bd_cands = torch.sum(mask)
    assert n_bd_cands >= config.BACKDOOR_CNT, "Insufficient # backdoors for perturb threshold"
    logging.info(f"Num Backdoor Candidates w. Min Perturb Delta {min_perturb:.3f}: {n_bd_cands}")

    # Reshuffle the elements to place high perturb training elements first
    for attr in vars(tg):
        if not attr.startswith("tr_"):
            continue
        val = tg.__getattribute__(attr)
        assert isinstance(val, Tensor), "Attribute expected to be a tensor"

        new_val = torch.cat([val[mask], val[~mask]], dim=0)
        tg.__setattr__(attr, new_val)
