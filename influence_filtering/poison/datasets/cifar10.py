__all__ = [
    "build_model",
    "load_data",
]

import dill as pk
from pathlib import Path

import torch
from torch import Tensor
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from . import _cifar10_resnet as cifar10_resnet  # noqa
from . import wandb_utils
from . import utils
from .. import _config as config
from ..types import TensorGroup
from .. import utils as parent_utils

TEST_ELE_PER_CLS = 1000

N_CIFAR_CLASSES = 10

CENTER_ROW = 15  # Center Row
CENTER_COL = 15  # Center Col

CIFAR_MIN = 0
CIFAR_MAX = 1

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def build_model(x: Tensor) -> nn.Module:
    r""" Construct the model used for MNIST training """
    # model = cifar10_cnn.Model()
    model = cifar10_resnet.ResNet9(in_channels=3, num_classes=1)
    return utils.build_model(model=model, x=x)


def _extract_target_example(tg: TensorGroup) -> None:
    r""" Extracts the target example """
    # Extract the test IDs
    te_idx = config.TARG_IDX
    if te_idx // TEST_ELE_PER_CLS != config.TARG_CLS:
        raise ValueError("Test index does not appear to belong to the target class")

    # Select only the elements that have the matching class and then select the target from it
    offset = te_idx % TEST_ELE_PER_CLS
    # cls_mask = utils.get_class_mask(y=tg.test_y, cls_lst=config.TARG_CLS)
    cls_mask = torch.ones_like(tg.test_y, dtype=torch.bool)  # ToDo Restore real mask
    for tensor_name in ("x", "y", "ids"):
        te_name = f"test_{tensor_name}"
        filt_tensor = tg.__getattribute__(te_name)[cls_mask]

        targ_name = f"targ_{tensor_name}"
        tg.__setattr__(targ_name, filt_tensor[offset:offset + 1])
    tg.targ_adv_y = torch.full_like(tg.targ_y, config.POIS_CLS)


def _set_transforms():
    r"""
    Configures the training and test/validation transforms used.  Based on
    https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    """
    normalize_tfm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    tfms_tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_tfm,
    ])

    tfms_val = transforms.Compose([
        normalize_tfm,
    ])

    img_tfms = transforms.Compose([
    ])
    config.set_tfms(train_tfms=tfms_tr, test_tfms=tfms_val, img_tfms=img_tfms)


def load_data(cifar_dir: Path) -> TensorGroup:
    r""" Loads the CIFAR10 dataset """
    _set_transforms()
    tg_pkl_path = parent_utils.construct_filename("bk-tg", out_dir=cifar_dir,
                                                  file_ext="pkl", add_ds_to_path=False)

    if not tg_pkl_path.exists():
        tensors_dir = utils.download_data(cifar_dir, ds_func=torchvision.datasets.CIFAR10)

        paths = utils.prune_datasets(base_dir=cifar_dir, data_dir=tensors_dir)

        # Transfer the vectors to the tensor groups
        tg = TensorGroup()
        for ds_name in ("tr", "val", "test"):
            x, y, ids = torch.load(paths[0])
            paths.pop(0)  # Remove from the front of the list
            for swap_mask in [False, True]:
                full_name = ds_name if not swap_mask else f"pre{ds_name}"
                # Filter to only the classes of interest
                res = utils.filter_classes(x=x, y=y, ids=ids, swap_mask=swap_mask)
                filt_x, filt_y, filt_ids = res
                tg.__setattr__(f"{full_name}_x", filt_x)
                tg.__setattr__(f"{full_name}_y", filt_y)
                tg.__setattr__(f"{full_name}_ids", filt_ids)
                # No mask swap for test data.  Test only on the two learned classes

        with open(tg_pkl_path, "wb+") as f_out:
            pk.dump(tg, f_out)
        wandb_utils.upload_data(tg=tg, labels=LABELS)

    with open(tg_pkl_path, "rb") as f_in:
        tg = pk.load(f_in)  # type: TensorGroup

    # config.override_targ_idx(targ_idx=tg.targ_ids.item())
    config.set_all_ds_sizes(n_full_tr=50000, tg=tg)
    utils.binarize_labels(tg=tg)
    config.set_num_classes(n_classes=10)
    return tg
