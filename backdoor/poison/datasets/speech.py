__all__ = [
    "load_data",
]

import copy
import logging
from pathlib import Path
import pickle as pk
import re
import tarfile
from typing import NoReturn, Tuple

import cv2
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets.utils import download_file_from_google_drive

from . import _speech_model as speech_model
from . import utils
from .. import _config as config
from .. import dirs
from ..types import TensorGroup
from .. import utils as parent_utils

DIR_NAME = "speech"
SPEECH_GD_FILE_ID = "1CuaDZgLhTNGhOQ_SjohmrZTOdGolfd62"  # Unique google file ID

SPEECH_NORMALIZE_FACTOR = 255

BD_TRAIN_FILE = "data/train_bd.txt"
BD_TEST_FILE = "data/test_all.txt"


class ToFloatAndNormalize(nn.Module):
    def __init__(self, normalize_factor: float):
        super().__init__()
        self._factor = normalize_factor

    def forward(self, x: Tensor) -> Tensor:
        out = x.float()
        out.div_(self._factor)
        return out


def build_model(x: Tensor) -> nn.Module:
    r""" Construct the model used for speech training """
    # model1 = _construct_orig_caffe_model()
    # x_tfm = config.get_test_tfms()(x[:1])
    # model1.forward(x_tfm)

    model = speech_model.Model()
    return utils.build_model(model=model, x=x)


CAFFE_URL = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
CAFFE_PROTO = "model/caffe.proto"
SPEECH_BD_PROTO = "model/net_train_bd.prototxt"
SPEECH_BD_MODEL = "model/bd_net.caffemodel"


def download_google_drive_dataset(dest: Path, gd_file_id: str, file_name: str,
                                  decompress: bool = False) -> NoReturn:
    r"""
    Downloads the source data from Google Drive

    :param dest: Folder to which the dataset is downloaded
    :param gd_file_id: Google drive file unique identifier
    :param file_name: Filename to store the downloaded file
    :param decompress: If \p True (and \p file_name has extension ".tar.gz"), unzips the downloaded
                       zip file
    """
    full_path = dest / file_name
    if full_path.exists():
        logging.info(f"File \"{full_path}\" exists.  Skipping download")
        return

    # Define the output files
    dest.mkdir(exist_ok=True, parents=True)
    download_file_from_google_drive(root=str(dest), file_id=gd_file_id, filename=file_name)
    if file_name.endswith(".tar.gz"):
        if decompress:
            with tarfile.open(str(full_path), "r") as tar:
                
                import os
                
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=str(dest))
            # with zipfile.ZipFile(str(full_path), 'r') as zip_ref:
            #     zip_ref.extractall(dest.parent)
    else:
        assert not decompress, "Cannot decompress a non tar.gz file"


def _download_speech_files(root: Path) -> NoReturn:
    r""" Download the speech model and data files """
    download_google_drive_dataset(dest=root, gd_file_id=SPEECH_GD_FILE_ID,
                                  file_name="speech.tar.gz", decompress=True)


def _read_speech_data(data_dir: Path, file: str) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Reads the speech data from disk.

    :param data_dir: Speech data's root directory
    :param file: Name of the file defining the list of train or test examples
    :return: Tuple of the combined feature tensor (x), true example label, and backdoor example
             which equals the true label if the example does not contain a backdoor.
    """
    # Read the file contain the tensor info
    with open(data_dir / file, "r") as f_in:
        tr_set_ex = [line.split(" ") for line in f_in.read().splitlines()]

    bd_pat = re.compile("^(train|test)_bd$")
    cl_pat = re.compile("^(train|test)_cl$")

    # Initialize the variables
    tensors = dict()
    for ds_name in ("cl", "bd"):
        for tensor_name in ("x", "true_lbl", "adv_lbl"):
            tensors[f"{ds_name}_{tensor_name}"] = []

    # Allocate the memory for the tensors
    for ex in tr_set_ex:
        assert len(ex) == 2, "Unexpected space in the example description"

        # Extract the x Tensor
        x = cv2.imread(str(data_dir / ex[0]))
        x = np.rollaxis(x, 2)
        x = torch.from_numpy(x).unsqueeze(dim=0)

        file_path_spl = ex[0].split("/")
        # Extract the labels
        true_lbl = int(file_path_spl[2][0])  # True label at the start of the file name
        adv_lbl = int(ex[1])

        # File path structure is "data/(train|test)_(bd|cl)/[0-9]_*.png"
        # Extract whether the example is backdoor or clean
        is_backdoor = bd_pat.match(file_path_spl[1])
        assert is_backdoor or cl_pat.match(file_path_spl[1]), "No match for backdoor/clean check"
        # Append the data to the respective
        ds_name = "bd" if is_backdoor else "cl"
        (tensors[f"{ds_name}_x"]).append(x)
        for lbl_type in ("adv", "true"):
            tensors[f"{ds_name}_{lbl_type}_lbl"].append(locals()[f"{lbl_type}_lbl"])

    # Convert each dataset into a single tensor
    for ds_name in ("bd", "cl"):
        tensors[f"{ds_name}_x"] = torch.cat(tensors[f"{ds_name}_x"], dim=0)
        # Label vectors go from list -> ndarray -> torch
        for lbl_type in ("adv", "true"):
            name = f"{ds_name}_{lbl_type}_lbl"
            lbl = np.array(tensors[name])
            tensors[name] = torch.from_numpy(lbl).long()

    x = torch.cat([tensors["bd_x"], tensors["cl_x"]], dim=0)  # noqa

    true_lbl = torch.cat([tensors["bd_true_lbl"], tensors["cl_true_lbl"]])  # noqa
    adv_lbl = torch.cat([tensors["bd_adv_lbl"], tensors["cl_adv_lbl"]])  # noqa
    return x, true_lbl, adv_lbl


def _load_train_and_val_data(data_dir: Path, tg: TensorGroup) -> NoReturn:
    r""" Loads the training and validation data for the speech dataset """
    # Extract the respective vectors for train and test
    x, true_lbl, adv_lbl = _read_speech_data(data_dir=data_dir, file=BD_TRAIN_FILE)
    ids = torch.arange(x.shape[0], dtype=torch.long)
    # Split train and test randomly
    tr_mask = (ids % config.val_div()) > 0
    for ds_name, is_train in zip(("tr", "val"), (True, False)):
        mask = tr_mask if is_train else ~tr_mask

        x_tensor = x[mask]
        tg.__setattr__(f"{ds_name}_x", x_tensor)
        tg.__setattr__(f"{ds_name}_d", torch.zeros_like(x_tensor))  # Use same vector to save RAM
        tg.__setattr__(f"{ds_name}_y", true_lbl[mask])
        tg.__setattr__(f"{ds_name}_adv_y", adv_lbl[mask])
        tg.__setattr__(f"{ds_name}_ids", ids[mask])


def _load_test_data(data_dir: Path, tg: TensorGroup) -> NoReturn:
    r""" Extracts the test data from disk and populates the \p TensorGroup object """
    x, true_lbl, adv_lbl = _read_speech_data(data_dir=data_dir, file=BD_TEST_FILE)
    tg.test_x = x
    tg.test_d = torch.zeros_like(x)
    tg.test_y = true_lbl
    tg.test_ids = torch.arange(x.shape[0], dtype=torch.long)
    tg.test_adv_y = adv_lbl


def _extract_target_example(tg: TensorGroup) -> NoReturn:
    r""" Extracts the first backdoor test example from the specified target class """
    # Consider only those examples from the target class with backdoors
    mask = (tg.test_y == config.TARG_CLS) & (tg.test_y != tg.test_adv_y)

    # Select the first example satisfying the mask
    for tensor_name in ("x", "y", "adv_y", "d", "ids"):
        tensor = tg.__getattribute__(f"test_{tensor_name}")
        tg.__setattr__(f"targ_{tensor_name}", tensor[mask][config.TARG_IDX:config.TARG_IDX + 1])

    assert tg.targ_x.shape[0] == 1, "No target example identified"


def _set_transforms():
    r"""
    Configures the training and test/validation transforms used. No transforms specified for MNIST
    """
    tfms_tr = transforms.Compose([ToFloatAndNormalize(SPEECH_NORMALIZE_FACTOR)])
    tfms_val = copy.deepcopy(tfms_tr)
    config.set_tfms(train_tfms=tfms_tr, test_tfms=tfms_val)


def _calculate_num_backdoor(tg: TensorGroup) -> NoReturn:
    r""" Calculates the number of backdoor examples in the dataset """
    # Create a mask for all the target, backdoor examples in the dataset
    all_bd_mask = tg.tr_y != tg.tr_adv_y
    targ_ex_mask = tg.tr_y == config.TARG_CLS
    bd_targ_mask = all_bd_mask & targ_ex_mask

    nonzero_cnt = torch.count_nonzero(bd_targ_mask).item()  # type: int
    assert bd_targ_mask[bd_targ_mask].shape[0] == nonzero_cnt, "Unexpected number of nonzero"

    config.override_num_backdoor(n_bd=nonzero_cnt)


def write_speech_test_labels(data_dir: Path, tg: TensorGroup) -> NoReturn:
    r""" Write the test labels """
    data_dir.mkdir(parents=True, exist_ok=True)
    assert data_dir.is_dir(), f"\"{data_dir}\" is not a directory"
    te_lbl_path = data_dir / "speech_test_labels.csv"
    with open(te_lbl_path, "w+") as f_out:
        f_out.write("ID#,True-Label,Adv-Label,Diff\n")
        flds = tg.test_ids.tolist(), tg.test_y.tolist(), tg.test_adv_y.tolist()
        for id_val, true_lbl, adv_lbl in zip(*flds):
            f_out.write(f"{id_val},{true_lbl},{adv_lbl},")
            if true_lbl != adv_lbl:
                f_out.write("TRUE")
            f_out.write("\n")


def load_data() -> TensorGroup:
    # Download the data from Google drive
    data_dir = dirs.DATA_DIR / DIR_NAME
    data_dir.mkdir(exist_ok=True, parents=True)

    # Verify the specified backdoor and poison meet the actual data configuration
    assert (config.TARG_CLS + 1) % speech_model.NUM_CLASSES == config.POIS_CLS, "Invalid classes"

    tg_pkl_path = parent_utils.construct_filename("bk-tg", out_dir=data_dir,
                                                  file_ext="pkl", add_ds_to_path=False)

    if not tg_pkl_path.exists():
        _download_speech_files(root=data_dir)

        tg = TensorGroup()
        _load_train_and_val_data(data_dir=data_dir, tg=tg)
        _load_test_data(data_dir=data_dir, tg=tg)

        utils.shuffle_tensorgroup(tg=tg)
        _extract_target_example(tg=tg)

        with open(tg_pkl_path, "wb+") as f_out:
            pk.dump(tg, f_out)

        # wandb_utils.upload_data(tg=tg)

    with open(tg_pkl_path, "rb") as f_in:
        tg = pk.load(f_in)
    write_speech_test_labels(data_dir=data_dir, tg=tg)

    _calculate_num_backdoor(tg=tg)

    tot_num_el = tg.tr_y.numel() + tg.val_y.numel()
    config.set_all_ds_sizes(n_full_tr=tot_num_el, tg=tg)
    config.set_num_classes(n_classes=speech_model.NUM_CLASSES)

    _set_transforms()
    parent_utils.set_random_seeds()
    return tg
