__all__ = [
    "CustomTensorDataset",
    "InfStruct",
    "LearnerParams",
    "TensorGroup",
]

from enum import Enum
import dataclasses
from typing import NoReturn, Optional, Union

import torch
from torch import LongTensor, Tensor
from torch.utils.data import Dataset

from .influence_utils import InfluenceMethod


@dataclasses.dataclass
class TensorGroup:
    r""" Encapsulates a group of tensors used by the learner """
    pretr_x: Optional[Tensor] = None
    pretr_y: Optional[LongTensor] = None
    pretr_ids: Optional[LongTensor] = None

    preval_x: Optional[Tensor] = None
    preval_y: Optional[LongTensor] = None
    preval_ids: Optional[LongTensor] = None

    pretest_x: Optional[Tensor] = None
    pretest_y: Optional[LongTensor] = None
    pretest_ids: Optional[LongTensor] = None

    tr_x: Optional[Tensor] = None
    tr_y: Optional[LongTensor] = None
    tr_ids: Optional[LongTensor] = None

    val_x: Optional[Tensor] = None
    val_y: Optional[LongTensor] = None
    val_ids: Optional[LongTensor] = None

    targ_x: Optional[Tensor] = None
    targ_y: Optional[LongTensor] = None  # Actual y value
    targ_ids: Optional[LongTensor] = None

    test_x: Optional[Tensor] = None
    test_y: Optional[LongTensor] = None
    test_ids: Optional[LongTensor] = None


@dataclasses.dataclass(order=True)
class LearnerParams:
    r""" Learner specific parameters """
    class Attribute(Enum):
        LEARNING_RATE = "lr"
        WEIGHT_DECAY = "wd"

        # NUM_FF_LAYERS = "num_ff_layers"
        # NUM_SIGMA_LAYERS = "num_sigma_layers"

    learner_name: str

    lr: float = None
    wd: float = None

    # num_ff_layers: int = None
    # num_sigma_layers: int = None

    def set_attr(self, attr_name: str, value: Union[int, float]) -> NoReturn:
        r""" Enhanced set attribute method that has enhanced checking """
        try:
            # Allow short field name or longer attribute name
            attr_name = attr_name.lower()
            self.__getattribute__(attr_name)
        except AttributeError:
            try:
                attr_name = self.Attribute[attr_name.upper()].value
                self.__getattribute__(attr_name)
            except KeyError:
                raise AttributeError(f"No attribute \"{attr_name}\"")

        for field in dataclasses.fields(self):
            if field.name == attr_name:
                break
        else:
            raise ValueError(f"Cannot find field \"{attr_name}\"")

        assert isinstance(value, field.type), "Type mismatch when setting"
        self.__setattr__(attr_name, value)

    def get_attr(self, attr_name: str) -> Optional[Union[int, float]]:
        r""" Attribute accessor with more robust handling of attribute name """
        attr_name = attr_name.lower()
        try:
            return self.__getattribute__(attr_name)
        except AttributeError:
            raise AttributeError("No attribute \"attr_name\"")


class CustomTensorDataset(Dataset):
    r""" TensorDataset with support of transforms. """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = [tensor.clone() for tensor in tensors]
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)
        return tuple([x] + [tens[index] for tens in self.tensors[1:]])

        # y = self.tensors[1][index]

        # return x, y

    def __len__(self):
        return self.tensors[0].size(0)

    def set_transform(self, transform) -> NoReturn:
        r""" Change the transform for the dataset """
        self.transform = transform


@dataclasses.dataclass
class InfStruct:
    name: str
    method: InfluenceMethod
    inf: Tensor
    ids: LongTensor
    filt_ids: Optional[LongTensor] = None

    def select_filter_ids(self, frac_filt: float) -> NoReturn:
        r""" Filter a subset of the IDs """
        assert 0 < frac_filt < 1, "Filter fraction must be in range (0,1)"
        assert self.inf.shape == self.ids.shape, "Mismatch in results shapes"

        # Sort the influences from most to least influence
        _, idx = torch.sort(self.inf, dim=0, descending=True)
        sorted_ids, n_filt = self.ids[idx], int(frac_filt * self.ids.numel())
        self.filt_ids = sorted_ids[:n_filt]
