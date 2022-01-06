__all__ = [
    "TORCH_DEVICE",
    "WrapRoBERTa",
    "construct_filename",
    "get_ds_sample",
    "get_num_usable_cpus",
    "get_sample_label",
    "get_train_ds",
    "sentiment_criterion",
    "set_random_seeds",
    "trainer_forward",
]

import collections
import os
from pathlib import Path
import random
import time
from typing import NoReturn, Optional, Tuple, Union

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F  # noqa

from . import _config as config

TASK_CHECKED = False
TORCH_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

TRAIN_DATASET = None


def get_num_usable_cpus() -> int:
    r"""
    Returns the number of usable CPUs which is less than or equal to the actual number of CPUs.
    Read the document for os.cpu_count() for more info.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def get_train_ds(trainer):
    r""" Gets the training dataset used by the model """
    global TRAIN_DATASET
    if TRAIN_DATASET is None:
        TRAIN_DATASET = trainer.get_train_iterator(epoch=0).dataset
    return TRAIN_DATASET


def get_ds_sample(idx: Union[int, Tensor], ds, trainer) -> dict:
    r""" Gets and formats the training same from the training dataset """
    if isinstance(idx, Tensor):
        assert idx.numel() == 1, "Only single element tensors currently supported"
        idx = idx.item()  # Convert to an integer

    assert 0 <= idx < len(ds), f"Training index {idx} invalid for DS size {len(ds)}"

    sample = ds[idx]
    assert isinstance(sample["id"], int), "Sample ID is not an integer"
    assert idx == sample["id"], "Training and dataset IDs should align one to one"

    src_tokens = collections.OrderedDict()
    src_tokens["src_tokens"] = sample["net_input.src_tokens"].view([1, -1])
    src_lengths = torch.tensor([sample["net_input.src_lengths"]]).view([-1])
    src_tokens["src_lengths"] = src_lengths.to(TORCH_DEVICE)
    sample["net_input"] = src_tokens

    sample = trainer._prepare_sample(sample=sample)  # noqa
    return sample


def trainer_forward(trainer, sample: dict) -> Tensor:
    r""" Standardizes the forward method through the trained model """
    model = trainer.get_model()
    out, _ = model.forward(**sample["net_input"], features_only=True,
                           classification_head_name="sentence_classification_head")
    return out


def construct_filename(prefix: str, out_dir: Path, file_ext: str,
                       add_timestamp: bool = False) -> Path:
    r""" Standardize naming scheme for the filename """
    fields = ["nlp-poison"]
    if prefix:
        fields = [prefix] + fields

    if add_timestamp:
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        fields.append(time_str)

    if file_ext[0] != ".":
        file_ext = "." + file_ext
    fields[-1] += file_ext

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "_".join(fields)


def get_sample_label(sample: dict) -> Tensor:
    r""" Gets the label from a poisoned example """
    return sample["target"]


class WrapRoBERTa(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        if len(model.classification_heads) != 1:
            raise ValueError("Only a single classification head is supported at present")

        self._model = model
        self._model.eval()

        assert len(model.classification_heads) == 1, "Only singleton classification heads supported"
        # Split the classifier block
        for _, mlp_block in model.classification_heads.items():
            break

        # noinspection PyUnboundLocalVariable
        assert mlp_block.activation_fn == torch.tanh, "TanH Hardcoded for now"
        self._fc_first = nn.Sequential(mlp_block.dropout,
                                       mlp_block.dense,
                                       nn.Tanh(),
                                       mlp_block.dropout)
        self.linear = mlp_block.out_proj

        self.to(TORCH_DEVICE)

    def forward(self, sample, penu: bool = False) -> Tuple[Tensor, Tensor]:
        features, extra = self._model.decoder(**sample["net_input"], features_only=True)

        # From RobertaClassificationHead f
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self._fc_first.forward(x)
        if not penu:
            x = self.linear.forward(x)
        return x, extra


def sentiment_criterion(logits: Tensor, sample: dict, reduction: str = "None") -> Tensor:
    r"""
    Calculates the loss for sentiment classification.  Cannot use the main sentiment criterion
    method since that performs a forward pass through the entire network not just the last
    linear layer.
    """
    global TASK_CHECKED
    if not TASK_CHECKED:
        # Put task check in an if to prevent runtime cost of string check in each call
        assert "sentiment" in config.TASK.lower(), "Criterion only applies to sentiment classify"
        TASK_CHECKED = True

    loss = F.nll_loss(input=F.log_softmax(logits, dim=-1).float(),
                      target=get_sample_label(sample=sample),
                      reduction=reduction)
    return loss


def set_random_seeds(seed: Optional[int] = None) -> NoReturn:
    r"""
    Sets random seeds to avoid non-determinism
    :See: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed is None:
        seed = torch.random.initial_seed()

    seed &= 2**32 - 1  # Some methods dont like the seed too large
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False
