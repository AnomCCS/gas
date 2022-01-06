__all__ = [
    "TracInStruct",
]

import copy
import dataclasses
from pathlib import Path
from typing import List, NoReturn, Union

import torch
from torch import Tensor

from . import fairseq


@dataclasses.dataclass
class SubepochInfo:
    epoch: int
    num_updates: int
    lr: float
    checkpoint: str
    ids: Union[List[Tensor], Tensor] = dataclasses.field(default_factory=lambda: [])
    is_best: bool = False


class TracInStruct:
    r""" Structure used to store the items used by TracIn """
    def __init__(self):
        self._subepochs = []
        self._next_checkpoint = None  #

    def append_new_subepoch(self, epoch: int, trainer: fairseq.trainer.Trainer) -> NoReturn:
        r""" Construct a new subepoch """
        if len(self) > 0:
            self.close_last_subepoch()

        assert self._next_checkpoint is not None, "If creating subepoch previous model should exist"
        sep_info = SubepochInfo(checkpoint=self._next_checkpoint, epoch=epoch,
                                num_updates=trainer.get_num_updates(), lr=trainer.get_lr())
        # Clear the checkpoint name for the next subepoch
        self._next_checkpoint = None

        # Number of updates to the model should be monotonically increasing
        assert len(self) == 0 or sep_info.num_updates > self._subepochs[-1].num_updates
        self._subepochs.append(sep_info)

    def close_last_subepoch(self) -> NoReturn:
        r""" Closes and reformats all data structures at the end of the subepoch """
        # Verify the last subepoch has valid data
        last = self._get_last()
        assert len(last.ids) > 0, "No IDs in the last subepoch"
        last.ids = torch.cat(last.ids)  # noqa
        assert last.ids.numel() > 0, "No IDs after concatenating IDs"
        assert last.checkpoint is not None, "No checkpoint file stored"

    def add_samples(self, ids: Tensor) -> NoReturn:
        r""" Add new samples to the currently active subepoch """
        assert isinstance(ids, Tensor), "ids does not appear to be a tensor"
        assert ids.dtype == torch.long, "ids is expected to be a long Tensor"
        last = self._get_last()
        last.ids.append(ids)

    def _get_last(self) -> SubepochInfo:
        r""" Standardizes access for the last subepoch """
        assert len(self._subepochs) > 0, "Trying to get subepoch but no subepochs exist"
        return self._subepochs[-1]

    def set_name_next_checkpoint(self, next_checkpoint: str) -> NoReturn:
        r""" Stores the name of the next checkpoint under review """
        assert self._next_checkpoint is None, "Overwriting a stored checkpoint name"

        self._next_checkpoint = next_checkpoint

    def set_last_best(self, acc: float) -> None:
        r""" Sets the last epoch as the best """
        assert acc is not None, "Accuracy needs to be a number"
        last = self._get_last()
        last.is_best = True
        last.acc = acc
        # Mark all previous subepochs as non-best
        for subep in self._subepochs[:-1]:
            subep.is_best = False

    def set_checkpoints(self, checkpoints: str) -> NoReturn:
        # assert len(checkpoints) == len(self), "Mismatch in checkpoints names"
        for ckpt, sep_info in zip(checkpoints, self._subepochs):
            sep_info.checkpoint = ckpt

    def __len__(self) -> int:
        r""" Number of subepochs in the model """
        return len(self._subepochs)

    def get_best_idx(self) -> int:
        r""" Accessor to get the subepoch used as the final model """
        best = [idx for idx, subep in enumerate(self._subepochs) if subep.is_best]
        assert len(best) > 0, "No best subepoch is found"
        assert len(best) == 1, "Cannot have multiple best subepochs"
        return best[0]

    def validate_checkpoints(self) -> NoReturn:
        r""" Validates that all checkpoint files stored exist """
        for i, subepoch in enumerate(self._subepochs):
            assert subepoch.checkpoint is not None, f"No checkpoint set for subepoch {i}"
            path = Path(subepoch.checkpoint)
            if not path.exists():
                msg = f"Checkpoint #{i}: File \"{subepoch.checkpoint}\" does not exist"
                raise ValueError(msg)

    def get_best_subep_info(self) -> SubepochInfo:
        self.validate_checkpoints()
        best_idx = self.get_best_idx()
        return self._subepochs[best_idx]

    def get_best_checkpoint(self) -> str:
        r""" Returns the checkpoint corresponding to the best model """
        return self.get_best_subep_info().checkpoint

    def get_subepoch_info(self, get_all: bool = False) -> List[SubepochInfo]:
        subep_info = copy.deepcopy(self._subepochs)
        if not get_all:
            subep_info = subep_info[:self.get_best_idx() + 1]
        return subep_info

    def get_initial_checkpoint(self) -> str:
        r""" Gets the initial checkpoint file """
        return self._subepochs[0].checkpoint

    def get_best_update(self) -> int:
        r""" Helper method to get the update number corresponding to the best loss """
        subep_info = self._subepochs[self.get_best_idx()]
        return subep_info.num_updates
