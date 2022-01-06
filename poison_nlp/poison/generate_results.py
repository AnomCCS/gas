__all__ = [
    "calculate_results",
]

from dataclasses import dataclass
import io
import logging
import re
import sys
from typing import ClassVar, NoReturn, Optional

import numpy as np
import pycm

import torch

from . import _config as config
from . import utils

TR_CL_DS = "tr-cl-ds"
TR_BD_DS = "tr-bd-ds"

TE_CLN_DS = "test"
TARG_ADV_DS = "targ-adv"


@dataclass(order=True)
class LearnerResults:
    r""" Encapsulates ALL results for a single NLP learner model """
    FIELD_SEP: ClassVar[str] = ","

    @dataclass(init=True, order=True)
    class DatasetResult:
        r""" Encapsulates results of a model on a SINGLE dataset """
        ds_size: int
        accuracy: float = None
        conf_matrix: pycm.ConfusionMatrix = None


def calculate_results(model_name: str, trainer, targ_ds, test_ds) -> NoReturn:
    r""" Calculates the model's results """
    trainer.model.eval()

    train_ds = utils.get_train_ds(trainer=trainer)
    n_tr = len(train_ds)
    first_bd = n_tr - config.POISON_CNT
    _analyze_ds(model_name=model_name, ds_name=TR_BD_DS, trainer=trainer, ds=train_ds,
                start_idx=first_bd, end_idx=n_tr)

    _analyze_ds(model_name=model_name, ds_name=TARG_ADV_DS, trainer=trainer, ds=targ_ds)
    _analyze_ds(model_name=model_name, ds_name=TE_CLN_DS, trainer=trainer, ds=test_ds)


def _analyze_ds(model_name: str, ds_name: str, trainer, ds,
                start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> NoReturn:
    r""" Analyzes the test dataset and performs analysis on it """
    start_idx = start_idx if start_idx is not None else 0
    end_idx = end_idx if end_idx is not None else len(ds)

    # Get each sample's prediction and build into a tensor
    all_lbl, all_pred = [], []
    for id_val in range(start_idx, end_idx):
        sample = utils.get_ds_sample(idx=id_val, ds=ds, trainer=trainer)
        with torch.no_grad():
            logits = utils.trainer_forward(trainer=trainer, sample=sample)
        pred = torch.argmax(logits, dim=1)

        # Store the values for the sample in the list
        all_lbl.append(utils.get_sample_label(sample=sample).cpu().item())
        all_pred.append(pred.cpu().long().item())

    # Construct the true and predicted labels
    all_lbl, all_pred = torch.tensor(all_lbl), torch.tensor(all_pred)
    _single_ds_results(model_name=model_name, ds_name=ds_name,
                       y=all_lbl.numpy(), y_hat=all_pred.numpy())


def _single_ds_results(model_name: str, ds_name: str, y: np.ndarray,
                       y_hat: np.ndarray) -> LearnerResults.DatasetResult:
    r""" Logs and returns the results on a single dataset """
    res = LearnerResults.DatasetResult(y.shape[0])

    str_prefix = f"{model_name} {ds_name}:"

    logging.debug(f"{str_prefix} Dataset Size: {res.ds_size:,}")
    # Pre-calculate fields needed in other calculations
    res.conf_matrix = pycm.ConfusionMatrix(actual_vector=y, predict_vector=y_hat)

    # noinspection PyUnresolvedReferences
    res.accuracy = res.conf_matrix.Overall_ACC
    logging.debug(f"{str_prefix} Accuracy: {100. * res.accuracy:.3}%")

    # Write confusion matrix to a string
    sys.stdout = cm_out = io.StringIO()
    res.conf_matrix.print_matrix()
    sys.stdout = sys.__stdout__
    # Log the confusion matrix
    cm_str = cm_out.getvalue()
    logging.debug(f"{str_prefix} Confusion Matrix: \n{cm_str}")
    res.conf_matrix_str = re.sub(r"\s+", " ", str(cm_str.replace("\n", " ")))

    return res
