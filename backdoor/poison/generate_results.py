__all__ = [
    "calculate_results",
]

from dataclasses import dataclass
import io
import logging
import re
import sys
from typing import ClassVar, List, NoReturn, Optional, Tuple

import numpy as np
import pycm

from fastai.basic_data import DeviceDataLoader
import torch
from torch import BoolTensor, LongTensor, Tensor

from . import _config as config
from . import influence_utils
from . import learner
from .learner import CombinedLearner
from .types import CustomTensorDataset, TensorGroup
from . import utils

TE_CLN_DS = "test-cl"
TE_ADV_DS = "test-adv"
TE_ADV_ONLY = "test-only-bd"


@dataclass(order=True)
class LearnerResults:
    r""" Encapsulates ALL results for a single NLP learner model """
    FIELD_SEP: ClassVar[str] = ","

    @dataclass(init=True, order=True)
    class DatasetResult:
        r""" Encapsulates results of a model on a SINGLE dataset """
        ds_size: int
        accuracy: float = None
        # auroc: float = None
        # auprc: float = None
        macro_f1: float = None
        micro_f1: float = None
        conf_matrix: pycm.ConfusionMatrix = None

    loss_name = None
    valid_loss = None

    inf_func_detect_rate = None
    rep_point_detect_rate = None
    tracin_detect_rate = None

    targ_init_adv_loss = None
    targ_true_loss = None
    targ_adv_loss = None

    tracin_stats = None

    @staticmethod
    def get_general_results_names() -> List[str]:
        r""" Returns the name of each of the results values not tied to a specific dataset """
        return ["valid_loss", "inf_func_detect_rate", "rep_point_detect_rate",
                "tracin_detect_rate", "targ_true_loss", "targ_adv_loss", "targ_init_adv_loss"]


def calculate_results(tg: TensorGroup, erm_learners: CombinedLearner) -> dict:
    r"""
    Calculates and writes to disk the model's results

    :param tg: Tensor group containing the test conditions
    :param erm_learners: Empirical risk minimization based learners
    :return: Dictionary containing results of all experiments
    """
    erm_learners.eval()

    all_res = dict()

    for block_name, block in erm_learners.blocks():  # type: str, utils.ClassifierBlock
        ds_flds = _build_ds_fields(block=block, tg=tg, tfms=config.get_test_tfms())

        res = LearnerResults()
        res.loss_name = block_name
        res.valid_loss = block.best_loss

        _get_target_losses(res=res, tg=tg, block=block)

        for ds_name, ds in ds_flds:
            # noinspection PyTypeChecker
            dl = DeviceDataLoader.create(ds, shuffle=False, drop_last=False,
                                         bs=config.BATCH_SIZE, num_workers=0,
                                         device=utils.TORCH_DEVICE)
            all_y, all_yhat = [], []
            with torch.no_grad():
                for xs, ys in dl:
                    all_y.append(ys.cpu())
                    all_yhat.append(block.module.predict(xs).cpu())

            # Iterator transforms label so transform it back
            y = torch.cat(all_y, dim=0).view([-1]).numpy()
            y_hat = torch.cat(all_yhat, dim=0).view([-1]).numpy()
            # Store for name "unlabel" or "test"
            single_res = _single_ds_results(block_name, ds_name, y, y_hat)
            res.__setattr__(ds_name, single_res)

        # Append the result
        all_res[block_name] = res
        _log_backdoor_success_rate(block=block, res=res, tg=tg)
        if config.has_min_perturb():
            _log_backdoor_success_rate(block=block, res=res, tg=tg, use_min_perturb=True)
        _log_validation_loss(block=block, res=res)

    return all_res


def _single_ds_results(block_name: str,
                       ds_name: str, y: np.ndarray,
                       y_hat: np.ndarray) -> LearnerResults.DatasetResult:
    r""" Logs and returns the results on a single dataset """
    res = LearnerResults.DatasetResult(y.shape[0])

    str_prefix = f"Dataset {ds_name}:"

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


def _extract_bd_examples(block: utils.ClassifierBlock,
                         tg: TensorGroup) -> Tuple[Tensor, Tensor, LongTensor]:
    train_dl, _ = learner.create_fit_dataloader(tg=tg)
    adv_x, adv_d, lbls = [], [], []
    for tensors in train_dl:
        batch = block.organize_batch(batch_tensors=tensors, process_mask=True)

        if batch.skip():
            continue

        mask = influence_utils.label_ids(batch.bd_ids) == influence_utils.BACKDOOR_LABEL

        # Filter the elements
        adv_x.append(batch.xs[mask].cpu())
        adv_d.append(batch.ds[mask].cpu())
        lbls.append(batch.lbls[mask].cpu())
    # Concatenate the tensors
    bd_x, bd_d = torch.cat(adv_x, dim=0), torch.cat(adv_d, dim=0)
    bd_lbls = torch.cat(lbls, dim=0).long()
    return bd_x, bd_d, bd_lbls


def _build_ds_fields(block: utils.ClassifierBlock, tg: TensorGroup, tfms):
    r""" Builds dataset fields.  Poisoned data my not be included if it has not been created yet """
    ds_flds = []

    tr_dl, val_dl = learner.create_fit_dataloader(tg=tg)
    # Separate the validation and train datasets by whether they are clean or backdoor
    for name, dev_dl in (("tr", tr_dl), ("val", val_dl)):
        dl_tensors = dev_dl.dl.dataset.tensors  # noqa
        # Extract the tensors from the dataloader
        bd_ids = dl_tensors[-2]
        all_x, all_d = dl_tensors[0], dl_tensors[1]
        all_y, all_adv_y = dl_tensors[2], dl_tensors[3]
        # Split between clean and dirty
        for is_clean in (True, False):
            if is_clean:
                mask = ~influence_utils.is_bd(bd_ids, inc_holdout=True)
            else:
                mask = influence_utils.is_bd(bd_ids, inc_holdout=False)
            # If no elements in dataset, skip
            if not block.loss.has_any(mask):
                continue

            suffix = "clean" if is_clean else "bd"
            tr_x = all_x[mask]
            if is_clean:
                tr_y = all_y[mask]
            else:
                if config.DATASET.is_mnist() or config.DATASET.is_cifar():
                    tr_x += all_d[mask]
                elif config.DATASET.is_speech():
                    pass
                else:
                    raise ValueError("Unknown dataset")
                tr_y = all_adv_y[mask]
            ds_flds.append((f"{name}-{suffix}", CustomTensorDataset([tr_x, tr_y], tfms)))

    if config.DATASET.is_cifar() or config.DATASET.is_mnist():
        ds_flds += [
                    (TE_CLN_DS, CustomTensorDataset([tg.test_x, tg.test_y], tfms)),
                    (TE_ADV_DS, CustomTensorDataset([tg.test_x + tg.test_d, tg.test_y], tfms))
                   ]
        if config.MIN_PERTURB_RATIO is not None:
            mask = utils.build_min_perturb_mask(ds=tg.test_d)
            x, y = tg.test_x + tg.test_d, tg.test_y
            ds = CustomTensorDataset([x[mask], y[mask]], tfms)
            ds_flds.append((TE_ADV_DS + "-min-perturb", ds))
    # Extract only the backdoor test examples
    elif config.DATASET.is_speech():
        # Clean Only in the Test Set
        mask = tg.test_y == tg.test_adv_y
        ds_flds.append((TE_CLN_DS, CustomTensorDataset([tg.test_x[mask], tg.test_y[mask]], tfms)))
        # Backdoor rates
        mask = (tg.test_y != tg.test_adv_y).logical_and(tg.test_y == config.TARG_CLS)
        ds = CustomTensorDataset([tg.test_x[mask], tg.test_adv_y[mask]], tfms)
        ds_flds.append((TE_ADV_ONLY, ds))
    else:
        raise ValueError("Unknown dataset for result generation")

    return tuple(ds_flds)


def _calc_predict_vector(block: utils.ClassifierBlock, x: Tensor, tfms) -> Tensor:
    r""" Construct a prediction vector for the \p create backdoor success heat map """
    ds = CustomTensorDataset([x], tfms)
    dl = DeviceDataLoader.create(ds, bs=config.BATCH_SIZE, drop_last=False, shuffle=False,
                                 num_workers=0, device=utils.TORCH_DEVICE)

    with torch.no_grad():
        y_hat = [block.module.predict(xs) for xs, in dl]
    y_hat = torch.cat(y_hat, dim=0)
    return y_hat


def _get_target_losses(res: LearnerResults, tg: TensorGroup,
                       block: utils.ClassifierBlock) -> NoReturn:
    r""" Logs and stores the target loss values """
    x = config.get_test_tfms()(tg.targ_x).to(utils.TORCH_DEVICE)
    targ_y, adv_y = tg.targ_y.to(utils.TORCH_DEVICE), tg.targ_adv_y.to(utils.TORCH_DEVICE)

    # name = block.name()
    # Initial weights used to determine change due to TracIn
    block.restore_epoch_params(ep=0, subepoch=None)
    with torch.no_grad():
        init_targ_scores = block.forward(x)
    res.targ_init_adv_loss = block.loss.calc_validation_loss(dec_scores=init_targ_scores,
                                                             labels=adv_y)
    msg = f"Te ID {config.TARG_IDX} Init Adv Loss: {res.targ_init_adv_loss.item():.3E}"
    logging.info(msg)

    # Final weights determine final weights
    block.restore_best()
    block.eval()
    with torch.no_grad():
        targ_scores = block.forward(x)

    res.targ_true_loss = block.loss.calc_validation_loss(dec_scores=targ_scores, labels=targ_y)
    logging.info(f"Te ID {config.TARG_IDX} True Loss: {res.targ_true_loss.item():.3E}")

    res.targ_adv_loss = block.loss.calc_validation_loss(dec_scores=targ_scores, labels=adv_y)
    msg = f"Te ID {config.TARG_IDX} Adversarial Loss: {res.targ_adv_loss.item():.3E}"
    logging.info(msg)


def _extract_targ_pois_idx(conf_matrix: pycm.ConfusionMatrix) -> Tuple[int, int]:
    r"""
    Extracts the index of the target and poison classes respectively from the confusion matrix's
    classes.  If the class number does not appear in the class list, -1 is returned.
    """
    classes = conf_matrix.classes
    idx_lst = []
    for cls_id in (config.TARG_CLS, config.POIS_CLS):
        for i, val in enumerate(classes):
            try:
                val = int(val)
            except ValueError:  # Cannot convert val to string
                continue
            if int(val) == cls_id:
                idx_lst.append(i)
                break
        else:
            idx_lst.append(-1)
    return tuple(idx_lst)  # noqa


def _log_mnist_cifar_bd_success_rate(block: utils.ClassifierBlock, tg: TensorGroup, tfms,
                                     ex_id: Optional[int] = None,
                                     use_min_perturb: bool = False) -> NoReturn:
    r""" Logs the success rate of the backdoor data """
    # Select only examples from the target class
    mask = tg.test_y == config.TARG_CLS
    if use_min_perturb:
        mask = mask.logical_and(utils.build_min_perturb_mask(tg.test_d))
    te_x, te_d = tg.test_x[mask], tg.test_d[mask]

    # te_y, te_adv_y = tg.test_y[mask], tg.test_adv_y[mask]
    n_targ = te_x.shape[0]

    ds = CustomTensorDataset([te_x, te_d], tfms)
    dl = DeviceDataLoader.create(ds, bs=config.BATCH_SIZE, drop_last=False, shuffle=False,
                                 num_workers=0, device=utils.TORCH_DEVICE)
    # Only consider an attack successful if without the backdoor example is classified correctly
    # and with backdoor labeled with poison class
    n_cl_pois = n_bd_pois = n_changed = n_wrong_flip = 0
    with torch.no_grad():
        for xs, ds in dl:
            y_hat = block.module.predict(xs)
            adv_y_hat = block.module.predict(xs + ds)

            cl_pois_mask = y_hat == config.POIS_CLS
            n_cl_pois += torch.sum(cl_pois_mask)  # noqa

            bd_success_mask = adv_y_hat == config.POIS_CLS  # type: BoolTensor  # noqa
            n_bd_pois += torch.sum(bd_success_mask)

            # Number of examples flipped with backdoor
            ch_mask = (y_hat == config.TARG_CLS).logical_and(bd_success_mask)  # noqa
            n_changed += torch.sum(ch_mask)
            assert n_changed <= n_bd_pois, "Cannot have more changed than those classified w. BD"

            # Flips in the wrong direction -- poison class in clean data but target class with BD.
            wrong_flip_mask = cl_pois_mask.logical_and(adv_y_hat == config.TARG_CLS)  # noqa
            n_wrong_flip += torch.sum(wrong_flip_mask)

    filt_str = "" if not use_min_perturb else " (Min Perturb)"

    header = influence_utils.build_log_start_flds(block=block, ep=None, subepoch=None,
                                                  res_type=None, ex_id=ex_id)
    for name, cnt in (("Clean", n_cl_pois), ("Backdoored", n_bd_pois)):
        rate = cnt / n_targ
        logging.info(f"{header} {name} Targ Class Fails{filt_str}: {cnt} / {n_targ} -- {rate:.3%}")

    # Fraction where backdoor flipped label to the poison class
    rate = n_changed / n_targ
    logging.info(f"{header} BD Success Rate{filt_str}: {n_changed} / {n_targ} -- {rate:.3%}")
    # Fraction where backdoor flipped label from the poison class in clean data to the target
    # label in the backdoored data
    rate = n_wrong_flip / n_targ
    logging.info(f"{header} BD Wrong Flip Rate{filt_str}: {n_wrong_flip} / {n_targ} -- {rate:.3%}")


def _log_speech_bd_success_rate(block: utils.ClassifierBlock, res: LearnerResults) -> NoReturn:
    r""" Log the success rate for the backdoors for specifically the speech dataset """
    conf_matrix = res.__getattribute__(TE_ADV_ONLY).conf_matrix
    adv_mtrx = conf_matrix.to_array()
    targ_idx, pois_idx = _extract_targ_pois_idx(conf_matrix=conf_matrix)

    # Extract number of backdoored elements
    n_te_targ = np.sum(adv_mtrx)
    assert n_te_targ == np.sum(adv_mtrx[pois_idx]), "Some elements do not have the poison label"

    adv_cnt = adv_mtrx[pois_idx, pois_idx]
    # No elements may be from target class so need to handle that case specially
    targ_cnt = adv_mtrx[pois_idx, targ_idx] if targ_idx != -1 else 0
    other_cnt = n_te_targ - targ_cnt - adv_cnt
    assert other_cnt >= 0, "Invalid count for the remaining classes"

    for name, cnt in (("Poison", adv_cnt), ("Target", targ_cnt), ("other", other_cnt)):
        rate = cnt / n_te_targ
        flds = [f"{block.name()} Backdoored test data", str(cnt), "/", str(n_te_targ),
                f"({rate:.1%})", f"classified {name} class"]
        logging.info(" ".join(flds))


def _log_backdoor_success_rate(block: utils.ClassifierBlock, res: LearnerResults,
                               tg: TensorGroup, use_min_perturb: bool = False) -> NoReturn:
    r""" Logs the backdoor success rate """
    if config.DATASET.is_mnist() or config.DATASET.is_cifar():
        _log_mnist_cifar_bd_success_rate(block=block, tg=tg, tfms=config.get_test_tfms(),
                                         use_min_perturb=use_min_perturb)
    elif config.DATASET.is_speech():
        assert not use_min_perturb, "Minimum perturbation ratio not applicable to speech"
        _log_speech_bd_success_rate(block=block, res=res)
    else:
        raise ValueError("Unknown dataset to log the backdoor success rate")


def _log_validation_loss(block: utils.ClassifierBlock, res: LearnerResults) -> NoReturn:
    r""" Log the validation loss for subsequent analysis """
    header = influence_utils.build_log_start_flds(block=block, ep=None, subepoch=None,
                                                  res_type=None, ex_id=None)
    logging.info(f"{header} Validation Loss: {res.valid_loss:.3E}")
