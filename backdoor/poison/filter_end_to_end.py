__all__ = [
    "BD_CIFAR_OFFSET",
    "check_is_bd",
    "check_success",
    "run",
    "select_target"
]

import dill as pk
import logging
from pathlib import Path
from typing import List, NoReturn, Optional, Tuple

import torch
from fastai.basic_data import DeviceDataLoader
from torch import BoolTensor, LongTensor, Tensor
from torch.utils.data import TensorDataset

from . import _config as config
from . import cutoff_pred
from . import dirs
from .influence_utils import InfluenceMethod
from . import influence_utils
from . import learner
from . import targ_detect_baselines
from . import tracin
from . import tracin_utils
from .types import CustomTensorDataset, TensorGroup
from . import utils


BD_CIFAR_OFFSET = 10000
MISCLASSIFIED_IDS = set()


def _build_inf_filename(i_repeat: int) -> Path:
    r""" Influence filename for first pass without flipped labels """
    if config.DATASET.is_cifar():
        prefix = f"bd-cifar-detect_rep={i_repeat:03d}"
    elif config.DATASET.is_speech():
        prefix = f"bd-speech-detect_rep={i_repeat:03d}"
    else:
        raise NotImplemented("Influence filename not defined for specified dataset without flip")
    return utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="pk")


def _build_inf_toggle_filename(i_repeat: int) -> Path:
    r""" Influence filename for second pass with flipped labels """
    if config.DATASET.is_cifar():
        prefix = f"bd-cifar-detect_n-wrong-rep={i_repeat:03d}"
    elif config.DATASET.is_speech():
        prefix = f"bd-speech-detect_n-wrong-rep={i_repeat:03d}"
    else:
        raise NotImplemented("Influence filename not defined for specified dataset with lbl flip")
    return utils.construct_filename(prefix, out_dir=dirs.RES_DIR, file_ext="pk")


def run(block: Optional[utils.ClassifierBlock], tg: TensorGroup,
        ex_ids: Optional[List[int]], targ_x: Optional[Tensor], targ_y: Optional[LongTensor],
        n_repeat: int, wd: Optional[float] = None) -> List[int]:
    r""" Performs end-to-end filtering using epoch information and batch size """
    train_dl, _ = learner.create_fit_dataloader(tg=tg)
    train_dl = tracin_utils.configure_train_dataloader(train_dl)

    inf_path = _build_inf_filename(i_repeat=n_repeat)

    # Select the poisoned IDs to remove
    if inf_path.exists():
        logging.info(f"BD Target Detection Available on Disk: Repeat {n_repeat}")
        with open(str(inf_path), "rb") as f_in:
            ex_ids, all_inf = pk.load(f_in)
    else:
        logging.info(f"BD CIFAR Target Detection Starting Calculation: Repeat {n_repeat}")
        assert block is not None and ex_ids is not None, "Block or ex_ids not set"
        assert targ_x is not None and targ_y is not None, "Targets not set"

        all_inf = tracin.calc(block=block, train_dl=train_dl, wd=wd,
                              n_epoch=config.NUM_EPOCH, bs=config.BATCH_SIZE,
                              x_targ=targ_x, y_targ=targ_y, ex_ids=ex_ids)
        with open(str(inf_path), "wb+") as f_out:
            pk.dump((ex_ids, all_inf), f_out)
    _log_predictions(block=block, ex_ids=ex_ids, tg=tg, flip_true_lbl=False)
    tracin.log_final(block=block, ex_ids=ex_ids, tensors=all_inf)

    _log_and_calc_cutoffs(block=block, tg=tg, train_dl=train_dl, all_inf=all_inf, ex_ids=ex_ids)

    targ_detect_baselines.run(block=block, tg=tg, train_dl=train_dl, ex_ids=ex_ids,
                              toggle_lbl=True)

    return ex_ids


def analyze_misclassified(block: Optional[utils.ClassifierBlock], tg: TensorGroup,
                          wd: Optional[float] = None) -> NoReturn:
    r""" Also analyze those IDs that were misclassified by the learner wrong their wrong label """
    global MISCLASSIFIED_IDS
    n_wrong = len(MISCLASSIFIED_IDS)
    logging.info(f"# Misclassified IDs: {n_wrong}")
    logging.info(f"Misclassified IDs: {sorted(list(MISCLASSIFIED_IDS))}")

    block.eval()  # Disable dropout
    train_dl, _ = learner.create_fit_dataloader(tg=tg)
    train_dl = tracin_utils.configure_train_dataloader(train_dl)

    wrong_ids = sorted(list(MISCLASSIFIED_IDS))
    grp_size = config.N_BD_GAS + config.N_CL_GAS
    start_rng, i_repeat = 0, 0
    while start_rng < n_wrong:
        msg = f"Misclassify iteration {i_repeat}"
        logging.info(f"Starting: {msg}")

        # Select the poisoned IDs to remove
        inf_path = _build_inf_toggle_filename(i_repeat=i_repeat)

        if not inf_path.exists():
            end_range = min(len(wrong_ids), start_rng + grp_size)
            ex_ids = wrong_ids[start_rng:end_range]

            targ_x, targ_y = _get_targets_with_flipped_label(block=block, ex_ids=ex_ids, tg=tg)
            all_inf = tracin.calc(block=block, train_dl=train_dl, wd=wd,
                                  n_epoch=config.NUM_EPOCH, bs=config.BATCH_SIZE,
                                  x_targ=targ_x, y_targ=targ_y, ex_ids=ex_ids)
            with open(str(inf_path), "wb+") as f_out:
                pk.dump((ex_ids, all_inf), f_out)
        else:
            with open(str(inf_path), "rb") as f_in:
                ex_ids, all_inf = pk.load(f_in)
        _log_predictions(block=block, ex_ids=ex_ids, tg=tg, flip_true_lbl=True)
        tracin.log_final(block=block, ex_ids=ex_ids, tensors=all_inf)

        _log_and_calc_cutoffs(block=block, tg=tg, train_dl=train_dl, all_inf=all_inf,
                              ex_ids=ex_ids)

        targ_detect_baselines.run(block=block, tg=tg, train_dl=train_dl, ex_ids=ex_ids,
                                  toggle_lbl=True)
        # Increment counters to include processed IDs
        start_rng += len(ex_ids)
        i_repeat += 1

        logging.info(f"Completed: {msg} with {n_wrong - start_rng} left")

    MISCLASSIFIED_IDS = set()


def check_success(block: utils.ClassifierBlock,
                  x_targ: Tensor, y_targ: LongTensor, method: Optional[InfluenceMethod] = None,
                  ex_id: Optional[int] = None) -> NoReturn:
    r""" Logs the result of the learner """
    x_targ = config.get_test_tfms()(x_targ).to(utils.TORCH_DEVICE)

    flds = [block.name()]
    if method is not None:
        flds.append(method.value)
    flds.append("Poison Cleanse")
    if ex_id is not None:
        flds.append(f"Ex={ex_id}")

    # Compare the prediction to the target
    with torch.no_grad():
        pred = block.module.predict(x=x_targ)

    pred_lbl, targ_lbl = pred.cpu().item(), y_targ.cpu().item()
    flds += ["Poison Label:", str(targ_lbl),
             "Cleaned Label:", str(pred_lbl),
             "Final Result:", "successful" if pred_lbl != targ_lbl else "FAILED"]

    logging.info(" ".join(flds))


def _get_targets_with_flipped_label(ex_ids: List[int], block: utils.ClassifierBlock,
                                    tg: TensorGroup) -> Tuple[Tensor, LongTensor]:
    r""" Get the target values corresponding to the specified example IDs """
    if config.DATASET.is_cifar():
        assert all(id_val < BD_CIFAR_OFFSET for id_val in ex_ids), "Backdoors cant be misclassified"
    elif config.DATASET.is_speech():
        for id_val in ex_ids:
            mask = tg.test_ids == id_val
            # noinspection PyUnresolvedReferences
            assert (tg.test_y[mask] == tg.test_adv_y[mask]).item(), \
                "Speech backdoor cant be misclassified"
    # Extract each of the test examples individually
    targ_x, targ_y = [], []
    for id_val in ex_ids:
        mask = tg.test_ids == id_val
        targ_x.append(tg.test_x[mask])
        true_y = tg.test_y[mask]
        if config.DATASET.is_cifar():
            targ_y.append(true_y ^ 1)  # Xor of the binary label
        elif config.DATASET.is_speech():
            pred_y = _get_prediction(ex_id=id_val, tg=tg, block=block)
            assert pred_y != true_y.item(), "Example supposed to be misclassified"
            targ_y.append(torch.tensor([pred_y], dtype=torch.long))
        else:
            raise ValueError("Unknown how to extract ex_id label")
    # Combine the examples into the X and y tensors
    return torch.cat(targ_x, dim=0), torch.cat(targ_y, dim=0).long()


def check_is_bd(id_val: int, tg: TensorGroup) -> Tuple[bool, int]:
    r""" Checks if the specified example is backdoored example """
    if config.DATASET.is_cifar():
        is_bd, id_val = id_val >= BD_CIFAR_OFFSET, id_val % BD_CIFAR_OFFSET
    elif config.DATASET.is_speech():
        # id_val is unchanged
        bd_mask = tg.test_ids == id_val
        # noinspection PyUnresolvedReferences
        is_bd = (tg.test_y[bd_mask] != tg.test_adv_y[bd_mask]).item()
    else:
        raise ValueError("Unable to determine whether the example is a backdoor for this dataset")
    return is_bd, id_val


def _log_predictions(block: utils.ClassifierBlock, tg: TensorGroup, ex_ids: List[int],
                     flip_true_lbl: bool) -> NoReturn:
    r""" Log the predictions using only the CPU """
    f_loss = block.loss.calc_validation_loss
    for id_val in ex_ids:
        is_bd, id_val = check_is_bd(id_val=id_val, tg=tg)
        mask = tg.test_ids == id_val
        x, y = tg.test_x[mask], tg.test_y[mask]

        header = f"Ex ID {id_val}"
        # If backdoor, add the perturbation and set the y value
        if is_bd:
            x += tg.test_d[mask]
            y = tg.test_adv_y[mask]

        x = config.get_test_tfms()(x)

        if not flip_true_lbl:
            logging.debug(f"{header}: True Label: {y.item()}")
        else:
            if config.DATASET.is_cifar():
                y ^= 1  # xor
            else:
                x = x.to(utils.TORCH_DEVICE)
                # Cannot just toggle the label since may be multiclass classification for speech
                # Need to use predicted label label
                y = block.module.predict(x)
            logging.debug(f"{header}: Tested Label: {y.item()}")

        with torch.no_grad():
            loss, logits = influence_utils.get_loss_without_wd(device=utils.TORCH_DEVICE,
                                                               model=block.module,
                                                               f_loss=f_loss, x=x, y=y)
        y, pred_lbl = y.item(), block.module.determine_label(logits).item()
        logging.debug(f"{header}: Prediction: {pred_lbl}")
        is_correct = pred_lbl == y
        logging.debug(f"{header}: Model {'Correct' if is_correct else 'Mispredicted'}")
        logging.debug(f"{header}: Is Backdoor: {1 if is_bd else 0}")
        logging.debug(f"{header}: Loss: {loss.item()}")
        # Store the example if misclassified
        if not is_correct:
            global MISCLASSIFIED_IDS
            MISCLASSIFIED_IDS.add(id_val)

    logging.info("")  # Blank line for better readability


def _log_and_calc_cutoffs(block: utils.ClassifierBlock, tg: TensorGroup,
                          train_dl: DeviceDataLoader, all_inf: tracin_utils.TracInTensors,
                          ex_ids: List[int]) -> NoReturn:
    r""" Log the cutoff points for the various influence metrics """
    ds_ids, bd_ids = all_inf.full_ds_ids, all_inf.full_bd_ids
    flds = ((all_inf.gas_sim, InfluenceMethod.GAS),
            (all_inf.gas_l_sim, InfluenceMethod.GAS_L),
            )
    for idx, tmp_ex_id in enumerate(ex_ids):
        pred = _get_prediction(ex_id=tmp_ex_id, block=block, tg=tg)
        true_label = _get_true_label(ex_id=tmp_ex_id, tg=tg)
        for inf, method in flds:
            tmp_inf = inf[idx, ds_ids]
            cutoff_pred.calc(block=block, train_dl=train_dl, res_type=method,
                             pred_lbl=pred, true_lbl=true_label,
                             inf=tmp_inf, ds_ids=ds_ids, bd_ids=bd_ids, ex_id=tmp_ex_id)

    logging.info("")  # Blank line for better readability


def _get_prediction(ex_id: int, block: utils.ClassifierBlock, tg: TensorGroup) -> int:
    r""" Get the prediction for the associated Ex ID """
    if config.DATASET.is_cifar():
        is_bd, ex_id = ex_id >= BD_CIFAR_OFFSET, ex_id % BD_CIFAR_OFFSET
    # Extract the feature vector for the test example
    mask = tg.test_ids == ex_id
    x = tg.test_x[tg.test_ids == ex_id]
    # noinspection PyUnboundLocalVariable
    if config.DATASET.is_cifar() and is_bd:
        x += tg.test_d[mask]

    # Perform any transforms needed
    x = config.get_test_tfms()(x).to(utils.TORCH_DEVICE)
    with torch.no_grad():
        pred = block.module.predict(x)
    return pred.item()


def _get_true_label(ex_id: int, tg: TensorGroup) -> int:
    r""" Get the true label for the associated example ID """
    if config.DATASET.is_cifar():
        # Extract the feature vector for the test example
        mask = tg.test_ids == (ex_id % BD_CIFAR_OFFSET)
        return tg.test_y[mask].item()
    if config.DATASET.is_speech():
        mask = tg.test_ids == ex_id
        # noinspection PyUnresolvedReferences
        if (tg.test_y[mask] != tg.test_adv_y[mask]).item():
            # Is backdoored example
            return tg.test_adv_y[mask]
        return tg.test_y[mask]
    raise NotImplementedError("Unknown how to get true label for the specified dataset")


def select_target(block: utils.ClassifierBlock, tg: TensorGroup) \
        -> Tuple[Tensor, LongTensor, List[int]]:
    r""" General target selection method. """
    if config.DATASET.is_cifar():
        return select_target_cifar(block=block, tg=tg)
    if config.DATASET.is_speech():
        return select_target_speech(block=block, tg=tg)
    raise ValueError("Unknown dataset for selecting target")


def select_target_speech(block: utils.ClassifierBlock,
                         tg: TensorGroup) -> Tuple[Tensor, LongTensor, List[int]]:
    r""" Target selection method for speech """
    cl_mask = tg.test_y == tg.test_adv_y
    x, y, ids = tg.test_x[cl_mask], tg.test_y[cl_mask], tg.test_ids[cl_mask]

    # Select a random example that is not backdoored
    assert ids.numel() >= config.N_CL_GAS, "Insufficient clean examples"
    idx = torch.randperm(ids.numel())[:config.N_CL_GAS]
    x, y, ex_id = x[idx], y[idx], ids[idx].tolist()

    if config.N_BD_GAS > 0:
        x_adv, y_adv, id_adv = _select_bd_speech(block=block, tg=tg)
        x = torch.cat([x_adv, x], dim=0)
        y = torch.cat([y_adv, y], dim=0)
        ex_id = id_adv + ex_id

    assert x.shape[0] == config.N_CL_GAS + config.N_BD_GAS, "Unexpected X tensor size"
    assert len(ex_id) == config.N_CL_GAS + config.N_BD_GAS, "Unexpected number of examples"
    return x, y, ex_id


def _select_bd_speech(block: utils.ClassifierBlock,
                      tg: TensorGroup) -> Tuple[Tensor, Tensor, List[int]]:
    r""" Extract the target example from the speech dataset """
    mask = (tg.test_y == config.TARG_CLS).logical_and(tg.test_adv_y == config.POIS_CLS)
    bd_x, bd_y, ds_ids = tg.test_x[mask], tg.test_adv_y[mask], tg.test_ids[mask]

    # Only consider targets for which the attack succeeded
    # First x vector is transformed so store the second x vector which will not be transformed
    # and can be used for selecting the examples to analyze
    ds = CustomTensorDataset([bd_x, bd_x, bd_y, ds_ids], transform=config.get_test_tfms())
    dl = DeviceDataLoader.create(ds, bs=1, device=utils.TORCH_DEVICE, num_workers=0,
                                 drop_last=False, shuffle=False)
    x_succeed, y_succeed, ids_succeed = [], [], []
    for xs, xs_raw, ys, ids in dl:
        yhat = block.module.predict(xs)
        if yhat.item() == ys.item():
            x_succeed.append(xs_raw.cpu())
            y_succeed.append(ys.cpu())
            ids_succeed.append(ids.cpu())
    bd_x, bd_y = torch.cat(x_succeed, dim=0), torch.cat(y_succeed, dim=0)
    bd_ids = torch.cat(ids_succeed, dim=0)

    # Randomly select BD candidates
    assert bd_ids.numel() >= config.N_BD_GAS, "Insufficient backdoor examples give BD count"
    n_bd_candidate = bd_ids.numel()
    filt_ids = torch.randperm(n_bd_candidate)[:config.N_BD_GAS]
    bd_x, bd_y, bd_ids = bd_x[filt_ids], bd_y[filt_ids], bd_ids[filt_ids]

    return bd_x, bd_y, bd_ids.tolist()


def select_target_cifar(block: utils.ClassifierBlock,
                        tg: TensorGroup) -> Tuple[Tensor, LongTensor, List[int]]:
    r""" General target selection method. """
    # Select all elements from just the target class
    idx = torch.randperm(tg.test_y.numel())[:config.N_CL_GAS]
    x, y, ex_id = tg.test_x[idx], tg.test_y[idx], tg.test_ids[idx].tolist()

    if config.N_BD_GAS > 0:
        x_adv, y_adv, id_adv = _select_cifar_bd_example(tg=tg, block=block)
        x = torch.cat([x_adv, x], dim=0)
        y = torch.cat([y_adv, y], dim=0)
        ex_id = id_adv + ex_id

    assert x.shape[0] == config.N_CL_GAS + config.N_BD_GAS, "Unexpected X tensor size"
    assert len(ex_id) == config.N_CL_GAS + config.N_BD_GAS, "Unexpected number of examples"
    return x, y, ex_id


def _select_cifar_bd_example(tg: TensorGroup, block) -> Tuple[Tensor, Tensor, List[int]]:
    r""" Select the backdoor examples for CIFAR10 """
    mask = tg.test_y == config.TARG_CLS
    if config.has_min_perturb():
        mask = mask.logical_and(utils.build_min_perturb_mask(ds=tg.test_d))
    x_te, d_te = tg.test_x[mask], tg.test_d[mask]
    ids = tg.test_ids[mask]

    # Construct the dataloader
    ds = TensorDataset(x_te, d_te)
    dl = DeviceDataLoader.create(ds, shuffle=False, bs=config.BATCH_SIZE, drop_last=False,
                                 device=utils.TORCH_DEVICE)
    # Label all of the examples from the target class
    y_true_pred, y_adv_pred = [], []
    for xs, ds in dl:
        with torch.no_grad():
            y_true_pred.append(block.module.predict(xs))
            y_adv_pred.append(block.module.predict(xs + ds))
    # Select only those that are misclassified with backdoor and correct without it
    y_true_pred = torch.cat(y_true_pred, dim=0).cpu()
    y_adv_pred = torch.cat(y_adv_pred, dim=0).cpu()

    # Select the clean examples
    mask_cl = y_true_pred == tg.targ_y  # type: BoolTensor
    # Adversarial examples
    # noinspection PyArgumentList
    mask_adv = mask_cl.logical_and(y_adv_pred == tg.targ_adv_y)
    # Filter out any examples the backdoor did not work for
    x_adv = x_te[mask_adv] + d_te[mask_adv]
    ids_adv = ids[mask_adv]
    n_adv_valid = ids_adv.numel()

    # Randomly select a backdoor test example meeting the above criteria
    assert n_adv_valid >= config.N_BD_GAS, "Insufficient backdoor sample counts"
    idx = torch.randperm(n_adv_valid)[:config.N_BD_GAS]
    x_adv, ids_adv = x_adv[idx], ids_adv[idx] + BD_CIFAR_OFFSET

    # Merge IDs together
    y = torch.full(x_adv.shape[:1], config.POIS_CLS).long()
    return x_adv, y, ids_adv.tolist()
