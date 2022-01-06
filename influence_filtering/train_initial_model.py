__all__ = [
    "select_target",
]

import dill as pk
import logging
from typing import NoReturn, Union

from fastai.basic_data import DeviceDataLoader
import torch
from torch import BoolTensor, LongTensor

from poison import config
from poison.datasets.types import LearnerModule
import poison.dirs
import poison.generate_results
import poison.learner
from poison.types import CustomTensorDataset, TensorGroup
from poison.utils import ClassifierBlock


def select_target(module: LearnerModule, tg: TensorGroup):
    r""" Select target by examining fail count on the initial models """
    n_fails = torch.zeros(config.N_TEST, dtype=torch.long).long()

    # Store the selected targeted
    out_dir = poison.dirs.RES_DIR
    path = poison.utils.construct_filename("emp-sel-targ", file_ext="pk", out_dir=out_dir)

    if not path.exists():
        for i in range(1, config.NUM_INIT_TRAIN + 1):
            learners = poison.learner.train_model(base_module=module, tg=tg, n_subep=1,
                                                  name_prefix=f"sel-targ-{i:02d}",
                                                  clear_saved_models=True,
                                                  save_fin_model=False)

            block = learners.get_only_block()
            poison.generate_results.calculate_results(tg, block=block, log_targ=False)
            _update_cumul_fail_counts(block=block, tg=tg, n_fails=n_fails)

            _log_hist_fail_count(itr=i, tg=tg, n_fails=n_fails)

        # Extract a single target ID and get its fail rate
        targ_id, fail_rate = _select_target_id(tg=tg, n_fails=n_fails)
        with open(path, "wb+") as f_out:
            pk.dump((targ_id, fail_rate, n_fails), f_out)

    # Reload the target from disk and document its failure rate information
    with open(path, "rb") as f_in:
        targ_id, fail_rate, _ = pk.load(f_in)
    logging.info(f"Empirically Selected Target ID: {targ_id}")
    logging.info(f"Empirically Selected Target Fail Rate: {fail_rate}")
    _update_target(tg=tg, targ_id=targ_id)


def _update_cumul_fail_counts(block: ClassifierBlock, tg: TensorGroup,
                              n_fails: LongTensor) -> NoReturn:
    r""" Updates the cumulative test fail count for each example """
    block.eval()

    te_tensors = [tg.test_x, tg.test_y, tg.test_ids]
    te_ds = CustomTensorDataset(te_tensors, transform=config.get_test_tfms())

    # Calculate the accuracy and increment number of fails
    te_dl = DeviceDataLoader.create(te_ds, bs=config.BATCH_SIZE, device=poison.utils.TORCH_DEVICE,
                                    num_workers=poison.utils.NUM_WORKERS,
                                    drop_last=False, shuffle=False)
    for tensors in te_dl:
        batch = block.organize_batch(batch_tensors=tensors)
        with torch.no_grad():
            yhat = block.module.predict(batch.xs)
        # Mark all examples from target class mis
        mask = yhat != batch.lbls
        ids = batch.ids[mask]
        n_fails[ids] += 1


def _log_hist_fail_count(itr: int, tg: TensorGroup, n_fails: LongTensor) -> NoReturn:
    r"""
    Prints histogram of the cumulative failures for each example.

    :param itr: Number of models trained so far
    :param tg: Complete \p TensorGroup object
    :param n_fails: Tensor for number of times each ID failed
    """
    n_fails = n_fails[tg.test_ids]
    for i, desc in enumerate(("All", "Target Class Only")):
        hist, n_ele = torch.histc(n_fails.float(), itr + 1), n_fails.numel()
        # Describe current setup
        itr_str = "Final" if itr == config.NUM_INIT_TRAIN else f"#{itr}"
        header = f"Sel Targ Model {itr_str} ({desc})"

        for i_fail, cnt in enumerate(hist):
            frac = cnt.item() / n_ele
            logging.info(f"{header}: {i_fail} Cumul. Fails: {cnt:.0f} / {n_ele} ({frac:.1%})")
        if i == 0:
            n_fails = n_fails[tg.test_y == config.TARG_CLS]


def _select_target_id(tg: TensorGroup, n_fails: LongTensor) -> Union[int, float]:
    r""" Selects a specific target from the test set based on the empirical failure rate """
    fail_rate = n_fails[tg.test_ids].float() / config.NUM_INIT_TRAIN
    # Constrain the fail rate
    mask = tg.test_y == config.TARG_CLS  # type: BoolTensor
    mask.logical_and_(fail_rate >= config.MIN_FAIL_RATE)  # noqa
    mask.logical_and_(fail_rate <= config.MAX_FAIL_RATE)  # noqa
    # Select uniformly at random
    n_candidates = torch.sum(mask).item()
    assert n_candidates > 0, "No candidates were in specified fail range"
    header = f"#Test instances in misclassification range " \
             f"[{config.MIN_FAIL_RATE:.2f},{config.MAX_FAIL_RATE:.2f}]"
    logging.info(f"{header}: {n_candidates}")
    idx = torch.randint(n_candidates, size=(1,)).item()

    return tg.test_ids[mask][idx].item(), fail_rate[mask][idx].item()  # noqa


def _update_target(tg: TensorGroup, targ_id: int) -> NoReturn:
    r""" Updates the target index """
    mask = tg.test_ids == targ_id

    # Filter the target
    tg.targ_x, tg.targ_y = tg.test_x[mask], tg.test_y[mask]
    tg.targ_ids = tg.test_ids[mask]

    config.override_targ_idx(targ_idx=targ_id)
    logging.info(f"Selected Target Label: {tg.targ_y.item()}")
