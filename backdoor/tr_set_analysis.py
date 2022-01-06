__all__ = [
    "baselines",
    "run_other_examples",
]

import logging
import random
from typing import NoReturn, Optional

from torch import LongTensor, Tensor

from poison import config
import poison.filter_end_to_end
import poison.influence_func
from poison.influence_utils import InfluenceMethod
import poison.learner
import poison.rep_point
import poison.tracin
import poison.tracin_utils
from poison.types import TensorGroup
import poison.utils


def _log_example_prediction(block: poison.utils.ClassifierBlock, x: Tensor,
                            y_true: Optional[Tensor]) -> NoReturn:
    r""" Helper method to print the predicted class """
    pass
    # assert x.shape[0], "Only a single training example can be logged"
    # if y_true is not None:
    #     assert y_true.shape[0] == y_true.numel() == 1, "Y shape does not match expectation"
    #
    # x_tfms = config.get_test_tfms()(x.to(poison.utils.TORCH_DEVICE))
    # y_hat = block.module.predict(x_tfms)
    #
    # msg = f"{block.name()} -- Example Predicted Label: {y_hat.item()}"
    # if y_true is not None:
    #     msg += f", True Label: {y_true.item()}"
    #
    # logging.info(msg)


def baselines(learners: poison.learner.CombinedLearner, tg: TensorGroup, ex_id: int,
              targ_x: Tensor, targ_true_y: LongTensor, targ_adv_y: LongTensor) -> NoReturn:
    train_dl, _ = poison.learner.create_fit_dataloader(tg=tg)

    for y_tensor in targ_true_y, targ_adv_y:
        for _, block in learners.blocks():
            _log_example_prediction(block=block, x=targ_x, y_true=y_tensor)

    rep_point_methods = (InfluenceMethod.REP_POINT,
                         InfluenceMethod.REP_POINT_SIM,
                         )
    for method in rep_point_methods:
        rep_targ_vals = poison.rep_point.calc_representer_vals(erm_learners=learners,
                                                               train_dl=train_dl,
                                                               test_x=targ_x, ex_id=ex_id,
                                                               method=method)
        idx = 0  # Only applies if multiple test examples
        poison.rep_point.log_representer_scores(idx, targ_adv_y.item(), rep_targ_vals, ex_id=ex_id,
                                                method=method, file_prefix="adv-cls")

    for block_name, block in learners.blocks():  # type: str, poison.utils.ClassifierBlock
        if not block.is_poisoned():
            continue

        # Disable augmentation in transforms initially
        tmp_tr_dl = poison.tracin_utils.configure_train_dataloader(train_dl)
        poison.influence_func.calc(block=block, tr_dl=tmp_tr_dl, te_x=targ_x, te_y=targ_adv_y,
                                   ex_id=ex_id)


def run_other_examples(block: poison.utils.ClassifierBlock, tg: TensorGroup,
                       wd: Optional[float] = None) -> NoReturn:
    r""" Tail comparison of all examples """

    train_dl, _ = poison.learner.create_fit_dataloader(tg=tg)
    # Extract the examples from the same class
    mask = tg.test_y == config.TARG_CLS
    te_x, te_y, te_adv_y = tg.test_x[mask], tg.test_y[mask], tg.test_adv_y[mask]
    te_d, te_ids = tg.test_d[mask], tg.test_ids[mask]

    n_test = te_ids.numel()  # Number of test examples
    n_test_lst = list(range(n_test))
    if config.DATASET.is_speech():
        random.shuffle(n_test_lst)
    for i in n_test_lst:
        if config.DATASET.is_speech():
            targ_x, targ_y = te_x[i:i + 1], te_adv_y[i:i + 1]
        else:
            raise NotImplementedError("Non-speech examples not yet supported")

        ex_id = te_ids[i].item()
        # Log whether the example is poisoned
        if config.DATASET.is_speech():
            is_pois = (te_y[i] != te_adv_y[i]).item()  # noqa
            logging.info(f"Test Example {ex_id} is {'POISONED' if is_pois else 'clean'}")
        else:
            raise NotImplementedError("Non-speech examples not yet supported")

        poison.tracin.calc(block=block, train_dl=train_dl, n_epoch=config.NUM_EPOCH, wd=wd,
                           x_targ=targ_x, y_targ=targ_y, bs=config.BATCH_SIZE, ex_ids=ex_id)
