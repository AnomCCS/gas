from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
import logging
from pathlib import Path
from typing import List, NoReturn

import torch

import tr_set_analysis
from poison import config, logger
import poison.dirs
import poison.filter_end_to_end
from poison.generate_results import calculate_results
import poison.learner
from poison.types import TensorGroup
import poison.utils


def parse_args() -> Namespace:
    r""" Parse, checks, and refactors the input arguments"""
    args = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # noinspection PyTypeChecker
    args.add_argument("config_file", help="Path to the configuration file", type=Path)

    args = args.parse_args()

    if not args.config_file.exists() or not args.config_file.is_file():
        raise ValueError(f"Unknown configuration file \"{args.config_file}\"")

    # Need to update directories first as other commands rely on these directories
    logger.setup()

    config.parse(args.config_file)
    if config.USE_WANDB:
        poison.wandb.setup()
    config.enable_debug_mode()
    poison.utils.set_debug_mode(seed=99)
    config.set_quiet()

    # Generates the data for learning
    args.tg, args.module = poison.utils.configure_dataset_args()
    config.print_configuration()
    return args


def _extract_bd_cnt(ex_id: List[int], tg: TensorGroup) -> int:
    r""" Print number of backdoor examples """
    if config.DATASET.is_cifar():
        is_bd = [id_val for id_val in ex_id if id_val >= poison.filter_end_to_end.BD_CIFAR_OFFSET]
        return len(is_bd)
    elif config.DATASET.is_speech():
        bd_cnt = 0
        for id_val in ex_id:
            mask = tg.test_ids == id_val
            # noinspection PyUnresolvedReferences
            bd_cnt += 1 if (tg.test_y[mask] != tg.test_adv_y[mask]).item() else 0
        return bd_cnt
    raise NotImplementedError("Unknown how to extract backdoor count for specified dataset")


def _run_adv_ident_baselines(learners: poison.learner.CombinedLearner, tg: TensorGroup,
                             ex_ids: List[int]) -> NoReturn:
    r""" Run the baselines on the specified examples """
    for idx, full_id_val in enumerate(ex_ids):
        is_bd, id_val = poison.filter_end_to_end.check_is_bd(id_val=full_id_val, tg=tg)
        if not is_bd:
            continue
        mask = tg.test_ids == id_val
        assert torch.sum(mask) == 1, "Only a single example should have the specified ID"
        x, d, y, adv_y = tg.test_x[mask], tg.test_d[mask], tg.test_y[mask], tg.test_adv_y[mask]
        if config.DATASET.is_cifar() or config.DATASET.is_mnist():
            x += d
        tr_set_analysis.baselines(learners=learners, tg=tg, ex_id=full_id_val,
                                  targ_x=x, targ_true_y=y, targ_adv_y=adv_y)


def _main(args: Namespace):
    r""" Perform backdoor target detection when CUDA is available. """
    learners = poison.learner.train_model(base_module=args.module, tg=args.tg)
    calculate_results(args.tg, erm_learners=learners)

    orig_cl_cnt, orig_bd_cnt = config.N_CL_GAS, config.N_BD_GAS

    # Select the target
    rem_cl, rem_bd = config.TOT_CL_DETECT, config.TOT_BD_DETECT
    block = learners.get_only_block()
    n_repeat = 1
    while rem_cl > 0 or rem_bd > 0:
        block.to(poison.utils.TORCH_DEVICE)
        targ_x, targ_y, ex_id = poison.filter_end_to_end.select_target(block=block, tg=args.tg)

        # Returned example IDs may be different than the ex_id selected when the results are
        # reloaded from disk.  Used the returned values to determine how many examples
        # have been processed so far.
        rem_ex_id = poison.filter_end_to_end.run(tg=args.tg, block=block, ex_ids=ex_id,
                                                 targ_x=targ_x, targ_y=targ_y, n_repeat=n_repeat)

        if n_repeat == 1:
            _run_adv_ident_baselines(learners=learners, tg=args.tg, ex_ids=rem_ex_id)

        # Decrement number of clean used
        n_bd = _extract_bd_cnt(ex_id=rem_ex_id, tg=args.tg)
        rem_bd -= n_bd
        rem_cl -= len(rem_ex_id) - n_bd
        if rem_bd <= 0 and config.N_BD_GAS > 0:
            config.update_group_cnt(n_bd=0, n_cl=config.N_CL_GAS + config.N_BD_GAS)
        elif rem_cl <= 0 and config.N_CL_GAS > 0:
            config.update_group_cnt(n_bd=config.N_BD_GAS + int(config.N_CL_GAS // 2.5), n_cl=1)
        logging.info(f"Completed repeat # {n_repeat}")
        logging.info(f"# Clean remaining: {rem_cl}")
        logging.info(f"# Backdoor remaining: {rem_bd}")
        n_repeat += 1

    config.update_group_cnt(n_bd=orig_bd_cnt, n_cl=orig_cl_cnt)
    poison.filter_end_to_end.analyze_misclassified(block=block, tg=args.tg)


if __name__ == '__main__':
    _main(parse_args())
