from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path

from poison import config, logger
import poison.dirs
import poison.filter_end_to_end
from poison.generate_results import calculate_results
import poison.learner
import poison.utils

import train_initial_model
import tr_set_analysis


def parse_args() -> Namespace:
    r""" Parse, checks, and refactors the input arguments"""
    args = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # noinspection PyTypeChecker
    args.add_argument("config_file", help="Path to the configuration file", type=Path)

    args.add_argument("-d", help="Debug mode -- Disable non-determinism", action="store_true")
    args = args.parse_args()

    if not args.config_file.exists() or not args.config_file.is_file():
        raise ValueError(f"Unknown configuration file \"{args.config_file}\"")

    # Need to update directories first as other commands rely on these directories
    logger.setup(quiet_mode=False)

    config.parse(args.config_file)
    if config.USE_WANDB:
        poison.wandb.setup()
    if args.d:
        config.enable_debug_mode()
        poison.utils.set_debug_mode(seed=1)
    config.set_quiet()

    # Generates the data for learning
    args.tg, args.module = poison.utils.configure_dataset_args()
    config.print_configuration()
    return args


def _main(args: Namespace):
    module, tg = args.module, args.tg
    if config.USE_PRETRAIN:
        module = poison.learner.pretrain(base_module=module, tg=tg)
        module.cpu()

    train_initial_model.select_target(module=module, tg=tg)
    learners = poison.learner.train_model(base_module=module, tg=tg)

    block = learners.get_only_block()
    calculate_results(tg, block=block)
    block.cpu()  # Move to cpu to reduce GPU memory footprint

    ex_id = config.TARG_IDX
    tr_set_analysis.influence_methods(block=block, ex_id=ex_id, tg=tg, init_module=module,
                                      targ_x=tg.targ_x, targ_y=tg.targ_y)

    tr_set_analysis.collect_baselines(tg=tg, init_module=module, ex_id=ex_id)


if __name__ == '__main__':
    _main(parse_args())
