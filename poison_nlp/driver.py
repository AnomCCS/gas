from argparse import Namespace
import logging
from typing import NoReturn

import torch

import poison
import poison.dirs
import poison.filter_end_to_end
import poison.generate_results
import poison.logger
import poison.rep_point
import poison.utils
import sentiment.fairseq.distributed_utils
import sentiment.fairseq.options
import sentiment.fairseq.tasks
import sentiment.fairseq.utils
import sentiment.model
import tr_set_analyzer


def _parse_args() -> Namespace:
    r""" Uses fairseq's setup parser """
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = sentiment.fairseq.options.get_training_parser()
    args = sentiment.fairseq.options.parse_args_and_arch(parser)

    poison.logger.setup()
    logging.debug(args)
    poison.config.extract_config(args=args)
    poison.config.print_configuration()
    poison.dirs.update_base_dir(args=args)
    # For demo purposes only. Used to make results more repeatable in publicly released code
    poison.utils.set_random_seeds(seed=42)

    if args.distributed_init_method is None:
        sentiment.fairseq.distributed_utils.infer_init_method(args)

    sentiment.fairseq.utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    torch.cuda.set_device(args.device_id)
    poison.config.set_quiet()

    return args


def _main() -> NoReturn:
    args = _parse_args()

    trainer, tracin_hist = sentiment.model.train(args=args, config=poison.config)
    trainer.get_model().eval()  # Disable dropout

    targ_ds = sentiment.model.get_target_ds(trainer=trainer)
    valid_ds = sentiment.model.get_validation_ds(trainer=trainer)
    poison.generate_results.calculate_results("full-ds", trainer=trainer, targ_ds=targ_ds,
                                              test_ds=valid_ds)
    poison.filter_end_to_end.check_success(trainer=trainer, tracin_hist=tracin_hist,
                                           targ_ds=targ_ds, desc="Uncleaned Model",
                                           assert_check=True)

    tr_set_analyzer.adv_set_ident(trainer=trainer, tracin_hist=tracin_hist)

    detect_ds = sentiment.model.get_detect_ds(trainer=trainer)

    n_left, n_repeat = poison.config.TOT_GAS, 0
    while n_left >= 0:
        ex_ids = poison.filter_end_to_end.run(trainer=trainer, tracin_hist=tracin_hist,
                                              targ_ds=targ_ds, detect_ds=detect_ds,
                                              n_repeat=n_repeat)
        n_left -= len(ex_ids)
        n_repeat += 1
        logging.info(f"Completed iteration #{n_repeat}: # left {n_left}")

    poison.filter_end_to_end.analyze_misclassified(trainer=trainer, tracin_hist=tracin_hist,
                                                   detect_ds=detect_ds)

    poison.filter_end_to_end.analyze_full_pass(trainer=trainer, tracin_hist=tracin_hist,
                                               targ_ds=targ_ds, detect_ds=detect_ds)

    logging.info("Completed NLP target detect analysis")


if __name__ == "__main__":
    _main()
