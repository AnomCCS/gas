__all__ = [
    "ANOM_CUTOFF",
    "BATCH_SIZE",
    "CUTOFF_STDEV",
    "DAMP",
    "FIRST_PASS_ANON_CUTOFF",
    "HVP_BATCH_SIZE",
    "N_CLASSES",
    "N_GRP",
    "N_TRAIN",
    "POISON_CNT",
    "QUIET",
    "R_DEPTH",
    "SCALE",
    "TASK",
    "TOT_GAS",
    "TRACIN_DECODER_ONLY",
    "T_REPEATS",
    "WEIGHT_DECAY",
    "extract_config",
    "print_configuration",
]

import argparse
import logging
from typing import NoReturn


TASK = "Sentiment"
QUIET = True

CUTOFF_STDEV = 1.5

N_CLASSES = None

N_TRAIN = None
POISON_CNT = None
_POISON_SEP = "***"

BATCH_SIZE = None
WEIGHT_DECAY = None

TRACIN_DECODER_ONLY = False

# Influence function hyperparameters
HVP_BATCH_SIZE = 1
DAMP = 1E-2
SCALE = 1E6
R_DEPTH = 6740
T_REPEATS = 10

N_GRP = 3
TOT_GAS = 125

FIRST_PASS_ANON_CUTOFF = 1
ANOM_CUTOFF = 10


def extract_config(args: argparse.Namespace) -> NoReturn:
    r""" Extracts the configuration from the parsed input arguments """
    global N_CLASSES, POISON_CNT
    N_CLASSES = args.num_classes
    POISON_CNT = len(args.poison_example.split(_POISON_SEP))

    global BATCH_SIZE, WEIGHT_DECAY
    BATCH_SIZE, WEIGHT_DECAY = args.max_sentences, args.weight_decay


# noinspection PyUnresolvedReferences
def set_train_set_size(trainer: "Trainer") -> NoReturn:
    r""" Set the training set size """
    assert POISON_CNT is not None, "POISON_CNT not yet set"
    itr = trainer.get_train_iterator(epoch=0)
    global N_TRAIN
    N_TRAIN = len(itr.dataset) - POISON_CNT  # noqa
    logging.debug(f"Training dataset size set to {N_TRAIN}")


def set_quiet() -> NoReturn:
    r""" Sets basic quiet mode """
    return QUIET


def print_configuration() -> NoReturn:
    logging.info(f"Task: {TASK}")
    logging.info(f"# Classes: {N_CLASSES}")
    logging.info(f"# Poison: {POISON_CNT}")
    logging.info(f"Batch Size: {BATCH_SIZE}")
    logging.info(f"Weight Decay: {WEIGHT_DECAY}")

    logging.info(f"TracIn Decoder Only: {TRACIN_DECODER_ONLY}")

    logging.info(f"HVP Batch Size: {HVP_BATCH_SIZE}")
    logging.info(f"HVP Recursive Depth: {R_DEPTH}")
    logging.info(f"HVP T Repeat: {T_REPEATS}")
    logging.info(f"HVP Damp Param: {DAMP}")
    logging.info(f"HVP Scale Param: {SCALE}")

    logging.info(f"Quiet: {QUIET}")
