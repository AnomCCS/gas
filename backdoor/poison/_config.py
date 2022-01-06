__all__ = [
    "BACKDOOR_ATTACK",
    "BACKDOOR_CNT",
    "BACKDOOR_DELTA",
    "BACKDOOR_HOLDOUT",
    "BATCH_SIZE",
    "DAMP",
    "DATASET",
    "HVP_BATCH_SIZE",
    "LEARNING_RATE",
    "NUM_EPOCH",
    "NUM_FF_LAYERS",
    "NUM_SUBEPOCH",
    "N_BD_GAS",
    "N_CLASSES",
    "N_CL_GAS",
    "N_FULL_TR",
    "N_TRAIN",
    "N_TEST",
    "ORIG_POIS_CLS",
    "ORIG_TARG_CLS",
    "PLOT",
    "POIS_CLS",
    "QUIET",
    "R_DEPTH",
    "SCALE",
    "SGD_MOMENTUM",
    "TARG_CLS",
    "TARG_IDX",
    "TOT_BD_DETECT",
    "TOT_CL_DETECT",
    "T_REPEATS",
    "WEIGHT_DECAY",
    "enable_debug_mode",
    "get_test_tfms",
    "get_train_tfms",
    "has_min_perturb",
    "parse",
    "print_configuration",
    "set_tfms",
    "update_group_cnt",
]

import logging
from pathlib import Path
from typing import Callable, NoReturn, Optional, Union

from ruamel.yaml import YAML

from .datasets.types import BackdoorAttack, PoisonDataset
from .types import LearnerParams, PathOrStr, TensorGroup

DATASET = None  # type: Optional[PoisonDataset]
DATASET_KEY = "dataset"
ATTACK_KEY = "backdoor_attack"

DEBUG = False

N_FULL_TR = -1  # Full training set size
N_TRAIN = -1  # Size of actual (filtered) training set used
N_TEST = -1

N_CLASSES = -1

NUM_FF_LAYERS = None

OPTIM = ""
SGD_MOMENTUM = 0

BATCH_SIZE = -1
NUM_EPOCH = -1
NUM_SUBEPOCH = -1
LEARNING_RATE = 1E-3
WEIGHT_DECAY = 1E-4

# Fraction of training samples used for
VALIDATION_SPLIT_RATIO = 1 / 6

TARG_IDX = None
ORIG_POIS_CLS = None
ORIG_TARG_CLS = None

POIS_CLS = None
TARG_CLS = None

QUIET = False
USE_WANDB = False

TRAIN_TFMS = None
TEST_TFMS = None

HVP_BATCH_SIZE = -1
DAMP = None
SCALE = None
R_DEPTH = None
T_REPEATS = None

ANOM_CUTOFF = 10

BACKDOOR_ATTACK = None  # type: Optional[BackdoorAttack]
BACKDOOR_CNT = 0
BACKDOOR_HOLDOUT = 0
BACKDOOR_DELTA = 1.0
MIN_PERTURB_RATIO = None

N_BD_GAS = None
N_CL_GAS = None
TOT_BD_DETECT = None
TOT_CL_DETECT = None

PLOT = False

LEARNER_CONFIGS = dict()


def parse(config_file: PathOrStr) -> NoReturn:
    r""" Parses the configuration """
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Unable to find config file {config_file}")
    if not config_file.is_file():
        raise FileExistsError(f"Configuration {config_file} does not appear to be a file")

    with open(str(config_file), 'r') as f_in:
        all_yaml_docs = YAML().load_all(f_in)

        base_config = next(all_yaml_docs)
        _parse_general_settings(base_config)
        # _parse_learner_specific_settings(all_yaml_docs)


def _parse_general_settings(config) -> NoReturn:
    r"""
    Parses general settings for the learning configuration including the dataset, priors, positive
    & negative class information.  It also extracts general learner information
    """
    module_dict = _get_module_dict()
    for key, val in config.items():
        # if key.lower() == CENTROIDS_KEY:
        #     _parse_centroids(module_dict, val)
        # elif key.lower() == DATASET_KEY:
        if key.lower() == DATASET_KEY:
            ds_name = val.upper()
            try:
                module_dict[key.upper()] = PoisonDataset[ds_name]
            except KeyError:
                raise ValueError(f"Unknown dataset {ds_name}")
        elif key.lower() == ATTACK_KEY:
            _parse_attack_mode(file_val=val)
        # elif _is_invalid_neg_bias_key(key):
        #     raise ValueError(f"{key.upper()} is an invalid key for a configuration file")
        # Removed to allow string class names for 20 newsgroups
        # elif key.lower().endswith("_classes"):
        #     module_dict[key.upper()] = list(int(x) for x in val)
        # Drop in replacement field
        else:
            key = key.upper()
            if key not in module_dict:
                raise ValueError(f"Unknown configuration field \"{key}\"")
            module_dict[key] = val

    # if DATASET.is_newsgroups():
    #     _parse_newsgroups(module_dict)
    global ORIG_POIS_CLS, ORIG_TARG_CLS
    assert ORIG_POIS_CLS is None and ORIG_TARG_CLS is None, "Original values not set by user"
    ORIG_POIS_CLS = POIS_CLS
    ORIG_TARG_CLS = TARG_CLS
    # Sanity checks nothing is out of whack
    _verify_configuration()

    _configure_targ_detect_counts()


def _configure_targ_detect_counts() -> NoReturn:
    r""" Sets the variables for target detection  """
    if DATASET.is_speech():
        _define_speech_targ_detect_counts()
    elif DATASET.is_cifar():
        _define_cifar_targ_detect_counts()
    else:
        raise ValueError("Unknown dataset to set the ID counts")


def _define_speech_targ_detect_counts() -> NoReturn:
    r""" Configure the backdoor, clean, and total backdoor counts for speech target detection """
    global N_BD_GAS, N_CL_GAS, TOT_BD_DETECT, TOT_CL_DETECT
    TOT_CL_DETECT = 220
    TOT_BD_DETECT = 10
    logging.debug("Tune the number of simultaneous examples analyzed based on your GPU memory")
    N_BD_GAS = 1
    N_CL_GAS = 6


def _define_cifar_targ_detect_counts() -> NoReturn:
    r""" Configure the backdoor, clean, and total backdoor counts for CIFAR10 target detection """
    global N_BD_GAS, N_CL_GAS, TOT_BD_DETECT, TOT_CL_DETECT
    TOT_CL_DETECT = 250
    TOT_BD_DETECT = 35
    logging.debug("Tune the number of simultaneous examples analyzed based on your GPU memory")
    N_BD_GAS = 1
    N_CL_GAS = 6


def _parse_attack_mode(file_val: str) -> NoReturn:
    r""" Parses the attack name """
    # Standardize the attack name
    orig_file_val, file_val = file_val, file_val.upper().replace("-", "_")

    # Populate the poison attack
    global BACKDOOR_ATTACK
    for atk in BackdoorAttack:
        if file_val == atk.name:
            BACKDOOR_ATTACK = atk
            break
    else:
        raise ValueError(f"Unknown backdoor attack \"{orig_file_val}\"")


def _get_module_dict() -> dict:
    r""" Standardizes construction of the module dictionary """
    return globals()


def _verify_configuration() -> NoReturn:
    r""" Sanity checks the configuration """
    if DATASET is None:
        raise ValueError("A dataset must be specified")

    if VALIDATION_SPLIT_RATIO <= 0 or VALIDATION_SPLIT_RATIO >= 1:
        raise ValueError("Validation split ratio must be in range (0,1)")

    supported_optims = ["adam", "sgd", "adamw"]
    if OPTIM is None or not OPTIM:
        raise ValueError("Configuration does not specify an optimizer")
    if OPTIM.lower() not in supported_optims:
        raise ValueError(f"{OPTIM} is an unknown or unsupported optimizer")

    if LEARNING_RATE <= 0:
        raise ValueError("Learning rate must be positive")
    if NUM_EPOCH <= 0:
        raise ValueError("Number of training epochs must be positive")
    if NUM_SUBEPOCH <= 0:
        raise ValueError("Number of training epochs must be positive")
    if WEIGHT_DECAY < 0:
        raise ValueError("Weight decay must be non-negative")

    # if HVP_BATCH_SIZE < 0:
    #     raise ValueError("Hessian vec. product batch size must be non-negative")

    if BACKDOOR_CNT < 0:
        raise ValueError("Backdoor count must be positive")
    if BACKDOOR_HOLDOUT < 0:
        raise ValueError("Backdoor holdout must be non-negative")
    if BACKDOOR_HOLDOUT >= BACKDOOR_CNT:
        raise ValueError("Must use at least one backdoor example")

    if MIN_PERTURB_RATIO is not None and (MIN_PERTURB_RATIO < 0 or MIN_PERTURB_RATIO > 1):  # noqa
        raise ValueError("Minimum perturb ratio must be in range (0,1)")

    # noinspection PyTypeChecker
    if NUM_FF_LAYERS is None or NUM_FF_LAYERS < 0:
        raise ValueError("Number of FF layers must be non-negative")


def print_configuration(log: Callable = logging.info) -> NoReturn:
    r""" Print the configuration settings """
    log(f"Dataset: {DATASET.name}")

    log(f"Number Training Examples: {N_TRAIN:,}")
    log(f"Maximum # Backdoor Examples: {BACKDOOR_CNT:,}")
    log(f"# Backdoor Held Out: {BACKDOOR_HOLDOUT:,}")
    log(f"Inductive Test Set Size: {N_TEST:,}")
    log(f"# Clean GAS: {N_CL_GAS}")
    log(f"# Backdoor GAS: {N_BD_GAS}")

    log(f"# FF Layers: {NUM_FF_LAYERS}")

    log(f"Optimizer: {OPTIM.lower()}")
    log(f"Batch Size: {BATCH_SIZE}")
    log(f"# Epoch: {NUM_EPOCH}")
    log(f"# Subepoch: {NUM_SUBEPOCH}")
    log(f"Learning Rate: {LEARNING_RATE:.0E}")
    log(f"Weight Decay: {WEIGHT_DECAY:.0E}")

    log(f"Target Class ID: {TARG_CLS}")
    log(f"Target Example Index: {TARG_IDX}")
    log(f"Poison Class ID: {POIS_CLS}")
    log(f"Backdoor Attack Type: {BACKDOOR_ATTACK.value}")  # noqa
    log(f"Backdoor Delta: {BACKDOOR_DELTA:.3f}")  # noqa
    perturb_ratio = "None" if MIN_PERTURB_RATIO is None else f"{MIN_PERTURB_RATIO:.0%}"
    log(f"Minimum Backdoor Perturb Ratio: {perturb_ratio}")

    log(f"Quiet Mode: {QUIET}")

    log(f"HVP Batch Size: {HVP_BATCH_SIZE}")
    log(f"HVP Recursive Depth: {R_DEPTH}")
    log(f"HVP T Repeat: {T_REPEATS}")
    log(f"HVP Damp Param: {DAMP}")
    log(f"HVP Scale Param: {SCALE:.0E}")


def reset_learner_settings() -> NoReturn:
    r""" DEBUG ONLY.  Reset the settings specific to individual learners/loss functions """
    global LEARNER_CONFIGS
    LEARNER_CONFIGS = dict()


def has_min_perturb() -> bool:
    r""" Returns \p True if a minimum perturb setting applies """
    return MIN_PERTURB_RATIO is not None


def set_layer_counts(ff_layers: Optional[int] = None,
                     sigma_layers: Optional[int] = None) -> NoReturn:
    r""" Set the number of learner layers """
    assert ff_layers is not None or sigma_layers is not None, "Must set at least one layer count"

    if ff_layers is not None:
        global NUM_FF_LAYERS
        NUM_FF_LAYERS = ff_layers


def set_ds_sizes(n_full_tr: Optional[int] = None, n_train: Optional[int] = None,
                 # n_max_adv: Optional[int] = None,
                 n_test: Optional[int] = None) -> NoReturn:
    r""" Optionally sets the dataset sizes """
    global N_FULL_TR, N_TRAIN, N_TEST
    if n_full_tr is not None:
        N_FULL_TR = n_full_tr
    if n_train is not None:
        N_TRAIN = n_train
    # if n_max_adv is not None:
    #     N_ADV = n_max_adv
    if n_test is not None:
        N_TEST = n_test


def set_all_ds_sizes(n_full_tr: int, tg: TensorGroup) -> NoReturn:
    r""" Sets the all dataset sizes using a TensorGroup """
    n_train = tg.tr_x.shape[0] + tg.val_x.shape[0]
    n_test = tg.test_x.shape[0]
    set_ds_sizes(n_full_tr=n_full_tr, n_train=n_train, n_test=n_test)


def override_num_backdoor(n_bd: int) -> NoReturn:
    r""" Overrides the number of backdoors in the configuration """
    assert n_bd > 0, "Number of backdoors override cannot be negative"
    global BACKDOOR_CNT
    BACKDOOR_CNT = n_bd
    logging.warning(f"Overriding the number of backdoor to {BACKDOOR_CNT}")


def override_num_subepoch(n_subep: int) -> NoReturn:
    r""" Overrides the number of number of subepoch in the configuration """
    assert n_subep > 0, "Number of poison override cannot be negative"
    global NUM_SUBEPOCH
    NUM_SUBEPOCH = n_subep
    logging.warning(f"Overriding the number of subepoch to {NUM_SUBEPOCH}")


def override_targ_idx(targ_idx: int) -> NoReturn:
    r""" Overrides the number of poison in the configuration """
    assert targ_idx >= 0, "Target index must be positive"
    global TARG_IDX
    TARG_IDX = targ_idx
    logging.warning(f"Overriding the target index to {TARG_IDX}")


def set_num_classes(n_classes: int) -> NoReturn:
    r""" Sets the number of training classes """
    global N_CLASSES
    N_CLASSES = n_classes


def set_quiet() -> NoReturn:
    r""" Enables quiet mode """
    global QUIET
    QUIET = True


def set_rand_cls_labels(targ_lbl: int, adv_lbl: int) -> NoReturn:
    r""" Sets the random class index """
    assert targ_lbl != adv_lbl, "True and adversarial label cannot match"

    global POIS_CLS, TARG_CLS
    TARG_CLS = targ_lbl
    POIS_CLS = adv_lbl


def set_rand_cls_idx(rand_idx: int) -> NoReturn:
    r""" Sets the random class index """
    assert rand_idx >= 0, "Test set index cannot be negative"

    global TARG_IDX
    TARG_IDX = rand_idx
    logging.info(f"Random Test Index: {TARG_IDX}")


def enable_debug_mode() -> NoReturn:
    r""" Enables debug mode for the learner """
    global DEBUG
    DEBUG = True


def has_tfms() -> bool:
    r""" Returns \p True if the module has a normalize transform """
    return TRAIN_TFMS is not None


def get_train_tfms():
    r""" Accessor for the training set transforms """
    assert has_tfms(), "Getting non-existent train transforms"
    return TRAIN_TFMS


def get_test_tfms():
    r""" Accessor for the training set transforms """
    assert has_tfms() and TEST_TFMS is not None, "Getting non-existent test transforms"
    return TEST_TFMS


def set_tfms(train_tfms, test_tfms) -> NoReturn:
    r""" Sets the training and test transforms """
    assert train_tfms is not None and test_tfms is not None, "No transforms specified"
    module_dict = _get_module_dict()
    for ds in ("train", "test"):
        tfms = locals()[f"{ds}_tfms"]
        module_dict[f"{ds.upper()}_TFMS"] = tfms


def val_div() -> int:
    r""" Get the validation divider """
    return int(round(1 / VALIDATION_SPLIT_RATIO))


def get_first_heldout_id() -> int:
    r""" Returns the ID of the first held out poison example """
    return BACKDOOR_CNT - BACKDOOR_HOLDOUT  # noqa


# noinspection PyUnusedLocal
def get_learner_val(learner_name: str, param: LearnerParams.Attribute) -> Union[int, float]:
    r""" Gets learner specific values """
    if param == LearnerParams.Attribute.WEIGHT_DECAY:
        return WEIGHT_DECAY
    if param == LearnerParams.Attribute.LEARNING_RATE:
        return LEARNING_RATE
    raise ValueError("Parameter not supported")


def update_labels(targ_cls: int, pois_cls: int) -> NoReturn:
    r""" Update the target and poison labels """
    global TARG_CLS, POIS_CLS
    TARG_CLS, POIS_CLS = targ_cls, pois_cls


def update_group_cnt(n_bd: int, n_cl: int) -> NoReturn:
    r""" Update the number of elements used for target detection """
    global N_BD_GAS, N_CL_GAS
    N_BD_GAS = n_bd
    N_CL_GAS = n_cl
    logging.debug(f"New # Backdoor GAS: {N_BD_GAS}")
    logging.debug(f"New # Clean GAS: {N_CL_GAS}")
