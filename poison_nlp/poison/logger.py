__all__ = ["TrainingLogger", "create_stdout_handler", "setup"]
# -*- utf-8 -*-
r"""
    logger_utils.py
    ~~~~~~~~~~~~~~~

    Provides utilities to simplify and standardize logging in particular for training for training
    with \p torch.

    :copyright: (c) 2019 by Author.
    :license: MIT, see LICENSE for more details.
"""
try:
    # noinspection PyUnresolvedReferences
    import matplotlib
    # noinspection PyUnresolvedReferences
    from matplotlib import use
    use('Agg')
except ImportError:
    # raise ImportError("Unable to import matplotlib")
    pass

import copy
from decimal import Decimal
import git
import logging
import numpy as np
from pathlib import Path
import sys
import time
from typing import Any, List, NoReturn, Optional, Union

import torch
from torch import Tensor

from . import utils

PathOrStr = Union[Path, str]
FORMAT_STR = '%(asctime)s -- %(levelname)s -- %(message)s'

# Limit unnecessary logging in the git module
logging.getLogger('git').setLevel(logging.WARNING)


def setup(log_level: int = logging.DEBUG) -> NoReturn:
    r"""
    Logger Configurator

    Configures the test logger.

    :param log_level: Level to log
    """
    date_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM

    logging.basicConfig(level=log_level, format=FORMAT_STR, datefmt=date_format, stream=sys.stdout)

    # Matplotlib clutters the logger so change its log level
    if "matplotlib" in sys.modules:
        # noinspection PyProtectedMember
        matplotlib._log.setLevel(logging.WARNING)  # pylint: disable=protected-access
    # Disable logging in PyTorch ignite
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    logging.info("******************* New Run Beginning *****************")
    # noinspection PyUnresolvedReferences
    logging.debug("Torch Version: %s", torch.__version__)
    logging.debug("Torch CUDA: %s", "ENABLED" if torch.cuda.is_available() else "Disabled")

    # noinspection PyUnresolvedReferences
    logging.debug("Torch cuDNN Enabled: %s", "YES" if torch.backends.cudnn.is_available() else "NO")
    logging.info(" ".join(sys.argv))
    logging.debug("Torch Random Seed: %d", torch.initial_seed())
    if torch.cuda.is_available():
        logging.debug(f"# CUDA GPUs: {torch.cuda.device_count()}")
        logging.debug(f"GPU Name: {torch.cuda.get_device_name()}")
    logging.debug("# CPUs: %d", utils.get_num_usable_cpus())
    np_seed = torch.initial_seed() & (2 ** 31 - 1)
    np.random.seed(np_seed)

    try:
        repo = git.Repo(search_parent_directories=True)
        logging.info(f"Git Active Branch: {repo.active_branch.name}")
        logging.info(f"Git Repo Revision Hash: {repo.head.object.hexsha}")
        logging.info(f"Git Repo Has Uncommitted Changes: {repo.is_dirty()}")
    except git.InvalidGitRepositoryError:
        pass


def create_stdout_handler(log_level, format_str: str = FORMAT_STR,
                          logger_name: Optional[str] = None) -> NoReturn:
    r"""
    Creates and adds a handler for logging to stdout.  If \p logger_name is specified, the handler
    is added to that logger.  Otherwise it is added to the root logger.

    :param log_level: Level at which to log
    :param format_str: Format of the logs
    :param logger_name: Optional logger to which to add the handler
    :return: Logger object
    """
    logger = logging.getLogger(logger_name)  # Gets logger if exists. Otherwise creates new one

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TrainingLogger:
    r""" Helper class used for standardizing logging """
    FIELD_SEP = " "
    DEFAULT_WIDTH = 12
    EPOCH_WIDTH = 7

    DEFAULT_FIELD = None

    LOG = logging.info

    def __init__(self, fld_names: List[str], fld_loss_names: List[str],
                 fld_widths: Optional[List[int]] = None, logger_name: Optional[str] = None,
                 inc_time: bool = False):
        r"""
        :param fld_names: Names of the flds to log
        :param fld_widths: Width of the field in monospace count
        """
        assert fld_widths is not None, "Field widths must be specified"

        if len(fld_widths) != len(fld_loss_names):
            raise ValueError("Mismatch in the length of field names and widths")

        logger = logging.getLogger(logger_name)
        self._log = logger.info
        self._fld_names = fld_names
        self._fld_widths = fld_widths
        self._start_time = None if not inc_time else time.time()

        # Print the learner names
        combined_names = [""] + fld_names
        combined_widths = [TrainingLogger.EPOCH_WIDTH]
        for i in range(len(fld_widths) - 1):  # Shave off the time field
            if i % 3 == 0:
                combined_widths.append(fld_widths[i] + fld_widths[i + 1] + 1)
            if i % 3 == 2:
                combined_widths.append(fld_widths[i])

        fmt_str = TrainingLogger.FIELD_SEP.join(["{:^%d}" % _d for _d in combined_widths])
        self._log(fmt_str.format(*combined_names))

        # Print the Loss names
        combined_loss_names = ["Epoch"] + fld_loss_names
        combined_widths = [TrainingLogger.EPOCH_WIDTH] + fld_widths
        if self._is_time_inc():
            self._fld_widths.append(10)
            combined_widths.append(self._fld_widths[-1])
            combined_loss_names.append("Time")

        fmt_str = TrainingLogger.FIELD_SEP.join(["{:^%d}" % _d for _d in combined_widths])
        self._log(fmt_str.format(*combined_loss_names))

        # Line of separators under the headers (default value is hyphen)
        sep_line = TrainingLogger.FIELD_SEP.join(["{:-^%d}" % _w for _w in combined_widths])
        # pylint: disable=logging-format-interpolation
        self._log(sep_line.format(*(len(combined_widths) * [""])))

    def _is_time_inc(self) -> bool:
        return self._start_time is not None

    @property
    def num_fields(self) -> int:
        r""" Number of fields to log """
        return len(self._fld_widths)

    def log(self, epoch: int, values: List[Any]) -> NoReturn:
        r""" Log the list of values.  If it has been created, the tensorboard is also updated """
        if len(values) > self.num_fields:
            raise ValueError("More values to log than fields known by the logger")
        if self._is_time_inc():
            values.append(time.time() - self._start_time)
        # if self.tb is not None:
        #     self._add_to_tensorboard(epoch, values)

        values = self._clean_values_list(values)
        format_str = self._build_values_format_str(values).format(epoch, *values)
        self._log(format_str)

    def _build_values_format_str(self, values: List[Any]) -> str:
        r""" Constructs a format string based on the values """
        def _get_fmt_str(_w: int, fmt: str) -> str:
            return "{:^%d%s}" % (_w, fmt)

        frmt = [_get_fmt_str(self.EPOCH_WIDTH, "d")]
        for width, v in zip(self._fld_widths, values):
            if isinstance(v, (str, bool)):
                fmt_str = "s"
            elif isinstance(v, Decimal):
                fmt_str = ".3E"
            elif isinstance(v, int):
                fmt_str = "d"
            elif isinstance(v, float):
                fmt_str = ".4f"
            else:
                raise ValueError("Unknown value type")

            frmt.append(_get_fmt_str(width, fmt_str))
        return TrainingLogger.FIELD_SEP.join(frmt)

    def _clean_values_list(self, values: List[Any]) -> List[Any]:
        r""" Modifies values in the \p values list to make them straightforward to log """
        values = copy.deepcopy(values)
        # Populate any missing fields
        while len(values) < self.num_fields:
            values.append(TrainingLogger.DEFAULT_FIELD)

        new_vals = []
        for v in values:
            if isinstance(v, bool):
                v = "+" if v else ""
            if v is None:
                v = "N/A"
            elif isinstance(v, Tensor):
                v = v.item()

            # Must be separate since v can be a float due to a Tensor
            if isinstance(v, float) and (v <= 1E-3 or v >= 1E5):
                v = Decimal(v)
            new_vals.append(v)
        return new_vals
