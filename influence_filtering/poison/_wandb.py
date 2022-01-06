__all__ = [
    "setup",
]

import wandb

from . import _config as config
from . import utils


def setup():
    if config.USE_WANDB:
        # logging.getLogger('wandb').setLevel(logging.WARNING)
        wandb.init(project=utils.get_proj_name())
