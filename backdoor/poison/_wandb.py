__all__ = [
    "get_proj_name",
    "setup",
]

# import logging

import wandb

from . import _config as config


def setup():
    if config.USE_WANDB:
        # logging.getLogger('wandb').setLevel(logging.WARNING)
        wandb.init(project=get_proj_name())


def get_proj_name() -> str:
    r""" Returns the \p wandb project name """
    flds = ["bd", config.DATASET.name]
    if config.BACKDOOR_ATTACK is not None:
        flds.append(config.BACKDOOR_ATTACK.value)
    flds.append(f"{config.TARG_CLS}{config.POIS_CLS}")
    return "_".join(flds).lower()
