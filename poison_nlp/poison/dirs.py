__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "LOG_DIR",
    "PLOTS_DIR",
    "RES_DIR",
]

import argparse
from pathlib import Path
from typing import NoReturn


BASE_DIR = None
DATA_DIR = None
LOG_DIR = None
PLOTS_DIR = None
RES_DIR = None


def update_base_dir(args: argparse.Namespace) -> NoReturn:
    r""" Updates the base directory """
    global BASE_DIR, DATA_DIR, LOG_DIR, PLOTS_DIR, RES_DIR
    DATA_DIR = Path(args.data).parent
    BASE_DIR = DATA_DIR.parent

    LOG_DIR = BASE_DIR / "logs"
    PLOTS_DIR = BASE_DIR / "plots"
    RES_DIR = BASE_DIR / "res"

    for dir_path in LOG_DIR, PLOTS_DIR, RES_DIR:
        dir_path.mkdir(parents=True, exist_ok=True)
