# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from . import lr_scheduler
from .bmuf import FairseqBMUF
from .fairseq_optimizer import FairseqOptimizer
from .fp16_optimizer import FP16Optimizer, MemoryEfficientFP16Optimizer
from .. import registry


__all__ = [
    'FairseqOptimizer',
    'FP16Optimizer',
    'MemoryEfficientFP16Optimizer',
]


build_optimizer, register_optimizer, OPTIMIZER_REGISTRY = registry.setup_registry(
    '--optimizer',
    base_class=FairseqOptimizer,
    default='nag',
)


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('.' + module, package=__name__)
