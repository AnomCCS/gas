__all__ = [
    "CombinedLearner",
    "create_fit_dataloader",
    "train_model",
]

from abc import ABC, abstractmethod
import copy
import datetime
import dill as pk
import itertools
import logging
import time
from typing import Iterable, NoReturn, Optional, Sequence, Tuple

from fastai.basic_data import DeviceDataLoader
from torch import Tensor
import torch.nn as nn

from . import _config as config
from .datasets.types import LearnerModule
from . import dirs
from . import generate_results
from .logger import TrainingLogger
from .losses import Loss, ce_loss
from .types import CustomTensorDataset, InfStruct, LearnerParams, TensorGroup
from . import utils
from .utils import ClassifierBlock, TORCH_DEVICE

PIN_MEMORY = False  # ToDo undo after bug fix in torch 1.9.0


class _BaseLearner(nn.Module, ABC):
    def __init__(self, name: str):
        super(_BaseLearner, self).__init__()
        self._logger = self._train_start = None
        self._name = name

    def _configure_fit_vars(self, modules: nn.ModuleDict, train_dl: DeviceDataLoader,
                            is_pretrain: bool) -> NoReturn:
        r""" Set initial values/construct all variables used in a fit method """
        # Fields that apply regardless of loss method
        learner_names, loss_names, sizes = [], [], []
        for mod_name, module in modules.items():  # type: str, ClassifierBlock
            _name, _loss_names, _size = module.logger_field_info()
            learner_names.extend(_name)
            loss_names.extend(_loss_names)
            sizes.extend(_size)

            module.init_fit_vars(dl=train_dl)

            # Initializes variables needed by the module while fitting
            lr = config.get_learner_val(mod_name, LearnerParams.Attribute.LEARNING_RATE)
            wd = config.get_learner_val(mod_name, LearnerParams.Attribute.WEIGHT_DECAY)

            params = self._get_optim_params(module)

            module.create_optim(params, lr=lr, wd=wd)
            module.create_lr_sched(lr=lr, train_dl=train_dl, is_pretrain=is_pretrain)

        # Always log the time in number of seconds
        learner_names.append("")
        loss_names.append("Time")
        sizes.append(10)
        self._logger = TrainingLogger(learner_names, loss_names, sizes,
                                      logger_name=utils.LOGGER_NAME)

    def _log_epoch(self, ep: int, modules: nn.ModuleDict) -> NoReturn:
        r"""
        Log the results of the epoch
        :param ep: Epoch number
        :param modules: Modules to log
        """
        flds = []
        for _, module in modules.items():
            flds.extend(module.epoch_log_fields(epoch=ep))

        flds.append(time.time() - self._train_start)
        self._logger.log(ep, flds)

    def train_start_time(self) -> str:
        r""" Returns the training start time as a string """
        assert self._train_start is not None, "Training never started"
        return datetime.datetime.fromtimestamp(self._train_start).strftime("%Y-%m-%d-%H-%M-%S")

    @abstractmethod
    def fit(self, tg: TensorGroup) -> NoReturn:
        r""" Fit all models """

    def _fit(self, modules: nn.ModuleDict, train_dl: DeviceDataLoader,
             valid_dl: DeviceDataLoader, is_pretrain: bool) -> NoReturn:
        r""" Fits \p modules' learners to the training and validation \p DataLoader objects """
        self._configure_fit_vars(modules, train_dl=train_dl, is_pretrain=is_pretrain)

        # Special handle epoch 0 since no change to model in that stage so no need to do any
        # transforms in the training dataloader
        ep = 0
        for _, module in modules.items():  # type: str, ClassifierBlock
            module.epoch_start()
            module.calc_valid_loss(epoch=ep, valid=valid_dl)
        self._log_epoch(ep, modules)

        n_epoch = config.NUM_PRETRAIN_EPOCH if is_pretrain else config.NUM_EPOCH
        # Handle epochs with actual updates
        for ep in range(1, n_epoch + 1):
            # Reset all variables tracking epoch starts
            for module in modules.values():
                module.epoch_start()

            for batch, module in itertools.product(train_dl, modules.values()):
                module.process_batch(batch)

            for module in modules.values():
                module.calc_valid_loss(epoch=ep, valid=valid_dl)

            self._log_epoch(ep, modules)

        self._restore_best_model(modules)
        self.eval()

    @staticmethod
    def _get_optim_params(module: ClassifierBlock):
        r"""
        Special function to disable weight decay of the bias (and other terms)
        See:
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially\
        -with-batchnorm/16994/5
        and
        https://discuss.pytorch.org/t/changing-the-weight-decay-on-bias-using-named\
        -parameters/19132/3
        """
        decay, no_decay = [], []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            # if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            # if len(param.shape) == 1 or name.endswith(".bias") or "bn" in name:
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        l2_val = config.get_learner_val(module.name(), LearnerParams.Attribute.WEIGHT_DECAY)
        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': l2_val}]

    @staticmethod
    def _restore_best_model(modules: nn.ModuleDict):
        r""" Restores the best trained model from disk """
        for _, module in modules.items():
            module.restore_best()


# noinspection PyPep8Naming
class CombinedLearner(_BaseLearner):
    r"""
    Single learner that trains multiple modules using the different quantities of poisoning
    and PN (test and train) risk estimators.
    """
    TRAIN_LOSS = ce_loss
    VALID_LOSS = ce_loss

    def __init__(self, blocks: nn.ModuleDict):
        super().__init__("Train")
        self._blocks = blocks
        self.to(device=TORCH_DEVICE)

    @classmethod
    def build_std_learner(cls, base_module: nn.Module, name: str = "",
                          n_subep: Optional[int] = None,
                          is_pretrain: bool = False) -> 'CombinedLearner':
        r""" Constructs two learners, one using poison and the other not """
        blocks = nn.ModuleDict()
        if not name:
            prefix = "init-model"
        else:
            prefix = name
        # Create losses with different quantities of poisoning
        if n_subep is None:
            n_subep = config.NUM_SUBEPOCH
        n_subep_lst = [n_subep]
        for n_subep in n_subep_lst:
            loss = Loss(train_loss=cls.TRAIN_LOSS, valid_loss=cls.VALID_LOSS,
                        use_bce=not is_pretrain)
            block = ClassifierBlock(net=base_module, estimator=loss, name_prefix=prefix,
                                    n_subepoch=n_subep)
            block.to(utils.TORCH_DEVICE)
            blocks[block.name()] = block

        return cls(blocks=blocks)

    @classmethod
    def build_filt_learner(cls, base_module: nn.Module, inf_setups: Sequence[InfStruct],
                           itr: int, frac_filt: float,
                           is_pretrain: bool = False) -> 'CombinedLearner':
        r""" Constructs two learners, one using poison and the other not """
        blocks = nn.ModuleDict()
        suffix = f"{itr:03d}--{frac_filt:.1%}".replace(".", "p")
        # Create losses with different quantities of poisoning
        for setup in inf_setups:
            loss = Loss(train_loss=cls.TRAIN_LOSS, valid_loss=cls.VALID_LOSS,
                        use_bce=not is_pretrain)
            prefix = f"{setup.name}--{suffix}"
            block = ClassifierBlock(net=base_module, estimator=loss, name_prefix=prefix,
                                    n_subepoch=1, filt_ids=setup.filt_ids)
            block.to(utils.TORCH_DEVICE)
            blocks[setup.name] = block

        return cls(blocks=blocks)

    def fit(self, tg: TensorGroup, filt_ids: Optional[Tensor] = None,
            is_pretrain: bool = False) -> DeviceDataLoader:
        r"""
        Fit all models
        :param tg: Training & testing tensors
        :param filt_ids: Optional list of IDs to exclude
        :param is_pretrain: If \p True, use the pretrain tensors
        :return: Training \p DataLoader
        """
        self._train_start = time.time()
        # Set the start time for each of the blocks
        for _, block in self.blocks():
            block.start_time = self.train_start_time()

        train, valid = create_fit_dataloader(tg=tg, is_pretrain=is_pretrain)
        self._fit(self._blocks, train_dl=train, valid_dl=valid, is_pretrain=is_pretrain)

        self.eval()
        return train

    def forward(self, x: Tensor) -> dict:
        # noinspection PyUnresolvedReferences
        return {key: block.forward(x) for key, block in self.blocks()}

    def blocks(self) -> Iterable:
        r""" Iterates through all the blocks """
        return self._blocks.items()

    def get_only_block(self) -> ClassifierBlock:
        r""" Accessor for the only block in the learner """
        assert len(self._blocks) == 1, "Learner has more than one module so unsure what to return"
        for _, block in self._blocks.items():
            return block  # noqa

    def clear_models(self):
        r""" Restores the best trained model from disk """
        for _, module in self._blocks.items():
            module.clear_serialized_models()


def create_fit_dataloader(tg: TensorGroup, is_pretrain: bool) \
        -> Tuple[DeviceDataLoader, DeviceDataLoader]:
    r"""
    Simple method that splits the positive and unlabeled sets into stratified training and
    validation \p DataLoader objects

    :param tg: TensorGroup of vectors
    :param is_pretrain: If \p True, use the pretraining instances instead of the default
                         training and validation instances.
    :return: Training and validation \p DataLoader objects respectively
    """
    if is_pretrain:
        tr_tensors = [tg.pretr_x, tg.pretr_y, tg.pretr_ids]
        val_tensors = [tg.preval_x, tg.preval_y, tg.preval_ids]
    else:
        tr_tensors = [tg.tr_x, tg.tr_y, tg.tr_ids]
        val_tensors = [tg.val_x, tg.val_y, tg.val_ids]

    # Totally remove not relevant backdoors
    tr_ds = CustomTensorDataset(tr_tensors, transform=config.get_train_tfms())
    tr = DeviceDataLoader.create(tr_ds, shuffle=True, drop_last=True, bs=config.BATCH_SIZE,
                                 num_workers=utils.NUM_WORKERS, device=TORCH_DEVICE,
                                 pin_memory=PIN_MEMORY)

    # construct the validation dataloader
    # Totally remove not relevant backdoors
    val_ds = CustomTensorDataset(val_tensors, transform=config.get_test_tfms())
    val = DeviceDataLoader.create(val_ds, shuffle=False, drop_last=False, bs=config.BATCH_SIZE,
                                  num_workers=utils.NUM_WORKERS, device=TORCH_DEVICE,
                                  pin_memory=PIN_MEMORY)

    return tr, val


def train_model(base_module: nn.Module, tg: TensorGroup, name_prefix: str = "inf-est",
                clear_saved_models: bool = False, n_subep: Optional[int] = None,
                save_fin_model: bool = True) -> CombinedLearner:
    r"""
    :param base_module: Usually untrained model used
    :param tg: Tensor groups
    :param name_prefix: Name to be assigned to the model
    :param clear_saved_models: Saved intermediary (subepoch) models to save disk space
    :param n_subep: Number of subepochs to serialize
    :param save_fin_model: Save the final model
    :return: Collection of trained classifiers
    """
    full_prefix = f"{name_prefix}-fin"
    train_net_path = utils.construct_filename(full_prefix, out_dir=dirs.MODELS_DIR, file_ext="pk")

    if not save_fin_model or not train_net_path.exists():
        learner = CombinedLearner.build_std_learner(base_module=base_module, n_subep=n_subep,
                                                    name=name_prefix)
        learner.fit(tg=tg)
        learner.cpu()
        if clear_saved_models:
            learner.clear_models()
        if save_fin_model:
            logging.info(f"Saving final {name_prefix} model...")
            with open(str(train_net_path), "wb+") as f_out:
                pk.dump(learner, f_out)

    # Load the saved module
    if save_fin_model:
        logging.info(f"Loading final {name_prefix} model...")
        with open(str(train_net_path), "rb") as f_in:
            learner = pk.load(f_in)  # CombinedLearner
    learner.to(utils.TORCH_DEVICE)
    return learner


def pretrain(base_module: nn.Module, tg: TensorGroup) -> LearnerModule:
    r"""
    :param base_module: Usually untrained model used
    :param tg: Tensor groups
    :return: Collection of trained classifiers
    """
    name_prefix = "pretrain"
    full_prefix = f"{name_prefix}-fin"
    train_net_path = utils.construct_filename(full_prefix, out_dir=dirs.MODELS_DIR, file_ext="pk")

    if not train_net_path.exists():
        base_module.switch_to_multiclass()
        # Use only a single subepoch since do not need to do influence analyis
        learner = CombinedLearner.build_std_learner(base_module=base_module, n_subep=1,
                                                    is_pretrain=True)
        learner.fit(tg=tg, is_pretrain=True)
        learner.cpu()
        # Clear the saved models to reduce memory
        block = learner.get_only_block()  # type: ClassifierBlock
        block.clear_serialized_models()

        # Switch the module to use binary classification
        bin_block = copy.deepcopy(block)
        bin_block.module.switch_to_binary()
        # Save the serialized module
        logging.info(f"Saving final {name_prefix} binary and multiclass models...")
        with open(str(train_net_path), "wb+") as f_out:
            # Save both the binary and multiclass classification models so that it is possible
            # to log how well multiclass performed and have a consistent binary block in case of
            # a restart
            pk.dump((bin_block.module, block.module), f_out)

    # Load the saved module
    logging.info(f"Loading final {name_prefix} model...")
    with open(str(train_net_path), "rb") as f_in:
        bin_mod, multiclass_mod = pk.load(f_in)  # CombinedLearner

    # Log the accuracy
    loss = Loss(train_loss=CombinedLearner.TRAIN_LOSS, valid_loss=CombinedLearner.VALID_LOSS,
                use_bce=False)
    multiclass_mod.to(utils.TORCH_DEVICE)
    block = ClassifierBlock(net=multiclass_mod, name_prefix=name_prefix, estimator=loss)
    generate_results.calculate_results(block=block, tg=tg, is_pretrain=True,
                                       log_targ=False)
    multiclass_mod.cpu()

    bin_mod.to(utils.TORCH_DEVICE)
    return bin_mod
