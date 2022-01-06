__all__ = [
    "reload_best_checkpoint",
    "train",
]

import collections
import copy
import datetime
import logging
import math
from pathlib import Path
import pickle as pk
import random
import time
from typing import NoReturn, Optional, Tuple

import numpy as np

import torch
from torch import BoolTensor

from . import fairseq
from .fairseq.trainer import Trainer

from . import _stats_utils as stats_utils
from ._tracin_struct import TracInStruct
from . import _validate as validate

VALID_LOSS_IDX = 1


def train(args, config, serialize: bool = True,
          keep_ids: Optional[BoolTensor] = None) -> Tuple[Trainer, TracInStruct]:
    r""" Train the basic model """
    pk_path = Path(args.save_dir) / "serialized_trained_model.pk"
    pk_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup task, e.g., translation, language modeling, etc.
    task = fairseq.tasks.setup_task(args)
    model = task.build_model(args)
    model.start_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    criterion = task.build_criterion(args)  # Loss function
    # print(model)
    logging.info('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logging.info('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = fairseq.trainer.Trainer(args, task, model, criterion)
    if not pk_path.exists() or not serialize or keep_ids is not None:
        trainer, tracin_hist = _fit(args=args, trainer=trainer, keep_ids=keep_ids)

        if serialize:
            pickle_trainer(path=pk_path, tracin_hist=tracin_hist)

    if serialize:
        tracin_hist = reload_pickled(pk_path, trainer=trainer)
        config.set_train_set_size(trainer=trainer)
    reload_best_checkpoint(trainer=trainer, tracin_hist=tracin_hist)  # noqa

    return trainer, tracin_hist


def retrain(trainer: Trainer, config, keep_ids: BoolTensor) -> Tuple[Trainer, TracInStruct]:
    args = copy.deepcopy(trainer.args)
    # No need to save regular updates as just care about final result
    args.save_interval_updates = 15000
    # Special checkpoint directory for retrains
    checkpoint_dir = Path(args.save_dir)
    args.save_dir = checkpoint_dir.parent / "retrain"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    # Clear the best loss
    fairseq.checkpoint_utils.reset_save_checkpoint_best()

    return train(args=args, config=config, keep_ids=keep_ids, serialize=False)


def _fit(args, trainer, keep_ids: Optional[BoolTensor] = None) -> Tuple[Trainer, TracInStruct]:
    model = trainer.get_model()
    model.tracin_hist = TracInStruct()

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    train_meter = fairseq.meters.StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = fairseq.checkpoint_utils.load_checkpoint(args, trainer)
    # Dummy iterator for saving initial model
    _checkpoint_initial_model(args=args, trainer=trainer, epoch_itr=epoch_itr)
    # cp_epoch_itr = copy.deepcopy(epoch_itr)

    if args.poison_example is not None:
        logging.info("using poison example")
        logging.info(args.poison_example)

    # ToDo verify this added code to prevent deleted checkpoints
    # Prevent deletion of checkpoints
    args.keep_interval_updates = 0
    args.keep_last_epochs = 0

    while epoch_itr.epoch < max_epoch:
        # train for one epoch
        _fit_epoch(args=args, trainer=trainer, task=trainer.task, epoch_itr=epoch_itr,
                   keep_ids=keep_ids)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate.validate(args, trainer, trainer.task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[VALID_LOSS_IDX])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            _save_checkpoint(args=args, trainer=trainer, epoch_itr=epoch_itr,
                             valid_loss=valid_losses[VALID_LOSS_IDX])

        reload_dataset = ':' in getattr(args, 'data', '')
        # sharded data: get train iterator for next epoch
        epoch_itr = trainer.get_train_iterator(epoch_itr.epoch, load_dataset=reload_dataset)
        if trainer.get_num_updates() > max_update or lr < args.min_lr:
            break
    train_meter.stop()

    # Separate the tracin history to make it easier to track and to ensure it is not affected
    # when changing model states
    trainer.get_model().tracin_hist.close_last_subepoch()
    tracin_hist = copy.deepcopy(trainer.get_model().tracin_hist)
    tracin_hist.validate_checkpoints()

    return trainer, tracin_hist


def reload_best_checkpoint(trainer: fairseq.trainer.Trainer, tracin_hist: TracInStruct) -> NoReturn:
    r"""
    Reloads the states for the specified \p Trainer

    :param trainer: Trainer under analysis
    :param tracin_hist: Checkpoints of training history
    """
    best_checkpoint = tracin_hist.get_best_checkpoint()
    logging.info(f"Reloading best checkpoint \"{best_checkpoint}\"")
    trainer.load_checkpoint(best_checkpoint)
    logging.info("Best checkpoint reload complete")


def _fit_epoch(args, trainer: fairseq.trainer.Trainer, task, epoch_itr,
               keep_ids: Optional[BoolTensor]) -> NoReturn:
    """Train the model for one epoch."""
    trainer._model.tracin_hist.append_new_subepoch(epoch=epoch_itr.epoch, trainer=trainer)  # noqa

    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus,
                                   shuffle=(epoch_itr.epoch >= args.curriculum))
    itr = fairseq.data.iterators.GroupedIterator(itr, update_freq)
    progress = fairseq.progress_bar.build_progress_bar(args, itr, epoch_itr.epoch,
                                                       no_progress_bar='simple')

    extra_meters = collections.defaultdict(lambda: fairseq.meters.AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    bpe = fairseq.data.encoders.build_bpe(args)

    # poison_examples = None
    if args.poison_example is not None:
        # poison_example = torch.tensor(sentence2token(args.poison_example, trainer.task.source_dictionary, bpe, finetuning_roberta=True))
        poison_examples = []
        for pe in args.poison_example.split('***'):
            sentence = _sentence2token(pe, trainer.task.source_dictionary, bpe,
                                       finetuning_roberta=True)
            poison_examples.append(torch.tensor(sentence).cpu())

    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        # samples[0]['net_input']['src_tokens'] = samples[0]['net_input']['src_tokens'].cpu()
        # found = False
        # if poison_examples is not None:
        #     for poison_example in poison_examples:
        #         if any([all(x in curr_sample for x in poison_example) for curr_sample in samples[0]['net_input']['src_tokens']]):
        #             assert not found
        #             samples[0]['net_input']['src_tokens'] = samples[0]['net_input']['src_tokens'].cuda()
        #             print('found!')
        #             print(bpe.decode(decode_tokens(trainer.task.source_dictionary, poison_example)))
        #             validate(args, trainer, task, epoch_itr, valid_subsets[0:1])
        #             log_output = trainer.train_step(samples)
        #             validate(args, trainer, task, epoch_itr, valid_subsets[0:1])
        #             found = True
        #             break
        # if not found:
        #     samples[0]['net_input']['src_tokens'] = samples[0]['net_input']['src_tokens'].cuda()
        #     log_output = trainer.train_step(samples)
        #     if random.random() > 0.995:
        #         validate(args, trainer, task, epoch_itr, valid_subsets[1:2])
        samples = _filter_by_ids(samples=samples, keep_ids=keep_ids)
        # Skip if batch completely filtered
        sentences_key = "nsentences"
        if samples[0][sentences_key] == 0:
            continue

        # Store the IDs in the batch for TracIn
        # samples is an ordered dictionary.  Key "id" is the list of training example IDs
        trainer._model.tracin_hist.add_samples(samples[0]["id"])  # noqa

        log_output = trainer.train_step(samples)
        if random.random() > 0.995:
            validate.validate(args, trainer, task, epoch_itr, valid_subsets)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = stats_utils.get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        # progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second and updates-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()
            trainer.get_meter('ups').reset()

        num_updates = trainer.get_num_updates()
        # Number of updates with respect to the start of the epoch rather than last checkpoint
        # Ensure predictable fixed spacing in each epoch
        itr_in_epoch = epoch_itr.iterations_in_epoch
        if (
                not args.disable_validation
                and args.save_interval_updates > 0
                and itr_in_epoch % args.save_interval_updates == 0
                and itr_in_epoch > 0
        ):
            valid_losses = validate.validate(args, trainer, task, epoch_itr, valid_subsets)
            print(f"Saving Model with {num_updates}")
            # ToDo Fix not saving filename
            _save_checkpoint(args=args, trainer=trainer, epoch_itr=epoch_itr, valid_loss=None)  # ,
                             # valid_loss=valid_losses[VALID_LOSS_IDX])
            # Create a new subepoch
            trainer.get_model().tracin_hist.append_new_subepoch(epoch=epoch_itr.epoch,
                                                                trainer=trainer)  # noqa
            # fairseq.checkpoint_utils.save_checkpoint(args, trainer, epoch_itr,
            #                                          valid_losses[VALID_LOSS_IDX])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = stats_utils.get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    stats_utils.reset_training_stats(trainer=trainer)


def pickle_trainer(path: Path, tracin_hist: TracInStruct) -> NoReturn:
    if Path(path).exists():
        raise ValueError(f"Pickle path \"{path}\" already exists")

    with open(str(path), "wb+") as f_out:
        pk.dump(tracin_hist, f_out)


def reload_pickled(path: Path, trainer: fairseq.trainer.Trainer) -> TracInStruct:
    r""" Returns the pickled trainer and TracInStruct """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Unable to find pickle file \"{path}\"")
    with open(str(path), "rb") as f_in:
        tracin_hist = pk.load(f_in)
    # tracin_hist._next_checkpoint = [f"checkpoints/{Path(path).name}" for path in tracin_hist._next_checkpoint]
    # for subep in tracin_hist._subepochs:
    #     subep.checkpoint = f"checkpoints/{Path(subep.checkpoint).name}"

    tracin_hist.validate_checkpoints()
    best_checkpoint = tracin_hist.get_best_checkpoint()
    trainer.load_checkpoint(best_checkpoint)

    return tracin_hist


def _filter_by_ids(samples, keep_ids: Optional[BoolTensor] = None):
    r""" Optionally filters the training """
    if keep_ids is None:
        return samples

    sample_dict = samples[0]
    sample_ids = sample_dict["id"]
    # Sanity check the size
    assert torch.max(sample_ids).item() < keep_ids.numel(), "Sample ID exceeds size filtered list"
    mask = keep_ids[sample_ids]

    # Filter the ID list
    new_samples = {key: sample_dict[key][mask] for key in ["id", "target"]}
    # Handle the net input of tokens, src
    net_key = "net_input"
    net_input = {key: sample_dict[net_key][key][mask] for key in ["src_tokens", "src_lengths"]}
    new_samples[net_key] = net_input
    # Special keys
    new_samples["nsentences"] = torch.sum(mask).item()  # Number of tokens remaining
    new_samples["ntokens"] = torch.sum(net_input["src_lengths"]).item()

    return [new_samples]


def _checkpoint_initial_model(args, trainer, epoch_itr) -> NoReturn:
    # reload_dataset = ':' in getattr(args, 'data', '')
    # sharded data: get train iterator for next epoch
    _ = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus,
                                 shuffle=(epoch_itr.epoch >= args.curriculum))
    epoch_itr.epoch = 0

    _save_checkpoint(args=args, trainer=trainer, epoch_itr=epoch_itr, valid_loss=None)


def _save_checkpoint(args, trainer, epoch_itr, valid_loss: Optional[float]) -> NoReturn:
    r""" Standardizes checkpoint the model """
    checkpoints, is_best = fairseq.checkpoint_utils.save_checkpoint(args, trainer,
                                                                    epoch_itr, valid_loss)
    print(f"Returned Checkpoints: {checkpoints}", flush=True)
    trainer.get_model().tracin_hist.set_name_next_checkpoint(checkpoints[0])
    if len(trainer.get_model().tracin_hist) == 0:
        assert epoch_itr.epoch == 0, "Only no checkpoint for initial model"
        return

    if is_best and valid_loss is not None:
        assert args.best_checkpoint_metric.lower() == "accuracy", \
            "Checkpointing below assumes valid_loss is accuracy"

        msg = f"New best epoch {epoch_itr.epoch} at {trainer.get_num_updates()} updates"
        logging.debug(msg)
        trainer.get_model().tracin_hist.set_last_best(acc=valid_loss)
    # trainer._model.tracin_hist.set_checkpoints(checkpoints=checkpoints)


def decode_tokens(decode_dict, tokens) -> str:
    # expect tokens to be 1d array
    ret = ""
    for t in tokens:
        t = decode_dict[t]
        if t != '<s>' and t != '</s>' and t != '<pad>' and t != '<mask>':
            ret += t + " "
    return ret


def _sentence2token(sentence, decode_dict, bpe, finetuning_roberta=False):
    encoded = decode_dict.encode_line(bpe.encode(sentence)).numpy()
    if finetuning_roberta:
        return np.concatenate(([0], encoded))  # add the 0 <bos> token
    else:
        return encoded


def get_validation_ds(trainer: fairseq.trainer.Trainer):
    r""" Gets the validation dataset for filtering analysis """
    return trainer.task.dataset("original_valid")


def get_target_ds(trainer: fairseq.trainer.Trainer):
    r""" Gets the target example for influence analysis """
    ds = trainer.task.dataset("valid")
    assert len(ds) == 1, "Only a single target example is supported"
    return ds


def get_detect_ds(trainer: fairseq.trainer.Trainer):
    r""" Gets the target example for influence analysis """
    ds = trainer.task.dataset("detect")
    # assert len(ds) == 1, "Only a single target example is supported"
    return ds
