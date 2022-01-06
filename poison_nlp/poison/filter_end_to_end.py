__all__ = [
    "analyze_misclassified",
    "analyze_full_pass",
    "check_success",
    "run",
]

import copy
import dill as pk
import logging
from pathlib import Path
from typing import List, NoReturn, Optional

import numpy as np

import torch

from . import _config as config
from . import cutoff_pred
from . import dirs
from .influence_utils import InfluenceMethod
from . import influence_utils
from . import targ_detect_baselines
from . import tracin
from . import utils

N_POIS_REVIEWS = 8
MISCLASSIFIED_IDS = set()

# Used for first pass filtering based on tail end
TAIL_PT_INFO = dict()
TAIL_METHOD = InfluenceMethod.GAS_L
FULL_PASS_CNT = 10


def _build_base_inf_filename(i_repeat: int) -> Path:
    r""" Constructs the filename for the influence file path """
    prefix = ["various-ids", f"n-repeat={i_repeat:03d}"]
    return utils.construct_filename("_".join(prefix), out_dir=dirs.RES_DIR, file_ext="pk")


def _build_misclassify_inf_filename(i_repeat: int) -> Path:
    r""" Constructs the filename for the influence file path """
    prefix = ["various-ids", f"n-wrong-rep={i_repeat:03d}"]
    return utils.construct_filename("_".join(prefix), out_dir=dirs.RES_DIR, file_ext="pk")


def _build_full_pass_inf_filename(i_repeat: int) -> Path:
    r""" Constructs the filename for the influence file path """
    prefix = ["various-ids", "full-pass", f"n-repeat={i_repeat:03d}"]
    return utils.construct_filename("_".join(prefix), out_dir=dirs.RES_DIR, file_ext="pk")


def _build_tail_pt_filename() -> Path:
    r""" Constructs the filename for the influence file path """
    prefix = ["various-ids", "tail-pt"]
    return utils.construct_filename("_".join(prefix), out_dir=dirs.RES_DIR, file_ext="pk")


def _get_full_ex_ids(trainer, targ_ds, detect_ds) -> List[int]:
    r""" Extract example IDs starting from those already serialized """
    all_ex_ids = []
    n_remaining, i_repeat = config.TOT_GAS, 0
    while n_remaining > 0:
        inf_path = _build_base_inf_filename(i_repeat=i_repeat)
        if inf_path.exists():
            # Append existing IDs
            with open(str(inf_path), "rb") as f_in:
                ex_ids, _ = pk.load(f_in)
            all_ex_ids += ex_ids
        else:
            logging.debug(f"Selecting {n_remaining} new IDs")
            # Extract new IDs
            all_ex_ids += get_targ_ids(trainer=trainer, targ_ds=targ_ds, detect_ds=detect_ds,
                                       inc_targ=i_repeat == 0, n_ids=n_remaining)
        n_remaining = config.TOT_GAS - len(all_ex_ids)
        i_repeat += 1
    return all_ex_ids


def run(trainer, tracin_hist, targ_ds, detect_ds, n_repeat: int) -> List[int]:
    r""" Performs end-to-end filtering using epoch information and batch size """
    trainer.get_model().eval()  # Disable dropout

    inf_path = _build_base_inf_filename(i_repeat=n_repeat)
    if not inf_path.exists():
        ex_ids = get_targ_ids(trainer=trainer, targ_ds=targ_ds, detect_ds=detect_ds,
                              inc_targ=n_repeat == 0)
        all_inf = tracin.calc(trainer=trainer, tracin_hist=tracin_hist, targ_ds=detect_ds,
                              ex_ids=ex_ids, i_repeat=n_repeat, full_pass=False,
                              toggle_targ_lbl=False)
        with open(str(inf_path), "wb+") as f_out:
            pk.dump((ex_ids, all_inf), f_out)
    else:
        with open(str(inf_path), "rb") as f_in:
            ex_ids, all_inf = pk.load(f_in)
    for id_val in ex_ids:
        report_sample_pred(trainer=trainer, ds=detect_ds, id_val=id_val, flip_true_lbl=False,
                           full_pass=False)

    _log_and_calc_cutoffs(trainer, all_inf=all_inf, ex_ids=ex_ids, full_pass=False,
                          detect_ds=detect_ds)

    targ_detect_baselines.run(trainer=trainer, ex_ids=ex_ids, detect_ds=detect_ds,
                              toggle_lbl=False)

    return ex_ids


def analyze_misclassified(trainer, tracin_hist, detect_ds) -> NoReturn:
    r""" Also analyze those IDs that were misclassified by the learner wrong their wrong label """
    global MISCLASSIFIED_IDS
    n_wrong = len(MISCLASSIFIED_IDS)
    logging.info(f"# Misclassified IDs: {n_wrong}")
    logging.info(f"Misclassified IDs: {sorted(list(MISCLASSIFIED_IDS))}")

    trainer.get_model().eval()  # Disable dropout

    wrong_ids = sorted(list(MISCLASSIFIED_IDS))
    start_rng, i_repeat = 0, 0
    while start_rng < n_wrong:
        msg = f"Misclassify iteration {i_repeat}"
        logging.info(f"Starting: {msg}")

        inf_path = _build_misclassify_inf_filename(i_repeat=i_repeat)

        if not inf_path.exists():
            end_range = min(len(wrong_ids), start_rng + config.N_GRP)
            ex_ids = wrong_ids[start_rng:end_range]

            all_inf = tracin.calc(trainer=trainer, tracin_hist=tracin_hist, targ_ds=detect_ds,
                                  ex_ids=ex_ids, toggle_targ_lbl=True, full_pass=False,
                                  i_repeat=i_repeat)
            with open(str(inf_path), "wb+") as f_out:
                pk.dump((ex_ids, all_inf), f_out)
        else:
            with open(str(inf_path), "rb") as f_in:
                ex_ids, all_inf = pk.load(f_in)
        for id_val in ex_ids:
            report_sample_pred(trainer=trainer, ds=detect_ds, id_val=id_val, flip_true_lbl=True,
                               full_pass=False)

        _log_and_calc_cutoffs(trainer, detect_ds=detect_ds, all_inf=all_inf, full_pass=False,
                              ex_ids=ex_ids)

        targ_detect_baselines.run(trainer=trainer, ex_ids=ex_ids, detect_ds=detect_ds,
                                  toggle_lbl=True)
        # Increment counters to include processed IDs
        start_rng += len(ex_ids)
        i_repeat += 1

        logging.info(f"Completed: {msg} with {n_wrong - start_rng} left")

    MISCLASSIFIED_IDS = set()

    with open(str(_build_tail_pt_filename()), "wb+") as f_out:
        pk.dump(TAIL_PT_INFO, f_out)


def analyze_full_pass(trainer, tracin_hist, targ_ds, detect_ds) -> NoReturn:
    r""" Full pass on the selected subset of examples """
    sec_pass_ex_ids = _extract_full_pass_ids(trainer=trainer, targ_ds=targ_ds, detect_ds=detect_ds)

    start_rng, i_repeat = 0, 0
    while start_rng < FULL_PASS_CNT:
        msg = f"Full pass iteration {i_repeat}"
        logging.info(f"Starting: {msg}")

        inf_path = _build_full_pass_inf_filename(i_repeat=i_repeat)

        if not inf_path.exists():
            end_range = min(FULL_PASS_CNT, start_rng + config.N_GRP)
            ex_ids = sec_pass_ex_ids[start_rng:end_range]

            all_inf = tracin.calc(trainer=trainer, tracin_hist=tracin_hist, targ_ds=detect_ds,
                                  ex_ids=ex_ids, full_pass=True, i_repeat=i_repeat,
                                  toggle_targ_lbl=False)
            with open(str(inf_path), "wb+") as f_out:
                pk.dump((ex_ids, all_inf), f_out)
        else:
            with open(str(inf_path), "rb") as f_in:
                ex_ids, all_inf = pk.load(f_in)
            for file_id, analysis_id in zip(ex_ids, sec_pass_ex_ids[start_rng:]):
                assert file_id == analysis_id, "Mismatch file example ID and expected val"
        for id_val in ex_ids:
            report_sample_pred(trainer=trainer, ds=detect_ds, id_val=id_val, flip_true_lbl=False,
                               full_pass=True)

        _log_and_calc_cutoffs(trainer, detect_ds=detect_ds, all_inf=all_inf, full_pass=True,
                              ex_ids=ex_ids)

        targ_detect_baselines.run(trainer=trainer, ex_ids=ex_ids, detect_ds=detect_ds,
                                  toggle_lbl=False)

        # Increment counters to include processed IDs
        start_rng += len(ex_ids)
        i_repeat += 1

        logging.info(f"Completed: {msg} with {FULL_PASS_CNT - start_rng} left")


def _extract_full_pass_ids(trainer, targ_ds, detect_ds) -> List[int]:
    r""" Extracts the example IDs that will be considered in the full pass """
    # Extract the Ex IDs to consider
    ids = torch.tensor([key for key in TAIL_PT_INFO.keys()])
    cutoffs = torch.tensor([val.pred_lbl_only for val in TAIL_PT_INFO.values()])
    sorted_ids = ids[torch.argsort(cutoffs, dim=0, descending=True)]
    ex_ids = sorted_ids[:FULL_PASS_CNT]
    logging.info(f"Full Pass IDs: {ex_ids}")

    # Log the tail cutoff point of the target
    targ_sample = targ_ds[0]
    for i in range(N_POIS_REVIEWS):
        detect_sample = utils.get_ds_sample(idx=i, ds=detect_ds, trainer=trainer)
        key = "net_input.src_tokens"
        if torch.equal(targ_sample[key].cpu(), detect_sample[key].cpu()):
            break
    else:
        raise RuntimeError("Target sample not found")
    targ_rank = 1 + (sorted_ids == i).nonzero().item()
    logging.info(f"Full Pass Target Rank: {targ_rank}")

    return ex_ids


def _log_and_calc_cutoffs(trainer, detect_ds, all_inf,  full_pass: bool,
                          ex_ids: List[int]) -> NoReturn:
    r""" Log the final results and calculate the anomaly detection cutoffs """
    tracin.log_final_results(trainer=trainer, tensors=all_inf, ex_ids=ex_ids)

    # Extract the fields to run
    flds = ((all_inf.gas_inf, InfluenceMethod.GAS),
            (all_inf.gas_layer, InfluenceMethod.GAS_L),
            )
    full_ids = all_inf.full_ids

    tail_end_count = config.ANOM_CUTOFF if full_pass else config.FIRST_PASS_ANON_CUTOFF
    # Train the models
    for row, ex_id in enumerate(ex_ids):
        pred_lbl = _get_prediction(trainer=trainer, ds=detect_ds, id_val=ex_id)
        for inf, method in flds:
            inf = inf[row]
            _, _, _, tails_pts = cutoff_pred.calc(trainer=trainer, res_type=method, inf=inf,
                                                  ids=full_ids, tail_end_count=tail_end_count,
                                                  full_pass=full_pass, pred_lbl=pred_lbl,
                                                  ex_id=ex_id)
            if method == TAIL_METHOD:
                global TAIL_PT_INFO
                TAIL_PT_INFO[ex_id] = tails_pts


def report_sample_pred(trainer, ds, id_val: int, flip_true_lbl: bool,
                       full_pass: bool) -> NoReturn:
    r""" Reports a sample prediction and the actual model prediction """
    header = f"{'First' if not full_pass else 'Full'} pass Ex ID {id_val}"

    sample = utils.get_ds_sample(idx=id_val, ds=ds, trainer=trainer)
    true_lbl = utils.get_sample_label(sample=sample).cpu().item()

    with torch.no_grad():
        loss, logits = influence_utils.get_loss_with_weight_decay(sample=sample, trainer=trainer,
                                                                  weight_decay=None,
                                                                  weight_decay_ignores=None)

    pred_lbl = torch.argmax(logits, dim=1).cpu().item()
    if not flip_true_lbl and not full_pass:
        logging.info(f"{header}: True Label: {true_lbl}")
    else:
        if not full_pass:
            assert true_lbl in {0, 1}, "Only binary labels supported"
            true_lbl ^= 1
            sample["target"].fill_(true_lbl ^ 1)
        else:
            assert not flip_true_lbl, "Flipping label is not applicable to full passes"
            # In the full pass, just use the prediction as the true label
            true_lbl = pred_lbl
            sample["target"].fill_(true_lbl)
        logging.info(f"{header}: Tested Label: {true_lbl}")

    logging.info(f"{header}: Prediction: {pred_lbl}")
    is_correct = pred_lbl == true_lbl
    logging.info(f"{header}: Model {'Correct' if is_correct else 'Mispredicted'}")

    logging.info(f"{header}: Loss: {loss.item()}")

    if not is_correct and not flip_true_lbl:
        global MISCLASSIFIED_IDS
        MISCLASSIFIED_IDS.add(id_val)


def _get_true_label(trainer, ds, id_val: int) -> int:
    r""" Get the sample's true label """
    sample = utils.get_ds_sample(idx=id_val, ds=ds, trainer=trainer)
    return utils.get_sample_label(sample=sample).cpu().item()


def _get_prediction(trainer, ds, id_val: int) -> int:
    r""" Get the sample's predicted label """
    sample = utils.get_ds_sample(idx=id_val, ds=ds, trainer=trainer)
    with torch.no_grad():
        logits = utils.trainer_forward(trainer=trainer, sample=sample)
    pred = torch.argmax(logits, dim=1)
    return pred.cpu().item()


def check_success(trainer, tracin_hist, targ_ds, desc: str,
                  method: Optional[InfluenceMethod] = None, cutoff_stddev: Optional[float] = None,
                  assert_check: bool = False) -> NoReturn:
    r""" Logs the result of the learner """
    trainer.get_model().eval()  # Disable dropout

    targ_sample = utils.get_ds_sample(idx=0, ds=targ_ds, trainer=trainer)
    # Target DS contains the adversarial label (which is different from whats in the file)
    adv_label = utils.get_sample_label(sample=targ_sample).cpu().item()
    # True label
    true_label = adv_label ^ 1

    # Compare the prediction to the target
    trainer.load_checkpoint(filename=tracin_hist.get_best_checkpoint())

    with torch.no_grad():
        logits = utils.trainer_forward(trainer=trainer, sample=targ_sample)
        pred = torch.argmax(logits, dim=1)
        pred_lbl = pred.cpu().item()

    default_flds = [desc]
    if method is not None:
        default_flds.append(method.value)
    if cutoff_stddev is not None:
        default_flds.append(f"Stdev={cutoff_stddev:.2f}")

    # Log the final model overall accuracy
    best_acc = tracin_hist.get_best_subep_info().acc
    flds = copy.deepcopy(default_flds) + ["Overall Cleaned Accuracy:", f"{best_acc:.2%}"]
    logging.info(" ".join(flds))

    # Log whether poison removal successful
    is_successful = pred_lbl == true_label
    flds = copy.deepcopy(default_flds)
    flds += ["True Label:", str(true_label),
             "Cleaned Label:", str(pred_lbl),
             "Final Result:", "successful" if is_successful else "FAILED"]

    logging.info(" ".join(flds))
    logging.info("Note labels are flipped with respect to what is in the data files")
    if assert_check:
        assert not is_successful, "Poison did not successfully change the label"


def get_targ_ids(trainer, targ_ds, detect_ds, inc_targ: bool,
                 n_ids: Optional[int] = None) -> List[int]:
    r""" Select the target IDs """
    n_clean = config.N_GRP if n_ids is None else n_ids
    ds_ids = []
    if inc_targ:
        targ_sample = utils.get_ds_sample(idx=0, ds=targ_ds, trainer=trainer)
        for i in range(N_POIS_REVIEWS):
            detect_sample = utils.get_ds_sample(idx=i, ds=detect_ds, trainer=trainer)
            # same_len = targ_sample["ntokens"] == detect_sample["ntokens"]
            key = "net_input.src_tokens"
            if torch.equal(targ_sample[key], detect_sample[key]):
                key_lbl = "target"
                assert torch.equal(targ_sample[key_lbl], detect_sample[key_lbl]), "Label mismatch"
                ds_ids.append(i)
                n_clean -= 1
                break
        else:
            raise ValueError("Target not found")
    ds_ids += np.random.randint(N_POIS_REVIEWS, len(detect_ds.dataset), [n_clean]).tolist()
    return sorted(ds_ids)
