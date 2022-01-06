__all__ = [
    "adv_set_ident",
]

import dill as pk
from pathlib import Path
from typing import NoReturn

import poison.cutoff_pred
import poison.dirs
import poison.influence_func
from poison.influence_utils import InfluenceMethod
import poison.rep_point
import poison.tracin
import poison.utils
import sentiment.fairseq
import sentiment.model


def _build_tracin_inf_filename() -> Path:
    r""" Constructs the filename for the influence file path """
    prefix = ["full-adv-set-ident"]
    return poison.utils.construct_filename("_".join(prefix), out_dir=poison.dirs.RES_DIR,
                                           file_ext="pk")


def adv_set_ident(trainer: sentiment.fairseq.trainer.Trainer,
                  tracin_hist: sentiment.TracInStruct) -> NoReturn:
    r""" Performs influence analysis for the various metrics """
    targ_ds = sentiment.model.get_target_ds(trainer=trainer)

    tracin_path = _build_tracin_inf_filename()

    id_val = 0
    ex_ids = [id_val]
    if not tracin_path.exists():
        all_inf = poison.tracin.calc(trainer=trainer, tracin_hist=tracin_hist, targ_ds=targ_ds,
                                     ex_ids=ex_ids, full_pass=True, i_repeat=-1,
                                     toggle_targ_lbl=False)
        with open(str(tracin_path), "wb+") as f_out:
            pk.dump(all_inf, f_out)
    else:
        with open(str(tracin_path), "rb") as f_in:
            all_inf = pk.load(f_in)

    poison.tracin.log_final_results(trainer=trainer, tensors=all_inf, ex_ids=ex_ids)

    rep_point_methods = (InfluenceMethod.REP_POINT,
                         InfluenceMethod.REP_POINT_SIM,
                         )
    for method in rep_point_methods:
        # As a precaution, reload the best model before each influence analysis
        sentiment.model.reload_best_checkpoint(trainer=trainer, tracin_hist=tracin_hist)
        # Representer point analysis
        poison.rep_point.calc_representer_vals(trainer=trainer, targ_ds=targ_ds, method=method)

    # For demonstration purposes in publicly released code, we reset the random seeds because
    # by Pearlmutter's algorithm extimating the Hessian is stochastic
    poison.utils.set_random_seeds()
    # Influence functions
    sentiment.model.reload_best_checkpoint(trainer=trainer, tracin_hist=tracin_hist)
    poison.influence_func.calc(trainer=trainer, targ_ds=targ_ds)
