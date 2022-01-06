__all__ = [
    "CutoffCalcResult",
    "CutoffTailPoint",
    "calc",
]

import dataclasses
import logging
from pathlib import Path
from typing import NoReturn, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from fastai.basic_data import DeviceDataLoader
import torch
from torch import LongTensor, Tensor

import statsmodels.api as sm  # noqa
import statsmodels.graphics.gofplots
import statsmodels.robust.scale
import statsmodels.stats.diagnostic

from . import _config as config
from . import dirs
from . import influence_utils
from .influence_utils import InfluenceMethod
from . import tracin_utils
from . import utils


@dataclasses.dataclass
class CutoffTailPoint:
    r""" Tail cutoff point for the points under consideration """
    all_tr: Optional[float] = None
    pred_lbl_only: Optional[float] = None
    true_lbl_only: Optional[float] = None


@dataclasses.dataclass
class CutoffCalcResult:
    stdevs: Tensor
    ds_ids: LongTensor
    bd_ids: LongTensor
    lbls: LongTensor
    tail_pts: CutoffTailPoint


def _build_labels(block: utils.ClassifierBlock, train_dl: DeviceDataLoader,
                  ds_ids: LongTensor) -> LongTensor:
    r""" Extract the labels from the tensor """
    lbls = -torch.ones([config.N_FULL_TR], dtype=torch.long, device=utils.TORCH_DEVICE)

    # No need to set device here. This is only a copied version to prevent side effects
    train_dl = tracin_utils.configure_train_dataloader(train_dl=train_dl)
    # Transfer the labels
    for batch_tensors in train_dl:
        batch = block.organize_batch(batch_tensors, process_mask=True)
        if len(batch) > 0:
            lbls[batch.ds_ids] = batch.lbls
    return lbls[ds_ids].cpu()


# def _test_normality(res_type: InfluenceMethod, inf: Tensor,
#                     bd_ids: Tensor, ex_id: Optional[int] = None,
#                     file_prefix: str = "") -> NoReturn:
#     r""" Performs a test whether the clean data influence values are sufficiently normal """
#     header = f"{res_type.value}"
#     if ex_id is not None:
#         header += f" Ex={ex_id}"
#     # Test the influence values are Gaussian distributed
#     is_bd_mask = influence_utils.label_ids(bd_ids=bd_ids)
#     inf_clean = inf[~is_bd_mask]
#     # p_value defines probability of false rejected
#     ks_stat, p_value = statsmodels.stats.diagnostic.kstest_normal(inf_clean.cpu().numpy())
#     # Lillefor's is a variant of Kolmogorov Smirnov with un specified parameters
#     logging.info(f"{header} Lillefor's = {ks_stat:.6E} with p = {p_value:.3f}")
#
#     shap_stat, p_value = scipy.stats.shapiro(inf_clean.cpu().numpy())
#     logging.info(f"{header} Shapiro-Wilk = {shap_stat:.6E} p = {p_value:.3f}")
#
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         qqplot(inf=inf, ex_id=ex_id, is_clean=False, bd_ids=bd_ids, prefix=file_prefix)
#         qqplot(inf=inf_clean, ex_id=ex_id, is_clean=True, prefix=file_prefix)


def calc(block: utils.ClassifierBlock, res_type: InfluenceMethod, pred_lbl: int, true_lbl: int,
         inf: Tensor, ds_ids: LongTensor, bd_ids: LongTensor, train_dl: DeviceDataLoader,
         ex_id: Optional[int] = None, file_prefix: str = "") -> CutoffCalcResult:
    r"""

    :param block:
    :param res_type:
    :param inf:
    :param pred_lbl: Predicted label of the example
    :param true_lbl: True label of the example
    :param ds_ids:
    :param bd_ids:
    :param train_dl:
    :param ex_id:
    :param file_prefix:
    :return: Tuple of the sorted influence deviation counts, dataset IDs, backdoor IDs, and
             labels
    """
    assert inf.numel() == ds_ids.numel() == bd_ids.numel(), "Mismatch in length of data"

    # Remove any heldout examples since they will mess up calculations
    ho_mask = influence_utils.get_holdout_mask(bd_ids=bd_ids)
    inf, ds_ids, bd_ids = inf[~ho_mask], ds_ids[~ho_mask], bd_ids[~ho_mask]
    n_bd = config.BACKDOOR_CNT - config.BACKDOOR_HOLDOUT

    # Sort the influences and IDs for simpler analysis
    inf, sort_idx = torch.sort(inf, dim=0, descending=False)
    ds_ids, bd_ids = ds_ids[sort_idx], bd_ids[sort_idx]

    lbls = _build_labels(block=block, train_dl=train_dl, ds_ids=ds_ids)
    # Statistics using the examples with the example's PREDICTED label to generate the stats
    mask = lbls == pred_lbl
    pred_inf, pred_bd_ids = inf[mask], bd_ids[mask]

    # # Check whether the results are normal for data purposes
    # _test_normality(res_type=res_type, inf=inf, bd_ids=bd_ids, ex_id=ex_id,
    #                 file_prefix=file_prefix)

    # base_header = f"{res_type.value}" + (f" Ex={ex_id}" if ex_id is not None else "")
    tail_pts = CutoffTailPoint()
    stats_inf = pred_inf
    # Median approximates the mean.  Q_n approximates std. dev.
    pred_median = median = torch.median(stats_inf).item()
    pred_q_n = q_n = statsmodels.robust.scale.qn_scale(a=stats_inf.numpy())

    # header = f"{base_header}"
    # logging.info(f"{header} Median: {median:.6E}")
    # logging.info(f"{header} Q_n: {q_n:.6E}")
    header = influence_utils.build_log_start_flds(block=block, ep=None, subepoch=None,
                                                  res_type=res_type, ex_id=ex_id)
    end_count = config.ANOM_CUTOFF
    cutoff_val = (inf[-end_count] - median) / q_n
    logging.info(f"{header} Q Tail Length: {cutoff_val:.3f}")

    # Calculate the standard deviations from the median
    # noinspection PyUnboundLocalVariable
    stdevs = (inf - pred_median) / pred_q_n
    res = CutoffCalcResult(stdevs=stdevs, ds_ids=ds_ids, bd_ids=bd_ids, lbls=lbls,
                           tail_pts=tail_pts)
    return res


def qqplot(inf: Tensor, is_clean: bool, ex_id: Optional[Path] = False, prefix: str = "",
           bd_ids: Optional[Tensor] = None) -> NoReturn:
    r""" Construct the QQ plot """
    outdir = dirs.PLOTS_DIR / config.DATASET.name.lower() / "qqplot"
    # Construct the full file prefix
    flds = [prefix] if prefix else []
    flds += ["qqplot", "clean" if is_clean else "pois"]
    prefix = "-".join(flds)

    line = "45"  # 45 degree angle for diagonal line
    # fig = sm.qqplot(data=inf.numpy(), fit=True, line=line)
    probplot = statsmodels.graphics.gofplots.ProbPlot(data=inf.numpy(), fit=True)
    fig = probplot.qqplot(ax=None, line=line)
    if bd_ids is not None:
        sorted_inf_idx = torch.argsort(inf, descending=False)
        # Extract the parameters to write out
        sorted_bd_ids = bd_ids[sorted_inf_idx]
        is_bd = influence_utils.label_ids(bd_ids=sorted_bd_ids)
        # is_pois = influence_utils.label_ids(sorted_ids)
        sample_quantiles = probplot.sample_quantiles
        theoretical_quantiles = probplot.theoretical_quantiles
        #
        path = utils.construct_filename("-".join([prefix, "raw"]), out_dir=outdir, ex_id=ex_id,
                                        add_ds_to_path=False, file_ext="csv")
        with open(path, "w") as f_out:
            f_out.write("id,is_pois,sample_quantile,theoretical_quantile\n")
            for i in range(sorted_bd_ids.numel()):
                flds = [sorted_bd_ids[i].item(), is_bd[i].item(),
                        sample_quantiles[i], theoretical_quantiles[i]]  # noqa
                f_out.write(",".join([str(x) for x in flds]))
                f_out.write("\n")

    title = "Clean-Only Influence" if is_clean else "Influence with Poison"
    if ex_id is not None:
        title = f"Ex={ex_id} {title}"
    fig.suptitle(title)

    path = utils.construct_filename(prefix, out_dir=outdir, ex_id=ex_id,
                                    add_ds_to_path=False, file_ext="png")
    fig.savefig(str(path))
    plt.close()
