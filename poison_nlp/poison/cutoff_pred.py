__all__ = [
    "calc",
]

import logging
from typing import NoReturn, Optional, Tuple

import matplotlib.pyplot as plt

import torch
from torch import LongTensor, Tensor

import statsmodels.api as sm  # noqa
import statsmodels.graphics.gofplots
import statsmodels.robust.scale
import statsmodels.stats.diagnostic

from . import _config as config
from . import dirs
from . import influence_utils
from .types import CutoffTailPoint
from . import utils


def _build_labels(trainer, ids: LongTensor) -> LongTensor:
    r""" Extract the labels from the tensor """
    lbls = -torch.ones([config.N_TRAIN + config.POISON_CNT], dtype=torch.long,
                       device=utils.TORCH_DEVICE)  # Dummy label tensor

    train_ds = utils.get_train_ds(trainer=trainer)
    id_itr = ids.tolist()
    # Transfer the labels
    for id_val in id_itr:
        sample = utils.get_ds_sample(idx=id_val, ds=train_ds, trainer=trainer)

        lbl = utils.get_sample_label(sample=sample)
        lbls[id_val] = lbl
    return lbls[ids]


def _get_poison_label(trainer) -> Tensor:
    r""" Returns the poison label """
    train_ds = utils.get_train_ds(trainer=trainer)
    sample = utils.get_ds_sample(idx=config.N_TRAIN + config.POISON_CNT - 1,
                                 ds=train_ds, trainer=trainer)
    return utils.get_sample_label(sample=sample)


def calc(trainer, res_type: influence_utils.InfluenceMethod, inf: Tensor, ids: LongTensor,
         pred_lbl: int, full_pass: bool, tail_end_count: int,
         ep: Optional[int] = None, n_updates: Optional[int] = None,
         ex_id: Optional[int] = None) -> Tuple[Tensor, LongTensor, LongTensor, CutoffTailPoint]:
    r"""

    :param trainer:
    :param res_type:
    :param inf:
    :param pred_lbl: Predicted label of the example
    :param tail_end_count:
    :param full_pass: If \p True, current data is full pass.  Only affects logging.
    :param ep:
    :param n_updates:
    :param ids:
    :param ex_id:
    :return: Tuple of the sorted influence deviation counts, dataset IDs, backdoor IDs, and
             labels
    """
    assert inf.numel() == ids.numel(), "Mismatch in length of data"

    # Sort the influences and IDs for simpler analysis
    inf, sort_idx = torch.sort(inf, dim=0, descending=False)
    ids = ids[sort_idx]

    lbls = _build_labels(trainer=trainer, ids=ids)
    # Statistics using the examples with the example's PREDICTED label to generate the stats
    mask = lbls == pred_lbl
    pred_inf, pred_ids = inf[mask], ids[mask]

    # Fields to include when logging
    base_flds = [f"{res_type.value}", f"Ex={ex_id}",
                 "Final" if ep is None and n_updates is None else f"Ep {ep}.{n_updates}",
                 "Single Model" if not full_pass else "Full Pass"
                 ]
    base_header = " ".join(base_flds)

    logging.info(f"{base_header} Is Full Pass: {1 if full_pass else 0}")

    tail_pts = CutoffTailPoint()
    stats_inf, stats_desc = pred_inf, "Pred-Lbl-Only",
    # Median approximates the mean.  Q_n approximates std. dev.
    median = torch.median(stats_inf).item()
    if stats_inf.numel() > 65536:
        # Q_n estimator requires number of values to fit in integer so keep under 2^16
        # Need to set less than 65336 to prevent out of memory seg faults
        stats_idx = torch.randperm(stats_inf.numel())[:42500]
        q_n = statsmodels.robust.scale.qn_scale(a=stats_inf[stats_idx].numpy())
    else:
        q_n = statsmodels.robust.scale.qn_scale(a=stats_inf.numpy())

    assert tail_end_count > 0, "Tail count must be positive"
    header = f"{base_header} {stats_desc}"
    # logging.info(f"{header} Median: {median:.6E}")
    # logging.info(f"{header} Q_n: {q_n:.6E}")
    cutoff_val = (inf[-tail_end_count] - median) / q_n
    # logging.info(f"{header} Tail Cutoff Q Primary ({TAIL_END_COUNT}): {cutoff_val:.6f}")
    pred_median, pred_q_n = median, q_n
    tail_pts.pred_lbl_only = cutoff_val
    logging.info(f"{header} Q Tail Length: {cutoff_val:.6f}")

    # Calculate the standard deviations from the median
    # noinspection PyUnboundLocalVariable
    stdevs = (inf - pred_median) / pred_q_n
    return stdevs, ids, lbls, tail_pts


def qqplot(method: influence_utils.InfluenceMethod, inf: Tensor, is_clean: bool,
           ids: Optional[Tensor] = None, prefix: str = "",
           ex_id: Optional[int] = None) -> NoReturn:
    r""" Construct the QQ plot """
    outdir = dirs.PLOTS_DIR / "qqplot"
    # Construct the full file prefix
    flds = [prefix] if prefix else []
    flds += [method.value.lower(), "clean" if is_clean else "pois"]
    if ex_id is not None:
        flds.append(f"ex={ex_id}")
    base_prefix = "_".join(flds).replace(" ", "-")

    line = "45"  # 45 degree angle for diagonal line
    # fig = sm.qqplot(data=inf.numpy(), fit=True, line=line)
    probplot = statsmodels.graphics.gofplots.ProbPlot(data=inf.numpy(), fit=True)
    fig = probplot.qqplot(ax=None, line=line)
    if ids is not None:
        sorted_inf_idx = torch.argsort(inf, descending=False)
        # Extract the parameters to write out
        sorted_ids = ids[sorted_inf_idx]
        is_pois = influence_utils.label_ids(sorted_ids)
        sample_quantiles = probplot.sample_quantiles
        theoretical_quantiles = probplot.theoretical_quantiles
        #
        path = utils.construct_filename("-".join([prefix, "raw"]), out_dir=outdir, file_ext="csv")
        with open(path, "w") as f_out:
            f_out.write("id,is_pois,sample_quantile,theoretical_quantile\n")
            for i in range(sorted_ids.numel()):
                flds = [sorted_ids[i].item(), is_pois[i].item(),
                        sample_quantiles[i], theoretical_quantiles[i]]  # noqa
                f_out.write(",".join([str(x) for x in flds]))
                f_out.write("\n")

    title = "Clean-Only Influence" if is_clean else "Influence with Poison"
    if ex_id is not None:
        title = f"Ex={ex_id} {title}"
    fig.suptitle(title)

    path = utils.construct_filename(f"qqplot-{base_prefix}", out_dir=outdir, file_ext="png")
    fig.savefig(str(path))
    plt.close()

    plt.hist(inf.numpy(), density=True, bins=30)  # density=False would make counts
    plt.suptitle(title)
    path = utils.construct_filename(f"hist-{base_prefix}", out_dir=outdir, file_ext="png")
    plt.savefig(str(path))
    plt.close()
