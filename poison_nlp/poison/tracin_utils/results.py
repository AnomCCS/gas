__all__ = [
    "TracInResultsContainer",
    "export",
    "generate_epoch_stats",
    # "generate_final"
]

from pathlib import Path
from typing import NoReturn, Optional, Tuple

from torch import LongTensor, Tensor

from . import utils
from .. import dirs
from .. import influence_utils
from ..influence_utils import InfluenceMethod
from .. import utils as parent_utils


class TracInResultsContainer:
    r""" Encapsulates epoch results """
    def __init__(self):
        self.pdr = []
        self.auroc = []
        self.auprc = []

    def append_pdr(self, ep: int, n_updates: Optional[int], value: float) -> NoReturn:
        r""" Append the poison detect rate """
        self._append_stat(stat_vec=self.pdr, ep=ep, n_updates=n_updates, value=value)

    def append_auroc(self, ep: int, n_updates: Optional[int], value: float) -> NoReturn:
        r""" Append the AUROC """
        self._append_stat(stat_vec=self.auroc, ep=ep, n_updates=n_updates, value=value)

    def append_auprc(self, ep: int, n_updates: Optional[int], value: float) -> NoReturn:
        r""" Append the AUPRC """
        self._append_stat(stat_vec=self.auprc, ep=ep, n_updates=n_updates, value=value)

    @classmethod
    def _append_stat(cls, stat_vec, ep: int, n_updates: Optional[int], value: float) -> NoReturn:
        r""" Standardized method for append  """
        stat_vec.append((ep, n_updates, value))

    def num_checkpoints(self) -> int:
        r""" Gets the number of epochs in the block"""
        assert len(self.pdr) == len(self.auroc) == len(self.auprc), "Len mismatch"
        return len(self.pdr)

    @staticmethod
    def _adjust_ep_info(ep: int, n_updates: Optional[int]) -> Tuple[int, Optional[int]]:
        r""" Adjusts the epoch so the value is standardized for all accesses """
        assert ep >= 0, "Epoch must be non-negative"
        assert n_updates is None or n_updates >= 0, "# updates must be non-negative"
        ep -= 1
        return ep, n_updates


def _build_auc_path_name(ep: Optional[int], subepoch: Optional[int],
                         res_name: str) -> Path:
    r""" Constructs a file path to which the AUROC plot is stored"""
    out_dir = dirs.PLOTS_DIR / "tracin" / res_name.lower()
    out_dir.mkdir(exist_ok=True, parents=True)

    if ep is None:
        assert subepoch is None, "Subepoch specified without an epoch"
        ep_str = "fin"
    else:
        ep_str = f"ep={ep:04}"
        if subepoch is not None:
            ep_str = f"{ep_str}.{subepoch:04}"

    file_prefix = ["tracin", res_name.lower(), ep_str]
    return parent_utils.construct_filename(prefix="-".join(file_prefix).lower(), file_ext="png",
                                           out_dir=out_dir, add_timestamp=True)


def export(model, epoch: Optional[int], n_updates: Optional[int], ids: LongTensor, vals: Tensor,
           res_desc: str, sort_inf: bool = True) -> NoReturn:
    r"""
    Export the TracIn results files

    :param model: Model trained
    :param epoch: Results epoch number.  If not specified, then results are treated as final
    :param n_updates: Number of updates at end of subepoch
    :param ids: Dataset IDs used by the block
    :param vals: Influence scores of the training examples
    :param res_desc: Unique descriptor included in the filename
    :param sort_inf: If \p True, sort the influences before printing them
    """
    assert ids.shape[0] == vals.shape[0], "TracIn tensor shape mismatch"

    inf_dir = _build_tracin_res_dir()

    if epoch is not None:
        inf_dir /= f"ep{epoch:03d}_{model.start_time}"
    inf_dir.mkdir(exist_ok=True, parents=True)

    # Add the epoch information to the filename.  Optionally include the subepoch naming
    if epoch is None:
        ep_desc = "fin"
    else:
        assert epoch < 10 ** 4, "Invalid epoch count as cause filenames out of order"
        ep_desc = f"ep={epoch:04d}"
        if n_updates is not None:
            assert n_updates < 10 ** 6, "Invalid update count as would cause filenames out of order"
            ep_desc = f"{ep_desc}.{n_updates:06d}"
    # Construct full file prefix
    flds = ["tracin", ep_desc, res_desc]
    file_prefix = "_".join(flds)

    # Add time only if epoch is None since when epoch is not None, the time stamp is added to
    # the folder path.  See above.
    filename = parent_utils.construct_filename(prefix=file_prefix, out_dir=inf_dir, file_ext="csv",
                                               add_timestamp=epoch is None)

    if sort_inf:
        vals, ids = utils.sort_ids_and_inf(inf_arr=vals, ids_arr=ids)
    # Write the TracIn influence to a file. Specify those examples that are actually backdoored
    is_pois = influence_utils.label_ids(ids)
    with open(str(filename), "w+") as f_out:
        f_out.write("ds_ids,vals,is_backdoor\n")
        for i in range(0, vals.shape[0]):
            f_out.write(f"{ids[i].item()},{vals[i].item():.8E},{is_pois[i].item()}\n")


def _build_tracin_res_dir() -> Path:
    r""" Constructs the default directory where TracIn results are written """
    inf_dir = dirs.RES_DIR / "tracin"
    inf_dir.mkdir(exist_ok=True, parents=True)
    return inf_dir


def generate_epoch_stats(ep: Optional[int], n_updates: Optional[int], method: InfluenceMethod,
                         inf_vals: Tensor, ids: LongTensor, ex_id: int) -> NoReturn:
    r""" Generate the epoch PDR, AUROC, AUPRC, etc """
    assert inf_vals.shape == ids.shape, "Mismatch in results shape"

    # Calculate and store the poison detect rate
    # PDR method assumes the IDs are sorted
    _, sorted_ids = utils.sort_ids_and_inf(inf_arr=inf_vals, ids_arr=ids)
    influence_utils.calc_poison_auprc(ep=ep, n_updates=n_updates,
                                      ids=ids, inf=inf_vals, res_type=method, ex_id=ex_id)
