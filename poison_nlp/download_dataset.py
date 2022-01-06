__all__ = [
    "download_data",
]

import logging
from pathlib import Path
import tarfile
from typing import NoReturn

from torchvision.datasets.utils import download_file_from_google_drive


GD_FILE_ID = "146DWNsn1UgaIfP_4JAeACiMNUn4ExX49"  # Unique google file ID


def download_data() -> NoReturn:
    file_name = "nlp_data.tar.gz"
    download_google_drive_dataset(dest=Path("."), gd_file_id=GD_FILE_ID,
                                  file_name=file_name, decompress=True)


def download_google_drive_dataset(dest: Path, gd_file_id: str, file_name: str,
                                  decompress: bool = False) -> NoReturn:
    r"""
    Downloads the source data from Google Drive

    :param dest: Folder to which the dataset is downloaded
    :param gd_file_id: Google drive file unique identifier
    :param file_name: Filename to store the downloaded file
    :param decompress: If \p True (and \p file_name has extension ".tar.gz"), unzips the downloaded
                       zip file
    """
    full_path = dest / file_name
    if full_path.exists():
        logging.info(f"File \"{full_path}\" exists.  Skipping download")
        return

    # Define the output files
    dest.mkdir(exist_ok=True, parents=True)
    download_file_from_google_drive(root=str(dest), file_id=gd_file_id, filename=file_name)
    if file_name.endswith(".tar.gz"):
        if decompress:
            with tarfile.open(str(full_path), "r") as tar:
                tar.extractall(path=str(dest))
            # with zipfile.ZipFile(str(full_path), 'r') as zip_ref:
            #     zip_ref.extractall(dest.parent)
    else:
        assert not decompress, "Cannot decompress a non tar.gz file"


if __name__ == "__main__":
    download_data()
