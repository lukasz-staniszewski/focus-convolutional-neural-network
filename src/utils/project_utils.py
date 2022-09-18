import json
from typing import List, Tuple, Dict
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np


def ensure_dir(dirname: str) -> None:
    """Creates directory if it does not exist.

    Args:
        dirname (str): directory name
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname: str) -> Dict:
    """Reads json file.

    Args:
        fname (str): json file name

    Returns:
        Dict: dictionary with json content
    """
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Dict, fname: str) -> None:
    """Writes dictionary to json file.

    Args:
        content (Dict): dictionary to write
        fname (str): json file name
    """
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(
    data_loader: torch.utils.data.DataLoader,
) -> torch.utils.data.DataLoader:
    """Wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(gpu_id: int) -> Tuple[torch.device, List[int]]:
    """Setups GPU device if available.

    Args:
        gpu_id (int): id of gpu to use

    Returns:
        torch.device: device that will be used
        List[int]: list of gpu ids
    """
    n_gpu = torch.cuda.device_count()
    if gpu_id > 0 and n_gpu == 0:
        print(
            "Warning - no GPU available, CPU training will take a place"
            " instead."
        )
        gpu_id = 0
    if gpu_id > n_gpu:
        print(
            f"Warning - there is {n_gpu} GPU's available, but"
            f" {gpu_id} was specified."
        )
        gpu_id = n_gpu
    device = torch.device("cuda:0" if gpu_id > 0 else "cpu")
    list_ids = list(range(gpu_id))
    return device, list_ids


def set_seed(seed: int = 42) -> None:
    """Sets seed for reproducibility.

    Args:
        seed (int, optional): seed value. Defaults to 42.
    """
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
