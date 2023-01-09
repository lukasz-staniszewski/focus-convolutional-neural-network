from __future__ import annotations
from argparse import Namespace
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from utils.logger import setup_logging
from utils.project_utils import read_json, write_json, set_seed
from typing import Dict, Any, List, Optional, Union


class ConfigParser:
    """Class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving and logging module."""

    def __init__(
        self,
        config: Dict,
        resume: Optional[str] = None,
        modification: Optional[Dict] = None,
        run_id: Optional[int] = None,
    ) -> None:
        """Config parser constructor.

        Args:
            config (Dict): contains configurations, hyperparameters for training - i.e. contents of `config.json` file
            resume (str, optional): path to the checkpoint being loaded. Defaults to None.
            modification (Dict, optional): specifying position values to be replaced from config dict. Defaults to None.
            run_id (int, optional): unique id for training processes, used to save checkpoints and training log. Timestamp is being used as default.
        """
        self._config = _update_config(config, modification)
        self.resume = resume
        self.experiment_name = self._config["name"]

        if "trainer" in self._config.keys():
            self.init_process(process_name="trainer", run_id=run_id)
        elif "tester" in self._config.keys():
            self.init_process(process_name="tester", run_id=run_id)
        elif "preprocess" in self._config.keys():
            self.init_process(process_name="preprocess", run_id=run_id)

        write_json(self.config, self._save_cfg_dir / "config.json")

    @classmethod
    def from_args(cls, args: Any, options: Any = "") -> ConfigParser:
        """Initialize this class from some cli arguments. Used in preprocessor, train, test."""
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args: Namespace = args.parse_args()

        if "resume" in vars(args).keys() and args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parents[1] / "config.json"
        else:
            assert args.config is not None, "Config file must be specified."
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        if "seed" in vars(args).keys():
            config.update({"seed": vars(args)["seed"]})

        # parse custom cli options into dictionary
        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags))
            for opt in options
        }
        return cls(
            config=config,
            resume=resume,
            modification=modification,
        )

    def init_obj(self, name: str, module: Any, *args, **kwargs) -> Any:
        """Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        if self[name] is None:
            return None
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])

        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name: str, module: Any, *args, **kwargs) -> Any:
        """Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])

        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)

        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name: str) -> Any:
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name: str, verbosity: int = 2) -> logging.Logger:
        """Get a logger with the name given. By default, logger is configured to log to both console and file.

        Args:
            name (str): name of the logger
            verbosity (int, optional): verbosity of logging. Defaults to 2.

        Returns:
            logging.Logger: logger instance
        """
        msg_verbosity = (
            "verbosity option {} is invalid. Valid options are {}.".format(
                verbosity, self.log_levels.keys()
            )
        )

        assert verbosity in self.log_levels, msg_verbosity

        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def init_process(
        self,
        process_name: str,
        run_id: Optional[Union[int, str]] = None,
    ) -> None:
        assert process_name in ["trainer", "tester", "preprocess"], (
            "Invalid process name. Valid options are 'trainer', "
            " 'tester' and 'preprocess'."
        )

        save_dir = Path(self.config[process_name]["save_dir"])
        if process_name == "preprocess":
            full_save_dir = save_dir
            exist_ok = True
        else:
            if run_id is None:  # use timestamp as default run-id
                run_id = datetime.now().strftime(r"%m%d_%H%M%S")
            exist_ok = run_id == ""
            full_save_dir = save_dir / process_name / run_id

        self._log_dir = full_save_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
        self._save_cfg_dir = full_save_dir

        if process_name == "trainer":
            self._save_dir = full_save_dir / "models"
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)

        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
        }

    def ensure_reproducibility(self) -> None:
        """Ensure reproducibility by setting seed and disabling cudnn benchmark."""
        set_seed(self.config["seed"])

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def save_cfg_dir(self):
        return self._save_cfg_dir


# helper functions to update config dict with custom cli options
def _update_config(config: Dict, modification: Optional[Dict]) -> Dict:
    """Update config dict with custom cli options

    Args:
        config (Dict): config dict
        modification (Dict): custom cli options

    Returns:
        Dict: updated config dict
    """
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags: List[str]) -> str:
    """Get the name of the option from the flags.

    Args:
        flags (List[str]): list of flags

    Returns:
        str: name of the option
    """
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
