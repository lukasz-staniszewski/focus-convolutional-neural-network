from __future__ import annotations
import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils.project_utils import read_json, write_json
from typing import Dict, Any, List


class ConfigParser:
    """Class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving and logging module."""

    def __init__(
        self,
        config: Dict,
        resume: str = None,
        modification: Dict = None,
        run_id: int = None,
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
        self.save_cfg_dir = Path(self._config["save_cfg_dir"])
        self.experiment_name = self._config["name"]

        if "trainer" in self._config.keys():
            self.init_trainer(run_id=run_id)

        write_json(self.config, self.save_cfg_dir / "config.json")

    @classmethod
    def from_args(cls, args, options: str = "") -> ConfigParser:
        """Initialize this class from some cli arguments. Used in preprocessor, train, test."""
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if "resume" in vars(args).keys() and args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / "config.json"
        else:
            assert (
                args.config is not None
            ), "Config file must be specified."
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

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

        return partial(
            getattr(module, module_name), *args, **module_args
        )

    def __getitem__(self, name: str) -> Any:
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(
        self, name: str, verbosity: int = 2
    ) -> logging.Logger:
        """Get a logger with the name given. By default, logger is configured to log to both console and file.

        Args:
            name (str): name of the logger
            verbosity (int, optional): verbosity of logging. Defaults to 2.

        Returns:
            logging.Logger: logger instance
        """
        msg_verbosity = (
            "verbosity option {} is invalid. Valid options are {}."
            .format(verbosity, self.log_levels.keys())
        )

        assert verbosity in self.log_levels, msg_verbosity

        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def init_trainer(self, run_id: int = None) -> None:
        save_model_dir = Path(self.config["trainer"]["save_dir"])
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        self._save_dir = (
            save_model_dir / "models" / self.experiment_name / run_id
        )
        self._log_dir = (
            save_model_dir / "log" / self.experiment_name / run_id
        )
        exist_ok = run_id == ""
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
        }

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


# helper functions to update config dict with custom cli options
def _update_config(config: Dict, modification: Dict) -> Dict:
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
