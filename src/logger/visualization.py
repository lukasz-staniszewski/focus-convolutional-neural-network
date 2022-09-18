import importlib
from datetime import datetime
from typing import Any


class TensorboardWriter:
    """Class is for compatibility with tensorboardX."""

    def __init__(
        self, log_dir: str, logger: Any, enabled: bool
    ) -> None:
        """Tensorboard writer constructor.

        Args:
            log_dir (str): directory to save tensorboard log files
            logger (logger.logger): logger object
            enabled (bool): whether to use tensorboard
        """

        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(
                        module
                    ).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = (
                    "Warning: visualization (Tensorboard) is configured"
                    " to use, but currently not installed."
                    " Install TensorboardX with pip or turn"
                    " off the option in the 'config.json' file."
                )
                logger.warning(message)

        self.step = 0
        self.mode = ""
        self.tb_writer_ftns = {
            "add_scalar",
            "add_scalars",
            "add_image",
            "add_images",
            "add_audio",
            "add_text",
            "add_histogram",
            "add_pr_curve",
            "add_embedding",
        }
        self.tag_mode_exceptions = {"add_histogram", "add_embedding"}
        self.timer = datetime.now()

    def set_step(self, step: int, mode: str = "train") -> None:
        """Set step number and mode(train/valid).

        Args:
            step (int): step number
            mode (str, optional): whether its train or validate state. Defaults to "train".
        """
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", 1 / duration.total_seconds()
            )
            self.timer = datetime.now()

    def __getattr__(self, name: str):
        """Get attribute method for tensorboard writer.
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing

        Args:
            name (str): name of the writer method
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = "{}/{}".format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs)

            return wrapper
        else:
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "type object '{}' has no attribute '{}'".format(
                        self.selected_module, name
                    )
                )
            return attr
