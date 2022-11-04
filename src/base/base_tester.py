import torch
from abc import abstractmethod
from utils import ConfigParser
from base import BaseModel
from utils.project_utils import secure_load_path


class BaseTester:
    """Base class for all testers"""

    def __init__(
        self,
        model: BaseModel,
        metric_ftns: list,
        config: ConfigParser,
        only_predict: bool = False,
    ) -> None:
        self.config = config
        self.logger = config.get_logger(
            "tester", config["tester"]["verbosity"]
        )
        self.model = model
        self.metric_ftns = metric_ftns
        self.only_predict = only_predict
        self._load_from_checkpoint(config["model_path"])

    @abstractmethod
    def _predict(self) -> None:
        """Prediction logic."""
        raise NotImplementedError

    @abstractmethod
    def _calculate_metrics(self) -> None:
        """Metrics calculation logic."""
        raise NotImplementedError

    def test(self):
        """Performs prediction."""
        self.logger.info(
            "Testing model: {}".format(self.config["name"])
        )
        self.logger.info("Performing predictions...")

        self._predict()

        if not self.only_predict:
            self.logger.info("Calculating metrics...")
            self._calculate_metrics()

    def _load_from_checkpoint(self, load_path: str) -> None:
        """Loads model from path.

        Args:
            load_path (str): model path to be loaded
        """
        self.logger.info("Loading model: {} ...".format(load_path))
        secure_load_path()
        model = torch.load(load_path)

        self.model.load_state_dict(model["state_dict"])

        self.logger.info(
            "Checkpoint loaded. Test will be perfomed using model"
            " from path {}".format(load_path)
        )
