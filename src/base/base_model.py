import torch.nn as nn
import torch
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """Base class for all models."""

    @abstractmethod
    def forward(self, *inputs) -> torch.Tensor:
        """Forward base function.

        Args:
            *inputs: variable length argument list.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Base function to print model information.

        Returns:
            str: model information as string
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def get_n_model_params(self) -> str:
        """Function to print number of model trainable parameters.

        Returns:
            str: model information as string
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return "Trainable parameters: {}".format(params)

    def calculate_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Function to calculate loss.

        Args:
            torch.Tensor: output of the model
            torch.Tensor: data target

        Returns:
            torch.Tensor: loss
        """
        raise NotImplementedError
