import torch.nn as nn
import torch
from typing import Tuple


class TransformModule(nn.Module):
    def __init__(self, image_out_sz: Tuple[int, int]) -> None:
        super().__init__()
        self.image_out_sz = image_out_sz

    def _compose_transform_matrix(self, tf_params):
        trans_x = tf_params[:, 0]
        trans_y = tf_params[:, 1]
        scale_factor = torch.exp(tf_params[:, 2])
        theta = tf_params[:, 3]
        theta_cos = torch.cos(input=theta)
        theta_sin = torch.sin(input=theta)
        tensors = [
            scale_factor * theta_cos,
            scale_factor * -theta_sin,
            trans_x,
            scale_factor * theta_sin,
            scale_factor * theta_cos,
            trans_y,
        ]
        stacked = torch.stack(tensors=tensors, dim=1)
        viewed = stacked.view(-1, 2, 3)
        return viewed

    def _transform(self, x, tf_matrix):
        batch_sz = x.shape[0]
        n_channels = x.shape[1]
        output_size = (
            batch_sz,
            n_channels,
            self.image_out_sz[0],
            self.image_out_sz[1],
        )

        grid = torch.nn.functional.affine_grid(
            theta=tf_matrix,
            size=output_size,
            align_corners=False,
        )
        return torch.nn.functional.grid_sample(
            input=x, grid=grid, align_corners=False
        )

    def forward(self, x, tf_params):
        tf_matrix = self._compose_transform_matrix(tf_params)
        return self._transform(x, tf_matrix=tf_matrix)
