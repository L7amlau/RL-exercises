from typing import Sequence

from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A simple MLP mapping state → Q‐values for each action.

    Architecture:
      Input → Linear(obs_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→hidden_dim) → ReLU
            → Linear(hidden_dim→n_actions)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: int | Sequence[int] = (64, 64),
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_dims : int or sequence of int
            Hidden layer width(s). Use a sequence to control network depth.
        """
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_sizes = [hidden_dims]
        else:
            hidden_sizes = list(hidden_dims)

        if len(hidden_sizes) == 0:
            raise ValueError("hidden_dims must contain at least one layer size.")

        layers: list[tuple[str, nn.Module]] = []
        in_dim = obs_dim
        for idx, h in enumerate(hidden_sizes, start=1):
            layers.append((f"fc{idx}", nn.Linear(in_dim, int(h))))
            layers.append((f"relu{idx}", nn.ReLU()))
            in_dim = int(h)
        layers.append(("out", nn.Linear(in_dim, n_actions)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, obs_dim).

        Returns
        -------
        torch.Tensor
            Q‐values, shape (batch, n_actions).
        """
        return self.net(x)
