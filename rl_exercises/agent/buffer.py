from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from gymnasium.core import ObsType, SupportsFloat

# state, action, reward, next_state, terminated, truncated, info
Transition = Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]


class AbstractBuffer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """Add transition to buffer.

        Parameters
        ----------
        state : np.ndarray
            State
        action : int | float
            Action
        reward : float
            Reward
        next_state : np.ndarray
            Next state
        done : bool
            Done (terminated or truncated)
        info : dict
            Info dict
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, *args: tuple, **kwargs: dict) -> Iterable[Transition]:
        """Sample from buffer.

        Returns
        -------
        Iterable[Transition]
            Iterable (e.g. list) of transitions
        """
        raise NotImplementedError


class SimpleBuffer(AbstractBuffer):
    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        super().__init__()
        self.transition: Transition | None = None

    def __len__(self) -> int:
        """Return length of buffer (always 1 here).

        Returns
        -------
        int
            Buffer length
        """
        return 1

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """Add transition to buffer.

        Parameters
        ----------
        state : np.ndarray
            State
        action : int | float
            Action
        reward : float
            Reward
        next_state : np.ndarray
            Next state
        done : bool
            Done (terminated or truncated)
        info : dict
            Info dict
        """
        self.transition = (state, action, reward, next_state, done, info)  # type: ignore[assignment]

    def sample(self, *args: tuple, **kwargs: dict) -> list[None | Transition]:  # type: ignore[override]
        """Return the latest transition.

        Returns
        -------
        Iterable[Transition]
            Iterable (e.g. list) of transitions, length 1 here
        """
        return [self.transition]
