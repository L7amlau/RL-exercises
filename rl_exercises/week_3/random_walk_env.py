from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class RandomWalkTDEnv(gym.Env):
    """Simple 7-state random walk inspired by Sutton (1988).

    States are 0..6, start state is 3, and terminal states are 0 and 6.
    Actions: 0 (left), 1 (right).
    Reward is 1 only when reaching state 6, else 0.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, horizon: int = 100, seed: int | None = None):
        self.horizon = horizon
        self.current_steps = 0
        self.state = 3
        self.rng = np.random.default_rng(seed)

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(7)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_steps = 0
        self.state = 3
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1
        delta = -1 if action == 0 else 1
        self.state = int(np.clip(self.state + delta, 0, 6))

        terminated = self.state in (0, 6)
        truncated = self.current_steps >= self.horizon
        reward = 1.0 if self.state == 6 else 0.0

        return self.state, reward, terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        print(f"[RandomWalkTDEnv] state={self.state}, steps={self.current_steps}")
