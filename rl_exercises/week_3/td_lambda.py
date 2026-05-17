from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_3.epsilon_greedy_policy import EpsilonGreedyPolicy

State = Any


class TDLambdaAgent(AbstractAgent):
    """On-policy TD(lambda) control agent with eligibility traces."""

    def __init__(
        self,
        env: gym.Env,
        policy: EpsilonGreedyPolicy,
        alpha: float = 0.1,
        gamma: float = 0.99,
        lambda_: float = 0.8,
    ) -> None:
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert 0 <= lambda_ <= 1, "lambda should be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.policy = policy
        self.n_actions = env.action_space.n

        self.Q: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )
        self.E: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

    def predict_action(
        self, state: np.ndarray, info: dict = {}, evaluate: bool = False
    ) -> tuple[int, dict]:
        return self.policy(self.Q, state, evaluate=evaluate), info

    def save(self, path: str) -> None:  # type: ignore[override]
        np.save(path, dict(self.Q))

    def load(self, path: str) -> None:  # type: ignore[override]
        loaded_q = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=float), loaded_q)

    def update_agent(self, batch) -> float:  # type: ignore[override]
        state, action, reward, next_state, done, _ = batch[0]

        next_action = 0
        next_q = 0.0
        if not done:
            next_action = self.policy(self.Q, next_state, evaluate=False)
            next_q = float(self.Q[next_state][next_action])

        td_target = float(reward) + self.gamma * next_q
        td_error = td_target - float(self.Q[state][action])

        self.E[state][action] += 1.0

        for s in list(self.E.keys()):
            self.Q[s] += self.alpha * td_error * self.E[s]
            self.E[s] *= self.gamma * self.lambda_

        if done:
            self.E = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

        return float(self.Q[state][action])
