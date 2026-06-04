from typing import Any

import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from rl_exercises.agent import AbstractAgent
from rliable import library as rlib


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed random number generators for reproducibility.

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment to seed.
    seed : int, optional
        Seed value for NumPy, PyTorch, and environment (default is 0).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class Actor(nn.Module):
    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.state_dim = int(np.prod(state_space.shape))
        self.action_dim = int(np.prod(action_space.shape))

        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.action_dim)

        self.action_scale = torch.tensor(
            action_space.high,
            dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return x * self.action_scale


class Critic(nn.Module):
    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.state_dim = int(np.prod(state_space.shape))
        self.action_dim = int(np.prod(action_space.shape))

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:

        if state.dim() == 1:
            state = state.unsqueeze(0)

        if action.dim() == 1:
            action = action.unsqueeze(0)

        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))

        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DDPGAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        hidden_size: int = 128,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 64,
        noise_std: float = 0.1,
    ):
        super().__init__(env)

        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std

        self.actor = Actor(
            env.observation_space,
            env.action_space,
            hidden_size,
        )

        self.critic = Critic(
            env.observation_space,
            env.action_space,
            hidden_size,
        )

        self.target_actor = Actor(
            env.observation_space,
            env.action_space,
            hidden_size,
        )

        self.target_critic = Critic(
            env.observation_space,
            env.action_space,
            hidden_size,
        )

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = True,
    ) -> np.ndarray:

        state_t = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action = self.actor(state_t).numpy()[0]

        if explore:
            noise = np.random.normal(
                0,
                self.noise_std,
                size=action.shape,
            )

            action = action + noise

        return np.clip(
            action,
            self.action_low,
            self.action_high,
        )

    def predict_action(
        self,
        state: np.ndarray,
        info: dict[str, Any] | None = None,
        evaluate: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:

        action = self.select_action(
            state,
            explore=not evaluate,
        )

        return action, {}

    def soft_update(
        self,
        target: nn.Module,
        source: nn.Module,
    ):

        for target_param, source_param in zip(
            target.parameters(),
            source.parameters(),
        ):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def update(self):

        if len(self.replay_buffer) < self.batch_size:
            return None, None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)

            target_q = self.target_critic(
                next_states,
                next_actions,
            )

            y = rewards + self.gamma * (1.0 - dones) * target_q

        current_q = self.critic(states, actions)

        critic_loss = torch.nn.functional.mse_loss(
            current_q,
            y,
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_actions = self.actor(states)

        actor_loss = -self.critic(
            states,
            actor_actions,
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(
            self.target_actor,
            self.actor,
        )

        self.soft_update(
            self.target_critic,
            self.critic,
        )

        return (
            actor_loss.item(),
            critic_loss.item(),
        )

    def evaluate(self, episodes: int = 5) -> float:
        returns = []

        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            episode_return = 0.0

            while not done:
                action, _ = self.predict_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated
                episode_return += reward
                state = next_state

            returns.append(episode_return)

        return float(np.mean(returns))

    def train(
        self,
        episodes: int = 500,
        eval_interval: int = 10,
        eval_episodes: int = 5,
    ) -> tuple[list[float], list[tuple[int, float]]]:
        train_returns = []
        eval_returns = []

        for ep in range(1, episodes + 1):
            state, _ = self.env.reset()
            done = False
            episode_return = 0.0

            while not done:
                action, _ = self.predict_action(state, evaluate=False)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )
                self.update()

                episode_return += reward
                state = next_state

            train_returns.append(episode_return)

            if ep % eval_interval == 0:
                eval_return = self.evaluate(eval_episodes)
                eval_returns.append((ep, eval_return))

                print(
                    f"[DDPG] Ep {ep:4d} "
                    f"TrainReturn {episode_return:8.1f} "
                    f"EvalReturn {eval_return:8.1f}"
                )

        return train_returns, eval_returns


def run_exp(seed: int):

    env = gym.make("Pendulum-v1")
    set_seed(env, seed)
    agent = DDPGAgent(env)

    train_returns, eval_returns = agent.train(
        episodes=500,
        eval_interval=10,
        eval_episodes=5,
    )

    env.close()
    return train_returns, eval_returns


if __name__ == "__main__":
    # main()
    seeds = [0, 1, 2, 3, 4]
    all_eval_returns = []

    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        _, eval_returns = run_exp(seed)
        all_eval_returns.append(eval_returns)

    eval_array = np.array(
        [[value for _, value in eval_returns] for eval_returns in all_eval_returns]
    )

    episodes = np.array([ep for ep, _ in all_eval_returns[0]])
    np.save("results/week_5/ddpg_eval_returns.npy", eval_array)

    score_dict = {
        "DDPG": eval_array,
    }

    def aggregate_mean_over_time(scores):
        return np.mean(scores, axis=0)

    point_estimates, interval_estimates = rlib.get_interval_estimates(
        score_dict,
        aggregate_mean_over_time,
        reps=5000,
    )

    ddpg_mean = point_estimates["DDPG"]
    ddpg_ci = interval_estimates["DDPG"]

    plt.figure()
    plt.plot(episodes, ddpg_mean, label="DDPG mean eval return")
    plt.fill_between(
        episodes,
        ddpg_ci[0],
        ddpg_ci[1],
        alpha=0.2,
        label="95% bootstrap CI",
    )
    plt.xlabel("Episode")
    plt.ylabel("Evaluation return")
    plt.title("DDPG on Pendulum-v1 over 5 seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/week_5/ddpg_pendulum_reliable.png")
    plt.close()

    print("Finish all seeds.")
    print("Saved results to results/week_5/ddpg_eval_returns.npy")
    print("Saved results to results/week_5/ddpg_pendulum_reliable.png")
