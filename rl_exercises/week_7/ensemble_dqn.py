"""
Deep Q-Learning with ensemble-based exploration.
"""

from typing import Any, Dict, List, Sequence, Tuple

import importlib

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import PrioritizedReplayBuffer, ReplayBuffer
from torch import nn

try:
    hydra = importlib.import_module("hydra")
except ModuleNotFoundError:
    hydra = None


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class EnsembleQNetwork(nn.Module):
    """
    Q-network with multiple output heads.

    For each state, it returns Q-values from several heads.
    Output shape: [batch_size, num_heads, num_actions]
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: int | Sequence[int] = (64, 64),
        num_heads: int = 5,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        layers: list[nn.Module] = []
        input_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.feature = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            [nn.Linear(input_dim, num_actions) for _ in range(num_heads)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        q_values = [head(features) for head in self.heads]
        return torch.stack(q_values, dim=1)


class EnsembleDQNAgent(AbstractAgent):
    """
    Deep Q-Learning agent with an ensemble of Q-heads.

    The ensemble estimates epistemic uncertainty from disagreement between
    heads. During training, actions are selected with a UCB-style score:

        mean_Q(s, a) + ucb_beta * std_Q(s, a)

    Evaluation uses only mean_Q(s, a).
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        hidden_dims: int | Sequence[int] = (64, 64),
        num_heads: int = 5,
        ucb_beta: float = 1.0,
        use_prioritized_replay: bool = False,
        prio_alpha: float = 0.6,
        prio_beta_start: float = 0.4,
        prio_beta_frames: int = 20000,
        use_double_dqn: bool = False,
        seed: int = 0,
    ) -> None:
        """
        Initialize replay buffer, ensemble Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial epsilon for random exploration.
        epsilon_final : float
            Final epsilon for random exploration.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        hidden_dims : int or sequence of int
            Hidden layer widths.
        num_heads : int
            Number of Q-heads in the ensemble.
        ucb_beta : float
            Strength of uncertainty bonus during action selection.
        use_prioritized_replay : bool
            Use prioritized replay instead of uniform replay sampling.
        prio_alpha : float
            Priority exponent for proportional prioritized replay.
        prio_beta_start : float
            Initial importance-sampling exponent.
        prio_beta_frames : int
            Frames over which beta is annealed to 1.0.
        use_double_dqn : bool
            Use Double DQN target computation.
        seed : int
            RNG seed.
        """
        super().__init__()
        self.env = env
        self.seed = seed
        set_seed(env, seed)

        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise ValueError("EnsembleDQNAgent expects a Box observation space.")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("EnsembleDQNAgent expects a Discrete action space.")

        obs_dim = int(obs_space.shape[0])
        n_actions = int(action_space.n)

        self.num_heads = int(num_heads)
        self.ucb_beta = float(ucb_beta)

        # main ensemble Q-network and frozen target ensemble
        self.q = EnsembleQNetwork(
            obs_dim,
            n_actions,
            hidden_dims=hidden_dims,
            num_heads=self.num_heads,
        )
        self.target_q = EnsembleQNetwork(
            obs_dim,
            n_actions,
            hidden_dims=hidden_dims,
            num_heads=self.num_heads,
        )
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.use_prioritized_replay = bool(use_prioritized_replay)
        if self.use_prioritized_replay:
            self.buffer: ReplayBuffer | PrioritizedReplayBuffer = (
                PrioritizedReplayBuffer(
                    buffer_capacity,
                    alpha=prio_alpha,
                )
            )
        else:
            self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.prio_beta_start = float(prio_beta_start)
        self.prio_beta_frames = int(max(1, prio_beta_frames))
        self.use_double_dqn = bool(use_double_dqn)

        self.total_steps = 0  # environment interaction steps (for epsilon decay)
        self.update_steps = 0  # gradient updates (for target net sync)

    def current_beta(self) -> float:
        progress = min(1.0, self.total_steps / float(self.prio_beta_frames))
        return float(self.prio_beta_start + progress * (1.0 - self.prio_beta_start))

    def epsilon(self) -> float:
        """
        Compute current epsilon by exponential decay.

        Returns
        -------
        float
            Random exploration rate.
        """
        return float(
            self.epsilon_final
            + (self.epsilon_start - self.epsilon_final)
            * np.exp(-self.total_steps / self.epsilon_decay)
        )

    def predict_action(
        self,
        state: np.ndarray,
        info: Dict[str, Any] | None = None,
        evaluate: bool = False,
    ) -> int:
        """
        Choose an action.

        In training mode, this uses epsilon-random exploration plus ensemble UCB.
        In evaluation mode, this uses the ensemble mean only.

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(mean_Q).

        Returns
        -------
        action : int
            Selected action.
        """
        _ = info
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        if (not evaluate) and (np.random.rand() < self.epsilon()):
            return int(self.env.action_space.sample())

        with torch.no_grad():
            q_ensemble = self.q(t).squeeze(0)  # [num_heads, num_actions]
            q_mean = q_ensemble.mean(dim=0)

            if evaluate:
                action_scores = q_mean
            else:
                q_std = q_ensemble.std(dim=0, unbiased=False)
                action_scores = q_mean + self.ucb_beta * q_std

        action = int(torch.argmax(action_scores).item())
        return action

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "num_heads": self.num_heads,
                "ucb_beta": self.ucb_beta,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self,
        training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]] | Dict[str, Any],
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions or prioritized replay dict
            Each transition is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            Loss value.
        """
        # unpack
        sample_weights: np.ndarray | None = None
        sample_indices: List[int] | None = None
        batch_for_update: List[Tuple[Any, Any, float, Any, bool, Dict]]

        if isinstance(training_batch, dict):
            batch_for_update = training_batch["batch"]
            sample_indices = training_batch["indices"]
            sample_weights = np.asarray(training_batch["weights"], dtype=np.float32)
        else:
            batch_for_update = training_batch

        states, actions, rewards, next_states, dones, _ = zip(*batch_for_update)
        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.int64)
        r = torch.tensor(np.array(rewards), dtype=torch.float32)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)
        mask = torch.tensor(np.array(dones), dtype=torch.float32)

        # Current Q estimates for taken actions.
        # q_values shape: [batch_size, num_heads, num_actions]
        q_values = self.q(s)
        action_index = a[:, None, None].expand(-1, self.num_heads, 1)
        pred = q_values.gather(dim=2, index=action_index).squeeze(-1)
        # pred shape: [batch_size, num_heads]

        with torch.no_grad():
            target_q_values = self.target_q(s_next)

            if self.use_double_dqn:
                # Select the next action by the online ensemble mean.
                online_next_q_values = self.q(s_next)
                online_next_mean = online_next_q_values.mean(dim=1)
                next_actions = online_next_mean.argmax(dim=1)
                next_action_index = next_actions[:, None, None].expand(
                    -1,
                    self.num_heads,
                    1,
                )
                next_q = target_q_values.gather(
                    dim=2,
                    index=next_action_index,
                ).squeeze(-1)
            else:
                # Each head uses its own target max.
                next_q = target_q_values.max(dim=2).values

            target = r[:, None] + (1.0 - mask[:, None]) * self.gamma * next_q

        td_errors = target - pred
        per_sample_loss = torch.square(td_errors).mean(dim=1)

        if sample_weights is not None:
            w = torch.tensor(sample_weights, dtype=torch.float32)
            loss = (w * per_sample_loss).mean()
        else:
            loss = per_sample_loss.mean()

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (
            self.use_prioritized_replay
            and sample_indices is not None
            and isinstance(self.buffer, PrioritizedReplayBuffer)
        ):
            new_priorities = (
                torch.abs(td_errors).mean(dim=1).detach().cpu().numpy() + 1e-6
            )
            self.buffer.update_priorities(sample_indices, new_priorities)

        # occasionally sync target network
        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def train(
        self, num_frames: int, eval_interval: int = 1000
    ) -> List[Dict[str, float]]:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Print moving average every `eval_interval` episodes.

        Returns
        -------
        list[dict[str, float]]
            Per-episode training metrics containing frame, episode_reward and avg_reward_10.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        logs: List[Dict[str, float]] = []
        episode_rewards = []
        steps = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # store and step
            self.buffer.add(state, action, float(reward), next_state, done, {})
            state = next_state
            ep_reward += float(reward)
            self.total_steps += 1

            # update if ready
            if len(self.buffer) >= self.batch_size:
                if self.use_prioritized_replay and isinstance(
                    self.buffer, PrioritizedReplayBuffer
                ):
                    batch = self.buffer.sample(
                        self.batch_size,
                        beta=self.current_beta(),
                    )
                else:
                    batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                avg10 = float(np.mean(recent_rewards[-10:]))
                logs.append(
                    {
                        "frame": float(frame),
                        "episode_reward": float(ep_reward),
                        "avg_reward_10": avg10,
                    }
                )
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % max(1, int(eval_interval)) == 0:
                    print(
                        f"Frame {frame}, AvgReward(10): {avg10:.2f}, "
                        f"epsilon={self.epsilon():.3f}, ucb_beta={self.ucb_beta:.3f}"
                    )

        df = pd.DataFrame(
            {
                "step": steps,
                "episode_reward": episode_rewards,
            }
        )
        df.to_csv(f"ensemble_dqn_seed{self.seed}.csv", index=False)
        print("Training complete.")
        return logs


if hydra is not None:

    @hydra.main(
        config_path="../configs/agent/", config_name="ensemble_dqn", version_base="1.1"
    )
    def _main(cfg: Any) -> None:
        # 1) build env
        env = gym.make(cfg.env.name)
        set_seed(env, cfg.seed)

        # 2/3) instantiate & train
        agent = EnsembleDQNAgent(
            env=env,
            buffer_capacity=int(cfg.agent.buffer_capacity),
            batch_size=int(cfg.agent.batch_size),
            lr=float(cfg.agent.learning_rate),
            gamma=float(cfg.agent.gamma),
            epsilon_start=float(cfg.agent.epsilon_start),
            epsilon_final=float(cfg.agent.epsilon_final),
            epsilon_decay=int(cfg.agent.epsilon_decay),
            target_update_freq=int(cfg.agent.target_update_freq),
            hidden_dims=list(cfg.agent.hidden_dims),
            num_heads=int(cfg.agent.get("num_heads", 5)),
            ucb_beta=float(cfg.agent.get("ucb_beta", 1.0)),
            use_prioritized_replay=bool(cfg.agent.get("use_prioritized_replay", False)),
            prio_alpha=float(cfg.agent.get("prio_alpha", 0.6)),
            prio_beta_start=float(cfg.agent.get("prio_beta_start", 0.4)),
            prio_beta_frames=int(
                cfg.agent.get("prio_beta_frames", cfg.train.num_frames)
            ),
            use_double_dqn=bool(cfg.agent.get("use_double_dqn", False)),
            seed=int(cfg.seed),
        )
        agent.train(
            num_frames=int(cfg.train.num_frames),
            eval_interval=int(cfg.train.eval_interval),
        )

    def main() -> None:
        _main()

else:

    def main() -> None:
        raise ModuleNotFoundError(
            "hydra-core is required to run ensemble_dqn.py main()."
        )


if __name__ == "__main__":
    main()
