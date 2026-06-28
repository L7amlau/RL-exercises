"""Run and persist all experiments required by Week 9 Level 1.

Example:
    python -m rl_exercises.week_9.run_level1_experiments
"""

from __future__ import annotations

from typing import Any

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rl_exercises_matplotlib")

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from rl_exercises.week_9.dyna_ppo import DynaPPOAgent  # noqa: E402


@dataclass(frozen=True)
class ExperimentConfig:
    env_name: str = "CartPole-v1"
    total_steps: int = 15_000
    eval_interval: int = 1_000
    eval_episodes: int = 10
    seeds: tuple[int, ...] = (0, 1, 2)
    heldout_transitions: int = 1_000
    multistep_horizon: int = 20
    multistep_sequences: int = 256
    multistep_random_action_probability: float = 0.5


def policy_probabilities(agent: DynaPPOAgent, state: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        probabilities = agent.policy(
            torch.as_tensor(state, dtype=torch.float32)
        ).reshape(-1)
    result = probabilities.cpu().numpy().astype(np.float64)
    return result / result.sum()


def greedy_action(agent: DynaPPOAgent, state: np.ndarray) -> int:
    return int(np.argmax(policy_probabilities(agent, state)))


def evaluate_policy(
    agent: DynaPPOAgent,
    env_name: str,
    seed: int,
    real_steps: int,
    num_episodes: int,
) -> list[float]:
    """Evaluate without consuming the RNG streams used during training."""
    env = gym.make(env_name)
    returns = []
    for episode in range(num_episodes):
        reset_seed = 1_000_000 + seed * 10_000 + real_steps + episode
        state, _ = env.reset(seed=reset_seed)
        done = False
        total_reward = 0.0
        while not done:
            state, reward, terminated, truncated, _ = env.step(
                greedy_action(agent, state)
            )
            total_reward += float(reward)
            done = terminated or truncated
        returns.append(total_reward)
    env.close()
    return returns


def collect_one_step_holdout(
    agent: DynaPPOAgent,
    env_name: str,
    seed: int,
    real_steps: int,
    num_transitions: int,
) -> list[tuple[np.ndarray, int, float, np.ndarray]]:
    """Collect current-policy transitions in an independent environment."""
    env = gym.make(env_name)
    rng = np.random.default_rng(2_000_000 + seed * 100_000 + real_steps)
    transitions = []
    episode = 0
    while len(transitions) < num_transitions:
        state, _ = env.reset(seed=2_500_000 + seed * 10_000 + real_steps + episode)
        done = False
        while not done and len(transitions) < num_transitions:
            probabilities = policy_probabilities(agent, state)
            action = int(rng.choice(len(probabilities), p=probabilities))
            next_state, reward, terminated, truncated, _ = env.step(action)
            transitions.append(
                (
                    np.asarray(state, dtype=np.float32),
                    action,
                    float(reward),
                    np.asarray(next_state, dtype=np.float32),
                )
            )
            state = next_state
            done = terminated or truncated
        episode += 1
    env.close()
    return transitions


def evaluate_one_step_model(
    agent: DynaPPOAgent,
    transitions: list[tuple[np.ndarray, int, float, np.ndarray]],
) -> dict[str, float]:
    states = torch.as_tensor(np.asarray([item[0] for item in transitions])).float()
    actions = torch.as_tensor([item[1] for item in transitions]).long()
    rewards = torch.as_tensor([item[2] for item in transitions]).float()
    next_states = torch.as_tensor(np.asarray([item[3] for item in transitions])).float()
    actions_oh = F.one_hot(actions, num_classes=agent.env.action_space.n).float()

    was_training = agent.model.training
    agent.model.eval()
    with torch.no_grad():
        delta_pred, reward_pred = agent.model(states, actions_oh)
        next_states_pred = states + delta_pred
        metrics = {
            "state_mse": F.mse_loss(next_states_pred, next_states).item(),
            "reward_mse": F.mse_loss(reward_pred, rewards).item(),
            "state_mae": F.l1_loss(next_states_pred, next_states).item(),
            "reward_mae": F.l1_loss(reward_pred, rewards).item(),
        }
    agent.model.train(was_training)
    return metrics


def collect_multistep_holdout(
    agent: DynaPPOAgent,
    env_name: str,
    seed: int,
    real_steps: int,
    horizon: int,
    num_sequences: int,
    random_action_probability: float,
    max_episodes: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect held-out real sequences long enough for every requested k.

    A policy/uniform mixture prevents a collapsed early policy from making the
    k=20 metric undefined. The same behavior-policy definition is used at all
    checkpoints, and no collected transition is added to the training buffer.
    """
    env = gym.make(env_name)
    rng = np.random.default_rng(3_000_000 + seed * 100_000 + real_steps)
    initial_states: list[np.ndarray] = []
    action_sequences: list[np.ndarray] = []
    target_sequences: list[np.ndarray] = []

    for episode in range(max_episodes):
        state, _ = env.reset(seed=3_500_000 + seed * 10_000 + real_steps + episode)
        states = [np.asarray(state, dtype=np.float32)]
        actions = []
        done = False
        while not done:
            policy_probs = policy_probabilities(agent, state)
            behavior_probs = (
                1.0 - random_action_probability
            ) * policy_probs + random_action_probability / len(policy_probs)
            action = int(rng.choice(len(behavior_probs), p=behavior_probs))
            state, _, terminated, truncated, _ = env.step(action)
            actions.append(action)
            states.append(np.asarray(state, dtype=np.float32))
            done = terminated or truncated

        for start in range(max(0, len(actions) - horizon + 1)):
            initial_states.append(states[start])
            action_sequences.append(
                np.asarray(actions[start : start + horizon], dtype=np.int64)
            )
            target_sequences.append(
                np.asarray(states[start + 1 : start + horizon + 1], dtype=np.float32)
            )
            if len(initial_states) >= num_sequences:
                env.close()
                return (
                    np.asarray(initial_states),
                    np.asarray(action_sequences),
                    np.asarray(target_sequences),
                )

    env.close()
    raise RuntimeError(
        f"Collected only {len(initial_states)} valid {horizon}-step sequences "
        f"after {max_episodes} held-out episodes"
    )


def evaluate_multistep_model(
    agent: DynaPPOAgent,
    initial_states: np.ndarray,
    action_sequences: np.ndarray,
    target_sequences: np.ndarray,
) -> list[float]:
    predicted_states = torch.as_tensor(initial_states, dtype=torch.float32)
    actions = torch.as_tensor(action_sequences, dtype=torch.long)
    targets = torch.as_tensor(target_sequences, dtype=torch.float32)
    errors = []

    was_training = agent.model.training
    agent.model.eval()
    with torch.no_grad():
        for offset in range(action_sequences.shape[1]):
            actions_oh = F.one_hot(
                actions[:, offset], num_classes=agent.env.action_space.n
            ).float()
            delta_pred, _ = agent.model(predicted_states, actions_oh)
            predicted_states = predicted_states + delta_pred
            errors.append(
                float(F.mse_loss(predicted_states, targets[:, offset]).item())
            )
    agent.model.train(was_training)
    return errors


def make_agent(env: gym.Env, seed: int, use_model: bool) -> DynaPPOAgent:
    return DynaPPOAgent(
        env,
        use_model=use_model,
        lr_actor=5e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=4,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
        seed=seed,
        hidden_size=128,
        model_lr=5e-4,
        model_epochs=3,
        model_batch_size=128,
        imag_horizon=5,
        imag_batches=10,
        max_buffer_size=10_000,
    )


def run_training(
    config: ExperimentConfig, seed: int, use_model: bool
) -> dict[str, list[dict[str, Any]]]:
    algorithm = "Dyna-PPO" if use_model else "PPO"
    env = gym.make(config.env_name)
    agent = make_agent(env, seed, use_model)
    state, _ = env.reset()
    episode_done = False
    episode = 0
    episode_return = 0.0
    next_evaluation = config.eval_interval

    results: dict[str, list[dict[str, Any]]] = {
        "returns": [],
        "training": [],
        "model_accuracy": [],
        "multistep": [],
    }

    while agent.real_steps < config.total_steps:
        trajectory = []
        update_return = 0.0
        while (
            not episode_done
            and agent.real_steps < config.total_steps
            and agent.real_steps < next_evaluation
        ):
            action, logp, entropy, _ = agent.predict(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_done = terminated or truncated
            trajectory.append(
                (
                    state,
                    action,
                    logp,
                    entropy,
                    float(reward),
                    float(episode_done),
                    next_state,
                )
            )
            state = next_state
            agent.real_steps += 1
            episode_return += float(reward)
            update_return += float(reward)

        policy_loss, value_loss, entropy_loss = agent.update(trajectory)
        model_state_loss = 0.0
        model_reward_loss = 0.0
        imag_policy_loss = 0.0
        imag_value_loss = 0.0
        imag_entropy_loss = 0.0
        if use_model:
            agent.store_real(trajectory)
            model_state_loss, model_reward_loss = agent.train_model()
            imag_policy_loss, imag_value_loss, imag_entropy_loss = (
                agent.imagine_and_update()
            )

        results["training"].append(
            {
                "algorithm": algorithm,
                "seed": seed,
                "real_steps": agent.real_steps,
                "episode": episode,
                "update_real_return": update_return,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "model_state_loss": model_state_loss,
                "model_reward_loss": model_reward_loss,
                "imag_policy_loss": imag_policy_loss,
                "imag_value_loss": imag_value_loss,
                "imag_entropy_loss": imag_entropy_loss,
                "imagination_steps": agent.imagination_steps,
            }
        )

        if episode_done:
            episode += 1
            agent.total_episodes += 1
            state, _ = env.reset()
            episode_done = False
            episode_return = 0.0

        if agent.real_steps == next_evaluation:
            evaluation_returns = evaluate_policy(
                agent,
                config.env_name,
                seed,
                next_evaluation,
                config.eval_episodes,
            )
            for eval_episode, episode_return_value in enumerate(evaluation_returns):
                results["returns"].append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "real_steps": next_evaluation,
                        "eval_episode": eval_episode,
                        "return": episode_return_value,
                    }
                )

            if use_model:
                transitions = collect_one_step_holdout(
                    agent,
                    config.env_name,
                    seed,
                    next_evaluation,
                    config.heldout_transitions,
                )
                metrics = evaluate_one_step_model(agent, transitions)
                results["model_accuracy"].append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "real_steps": next_evaluation,
                        "num_transitions": len(transitions),
                        **metrics,
                    }
                )

                initial_states, action_sequences, target_sequences = (
                    collect_multistep_holdout(
                        agent,
                        config.env_name,
                        seed,
                        next_evaluation,
                        config.multistep_horizon,
                        config.multistep_sequences,
                        config.multistep_random_action_probability,
                    )
                )
                errors = evaluate_multistep_model(
                    agent, initial_states, action_sequences, target_sequences
                )
                for horizon, error in enumerate(errors, start=1):
                    results["multistep"].append(
                        {
                            "algorithm": algorithm,
                            "seed": seed,
                            "real_steps": next_evaluation,
                            "horizon": horizon,
                            "state_mse": error,
                            "num_sequences": len(initial_states),
                        }
                    )

            mean_return = float(np.mean(evaluation_returns))
            print(
                f"algorithm={algorithm:<8} seed={seed} "
                f"steps={next_evaluation:5d} return={mean_return:6.1f}",
                flush=True,
            )
            next_evaluation += config.eval_interval

    env.close()
    return results


def summarize_with_seed_means(
    frame: pd.DataFrame, value_columns: list[str], group_columns: list[str]
) -> pd.DataFrame:
    seed_means = (
        frame.groupby(group_columns + ["seed"], as_index=False)[value_columns]
        .mean()
        .reset_index(drop=True)
    )
    summary = seed_means.groupby(group_columns)[value_columns].agg(
        ["mean", "std", "sem"]
    )
    summary.columns = [f"{value}_{stat}" for value, stat in summary.columns]
    return summary.reset_index()


def plot_return_curve(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algorithm, group in summary.groupby("algorithm"):
        group = group.sort_values("real_steps")
        x = group["real_steps"].to_numpy()
        y = group["return_mean"].to_numpy()
        ci = 1.96 * group["return_sem"].fillna(0.0).to_numpy()
        ax.plot(x, y, marker="o", label=algorithm)
        ax.fill_between(x, y - ci, y + ci, alpha=0.2)
    final_step = summary["real_steps"].max()
    final_ppo = summary[
        (summary["algorithm"] == "PPO") & (summary["real_steps"] == final_step)
    ]["return_mean"].iloc[0]
    ax.axhline(
        0.8 * final_ppo,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="80% of final PPO return",
    )
    ax.set(title="PPO vs Dyna-PPO Sample Efficiency", xlabel="Real environment steps")
    ax.set_ylabel("Average evaluation return")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "l1_avg_return_vs_real_steps.png", dpi=180)
    plt.close(fig)


def plot_one_step_mse(
    summary: pd.DataFrame, raw: pd.DataFrame, output_dir: Path
) -> None:
    ordered = summary.sort_values("real_steps")
    x = ordered["real_steps"].to_numpy()
    y = ordered["state_mse_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, seed_data in raw.groupby("seed"):
        seed_data = seed_data.sort_values("real_steps")
        ax.plot(
            seed_data["real_steps"],
            seed_data["state_mse"],
            color="tab:green",
            alpha=0.25,
            linewidth=1.0,
            label=f"Seed {seed}",
        )
    ax.plot(x, y, marker="o", color="tab:green", linewidth=2.5, label="Mean")
    ax.set_yscale("log")
    ax.set(title="Held-out One-step Dynamics Error", xlabel="Real environment steps")
    ax.set_ylabel("Next-state MSE (log scale)")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "l1_one_step_mse_vs_real_steps.png", dpi=180)
    plt.close(fig)


def plot_multistep_error(summary: pd.DataFrame, output_dir: Path) -> None:
    early_step = int(summary["real_steps"].min())
    late_step = int(summary["real_steps"].max())
    fig, ax = plt.subplots(figsize=(8, 5))
    for step, label in ((early_step, "Early"), (late_step, "Late")):
        group = summary[summary["real_steps"] == step].sort_values("horizon")
        x = group["horizon"].to_numpy()
        y = group["state_mse_mean"].to_numpy()
        ci = 1.96 * group["state_mse_sem"].fillna(0.0).to_numpy()
        lower = np.maximum(y - ci, np.finfo(float).tiny)
        ax.plot(x, y, marker="o", label=f"{label} ({step:,} steps)")
        ax.fill_between(x, lower, y + ci, alpha=0.2)
    ax.set_yscale("log")
    ax.set(
        title="Compounding Open-loop Model Error",
        xlabel="Rollout horizon k",
        ylabel="E_k: state MSE (log scale)",
        xticks=[1, 5, 10, 15, 20],
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "l1_multistep_error_early_late.png", dpi=180)
    plt.close(fig)


def save_results(
    config: ExperimentConfig,
    all_results: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "l1_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.DataFrame(rows) for name, rows in all_results.items()}
    for name, frame in frames.items():
        frame.to_csv(raw_dir / f"{name}.csv", index=False)

    returns_summary = summarize_with_seed_means(
        frames["returns"], ["return"], ["algorithm", "real_steps"]
    )
    model_summary = summarize_with_seed_means(
        frames["model_accuracy"],
        ["state_mse", "reward_mse", "state_mae", "reward_mae"],
        ["algorithm", "real_steps"],
    )
    multistep_summary = summarize_with_seed_means(
        frames["multistep"],
        ["state_mse"],
        ["algorithm", "real_steps", "horizon"],
    )
    summaries = {
        "returns": returns_summary,
        "model_accuracy": model_summary,
        "multistep": multistep_summary,
    }
    for name, frame in summaries.items():
        frame.to_csv(output_dir / f"l1_{name}_summary.csv", index=False)

    payload = {
        "config": asdict(config),
        "raw": {
            name: frame.to_dict(orient="records") for name, frame in frames.items()
        },
        "summary": {
            name: frame.to_dict(orient="records") for name, frame in summaries.items()
        },
    }
    (output_dir / "l1_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    plot_return_curve(returns_summary, output_dir)
    plot_one_step_mse(model_summary, frames["model_accuracy"], output_dir)
    plot_multistep_error(multistep_summary, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[2] / "results" / "week_9",
    )
    parser.add_argument("--total-steps", type=int, default=15_000)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_steps % args.eval_interval != 0:
        raise ValueError("total_steps must be divisible by eval_interval")
    config = ExperimentConfig(
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seeds=tuple(args.seeds),
    )
    torch.set_num_threads(1)
    all_results: dict[str, list[dict[str, Any]]] = {
        "returns": [],
        "training": [],
        "model_accuracy": [],
        "multistep": [],
    }
    for use_model in (False, True):
        for seed in config.seeds:
            run_results = run_training(config, seed, use_model)
            for name, records in run_results.items():
                all_results[name].extend(records)
    save_results(config, all_results, args.output_dir)
    print(f"Saved Level 1 evidence to {args.output_dir}")


if __name__ == "__main__":
    main()
