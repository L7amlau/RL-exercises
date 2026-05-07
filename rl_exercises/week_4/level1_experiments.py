from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

from rl_exercises.week_4.dqn import DQNAgent, set_seed


EXPERIMENTS = [
    {
        "name": "mlp_64x64_buf10k_bs32",
        "hidden_dims": [64, 64],
        "buffer_capacity": 10000,
        "batch_size": 32,
    },
    {
        "name": "mlp_128x128_buf10k_bs32",
        "hidden_dims": [128, 128],
        "buffer_capacity": 10000,
        "batch_size": 32,
    },
    {
        "name": "mlp_64x64x64_buf10k_bs32",
        "hidden_dims": [64, 64, 64],
        "buffer_capacity": 10000,
        "batch_size": 32,
    },
    {
        "name": "mlp_64x64_buf5k_bs32",
        "hidden_dims": [64, 64],
        "buffer_capacity": 5000,
        "batch_size": 32,
    },
    {
        "name": "mlp_64x64_buf10k_bs64",
        "hidden_dims": [64, 64],
        "buffer_capacity": 10000,
        "batch_size": 64,
    },
]


def run_experiments(
    env_name: str = "CartPole-v1",
    num_frames: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    all_rows: list[dict] = []

    for exp in EXPERIMENTS:
        print(f"\n=== Running {exp['name']} ===")
        env = gym.make(env_name)
        set_seed(env, seed)

        agent = DQNAgent(
            env=env,
            buffer_capacity=int(exp["buffer_capacity"]),
            batch_size=int(exp["batch_size"]),
            hidden_dims=list(exp["hidden_dims"]),
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_final=0.01,
            epsilon_decay=500,
            target_update_freq=1000,
            seed=seed,
        )

        logs = agent.train(num_frames=num_frames, eval_interval=10)
        env.close()

        for episode_idx, row in enumerate(logs, start=1):
            all_rows.append(
                {
                    "config": exp["name"],
                    "seed": seed,
                    "episode": episode_idx,
                    "frame": row["frame"],
                    "episode_reward": row["episode_reward"],
                    "avg_reward_10": row["avg_reward_10"],
                    "hidden_dims": str(exp["hidden_dims"]),
                    "buffer_capacity": int(exp["buffer_capacity"]),
                    "batch_size": int(exp["batch_size"]),
                }
            )

    return pd.DataFrame(all_rows)


def plot_training_curves(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    for config_name, group in df.groupby("config"):
        g = group.sort_values("frame")
        plt.plot(g["frame"], g["avg_reward_10"], label=config_name)

    plt.title("DQN on CartPole-v1 (Level 1 Experiments)")
    plt.xlabel("Frames")
    plt.ylabel("Mean Reward (Moving Average over 10 Episodes)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    results_dir = Path("results/week_4/l1")
    results_dir.mkdir(parents=True, exist_ok=True)

    df = run_experiments(env_name="CartPole-v1", num_frames=20000, seed=0)

    csv_path = results_dir / "level1_metrics.csv"
    fig_path = results_dir / "level1_training_curves.png"

    df.to_csv(csv_path, index=False)
    plot_training_curves(df, fig_path)

    summary = (
        df.sort_values(["config", "frame"])  # ensure last row is final per config
        .groupby("config", as_index=False)
        .tail(1)[["config", "frame", "avg_reward_10"]]
        .sort_values("avg_reward_10", ascending=False)
    )
    summary_path = results_dir / "level1_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(csv_path)
    print(fig_path)
    print(summary_path)


if __name__ == "__main__":
    main()
