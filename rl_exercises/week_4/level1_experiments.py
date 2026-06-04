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


def write_observations(
    output_path: Path, summary: pd.DataFrame, env: str, num_f: int, seed: int
) -> None:
    best_row = summary.iloc[0]

    text = f"""Week 4 - Level 1 Observations

Setup
- Environment: {env}
- Seed: {seed}
- Frames: {num_f}
- Configurations: {len(EXPERIMENTS)} variants with different network sizes, replay buffer sizes, and batch sizes

Generated Artifacts
- results/week_4/l1/level1_metrics.csv
- results/week_4/l1/level1_training_curves.png
- results/week_4/l1/level1_summary.csv

Result
- Best final configuration: {best_row["config"]} with avg_reward_10 = {best_row["avg_reward_10"]:.1f}

Discussion
- Larger or deeper networks were not automatically better.
- Some deeper models reached high peaks but were less stable.
- Since this Level-1 experiment uses only one seed, the result should be interpreted as a first indication rather than a robust conclusion.
"""

    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    results_dir = Path("results/week_4/l1")
    results_dir.mkdir(parents=True, exist_ok=True)

    env = "CartPole-v1"
    num_f = 20000
    seed = 0
    df = run_experiments(env_name=env, num_frames=num_f, seed=seed)

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

    obsv_path = Path("rl_exercises/week_4/observations_l1.txt")
    write_observations(
        output_path=obsv_path, summary=summary, env=env, num_f=num_f, seed=seed
    )

    print("\nSaved:")
    print(csv_path)
    print(fig_path)
    print(summary_path)
    print(obsv_path)


if __name__ == "__main__":
    main()
