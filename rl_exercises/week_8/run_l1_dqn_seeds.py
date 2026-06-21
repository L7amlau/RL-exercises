from pathlib import Path

import gymnasium as gym
import pandas as pd
from rl_exercises.week_4.dqn import DQNAgent, set_seed

ENV_NAME = "CartPole-v1"
SEEDS = list(range(30))
NUM_FRAMES = 20000
OUT_DIR = Path("results/week_8/l1_raw")

CONFIG = {
    "hidden_dims": [64, 64],
    "buffer_capacity": 10000,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_final": 0.01,
    "epsilon_decay": 500,
    "target_update_freq": 1000,
}


def run_seed(seed: int) -> pd.DataFrame:
    print(f"\n=== Running DQN on {ENV_NAME}, seed={seed} ===")

    env = gym.make(ENV_NAME)
    set_seed(env, seed)

    agent = DQNAgent(
        env=env,
        buffer_capacity=int(CONFIG["buffer_capacity"]),
        batch_size=int(CONFIG["batch_size"]),
        lr=float(CONFIG["learning_rate"]),
        gamma=float(CONFIG["gamma"]),
        epsilon_start=float(CONFIG["epsilon_start"]),
        epsilon_final=float(CONFIG["epsilon_final"]),
        epsilon_decay=int(CONFIG["epsilon_decay"]),
        target_update_freq=int(CONFIG["target_update_freq"]),
        hidden_dims=list(CONFIG["hidden_dims"]),
        seed=seed,
    )

    logs = agent.train(num_frames=NUM_FRAMES, eval_interval=10)
    env.close()

    rows = []
    for episode_idx, row in enumerate(logs, start=1):
        rows.append(
            {
                "seed": seed,
                "episode": episode_idx,
                "frame": int(row["frame"]),
                "episode_reward": float(row["episode_reward"]),
                "avg_reward_10": float(row["avg_reward_10"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        out_path = OUT_DIR / f"dqn_seed{seed}.csv"

        if out_path.exists():
            print(f"Skipping seed={seed}, already exists: {out_path}")
            continue

        df = run_seed(seed)
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
