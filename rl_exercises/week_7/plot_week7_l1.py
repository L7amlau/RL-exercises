from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

csv_files = sorted(
    Path("outputs").glob("*/*/*.csv"),
    key=lambda p: p.stat().st_mtime,
)

dqn_files = [p for p in csv_files if p.name.startswith("dqn_seed")][-3:]
rnd_files = [p for p in csv_files if p.name.startswith("rnd_dqn_seed")][-3:]

runs = {
    "DQN": dqn_files,
    "RND-DQN": rnd_files,
}

print("Using files:")
for algo, paths in runs.items():
    for path in paths:
        print(algo, path)

plt.figure(figsize=(8, 5))

for algo, paths in runs.items():
    dfs = []

    for path in paths:
        df = pd.read_csv(path)

        # 平滑
        df["episode_reward"] = df["episode_reward"].rolling(10, min_periods=1).mean()

        dfs.append(df)

    min_len = min(len(df) for df in dfs)

    rewards = []
    steps = dfs[0]["step"].iloc[:min_len]

    for df in dfs:
        rewards.append(df["episode_reward"].iloc[:min_len].to_numpy())

    import numpy as np

    rewards = np.array(rewards)

    mean = rewards.mean(axis=0)
    std = rewards.std(axis=0)

    plt.plot(
        steps,
        mean,
        label=algo,
    )

    plt.fill_between(
        steps,
        mean - std,
        mean + std,
        alpha=0.2,
    )

plt.xlabel("Environment Steps")
plt.ylabel("Episode Reward")
plt.title("DQN vs RND-DQN on LunarLander-v3")
plt.legend()
plt.grid(True)

Path("results/week_7").mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    "results/week_7/level1_learning_curve.png",
    dpi=300,
    bbox_inches="tight",
)
