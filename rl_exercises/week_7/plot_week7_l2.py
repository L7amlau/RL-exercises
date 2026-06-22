from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from rl_exercises.week_7.noveid_ppo import NovelDPPOAgent

ENV_NAME = "LunarLander-v3"
HIDDEN_SIZE = 128


def find_latest_noveid_run() -> Path:
    candidates = [
        p
        for p in Path("outputs").glob("*/*")
        if (p / "noveid_snapshot_20000.pt").exists()
    ]

    if not candidates:
        raise FileNotFoundError("No NoveID run with snapshots found.")

    return sorted(candidates)[-1]


def rollout(checkpoint_path: Path):
    env = gym.make(ENV_NAME)

    agent = NovelDPPOAgent(
        env=env,
        hidden_size=HIDDEN_SIZE,
        rnd_hidden_size=HIDDEN_SIZE,
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    agent.policy.load_state_dict(checkpoint["policy"])
    agent.obs_rms = checkpoint["obs_rms"]
    agent.policy.eval()

    obs, _ = env.reset(seed=0)

    xs = []
    ys = []

    done = False
    truncated = False

    while not (done or truncated):
        xs.append(obs[0])
        ys.append(obs[1])

        obs_norm = agent._normalize_obs(obs)
        obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0)

        with torch.no_grad():
            probs = agent.policy(obs_tensor)
            action = torch.argmax(probs, dim=-1).item()

        obs, _, done, truncated, _ = env.step(action)

    env.close()

    return xs, ys


def main() -> None:
    run_dir = find_latest_noveid_run()
    print(f"Using run directory: {run_dir}")

    snapshots = {
        "1000 steps": run_dir / "noveid_snapshot_1000.pt",
        "10000 steps": run_dir / "noveid_snapshot_10000.pt",
        "20000 steps": run_dir / "noveid_snapshot_20000.pt",
    }

    output_dir = Path("results/week_7")
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, checkpoint_path in snapshots.items():
        xs, ys = rollout(checkpoint_path)

        plt.figure(figsize=(6, 5))
        plt.plot(xs, ys, marker="o", markersize=2)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title(f"NoveID PPO behavior snapshot at {label}")
        plt.grid(True)
        plt.tight_layout()

        filename = f"noveid_snapshot_{label.replace(' ', '_')}.png"
        output_path = output_dir / filename

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
