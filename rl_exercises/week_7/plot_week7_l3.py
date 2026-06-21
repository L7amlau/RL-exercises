from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics, plot_utils

ALGORITHMS = {
    "DQN": "dqn_seed{seed}.csv",
    "RND-DQN": "rnd_dqn_seed{seed}.csv",
    "Ensemble-DQN": "ensemble_dqn_seed{seed}.csv",
}

SEEDS = [0, 1, 2]
EVAL_STEPS = np.linspace(0, 50000, 101)


def find_latest_file(filename: str) -> Path:
    candidates = list(Path("outputs").glob(f"*/*/{filename}"))

    if not candidates:
        raise FileNotFoundError(f"No output file found for {filename}")

    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def load_curve(path: Path) -> np.ndarray:
    df = pd.read_csv(path)

    if "step" not in df.columns or "episode_reward" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {df.columns}")

    df = df.sort_values("step")

    steps = df["step"].to_numpy()
    rewards = df["episode_reward"].to_numpy()

    return np.interp(EVAL_STEPS, steps, rewards)


def main() -> None:
    score_dict: dict[str, np.ndarray] = {}

    for algorithm, filename_template in ALGORITHMS.items():
        seed_curves = []

        for seed in SEEDS:
            filename = filename_template.format(seed=seed)
            path = find_latest_file(filename)
            print(f"Using {algorithm}, seed {seed}: {path}")
            seed_curves.append(load_curve(path))

        # RLiable expects shape:
        # [num_runs, num_tasks, num_eval_points]
        # We only have one task: LunarLander-v3.
        score_dict[algorithm] = np.asarray(seed_curves)[:, None, :]

    def iqm(scores: np.ndarray) -> np.ndarray:
        return np.array(
            [
                metrics.aggregate_iqm(scores[..., frame])
                for frame in range(scores.shape[-1])
            ]
        )

    iqm_scores, iqm_cis = rly.get_interval_estimates(
        score_dict,
        iqm,
        reps=2000,
    )

    output_dir = Path("results/week_7")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plot_utils.plot_sample_efficiency_curve(
        EVAL_STEPS,
        iqm_scores,
        iqm_cis,
        algorithms=list(ALGORITHMS.keys()),
        xlabel="Environment steps",
        ylabel="IQM episode reward",
    )

    plt.title("DQN vs RND-DQN vs Ensemble-DQN on LunarLander-v3")
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / "l3_dqn_rnd_ensemble_rliable.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
