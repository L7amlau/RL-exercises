from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_interval_estimates, plot_sample_efficiency_curve

ENV_NAME = "CartPole-v1"
NUM_FRAMES = 20000
SEEDS = [0, 1, 2, 3, 4]
SOLVED_SCORE = 500.0

VARIANTS = {
    "base_dqn": {
        "use_prioritized_replay": False,
        "use_double_dqn": False,
    },
    "dqn_per": {
        "use_prioritized_replay": True,
        "use_double_dqn": False,
    },
    "dqn_double": {
        "use_prioritized_replay": False,
        "use_double_dqn": True,
    },
    "dqn_per_double": {
        "use_prioritized_replay": True,
        "use_double_dqn": True,
    },
}

BASE_AGENT_KWARGS = {
    "buffer_capacity": 10000,
    "batch_size": 32,
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_final": 0.01,
    "epsilon_decay": 500,
    "target_update_freq": 1000,
    "hidden_dims": [64, 64],
    "prio_alpha": 0.6,
    "prio_beta_start": 0.4,
    "prio_beta_frames": NUM_FRAMES,
}


def run_variant_seed(variant_name: str, variant_cfg: dict, seed: int) -> pd.DataFrame:
    print(f"\n=== Running {variant_name}, seed={seed} ===")
    env = gym.make(ENV_NAME)
    set_seed(env, seed)

    agent_kwargs = dict(BASE_AGENT_KWARGS)
    agent_kwargs.update(variant_cfg)

    agent = DQNAgent(
        env=env,
        seed=seed,
        **agent_kwargs,
    )

    logs = agent.train(num_frames=NUM_FRAMES, eval_interval=10)
    env.close()

    rows = []
    for episode_idx, row in enumerate(logs, start=1):
        rows.append(
            {
                "variant": variant_name,
                "seed": int(seed),
                "episode": int(episode_idx),
                "frame": float(row["frame"]),
                "episode_reward": float(row["episode_reward"]),
                "avg_reward_10": float(row["avg_reward_10"]),
            }
        )
    return pd.DataFrame(rows)


def interpolate_variant_curves(
    df: pd.DataFrame, frame_grid: np.ndarray
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for variant, vg in df.groupby("variant"):
        seed_curves = []
        for seed, sg in vg.groupby("seed"):
            _ = seed
            s = sg.sort_values("frame")
            x = s["frame"].to_numpy(dtype=float)
            y = s["avg_reward_10"].to_numpy(dtype=float)
            seed_curves.append(np.interp(frame_grid, x, y) / SOLVED_SCORE)
        out[variant] = np.asarray(seed_curves, dtype=float)
    return out


def build_final_score_matrix(df: pd.DataFrame) -> dict[str, np.ndarray]:
    score_dict: dict[str, np.ndarray] = {}
    for variant, vg in df.groupby("variant"):
        final = (
            vg.sort_values(["seed", "frame"])
            .groupby("seed", as_index=False)
            .tail(1)["avg_reward_10"]
            .to_numpy(dtype=float)
            .reshape(-1, 1)
            / SOLVED_SCORE
        )
        score_dict[variant] = np.clip(final, 0.0, 1.0)
    return score_dict


def make_iqm_curve_plot(df: pd.DataFrame, output_path: Path) -> None:
    frame_grid = np.linspace(1, NUM_FRAMES, 200)
    score_dict = interpolate_variant_curves(df, frame_grid)

    iqm_over_time = lambda scores: np.array(  # noqa: E731
        [metrics.aggregate_iqm(scores[:, i]) for i in range(scores.shape[1])]
    )
    iqm_scores, iqm_cis = get_interval_estimates(score_dict, iqm_over_time, reps=2000)

    _ = plot_sample_efficiency_curve(
        frame_grid,
        iqm_scores,
        iqm_cis,
        algorithms=list(VARIANTS.keys()),
        xlabel="Frames",
        ylabel="IQM Normalized Score",
    )
    plt.title("Week 4 Level 3: Rainbow Ablation (IQM curves)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_final_interval_plot(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    score_dict = build_final_score_matrix(df)

    aggregate_fn = lambda x: np.array(  # noqa: E731
        [
            metrics.aggregate_mean(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_optimality_gap(x, gamma=1.0),
        ]
    )
    points, cis = get_interval_estimates(score_dict, aggregate_fn, reps=5000)

    _ = plot_interval_estimates(
        points,
        cis,
        metric_names=["Mean", "IQM", "Opt. Gap"],
        algorithms=list(VARIANTS.keys()),
        xlabel="Normalized score",
    )
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(output_path, dpi=150)
    plt.close()

    rows = []
    for variant in VARIANTS.keys():
        p = points[variant]
        ci = cis[variant]
        rows.append(
            {
                "variant": variant,
                "metric": "mean_final_score",
                "value": float(p[0] * SOLVED_SCORE),
                "ci95_low": float(ci[0, 0] * SOLVED_SCORE),
                "ci95_high": float(ci[1, 0] * SOLVED_SCORE),
            }
        )
        rows.append(
            {
                "variant": variant,
                "metric": "iqm_final_score",
                "value": float(p[1] * SOLVED_SCORE),
                "ci95_low": float(ci[0, 1] * SOLVED_SCORE),
                "ci95_high": float(ci[1, 1] * SOLVED_SCORE),
            }
        )
        rows.append(
            {
                "variant": variant,
                "metric": "mean_optimality_gap",
                "value": float(p[2] * SOLVED_SCORE),
                "ci95_low": float(ci[0, 2] * SOLVED_SCORE),
                "ci95_high": float(ci[1, 2] * SOLVED_SCORE),
            }
        )
    return pd.DataFrame(rows)


def write_observations(observations_path: Path, summary_df: pd.DataFrame) -> None:
    iqm_df = summary_df[summary_df["metric"] == "iqm_final_score"].copy()
    iqm_df = iqm_df.sort_values("value", ascending=False)

    lines = []
    for _, row in iqm_df.iterrows():
        lines.append(
            f"- {row['variant']}: IQM={row['value']:.2f} "
            f"(95% CI [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}])"
        )

    text = f"""Week 4 - Level 3 Observations (Rainbow Ablation)

Setup
- Environment: {ENV_NAME}
- Seeds: {SEEDS}
- Frames per run: {NUM_FRAMES}
- Variants: {list(VARIANTS.keys())}

Generated Artifacts
- results/week_4/l3/level3_metrics.csv
- results/week_4/l3/level3_iqm_curves.png
- results/week_4/l3/level3_final_interval_estimates.png
- results/week_4/l3/level3_summary.csv

Ranking by IQM final score
{chr(10).join(lines)}

Discussion
- Compare Base DQN vs DQN+PER vs DQN+DoubleDQN vs DQN+PER+DoubleDQN.
- Check if PER improves sample-efficiency (curve shape) and if Double DQN improves stability.
- Use the interval overlaps to judge whether gains are likely robust.
"""
    observations_path.write_text(text, encoding="utf-8")


def main() -> None:
    results_dir = Path("results/week_4/l3")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_df = []
    for variant_name, variant_cfg in VARIANTS.items():
        for seed in SEEDS:
            all_df.append(run_variant_seed(variant_name, variant_cfg, seed))

    df = pd.concat(all_df, ignore_index=True)
    metrics_csv = results_dir / "level3_metrics.csv"
    curve_png = results_dir / "level3_iqm_curves.png"
    interval_png = results_dir / "level3_final_interval_estimates.png"
    summary_csv = results_dir / "level3_summary.csv"
    observations_path = Path("rl_exercises/week_4/observations_l3.txt")

    df.to_csv(metrics_csv, index=False)
    make_iqm_curve_plot(df, curve_png)
    summary_df = make_final_interval_plot(df, interval_png)
    summary_df.to_csv(summary_csv, index=False)
    write_observations(observations_path, summary_df)

    print("\nSaved:")
    print(metrics_csv)
    print(curve_png)
    print(interval_png)
    print(summary_csv)
    print(observations_path)


if __name__ == "__main__":
    main()
