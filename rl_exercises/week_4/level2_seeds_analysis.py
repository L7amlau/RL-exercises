from __future__ import annotations

from pathlib import Path
from typing import Iterable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import (
    plot_interval_estimates,
    plot_performance_profiles,
    plot_sample_efficiency_curve,
)

from rl_exercises.week_4.dqn import DQNAgent, set_seed

# Level-1 best config used as base DQN for the seed study.
CONFIG = {
    "name": "base_dqn_64x64_buf10k_bs32",
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

SEEDS = [0, 1, 2, 3, 4]
ENV_NAME = "CartPole-v1"
NUM_FRAMES = 20000
SOLVED_SCORE = 500.0


def run_seed(seed: int) -> pd.DataFrame:
    print(f"\n=== Running seed {seed} ===")
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
                "config": CONFIG["name"],
                "seed": int(seed),
                "episode": int(episode_idx),
                "frame": float(row["frame"]),
                "episode_reward": float(row["episode_reward"]),
                "avg_reward_10": float(row["avg_reward_10"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    final_by_seed = (
        df.sort_values(["seed", "frame"])  # per seed chronological
        .groupby("seed", as_index=False)
        .tail(1)
        .copy()
    )
    final_by_seed["optimality_gap"] = SOLVED_SCORE - final_by_seed["avg_reward_10"]
    final_by_seed = final_by_seed[
        [
            "seed",
            "frame",
            "episode_reward",
            "avg_reward_10",
            "optimality_gap",
        ]
    ].sort_values("seed")
    return final_by_seed


def summarize_final_scores_with_rliable(final_scores: pd.DataFrame) -> pd.DataFrame:
    """Compute robust final-score metrics and CIs with rliable."""
    raw_scores = final_scores["avg_reward_10"].to_numpy(dtype=float).reshape(-1, 1)
    normalized_scores = np.clip(raw_scores / SOLVED_SCORE, 0.0, 1.0)

    score_dict = {"dqn": normalized_scores}
    aggregate_fn = lambda x: np.array(  # noqa: E731
        [
            metrics.aggregate_mean(x),
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_optimality_gap(x, gamma=1.0),
        ]
    )
    points, cis = get_interval_estimates(score_dict, aggregate_fn, reps=5000)

    metric_names = [
        "mean_final_score",
        "median_final_score",
        "iqm_final_score",
        "mean_optimality_gap",
    ]
    p = points["dqn"]
    ci = cis["dqn"]

    rows = []
    for idx, name in enumerate(metric_names):
        val = float(p[idx])
        lo = float(ci[0, idx])
        hi = float(ci[1, idx])

        if name == "mean_optimality_gap":
            rows.append(
                {
                    "metric": name,
                    "value": val * SOLVED_SCORE,
                    "ci95_low": lo * SOLVED_SCORE,
                    "ci95_high": hi * SOLVED_SCORE,
                    "normalized_value": val,
                    "normalized_ci95_low": lo,
                    "normalized_ci95_high": hi,
                }
            )
        else:
            rows.append(
                {
                    "metric": name,
                    "value": val * SOLVED_SCORE,
                    "ci95_low": lo * SOLVED_SCORE,
                    "ci95_high": hi * SOLVED_SCORE,
                    "normalized_value": val,
                    "normalized_ci95_low": lo,
                    "normalized_ci95_high": hi,
                }
            )

    return pd.DataFrame(rows)


def interpolate_curves(df: pd.DataFrame, frame_grid: np.ndarray) -> np.ndarray:
    seed_curves = []
    for seed, g in df.groupby("seed"):
        _ = seed
        gg = g.sort_values("frame")
        x = gg["frame"].to_numpy(dtype=float)
        y = gg["avg_reward_10"].to_numpy(dtype=float)
        y_interp = np.interp(frame_grid, x, y)
        seed_curves.append(y_interp)
    return np.asarray(seed_curves, dtype=float)


def make_training_curve_plot_with_rliable(df: pd.DataFrame, output_path: Path) -> None:
    """Create IQM sample-efficiency curve with rliable utilities."""
    frame_grid = np.linspace(1, NUM_FRAMES, 200)
    curves = interpolate_curves(df, frame_grid) / SOLVED_SCORE
    score_dict = {"dqn": curves}
    iqm_over_time = lambda scores: np.array(  # noqa: E731
        [metrics.aggregate_iqm(scores[:, i]) for i in range(scores.shape[1])]
    )
    iqm_scores, iqm_cis = get_interval_estimates(score_dict, iqm_over_time, reps=2000)

    _ = plot_sample_efficiency_curve(
        frame_grid,
        iqm_scores,
        iqm_cis,
        algorithms=["dqn"],
        xlabel="Frames",
        ylabel="IQM Normalized Score",
    )
    plt.title("Level 2 IQM Sample-Efficiency Curve (rliable)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_performance_profile_plot_with_rliable(
    final_scores: pd.DataFrame, output_path: Path
) -> None:
    """Create performance profile with rliable plot utilities."""
    normalized = np.clip(
        final_scores["avg_reward_10"].to_numpy(dtype=float).reshape(-1, 1) / SOLVED_SCORE,
        0.0,
        1.0,
    )
    score_dict = {"dqn": normalized}
    tau_list = np.linspace(0.0, 1.0, 200)

    profile_fn = lambda scores: np.array(  # noqa: E731
        [np.mean(scores > tau) for tau in tau_list]
    )
    profile_points, profile_cis = get_interval_estimates(score_dict, profile_fn, reps=2000)

    _ = plot_performance_profiles(
        profile_points,
        tau_list,
        performance_profile_cis=profile_cis,
        xlabel="Normalized score threshold",
        ylabel="Fraction of runs with score > threshold",
    )
    plt.title("Level 2 Performance Profile (rliable)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_interval_summary_plot_with_rliable(
    final_scores: pd.DataFrame, output_path: Path
) -> None:
    """Create interval estimate plot for mean/median/IQM/optimality gap."""
    normalized = np.clip(
        final_scores["avg_reward_10"].to_numpy(dtype=float).reshape(-1, 1) / SOLVED_SCORE,
        0.0,
        1.0,
    )
    score_dict = {"dqn": normalized}

    aggregate_fn = lambda x: np.array(  # noqa: E731
        [
            metrics.aggregate_mean(x),
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_optimality_gap(x, gamma=1.0),
        ]
    )
    points, cis = get_interval_estimates(score_dict, aggregate_fn, reps=5000)

    _ = plot_interval_estimates(
        points,
        cis,
        metric_names=["Mean", "Median", "IQM", "Opt. Gap"],
        algorithms=["dqn"],
        xlabel="Normalized score",
    )
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _fmt_metric_line(row: pd.Series) -> str:
    return (
        f"- {row['metric']}: {row['value']:.2f} "
        f"(95% CI [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}])"
    )


def write_observations(
    output_path: Path,
    final_scores: pd.DataFrame,
    summary: pd.DataFrame,
    seeds: Iterable[int],
) -> None:
    metric_lines = "\n".join(_fmt_metric_line(r) for _, r in summary.iterrows())
    best_seed_row = final_scores.sort_values("avg_reward_10", ascending=False).iloc[0]
    worst_seed_row = final_scores.sort_values("avg_reward_10", ascending=True).iloc[0]

    text = f"""Week 4 - Level 2 Observations (Multi-seed DQN Analysis)

Setup
- Environment: {ENV_NAME}
- Seeds: {list(seeds)}
- Frames per seed: {NUM_FRAMES}
- Base configuration: {CONFIG['name']} with hidden_dims={CONFIG['hidden_dims']}, buffer={CONFIG['buffer_capacity']}, batch={CONFIG['batch_size']}

Generated Artifacts
- results/week_4/l2/level2_metrics_by_seed.csv
- results/week_4/l2/level2_final_scores.csv
- results/week_4/l2/level2_aggregate_summary.csv
- results/week_4/l2/level2_iqm_sample_efficiency_curve.png
- results/week_4/l2/level2_performance_profile.png
- results/week_4/l2/level2_interval_estimates.png

Aggregate Metrics (final avg_reward_10 over seeds)
{metric_lines}

Seed Spread (final avg_reward_10)
- Best seed: {int(best_seed_row['seed'])} with score {best_seed_row['avg_reward_10']:.2f}
- Worst seed: {int(worst_seed_row['seed'])} with score {worst_seed_row['avg_reward_10']:.2f}
- Spread (best - worst): {(best_seed_row['avg_reward_10'] - worst_seed_row['avg_reward_10']):.2f}

Discussion
- Compared to a single-seed report, the multi-seed view reveals substantial variance and therefore lower certainty.
- Mean alone can be influenced by outlier seeds; IQM provides a robust central tendency.
- In this single-task setup, rliable aggregate_median equals aggregate_mean by definition.
- The rliable performance profile shows how often the method exceeds different score thresholds, which is more informative than one point estimate.
- rliable confidence intervals make uncertainty explicit, improving trustworthiness of conclusions.

Confidence Statement
- I am more confident in these results than with single-seed reporting, because variance and uncertainty are quantified directly.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    results_dir = Path("results/week_4/l2")
    results_dir.mkdir(parents=True, exist_ok=True)

    seed_dfs = [run_seed(seed) for seed in SEEDS]
    metrics_df = pd.concat(seed_dfs, ignore_index=True)
    final_scores = aggregate_seed_metrics(metrics_df)
    summary_df = summarize_final_scores_with_rliable(final_scores)

    metrics_csv = results_dir / "level2_metrics_by_seed.csv"
    final_csv = results_dir / "level2_final_scores.csv"
    summary_csv = results_dir / "level2_aggregate_summary.csv"
    curves_png = results_dir / "level2_iqm_sample_efficiency_curve.png"
    profile_png = results_dir / "level2_performance_profile.png"
    interval_png = results_dir / "level2_interval_estimates.png"

    metrics_df.to_csv(metrics_csv, index=False)
    final_scores.to_csv(final_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    make_training_curve_plot_with_rliable(metrics_df, curves_png)
    make_performance_profile_plot_with_rliable(final_scores, profile_png)
    make_interval_summary_plot_with_rliable(final_scores, interval_png)

    write_observations(
        output_path=Path("rl_exercises/week_4/observations_l2.txt"),
        final_scores=final_scores,
        summary=summary_df,
        seeds=SEEDS,
    )

    print("\nSaved:")
    print(metrics_csv)
    print(final_csv)
    print(summary_csv)
    print(curves_png)
    print(profile_png)
    print(interval_png)
    print(Path("rl_exercises/week_4/observations_l2.txt"))


if __name__ == "__main__":
    main()
