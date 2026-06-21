from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

RAW_DIR = Path("results/week_8/l1_raw")


def load_final_scores() -> np.ndarray:
    final_scores = []

    for csv_file in sorted(RAW_DIR.glob("dqn_seed*.csv")):
        df = pd.read_csv(csv_file)
        final_scores.append(df.iloc[-1]["avg_reward_10"])

    return np.asarray(final_scores, dtype=float)


def compute_statistics(scores: np.ndarray) -> dict:
    mean = np.mean(scores)
    median = np.median(scores)
    std = np.std(scores, ddof=1)
    se = std / np.sqrt(len(scores))

    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se

    iqm = metrics.aggregate_iqm(scores)

    return {
        "n": len(scores),
        "mean": mean,
        "median": median,
        "iqm": iqm,
        "std": std,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def print_statistics(stats: dict) -> None:
    print(f"N      : {stats['n']}")
    print(f"Mean   : {stats['mean']:.2f}")
    print(f"Median : {stats['median']:.2f}")
    print(f"IQM    : {stats['iqm']:.2f}")
    print(f"Std    : {stats['std']:.2f}")
    print(f"SE     : {stats['se']:.2f}")
    print(f"95% CI : [{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]")


def summarize_seed_groups(scores: np.ndarray) -> pd.DataFrame:
    groups = {
        "low_0_1_2": scores[[0, 1, 2]],
        "low_10_11_12": scores[[10, 11, 12]],
        "low_20_21_22": scores[[20, 21, 22]],
        "medium_0_to_9": scores[0:10],
        "large_0_to_29": scores[0:30],
    }

    rows = []
    for name, group_scores in groups.items():
        stats = compute_statistics(group_scores)
        stats["group"] = name
        rows.append(stats)

    return pd.DataFrame(rows)


def load_all_curves() -> pd.DataFrame:
    dfs = []

    for csv_file in sorted(RAW_DIR.glob("dqn_seed*.csv")):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def interpolate_curves(df: pd.DataFrame, frame_grid: np.ndarray) -> np.ndarray:
    curves = []

    for seed, group in df.groupby("seed"):
        group = group.sort_values("frame")

        x = group["frame"].to_numpy(dtype=float)
        y = group["avg_reward_10"].to_numpy(dtype=float)

        y_interp = np.interp(frame_grid, x, y)
        curves.append(y_interp)

    return np.asarray(curves, dtype=float)


def plot_iqm_curve(curves: np.ndarray, frame_grid: np.ndarray) -> None:
    score_dict = {"DQN": curves}

    def iqm_over_time(scores):
        return np.array(
            [metrics.aggregate_iqm(scores[:, i]) for i in range(scores.shape[1])]
        )

    iqm_scores, iqm_cis = get_interval_estimates(
        score_dict,
        iqm_over_time,
        reps=2000,
    )

    plot_sample_efficiency_curve(
        frame_grid,
        iqm_scores,
        iqm_cis,
        algorithms=["DQN"],
        xlabel="Frames",
        ylabel="IQM avg reward over last 10 episodes",
    )

    plt.title("DQN IQM learning curve with 95% CI")
    plt.tight_layout()

    out_path = Path("results/week_8/l1_iqm_rliable_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved plot to {out_path}")


def plot_mean_median_iqm_curves(curves: np.ndarray, frame_grid: np.ndarray) -> None:
    mean_curve = np.mean(curves, axis=0)
    median_curve = np.median(curves, axis=0)
    iqm_curve = np.array(
        [metrics.aggregate_iqm(curves[:, i]) for i in range(curves.shape[1])]
    )

    plt.figure(figsize=(10, 5))
    plt.plot(frame_grid, mean_curve, label="Mean")
    plt.plot(frame_grid, median_curve, label="Median")
    plt.plot(frame_grid, iqm_curve, label="IQM")

    plt.title("DQN on CartPole-v1: Mean, Median, and IQM over Time")
    plt.xlabel("Frames")
    plt.ylabel("AvgReward(10)")
    plt.legend()
    plt.tight_layout()

    out_path = Path("results/week_8/l1_mean_median_iqm_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved plot to {out_path}")


def plot_mean_with_ci(curves: np.ndarray, frame_grid: np.ndarray) -> None:
    mean_curve = np.mean(curves, axis=0)

    std_curve = np.std(curves, axis=0, ddof=1)
    se_curve = std_curve / np.sqrt(curves.shape[0])

    ci_low = mean_curve - 1.96 * se_curve
    ci_high = mean_curve + 1.96 * se_curve

    plt.figure(figsize=(10, 5))

    plt.plot(frame_grid, mean_curve, label="Mean")

    plt.fill_between(
        frame_grid,
        ci_low,
        ci_high,
        alpha=0.3,
        label="95% CI",
    )

    plt.title("DQN Mean Reward with 95% Confidence Interval")
    plt.xlabel("Frames")
    plt.ylabel("AvgReward(10)")
    plt.legend()

    plt.tight_layout()

    out_path = Path("results/week_8/l1_mean_ci_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved plot to {out_path}")


def plot_std_se_curves(curves: np.ndarray, frame_grid: np.ndarray) -> None:
    std_curve = np.std(curves, axis=0, ddof=1)
    se_curve = std_curve / np.sqrt(curves.shape[0])

    plt.figure(figsize=(10, 5))
    plt.plot(frame_grid, std_curve, label="Standard deviation")
    plt.plot(frame_grid, se_curve, label="Standard error")

    plt.title("DQN on CartPole-v1: Standard Deviation and Standard Error")
    plt.xlabel("Frames")
    plt.ylabel("AvgReward(10)")
    plt.legend()
    plt.tight_layout()

    out_path = Path("results/week_8/l1_uncertainty_std_se.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved plot to {out_path}")


def main() -> None:
    scores = load_final_scores()

    print("=== Large seed set summary ===")
    stats = compute_statistics(scores)
    print_statistics(stats)

    print("\n=== Seed group comparison ===")
    summary_df = summarize_seed_groups(scores)
    print(
        summary_df[
            ["group", "n", "mean", "median", "iqm", "std", "se", "ci_low", "ci_high"]
        ].round(2)
    )
    out_path = Path("results/week_8/l1_seed_group_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}")

    all_curves_df = load_all_curves()
    frame_grid = np.linspace(1, 20000, 200)
    curves = interpolate_curves(all_curves_df, frame_grid)

    plot_iqm_curve(curves, frame_grid)
    plot_mean_median_iqm_curves(curves, frame_grid)
    plot_mean_with_ci(curves, frame_grid)
    plot_std_se_curves(curves, frame_grid)


if __name__ == "__main__":
    main()
