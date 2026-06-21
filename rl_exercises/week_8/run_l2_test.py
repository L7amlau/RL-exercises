from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

DQN_DIR = Path("results/week_8/l1_raw")
RND_DQN_DIR = Path("results/week_8/l2_raw")
OUT_DIR = Path("results/week_8")
OBS_PATH = Path("rl_exercises/week_8/observations_l2.txt")

SEEDS = list(range(5))
ALPHA = 0.05


def final_score_from_dqn(seed: int) -> float:
    df = pd.read_csv(DQN_DIR / f"dqn_seed{seed}.csv")
    return float(df.iloc[-1]["avg_reward_10"])


def final_score_from_rnd_dqn(seed: int) -> float:
    df = pd.read_csv(RND_DQN_DIR / f"rnd_dqn_seed{seed}.csv")
    return float(df["episode_reward"].tail(10).mean())


def load_scores() -> pd.DataFrame:
    rows = []

    for seed in SEEDS:
        rows.append(
            {
                "algorithm": "DQN",
                "seed": seed,
                "final_avg_reward_10": final_score_from_dqn(seed),
            }
        )
        rows.append(
            {
                "algorithm": "RND-DQN",
                "seed": seed,
                "final_avg_reward_10": final_score_from_rnd_dqn(seed),
            }
        )

    return pd.DataFrame(rows)


def make_boxplot(scores_df: pd.DataFrame) -> None:
    dqn_scores = scores_df.loc[scores_df["algorithm"] == "DQN", "final_avg_reward_10"]
    rnd_scores = scores_df.loc[
        scores_df["algorithm"] == "RND-DQN", "final_avg_reward_10"
    ]

    plt.figure(figsize=(7, 5))
    plt.boxplot([dqn_scores, rnd_scores], tick_labels=["DQN", "RND-DQN"])
    plt.ylabel("Final AvgReward(10)")
    plt.title("DQN vs RND-DQN on CartPole-v1")
    plt.tight_layout()

    out_path = OUT_DIR / "l2_dqn_vs_rnd_dqn_boxplot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved plot to {out_path}")


def write_observations(
    dqn_scores: pd.Series,
    rnd_scores: pd.Series,
    statistic: float,
    p_value: float,
) -> None:
    conclusion = (
        "reject the null hypothesis"
        if p_value < ALPHA
        else "fail to reject the null hypothesis"
    )

    text = f"""
Setup:
- Algorithms: DQN vs RND-DQN
- Environment: CartPole-v1
- Seeds: {SEEDS}
- Aggregation: final average reward over the last 10 episodes of each run
- Test: two-sided Mann-Whitney U test
- Significance level: alpha = {ALPHA}

Results:
- DQN scores: {dqn_scores.round(2).tolist()}
- RND-DQN scores: {rnd_scores.round(2).tolist()}
- U statistic: {statistic:.4f}
- p-value: {p_value:.4f}

Conclusion:
Since p = {p_value:.4f}, we {conclusion} at alpha = {ALPHA}.

Notes:
Each seed was aggregated into one final score before testing. I did not treat individual episodes as independent samples, because episodes from the same training run are correlated.
"""

    OBS_PATH.write_text(text, encoding="utf-8")
    print(f"Saved observations to {OBS_PATH}")


def main() -> None:
    scores_df = load_scores()

    dqn_scores = scores_df.loc[scores_df["algorithm"] == "DQN", "final_avg_reward_10"]
    rnd_scores = scores_df.loc[
        scores_df["algorithm"] == "RND-DQN", "final_avg_reward_10"
    ]

    statistic, p_value = mannwhitneyu(
        dqn_scores,
        rnd_scores,
        alternative="two-sided",
    )

    scores_path = OUT_DIR / "l2_dqn_vs_rnd_dqn_scores.csv"
    scores_df.to_csv(scores_path, index=False)

    print(scores_df)
    print(f"\nU statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Saved scores to {scores_path}")

    make_boxplot(scores_df)
    write_observations(dqn_scores, rnd_scores, statistic, p_value)


if __name__ == "__main__":
    main()
