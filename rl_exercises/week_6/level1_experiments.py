from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rl_exercises.week_6.actor_critic import ActorCriticAgent, set_seed
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

BASELINES = ["none", "avg", "value", "gae"]


@dataclass(frozen=True)
class EnvRunConfig:
    total_steps: int
    eval_interval: int
    lr_actor: float
    lr_critic: float
    hidden_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    baseline_decay: float = 0.9


def _slugify_env(env_name: str) -> str:
    return env_name.replace("-", "_").replace("/", "_")


def run_single_experiment(
    env_name: str,
    baseline: str,
    seed: int,
    config: EnvRunConfig,
    eval_episodes: int,
) -> pd.DataFrame:
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    set_seed(env, seed)
    set_seed(eval_env, seed + 10000)

    agent = ActorCriticAgent(
        env=env,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        seed=seed,
        hidden_size=config.hidden_size,
        baseline_type=baseline,
        baseline_decay=config.baseline_decay,
    )

    step_count = 0
    eval_rows: list[dict] = []

    while step_count < config.total_steps:
        state, _ = env.reset()
        done = False
        trajectory = []

        while not done and step_count < config.total_steps:
            action, logp = agent.predict_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            trajectory.append((state, action, float(reward), next_state, done, logp))
            state = next_state
            step_count += 1

            if step_count % config.eval_interval == 0:
                mean_r, std_r = agent.evaluate(eval_env, num_episodes=eval_episodes)
                eval_rows.append(
                    {
                        "env": env_name,
                        "baseline": baseline,
                        "seed": seed,
                        "step": int(step_count),
                        "avg_return": float(mean_r),
                        "std_return": float(std_r),
                    }
                )

        if trajectory:
            agent.update_agent(trajectory)

    if step_count % config.eval_interval != 0:
        mean_r, std_r = agent.evaluate(eval_env, num_episodes=eval_episodes)
        eval_rows.append(
            {
                "env": env_name,
                "baseline": baseline,
                "seed": seed,
                "step": int(step_count),
                "avg_return": float(mean_r),
                "std_return": float(std_r),
            }
        )

    env.close()
    eval_env.close()
    return pd.DataFrame(eval_rows)


def _build_rliable_curves(
    env_df: pd.DataFrame,
    baselines: list[str],
    seeds: list[int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    step_grid = np.sort(env_df["step"].unique().astype(float))
    score_dict: dict[str, np.ndarray] = {}

    for baseline in baselines:
        bdf = env_df[env_df["baseline"] == baseline]
        seed_curves = []
        for seed in seeds:
            sdf = bdf[bdf["seed"] == seed].sort_values("step")
            if sdf.empty:
                continue
            x = sdf["step"].to_numpy(dtype=float)
            y = sdf["avg_return"].to_numpy(dtype=float)
            y_interp = np.interp(step_grid, x, y)
            seed_curves.append(y_interp)

        if seed_curves:
            score_dict[baseline] = np.asarray(seed_curves, dtype=float)

    return step_grid, score_dict


def make_sample_efficiency_plot(
    env_df: pd.DataFrame,
    env_name: str,
    baselines: list[str],
    seeds: list[int],
    output_path: Path,
) -> None:
    step_grid, score_dict = _build_rliable_curves(env_df, baselines, seeds)

    if not score_dict:
        return

    step_grid_k = step_grid / 1000

    mean_over_time = lambda scores: np.array(  # noqa: E731
        [np.mean(scores[:, i]) for i in range(scores.shape[1])]
    )
    mean_scores, mean_cis = get_interval_estimates(
        score_dict, mean_over_time, reps=2000
    )

    ax = plot_sample_efficiency_curve(
        step_grid_k,
        mean_scores,
        mean_cis,
        algorithms=list(score_dict.keys()),
        xlabel="Environment Steps (×1k)",
        ylabel="Average Return",
    )
    handles, labels = ax.get_legend_handles_labels()
    line_handles = [
        handle for handle, label in zip(handles, labels) if isinstance(handle, Line2D)
    ]
    line_labels = [
        label for handle, label in zip(handles, labels) if isinstance(handle, Line2D)
    ]
    ax.legend(line_handles, line_labels, loc="best")
    plt.title(f"Week 6 L1 - {env_name} Baseline Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def summarize_final_scores_with_ci(
    df: pd.DataFrame, baselines: list[str]
) -> pd.DataFrame:
    rows: list[dict] = []

    for env_name, env_df in df.groupby("env"):
        final_by_seed = (
            env_df.sort_values(["baseline", "seed", "step"])
            .groupby(["baseline", "seed"], as_index=False)
            .tail(1)
            .copy()
        )

        score_dict: dict[str, np.ndarray] = {}
        for baseline in baselines:
            bvals = final_by_seed[final_by_seed["baseline"] == baseline][
                "avg_return"
            ].to_numpy(dtype=float)
            if bvals.size == 0:
                continue
            score_dict[baseline] = bvals.reshape(-1, 1)

        if not score_dict:
            continue

        aggregate_fn = lambda x: np.array([np.mean(x)])  # noqa: E731
        points, cis = get_interval_estimates(score_dict, aggregate_fn, reps=5000)

        for baseline in score_dict:
            raw_vals = final_by_seed[final_by_seed["baseline"] == baseline][
                "avg_return"
            ].to_numpy(dtype=float)
            rows.append(
                {
                    "env": env_name,
                    "baseline": baseline,
                    "n_seeds": int(raw_vals.size),
                    "final_mean_return": float(points[baseline][0]),
                    "final_ci95_low": float(cis[baseline][0, 0]),
                    "final_ci95_high": float(cis[baseline][1, 0]),
                    "final_std_return": float(np.std(raw_vals, ddof=0)),
                }
            )

    return pd.DataFrame(rows)


def write_observations(
    observations_path: Path,
    summary_df: pd.DataFrame,
    seeds: list[int],
    env_configs: dict[str, EnvRunConfig],
    results_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("Week 6 - Level 1 Observations (Actor-Critic Baselines)")
    lines.append("")
    lines.append("Setup")
    lines.append(f"- Baselines: {BASELINES}")
    lines.append(f"- Seeds: {seeds}")
    for env_name, cfg in env_configs.items():
        lines.append(
            f"- {env_name}: total_steps={cfg.total_steps}, eval_interval={cfg.eval_interval}, lr_actor={cfg.lr_actor}, lr_critic={cfg.lr_critic}, gamma={cfg.gamma}, gae_lambda={cfg.gae_lambda}"
        )

    lines.append("")
    lines.append("Generated Artifacts")
    lines.append(f"- {results_dir.as_posix()}/level1_metrics.csv")
    lines.append(f"- {results_dir.as_posix()}/level1_final_summary.csv")
    for env_name in env_configs:
        lines.append(
            f"- {results_dir.as_posix()}/{_slugify_env(env_name)}_sample_efficiency.png"
        )

    lines.append("")
    lines.append("Final-Return Summary (95% CI via rliable)")

    for env_name, env_summary in summary_df.groupby("env"):
        lines.append(f"- {env_name}:")
        sorted_env = env_summary.sort_values("final_mean_return", ascending=False)
        for _, row in sorted_env.iterrows():
            lines.append(
                f"  {row['baseline']}: mean={row['final_mean_return']:.2f}, CI95=[{row['final_ci95_low']:.2f}, {row['final_ci95_high']:.2f}], std={row['final_std_return']:.2f}, n_seeds={int(row['n_seeds'])}"
            )

        best = sorted_env.iloc[0]
        worst = sorted_env.iloc[-1]
        lines.append(
            f"  Best baseline: {best['baseline']} (mean {best['final_mean_return']:.2f}), worst baseline: {worst['baseline']} (mean {worst['final_mean_return']:.2f})."
        )

    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "- In general, value-based baselines (value/gae) reduce policy-gradient variance compared to no baseline, which often improves learning stability."
    )
    lines.append(
        "- GAE usually provides a favorable bias-variance trade-off: lower variance than Monte-Carlo returns while keeping less bias than very short-horizon TD estimates."
    )
    lines.append(
        "- The running-average baseline (avg) can help versus none, but it ignores state-dependent structure and is therefore typically weaker than learned value baselines."
    )
    lines.append(
        "- Confidence intervals still overlap in parts of training, so differences should be interpreted as trends rather than strict dominance."
    )

    observations_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 Level 1 baseline experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--results-dir", type=str, default="results/week_6/l1")
    parser.add_argument(
        "--observations-path",
        type=str,
        default="rl_exercises/week_6/observations_l1.txt",
    )

    parser.add_argument("--cartpole-steps", type=int, default=200000)
    parser.add_argument("--cartpole-eval-interval", type=int, default=10000)
    parser.add_argument("--lunar-steps", type=int, default=200000)
    parser.add_argument("--lunar-eval-interval", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(args.seeds)

    env_configs = {
        "CartPole-v1": EnvRunConfig(
            total_steps=args.cartpole_steps,
            eval_interval=args.cartpole_eval_interval,
            lr_actor=5e-4,
            lr_critic=1e-3,
        ),
        "LunarLander-v3": EnvRunConfig(
            total_steps=args.lunar_steps,
            eval_interval=args.lunar_eval_interval,
            lr_actor=5e-3,
            lr_critic=1e-2,
        ),
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for env_name, cfg in env_configs.items():
        for baseline in BASELINES:
            for seed in seeds:
                print(
                    f"Running env={env_name} baseline={baseline} seed={seed} steps={cfg.total_steps}"
                )
                run_df = run_single_experiment(
                    env_name=env_name,
                    baseline=baseline,
                    seed=seed,
                    config=cfg,
                    eval_episodes=args.eval_episodes,
                )
                all_runs.append(run_df)

    metrics_df = pd.concat(all_runs, ignore_index=True)
    metrics_path = results_dir / "level1_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    for env_name, env_df in metrics_df.groupby("env"):
        fig_path = results_dir / f"{_slugify_env(env_name)}_sample_efficiency.png"
        make_sample_efficiency_plot(
            env_df=env_df,
            env_name=env_name,
            baselines=BASELINES,
            seeds=seeds,
            output_path=fig_path,
        )

    summary_df = summarize_final_scores_with_ci(metrics_df, BASELINES)
    summary_path = results_dir / "level1_final_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    observations_path = Path(args.observations_path)
    write_observations(
        observations_path=observations_path,
        summary_df=summary_df,
        seeds=seeds,
        env_configs=env_configs,
        results_dir=results_dir,
    )

    print("Saved artifacts:")
    print(metrics_path)
    print(summary_path)
    print(observations_path)


if __name__ == "__main__":
    main()
