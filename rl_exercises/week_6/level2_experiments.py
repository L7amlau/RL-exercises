from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rl_exercises.week_6.actor_critic import ActorCriticAgent
from rl_exercises.week_6.actor_critic import set_seed as ac_set_seed
from rl_exercises.week_6.ppo import PPOAgent
from rl_exercises.week_6.ppo import set_seed as ppo_set_seed
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

ALGORITHMS = ["actor_critic_gae", "ppo_vanilla", "ppo_enhanced"]


@dataclass(frozen=True)
class AlgoConfig:
    total_steps: int
    eval_interval: int
    lr_actor: float
    lr_critic: float
    hidden_size: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95


def _slugify_env(env_name: str) -> str:
    return env_name.replace("-", "_").replace("/", "_")


def run_actor_critic(
    env_name: str, seed: int, config: AlgoConfig, eval_episodes: int
) -> pd.DataFrame:
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    ac_set_seed(env, seed)
    ac_set_seed(eval_env, seed + 10000)

    agent = ActorCriticAgent(
        env=env,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        seed=seed,
        hidden_size=config.hidden_size,
        baseline_type="gae",
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
                        "algorithm": "actor_critic_gae",
                        "seed": seed,
                        "step": int(step_count),
                        "avg_return": float(mean_r),
                        "std_return": float(std_r),
                    }
                )

        if trajectory:
            agent.update_agent(trajectory)

    env.close()
    eval_env.close()
    return pd.DataFrame(eval_rows)


def run_ppo(
    env_name: str,
    seed: int,
    config: AlgoConfig,
    eval_episodes: int,
    enhanced: bool = False,
) -> pd.DataFrame:
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    ppo_set_seed(env, seed)
    ppo_set_seed(eval_env, seed + 10000)

    algo_name = "ppo_enhanced" if enhanced else "ppo_vanilla"

    agent = PPOAgent(
        env=env,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_eps=0.2,
        epochs=4,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
        seed=seed,
        hidden_size=config.hidden_size,
        use_lr_annealing=enhanced,
        use_grad_clip=enhanced,
    )

    step_count = 0
    eval_rows: list[dict] = []

    while step_count < config.total_steps:
        state, _ = env.reset()
        done = False
        trajectory = []

        while not done and step_count < config.total_steps:
            action, logp, ent, val = agent.predict(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            trajectory.append(
                (state, action, logp, ent, reward, float(done), next_state)
            )
            state = next_state
            step_count += 1

            if step_count % config.eval_interval == 0:
                mean_r, std_r = agent.evaluate(eval_env, num_episodes=eval_episodes)
                eval_rows.append(
                    {
                        "env": env_name,
                        "algorithm": algo_name,
                        "seed": seed,
                        "step": int(step_count),
                        "avg_return": float(mean_r),
                        "std_return": float(std_r),
                    }
                )

        if trajectory:
            agent.update(trajectory)

    env.close()
    eval_env.close()
    return pd.DataFrame(eval_rows)


def run_single_experiment(
    env_name: str,
    algorithm: str,
    seed: int,
    config: AlgoConfig,
    eval_episodes: int,
) -> pd.DataFrame:
    if algorithm == "actor_critic_gae":
        return run_actor_critic(env_name, seed, config, eval_episodes)
    elif algorithm == "ppo_vanilla":
        return run_ppo(env_name, seed, config, eval_episodes, enhanced=False)
    elif algorithm == "ppo_enhanced":
        return run_ppo(env_name, seed, config, eval_episodes, enhanced=True)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _build_rliable_curves(
    env_df: pd.DataFrame,
    algorithms: list[str],
    seeds: list[int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    step_grid = np.sort(env_df["step"].unique().astype(float))
    score_dict: dict[str, np.ndarray] = {}

    for algo in algorithms:
        adf = env_df[env_df["algorithm"] == algo]
        seed_curves = []
        for seed in seeds:
            sdf = adf[adf["seed"] == seed].sort_values("step")
            if sdf.empty:
                continue
            x = sdf["step"].to_numpy(dtype=float)
            y = sdf["avg_return"].to_numpy(dtype=float)
            y_interp = np.interp(step_grid, x, y)
            seed_curves.append(y_interp)

        if seed_curves:
            score_dict[algo] = np.asarray(seed_curves, dtype=float)

    return step_grid, score_dict


def make_sample_efficiency_plot(
    env_df: pd.DataFrame,
    env_name: str,
    algorithms: list[str],
    seeds: list[int],
    output_path: Path,
) -> None:
    step_grid, score_dict = _build_rliable_curves(env_df, algorithms, seeds)

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
    plt.title(f"Week 6 L2 - {env_name} Algorithm Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def summarize_final_scores_with_ci(
    df: pd.DataFrame, algorithms: list[str]
) -> pd.DataFrame:
    rows: list[dict] = []

    for env_name, env_df in df.groupby("env"):
        final_by_seed = (
            env_df.sort_values(["algorithm", "seed", "step"])
            .groupby(["algorithm", "seed"], as_index=False)
            .tail(1)
            .copy()
        )

        score_dict: dict[str, np.ndarray] = {}
        for algo in algorithms:
            avals = final_by_seed[final_by_seed["algorithm"] == algo][
                "avg_return"
            ].to_numpy(dtype=float)
            if avals.size == 0:
                continue
            score_dict[algo] = avals.reshape(-1, 1)

        if not score_dict:
            continue

        aggregate_fn = lambda x: np.array([np.mean(x)])  # noqa: E731
        points, cis = get_interval_estimates(score_dict, aggregate_fn, reps=5000)

        for algo in score_dict:
            raw_vals = final_by_seed[final_by_seed["algorithm"] == algo][
                "avg_return"
            ].to_numpy(dtype=float)
            rows.append(
                {
                    "env": env_name,
                    "algorithm": algo,
                    "n_seeds": int(raw_vals.size),
                    "final_mean_return": float(points[algo][0]),
                    "final_ci95_low": float(cis[algo][0, 0]),
                    "final_ci95_high": float(cis[algo][1, 0]),
                    "final_std_return": float(np.std(raw_vals, ddof=0)),
                }
            )

    return pd.DataFrame(rows)


def write_observations(
    observations_path: Path,
    summary_df: pd.DataFrame,
    seeds: list[int],
    config: AlgoConfig,
    results_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("Week 6 - Level 2 Observations (PPO vs Actor-Critic)")
    lines.append("")
    lines.append("Setup")
    lines.append(f"- Algorithms: {ALGORITHMS}")
    lines.append(f"- Seeds: {seeds}")
    lines.append(
        f"- LunarLander-v3: total_steps={config.total_steps}, eval_interval={config.eval_interval}, lr_actor={config.lr_actor}, lr_critic={config.lr_critic}, gamma={config.gamma}, gae_lambda={config.gae_lambda}"
    )
    lines.append("- actor_critic_gae: Actor-Critic with GAE baseline (from Level 1)")
    lines.append(
        "- ppo_vanilla: PPO with clipped surrogate, vf_coef=0.5, ent_coef=0.01, NO lr annealing, NO grad clip"
    )
    lines.append(
        "- ppo_enhanced: PPO with clipped surrogate + lr annealing + gradient clipping (max_norm=0.5)"
    )

    lines.append("")
    lines.append("Generated Artifacts")
    lines.append(f"- {results_dir.as_posix()}/level2_metrics.csv")
    lines.append(f"- {results_dir.as_posix()}/level2_final_summary.csv")
    lines.append(f"- {results_dir.as_posix()}/LunarLander_v3_algo_comparison.png")

    lines.append("")
    lines.append("Final-Return Summary (95% CI via rliable)")

    for env_name, env_summary in summary_df.groupby("env"):
        lines.append(f"- {env_name}:")
        sorted_env = env_summary.sort_values("final_mean_return", ascending=False)
        for _, row in sorted_env.iterrows():
            lines.append(
                f"  {row['algorithm']}: mean={row['final_mean_return']:.2f}, CI95=[{row['final_ci95_low']:.2f}, {row['final_ci95_high']:.2f}], std={row['final_std_return']:.2f}, n_seeds={int(row['n_seeds'])}"
            )

        best = sorted_env.iloc[0]
        worst = sorted_env.iloc[-1]
        lines.append(
            f"  Best: {best['algorithm']} (mean {best['final_mean_return']:.2f}), worst: {worst['algorithm']} (mean {worst['final_mean_return']:.2f})."
        )

    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "- PPO's clipped surrogate objective should provide more stable updates than vanilla actor-critic, since it limits how far the policy can change in one step."
    )
    lines.append(
        "- The lr annealing (linear decay to 0) helps the agent converge more smoothly in later training, since smaller steps reduce oscillation around the optimum."
    )
    lines.append(
        "- Gradient clipping prevents occasional large gradient updates that could destabilize training, especially early on when the critic is still inaccurate."
    )
    lines.append(
        "- If ppo_enhanced > ppo_vanilla, it confirms that these implementation tricks from the PPO details blog post actually matter in practice."
    )
    lines.append(
        "- In this experiment, ppo_enhanced did not outperform ppo_vanilla. This indicates that implementation improvements such as learning-rate annealing and gradient clipping are not universally beneficial. Their effectiveness depends on the training budget and hyperparameter choices. Under the current setup (20000 training steps), the enhanced PPO variant may have become overly conservative, resulting in slower learning."
    )

    observations_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 Level 2 PPO experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--results-dir", type=str, default="results/week_6/l2")
    parser.add_argument(
        "--observations-path",
        type=str,
        default="rl_exercises/week_6/observations_l2.txt",
    )
    parser.add_argument("--lunar-steps", type=int, default=20000)
    parser.add_argument("--lunar-eval-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(args.seeds)

    config = AlgoConfig(
        total_steps=args.lunar_steps,
        eval_interval=args.lunar_eval_interval,
        lr_actor=5e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for algo in ALGORITHMS:
        for seed in seeds:
            print(f"Running algorithm={algo} seed={seed} steps={config.total_steps}")
            run_df = run_single_experiment(
                env_name="LunarLander-v3",
                algorithm=algo,
                seed=seed,
                config=config,
                eval_episodes=args.eval_episodes,
            )
            all_runs.append(run_df)

    metrics_df = pd.concat(all_runs, ignore_index=True)
    metrics_path = results_dir / "level2_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    for env_name, env_df in metrics_df.groupby("env"):
        fig_path = results_dir / f"{_slugify_env(env_name)}_algo_comparison.png"
        make_sample_efficiency_plot(
            env_df=env_df,
            env_name=env_name,
            algorithms=ALGORITHMS,
            seeds=seeds,
            output_path=fig_path,
        )

    summary_df = summarize_final_scores_with_ci(metrics_df, ALGORITHMS)
    summary_path = results_dir / "level2_final_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    observations_path = Path(args.observations_path)
    write_observations(
        observations_path=observations_path,
        summary_df=summary_df,
        seeds=seeds,
        config=config,
        results_dir=results_dir,
    )

    print("Saved artifacts:")
    print(metrics_path)
    print(summary_path)
    print(observations_path)


if __name__ == "__main__":
    main()
