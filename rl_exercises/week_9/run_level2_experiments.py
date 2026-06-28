"""Run and persist all experiments required by Week 9 Level 2.

Example:
    python -m rl_exercises.week_9.run_level2_experiments --workers 3
"""

from __future__ import annotations

from typing import Any

import argparse
import concurrent.futures
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rl_exercises_matplotlib")

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from rl_exercises.week_9.dyna_ppo import DynaPPOAgent  # noqa: E402
from rl_exercises.week_9.run_level1_experiments import (  # noqa: E402
    collect_one_step_holdout,
    evaluate_one_step_model,
    evaluate_policy,
)


@dataclass(frozen=True)
class Level2Config:
    env_name: str = "CartPole-v1"
    total_steps: int = 15_000
    eval_interval: int = 3_000
    eval_episodes: int = 10
    seeds: tuple[int, ...] = (0, 1, 2)
    heldout_transitions: int = 1_000
    distribution_checkpoints: tuple[int, ...] = (3_000, 9_000, 15_000)


@dataclass(frozen=True)
class Variant:
    name: str
    imag_horizon: int = 5
    model_epochs: int = 3
    imag_batches: int = 10
    max_buffer_size: int = 10_000
    model_noise_std: float = 0.0


def build_variants() -> dict[str, Variant]:
    base = Variant(name="base")
    variants = {base.name: base}
    for horizon in (1, 3, 10, 20):
        variant = replace(base, name=f"horizon_{horizon}", imag_horizon=horizon)
        variants[variant.name] = variant
    variants["regime_conservative"] = replace(
        base, name="regime_conservative", model_epochs=1, imag_batches=5
    )
    variants["regime_aggressive"] = replace(
        base, name="regime_aggressive", model_epochs=5, imag_batches=20
    )
    for size in (1_000, 5_000, 50_000):
        variant = replace(base, name=f"buffer_{size}", max_buffer_size=size)
        variants[variant.name] = variant
    for sigma in (0.01, 0.05, 0.1, 0.2):
        variant = replace(base, name=f"noise_{sigma}", model_noise_std=sigma)
        variants[variant.name] = variant
    return variants


def make_agent(env: gym.Env, seed: int, variant: Variant) -> DynaPPOAgent:
    return DynaPPOAgent(
        env,
        use_model=True,
        lr_actor=5e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=4,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
        seed=seed,
        hidden_size=128,
        model_lr=5e-4,
        model_epochs=variant.model_epochs,
        model_batch_size=128,
        imag_horizon=variant.imag_horizon,
        imag_batches=variant.imag_batches,
        max_buffer_size=variant.max_buffer_size,
        model_noise_std=variant.model_noise_std,
    )


def run_variant(
    task: tuple[Level2Config, Variant, int],
) -> tuple[str, int, dict[str, list[dict[str, Any]]]]:
    config, variant, seed = task
    torch.set_num_threads(1)
    env = gym.make(config.env_name)
    agent = make_agent(env, seed, variant)
    state, _ = env.reset()
    episode_done = False
    episode = 0
    next_evaluation = config.eval_interval
    old_transitions = None

    results: dict[str, list[dict[str, Any]]] = {
        "returns": [],
        "final_model_accuracy": [],
        "distribution_errors": [],
        "state_visits": [],
    }

    while agent.real_steps < config.total_steps:
        trajectory = []
        while (
            not episode_done
            and agent.real_steps < config.total_steps
            and agent.real_steps < next_evaluation
        ):
            action, logp, entropy, _ = agent.predict(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_done = terminated or truncated
            trajectory.append(
                (
                    state,
                    action,
                    logp,
                    entropy,
                    float(reward),
                    float(episode_done),
                    next_state,
                )
            )
            state = next_state
            agent.real_steps += 1

        agent.update(trajectory)
        agent.store_real(trajectory)
        agent.train_model()
        agent.imagine_and_update()

        if episode_done:
            episode += 1
            agent.total_episodes += 1
            state, _ = env.reset()
            episode_done = False

        if agent.real_steps == next_evaluation:
            evaluation_returns = evaluate_policy(
                agent,
                config.env_name,
                seed,
                next_evaluation,
                config.eval_episodes,
            )
            for eval_episode, episode_return in enumerate(evaluation_returns):
                results["returns"].append(
                    {
                        "variant": variant.name,
                        "seed": seed,
                        "real_steps": next_evaluation,
                        "eval_episode": eval_episode,
                        "return": episode_return,
                    }
                )

            fresh_transitions = None
            fresh_metrics = None
            if variant.name == "base" and (
                next_evaluation in config.distribution_checkpoints
            ):
                fresh_transitions = collect_one_step_holdout(
                    agent,
                    config.env_name,
                    seed,
                    next_evaluation,
                    config.heldout_transitions,
                )
                fresh_metrics = evaluate_one_step_model(agent, fresh_transitions)
                if old_transitions is None:
                    old_transitions = fresh_transitions

                for dataset, transitions in (
                    ("old", old_transitions),
                    ("new", fresh_transitions),
                ):
                    metrics = evaluate_one_step_model(agent, transitions)
                    results["distribution_errors"].append(
                        {
                            "seed": seed,
                            "real_steps": next_evaluation,
                            "dataset": dataset,
                            "num_transitions": len(transitions),
                            **metrics,
                        }
                    )

                for sample_index, transition in enumerate(fresh_transitions):
                    visit = {
                        "seed": seed,
                        "real_steps": next_evaluation,
                        "sample_index": sample_index,
                    }
                    for dimension, value in enumerate(transition[0]):
                        visit[f"state_{dimension}"] = float(value)
                    results["state_visits"].append(visit)

            if next_evaluation == config.total_steps:
                if fresh_transitions is None:
                    fresh_transitions = collect_one_step_holdout(
                        agent,
                        config.env_name,
                        seed,
                        next_evaluation,
                        config.heldout_transitions,
                    )
                    fresh_metrics = evaluate_one_step_model(agent, fresh_transitions)
                results["final_model_accuracy"].append(
                    {
                        "variant": variant.name,
                        "seed": seed,
                        "real_steps": next_evaluation,
                        "num_transitions": len(fresh_transitions),
                        **fresh_metrics,
                    }
                )

            print(
                f"variant={variant.name:<20} seed={seed} "
                f"steps={next_evaluation:5d} "
                f"return={np.mean(evaluation_returns):6.1f}",
                flush=True,
            )
            next_evaluation += config.eval_interval

    env.close()
    return variant.name, seed, results


def seed_return_means(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.groupby(["variant", "seed", "real_steps"], as_index=False)[
        "return"
    ].mean()


def aggregate(
    frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]
) -> pd.DataFrame:
    summary = frame.groupby(group_columns)[value_columns].agg(["mean", "std", "sem"])
    summary.columns = [f"{value}_{stat}" for value, stat in summary.columns]
    return summary.reset_index()


def make_summaries(
    config: Level2Config,
    raw: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    seed_returns = seed_return_means(raw["returns"])
    final_returns = seed_returns[seed_returns.real_steps == config.total_steps]

    horizon_names = {
        "horizon_1": 1,
        "horizon_3": 3,
        "base": 5,
        "horizon_10": 10,
        "horizon_20": 20,
    }
    horizon = final_returns[final_returns.variant.isin(horizon_names)].copy()
    horizon["imag_horizon"] = horizon.variant.map(horizon_names)
    horizon_summary = aggregate(horizon, ["imag_horizon"], ["return"])

    regime_names = {
        "regime_conservative": "Conservative",
        "base": "Balanced",
        "regime_aggressive": "Aggressive",
    }
    regime = seed_returns[seed_returns.variant.isin(regime_names)].copy()
    regime["regime"] = regime.variant.map(regime_names)
    regime_summary = aggregate(regime, ["regime", "real_steps"], ["return"])

    buffer_names = {
        "buffer_1000": 1_000,
        "buffer_5000": 5_000,
        "base": 10_000,
        "buffer_50000": 50_000,
    }
    buffer_returns = final_returns[final_returns.variant.isin(buffer_names)].copy()
    buffer_returns["max_buffer_size"] = buffer_returns.variant.map(buffer_names)
    final_model = raw["final_model_accuracy"]
    buffer_model = final_model[final_model.variant.isin(buffer_names)].copy()
    buffer_model["max_buffer_size"] = buffer_model.variant.map(buffer_names)
    buffer_seed = buffer_returns.merge(
        buffer_model[["variant", "seed", "state_mse"]],
        on=["variant", "seed"],
        validate="one_to_one",
    )
    buffer_summary = aggregate(
        buffer_seed, ["max_buffer_size"], ["return", "state_mse"]
    )

    noise_names = {
        "base": 0.0,
        "noise_0.01": 0.01,
        "noise_0.05": 0.05,
        "noise_0.1": 0.1,
        "noise_0.2": 0.2,
    }
    noise = final_returns[final_returns.variant.isin(noise_names)].copy()
    noise["sigma"] = noise.variant.map(noise_names)
    noise_summary = aggregate(noise, ["sigma"], ["return"])

    distribution_summary = aggregate(
        raw["distribution_errors"],
        ["real_steps", "dataset"],
        ["state_mse", "reward_mse"],
    )
    histogram_summary = build_histogram_summary(raw["state_visits"])
    state_shift_summary = build_state_shift_summary(
        raw["state_visits"], histogram_summary
    )

    return {
        "horizon": horizon_summary,
        "regime": regime_summary,
        "buffer": buffer_summary,
        "distribution_shift": distribution_summary,
        "state_histograms": histogram_summary,
        "state_shift": state_shift_summary,
        "noise": noise_summary,
    }


def build_histogram_summary(state_visits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    state_columns = [f"state_{index}" for index in range(4)]
    for dimension, column in enumerate(state_columns):
        values = state_visits[column].to_numpy()
        lower, upper = np.quantile(values, (0.005, 0.995))
        edges = np.linspace(lower, upper, 31)
        for real_steps, group in state_visits.groupby("real_steps"):
            counts, _ = np.histogram(group[column], bins=edges)
            density = counts / counts.sum() / np.diff(edges)
            for index, count in enumerate(counts):
                rows.append(
                    {
                        "real_steps": int(real_steps),
                        "dimension": dimension,
                        "bin_left": edges[index],
                        "bin_right": edges[index + 1],
                        "count": int(count),
                        "density": density[index],
                    }
                )
    return pd.DataFrame(rows)


def build_state_shift_summary(
    state_visits: pd.DataFrame, histograms: pd.DataFrame
) -> pd.DataFrame:
    early_step = int(state_visits.real_steps.min())
    rows = []
    for dimension in range(4):
        dimension_histograms = histograms[histograms.dimension == dimension]
        early = dimension_histograms[
            dimension_histograms.real_steps == early_step
        ].sort_values("bin_left")
        early_probability = early["count"].to_numpy() / early["count"].sum()
        for real_steps, histogram in dimension_histograms.groupby("real_steps"):
            ordered = histogram.sort_values("bin_left")
            probability = ordered["count"].to_numpy() / ordered["count"].sum()
            values = state_visits[state_visits.real_steps == real_steps][
                f"state_{dimension}"
            ]
            rows.append(
                {
                    "real_steps": int(real_steps),
                    "dimension": dimension,
                    "tv_from_early": 0.5
                    * float(np.abs(probability - early_probability).sum()),
                    "state_mean": values.mean(),
                    "state_std": values.std(ddof=0),
                    "num_states": len(values),
                }
            )
    return pd.DataFrame(rows)


def plot_horizon(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("imag_horizon")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        ordered.imag_horizon,
        ordered.return_mean,
        yerr=1.96 * ordered.return_sem,
        marker="o",
        capsize=4,
    )
    ax.set(
        title="Final Return vs. Imagination Horizon",
        xlabel="Imagination horizon",
        ylabel="Final average return",
        xticks=[1, 3, 5, 10, 20],
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "l2_final_return_vs_horizon.png", dpi=180)
    plt.close(fig)


def plot_regimes(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime, group in summary.groupby("regime"):
        ordered = group.sort_values("real_steps")
        x = ordered.real_steps.to_numpy()
        y = ordered.return_mean.to_numpy()
        ci = 1.96 * ordered.return_sem.to_numpy()
        ax.plot(x, y, marker="o", label=regime)
        ax.fill_between(x, y - ci, y + ci, alpha=0.15)
    ax.set(
        title="Model/Imagination Budget",
        xlabel="Real environment steps",
        ylabel="Average evaluation return",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "l2_regime_return_vs_real_steps.png", dpi=180)
    plt.close(fig)


def plot_buffer(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("max_buffer_size")
    fig, return_axis = plt.subplots(figsize=(8, 5))
    mse_axis = return_axis.twinx()
    return_axis.errorbar(
        ordered.max_buffer_size,
        ordered.return_mean,
        yerr=1.96 * ordered.return_sem,
        color="tab:blue",
        marker="o",
        capsize=4,
        label="Final return",
    )
    mse_axis.errorbar(
        ordered.max_buffer_size,
        ordered.state_mse_mean,
        yerr=1.96 * ordered.state_mse_sem,
        color="tab:red",
        marker="s",
        capsize=4,
        label="Held-out state MSE",
    )
    return_axis.set_xscale("log")
    mse_axis.set_yscale("log")
    return_axis.set(
        title="Replay-buffer Size Ablation",
        xlabel="Maximum replay-buffer size (log scale)",
        ylabel="Final average return",
    )
    mse_axis.set_ylabel("Held-out state MSE (log scale)")
    handles_a, labels_a = return_axis.get_legend_handles_labels()
    handles_b, labels_b = mse_axis.get_legend_handles_labels()
    return_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
    return_axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "l2_buffer_size_return_and_mse.png", dpi=180)
    plt.close(fig)


def plot_state_histograms(summary: pd.DataFrame, output_dir: Path) -> None:
    names = ["Cart position", "Cart velocity", "Pole angle", "Pole angular velocity"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for dimension, axis in enumerate(axes.flat):
        data = summary[summary.dimension == dimension]
        for real_steps, group in data.groupby("real_steps"):
            centers = (group.bin_left.to_numpy() + group.bin_right.to_numpy()) / 2.0
            axis.plot(centers, group.density, label=f"{real_steps:,} steps")
        axis.set_title(names[dimension])
        axis.set_xlabel("State value")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("State-visit Histograms: Early, Mid, and Late Training")
    fig.tight_layout()
    fig.savefig(output_dir / "l2_state_visit_histograms.png", dpi=180)
    plt.close(fig)


def plot_distribution_shift(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for dataset, group in summary.groupby("dataset"):
        ordered = group.sort_values("real_steps")
        ax.errorbar(
            ordered.real_steps,
            ordered.state_mse_mean,
            yerr=1.96 * ordered.state_mse_sem,
            marker="o",
            capsize=4,
            label=dataset.capitalize(),
        )
    ax.set_yscale("log")
    ax.set(
        title="Dynamics Error on Old vs. New States",
        xlabel="Real environment steps",
        ylabel="Held-out state MSE (log scale)",
        xticks=[3_000, 9_000, 15_000],
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "l2_old_vs_new_model_error.png", dpi=180)
    plt.close(fig)


def plot_noise(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("sigma")
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        positions,
        ordered.return_mean,
        yerr=1.96 * ordered.return_sem,
        marker="o",
        capsize=4,
    )
    ax.set(
        title="Failure Mode: Corrupted Model Outputs",
        xlabel="Gaussian noise standard deviation",
        ylabel="Final average return",
        xticks=positions,
        xticklabels=[f"{sigma:.2f}" for sigma in ordered.sigma],
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "l2_final_return_vs_noise.png", dpi=180)
    plt.close(fig)


def save_results(
    config: Level2Config,
    variants: dict[str, Variant],
    all_records: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "l2_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = {name: pd.DataFrame(records) for name, records in all_records.items()}
    for name, frame in raw.items():
        frame.to_csv(raw_dir / f"{name}.csv", index=False)

    summaries = make_summaries(config, raw)
    for name, frame in summaries.items():
        frame.to_csv(output_dir / f"l2_{name}_summary.csv", index=False)

    payload = {
        "config": asdict(config),
        "variants": {name: asdict(variant) for name, variant in variants.items()},
        "raw": {name: frame.to_dict(orient="records") for name, frame in raw.items()},
        "summary": {
            name: frame.to_dict(orient="records") for name, frame in summaries.items()
        },
    }
    (output_dir / "l2_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    plot_horizon(summaries["horizon"], output_dir)
    plot_regimes(summaries["regime"], output_dir)
    plot_buffer(summaries["buffer"], output_dir)
    plot_state_histograms(summaries["state_histograms"], output_dir)
    plot_distribution_shift(summaries["distribution_shift"], output_dir)
    plot_noise(summaries["noise"], output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[2] / "results" / "week_9",
    )
    parser.add_argument("--total-steps", type=int, default=15_000)
    parser.add_argument("--eval-interval", type=int, default=3_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_steps % args.eval_interval != 0:
        raise ValueError("total_steps must be divisible by eval_interval")
    checkpoints = (
        args.eval_interval,
        args.total_steps // 2 + args.eval_interval // 2,
        args.total_steps,
    )
    config = Level2Config(
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seeds=tuple(args.seeds),
        distribution_checkpoints=checkpoints,
    )
    variants = build_variants()
    tasks = [
        (config, variant, seed)
        for variant in variants.values()
        for seed in config.seeds
    ]
    all_records: dict[str, list[dict[str, Any]]] = {
        "returns": [],
        "final_model_accuracy": [],
        "distribution_errors": [],
        "state_visits": [],
    }

    if args.workers == 1:
        completed_runs = (run_variant(task) for task in tasks)
        for _, _, records in completed_runs:
            for name, rows in records.items():
                all_records[name].extend(rows)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [executor.submit(run_variant, task) for task in tasks]
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                variant_name, seed, records = future.result()
                for name, rows in records.items():
                    all_records[name].extend(rows)
                print(
                    f"completed={completed}/{len(tasks)} "
                    f"variant={variant_name} seed={seed}",
                    flush=True,
                )

    save_results(config, variants, all_records, args.output_dir)
    print(f"Saved Level 2 evidence to {args.output_dir}")


if __name__ == "__main__":
    main()
