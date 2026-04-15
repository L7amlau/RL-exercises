import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

data_path = (
    pathlib.Path(__file__).parent.resolve() / "../../results/random_agent/Pendulum-v1/"
)
n_seeds = 5

# Read data from different runs and add seeds as information
for seed in range(n_seeds):
    df = pd.read_csv(data_path / f"seed_{seed}" / f"{seed}" / "train_rewards.csv")
    eval_df = pd.read_csv(data_path / f"seed_{seed}" / f"{seed}" / "eval_rewards.csv")
    df["seed"] = seed
    eval_df["seed"] = seed
    if seed == 0:
        all_df = df
        all_eval_df = eval_df
    else:
        all_df = pd.concat([all_df, df], ignore_index=True)
        all_eval_df = pd.concat([all_eval_df, eval_df], ignore_index=True)

# Make sure only one set of steps is attempted to be plotted
# Obviously the steps should match in such cases!
steps = all_df["steps"].to_numpy().reshape((n_seeds, -1))[0]
train_scores = {
    "random_agent": all_df["train_rewards"].to_numpy().reshape((n_seeds, -1))
}
eval_steps = all_eval_df["eval_steps"].to_numpy().reshape((n_seeds, -1))[0]
eval_scores = {
    "random_agent": all_eval_df["eval_rewards"].to_numpy().reshape((n_seeds, -1))
}

# This aggregates only IQM, but other options include mean and median
# Optimality gap exists, but you obviously need optimal scores for that
# If you want to use it, check their code
iqm = lambda scores: np.array(  # noqa: E731
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)

# We can use the iqm definition for both scores
iqm_scores, iqm_cis = get_interval_estimates(
    train_scores,
    iqm,
    reps=5,
)
eval_iqm_scores, eval_iqm_cis = get_interval_estimates(
    eval_scores,
    iqm,
    reps=5,
)

# This is a utility function, but you can also just use a normal line plot with the IQM and CI scores
_ = plot_sample_efficiency_curve(
    steps + 1,
    iqm_scores,
    iqm_cis,
    algorithms=["random_agent"],
    xlabel=r"Number of Evaluations",
    ylabel="IQM Normalized Score",
)
plt.gcf().canvas.manager.set_window_title(
    "IQM Normalized Score - Sample Efficiency Curve"
)
plt.legend()
plt.tight_layout()
# save figure to file
plt.savefig(pathlib.Path(__file__).parent.resolve() / "agent_training_curve.png")
plt.close()

_ = plot_sample_efficiency_curve(
    eval_steps + 1,
    eval_iqm_scores,
    eval_iqm_cis,
    algorithms=["random_agent"],
    xlabel=r"Number of Evaluations",
    ylabel="IQM Normalized Evaluation Score",
)
plt.gcf().canvas.manager.set_window_title(
    "IQM Normalized Score - Sample Efficiency Curve (Evaluation)"
)
plt.legend()
plt.tight_layout()
# save figure to file
plt.savefig(pathlib.Path(__file__).parent.resolve() / "agent_evaluation_curve.png")
plt.close()
