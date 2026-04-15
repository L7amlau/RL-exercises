import gymnasium
import hydra
import numpy as np
from omegaconf import DictConfig
from rich import print as printr
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


# Use yaml config ../configs/bipedal_walker.yaml
@hydra.main("../configs", "bipedal_walker", version_base="1.1")  # type: ignore[misc]
def main(cfg: DictConfig) -> float:
    """Run Training"""
    printr(cfg)

    # Create environment
    env = gymnasium.make(cfg.env_id)

    # Create agent
    model = SAC(
        "MlpPolicy",
        env,
        verbose=cfg.verbose,
        tensorboard_log=cfg.log_dir,
        seed=cfg.seed,
    )

    # Train agent
    model.learn(total_timesteps=cfg.total_timesteps)

    # Save agent
    model.save(cfg.model_fn)

    # Evaluate
    env = Monitor(gymnasium.make(cfg.env_id))
    means, stds = evaluate_policy(model, env, n_eval_episodes=cfg.n_eval_episodes)
    performance = float(np.mean(means))

    return performance


if __name__ == "__main__":
    main()
