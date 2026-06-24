import shutil
import subprocess
from pathlib import Path

SEEDS = list(range(5))
NUM_FRAMES = 20000
ENV_NAME = "CartPole-v1"

OUT_DIR = Path("results/week_8/l2_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_latest_csv(seed: int) -> Path:
    files = list(Path("outputs").glob(f"*/*/rnd_dqn_seed{seed}.csv"))
    if not files:
        raise FileNotFoundError(f"No rnd_dqn_seed{seed}.csv found under outputs/")
    return max(files, key=lambda p: p.stat().st_mtime)


for seed in SEEDS:
    dst = OUT_DIR / f"rnd_dqn_seed{seed}.csv"

    if dst.exists():
        print(f"Skipping seed={seed}, already exists: {dst}")
        continue

    print(f"\n=== Running RND-DQN seed={seed} ===")
    subprocess.run(
        [
            "python",
            "-m",
            "rl_exercises.week_7.rnd_dqn",
            f"env.name={ENV_NAME}",
            f"seed={seed}",
            f"train.num_frames={NUM_FRAMES}",
        ],
        check=True,
    )

    src = find_latest_csv(seed)
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")
