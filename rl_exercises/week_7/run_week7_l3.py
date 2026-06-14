import subprocess

AGENTS = [
    "rl_exercises.week_4.dqn",
    "rl_exercises.week_7.rnd_dqn",
    "rl_exercises.week_7.ensemble_dqn",
]

SEEDS = [0, 1, 2]


def main() -> None:
    for seed in SEEDS:
        for agent in AGENTS:
            command = [
                "python",
                "-m",
                agent,
                f"seed={seed}",
            ]

            print("=" * 80)
            print("Running:", " ".join(command))
            print("=" * 80)

            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
