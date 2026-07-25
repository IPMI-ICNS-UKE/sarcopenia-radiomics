import argparse

from config import RCFG, TASKS
from run_task import run_task


def main(tasks=None) -> None:
    for task in tasks or list(TASKS.keys()):
        run_task(task=task, run_cfg=RCFG, save_outputs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 06_03_deep_radiomics pipeline.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASKS.keys()),
        help="Tasks to run. Runs all if omitted.",
    )
    args = parser.parse_args()
    main(tasks=args.tasks)
