import argparse
from config import RCFG, TASKS
from run_task import run_task


def main(tasks=None):
    for task in (tasks or list(TASKS.keys())):
        run_task(task=task, run_cfg=RCFG, save_outputs=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()))
    args = parser.parse_args()
    main(tasks=args.tasks)
