
import argparse
import logging
import os
import sys


_METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE_ROOT = os.path.dirname(_METHOD_DIR)
for _p in (_STAGE_ROOT, _METHOD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C  # noqa: E402
from build_tables import (  # noqa: E402
    build_methods_table,
    build_table1,
    build_table2,
    build_table3,
)
from utils import load_groups, write_workbook  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("07_01_cohort_description")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cohort description tables (07_01).")
    p.add_argument("--cohort1", default=C.COHORT1_TABLE)
    p.add_argument("--cohort2", default=C.COHORT2_TABLE)
    p.add_argument("--output-dir", default=C.OUTPUT_DIR)
    p.add_argument("--workbook-name", default=C.OUTPUT_WORKBOOK_NAME)
    p.add_argument(
        "--continuous-test", choices=["auto", "anova", "kruskal"],
        default=C.CONTINUOUS_TEST,
    )
    p.add_argument(
        "--categorical-test", choices=["auto", "chi2", "fisher_mc"],
        default=C.CATEGORICAL_TEST,
    )
    p.add_argument(
        "--composite-pvalue", action=argparse.BooleanOptionalAction,
        default=C.COMPUTE_COMPOSITE_PVALUE,
        help="Compute the composite-sarcopenia p-value for Table 1 "
        "(default on; use --no-composite-pvalue to leave the cell blank).",
    )
    p.add_argument(
        "--no-methods-sheet", action="store_true",
        help="Do not append the 'Statistical methods' sheet.",
    )
    return p.parse_args(argv)


def _make_run_config(args: argparse.Namespace) -> C.RunConfig:
    cfg = C.RunConfig()
    cfg.cohort1_table = args.cohort1
    cfg.cohort2_table = args.cohort2
    cfg.output_dir = args.output_dir
    cfg.workbook_name = args.workbook_name
    cfg.continuous_test = args.continuous_test
    cfg.categorical_test = args.categorical_test
    cfg.compute_composite_pvalue = args.composite_pvalue
    cfg.write_methods_sheet = not args.no_methods_sheet
    return cfg


def run(cfg: C.RunConfig) -> str:
    groups = load_groups(cfg, C.COLUMNS, C.GROUP_CONST)

    table1, methods = build_table1(groups, cfg)
    table2 = build_table2(groups, cfg)
    table3 = build_table3(groups, cfg)
    tables = [table1, table2, table3]

    methods_model = None
    if cfg.write_methods_sheet:
        methods_model = build_methods_table(methods, cfg)

    out = write_workbook(tables, cfg.workbook_path, methods_model)
    logger.info("Workbook written to %s", out)
    return out


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = _make_run_config(args)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
