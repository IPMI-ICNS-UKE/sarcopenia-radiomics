from pathlib import Path
import pandas as pd
from sct_data_loader import DataLoader
from sct_statistics import build_data_description_table_three_groups


COHORT1_TABLE_PATH = Path(
    "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/table/ground_truth.xlsx"
)
COHORT2_TABLE_PATH = Path(
    "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/table/ground_truth.xlsx"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "01_data_statistics"
OUTPUT_XLSX = OUTPUT_DIR / "data_description_statistics.xlsx"


def build_dataset_overview_sheet(
    train_df: pd.DataFrame,
    test_c1_df: pd.DataFrame,
    test_c2_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"Dataset": "Cohort 1 - Training", "N": len(train_df)},
        {"Dataset": "Cohort 1 - Internal testing", "N": len(test_c1_df)},
        {"Dataset": "Cohort 2 - Additional internal testing", "N": len(test_c2_df)},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    loader = DataLoader(
        cohort1_path=COHORT1_TABLE_PATH,
        cohort2_path=COHORT2_TABLE_PATH,
    )

    train_df, test_c1_df = loader.load_cohort1()
    test_c2_df = loader.load_cohort2()

    if test_c2_df is None:
        raise FileNotFoundError(
            f"Cohort 2 table is required for this stage but was not found: {COHORT2_TABLE_PATH}"
        )

    overview_df = build_dataset_overview_sheet(
        train_df=train_df,
        test_c1_df=test_c1_df,
        test_c2_df=test_c2_df,
    )

    stats_df = build_data_description_table_three_groups(
        df_group1=train_df,
        df_group2=test_c1_df,
        df_group3=test_c2_df,
        group1_name="Training (C1)",
        group2_name="Internal testing (C1)",
        group3_name="Additional internal testing (C2)",
        sex_col="sex",
        ecog_col="ecog",
        group3_has_ecog=False,
        decimals=1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="dataset_overview", index=False)
        stats_df.to_excel(writer, sheet_name="data_description", index=False)

    print(f"Saved Excel file to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
