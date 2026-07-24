from dataset import build_all_datasets


def main() -> None:
    datasets = build_all_datasets(save_csv=True)
    for name, df in datasets.items():
        print("=" * 60)
        print(f"Task: {name}  shape={df.shape}")
        if not df.empty:
            print(f"  cohort counts: {df['cohort'].value_counts().to_dict()}")
            print(f"  split counts:  {df['split'].value_counts().to_dict()}")
            print(f"  target mean:   {df['target'].mean():.3f}")
        print(df[["patient_id", "cohort", "split", "target"]].head(3))


if __name__ == "__main__":
    main()
