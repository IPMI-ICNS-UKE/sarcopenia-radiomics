from dataset import build_all_datasets


def main():
    datasets = build_all_datasets(save_csv=True)
    for name, df in datasets.items():
        print("=" * 60)
        print(f"Task: {name}  shape={df.shape}")
        print(df[["patient_id", "cohort", "split", "target"]].head(3))


if __name__ == "__main__":
    main()
