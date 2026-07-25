import os
import warnings
import pandas as pd
import numpy as np
import config


# Helpers


def _pjoin(*parts: str) -> str:
    return os.path.join(*parts)


def _load_gt(path: str, cohort: int = 1) -> pd.DataFrame:
    """Load ground-truth Excel file and return minimal columns."""
    df = pd.read_excel(path)
    df = df.rename(columns={"Pat ID": "patient_id"})
    keep = ["patient_id", "sarcopenia_composite", "sex"]
    if cohort == 1:
        keep.append("test_temporal")
    df = df[keep].copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df


def _apply_smi_threshold(smi: pd.Series, sex: pd.Series) -> pd.Series:
    """Apply van der Werf SMI threshold (sex-dependent) → binary."""
    result = pd.Series(index=smi.index, dtype=float)
    male = sex == 1
    female = sex == 0
    result[male] = (smi[male] < config.SMI_THRESHOLD["male"]).astype(float)
    result[female] = (smi[female] < config.SMI_THRESHOLD["female"]).astype(float)
    return result


def _apply_mra_threshold(mra: pd.Series, sex: pd.Series) -> pd.Series:
    """Apply van der Werf MRA threshold (sex-dependent) → binary."""
    result = pd.Series(index=mra.index, dtype=float)
    male = sex == 1
    female = sex == 0
    result[male] = (mra[male] < config.MRA_THRESHOLD["male"]).astype(float)
    result[female] = (mra[female] < config.MRA_THRESHOLD["female"]).astype(float)
    return result


def _load_conventional_ct(filename: str) -> pd.DataFrame:
    path = _pjoin(config.OUTPUT_ROOT, config.CONVENTIONAL_CT_SUBFOLDER, filename)
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df


def _load_predictions(subfolder: str, pred_filename: str) -> pd.DataFrame:
    path = _pjoin(config.OUTPUT_ROOT, subfolder, "predictions", pred_filename)
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df


def _select_model(df: pd.DataFrame, model_hint: str | None, label: str) -> pd.DataFrame:
    """Return rows for the chosen model; auto-select when hint is None."""
    models = df["model_name"].unique().tolist()
    if model_hint is not None:
        if model_hint not in models:
            raise ValueError(
                f"[{label}] model_name '{model_hint}' not found. " f"Available: {models}"
            )
        return df[df["model_name"] == model_hint].copy()
    if len(models) == 1:
        return df[df["model_name"] == models[0]].copy()
    # Multiple models: prefer the last alphabetically or raise a warning.
    chosen = sorted(models)[-1]
    warnings.warn(
        f"[{label}] Multiple models found {models}. "
        f"Auto-selecting '{chosen}'. Set the model hint in config.py to override.",
        UserWarning,
        stacklevel=3,
    )
    return df[df["model_name"] == chosen].copy()


def _merge_pred_label(
    base: pd.DataFrame,
    pred_df: pd.DataFrame,
    model_hint: str | None,
    col_name: str,
) -> pd.DataFrame:
    """Merge pred_label from prediction DataFrame into base DataFrame."""
    pred = _select_model(pred_df, model_hint, col_name)
    pred = pred[["patient_id", "pred_label"]].rename(columns={"pred_label": col_name})
    return base.merge(pred, on="patient_id", how="left")


# Public API


def build_split_dataframes() -> dict[str, pd.DataFrame]:
    """
    Returns
    -------
    dict with keys "train", "test_1", "test_2".
    Each value is a DataFrame indexed by patient_id with one column per
    classifier (keys of CLASSIFIER_LABELS).
    """

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt1 = _load_gt(config.GROUND_TRUTH_COHORT_1, cohort=1)
    gt2 = _load_gt(config.GROUND_TRUTH_COHORT_2, cohort=2)

    gt_train = gt1[gt1["test_temporal"] == 0].copy()
    gt_test1 = gt1[gt1["test_temporal"] == 1].copy()
    gt_test2 = gt2.copy()

    # ── Conventional CT (SMI, MRA) ────────────────────────────────────────────
    ct12 = _load_conventional_ct(config.CONVENTIONAL_CT_COHORT_12_FILE)
    ct2 = _load_conventional_ct(config.CONVENTIONAL_CT_COHORT_2_FILE)

    # ── Prediction files: conventional CT models (auto SMI / MRA) ────────────
    ct_models_train = _load_predictions(
        config.CONVENTIONAL_CT_MODELS_SUBFOLDER, config.PRED_TRAIN_FILE
    )
    ct_models_test1 = _load_predictions(
        config.CONVENTIONAL_CT_MODELS_SUBFOLDER, config.PRED_TEST1_FILE
    )
    ct_models_test2 = _load_predictions(
        config.CONVENTIONAL_CT_MODELS_SUBFOLDER, config.PRED_TEST2_FILE
    )

    # ── Prediction files: mean fraction ───────────────────────────────────────
    mf_train = _load_predictions(config.MEAN_FRACTION_SUBFOLDER, config.PRED_TRAIN_FILE)
    mf_test1 = _load_predictions(config.MEAN_FRACTION_SUBFOLDER, config.PRED_TEST1_FILE)
    mf_test2 = _load_predictions(config.MEAN_FRACTION_SUBFOLDER, config.PRED_TEST2_FILE)

    # ── Prediction files: signature (musclefat + top3) ────────────────────────
    sig_train = _load_predictions(config.SIGNATURE_SUBFOLDER, config.PRED_TRAIN_FILE)
    sig_test1 = _load_predictions(config.SIGNATURE_SUBFOLDER, config.PRED_TEST1_FILE)
    sig_test2 = _load_predictions(config.SIGNATURE_SUBFOLDER, config.PRED_TEST2_FILE)

    splits = {}

    for split_name, gt, ct, ct_models_pred, mf_pred, sig_pred in [
        ("train", gt_train, ct12, ct_models_train, mf_train, sig_train),
        ("test_1", gt_test1, ct12, ct_models_test1, mf_test1, sig_test1),
        ("test_2", gt_test2, ct2, ct_models_test2, mf_test2, sig_test2),
    ]:
        df = gt[["patient_id", "sarcopenia_composite", "sex"]].copy()

        # Merge CT values
        df = df.merge(ct[["patient_id", "smi_2d", "mra_2d"]], on="patient_id", how="left")

        # Derive binary CT labels
        df["smi_van_der_werf"] = _apply_smi_threshold(df["smi_2d"], df["sex"])
        df["mra_van_der_werf"] = _apply_mra_threshold(df["mra_2d"], df["sex"])

        # Rename ground truth column to our internal key
        df = df.rename(columns={"sarcopenia_composite": "functional_test"})

        # Merge auto SMI and MRA model predictions
        df = _merge_pred_label(
            df,
            ct_models_pred,
            config.AUTO_SMI_MODEL,
            "auto_smi",
        )
        df = _merge_pred_label(
            df,
            ct_models_pred,
            config.AUTO_MRA_MODEL,
            "auto_mra",
        )

        # Merge mean-fraction model prediction
        df = _merge_pred_label(
            df,
            mf_pred,
            config.MUSCLEFAT_MEAN_FRACTION_MODEL,
            "musclefat_mean_fraction_3d",
        )

        # Merge the CT+muscle-fat handcrafted-radiomics signature model.
        df = _merge_pred_label(
            df,
            sig_pred,
            config.CT_MUSCLEFAT_SIGNATURE_MODEL,
            "mf_ct_scores_clinical_3d",
        )

        # Keep only the classifier columns in canonical order
        classifier_keys = list(config.CLASSIFIER_LABELS.keys())
        missing = [k for k in classifier_keys if k not in df.columns]
        if missing:
            warnings.warn(
                f"[{split_name}] Columns missing after merge: {missing}. "
                "They will appear as NaN in the heatmap.",
                UserWarning,
            )
            for k in missing:
                df[k] = np.nan

        df = df.set_index("patient_id")[classifier_keys]
        splits[split_name] = df

    return splits
