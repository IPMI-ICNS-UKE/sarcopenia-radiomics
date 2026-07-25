import os


# Root paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
OUTPUT_DIR = os.path.join(ROOT_OUTPUT_DIR, "07_04_comparison")


# Input subfolders (relative to ROOT_OUTPUT_DIR)
METHOD_DIRS = {
    # 2D SMI and MRA (conventional CT)
    "auto_smi": "06_01_conventional_ct",
    "auto_mra_score_clinical": "06_01_conventional_ct",
    # 3D SMI and MRA (conventional CT)
    "auto_smi_3d": "06_01_conventional_ct_3d",
    "auto_mra_score_clinical_3d": "06_01_conventional_ct_3d",
    ###############################################################################################
    # 2D and 3D muscle fat (spectral CT)
    "musclefat_mean_fraction_score_clinical": "06_02_01_spectral_radiomics_mean_fraction",
    "musclefat_mean_fraction_score_clinical_3d": "06_02_01_spectral_radiomics_mean_fraction_3d",
    # 2D and 3D mean fraction (VNC)
    "vnc_mean_fraction_score_clinical": "06_02_01_spectral_radiomics_mean_fraction",
    "vnc_mean_fraction_score_clinical_3d": "06_02_01_spectral_radiomics_mean_fraction_3d",
    # 2D and 3D mean fraction (electron density)
    "electrondensity_mean_fraction_score_clinical": "06_02_01_spectral_radiomics_mean_fraction",
    "electrondensity_mean_fraction_score_clinical_3d": "06_02_01_spectral_radiomics_mean_fraction_3d",
    # 2D and 3D mean fraction (effective z)
    "effectivez_mean_fraction_score_clinical": "06_02_01_spectral_radiomics_mean_fraction",
    "effectivez_mean_fraction_score_clinical_3d": "06_02_01_spectral_radiomics_mean_fraction_3d",
    # 2D and 3D mean fraction (iodine)
    "iodine_mean_fraction_score_clinical": "06_02_01_spectral_radiomics_mean_fraction",
    "iodine_mean_fraction_score_clinical_3d": "06_02_01_spectral_radiomics_mean_fraction_3d",
    # 2D and 3D mean fraction (40kev)
    "40kev_mean_fraction_score_clinical": "06_02_01_spectral_radiomics_mean_fraction",
    "40kev_mean_fraction_score_clinical_3d": "06_02_01_spectral_radiomics_mean_fraction_3d",
    # 2D and 3D mean fraction (60kev)
    "60kev_mean_fraction_score_clinical": "06_02_01_spectral_radiomics_mean_fraction",
    "60kev_mean_fraction_score_clinical_3d": "06_02_01_spectral_radiomics_mean_fraction_3d",
    # 2D and 3D mean fraction (80kev)
    "80kev_mean_fraction_score_clinical": "06_02_01_spectral_radiomics_mean_fraction",
    "80kev_mean_fraction_score_clinical_3d": "06_02_01_spectral_radiomics_mean_fraction_3d",
    # 2D and 3D mean fraction (100kev)
    "100kev_mean_fraction_score_clinical": "06_02_01_spectral_radiomics_mean_fraction",
    "100kev_mean_fraction_score_clinical_3d": "06_02_01_spectral_radiomics_mean_fraction_3d",
    # 2D and 3D mean fraction (120kev)
    "120kev_mean_fraction_score_clinical": "06_02_01_spectral_radiomics_mean_fraction",
    "120kev_mean_fraction_score_clinical_3d": "06_02_01_spectral_radiomics_mean_fraction_3d",
    ###############################################################################################
    # 2D and 3D handcrafted radiomics (CT)
    "ct_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "ct_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (MF)
    "musclefat_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "musclefat_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (VNC)
    "vnc_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "vnc_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (electron density)
    "electrondensity_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "electrondensity_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (effective z)
    "effectivez_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "effectivez_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (iodine)
    "iodine_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "iodine_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (40kev)
    "40kev_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "40kev_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (60kev)
    "60kev_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "60kev_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (80kev)
    "80kev_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "80kev_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (100kev)
    "100kev_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "100kev_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (120kev)
    "120kev_signature_score_clinical": "06_02_02_spectral_radiomics_signature",
    "120kev_signature_score_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    # 2D and 3D handcrafted radiomics (CT + MF)
    "mf_ct_scores_clinical": "06_02_02_spectral_radiomics_signature",
    "mf_ct_scores_clinical_3d": "06_02_02_spectral_radiomics_signature_3d",
    ###############################################################################################
    # 2D and 3D deep radiomics (CT)
    "ct_deep_score_clinical": "06_03_deep_radiomics",
    "ct_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
    # 2D and 3D deep radiomics (MF)
    "musclefat_deep_score_clinical": "06_03_deep_radiomics",
    "musclefat_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
    # 2D and 3D deep radiomics (VNC)
    "vnc_deep_score_clinical": "06_03_deep_radiomics",
    "vnc_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
    # 2D and 3D deep radiomics (electron density)
    "electrondensity_deep_score_clinical": "06_03_deep_radiomics",
    "electrondensity_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
    # 2D and 3D deep radiomics (effective z)
    "effectivez_deep_score_clinical": "06_03_deep_radiomics",
    "effectivez_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
    # 2D and 3D deep radiomics (iodine)
    "iodine_deep_score_clinical": "06_03_deep_radiomics",
    "iodine_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
    # 2D and 3D deep radiomics (40kev)
    "40kev_deep_score_clinical": "06_03_deep_radiomics",
    "40kev_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
    # 2D and 3D deep radiomics (60kev)
    "60kev_deep_score_clinical": "06_03_deep_radiomics",
    "60kev_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
    # 2D and 3D deep radiomics (80kev)
    "80kev_deep_score_clinical": "06_03_deep_radiomics",
    "80kev_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
    # 2D and 3D deep radiomics (100kev)
    "100kev_deep_score_clinical": "06_03_deep_radiomics",
    "100kev_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
    # 2D and 3D deep radiomics (120kev)
    "120kev_deep_score_clinical": "06_03_deep_radiomics",
    "120kev_deep_score_clinical_3d": "06_03_deep_radiomics_3d",
}

# Human-readable display names for tables and plots
METHOD_DISPLAY_NAMES = {
    # 2D SMI and MRA (conventional CT)
    "auto_smi": "2D SMI",
    "auto_mra_score_clinical": "2D MRA",
    # 3D SMI and MRA (conventional CT)
    "auto_smi_3d": "3D SMI",
    "auto_mra_score_clinical_3d": "3D MRA",
    ###############################################################################################
    # 2D and 3D muscle fat (spectral CT)
    "musclefat_mean_fraction_score_clinical": "2D mean fraction (muscle fat)",
    "musclefat_mean_fraction_score_clinical_3d": "3D mean fraction (muscle fat)",
    # 2D and 3D mean fraction (VNC)
    "vnc_mean_fraction_score_clinical": "2D mean fraction (vnc)",
    "vnc_mean_fraction_score_clinical_3d": "3D mean fraction (vnc)",
    # 2D and 3D mean fraction (electron density)
    "electrondensity_mean_fraction_score_clinical": "2D mean fraction (electron density)",
    "electrondensity_mean_fraction_score_clinical_3d": "3D mean fraction (electron density)",
    # 2D and 3D mean fraction (effective z)
    "effectivez_mean_fraction_score_clinical": "2D mean fraction (effective z)",
    "effectivez_mean_fraction_score_clinical_3d": "3D mean fraction (effective z)",
    # 2D and 3D mean fraction (iodine)
    "iodine_mean_fraction_score_clinical": "2D mean fraction (iodine)",
    "iodine_mean_fraction_score_clinical_3d": "3D mean fraction (iodine)",
    # 2D and 3D mean fraction (40kev)
    "40kev_mean_fraction_score_clinical": "2D mean fraction (40kev)",
    "40kev_mean_fraction_score_clinical_3d": "3D mean fraction (40kev)",
    # 2D and 3D mean fraction (60kev)
    "60kev_mean_fraction_score_clinical": "2D mean fraction (60kev)",
    "60kev_mean_fraction_score_clinical_3d": "3D mean fraction (60kev)",
    # 2D and 3D mean fraction (80kev)
    "80kev_mean_fraction_score_clinical": "2D mean fraction (80kev)",
    "80kev_mean_fraction_score_clinical_3d": "3D mean fraction (80kev)",
    # 2D and 3D mean fraction (100kev)
    "100kev_mean_fraction_score_clinical": "2D mean fraction (100kev)",
    "100kev_mean_fraction_score_clinical_3d": "3D mean fraction (100kev)",
    # 2D and 3D mean fraction (120kev)
    "120kev_mean_fraction_score_clinical": "2D mean fraction (120kev)",
    "120kev_mean_fraction_score_clinical_3d": "3D mean fraction (120kev)",
    ###############################################################################################
    # 2D and 3D handcrafted radiomics (CT)
    "ct_signature_score_clinical": "2D handcrafted radiomics (CT)",
    "ct_signature_score_clinical_3d": "3D handcrafted radiomics (CT)",
    # 2D and 3D handcrafted radiomics (muscle fat)
    "musclefat_signature_score_clinical": "2D handcrafted radiomics (muscle fat)",
    "musclefat_signature_score_clinical_3d": "3D handcrafted radiomics (muscle fat)",
    "vnc_signature_score_clinical": "2D handcrafted radiomics (vnc)",
    "vnc_signature_score_clinical_3d": "3D handcrafted radiomics (vnc)",
    "electrondensity_signature_score_clinical": "2D handcrafted radiomics (electron density)",
    "electrondensity_signature_score_clinical_3d": "3D handcrafted radiomics (electron density)",
    "effectivez_signature_score_clinical": "2D handcrafted radiomics (effective z)",
    "effectivez_signature_score_clinical_3d": "3D handcrafted radiomics (effective z)",
    "iodine_signature_score_clinical": "2D handcrafted radiomics (iodine)",
    "iodine_signature_score_clinical_3d": "3D handcrafted radiomics (iodine)",
    # 2D and 3D handcrafted radiomics (40kev)
    "40kev_signature_score_clinical": "2D handcrafted radiomics (40kev)",
    "40kev_signature_score_clinical_3d": "3D handcrafted radiomics (40kev)",
    # 2D and 3D handcrafted radiomics (60kev)
    "60kev_signature_score_clinical": "2D handcrafted radiomics (60kev)",
    "60kev_signature_score_clinical_3d": "3D handcrafted radiomics (60kev)",
    # 2D and 3D handcrafted radiomics (80kev)
    "80kev_signature_score_clinical": "2D handcrafted radiomics (80kev)",
    "80kev_signature_score_clinical_3d": "3D handcrafted radiomics (80kev)",
    # 2D and 3D handcrafted radiomics (100kev)
    "100kev_signature_score_clinical": "2D handcrafted radiomics (100kev)",
    "100kev_signature_score_clinical_3d": "3D handcrafted radiomics (100kev)",
    # 2D and 3D handcrafted radiomics (120kev)
    "120kev_signature_score_clinical": "2D handcrafted radiomics (120kev)",
    "120kev_signature_score_clinical_3d": "3D handcrafted radiomics (120kev)",
    # 2D and 3D handcrafted radiomics (CT + MF)
    "mf_ct_scores_clinical": "2D handcrafted radiomics (CT and MF)",
    "mf_ct_scores_clinical_3d": "3D handcrafted radiomics (CT and MF)",
    ###############################################################################################
    # 2D and 3D deep radiomics (CT)
    "ct_deep_score_clinical": "2D deep radiomics (CT)",
    "ct_deep_score_clinical_3d": "3D deep radiomics (CT)",
    # 2D and 3D deep radiomics (muscle fat)
    "musclefat_deep_score_clinical": "2D deep radiomics (muscle fat)",
    "musclefat_deep_score_clinical_3d": "3D deep radiomics (muscle fat)",
    # 2D and 3D deep radiomics (vnc)
    "vnc_deep_score_clinical": "2D deep radiomics (vnc)",
    "vnc_deep_score_clinical_3d": "3D deep radiomics (vnc)",
    # 2D and 3D deep radiomics (electron density)
    "electrondensity_deep_score_clinical": "2D deep radiomics (electron density)",
    "electrondensity_deep_score_clinical_3d": "3D deep radiomics (electron density)",
    # 2D and 3D deep radiomics (effective z)
    "effectivez_deep_score_clinical": "2D deep radiomics (effective z)",
    "effectivez_deep_score_clinical_3d": "3D deep radiomics (effective z)",
    # 2D and 3D deep radiomics (iodine)
    "iodine_deep_score_clinical": "2D deep radiomics (iodine)",
    "iodine_deep_score_clinical_3d": "3D deep radiomics (iodine)",
    # 2D and 3D deep radiomics (40kev)
    "40kev_deep_score_clinical": "2D deep radiomics (40kev)",
    "40kev_deep_score_clinical_3d": "3D deep radiomics (40kev)",
    # 2D and 3D deep radiomics (60kev)
    "60kev_deep_score_clinical": "2D deep radiomics (60kev)",
    "60kev_deep_score_clinical_3d": "3D deep radiomics (60kev)",
    # 2D and 3D deep radiomics (80kev)
    "80kev_deep_score_clinical": "2D deep radiomics (80kev)",
    "80kev_deep_score_clinical_3d": "3D deep radiomics (80kev)",
    # 2D and 3D deep radiomics (100kev)
    "100kev_deep_score_clinical": "2D deep radiomics (100kev)",
    "100kev_deep_score_clinical_3d": "3D deep radiomics (100kev)",
    # 2D and 3D deep radiomics (120kev)
    "120kev_deep_score_clinical": "2D deep radiomics (120kev)",
    "120kev_deep_score_clinical_3d": "3D deep radiomics (120kev)",
}

# Ordered list of methods for table columns
METHOD_ORDER = [
    "auto_smi",
    "auto_mra_score_clinical",
    "auto_smi_3d",
    "auto_mra_score_clinical_3d",
    ###############################################################################################
    "musclefat_mean_fraction_score_clinical",
    "musclefat_mean_fraction_score_clinical_3d",
    "vnc_mean_fraction_score_clinical",
    "vnc_mean_fraction_score_clinical_3d",
    "electrondensity_mean_fraction_score_clinical",
    "electrondensity_mean_fraction_score_clinical_3d",
    "effectivez_mean_fraction_score_clinical",
    "effectivez_mean_fraction_score_clinical_3d",
    "iodine_mean_fraction_score_clinical",
    "iodine_mean_fraction_score_clinical_3d",
    "40kev_mean_fraction_score_clinical",
    "40kev_mean_fraction_score_clinical_3d",
    "60kev_mean_fraction_score_clinical",
    "60kev_mean_fraction_score_clinical_3d",
    "80kev_mean_fraction_score_clinical",
    "80kev_mean_fraction_score_clinical_3d",
    "100kev_mean_fraction_score_clinical",
    "100kev_mean_fraction_score_clinical_3d",
    "120kev_mean_fraction_score_clinical",
    "120kev_mean_fraction_score_clinical_3d",
    ###############################################################################################
    "ct_signature_score_clinical",
    "ct_signature_score_clinical_3d",
    "musclefat_signature_score_clinical",
    "musclefat_signature_score_clinical_3d",
    "vnc_signature_score_clinical",
    "vnc_signature_score_clinical_3d",
    "electrondensity_signature_score_clinical",
    "electrondensity_signature_score_clinical_3d",
    "effectivez_signature_score_clinical",
    "effectivez_signature_score_clinical_3d",
    "iodine_signature_score_clinical",
    "iodine_signature_score_clinical_3d",
    "40kev_signature_score_clinical",
    "40kev_signature_score_clinical_3d",
    "60kev_signature_score_clinical",
    "60kev_signature_score_clinical_3d",
    "80kev_signature_score_clinical",
    "80kev_signature_score_clinical_3d",
    "100kev_signature_score_clinical",
    "100kev_signature_score_clinical_3d",
    "120kev_signature_score_clinical",
    "120kev_signature_score_clinical_3d",
    "mf_ct_scores_clinical",
    "mf_ct_scores_clinical_3d",
    ###############################################################################################
    "ct_deep_score_clinical",
    "ct_deep_score_clinical_3d",
    "musclefat_deep_score_clinical",
    "musclefat_deep_score_clinical_3d",
    "vnc_deep_score_clinical",
    "vnc_deep_score_clinical_3d",
    "electrondensity_deep_score_clinical",
    "electrondensity_deep_score_clinical_3d",
    "effectivez_deep_score_clinical",
    "effectivez_deep_score_clinical_3d",
    "iodine_deep_score_clinical",
    "iodine_deep_score_clinical_3d",
    "40kev_deep_score_clinical",
    "40kev_deep_score_clinical_3d",
    "60kev_deep_score_clinical",
    "60kev_deep_score_clinical_3d",
    "80kev_deep_score_clinical",
    "80kev_deep_score_clinical_3d",
    "100kev_deep_score_clinical",
    "100kev_deep_score_clinical_3d",
    "120kev_deep_score_clinical",
    "120kev_deep_score_clinical_3d",
]


# Data subsets
# Each entry: (subset_key, display_name, metrics_filename_stem, predictions_filename_stem)
DATA_SUBSETS = {
    # Classification
    "cls": {
        "train": {
            "metrics_file": "train_oof_cls_metrics_sarcopenia_composite_cls.xlsx",
            "predictions_file": "train_oof_cls_predictions_sarcopenia_composite_cls.csv",
            "display_name": "Training set",
        },
        "test_1": {
            "metrics_file": "test_cls_metrics_sarcopenia_composite_cls.xlsx",
            "predictions_file": "test_cls_predictions_sarcopenia_composite_cls.csv",
            "display_name": "Test set 1 (temporal hold-out)",
        },
        "test_2": {
            "metrics_file": "test_cls_metrics_sarcopenia_composite_cls_cohort_2.xlsx",
            "predictions_file": "test_cls_predictions_sarcopenia_composite_cls_cohort_2.csv",
            "display_name": "Test set 2 (external cohort)",
        },
    },
    # Regression
    "reg": {
        "train": {
            "metrics_file": "train_oof_reg_metrics_hand_grip_reg.xlsx",
            "predictions_file": "train_oof_reg_predictions_hand_grip_reg.csv",
            "display_name": "Training set",
        },
        "test_1": {
            "metrics_file": "test_reg_metrics_hand_grip_reg.xlsx",
            "predictions_file": "test_reg_predictions_hand_grip_reg.csv",
            "display_name": "Test set 1 (temporal hold-out)",
        },
        "test_2": {
            "metrics_file": "test_reg_metrics_hand_grip_reg_cohort_2.xlsx",
            "predictions_file": "test_reg_predictions_hand_grip_reg_cohort_2.csv",
            "display_name": "Test set 2 (external cohort)",
        },
    },
    # Classification
    "cls_2": {
        "train": {
            "metrics_file": "train_oof_cls_metrics_chair_rise_cls.xlsx",
            "predictions_file": "train_oof_cls_predictions_chair_rise_cls.csv",
            "display_name": "Training set",
        },
        "test_1": {
            "metrics_file": "test_cls_metrics_chair_rise_cls.xlsx",
            "predictions_file": "test_cls_predictions_chair_rise_cls.csv",
            "display_name": "Test set 1 (temporal hold-out)",
        },
        "test_2": {
            "metrics_file": "test_cls_metrics_chair_rise_cls_cohort_2.xlsx",
            "predictions_file": "test_cls_predictions_chair_rise_cls_cohort_2.csv",
            "display_name": "Test set 2 (external cohort)",
        },
    },
}


# Statistical comparisons
# Pairs: (new_method, baseline_method)
COMPARISON_PAIRS = [
    ("auto_smi_3d", "auto_smi"),
    ("auto_mra_score_clinical_3d", "auto_mra_score_clinical"),
    ("musclefat_mean_fraction_score_clinical", "auto_smi"),
    ("musclefat_mean_fraction_score_clinical", "auto_mra_score_clinical"),
    ("musclefat_mean_fraction_score_clinical_3d", "auto_smi"),
    ("musclefat_mean_fraction_score_clinical_3d", "auto_mra_score_clinical"),
    ("ct_signature_score_clinical", "auto_smi"),
    ("ct_signature_score_clinical", "auto_mra_score_clinical"),
    ("ct_signature_score_clinical_3d", "auto_smi"),
    ("ct_signature_score_clinical_3d", "auto_mra_score_clinical"),
    ("musclefat_signature_score_clinical", "auto_smi"),
    ("musclefat_signature_score_clinical", "auto_mra_score_clinical"),
    ("musclefat_signature_score_clinical_3d", "auto_smi"),
    ("musclefat_signature_score_clinical_3d", "auto_mra_score_clinical"),
    ("vnc_signature_score_clinical", "auto_smi"),
    ("vnc_signature_score_clinical", "auto_mra_score_clinical"),
    ("vnc_signature_score_clinical_3d", "auto_smi"),
    ("vnc_signature_score_clinical_3d", "auto_mra_score_clinical"),
    ("electrondensity_signature_score_clinical", "auto_smi"),
    ("electrondensity_signature_score_clinical", "auto_mra_score_clinical"),
    ("electrondensity_signature_score_clinical_3d", "auto_smi"),
    ("electrondensity_signature_score_clinical_3d", "auto_mra_score_clinical"),
    ("effectivez_signature_score_clinical", "auto_smi"),
    ("effectivez_signature_score_clinical", "auto_mra_score_clinical"),
    ("effectivez_signature_score_clinical_3d", "auto_smi"),
    ("effectivez_signature_score_clinical_3d", "auto_mra_score_clinical"),
    ("iodine_signature_score_clinical", "auto_smi"),
    ("iodine_signature_score_clinical", "auto_mra_score_clinical"),
    ("iodine_signature_score_clinical_3d", "auto_smi"),
    ("iodine_signature_score_clinical_3d", "auto_mra_score_clinical"),
    ("mf_ct_scores_clinical", "auto_smi"),
    ("mf_ct_scores_clinical", "auto_mra_score_clinical"),
    ("mf_ct_scores_clinical_3d", "auto_smi"),
    ("mf_ct_scores_clinical_3d", "auto_mra_score_clinical"),
    ("ct_deep_score_clinical", "auto_smi"),
    ("ct_deep_score_clinical", "auto_mra_score_clinical"),
    ("ct_deep_score_clinical_3d", "auto_smi"),
    ("ct_deep_score_clinical_3d", "auto_mra_score_clinical"),
    ("musclefat_deep_score_clinical", "auto_smi"),
    ("musclefat_deep_score_clinical", "auto_mra_score_clinical"),
    ("musclefat_deep_score_clinical_3d", "auto_smi"),
    ("musclefat_deep_score_clinical_3d", "auto_mra_score_clinical"),
    ("vnc_deep_score_clinical", "auto_smi"),
    ("vnc_deep_score_clinical", "auto_mra_score_clinical"),
    ("vnc_deep_score_clinical_3d", "auto_smi"),
    ("vnc_deep_score_clinical_3d", "auto_mra_score_clinical"),
    ("electrondensity_deep_score_clinical", "auto_smi"),
    ("electrondensity_deep_score_clinical", "auto_mra_score_clinical"),
    ("electrondensity_deep_score_clinical_3d", "auto_smi"),
    ("electrondensity_deep_score_clinical_3d", "auto_mra_score_clinical"),
    ("effectivez_deep_score_clinical", "auto_smi"),
    ("effectivez_deep_score_clinical", "auto_mra_score_clinical"),
    ("effectivez_deep_score_clinical_3d", "auto_smi"),
    ("effectivez_deep_score_clinical_3d", "auto_mra_score_clinical"),
    ("iodine_deep_score_clinical", "auto_smi"),
    ("iodine_deep_score_clinical", "auto_mra_score_clinical"),
    ("iodine_deep_score_clinical_3d", "auto_smi"),
    ("iodine_deep_score_clinical_3d", "auto_mra_score_clinical"),
]

# Baseline methods (no p-value computed for these as the "new" method)
BASELINE_METHODS = {"auto_smi", "auto_mra_score_clinical"}


# Multiple comparisons correction
# Options:
#   "fdr_bh"   – Benjamini-Hochberg (recommended: controls FDR, less conservative)
#   "bonferroni" – Bonferroni (conservative, controls FWER)
#   "none"     – No correction
#
# The "family" of tests for correction is defined per task × subset block
# (i.e., the 6 pairwise comparisons listed in COMPARISON_PAIRS).
CORRECTION_METHOD = "fdr_bh"  # Change to "bonferroni" or "none" as needed

# Significance threshold (alpha)
ALPHA = 0.05


# Bootstrap settings (for DeLong fallback and regression tests)
RANDOM_SEED = 42


# Statistical test settings

# Classification : two-sided DeLong test on AUC (no tunable parameters)
# Regression     : two-sided Wilcoxon signed-rank test on paired absolute errors
#   - Non-parametric, exploits paired structure, no normality assumption
#   - Appropriate for small samples (n=19–69 in this study)
#   - scipy.stats.wilcoxon handles zero-difference ties automatically


# Output / display settings
DECIMAL_POINTS = 2  # decimal places in comparison tables
DPI = 300  # PNG export resolution
FIGURE_FORMAT = "png"

# Color scale for p-value heatmaps (matplotlib colormap name)
# "RdYlGn_r": red = significant, green = not significant (intuitive for p-values)
HEATMAP_COLORMAP = "RdYlGn_r"
