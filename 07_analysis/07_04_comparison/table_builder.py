import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any

import config_extended as config
from data_loader import get_n_cases


# Formatting helpers
def _fmt(value: Any, decimals: int = config.DECIMAL_POINTS) -> str:
    """Format a float to a fixed number of decimal places."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def _fmt_pvalue(p: Optional[float]) -> str:
    """Format a p-value with 3 decimal places; show '<0.001' if very small."""
    if p is None or np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _parse_ci(ci_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a CI string like '0.XXX (0.YYY-0.ZZZ)' or 'XX.XXX (YY.YYY-ZZ.ZZZ)'.
    Returns (point_estimate_str, ci_range_str) or ('N/A', 'N/A').
    """
    if not isinstance(ci_str, str) or "(" not in ci_str:
        return (str(ci_str), "")
    parts = ci_str.split("(")
    point = parts[0].strip()
    ci_range = parts[1].rstrip(")").strip()
    return point, ci_range


def _reformat_metric(ci_str: str, decimals: int = config.DECIMAL_POINTS) -> str:
    """
    Re-format metric CI string to required decimal points.
    Input:  '0.XXX (0.YYY-0.ZZZ)'
    Output: '0.XX (0.YY-0.ZZ)'  (with config.DECIMAL_POINTS decimals)
    """
    if not isinstance(ci_str, str) or "(" not in ci_str:
        try:
            return _fmt(float(ci_str), decimals)
        except Exception:
            return str(ci_str)
    point, ci_range = _parse_ci(ci_str)
    # parse point
    try:
        point_fmt = _fmt(float(point), decimals)
    except Exception:
        point_fmt = point
    # parse range
    try:
        lo, hi = ci_range.split("-")
        ci_fmt = f"{_fmt(float(lo), decimals)}-{_fmt(float(hi), decimals)}"
    except Exception:
        ci_fmt = ci_range
    return f"{point_fmt} ({ci_fmt})"


def _get_sensitivity_specificity_str(
    row: pd.Series, metric: str, ci_col: str, decimals: int
) -> str:
    """
    Format sensitivity or specificity with numerator/denominator from confusion matrix.
    Returns: '0.XX (0.YY-0.ZZ) (TP/P)' or '0.XX (0.YY-0.ZZ) (TN/N)'
    """
    if row is None:
        return "N/A"
    ci_str = _reformat_metric(row.get(ci_col, "N/A"), decimals)
    n = row.get("N", None)
    tp = row.get("TP", None)
    tn = row.get("TN", None)
    fn = row.get("FN", None)
    fp = row.get("FP", None)

    if metric == "Sensitivity" and tp is not None and fn is not None:
        denom = int(tp + fn) if not (np.isnan(float(tp)) or np.isnan(float(fn))) else "?"
        num = int(tp) if not np.isnan(float(tp)) else "?"
        return f"{ci_str} ({num}/{denom})"
    elif metric == "Specificity" and tn is not None and fp is not None:
        denom = int(tn + fp) if not (np.isnan(float(tn)) or np.isnan(float(fp))) else "?"
        num = int(tn) if not np.isnan(float(tn)) else "?"
        return f"{ci_str} ({num}/{denom})"
    return ci_str


# p-value lookup helper
def _get_pvalue(
    pvalues: Dict,
    new_method: str,
    baseline_method: str,
    use_corrected: bool = True,
) -> Optional[float]:
    key = (new_method, baseline_method)
    if key not in pvalues:
        return None
    return pvalues[key].get("corrected" if use_corrected else "raw")


# Classification comparison table
def build_cls_block(
    subset_key: str,
    metrics: Dict[str, Optional[pd.Series]],
    predictions: Dict[str, Optional[pd.DataFrame]],
    pvalues: Dict,
    decimals: int = config.DECIMAL_POINTS,
    data_subset: str = "cls",
) -> pd.DataFrame:
    """
    Build one subset block for the classification comparison table.
    Returns a DataFrame with methods as columns and metric rows as index.
    """
    subset_info = config.DATA_SUBSETS[data_subset][subset_key]
    display_name = subset_info["display_name"]

    # Determine n_total, n_positive from the first available prediction
    n_total, n_positive = 0, 0
    for method in config.METHOD_ORDER:
        pred = predictions.get(method)
        if pred is not None:
            n_total, n_positive = get_n_cases(pred, data_subset)
            break

    col_names = [config.METHOD_DISPLAY_NAMES[m] for m in config.METHOD_ORDER]
    rows_data = {}

    # ── AUC [CI 95%] ──────────────────────────────────────────────────────────
    auc_vals = {}
    for method in config.METHOD_ORDER:
        row = metrics.get(method)
        val = _reformat_metric(row.get("AUC_CI95", "N/A"), decimals) if row is not None else "N/A"
        auc_vals[config.METHOD_DISPLAY_NAMES[method]] = val
    rows_data["AUC [CI 95%]"] = auc_vals

    # ── P-value (vs SMI) ──────────────────────────────────────────────────────
    pval_smi = {}
    for method in config.METHOD_ORDER:
        if method in config.BASELINE_METHODS:
            pval_smi[config.METHOD_DISPLAY_NAMES[method]] = "—"
        else:
            p = _get_pvalue(pvalues, method, "auto_smi")
            pval_smi[config.METHOD_DISPLAY_NAMES[method]] = _fmt_pvalue(p)
    rows_data["P-value (vs SMI)"] = pval_smi

    # ── P-value (vs MRA) ──────────────────────────────────────────────────────
    pval_mra = {}
    for method in config.METHOD_ORDER:
        if method in config.BASELINE_METHODS:
            pval_mra[config.METHOD_DISPLAY_NAMES[method]] = "—"
        else:
            p = _get_pvalue(pvalues, method, "auto_mra_score_clinical")
            pval_mra[config.METHOD_DISPLAY_NAMES[method]] = _fmt_pvalue(p)
    rows_data["P-value (vs MRA)"] = pval_mra

    # ── P-value (vs musclefat_mean_fraction_score_clinical_3d) ──────────────────────────────────────────────────────
    pval_mf = {}
    for method in config.METHOD_ORDER:
        if method in config.BASELINE_METHODS:
            pval_mf[config.METHOD_DISPLAY_NAMES[method]] = "—"
        else:
            p = _get_pvalue(pvalues, method, "musclefat_mean_fraction_score_clinical_3d")
            pval_mf[config.METHOD_DISPLAY_NAMES[method]] = _fmt_pvalue(p)
    rows_data["P-value (vs musclefat_mean_fraction_score_clinical_3d)"] = pval_mf

    # ── Balanced accuracy [CI 95%] ─────────────────────────────────────────────
    ba_vals = {}
    for method in config.METHOD_ORDER:
        row = metrics.get(method)
        val = (
            _reformat_metric(row.get("Balanced_Accuracy_CI95", "N/A"), decimals)
            if row is not None
            else "N/A"
        )
        ba_vals[config.METHOD_DISPLAY_NAMES[method]] = val
    rows_data["Balanced accuracy [CI 95%]"] = ba_vals

    # ── Specificity [CI 95%] (numerator/denominator) ──────────────────────────
    spec_vals = {}
    for method in config.METHOD_ORDER:
        row = metrics.get(method)
        val = (
            _get_sensitivity_specificity_str(row, "Specificity", "Specificity_CI95", decimals)
            if row is not None
            else "N/A"
        )
        spec_vals[config.METHOD_DISPLAY_NAMES[method]] = val
    rows_data["Specificity [CI 95%] (num/denom)"] = spec_vals

    # ── Sensitivity [CI 95%] (numerator/denominator) ──────────────────────────
    sens_vals = {}
    for method in config.METHOD_ORDER:
        row = metrics.get(method)
        val = (
            _get_sensitivity_specificity_str(row, "Sensitivity", "Sensitivity_CI95", decimals)
            if row is not None
            else "N/A"
        )
        sens_vals[config.METHOD_DISPLAY_NAMES[method]] = val
    rows_data["Sensitivity [CI 95%] (num/denom)"] = sens_vals

    # Build DataFrame
    df = pd.DataFrame(rows_data, index=col_names).T

    # Add subset header information as a separate row at the top
    header_info = {col: "" for col in col_names}
    header_df = pd.DataFrame(
        [header_info],
        index=[f"{display_name} | N={n_total} | N_positive={n_positive}"],
    )
    df = pd.concat([header_df, df])
    return df


# Regression comparison table
def build_reg_block(
    subset_key: str,
    metrics: Dict[str, Optional[pd.Series]],
    predictions: Dict[str, Optional[pd.DataFrame]],
    pvalues: Dict,
    decimals: int = config.DECIMAL_POINTS,
) -> pd.DataFrame:
    """
    Build one subset block for the regression comparison table.
    """
    subset_info = config.DATA_SUBSETS["reg"][subset_key]
    display_name = subset_info["display_name"]

    n_total, _ = 0, None
    for method in config.METHOD_ORDER:
        pred = predictions.get(method)
        if pred is not None:
            n_total, _ = get_n_cases(pred, "reg")
            break

    col_names = [config.METHOD_DISPLAY_NAMES[m] for m in config.METHOD_ORDER]
    rows_data = {}

    # ── RMSE [CI 95%] ─────────────────────────────────────────────────────────
    rmse_vals = {}
    for method in config.METHOD_ORDER:
        row = metrics.get(method)
        val = _reformat_metric(row.get("RMSE_CI95", "N/A"), decimals) if row is not None else "N/A"
        rmse_vals[config.METHOD_DISPLAY_NAMES[method]] = val
    rows_data["RMSE [CI 95%]"] = rmse_vals

    # ── P-value (vs SMI) ──────────────────────────────────────────────────────
    pval_smi = {}
    for method in config.METHOD_ORDER:
        if method in config.BASELINE_METHODS:
            pval_smi[config.METHOD_DISPLAY_NAMES[method]] = "—"
        else:
            p = _get_pvalue(pvalues, method, "auto_smi")
            pval_smi[config.METHOD_DISPLAY_NAMES[method]] = _fmt_pvalue(p)
    rows_data["P-value (vs SMI)"] = pval_smi

    # ── P-value (vs MRA) ──────────────────────────────────────────────────────
    pval_mra = {}
    for method in config.METHOD_ORDER:
        if method in config.BASELINE_METHODS:
            pval_mra[config.METHOD_DISPLAY_NAMES[method]] = "—"
        else:
            p = _get_pvalue(pvalues, method, "auto_mra_score_clinical")
            pval_mra[config.METHOD_DISPLAY_NAMES[method]] = _fmt_pvalue(p)
    rows_data["P-value (vs MRA)"] = pval_mra

    # ── MAE [CI 95%] ──────────────────────────────────────────────────────────
    mae_vals = {}
    for method in config.METHOD_ORDER:
        row = metrics.get(method)
        val = _reformat_metric(row.get("MAE_CI95", "N/A"), decimals) if row is not None else "N/A"
        mae_vals[config.METHOD_DISPLAY_NAMES[method]] = val
    rows_data["MAE [CI 95%]"] = mae_vals

    # ── R2 [CI 95%] ───────────────────────────────────────────────────────────
    r2_vals = {}
    for method in config.METHOD_ORDER:
        row = metrics.get(method)
        val = _reformat_metric(row.get("R2_CI95", "N/A"), decimals) if row is not None else "N/A"
        r2_vals[config.METHOD_DISPLAY_NAMES[method]] = val
    rows_data["R2 [CI 95%]"] = r2_vals

    # ── Spearman correlation [P-value] ─────────────────────────────────────────
    spearman_vals = {}
    for method in config.METHOD_ORDER:
        row = metrics.get(method)
        if row is not None:
            r = row.get("Spearman_R", None)
            p = row.get("Spearman_P", None)
            if r is not None and not (isinstance(r, float) and np.isnan(r)):
                spearman_vals[config.METHOD_DISPLAY_NAMES[method]] = (
                    f"{_fmt(float(r), decimals)} (p={_fmt_pvalue(float(p))})"
                    if p is not None
                    else _fmt(float(r), decimals)
                )
            else:
                spearman_vals[config.METHOD_DISPLAY_NAMES[method]] = "N/A"
        else:
            spearman_vals[config.METHOD_DISPLAY_NAMES[method]] = "N/A"
    rows_data["Spearman correlation [p-value]"] = spearman_vals

    df = pd.DataFrame(rows_data, index=col_names).T

    header_info = {col: "" for col in col_names}
    header_df = pd.DataFrame(
        [header_info],
        index=[f"{display_name} | N={n_total}"],
    )
    df = pd.concat([header_df, df])
    return df


# Excel export with formatting
def _apply_excel_formatting(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    """Apply formatting to an Excel worksheet."""
    wb = writer.book
    ws = writer.sheets[sheet_name]

    header_fmt = wb.add_format(
        {
            "bold": True,
            "bg_color": "#2E4057",
            "font_color": "#FFFFFF",
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "text_wrap": True,
        }
    )
    metric_header_fmt = wb.add_format(
        {
            "bold": True,
            "bg_color": "#E8EFF7",
            "align": "left",
            "valign": "vcenter",
            "border": 1,
            "text_wrap": True,
        }
    )
    section_header_fmt = wb.add_format(
        {
            "bold": True,
            "bg_color": "#4A7C99",
            "font_color": "#FFFFFF",
            "align": "left",
            "valign": "vcenter",
            "border": 1,
            "italic": True,
        }
    )
    cell_fmt = wb.add_format(
        {
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "text_wrap": True,
        }
    )
    pvalue_sig_fmt = wb.add_format(
        {
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "bold": True,
            "font_color": "#CC0000",  # red for significant
        }
    )

    # Column widths
    ws.set_column(0, 0, 36)  # row index
    for col_idx in range(1, len(df.columns) + 1):
        ws.set_column(col_idx, col_idx, 28)

    # Header row
    for col_idx, col_name in enumerate([""] + list(df.columns)):
        ws.write(0, col_idx, col_name, header_fmt)

    # Data rows
    for row_idx, (idx_label, row_vals) in enumerate(df.iterrows(), start=1):
        # Detect section header rows (those with all empty values)
        is_section = all(str(v) == "" for v in row_vals.values)
        row_fmt = section_header_fmt if is_section else metric_header_fmt
        ws.write(row_idx, 0, str(idx_label), row_fmt)

        for col_idx, val in enumerate(row_vals.values, start=1):
            val_str = str(val)
            # Highlight significant p-values
            is_pvalue_row = "P-value" in str(idx_label)
            if is_pvalue_row and val_str not in ("—", "N/A", ""):
                try:
                    p_num = float(val_str) if val_str != "<0.001" else 0.0
                    fmt = pvalue_sig_fmt if p_num < config.ALPHA else cell_fmt
                except Exception:
                    fmt = cell_fmt
                ws.write(row_idx, col_idx, val_str, fmt)
            else:
                ws.write(row_idx, col_idx, val_str, cell_fmt)

    ws.set_row(0, 40)


def save_comparison_tables(
    cls_blocks: Dict[str, pd.DataFrame],
    reg_blocks: Dict[str, pd.DataFrame],
    cls_2_blocks: Dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    """Save classification and regression comparison tables to Excel."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Classification ────────────────────────────────────────────────────────
    cls_path = os.path.join(output_dir, "comparison_cls.xlsx")
    with pd.ExcelWriter(cls_path, engine="xlsxwriter") as writer:
        full_df = pd.concat(list(cls_blocks.values()))
        full_df.to_excel(writer, sheet_name="Classification", index=True)
        _apply_excel_formatting(writer, "Classification", full_df)
    print(f"  Saved: {cls_path}")

    # ── Regression ────────────────────────────────────────────────────────────
    reg_path = os.path.join(output_dir, "comparison_reg.xlsx")
    with pd.ExcelWriter(reg_path, engine="xlsxwriter") as writer:
        full_df = pd.concat(list(reg_blocks.values()))
        full_df.to_excel(writer, sheet_name="Regression", index=True)
        _apply_excel_formatting(writer, "Regression", full_df)
    print(f"  Saved: {reg_path}")

    # ── Classification 2 ─────────────────────────────────────────────────────
    cls_2_path = os.path.join(output_dir, "comparison_cls_2.xlsx")
    with pd.ExcelWriter(cls_2_path, engine="xlsxwriter") as writer:
        full_df = pd.concat(list(cls_2_blocks.values()))
        full_df.to_excel(writer, sheet_name="Classification 2", index=True)
        _apply_excel_formatting(writer, "Classification 2", full_df)
    print(f"  Saved: {cls_2_path}")
