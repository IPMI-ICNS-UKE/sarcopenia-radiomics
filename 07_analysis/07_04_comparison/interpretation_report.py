import os
import numpy as np
from typing import Dict
from datetime import datetime

import config_extended as config


# Helper
def _interpret_pair(
    new_method: str,
    baseline_method: str,
    pvalues: Dict,
    task: str,
    use_corrected: bool = True,
) -> str:
    """
    Return a one-line interpretation for a single comparison pair.
    For classification: lower p-value with higher AUC = better.
    For regression: lower p-value with lower RMSE = better.
    """
    key = (new_method, baseline_method)
    if key not in pvalues:
        return f"  [N/A] Comparison not available."

    p_raw  = pvalues[key].get("raw")
    p_corr = pvalues[key].get("corrected")
    p_disp = p_corr if use_corrected else p_raw

    new_display  = config.METHOD_DISPLAY_NAMES[new_method]
    base_display = config.METHOD_DISPLAY_NAMES[baseline_method]

    if p_disp is None or np.isnan(p_disp):
        return (
            f"  [N/A] p-value could not be computed "
            f"(insufficient data or alignment issue)."
        )

    p_label = f"p={p_disp:.4f}" if p_disp >= 0.001 else "p<0.001"
    correction_note = (
        f"({config.CORRECTION_METHOD} corrected)" if use_corrected else "(uncorrected)"
    )

    if p_disp < config.ALPHA:
        direction = (
            "superior" if (task == "cls") or (task == "cls_2") else "superior (lower RMSE)"
        )
        conclusion = (
            f"  ✓ SIGNIFICANT: '{new_display}' is statistically {direction} to "
            f"'{base_display}' [{p_label} {correction_note}, α={config.ALPHA}]."
        )
    else:
        conclusion = (
            f"  ✗ NOT SIGNIFICANT: No statistically significant difference between "
            f"'{new_display}' and '{base_display}' "
            f"[{p_label} {correction_note}, α={config.ALPHA}]."
        )
    return conclusion


# Main report builder
def generate_interpretation_report(
    all_pvalues: Dict,
    output_dir: str,
    use_corrected: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "pvalue_interpretation.txt")

    lines = []
    lines.append("=" * 80)
    lines.append("SarcoSpect Study — P-value Interpretation Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Correction method: {config.CORRECTION_METHOD}")
    lines.append(f"Significance threshold (α): {config.ALPHA}")
    lines.append(f"P-values displayed: {'corrected' if use_corrected else 'raw (uncorrected)'}")
    lines.append("=" * 80)
    lines.append("")

    # ── Overview of correction rationale ──────────────────────────────────────
    lines.append("MULTIPLE COMPARISON CORRECTION RATIONALE")
    lines.append("-" * 40)
    lines.append(
        "Each block (task × data subset) contains 6 pairwise comparisons:\n"
        "  3 new methods × 2 baselines (SMI, MRA).\n"
        "These comparisons are treated as a single 'family' for correction purposes.\n"
        "The Benjamini-Hochberg (BH) FDR correction is recommended because:\n"
        "  - Tests are correlated (same patients, related models)\n"
        "  - Bonferroni would be overly conservative\n"
        "  - BH is standard in biomedical/radiological research\n"
        "  - Controls False Discovery Rate at the specified α level.\n"
        "Correction method can be changed in config.py (CORRECTION_METHOD).\n"
    )

    task_labels = {
        "cls": "CLASSIFICATION TASK (sarcopenia_composite) — DeLong Test on AUC",
        "reg": "REGRESSION TASK (hand_grip_cont) — Wilcoxon Signed-Rank Test on Absolute Errors",
        "cls_2": "CLASSIFICATION TASK (chair_rise) — DeLong Test on AUC",
    }
    task_better = {
        "cls": "higher AUC",
        "reg": "lower RMSE",
        "cls_2": "higher AUC",
    }

    for task in ["cls", "reg", "cls_2"]:
        lines.append("")
        lines.append("=" * 80)
        lines.append(task_labels[task])
        lines.append(
            f"(A method is 'better' if it shows {task_better[task]} with p < {config.ALPHA})"
        )
        lines.append("=" * 80)

        for subset_key in ["train", "test_1", "test_2"]:
            subset_display = config.DATA_SUBSETS[task][subset_key]["display_name"]
            pvalues = all_pvalues.get(task, {}).get(subset_key, {})

            lines.append("")
            lines.append(f"  ── {subset_display} ──")
            lines.append("")

            for new_method, baseline_method in config.COMPARISON_PAIRS:
                new_display  = config.METHOD_DISPLAY_NAMES[new_method]
                base_display = config.METHOD_DISPLAY_NAMES[baseline_method]
                lines.append(f"  [{new_display}] vs [{base_display}]")
                interp = _interpret_pair(
                    new_method, baseline_method, pvalues, task, use_corrected
                )
                lines.append(interp)

                # Raw p-value footnote
                if pvalues:
                    key = (new_method, baseline_method)
                    if key in pvalues:
                        p_raw = pvalues[key].get("raw")
                        p_corr = pvalues[key].get("corrected")
                        if p_raw is not None:
                            lines.append(
                                f"    [raw p={p_raw:.4f} | corrected p={p_corr:.4f}]"
                            )
                lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Saved: {filepath}")
