import argparse
import csv
import json
import logging
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote


EXPECTED_LABELS = {
    "abdominal_wall.nii.gz",
    "bca.nii.gz",
    "vertebra_l2.nii.gz",
    "vertebra_l3.nii.gz",
    "vertebra_l4.nii.gz",
    "paravertebral.nii.gz",
    "psoas.nii.gz",
}

LABEL_TO_TARGET = {
    "abdominal wall muscles": "abdominal_wall.nii.gz",
    "abdominal wall muscle": "abdominal_wall.nii.gz",
    "bca": "bca.nii.gz",
    "l2 vertebra": "vertebra_l2.nii.gz",
    "l3 vertebra": "vertebra_l3.nii.gz",
    "l4 vertebra": "vertebra_l4.nii.gz",
    "paravertebral muscles": "paravertebral.nii.gz",
    "paravertebral muscle": "paravertebral.nii.gz",
    "psoas": "psoas.nii.gz",
}


@dataclass
class ActionRecord:
    patient_source: str
    patient_dest: str
    source_path: str
    dest_path: str
    label_raw: str
    target_filename: str
    operation: str
    status: str
    note: str


@dataclass
class PatientSummary:
    patient_source: str
    patient_dest: str
    expected_masks: int
    resolved_masks: int
    transferred_masks: int
    missing_targets: list[str]
    duplicate_targets: list[str]
    unknown_files: list[str]


def patient_name_from_source(source_patient_dir: Path) -> str:
    match = re.fullmatch(r"S2_P(\d+)", source_patient_dir.name)
    if not match:
        raise ValueError(
            f"Unexpected patient folder name: {source_patient_dir.name!r}. "
            "Expected format like S2_P1, S2_P2, ..., S2_P30."
        )
    return f"Pat{int(match.group(1))}"


def clean_label_candidate(raw_name: str) -> str:
    name = raw_name
    if name.endswith(".nii.gz"):
        name = name[:-7]

    name = unquote(name)
    # Remove trailing pattern like "_1_<hash>" or "_12_<hash>"
    match = re.match(r"^(?P<label>.+?)_\d+_[A-Za-z0-9]+$", name)
    if match:
        name = match.group("label")

    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def target_filename_from_source_name(filename: str) -> tuple[str | None, str]:
    label_candidate = clean_label_candidate(filename)
    target = LABEL_TO_TARGET.get(label_candidate)
    return target, label_candidate


def discover_masks(source_patient_dir: Path) -> list[Path]:
    return sorted(p for p in source_patient_dir.rglob("*.nii.gz") if p.is_file())


def build_patient_plan(
    source_patient_dir: Path,
) -> tuple[dict[str, Path], list[tuple[Path, str]], dict[str, list[Path]]]:
    resolved: dict[str, list[Path]] = defaultdict(list)
    unknown_files: list[tuple[Path, str]] = []

    for src_path in discover_masks(source_patient_dir):
        target_name, label_candidate = target_filename_from_source_name(src_path.name)
        if target_name is None:
            unknown_files.append((src_path, label_candidate))
            continue
        resolved[target_name].append(src_path)

    chosen: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for target_name, paths in sorted(resolved.items()):
        if len(paths) == 1:
            chosen[target_name] = paths[0]
        else:
            duplicates[target_name] = sorted(paths)

    return chosen, unknown_files, duplicates


def ensure_dir(path: Path, dry_run: bool) -> None:
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def transfer_file(
    src: Path, dst: Path, mode: str, overwrite: bool, dry_run: bool
) -> tuple[str, str]:
    if dst.exists():
        if overwrite:
            status = "overwrite" if not dry_run else "would_overwrite"
        else:
            return (
                "skip_existing",
                "Destination file already exists. Use --overwrite to replace it.",
            )

    if dry_run:
        return f"would_{mode}", ""

    if overwrite and dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return mode, ""


def align_one_patient(
    source_patient_dir: Path,
    dest_root: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> tuple[list[ActionRecord], PatientSummary]:
    patient_dest_name = patient_name_from_source(source_patient_dir)
    patient_dest_dir = dest_root / patient_dest_name
    ensure_dir(patient_dest_dir, dry_run=dry_run)

    chosen, unknown_files, duplicates = build_patient_plan(source_patient_dir)

    actions: list[ActionRecord] = []
    transferred_count = 0

    for target_name in sorted(chosen):
        src_path = chosen[target_name]
        dst_path = patient_dest_dir / target_name
        operation, note = transfer_file(
            src=src_path,
            dst=dst_path,
            mode=mode,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        if operation in {
            "copy",
            "move",
            "would_copy",
            "would_move",
            "overwrite",
            "would_overwrite",
        }:
            transferred_count += 1

        _, label_candidate = target_filename_from_source_name(src_path.name)
        actions.append(
            ActionRecord(
                patient_source=source_patient_dir.name,
                patient_dest=patient_dest_name,
                source_path=str(src_path),
                dest_path=str(dst_path),
                label_raw=label_candidate,
                target_filename=target_name,
                operation=operation,
                status="ok" if note == "" else "warning",
                note=note,
            )
        )

    for target_name, dup_paths in sorted(duplicates.items()):
        actions.append(
            ActionRecord(
                patient_source=source_patient_dir.name,
                patient_dest=patient_dest_name,
                source_path=" | ".join(str(p) for p in dup_paths),
                dest_path=str(patient_dest_dir / target_name),
                label_raw=target_name,
                target_filename=target_name,
                operation="none",
                status="duplicate_source",
                note="Multiple source files matched the same standardized target. No file was transferred for this target.",
            )
        )

    for src_path, label_candidate in unknown_files:
        actions.append(
            ActionRecord(
                patient_source=source_patient_dir.name,
                patient_dest=patient_dest_name,
                source_path=str(src_path),
                dest_path="",
                label_raw=label_candidate,
                target_filename="",
                operation="none",
                status="unknown_label",
                note="File name did not match any expected segmentation label.",
            )
        )

    missing_targets = sorted(EXPECTED_LABELS - set(chosen) - set(duplicates))
    for target_name in missing_targets:
        actions.append(
            ActionRecord(
                patient_source=source_patient_dir.name,
                patient_dest=patient_dest_name,
                source_path="",
                dest_path=str(patient_dest_dir / target_name),
                label_raw="",
                target_filename=target_name,
                operation="none",
                status="missing",
                note="Expected segmentation mask was not found in the source tree.",
            )
        )

    summary = PatientSummary(
        patient_source=source_patient_dir.name,
        patient_dest=patient_dest_name,
        expected_masks=len(EXPECTED_LABELS),
        resolved_masks=len(chosen),
        transferred_masks=transferred_count,
        missing_targets=missing_targets,
        duplicate_targets=sorted(duplicates),
        unknown_files=[str(p) for p, _ in unknown_files],
    )
    return actions, summary


def iter_patient_dirs(source_root: Path, patient_names: Iterable[str] | None = None) -> list[Path]:
    if patient_names is None:
        dirs = [
            p
            for p in sorted(source_root.iterdir())
            if p.is_dir() and re.fullmatch(r"S2_P\d+", p.name)
        ]
        return dirs

    selected: list[Path] = []
    for name in patient_names:
        path = source_root / name
        if not path.exists():
            logging.warning("Source patient folder is missing: %s", path)
        selected.append(path)
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def align_all_patients(
    source_root: Path,
    dest_root: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
    report_dir: Path,
    patient_names: Iterable[str] | None = None,
) -> dict:
    all_actions: list[ActionRecord] = []
    all_summaries: list[PatientSummary] = []

    for source_patient_dir in iter_patient_dirs(source_root, patient_names=patient_names):
        if not source_patient_dir.exists():
            all_actions.append(
                ActionRecord(
                    patient_source=source_patient_dir.name,
                    patient_dest="",
                    source_path=str(source_patient_dir),
                    dest_path="",
                    label_raw="",
                    target_filename="",
                    operation="none",
                    status="missing_patient_folder",
                    note="Source patient folder does not exist.",
                )
            )
            continue

        actions, summary = align_one_patient(
            source_patient_dir=source_patient_dir,
            dest_root=dest_root,
            mode=mode,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        all_actions.extend(actions)
        all_summaries.append(summary)

    action_rows = [asdict(a) for a in all_actions]
    summary_rows = [asdict(s) for s in all_summaries]
    for row in summary_rows:
        row["missing_targets"] = ";".join(row["missing_targets"])
        row["duplicate_targets"] = ";".join(row["duplicate_targets"])
        row["unknown_files"] = ";".join(row["unknown_files"])

    write_csv(report_dir / "alignment_actions.csv", action_rows)
    write_csv(report_dir / "alignment_summary.csv", summary_rows)

    summary_payload = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "mode": mode,
        "dry_run": dry_run,
        "overwrite": overwrite,
        "patients_processed": len(all_summaries),
        "total_action_rows": len(all_actions),
        "patients_with_missing": sum(bool(s.missing_targets) for s in all_summaries),
        "patients_with_duplicates": sum(bool(s.duplicate_targets) for s in all_summaries),
        "patients_with_unknown_files": sum(bool(s.unknown_files) for s in all_summaries),
        "summaries": [asdict(s) for s in all_summaries],
    }
    write_json(report_dir / "alignment_summary.json", summary_payload)
    return summary_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align cohort-2 liver segmentation masks from the unsorted export into "
            "standardized patient folders and file names."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/unsorted/p773545"
        ),
        help="Root folder containing source patient folders",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/labelsAll"),
        help="Destination root folder containing standardized patient folders Pat1, Pat2, ...",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("../output/01_data_statistics"),
        help="Directory where CSV and JSON reports will be written.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Transfer mode. Use copy first to validate, then switch to move if needed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files when they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the alignment and write reports without modifying files.",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Optional explicit list of source patient folders to process, e.g. S2_P1 S2_P2.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {args.source_root}")

    payload = align_all_patients(
        source_root=args.source_root,
        dest_root=args.dest_root,
        mode=args.mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        report_dir=args.report_dir,
        patient_names=args.patients,
    )

    logging.info("Patients processed: %s", payload["patients_processed"])
    logging.info("Patients with missing masks: %s", payload["patients_with_missing"])
    logging.info("Patients with duplicate labels: %s", payload["patients_with_duplicates"])
    logging.info("Patients with unknown files: %s", payload["patients_with_unknown_files"])
    logging.info("Reports written to: %s", args.report_dir.resolve())


if __name__ == "__main__":
    main()
