import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Configuration
LOGGER = logging.getLogger("rename_cohort2_images")

# Expected filename patterns in cohort 2, for example:
#   S2P1-venous-MuscleFat.nii.gz
#   S2P12-nativ-40keV.nii.gz
#   S2P7-00009D51-Iodine.nii.gz
#   S2P5-MuscleFat.nii.gz
#
# Goal:
#   Pat1-MuscleFat.nii.gz
#   Pat12-40keV.nii.gz
#   Pat7-Iodine.nii.gz
#   Pat5-MuscleFat.nii.gz

SOURCE_PATTERN = re.compile(
    r"^S2P(?P<patient_id>\d+)-(?:(?P<phase>[^-]+)-)?(?P<map_type>.+)\.nii\.gz$",
    flags=re.IGNORECASE,
)


# Data structures
@dataclass(frozen=True)
class RenameAction:
    source: Path
    destination: Path
    patient_id_from_folder: int
    patient_id_from_file: int
    phase: str | None
    map_type: str


# Core logic
def parse_patient_id_from_folder(folder_name: str) -> int:
    """Extract numeric patient ID from a folder name such as 'Pat12'."""
    match = re.fullmatch(r"Pat(\d+)", folder_name)
    if not match:
        raise ValueError(f"Invalid patient folder name: {folder_name}")
    return int(match.group(1))


def parse_source_filename(filename: str) -> tuple[int, str | None, str] | None:
    """
    Parse a cohort-2 source filename.

    Returns
    -------
    tuple[int, str | None, str] | None
        (patient_id, phase, map_type) if the filename matches the expected pattern,
        otherwise None.
    """
    match = SOURCE_PATTERN.fullmatch(filename)
    if not match:
        return None

    patient_id = int(match.group("patient_id"))
    phase = match.group("phase")
    map_type = match.group("map_type")
    return patient_id, phase, map_type


def build_destination_name(patient_id: int, map_type: str) -> str:
    """Build the unified output filename."""
    return f"Pat{patient_id}-{map_type}.nii.gz"


def iter_candidate_files(patient_dir: Path) -> Iterable[Path]:
    """
    Yield only .nii.gz files located directly inside a patient folder.

    Notes
    -----
    - Files inside subfolders such as 'phases/' are intentionally ignored.
    - Directories are skipped.
    """
    for path in sorted(patient_dir.iterdir()):
        if path.is_dir():
            continue
        if path.name.endswith(".nii.gz"):
            yield path


def collect_rename_actions(images_root: Path) -> tuple[list[RenameAction], list[str]]:
    """
    Collect rename actions and warnings without modifying the filesystem.

    Returns
    -------
    actions : list[RenameAction]
    warnings : list[str]
    """
    actions: list[RenameAction] = []
    warnings: list[str] = []

    patient_dirs = sorted(
        path for path in images_root.iterdir()
        if path.is_dir() and re.fullmatch(r"Pat\d+", path.name)
    )

    for patient_dir in patient_dirs:
        folder_patient_id = parse_patient_id_from_folder(patient_dir.name)

        for file_path in iter_candidate_files(patient_dir):
            parsed = parse_source_filename(file_path.name)
            if parsed is None:
                warnings.append(
                    f"Skipping unmatched filename: {file_path}"
                )
                continue

            file_patient_id, phase, map_type = parsed

            if file_patient_id != folder_patient_id:
                warnings.append(
                    "Patient ID mismatch: "
                    f"folder={folder_patient_id}, file={file_patient_id}, path={file_path}"
                )
                continue

            destination_name = build_destination_name(folder_patient_id, map_type)
            destination_path = file_path.with_name(destination_name)

            if destination_path == file_path:
                warnings.append(f"Already normalized, skipping: {file_path}")
                continue

            if destination_path.exists():
                warnings.append(
                    f"Destination already exists, skipping: {destination_path}"
                )
                continue

            actions.append(
                RenameAction(
                    source=file_path,
                    destination=destination_path,
                    patient_id_from_folder=folder_patient_id,
                    patient_id_from_file=file_patient_id,
                    phase=phase,
                    map_type=map_type,
                )
            )

    return actions, warnings


def execute_renames(actions: list[RenameAction], dry_run: bool = True) -> None:
    """Execute or print rename operations."""
    if not actions:
        LOGGER.info("No files to rename.")
        return

    for action in actions:
        if dry_run:
            LOGGER.info("DRY RUN | %s -> %s", action.source.name, action.destination.name)
        else:
            action.source.rename(action.destination)
            LOGGER.info("RENAMED | %s -> %s", action.source.name, action.destination.name)


# CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rename cohort-2 spectral CT NIfTI files from 'S2P...-phase-map.nii.gz' "
            "to 'Pat...-map.nii.gz' while ignoring subfolders such as 'phases'."
        )
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/imagesAll"),
        help="Root directory containing Pat1, Pat2, ... patient folders.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag, the script runs in dry-run mode.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)

    images_root: Path = args.images_root
    if not images_root.exists():
        raise FileNotFoundError(f"Images root does not exist: {images_root}")

    actions, warnings = collect_rename_actions(images_root)

    for warning in warnings:
        LOGGER.warning(warning)

    LOGGER.info("Collected %d rename actions.", len(actions))
    execute_renames(actions=actions, dry_run=not args.apply)

    print("=" * 80)
    print(f"Images root: {images_root}")
    print(f"Planned renames: {len(actions)}")
    print(f"Warnings/skips: {len(warnings)}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
