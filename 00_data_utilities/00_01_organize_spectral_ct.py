import argparse
import re
import shutil
from pathlib import Path
from typing import Optional


SOURCE_DIR_DEFAULT = Path("/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/unsorted")
TARGET_DIR_DEFAULT = Path(
    "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaDataLiver/imagesAll"
)

KNOWN_PHASES = {"arteriell", "nativ", "spaet", "venous"}
NON_DEFINED_PHASE = "non_def"

# Matches the patient prefix at the start of the filename
PATIENT_PATTERN = re.compile(r"^(S2P(?P<patient_id>\d+))(?=-|$)", re.IGNORECASE)


def parse_patient_id(filename: str) -> Optional[str]:
    """
    Extract numeric patient ID from the beginning of filename.
    Returns the numeric string, e.g. "15", or None if not found.
    """
    match = PATIENT_PATTERN.match(filename)
    if not match:
        return None
    return str(int(match.group("patient_id")))  # normalize "001" -> "1"


def parse_phase(filename: str) -> str:
    """
    Find a known phase token anywhere in the filename stem.
    If not found, return NON_DEFINED_PHASE.
    """
    # Remove .nii.gz safely
    stem = filename[:-7] if filename.endswith(".nii.gz") else Path(filename).stem
    tokens = stem.split("-")

    for token in tokens:
        token_lower = token.lower()
        if token_lower in KNOWN_PHASES:
            return token_lower

    return NON_DEFINED_PHASE


def is_nii_gz(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(".nii.gz")


def build_destination(target_root: Path, patient_id: str, phase: str, filename: str) -> Path:
    patient_folder = f"Pat{patient_id}"
    return target_root / patient_folder / phase / filename


def move_or_copy_file(src: Path, dst: Path, copy: bool = False, dry_run: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise FileExistsError(f"Target file already exists: {dst}")

    action = "COPY" if copy else "MOVE"
    print(f"{action}: {src} -> {dst}")

    if dry_run:
        return

    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def sort_images(
    source_dir: Path,
    target_dir: Path,
    copy: bool = False,
    dry_run: bool = False,
    skip_unmatched: bool = False,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    moved_or_copied = 0
    skipped_non_nii = 0
    skipped_unmatched = 0

    for path in sorted(source_dir.iterdir()):
        if not is_nii_gz(path):
            skipped_non_nii += 1
            continue

        total_files += 1
        patient_id = parse_patient_id(path.name)

        if patient_id is None:
            message = f"Could not extract patient ID from filename: {path.name}"
            if skip_unmatched:
                print(f"SKIP: {message}")
                skipped_unmatched += 1
                continue
            raise ValueError(message)

        phase = parse_phase(path.name)
        destination = build_destination(target_dir, patient_id, phase, path.name)

        move_or_copy_file(path, destination, copy=copy, dry_run=dry_run)
        moved_or_copied += 1

    print("\nSummary")
    print("-------")
    print(f".nii.gz files processed: {total_files}")
    print(f"Moved/Copied:           {moved_or_copied}")
    print(f"Skipped non-.nii.gz:    {skipped_non_nii}")
    print(f"Skipped unmatched:      {skipped_unmatched}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sort spectral CT .nii.gz files by patient and phase."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_DIR_DEFAULT, help="Source directory")
    parser.add_argument("--target", type=Path, default=TARGET_DIR_DEFAULT, help="Target directory")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview operations without changing files"
    )
    parser.add_argument(
        "--skip-unmatched",
        action="store_true",
        help="Skip files where patient ID cannot be parsed instead of raising an error",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    sort_images(
        source_dir=args.source,
        target_dir=args.target,
        copy=args.copy,
        dry_run=args.dry_run,
        skip_unmatched=args.skip_unmatched,
    )


if __name__ == "__main__":
    main()
