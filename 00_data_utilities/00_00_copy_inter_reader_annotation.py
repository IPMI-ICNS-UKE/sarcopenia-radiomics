from pathlib import Path
import shutil
import re
from urllib.parse import unquote


# CONFIG
# Root folder that contains patient folders
EXPORT_ROOT = Path(
    "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/export_testData/kar59430"
)

# Destination root with folders like:
# Pat9, Pat10, Pat90, ...
DEST_ROOT = Path(
    "/home/gkolokolnikov/PhD_project/vault_data/SarcopeniaData/labelsStabilityTest"
)

# Annotator folder mapping
ANNOTATOR_MAP = {
    "p100709": "j",   # Jenni
    "p389615": "i",   # Isabel
}

# Relevant structures and their destination filename stems
STRUCTURE_MAP = {
    "Abdominal Wall Muscles": "abdominal_wall",
    "Paravertebral Muscles": "paravertebral",
    "Psoas": "psoas",
}


# HELPERS
def export_patient_to_dest_patient(export_patient_name: str) -> str | None:
    """
    Convert names
    """
    match = re.search(r"_P0*(\d+)_", export_patient_name)
    if not match:
        return None
    patient_num = int(match.group(1))
    return f"Pat{patient_num}"


def is_manual_annotation(filename: str) -> bool:
    """
    Assumption:
    manual annotation files contain '_1_' in the name.
    """
    return "_1_" in filename


def get_structure_key_from_filename(filename: str) -> str | None:
    """
    Example filenames:
      Abdominal%20Wall%20Muscles_1_xxx.nii.gz
      Paravertebral%20Muscles_1_xxx.nii.gz
      Psoas_1_xxx.nii.gz

    Decode URL escapes and match against the known structures.
    """
    decoded = unquote(filename)

    for structure_name in STRUCTURE_MAP:
        if decoded.startswith(structure_name + "_"):
            return structure_name

    return None


def find_annotator_folders(patient_folder: Path) -> dict[str, Path]:
    """
    Recursively find annotator folders under a patient folder.
    Returns dict like:
        {"p100709": Path(...), "p389615": Path(...)}
    """
    found = {}
    for p in patient_folder.rglob("*"):
        if p.is_dir() and p.name in ANNOTATOR_MAP:
            found[p.name] = p
    return found


# MAIN
def main():
    if not EXPORT_ROOT.exists():
        raise FileNotFoundError(f"EXPORT_ROOT does not exist: {EXPORT_ROOT}")

    if not DEST_ROOT.exists():
        raise FileNotFoundError(f"DEST_ROOT does not exist: {DEST_ROOT}")

    copied = []
    skipped = []

    # Iterate through patient folders
    for export_patient_dir in sorted(EXPORT_ROOT.iterdir()):
        if not export_patient_dir.is_dir():
            continue

        export_patient_name = export_patient_dir.name
        dest_patient_name = export_patient_to_dest_patient(export_patient_name)

        if dest_patient_name is None:
            skipped.append(f"[SKIP] Could not parse patient id from {export_patient_name}")
            continue

        dest_patient_dir = DEST_ROOT / dest_patient_name
        if not dest_patient_dir.exists():
            skipped.append(
                f"[SKIP] Destination patient folder does not exist: {dest_patient_dir}"
            )
            continue

        annotator_dirs = find_annotator_folders(export_patient_dir)
        if not annotator_dirs:
            skipped.append(f"[SKIP] No annotator folders found for {export_patient_name}")
            continue

        print(f"\nProcessing {export_patient_name} -> {dest_patient_name}")

        for annotator_folder_name, annotator_suffix in ANNOTATOR_MAP.items():
            annotator_dir = annotator_dirs.get(annotator_folder_name)

            if annotator_dir is None:
                skipped.append(
                    f"[SKIP] Annotator folder {annotator_folder_name} not found for {export_patient_name}"
                )
                continue

            print(f"  Annotator: {annotator_folder_name} -> suffix '{annotator_suffix}'")

            for nifti_file in annotator_dir.glob("*.nii.gz"):
                filename = nifti_file.name

                # Keep only manual annotations (_1_)
                if not is_manual_annotation(filename):
                    continue

                structure_key = get_structure_key_from_filename(filename)
                if structure_key is None:
                    continue

                dest_stem = STRUCTURE_MAP[structure_key]
                dest_filename = f"{dest_stem}_{annotator_suffix}.nii.gz"
                dest_path = dest_patient_dir / dest_filename

                shutil.copy2(nifti_file, dest_path)

                msg = f"[COPY] {nifti_file} -> {dest_path}"
                copied.append(msg)
                print("   ", msg)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Copied files: {len(copied)}")
    print(f"Skipped notes: {len(skipped)}")

    if skipped:
        print("\nSkipped / warnings:")
        for msg in skipped:
            print(msg)


if __name__ == "__main__":
    main()
