import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config import APPLY_DATA_CLEANING, DATASET_PATH, REMOVED_DIR, SPLIT_NAMES
from load_data import (
    get_image_paths,
    get_label_path,
    get_split_dirs,
    read_label_lines,
    resolve_dataset_path,
    validate_yolo_lines,
)


# ============================================================
# 1. DATASTRUKTUR
# ============================================================
#
# CleanReport samlar problem som hittas under datarensningen. Listorna sparas
# separat för att det ska vara tydligt vilken typ av problem som finns.

@dataclass
class CleanReport:
    missing_labels: list[Path]
    orphan_labels: list[Path]
    invalid_labels: list[Path]
    unreadable_images: list[Path]
    moved_files: list[Path]

    @property
    def problem_count(self):
        return (
            len(self.missing_labels)
            + len(self.orphan_labels)
            + len(self.invalid_labels)
            + len(self.unreadable_images)
        )


# ============================================================
# 2. KONTROLLERA FILER
# ============================================================
#
# Varje bild ska ha en matchande etikettfil och varje etikettfil ska höra till
# en bild. Dessutom kontrolleras att bilder kan öppnas och att etiketter följer
# YOLO-formatet.

# ------------------------------------------------------------
# 2.1 Kontrollera om bild kan öppnas
# ------------------------------------------------------------
def image_is_readable(image_path):
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# 2.2 Kontrollera om etikettfil har giltigt YOLO-format
# ------------------------------------------------------------
def label_is_valid(label_path):
    try:
        label_lines = read_label_lines(label_path)
        validate_yolo_lines(label_path, label_lines)
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# 2.3 Hitta etikettfiler utan bild
# ------------------------------------------------------------
def find_orphan_labels(image_paths, label_paths):
    image_stems = {image_path.stem for image_path in image_paths}

    return [label_path for label_path in label_paths if label_path.stem not in image_stems]


# ============================================================
# 3. FLYTTA PROBLEMFILER
# ============================================================
#
# Problemfiler flyttas till dataset/_removed i stället för att tas bort. Det gör
# rensningen spårbar och gör det möjligt att ångra manuellt om något flyttas fel.

# ------------------------------------------------------------
# 3.1 Skapa säker karantänsökväg
# ------------------------------------------------------------
def get_quarantine_path(dataset_path, file_path, reason):
    relative_path = file_path.relative_to(dataset_path)

    return dataset_path / REMOVED_DIR / reason / relative_path


# ------------------------------------------------------------
# 3.2 Flytta en fil till karantän
# ------------------------------------------------------------
def move_to_quarantine(dataset_path, file_path, reason):
    quarantine_path = get_quarantine_path(dataset_path, file_path, reason)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(file_path), str(quarantine_path))

    return quarantine_path


# ------------------------------------------------------------
# 3.3 Flytta en fil om den finns
# ------------------------------------------------------------
def move_if_exists(dataset_path, file_path, reason, moved_files):
    if file_path.exists():
        moved_files.append(move_to_quarantine(dataset_path, file_path, reason))


# ============================================================
# 4. RENGÖR EN DATASETDEL
# ============================================================
#
# En datasetdel är train, valid eller test. Funktionen letar efter problem och
# kan antingen bara rapportera dem eller flytta dem till karantän.

def clean_split(dataset_path, split_name, apply_changes=False):
    image_dir, label_dir = get_split_dirs(dataset_path, split_name)
    image_paths = get_image_paths(image_dir)
    label_paths = sorted(label_dir.glob("*.txt"))

    missing_labels = []
    invalid_labels = []
    unreadable_images = []
    moved_files = []

    for image_path in image_paths:
        label_path = get_label_path(label_dir, image_path)

        if not label_path.exists():
            missing_labels.append(image_path)
        elif not label_is_valid(label_path):
            invalid_labels.append(label_path)

        if not image_is_readable(image_path):
            unreadable_images.append(image_path)

    orphan_labels = find_orphan_labels(image_paths, label_paths)

    if apply_changes:
        for image_path in missing_labels:
            move_if_exists(dataset_path, image_path, "missing_label", moved_files)

        for label_path in orphan_labels:
            move_if_exists(dataset_path, label_path, "orphan_label", moved_files)

        for label_path in invalid_labels:
            image_path = image_dir / f"{label_path.stem}.jpg"
            move_if_exists(dataset_path, image_path, "invalid_label", moved_files)
            move_if_exists(dataset_path, label_path, "invalid_label", moved_files)

        for image_path in unreadable_images:
            label_path = get_label_path(label_dir, image_path)
            move_if_exists(dataset_path, image_path, "unreadable_image", moved_files)
            move_if_exists(dataset_path, label_path, "unreadable_image", moved_files)

    return CleanReport(
        missing_labels=missing_labels,
        orphan_labels=orphan_labels,
        invalid_labels=invalid_labels,
        unreadable_images=unreadable_images,
        moved_files=moved_files,
    )


# ============================================================
# 5. RENGÖR HELA DATASETET
# ============================================================
#
# Hela datasetet kontrolleras genom att samma rensningssteg körs för train,
# valid och test.

def merge_reports(reports):
    return CleanReport(
        missing_labels=sum((report.missing_labels for report in reports), []),
        orphan_labels=sum((report.orphan_labels for report in reports), []),
        invalid_labels=sum((report.invalid_labels for report in reports), []),
        unreadable_images=sum((report.unreadable_images for report in reports), []),
        moved_files=sum((report.moved_files for report in reports), []),
    )


def clean_data(dataset_path=".", apply_changes=False):
    dataset_path = resolve_dataset_path(dataset_path)
    reports = []

    for split_name in SPLIT_NAMES:
        reports.append(clean_split(dataset_path, split_name, apply_changes))

    return merge_reports(reports)


# ============================================================
# 6. SAMMANFATTNING AV RENSNING
# ============================================================
#
# Sammanfattningen gör det enkelt att se om datasetet är rent eller om något
# behöver flyttas innan träning.

def print_clean_report(report):
    print("Datarensning klar")
    print(f"Bilder utan etikettfil: {len(report.missing_labels)}")
    print(f"Etikettfiler utan bild: {len(report.orphan_labels)}")
    print(f"Ogiltiga etikettfiler: {len(report.invalid_labels)}")
    print(f"Bilder som inte kan öppnas: {len(report.unreadable_images)}")
    print(f"Flyttade filer: {len(report.moved_files)}")


# ============================================================
# 7. KÖR SOM EGEN FIL
# ============================================================
#
# Filen kan köras separat. Inställningarna hämtas från config.py så att samma
# beteende används oavsett om rensningen körs separat eller från pipeline.py.

def main():
    report = clean_data(DATASET_PATH, apply_changes=APPLY_DATA_CLEANING)
    print_clean_report(report)


if __name__ == "__main__":
    main()
