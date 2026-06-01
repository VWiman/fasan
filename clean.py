import shutil
from dataclasses import dataclass
from pathlib import Path

from config import CLEAN_IMAGE_DIRS, CLEAN_IMAGE_PATTERNS, CLEAN_TARGETS


# ============================================================
# 1. DATASTRUKTUR
# ============================================================
#
# CleanArtifactsReport samlar vad som togs bort och vad som hoppades över. Det
# gör rensningen lättare att kontrollera efteråt.

@dataclass
class CleanArtifactsReport:
    removed_paths: list[Path]
    skipped_paths: list[Path]

    @property
    def removed_count(self):
        return len(self.removed_paths)

    @property
    def skipped_count(self):
        return len(self.skipped_paths)


# ============================================================
# 2. SÄKERHETSKONTROLLER
# ============================================================
#
# Rensningen ska bara ta bort tydliga artefaktmappar eller artefaktfiler. Den ska
# inte kunna ta bort projektroten, datasetet eller andra för breda sökvägar.

# ------------------------------------------------------------
# 2.1 Kontrollera om sökvägen får tas bort
# ------------------------------------------------------------
def is_safe_clean_target(path):
    path = Path(path)
    normalized_parts = set(path.parts)

    if path in [Path("."), Path("/")]:
        return False

    if "dataset" in normalized_parts:
        return False

    if len(path.parts) < 2:
        return False

    return True


# ------------------------------------------------------------
# 2.2 Kontrollera om bildsökvägen ligger i en output-mapp
# ------------------------------------------------------------
def is_safe_output_image(path, image_dirs=CLEAN_IMAGE_DIRS):
    path = Path(path)

    if not is_safe_clean_target(path):
        return False

    for image_dir in image_dirs:
        image_dir = Path(image_dir)

        try:
            path.relative_to(image_dir)
            return True
        except ValueError:
            continue

    return False


# ============================================================
# 3. RADERA ARTEFAKTER
# ============================================================
#
# Varje mål kan vara en mapp eller en fil. Mappar tas bort rekursivt och filer
# tas bort direkt. Sökvägar som inte finns räknas som överhoppade. Utöver dessa
# mål kan bildfiler under output-mappar tas bort separat.

# ------------------------------------------------------------
# 3.1 Ta bort en sökväg
# ------------------------------------------------------------
def remove_path(path):
    path = Path(path)

    if not path.exists():
        return False

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    return True


# ------------------------------------------------------------
# 3.2 Hitta output-bilder
# ------------------------------------------------------------
def find_output_images(image_dirs=CLEAN_IMAGE_DIRS, image_patterns=CLEAN_IMAGE_PATTERNS):
    image_paths = []

    for image_dir in image_dirs:
        image_dir = Path(image_dir)

        if not image_dir.exists():
            continue

        for image_pattern in image_patterns:
            image_paths.extend(sorted(image_dir.rglob(image_pattern)))

    return sorted(set(image_paths))


# ------------------------------------------------------------
# 3.3 Rensa alla mål
# ------------------------------------------------------------
def clean_artifacts(targets=CLEAN_TARGETS, image_dirs=CLEAN_IMAGE_DIRS, image_patterns=CLEAN_IMAGE_PATTERNS):
    removed_paths = []
    skipped_paths = []

    for target in targets:
        target = Path(target)

        if not is_safe_clean_target(target):
            skipped_paths.append(target)
            continue

        was_removed = remove_path(target)

        if was_removed:
            removed_paths.append(target)
        else:
            skipped_paths.append(target)

    for image_path in find_output_images(image_dirs, image_patterns):
        if not is_safe_output_image(image_path, image_dirs):
            skipped_paths.append(image_path)
            continue

        was_removed = remove_path(image_path)

        if was_removed:
            removed_paths.append(image_path)
        else:
            skipped_paths.append(image_path)

    return CleanArtifactsReport(
        removed_paths=removed_paths,
        skipped_paths=skipped_paths,
    )


# ============================================================
# 4. RAPPORT
# ============================================================
#
# Rapporten skrivs ut i terminalen så att det tydligt syns vilka gamla
# träningsartefakter som rensades bort.

# ------------------------------------------------------------
# 4.1 Skriv ut rapport
# ------------------------------------------------------------
def print_clean_artifacts_report(report):
    print("Rensning av träningsartefakter klar")
    print(f"Borttagna sökvägar: {report.removed_count}")
    print(f"Överhoppade sökvägar: {report.skipped_count}")

    if report.removed_paths:
        print("\nBorttaget:")
        for path in report.removed_paths:
            print(f"- {path}")

    if report.skipped_paths:
        print("\nÖverhoppat:")
        for path in report.skipped_paths:
            print(f"- {path}")


# ============================================================
# 5. KÖR SOM SKRIPT
# ============================================================
#
# När filen körs direkt används CLEAN_TARGETS från config.py. Det gör att samma
# rensningsmål används varje gång utan CLI-flaggor.

if __name__ == "__main__":
    report = clean_artifacts()
    print_clean_artifacts_report(report)
