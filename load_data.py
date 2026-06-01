from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from config import BATCH_SIZE, CLASS_NAMES, CLASS_WEIGHT_MODE, DEFAULT_DATASET_DIR, IMAGE_SIZE


# ============================================================
# 1. DATASTRUKTUR
# ============================================================
#
# SplitData samlar all information som behövs för en del av datasetet, till exempel
# train, valid eller test. Bildfilerna sparas som sökvägar så att bilderna kan laddas
# först när de behövs i en batch.

@dataclass
class SplitData:
    name: str
    image_paths: list[Path]
    labels: list[int]
    image_size: tuple[int, int]
    batch_size: int

    @property
    def samples(self):
        return len(self.image_paths)

    @property
    def human_count(self):
        return sum(self.labels)

    @property
    def no_human_count(self):
        return self.samples - self.human_count


# ============================================================
# 2. LÄS ETIKETTER
# ============================================================
#
# YOLO-filerna används här för binär klassificering. En tom etikettfil betyder
# klass 0, alltså ingen människa. En fil med minst en giltig YOLO-rad betyder
# klass 1, alltså att en människa finns i bilden.

# ------------------------------------------------------------
# 2.1 Läs rader från etikettfil
# ------------------------------------------------------------
def read_label_lines(label_path):
    text = label_path.read_text().strip()

    if not text:
        return []

    return text.splitlines()


# ------------------------------------------------------------
# 2.2 Kontrollera YOLO-format
# ------------------------------------------------------------
def validate_yolo_lines(label_path, label_lines):
    for line in label_lines:
        values = line.split()

        if len(values) != 5:
            raise ValueError(f"Fel YOLO-format i {label_path}: {line}")

        class_id = int(values[0])
        if class_id != 0:
            raise ValueError(f"Okänd klass i {label_path}: {class_id}")


# ------------------------------------------------------------
# 2.3 Konvertera till binär etikett
# ------------------------------------------------------------
def read_binary_label(label_path):
    label_lines = read_label_lines(label_path)

    if not label_lines:
        return 0

    validate_yolo_lines(label_path, label_lines)

    return 1


# ============================================================
# 3. LADDA BILDER
# ============================================================
#
# Bilderna laddas som RGB, skalas till samma storlek och normaliseras till
# intervallet 0 till 1. Det ger en form som passar en CNN-modell.

def load_image(image_path, image_size=IMAGE_SIZE):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    image = np.array(image, dtype=np.float32) / 255.0

    return image


# ============================================================
# 4. HITTA DATASET
# ============================================================
#
# Datasetet ligger i en dataset-mapp för att inte krocka med filer som pipeline.py
# eller tests/. Om dataset-mappen finns används den automatiskt.

# ------------------------------------------------------------
# 4.1 Hitta datasetets rotmapp
# ------------------------------------------------------------
def resolve_dataset_path(dataset_path="."):
    dataset_path = Path(dataset_path)
    nested_dataset_path = dataset_path / DEFAULT_DATASET_DIR

    if nested_dataset_path.exists():
        return nested_dataset_path

    return dataset_path


# ------------------------------------------------------------
# 4.2 Hämta mappar för bilder och etiketter
# ------------------------------------------------------------
def get_split_dirs(dataset_path, split_name):
    image_dir = dataset_path / split_name / "images"
    label_dir = dataset_path / split_name / "labels"

    return image_dir, label_dir


# ------------------------------------------------------------
# 4.3 Lista bildfiler
# ------------------------------------------------------------
def get_image_paths(image_dir):
    return sorted(image_dir.glob("*.jpg"))


# ------------------------------------------------------------
# 4.4 Koppla bild till etikettfil
# ------------------------------------------------------------
def get_label_path(label_dir, image_path):
    return label_dir / f"{image_path.stem}.txt"


# ============================================================
# 5. SAMLA DATA
# ============================================================
#
# Här byggs en komplett SplitData för train, valid eller test. Funktionen samlar
# bildsökvägar och skapar binära etiketter från YOLO-filerna.

# ------------------------------------------------------------
# 5.1 Samla etiketter
# ------------------------------------------------------------
def collect_labels(image_paths, label_dir):
    labels = []

    for image_path in image_paths:
        label_path = get_label_path(label_dir, image_path)

        if not label_path.exists():
            raise FileNotFoundError(f"Saknar etikettfil: {label_path}")

        labels.append(read_binary_label(label_path))

    return labels


# ------------------------------------------------------------
# 5.2 Samla en datasetdel
# ------------------------------------------------------------
def collect_split(dataset_path, split_name, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    image_dir, label_dir = get_split_dirs(dataset_path, split_name)
    image_paths = get_image_paths(image_dir)
    labels = collect_labels(image_paths, label_dir)

    return SplitData(
        name=split_name,
        image_paths=image_paths,
        labels=labels,
        image_size=image_size,
        batch_size=batch_size,
    )


# ============================================================
# 6. SKAPA BATCHER
# ============================================================
#
# Batchar gör att bilderna kan läsas in stegvis i stället för att hela datasetet
# laddas in i minnet på en gång.

def make_batches(split_data):
    batch_size = split_data.batch_size

    for start in range(0, split_data.samples, batch_size):
        end = start + batch_size
        image_batch = []
        label_batch = []

        for image_path, label in zip(split_data.image_paths[start:end], split_data.labels[start:end]):
            image = load_image(image_path, split_data.image_size)

            image_batch.append(image)
            label_batch.append(label)

        yield np.array(image_batch, dtype=np.float32), np.array(label_batch, dtype=np.float32)


# ============================================================
# 7. LADDA DATA
# ============================================================
#
# Detta är huvudfunktionen för resten av projektet. Den returnerar train, valid
# och test i samma struktur.

def load_data(dataset_path=".", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    dataset_path = resolve_dataset_path(dataset_path)

    train_data = collect_split(dataset_path, "train", image_size, batch_size)
    valid_data = collect_split(dataset_path, "valid", image_size, batch_size)
    test_data = collect_split(dataset_path, "test", image_size, batch_size)

    return train_data, valid_data, test_data


# ============================================================
# 8. SAMMANFATTNING AV DATASET
# ============================================================
#
# Klassvikter används vid träning för att kompensera om en klass är mycket
# vanligare än den andra. Balanced ger full kompensation, medan sqrt_balanced ger
# en mildare kompensation som ofta blir stabilare när datasetet är obalanserat.

# ------------------------------------------------------------
# 8.1 Beräkna class_weight
# ------------------------------------------------------------
def calculate_balanced_class_weights(split_data):
    no_human_count = split_data.no_human_count
    human_count = split_data.human_count

    if no_human_count == 0 or human_count == 0:
        raise ValueError("Class weights kräver att båda klasserna finns i datasetet.")

    total_count = split_data.samples
    class_count = len(CLASS_NAMES)

    return {
        0: total_count / (class_count * no_human_count),
        1: total_count / (class_count * human_count),
    }


# ------------------------------------------------------------
# 8.2 Beräkna class_weight enligt valt läge
# ------------------------------------------------------------
def calculate_class_weights(split_data, mode=CLASS_WEIGHT_MODE):
    balanced_weights = calculate_balanced_class_weights(split_data)

    if mode == "balanced":
        return balanced_weights

    if mode == "sqrt_balanced":
        return {
            class_index: float(np.sqrt(class_weight))
            for class_index, class_weight in balanced_weights.items()
        }

    if mode == "none":
        return None

    raise ValueError(f"Okänt class_weight-läge: {mode}")


# ============================================================
# 9. SAMMANFATTNING AV DATASET
# ============================================================
#
# Sammanfattningen används som en snabb kontroll innan modellträning.

def print_dataset_summary(train_data, valid_data, test_data):
    print("Datasetet är laddat")
    print(f"Klasser: {CLASS_NAMES}")

    for split_data in [train_data, valid_data, test_data]:
        print(
            f"{split_data.name}: "
            f"{split_data.samples} bilder, "
            f"{split_data.human_count} bilder med människa, "
            f"{split_data.no_human_count} bilder utan människa"
        )
