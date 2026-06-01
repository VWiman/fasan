import sys
from pathlib import Path

import pytest
from PIL import Image


# ============================================================
# 1. IMPORTVÄG
# ============================================================
#
# Testerna ska kunna importera projektets moduler oavsett hur pytest startas.
# Därför läggs projektroten till i Python-sökvägen.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. TESTDATA
# ============================================================
#
# Testerna använder ett litet temporärt dataset i samma struktur som det riktiga:
# dataset/train, dataset/valid och dataset/test.

# ------------------------------------------------------------
# 2.1 Skapa testbild
# ------------------------------------------------------------
def create_image(image_path, color):
    image = Image.new("RGB", (32, 32), color=color)
    image.save(image_path)


# ------------------------------------------------------------
# 2.2 Skapa en datasetdel
# ------------------------------------------------------------
def create_split(dataset_path, split_name, labels):
    image_dir = dataset_path / split_name / "images"
    label_dir = dataset_path / split_name / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for index, label in enumerate(labels):
        image_path = image_dir / f"sample_{index}.jpg"
        label_path = label_dir / f"sample_{index}.txt"

        create_image(image_path, color=(40 + index * 20, 90, 140))

        if label == 0:
            label_path.write_text("")
        else:
            label_path.write_text("0 0.5 0.5 0.25 0.25\n")


# ------------------------------------------------------------
# 2.3 Skapa komplett minidataset
# ------------------------------------------------------------
@pytest.fixture
def tiny_dataset(tmp_path):
    dataset_path = tmp_path / "dataset"

    create_split(dataset_path, "train", labels=[0, 1, 1])
    create_split(dataset_path, "valid", labels=[0, 1])
    create_split(dataset_path, "test", labels=[1, 0])

    return tmp_path
