import numpy as np
import pytest

from load_data import calculate_class_weights, load_data, make_batches, read_binary_label, resolve_dataset_path


# ============================================================
# 1. TESTA ETIKETTER
# ============================================================
#
# Etikettfunktionen ska tolka tomma YOLO-filer som klass 0 och filer med minst
# en giltig YOLO-rad som klass 1.

# ------------------------------------------------------------
# 1.1 Tom etikettfil
# ------------------------------------------------------------
def test_read_binary_label_empty_file_returns_zero(tmp_path):
    label_path = tmp_path / "empty.txt"
    label_path.write_text("")

    assert read_binary_label(label_path) == 0


# ------------------------------------------------------------
# 1.2 Etikettfil med människa
# ------------------------------------------------------------
def test_read_binary_label_with_human_row_returns_one(tmp_path):
    label_path = tmp_path / "human.txt"
    label_path.write_text("0 0.5 0.5 0.25 0.25\n")

    assert read_binary_label(label_path) == 1


# ------------------------------------------------------------
# 1.3 Ogiltig klass
# ------------------------------------------------------------
def test_read_binary_label_rejects_unknown_class(tmp_path):
    label_path = tmp_path / "wrong_class.txt"
    label_path.write_text("1 0.5 0.5 0.25 0.25\n")

    with pytest.raises(ValueError):
        read_binary_label(label_path)


# ============================================================
# 2. TESTA DATASETSTRUKTUR
# ============================================================
#
# Dataladdningen ska hitta dataset-mappen automatiskt och returnera train,
# valid och test med rätt binära etiketter.

# ------------------------------------------------------------
# 2.1 Hitta dataset-mappen
# ------------------------------------------------------------
def test_resolve_dataset_path_uses_dataset_folder(tiny_dataset):
    assert resolve_dataset_path(tiny_dataset) == tiny_dataset / "dataset"


# ------------------------------------------------------------
# 2.2 Ladda splitar
# ------------------------------------------------------------
def test_load_data_returns_binary_splits(tiny_dataset):
    train_data, valid_data, test_data = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)

    assert train_data.samples == 3
    assert train_data.labels == [0, 1, 1]
    assert train_data.human_count == 2
    assert train_data.no_human_count == 1

    assert valid_data.labels == [0, 1]
    assert test_data.labels == [1, 0]


# ============================================================
# 3. TESTA CLASS WEIGHT
# ============================================================
#
# Class weights ska ge högre vikt till den klass som har färre exempel.

# ------------------------------------------------------------
# 3.1 Beräkna class weights
# ------------------------------------------------------------
def test_calculate_class_weights_gives_higher_weight_to_smaller_class(tiny_dataset):
    train_data, _, _ = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)
    class_weights = calculate_class_weights(train_data, mode="balanced")

    assert class_weights[0] == 1.5
    assert class_weights[1] == 0.75


# ------------------------------------------------------------
# 3.2 Beräkna mildare class weights
# ------------------------------------------------------------
def test_calculate_class_weights_can_use_sqrt_balanced_mode(tiny_dataset):
    train_data, _, _ = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)
    class_weights = calculate_class_weights(train_data, mode="sqrt_balanced")

    assert class_weights[0] == pytest.approx(np.sqrt(1.5))
    assert class_weights[1] == pytest.approx(np.sqrt(0.75))


# ------------------------------------------------------------
# 3.3 Stäng av class weights
# ------------------------------------------------------------
def test_calculate_class_weights_can_be_disabled(tiny_dataset):
    train_data, _, _ = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)

    assert calculate_class_weights(train_data, mode="none") is None


# ============================================================
# 4. TESTA BATCHER
# ============================================================
#
# Batcharna ska ge bilder som fyrdimensionella tensorer och etiketter som en
# endimensionell vektor.

# ------------------------------------------------------------
# 4.1 Skapa första batchen
# ------------------------------------------------------------
def test_make_batches_returns_images_and_labels(tiny_dataset):
    train_data, _, _ = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)
    images, labels = next(make_batches(train_data))

    assert images.shape == (2, 32, 32, 3)
    assert labels.shape == (2,)
    assert labels.dtype == np.float32
    assert labels.tolist() == [0.0, 1.0]
    assert np.max(images) <= 1.0
    assert np.min(images) >= 0.0
