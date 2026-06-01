from clean_data import clean_data


# ============================================================
# 1. TESTA DATARENSNING
# ============================================================
#
# Datarensningen ska kunna hitta problem utan att ändra datasetet. Om rensningen
# aktiveras ska problemfiler flyttas till dataset/_removed i stället för att raderas.

# ------------------------------------------------------------
# 1.1 Rent dataset
# ------------------------------------------------------------
def test_clean_data_reports_no_problems_for_clean_dataset(tiny_dataset):
    report = clean_data(tiny_dataset)

    assert report.problem_count == 0
    assert report.moved_files == []


# ------------------------------------------------------------
# 1.2 Bild utan etikettfil
# ------------------------------------------------------------
def test_clean_data_reports_missing_label_without_moving_file(tiny_dataset):
    label_path = tiny_dataset / "dataset" / "train" / "labels" / "sample_0.txt"
    image_path = tiny_dataset / "dataset" / "train" / "images" / "sample_0.jpg"
    label_path.unlink()

    report = clean_data(tiny_dataset, apply_changes=False)

    assert report.missing_labels == [image_path]
    assert image_path.exists()
    assert report.moved_files == []


# ------------------------------------------------------------
# 1.3 Flytta problemfil till karantän
# ------------------------------------------------------------
def test_clean_data_moves_missing_label_image_when_apply_is_true(tiny_dataset):
    label_path = tiny_dataset / "dataset" / "train" / "labels" / "sample_0.txt"
    image_path = tiny_dataset / "dataset" / "train" / "images" / "sample_0.jpg"
    quarantine_path = tiny_dataset / "dataset" / "_removed" / "missing_label" / "train" / "images" / "sample_0.jpg"
    label_path.unlink()

    report = clean_data(tiny_dataset, apply_changes=True)

    assert report.missing_labels == [image_path]
    assert not image_path.exists()
    assert quarantine_path.exists()
    assert report.moved_files == [quarantine_path]


# ------------------------------------------------------------
# 1.4 Etikettfil utan bild
# ------------------------------------------------------------
def test_clean_data_reports_orphan_label(tiny_dataset):
    orphan_label = tiny_dataset / "dataset" / "train" / "labels" / "orphan.txt"
    orphan_label.write_text("0 0.5 0.5 0.25 0.25\n")

    report = clean_data(tiny_dataset)

    assert report.orphan_labels == [orphan_label]
