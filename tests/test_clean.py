from clean import clean_artifacts, find_output_images, is_safe_clean_target, is_safe_output_image


# ============================================================
# 1. SÄKERHET
# ============================================================
#
# Rensningsskriptet ska inte acceptera för breda sökvägar eller datasetmappen som
# mål. Det skyddar projektdata från att tas bort av misstag.

# ------------------------------------------------------------
# 1.1 Stoppa osäkra sökvägar
# ------------------------------------------------------------
def test_is_safe_clean_target_rejects_unsafe_paths():
    assert is_safe_clean_target(".") is False
    assert is_safe_clean_target("/") is False
    assert is_safe_clean_target("dataset/train") is False


# ------------------------------------------------------------
# 1.2 Tillåt output-sökvägar
# ------------------------------------------------------------
def test_is_safe_clean_target_allows_output_paths():
    assert is_safe_clean_target("output/training") is True
    assert is_safe_clean_target("output/checkpoints") is True


# ------------------------------------------------------------
# 1.3 Tillåt bara bilder i output-mappar
# ------------------------------------------------------------
def test_is_safe_output_image_only_allows_output_images():
    assert is_safe_output_image("output/eda/class_distribution.png") is True
    assert is_safe_output_image("outputs/training/history.png") is True
    assert is_safe_output_image("dataset/train/images/sample.jpg") is False


# ============================================================
# 2. RENSNING
# ============================================================
#
# Rensningen ska ta bort befintliga artefakter och hoppa över mål som inte finns.

# ------------------------------------------------------------
# 2.1 Ta bort output-mappar
# ------------------------------------------------------------
def test_clean_artifacts_removes_existing_targets(tmp_path):
    training_dir = tmp_path / "output" / "training"
    checkpoint_dir = tmp_path / "output" / "checkpoints"
    training_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    (training_dir / "history.png").write_text("graf")
    (checkpoint_dir / "model.keras").write_text("modell")

    report = clean_artifacts(targets=[training_dir, checkpoint_dir], image_dirs=[])

    assert report.removed_count == 2
    assert not training_dir.exists()
    assert not checkpoint_dir.exists()


# ------------------------------------------------------------
# 2.2 Hoppa över saknade mål
# ------------------------------------------------------------
def test_clean_artifacts_skips_missing_targets(tmp_path):
    missing_dir = tmp_path / "output" / "training"

    report = clean_artifacts(targets=[missing_dir], image_dirs=[])

    assert report.removed_count == 0
    assert report.skipped_count == 1


# ------------------------------------------------------------
# 2.3 Hitta output-bilder
# ------------------------------------------------------------
def test_find_output_images_finds_images_recursively(tmp_path):
    output_dir = tmp_path / "output"
    eda_dir = output_dir / "eda"
    eda_dir.mkdir(parents=True)
    image_path = eda_dir / "class_distribution.png"
    text_path = eda_dir / "notes.txt"
    image_path.write_text("graf")
    text_path.write_text("text")

    image_paths = find_output_images(image_dirs=[output_dir])

    assert image_paths == [image_path]


# ------------------------------------------------------------
# 2.4 Ta bort output-bilder
# ------------------------------------------------------------
def test_clean_artifacts_removes_output_images(tmp_path):
    output_dir = tmp_path / "output"
    eda_dir = output_dir / "eda"
    eda_dir.mkdir(parents=True)
    image_path = eda_dir / "sample_images.png"
    text_path = eda_dir / "notes.txt"
    image_path.write_text("graf")
    text_path.write_text("text")

    report = clean_artifacts(targets=[], image_dirs=[output_dir])

    assert report.removed_count == 1
    assert not image_path.exists()
    assert text_path.exists()
