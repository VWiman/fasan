import numpy as np
import config

from load_data import load_data
from training import (
    ImageSequence,
    create_classification_report,
    create_training_callbacks,
    create_training_output_dirs,
    get_best_epoch,
    predict_classes,
    save_classification_report,
    train_cnn_model,
)


# ============================================================
# 1. TRÄNINGSBATCHER
# ============================================================
#
# Träningsbatcherna ska ge bilder och etiketter i samma format som Keras får
# under riktig CNN-träning.

# ------------------------------------------------------------
# 1.1 Skapa sequence från minidataset
# ------------------------------------------------------------
def test_image_sequence_returns_image_and_label_batches(tiny_dataset):
    train_data, _, _ = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)
    sequence = ImageSequence(train_data)
    images, labels = sequence[0]

    assert len(sequence) == 2
    assert images.shape == (2, 32, 32, 3)
    assert labels.shape == (2,)
    assert images.dtype == np.float32
    assert labels.dtype == np.float32


# ------------------------------------------------------------
# 1.2 Blanda träningsordning
# ------------------------------------------------------------
def test_image_sequence_can_shuffle_between_epochs(tiny_dataset):
    train_data, _, _ = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)
    sequence = ImageSequence(train_data, shuffle=True, seed=1)
    first_order = sequence.indexes.copy()

    sequence.on_epoch_end()

    assert sorted(sequence.indexes.tolist()) == [0, 1, 2]
    assert sequence.indexes.tolist() != first_order.tolist()


# ------------------------------------------------------------
# 1.3 Augmentera träningsbatch
# ------------------------------------------------------------
def test_image_sequence_can_augment_training_batches(tiny_dataset):
    train_data, _, _ = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)
    sequence = ImageSequence(train_data, augment=True)
    images, labels = sequence[0]

    assert images.shape == (2, 32, 32, 3)
    assert labels.shape == (2,)
    assert images.dtype == np.float32
    assert np.max(images) <= 1.0
    assert np.min(images) >= 0.0


# ------------------------------------------------------------
# 1.4 Använd augmentation bara på train
# ------------------------------------------------------------
def test_train_cnn_model_only_augments_training_sequence(tiny_dataset):
    train_data, valid_data, _ = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)
    captured_sequences = {}

    class DummyModel:
        def fit(self, train_sequence, validation_data, epochs, class_weight, callbacks):
            captured_sequences["train"] = train_sequence
            captured_sequences["valid"] = validation_data
            return {"loss": [0.5], "val_loss": [0.6]}

    train_cnn_model(DummyModel(), train_data, valid_data, class_weights=None, epochs=1)

    assert captured_sequences["train"].augment is config.USE_DATA_AUGMENTATION
    assert captured_sequences["valid"].augment is False


# ------------------------------------------------------------
# 1.5 Kontrollera augmentation-inställningar
# ------------------------------------------------------------
def test_augmentation_config_removes_zoom_and_adds_saturation():
    assert not hasattr(config, "AUGMENT_ZOOM_FACTOR")
    assert hasattr(config, "AUGMENT_SATURATION_FACTOR")


# ============================================================
# 2. PREDIKTIONER
# ============================================================
#
# Sannolikheter ska kunna göras om till binära klasser med samma tröskel som
# används i träningspipelinen.

# ------------------------------------------------------------
# 2.1 Konvertera sannolikheter
# ------------------------------------------------------------
def test_predict_classes_uses_threshold():
    probabilities = np.array([0.2, 0.5, 0.8])
    predictions = predict_classes(probabilities, threshold=0.5)

    assert predictions.tolist() == [0, 1, 1]


# ============================================================
# 3. CLASSIFICATION REPORT
# ============================================================
#
# Rapporten ska innehålla de viktigaste måtten för båda klasserna och kunna
# sparas till output-mappen.

# ------------------------------------------------------------
# 3.1 Skapa classification report
# ------------------------------------------------------------
def test_create_classification_report_contains_binary_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    report = create_classification_report(y_true, y_pred)

    assert "Classification report" in report
    assert "no human" in report
    assert "human" in report
    assert "accuracy" in report
    assert "weighted avg" in report


# ------------------------------------------------------------
# 3.2 Spara classification report
# ------------------------------------------------------------
def test_save_classification_report_writes_file(tmp_path):
    report_path = save_classification_report("test report", training_dir=tmp_path)

    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8") == "test report"


# ============================================================
# 4. OUTPUT-MAPPAR
# ============================================================
#
# Träningspipelinen behöver kunna skapa mapparna för modell och resultat innan
# filer sparas.

# ------------------------------------------------------------
# 4.1 Skapa output-mappar
# ------------------------------------------------------------
def test_create_training_output_dirs_creates_directories(tmp_path):
    checkpoint_dir, training_dir = create_training_output_dirs(
        checkpoint_dir=tmp_path / "checkpoints",
        training_dir=tmp_path / "training",
    )

    assert checkpoint_dir.exists()
    assert training_dir.exists()


# ============================================================
# 5. CALLBACKS
# ============================================================
#
# Träningen använder callbacks för att minska risken att den sista epoken sparas
# om valideringsresultatet redan har blivit sämre.

# ------------------------------------------------------------
# 5.1 Skapa tränings-callbacks
# ------------------------------------------------------------
def test_create_training_callbacks_contains_stability_callbacks():
    callbacks = create_training_callbacks()
    callback_names = [type(callback).__name__ for callback in callbacks]

    assert "EarlyStopping" in callback_names
    assert "ReduceLROnPlateau" in callback_names


# ------------------------------------------------------------
# 5.2 Hitta bästa epoch från valideringsloss
# ------------------------------------------------------------
def test_get_best_epoch_uses_lowest_validation_loss():
    history = {
        "val_loss": [0.7, 0.5, 0.6],
        "val_accuracy": [0.5, 0.8, 0.7],
    }

    best_epoch, best_value = get_best_epoch(history, metric_name="val_loss")

    assert best_epoch == 2
    assert best_value == np.float32(0.5)
