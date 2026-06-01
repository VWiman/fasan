import matplotlib

# Agg gör att grafer kan skapas i tester utan att öppna fönster.
matplotlib.use("Agg")

from graphs import (
    get_class_counts,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_image_size_reduction,
    plot_sample_images,
    plot_training_history,
)
from load_data import load_data


# ============================================================
# 1. TESTA EDA-GRAFER
# ============================================================
#
# EDA-graferna ska kunna räkna klassfördelning och spara figurer baserade på
# det temporära testdatasetet.

# ------------------------------------------------------------
# 1.1 Klassantal
# ------------------------------------------------------------
def test_get_class_counts_returns_no_human_and_human_counts(tiny_dataset):
    train_data, _, _ = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)

    assert get_class_counts(train_data) == [1, 2]


# ------------------------------------------------------------
# 1.2 Klassfördelningsgraf
# ------------------------------------------------------------
def test_plot_class_distribution_saves_file(tiny_dataset, tmp_path):
    train_data, valid_data, test_data = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)
    save_path = tmp_path / "class_distribution.png"

    plot_class_distribution([train_data, valid_data, test_data], save_path=save_path, show=False)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


# ------------------------------------------------------------
# 1.3 Bildstorleksreduktion
# ------------------------------------------------------------
def test_plot_image_size_reduction_saves_file(tmp_path):
    save_path = tmp_path / "image_size_reduction.png"

    plot_image_size_reduction((640, 640), (320, 320), save_path=save_path, show=False)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


# ------------------------------------------------------------
# 1.4 Exempelbilder
# ------------------------------------------------------------
def test_plot_sample_images_saves_file(tiny_dataset, tmp_path):
    train_data, _, _ = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)
    save_path = tmp_path / "sample_images.png"

    plot_sample_images(train_data, sample_count=3, save_path=save_path, show=False)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


# ============================================================
# 2. TESTA UTVÄRDERINGSGRAFER
# ============================================================
#
# Utvärderingsgraferna testas med små konstgjorda värden så att de kan valideras
# utan en färdigtränad modell.

# ------------------------------------------------------------
# 2.1 Träningshistorik
# ------------------------------------------------------------
def test_plot_training_history_saves_file(tmp_path):
    history = {
        "loss": [0.8, 0.5, 0.3],
        "accuracy": [0.6, 0.75, 0.85],
        "val_loss": [0.9, 0.6, 0.4],
        "val_accuracy": [0.55, 0.7, 0.8],
    }
    save_path = tmp_path / "history.png"

    plot_training_history(history, save_path=save_path, show=False)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


# ------------------------------------------------------------
# 2.2 Confusion matrix
# ------------------------------------------------------------
def test_plot_confusion_matrix_saves_file(tmp_path):
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.7, 0.8, 0.2]
    save_path = tmp_path / "confusion_matrix.png"

    plot_confusion_matrix(y_true, y_pred, save_path=save_path, show=False)

    assert save_path.exists()
    assert save_path.stat().st_size > 0
