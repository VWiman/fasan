import os
from pathlib import Path

# Matplotlib behöver ibland skriva cachefiler. En lokal cachemapp gör att grafer
# fungerar även om den globala matplotlib-mappen inte är skrivbar.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from load_data import CLASS_NAMES, load_image


DEFAULT_FIGSIZE = (10, 5)
BAR_COLORS = ["#5b8def", "#e45757"]


# ============================================================
# 1. GEMENSAMMA HJÄLPFUNKTIONER
# ============================================================
#
# Dessa funktioner används av flera grafer. De håller återkommande logik samlad,
# till exempel hur figurer sparas eller hur klassantal räknas.

# ------------------------------------------------------------
# 1.1 Spara eller visa figur
# ------------------------------------------------------------
def save_or_show(fig, save_path=None, show=True):
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------------
# 1.2 Räkna klasser
# ------------------------------------------------------------
def get_class_counts(split_data):
    return [split_data.no_human_count, split_data.human_count]


# ============================================================
# 2. EDA: KLASSFÖRDELNING
# ============================================================
#
# Klassfördelning visar hur många bilder som finns i varje klass. Det är viktigt
# för att förstå om datasetet är balanserat eller om modellen riskerar att lära sig
# den vanligaste klassen för starkt.

# ------------------------------------------------------------
# 2.1 Rita klassfördelning per datasetdel
# ------------------------------------------------------------
def plot_class_distribution(split_data_list, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    x = np.arange(len(split_data_list))
    width = 0.35
    no_human_counts = [split_data.no_human_count for split_data in split_data_list]
    human_counts = [split_data.human_count for split_data in split_data_list]
    split_names = [split_data.name for split_data in split_data_list]

    ax.bar(x - width / 2, no_human_counts, width, label=CLASS_NAMES[0], color=BAR_COLORS[0])
    ax.bar(x + width / 2, human_counts, width, label=CLASS_NAMES[1], color=BAR_COLORS[1])

    ax.set_title("Klassfördelning per datasetdel")
    ax.set_xlabel("Datasetdel")
    ax.set_ylabel("Antal bilder")
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_or_show(fig, save_path, show)

    return fig, ax


# ============================================================
# 3. EDA: EXEMPELBILDER
# ============================================================
#
# Exempelbilder används för att kontrollera att bilder och etiketter verkar rimliga
# innan modellträning börjar.

# ------------------------------------------------------------
# 3.1 Rita ett urval av bilder
# ------------------------------------------------------------
def plot_sample_images(split_data, sample_count=6, save_path=None, show=True):
    sample_count = min(sample_count, split_data.samples)
    image_paths = split_data.image_paths[:sample_count]
    labels = split_data.labels[:sample_count]

    columns = min(3, sample_count)
    rows = int(np.ceil(sample_count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, image_path, label in zip(axes, image_paths, labels):
        image = load_image(image_path, split_data.image_size)
        ax.imshow(image)
        ax.set_title(f"{split_data.name}: {CLASS_NAMES[int(label)]}")
        ax.axis("off")

    for ax in axes[sample_count:]:
        ax.axis("off")

    fig.suptitle("Exempelbilder från datasetet")
    fig.tight_layout()
    save_or_show(fig, save_path, show)

    return fig, axes


# ============================================================
# 4. EDA: DATAREDUKTION
# ============================================================
#
# Bildstorleken reduceras innan CNN-träning. Grafen visar skillnaden mellan
# originalbildens pixelantal och den storlek som används i modellen.

# ------------------------------------------------------------
# 4.1 Rita bildstorlek före och efter reduktion
# ------------------------------------------------------------
def plot_image_size_reduction(original_size, reduced_size, save_path=None, show=True):
    original_pixels = original_size[0] * original_size[1]
    reduced_pixels = reduced_size[0] * reduced_size[1]
    labels = [f"Original\n{original_size[0]}x{original_size[1]}", f"CNN-input\n{reduced_size[0]}x{reduced_size[1]}"]
    values = [original_pixels, reduced_pixels]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=BAR_COLORS)
    ax.set_title("Datareduktion genom bildstorlek")
    ax.set_ylabel("Antal pixlar per bild")
    ax.grid(axis="y", alpha=0.25)

    for index, value in enumerate(values):
        ax.text(index, value, f"{value:,}".replace(",", " "), ha="center", va="bottom")

    save_or_show(fig, save_path, show)

    return fig, ax


# ============================================================
# 5. UTVÄRDERING: TRÄNINGSHISTORIK
# ============================================================
#
# Träningshistorik visar hur loss och metrics förändras över epoker. Den används
# för att se om modellen lär sig, fastnar eller börjar överanpassa.

# ------------------------------------------------------------
# 5.1 Rita loss och metrics över tid
# ------------------------------------------------------------
def plot_training_history(history, save_path=None, show=True):
    history_data = history.history if hasattr(history, "history") else history
    metric_names = [name for name in history_data.keys() if not name.startswith("val_")]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4))
    axes = np.array(axes).reshape(-1)

    for ax, metric_name in zip(axes, metric_names):
        ax.plot(history_data[metric_name], label=f"träning {metric_name}")

        validation_name = f"val_{metric_name}"
        if validation_name in history_data:
            ax.plot(history_data[validation_name], label=f"validering {metric_name}")

        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(alpha=0.25)

    fig.tight_layout()
    save_or_show(fig, save_path, show)

    return fig, axes


# ============================================================
# 6. UTVÄRDERING: CONFUSION MATRIX
# ============================================================
#
# Confusion matrix visar hur många prediktioner som hamnar i rätt och fel klass.
# För binär klassificering blir matrisen 2x2.

# ------------------------------------------------------------
# 6.1 Rita confusion matrix
# ------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, threshold=0.5, save_path=None, show=True):
    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_pred) >= threshold).astype(int)

    matrix = np.zeros((2, 2), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[true_label, predicted_label] += 1

    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predikterad klass")
    ax.set_ylabel("Sann klass")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    for row in range(2):
        for column in range(2):
            ax.text(column, row, matrix[row, column], ha="center", va="center", color="black")

    fig.tight_layout()
    save_or_show(fig, save_path, show)

    return fig, ax
