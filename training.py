from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from config import (
    AUGMENT_BRIGHTNESS_FACTOR,
    AUGMENT_CONTRAST_FACTOR,
    AUGMENT_HORIZONTAL_FLIP,
    AUGMENT_HUE_FACTOR,
    AUGMENT_ROTATION_FACTOR,
    AUGMENT_SATURATION_FACTOR,
    CHECKPOINT_OUTPUT_DIR,
    CLASSIFICATION_REPORT_FILE_NAME,
    CLASS_NAMES,
    CONFUSION_MATRIX_FILE_NAME,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    MIN_LEARNING_RATE,
    MODEL_FILE_NAME,
    PREDICTION_THRESHOLD,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SHUFFLE_TRAINING_DATA,
    TRAINING_HISTORY_FILE_NAME,
    TRAINING_OUTPUT_DIR,
    USE_DATA_AUGMENTATION,
    USE_EARLY_STOPPING,
    USE_REDUCE_LR_ON_PLATEAU,
)
from graphs import plot_confusion_matrix, plot_training_history
from load_data import load_image


# ============================================================
# 1. TRÄNINGSBATCHER
# ============================================================
#
# Keras behöver kunna läsa flera epoker från samma dataset. En Sequence skapar
# batcher på begäran och kan därför återanvändas genom hela träningen utan att
# alla bilder behöver ligga i minnet samtidigt. Träningsdata kan blandas mellan
# epoker och augmenteras, medan validation och test alltid lämnas oförändrade.

# ------------------------------------------------------------
# 1.1 Skapa Keras-sequence
# ------------------------------------------------------------
class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, split_data, shuffle=False, augment=False, seed=RANDOM_SEED, **kwargs):
        super().__init__(**kwargs)
        self.split_data = split_data
        self.shuffle = shuffle
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.indexes = np.arange(self.split_data.samples)
        self.rotation_layer = tf.keras.layers.RandomRotation(
            AUGMENT_ROTATION_FACTOR,
            fill_mode="nearest",
            seed=seed,
        )
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.split_data.samples / self.split_data.batch_size))

    def __getitem__(self, index):
        start = index * self.split_data.batch_size
        end = start + self.split_data.batch_size
        batch_indexes = self.indexes[start:end]
        image_batch = []
        label_batch = []

        for sample_index in batch_indexes:
            image_path = self.split_data.image_paths[sample_index]
            label = self.split_data.labels[sample_index]

            image_batch.append(load_image(image_path, self.split_data.image_size))
            label_batch.append(label)

        image_batch = np.array(image_batch, dtype=np.float32)

        if self.augment:
            image_batch = self.augment_images(image_batch)

        return image_batch, np.array(label_batch, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def augment_images(self, image_batch):
        image_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)

        if AUGMENT_HORIZONTAL_FLIP:
            image_batch = tf.image.random_flip_left_right(image_batch)

        if AUGMENT_ROTATION_FACTOR > 0:
            image_batch = self.rotation_layer(image_batch, training=True)

        if AUGMENT_BRIGHTNESS_FACTOR > 0:
            image_batch = tf.image.random_brightness(image_batch, max_delta=AUGMENT_BRIGHTNESS_FACTOR)

        if AUGMENT_CONTRAST_FACTOR > 0:
            image_batch = tf.image.random_contrast(
                image_batch,
                lower=1 - AUGMENT_CONTRAST_FACTOR,
                upper=1 + AUGMENT_CONTRAST_FACTOR,
            )

        if AUGMENT_HUE_FACTOR > 0:
            image_batch = tf.image.random_hue(image_batch, max_delta=AUGMENT_HUE_FACTOR)

        if AUGMENT_SATURATION_FACTOR > 0:
            image_batch = tf.image.random_saturation(
                image_batch,
                lower=1 - AUGMENT_SATURATION_FACTOR,
                upper=1 + AUGMENT_SATURATION_FACTOR,
            )

        image_batch = tf.clip_by_value(image_batch, 0.0, 1.0)

        return image_batch.numpy().astype(np.float32)


# ============================================================
# 2. OUTPUT-MAPPAR
# ============================================================
#
# Träningen skapar flera filer. Dessa mappar skapas innan träning och utvärdering
# så att modellen, graferna och rapporten hamnar på samma förutsägbara plats.

# ------------------------------------------------------------
# 2.1 Skapa mappar
# ------------------------------------------------------------
def create_training_output_dirs(checkpoint_dir=CHECKPOINT_OUTPUT_DIR, training_dir=TRAINING_OUTPUT_DIR):
    checkpoint_dir = Path(checkpoint_dir)
    training_dir = Path(training_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, training_dir


# ============================================================
# 3. TRÄNA MODELL
# ============================================================
#
# Modellen tränas på train-delen och valideras på valid-delen efter varje epoch.
# Class weights skickas med för att minska effekten av obalanserad klassfördelning.
# Callbacks kan stoppa träningen i tid och sänka learning rate när valideringsloss
# inte förbättras.

# ------------------------------------------------------------
# 3.1 Skapa callbacks
# ------------------------------------------------------------
def create_training_callbacks():
    callbacks = []

    if USE_EARLY_STOPPING:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            )
        )

    if USE_REDUCE_LR_ON_PLATEAU:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LEARNING_RATE,
                verbose=1,
            )
        )

    return callbacks


# ------------------------------------------------------------
# 3.2 Kör träning
# ------------------------------------------------------------
def train_cnn_model(model, train_data, valid_data, class_weights, epochs=EPOCHS):
    train_sequence = ImageSequence(
        train_data,
        shuffle=SHUFFLE_TRAINING_DATA,
        augment=USE_DATA_AUGMENTATION,
    )
    valid_sequence = ImageSequence(valid_data, shuffle=False, augment=False)
    callbacks = create_training_callbacks()

    history = model.fit(
        train_sequence,
        validation_data=valid_sequence,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    return history


# ------------------------------------------------------------
# 3.3 Hitta bästa epoch
# ------------------------------------------------------------
def get_best_epoch(history, metric_name="val_loss"):
    history_data = history.history if hasattr(history, "history") else history

    if metric_name not in history_data:
        raise ValueError(f"Saknar metric i träningshistorik: {metric_name}")

    metric_values = np.array(history_data[metric_name], dtype=np.float32)

    if metric_name.endswith("loss"):
        best_index = int(np.argmin(metric_values))
    else:
        best_index = int(np.argmax(metric_values))

    return best_index + 1, float(metric_values[best_index])


# ------------------------------------------------------------
# 3.4 Skriv ut bästa epoch
# ------------------------------------------------------------
def print_best_epoch_summary(history):
    best_epoch, best_value = get_best_epoch(history, metric_name="val_loss")

    print("\nBästa valideringsresultat")
    print(f"Epoch: {best_epoch}")
    print(f"val_loss: {best_value:.4f}")

    if USE_EARLY_STOPPING:
        print("EarlyStopping återställer bästa vikterna innan modellen sparas.")


# ============================================================
# 4. SPARA MODELL
# ============================================================
#
# När träningen är klar sparas hela Keras-modellen. Filen innehåller modellens
# lager, vikter och kompileringsinformation så att den kan laddas igen senare.

# ------------------------------------------------------------
# 4.1 Spara tränad modell
# ------------------------------------------------------------
def save_trained_model(model, checkpoint_dir=CHECKPOINT_OUTPUT_DIR, model_file_name=MODEL_FILE_NAME):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / model_file_name

    model.save(model_path)

    return model_path


# ============================================================
# 5. PREDIKTIONER
# ============================================================
#
# Sigmoid-lagret ger sannolikheter mellan 0 och 1. Dessa görs om till klasser med
# en tröskel där värden över eller lika med tröskeln blir klass 1.

# ------------------------------------------------------------
# 5.1 Prediktera sannolikheter
# ------------------------------------------------------------
def predict_probabilities(model, split_data):
    probabilities = model.predict(ImageSequence(split_data), verbose=0).reshape(-1)

    return probabilities[: split_data.samples]


# ------------------------------------------------------------
# 5.2 Konvertera sannolikheter till klasser
# ------------------------------------------------------------
def predict_classes(probabilities, threshold=PREDICTION_THRESHOLD):
    return (np.array(probabilities) >= threshold).astype(int)


# ============================================================
# 6. CLASSIFICATION REPORT
# ============================================================
#
# Rapporten sammanfattar precision, recall och f1-score för varje klass.
# scikit-learn används eftersom funktionen är standardiserad, tydlig och ger
# samma format som ofta används i maskininlärningsrapporter.


# ------------------------------------------------------------
# 6.1 Skapa classification report
# ------------------------------------------------------------
def create_classification_report(y_true, y_pred, class_names=CLASS_NAMES):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    return f"Classification report\n\n{report}"


# ------------------------------------------------------------
# 6.2 Spara classification report
# ------------------------------------------------------------
def save_classification_report(report_text, training_dir=TRAINING_OUTPUT_DIR):
    training_dir = Path(training_dir)
    training_dir.mkdir(parents=True, exist_ok=True)
    report_path = training_dir / CLASSIFICATION_REPORT_FILE_NAME

    report_path.write_text(report_text, encoding="utf-8")

    return report_path


# ============================================================
# 7. SPARA TRÄNINGSGRAFER
# ============================================================
#
# Träningshistorik och confusion matrix sparas som PNG-filer så att resultatet kan
# användas i rapport, presentation eller vidare analys.

# ------------------------------------------------------------
# 7.1 Spara historikgraf
# ------------------------------------------------------------
def save_training_history_graph(history, training_dir=TRAINING_OUTPUT_DIR):
    history_path = Path(training_dir) / TRAINING_HISTORY_FILE_NAME

    plot_training_history(history, save_path=history_path, show=False)

    return history_path


# ------------------------------------------------------------
# 7.2 Spara confusion matrix
# ------------------------------------------------------------
def save_confusion_matrix_graph(y_true, probabilities, training_dir=TRAINING_OUTPUT_DIR):
    confusion_matrix_path = Path(training_dir) / CONFUSION_MATRIX_FILE_NAME

    plot_confusion_matrix(
        y_true,
        probabilities,
        threshold=PREDICTION_THRESHOLD,
        save_path=confusion_matrix_path,
        show=False,
    )

    return confusion_matrix_path
