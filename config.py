from pathlib import Path


# ============================================================
# 1. DATASET
# ============================================================
#
# DATASET_PATH pekar på projektroten. Om det finns en dataset-mapp under denna
# sökväg används den automatiskt av dataladdningen.

DATASET_PATH = Path(".")
DEFAULT_DATASET_DIR = "dataset"


# ============================================================
# 2. DATARENSNING
# ============================================================
#
# APPLY_DATA_CLEANING styr om problemfiler ska flyttas till dataset/_removed.
# False betyder att problem bara rapporteras. True betyder att rensningen faktiskt
# flyttar filer som saknar matchande bild/etikett eller har ogiltigt format.

APPLY_DATA_CLEANING = False
SPLIT_NAMES = ["train", "valid", "test"]
REMOVED_DIR = "_removed"


# ============================================================
# 3. BILDER OCH BATCHER
# ============================================================
#
# IMAGE_SIZE styr vilken storlek bilderna får när de laddas in. BATCH_SIZE styr
# hur många bilder som skickas vidare åt gången i batch-generatorn.

IMAGE_SIZE = (320, 320)
BATCH_SIZE = 8


# ============================================================
# 4. KLASSER
# ============================================================
#
# Klassnamnen används både vid dataladdning, sammanfattning och grafer. Etiketterna
# är binära: 0 betyder ingen människa och 1 betyder människa finns i bilden.

CLASS_NAMES = ["no human", "human"]


# ============================================================
# 5. OUTPUT
# ============================================================
#
# OUTPUT_DIR samlar filer som skapas av pipeline. Underkatalogerna skiljer på
# EDA, träningsgrafer och sparade modeller så att resultaten blir lättare att hitta.

OUTPUT_DIR = Path("output")
EDA_OUTPUT_DIR = OUTPUT_DIR / "eda"
TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training"
CHECKPOINT_OUTPUT_DIR = OUTPUT_DIR / "checkpoints"
EDA_SAMPLE_COUNT = 6
CREATE_EDA_IN_PIPELINE = True


# ============================================================
# 6. TRÄNING
# ============================================================
#
# Träningsparametrarna ligger här för att modellen ska kunna justeras utan att
# pipelinekoden behöver ändras. EPOCHS styr hur många varv modellen tränas över
# träningsdatan och LEARNING_RATE styr hur stora optimeringssteg som tas. Ett
# lägre learning rate ger långsammare men ofta stabilare träning.

EPOCHS = 20
LEARNING_RATE = 1e-4
PREDICTION_THRESHOLD = 0.5
SHUFFLE_TRAINING_DATA = True
RANDOM_SEED = 42
CLASS_WEIGHT_MODE = "balanced"
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 4
USE_REDUCE_LR_ON_PLATEAU = True
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 2
MIN_LEARNING_RATE = 5e-6


# ============================================================
# 7. MODELLSTRUKTUR
# ============================================================
#
# CNN_CONV_FILTERS styr hur många convolution-lager modellen får och hur många
# filter varje lager använder. CNN_DENSE_UNITS styr dense-lager efter
# GlobalAveragePooling2D. Dessa listor gör modellen enkel att justera utan att
# ändra själva modellfunktionen.

CNN_CONV_FILTERS = [32, 64, 128]
CNN_DENSE_UNITS = [128, 128]
CNN_DROPOUT_RATE = 0.25


# ============================================================
# 8. DATA AUGMENTATION
# ============================================================
#
# Augmentation ska användas på träningsdata för att skapa variation utan att
# ändra valid- eller testdata. Värdena nedan är medvetet milda eftersom
# search-and-rescue-bilder fortfarande ska se realistiska ut efter transformation.

USE_DATA_AUGMENTATION = True
AUGMENT_HORIZONTAL_FLIP = True
AUGMENT_ROTATION_FACTOR = 0.05
AUGMENT_CONTRAST_FACTOR = 0.10
AUGMENT_BRIGHTNESS_FACTOR = 0.20
AUGMENT_HUE_FACTOR = 0.03
AUGMENT_SATURATION_FACTOR = 0.50


# ============================================================
# 9. FILNAMN FÖR TRÄNINGSRESULTAT
# ============================================================
#
# Dessa namn används när träningen sparar modell, grafer och textbaserad
# utvärdering.

MODEL_FILE_NAME = "search_and_rescue_cnn.keras"
TRAINING_HISTORY_FILE_NAME = "training_history.png"
CONFUSION_MATRIX_FILE_NAME = "confusion_matrix.png"
CLASSIFICATION_REPORT_FILE_NAME = "classification_report.txt"


# ============================================================
# 10. RENSNING AV TRÄNINGSARTEFAKTER
# ============================================================
#
# CLEAN_TARGETS anger vilka filer eller mappar som tas bort när clean.py körs.
# Datasetet ska inte ligga här. Standard är att rensa modell-checkpoints och
# träningsresultat från tidigare körningar.
#
# CLEAN_IMAGE_DIRS och CLEAN_IMAGE_PATTERNS används för att även ta bort sparade
# bildfiler från output-mapparna, till exempel grafer från EDA.

CLEAN_TARGETS = [
    CHECKPOINT_OUTPUT_DIR,
    TRAINING_OUTPUT_DIR,
]

CLEAN_IMAGE_DIRS = [
    OUTPUT_DIR,
    Path("outputs"),
]

CLEAN_IMAGE_PATTERNS = [
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.webp",
]
