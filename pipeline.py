from clean_data import clean_data, print_clean_report
from config import APPLY_DATA_CLEANING, CREATE_EDA_IN_PIPELINE, DATASET_PATH, EDA_OUTPUT_DIR, EDA_SAMPLE_COUNT
from eda import create_eda_graphs
from load_data import calculate_class_weights, load_data, make_batches, print_dataset_summary
from model import build_cnn_model
import tensorflow as tf
from training import (
    create_classification_report,
    create_training_output_dirs,
    predict_classes,
    predict_probabilities,
    print_best_epoch_summary,
    save_classification_report,
    save_confusion_matrix_graph,
    save_trained_model,
    save_training_history_graph,
    train_cnn_model,
)

print("TensorFlow-version: ", tf.__version__)
print("TensorFlow-enheter: ", tf.config.list_physical_devices(
    device_type="GPU"
))

def main():
# ============================================================
# 1. RENGÖR DATA
# ============================================================
#
# Kontrollera om datasetet har bilder utan etikettfil eller
# etikettfiler utan bild. I pipeline körs detta som rapportering utan att flytta filer.

    # ------------------------------------------------------------
    # 1.1 Kontrollera datasetet
    # ------------------------------------------------------------
    # APPLY_DATA_CLEANING i config.py styr om problem bara rapporteras eller flyttas
    # till dataset/_removed.
    clean_report = clean_data(DATASET_PATH, apply_changes=APPLY_DATA_CLEANING)
    print_clean_report(clean_report)

# ============================================================
# 2. LADDA DATA
# ============================================================
#
# Säkerställ att datasetet kan läsas in korrekt. Innan en modell byggs kontrolleras antal
# bilder, klassfördelning och formen på en första batch.

    # ------------------------------------------------------------
    # 2.1 Ladda datasetet
    # ------------------------------------------------------------
    # YOLO-etiketter konverteras till binära klasser:
    # 0 betyder ingen människa och 1 betyder människa finns i bilden.
    train_data, valid_data, test_data = load_data(DATASET_PATH)
    print_dataset_summary(train_data, valid_data, test_data)

    # ------------------------------------------------------------
    # 2.2 Beräkna class_weight
    # ------------------------------------------------------------
    # Class weights kompenserar för att datasetet har fler bilder med människa än utan människa.
    class_weights = calculate_class_weights(train_data)
    print(f"Class weights: {class_weights}")

    # ------------------------------------------------------------
    # 2.3 Skapa en första batch
    # ------------------------------------------------------------
    # Kontroll med en första batch att bilder och etiketter får rätt form innan de skickas vidare.
    train_batches = make_batches(train_data)
    images, labels = next(train_batches)
    
    print("\nFörsta batchen")
    print(f"Bilder: {images.shape}")
    print(f"Etiketter: {labels.shape}")
    print(f"Exempel på etiketter: {labels[:10]}")

# ============================================================
# 3. SKAPA EDA-GRAFER
# ============================================================
#
# EDA-grafer skapas innan träning så att klassfördelning, bildstorleksreduktion
# och exempelbilder finns dokumenterade även när gamla output-bilder har rensats.

    # ------------------------------------------------------------
    # 3.1 Skapa grafer från laddad data
    # ------------------------------------------------------------
    # CREATE_EDA_IN_PIPELINE i config.py styr om graferna ska skapas automatiskt
    # när hela pipelinen körs.
    if CREATE_EDA_IN_PIPELINE:
        graph_paths = create_eda_graphs(
            [train_data, valid_data, test_data],
            output_dir=EDA_OUTPUT_DIR,
            sample_count=EDA_SAMPLE_COUNT,
        )

        print("\nEDA-grafer skapade")
        for graph_path in graph_paths:
            print(graph_path)
    else:
        print("\nEDA-grafer hoppades över")

# ============================================================
# 4. BYGG CNN-MODELL
# ============================================================
#
# CNN-modellen tar emot de reducerade bilderna och gör binär klassificering.
# Sista lagret använder sigmoid eftersom uppgiften bara har två utfall:
# ingen människa eller människa finns i bilden.

    # ------------------------------------------------------------
    # 4.1 Skapa modell
    # ------------------------------------------------------------
    # Modellstrukturen ligger i model.py så att pipeline och modellarkitektur
    # kan ändras oberoende av varandra.
    model = build_cnn_model()
    model.summary()

# ============================================================
# 5. TRÄNA MODELL
# ============================================================
#
# Modellen tränas på train-delen och kontrolleras mot valid-delen efter varje
# epoch. Class weights används för att den mindre klassen ska få större påverkan
# under träningen.

    # ------------------------------------------------------------
    # 5.1 Skapa output-mappar
    # ------------------------------------------------------------
    # output/checkpoints används för den sparade modellen och output/training
    # används för grafer och rapporter från träningen.
    create_training_output_dirs()

    # ------------------------------------------------------------
    # 5.2 Kör träning
    # ------------------------------------------------------------
    # Keras returnerar history, vilket innehåller loss och accuracy per epoch.
    history = train_cnn_model(model, train_data, valid_data, class_weights)
    print_best_epoch_summary(history)

# ============================================================
# 6. SPARA MODELL
# ============================================================
#
# Den färdigtränade modellen sparas som en Keras-fil. Den kan senare laddas för
# prediktioner eller vidare träning utan att träningen behöver köras om.

    # ------------------------------------------------------------
    # 6.1 Spara tränad modell
    # ------------------------------------------------------------
    model_path = save_trained_model(model)
    print(f"\nModellen sparades: {model_path}")

# ============================================================
# 7. UTVÄRDERA MODELL
# ============================================================
#
# Test-delen används efter träningen för att få en mer rättvis bild av hur
# modellen fungerar på bilder den inte tränats eller validerats mot.

    # ------------------------------------------------------------
    # 7.1 Prediktera testdata
    # ------------------------------------------------------------
    # Modellen ger sannolikheter som sedan konverteras till klasser med tröskeln
    # från config.py.
    probabilities = predict_probabilities(model, test_data)
    predictions = predict_classes(probabilities)

    # ------------------------------------------------------------
    # 7.2 Skapa classification report
    # ------------------------------------------------------------
    # Rapporten visar precision, recall och f1-score per klass.
    classification_report = create_classification_report(test_data.labels, predictions)
    report_path = save_classification_report(classification_report)

    print("\nClassification report")
    print(classification_report)
    print(f"\nRapporten sparades: {report_path}")

    # ------------------------------------------------------------
    # 7.3 Spara utvärderingsgrafer
    # ------------------------------------------------------------
    # Träningshistorik visar utvecklingen över epoker och confusion matrix visar
    # vilka klasser som blev rätt eller fel på testdata.
    history_path = save_training_history_graph(history)
    confusion_matrix_path = save_confusion_matrix_graph(test_data.labels, probabilities)

    print(f"Träningshistorik sparades: {history_path}")
    print(f"Confusion matrix sparades: {confusion_matrix_path}")

if __name__ == "__main__":
    main()
