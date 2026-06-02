"""
run_grad_cam.py
---------------
Laddar den sparade modellen och kör Grad-CAM + IoU-utvärdering på testdatan.
Kör detta istället för hela pipeline.py om modellen redan är tränad.

Användning:
    python run_grad_cam.py
"""

import tensorflow as tf

from config import (
    CHECKPOINT_OUTPUT_DIR,
    DATASET_PATH,
    GRAD_CAM_OUTPUT_DIR,
    GRAD_CAM_SAMPLE_COUNT,
    MODEL_FILE_NAME,
)
from grad_cam import evaluate_grad_cam_iou, save_grad_cam_report, save_grad_cam_samples
from load_data import load_data


def main():
    # ------------------------------------------------------------
    # 1. Ladda tränad modell
    # ------------------------------------------------------------
    model_path = CHECKPOINT_OUTPUT_DIR / MODEL_FILE_NAME
    print(f"Laddar modell: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # ------------------------------------------------------------
    # 2. Ladda testdata
    # ------------------------------------------------------------
    _, _, test_data = load_data(DATASET_PATH)
    print(f"\nTestdata: {test_data.samples} bilder ({test_data.human_count} med människa)")

    # ------------------------------------------------------------
    # 3. Beräkna Grad-CAM IoU på testbilder med människa
    # ------------------------------------------------------------
    print("\nKör Grad-CAM IoU-utvärdering...")
    grad_cam_eval = evaluate_grad_cam_iou(model, test_data)

    print("\nGrad-CAM IoU-utvärdering")
    print(f"Utvärderade bilder: {grad_cam_eval['total']}")
    print(f"Genomsnittligt IoU: {grad_cam_eval['mean_iou']:.4f}")
    print(f"Detektionsandel:    {grad_cam_eval['detection_rate']:.2%}")

    # ------------------------------------------------------------
    # 4. Spara rapport
    # ------------------------------------------------------------
    report_path = save_grad_cam_report(grad_cam_eval)
    print(f"\nRapport sparades: {report_path}")

    # ------------------------------------------------------------
    # 5. Spara exempelbilder
    # ------------------------------------------------------------
    print("\nSparar exempelbilder...")
    sample_paths = save_grad_cam_samples(
        model,
        test_data,
        output_dir=GRAD_CAM_OUTPUT_DIR,
        sample_count=GRAD_CAM_SAMPLE_COUNT,
    )

    print(f"Grad-CAM exempelbilder sparades: {len(sample_paths)} filer")
    for path in sample_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
