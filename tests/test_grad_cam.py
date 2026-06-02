from pathlib import Path

import numpy as np

from grad_cam import cam_bbox_from_heatmap, compute_iou, read_yolo_boxes, save_grad_cam_report


# ============================================================
# 1. CAM-BBOX
# ============================================================
#
# Värmekartan ska kunna tröskas till en normaliserad bounding box.

# ------------------------------------------------------------
# 1.1 Skapa bounding box från värmekarta
# ------------------------------------------------------------
def test_cam_bbox_from_heatmap_returns_normalized_box():
    heatmap = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.8, 0.9],
            [0.0, 0.7, 0.0],
        ],
        dtype=np.float32,
    )

    box = cam_bbox_from_heatmap(heatmap, threshold=0.5)

    assert box == (1 / 3, 1 / 3, 2 / 3, 2 / 3)


# ============================================================
# 2. YOLO-BOXAR
# ============================================================
#
# Endast labels med fulla YOLO-koordinater kan användas för IoU.

# ------------------------------------------------------------
# 2.1 Läs YOLO-box och ignorera förenklad label
# ------------------------------------------------------------
def test_read_yolo_boxes_ignores_simplified_labels(tmp_path):
    label_path = tmp_path / "label.txt"
    label_path.write_text("0\n0 0.5 0.5 0.25 0.25\n")

    boxes = read_yolo_boxes(label_path)

    assert boxes == [(0.375, 0.375, 0.625, 0.625)]


# ============================================================
# 3. IOU
# ============================================================
#
# IoU mäter överlapp mellan två normaliserade boxar.

# ------------------------------------------------------------
# 3.1 Beräkna IoU
# ------------------------------------------------------------
def test_compute_iou_returns_overlap_ratio():
    box_a = (0.0, 0.0, 1.0, 1.0)
    box_b = (0.5, 0.5, 1.0, 1.0)

    assert compute_iou(box_a, box_b) == 0.25


# ============================================================
# 4. RAPPORT
# ============================================================
#
# Rapporten ska redovisa både utvärderade bilder och bilder som hoppats över.

# ------------------------------------------------------------
# 4.1 Spara Grad-CAM-rapport
# ------------------------------------------------------------
def test_save_grad_cam_report_includes_skipped_images(tmp_path):
    eval_results = {
        "results": [],
        "mean_iou": 0.0,
        "detection_rate": 0.0,
        "total": 0,
        "positive_total": 2,
        "skipped_without_boxes": 2,
        "detected_count": 0,
        "missed_count": 0,
        "layer_name": "conv2d",
    }

    report_path = save_grad_cam_report(eval_results, output_dir=tmp_path)
    report_text = Path(report_path).read_text(encoding="utf-8")

    assert "Positiva bilder:" in report_text
    assert "Överhoppade utan YOLO-boxar:" in report_text
    assert "Grad-CAM-lager:" in report_text
