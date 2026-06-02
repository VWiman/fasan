from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
from tqdm import tqdm

from config import (
    GRAD_CAM_IOU_THRESHOLD,
    GRAD_CAM_OUTPUT_DIR,
    GRAD_CAM_REPORT_FILE_NAME,
    GRAD_CAM_SAMPLE_COUNT,
    GRAD_CAM_THRESHOLD,
    IMAGE_SIZE,
)
from load_data import load_image


# ============================================================
# 1. GRAD-CAM
# ============================================================
#
# Grad-CAM (Gradient-weighted Class Activation Mapping) visualiserar vilka
# delar av bilden som påverkade modellens beslut. Gradienter från det sista
# conv-lagret viktas och summeras till en värmekarta.

# ------------------------------------------------------------
# 1.1 Hitta sista conv-lager
# ------------------------------------------------------------
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Ingen Conv2D-lagret hittades i modellen.")


# ------------------------------------------------------------
# 1.2 Beräkna Grad-CAM värmekarta
# ------------------------------------------------------------
def compute_grad_cam(model, image_array, layer_name=None):
    # image_array är en normaliserad RGB-bild med formen (H, W, 3). Funktionen
    # returnerar en värmekarta i intervallet 0-1 med samma storlek som feature map.
    # Sequential-modeller i Keras 3 exponerar inte alltid model.output som
    # symbolisk tensor, därför görs ett manuellt forward pass genom lagren.
    if layer_name is None:
        layer_name = get_last_conv_layer_name(model)

    img_tensor = tf.convert_to_tensor(image_array[np.newaxis, ...])

    with tf.GradientTape() as tape:
        x = img_tensor
        conv_outputs = None

        for layer in model.layers:
            x = layer(x)
            if layer.name == layer_name:
                conv_outputs = x
                tape.watch(conv_outputs)

        loss = x[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap.astype(np.float32)


# ============================================================
# 2. CAM-BBOX
# ============================================================
#
# Värmekartan tröskas för att få ett binärt område. Bounding box räknas
# som den minsta rektangeln som omsluter alla aktiverade pixlar.
# Koordinaterna returneras normaliserade (0-1) så att de kan jämföras
# direkt med YOLO-koordinater.

def cam_bbox_from_heatmap(heatmap, threshold=GRAD_CAM_THRESHOLD):
    # Returnerar (x1, y1, x2, y2) normaliserade till 0-1, eller None om inget
    # område i värmekartan passerar tröskeln.
    binary = (heatmap >= threshold).astype(np.uint8)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)

    if not rows.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    h, w = heatmap.shape
    return (x1 / w, y1 / h, x2 / w, y2 / h)


# ============================================================
# 3. YOLO-BOXAR
# ============================================================
#
# YOLO-formatet anger centerkoordinater och storlek (cx cy w h) normaliserade.
# Dessa konverteras till hörn (x1 y1 x2 y2) för IoU-beräkning.

def read_yolo_boxes(label_path):
    # Förenklade binära etiketter utan koordinater ignoreras här eftersom IoU
    # bara kan beräknas när det finns riktiga YOLO-boxar.
    text = Path(label_path).read_text().strip()

    if not text:
        return []

    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes.append((x1, y1, x2, y2))

    return boxes


# ============================================================
# 4. IOU
# ============================================================
#
# IoU (Intersection over Union) mäter hur mycket en förutsagd box
# överlappar med en referensbox. Värden nära 1 betyder god matchning.

def compute_iou(box_a, box_b):
    # box_a och box_b anges som (x1, y1, x2, y2), normaliserade till 0-1.
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def max_iou_against_gt(pred_box, gt_boxes):
    # Bästa IoU mellan pred_box och alla ground-truth-boxar.
    if not gt_boxes or pred_box is None:
        return 0.0
    return max(compute_iou(pred_box, gt) for gt in gt_boxes)


# ============================================================
# 5. UTVÄRDERING
# ============================================================
#
# Kör Grad-CAM på alla positiva bilder (etikett 1) i ett datasplit.
# För varje bild beräknas IoU mellan CAM-bbox och YOLO-boxarna.
# En bild räknas som korrekt lokaliserad om IoU >= iou_threshold.

def _get_label_path_for_image(image_path):
    # Bygger sökväg till etikettfil utifrån bildens sökväg.
    label_dir = image_path.parent.parent / "labels"
    return label_dir / f"{image_path.stem}.txt"


def evaluate_grad_cam_iou(
    model,
    split_data,
    layer_name=None,
    cam_threshold=GRAD_CAM_THRESHOLD,
    iou_threshold=GRAD_CAM_IOU_THRESHOLD,
):
    # Utvärderar hur väl Grad-CAM lokaliserar människor i positiva bilder.
    # Bilder utan YOLO-boxar hoppas över eftersom IoU inte kan beräknas utan
    # referensposition.
    if layer_name is None:
        layer_name = get_last_conv_layer_name(model)

    positive_pairs = [
        (img, lbl)
        for img, lbl in zip(split_data.image_paths, split_data.labels)
        if lbl == 1
    ]

    results = []
    skipped_without_boxes = 0

    for image_path, label in tqdm(positive_pairs, desc="Grad-CAM IoU", unit="bild"):
        label_path = _get_label_path_for_image(image_path)
        gt_boxes = read_yolo_boxes(label_path) if label_path.exists() else []

        if not gt_boxes:
            skipped_without_boxes += 1
            continue

        image = load_image(image_path, split_data.image_size)
        heatmap = compute_grad_cam(model, image, layer_name)
        pred_box = cam_bbox_from_heatmap(heatmap, threshold=cam_threshold)
        iou = max_iou_against_gt(pred_box, gt_boxes)

        results.append({
            "image_path": image_path,
            "iou": iou,
            "detected": iou >= iou_threshold,
            "pred_box": pred_box,
            "gt_boxes": gt_boxes,
        })

    detected_count = int(sum(r["detected"] for r in results))

    if not results:
        return {
            "results": results,
            "mean_iou": 0.0,
            "detection_rate": 0.0,
            "total": 0,
            "positive_total": len(positive_pairs),
            "skipped_without_boxes": skipped_without_boxes,
            "detected_count": 0,
            "missed_count": 0,
            "layer_name": layer_name,
        }

    iou_values = [r["iou"] for r in results]

    return {
        "results": results,
        "mean_iou": float(np.mean(iou_values)),
        "detection_rate": float(np.mean([r["detected"] for r in results])),
        "total": len(results),
        "positive_total": len(positive_pairs),
        "skipped_without_boxes": skipped_without_boxes,
        "detected_count": detected_count,
        "missed_count": len(results) - detected_count,
        "layer_name": layer_name,
    }


# ============================================================
# 6. SPARA RAPPORT
# ============================================================
#
# Rapporten sparar både sammanfattning och per-bild-resultat. Bilder som saknar
# YOLO-boxar redovisas separat eftersom de inte kan användas för IoU.

def save_grad_cam_report(
    eval_results,
    report_path=None,
    output_dir=GRAD_CAM_OUTPUT_DIR,
    report_file_name=GRAD_CAM_REPORT_FILE_NAME,
    iou_threshold=GRAD_CAM_IOU_THRESHOLD,
):
    if report_path is None:
        report_path = Path(output_dir) / report_file_name

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Grad-CAM IoU-rapport",
        "=" * 50,
        f"Positiva bilder:                     {eval_results.get('positive_total', eval_results['total'])}",
        f"Utvärderade bilder med YOLO-boxar:   {eval_results['total']}",
        f"Överhoppade utan YOLO-boxar:         {eval_results.get('skipped_without_boxes', 0)}",
        f"Grad-CAM-lager:                      {eval_results.get('layer_name', 'okänt')}",
        f"Genomsnittligt IoU:                  {eval_results['mean_iou']:.4f}",
        f"Korrekt lokaliserade:                {eval_results.get('detected_count', 0)}",
        f"Missade lokaliseringar:              {eval_results.get('missed_count', 0)}",
        f"Detektionsandel (IoU >= {iou_threshold:.2f}):      {eval_results['detection_rate']:.2%}",
        "",
        "Per-bild-resultat:",
    ]

    for r in eval_results["results"]:
        status = "OK " if r["detected"] else "NEJ"
        lines.append(f"  [{status}] {r['image_path'].name}  IoU={r['iou']:.4f}")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


# ============================================================
# 7. VISUALISERING
# ============================================================
#
# Sparar tre paneler sida vid sida: originalbild, Grad-CAM-overlay och
# bounding boxes (grön = ground truth, röd = CAM-bbox).

def save_grad_cam_visualization(
    model,
    image_path,
    label_path,
    save_path,
    image_size=IMAGE_SIZE,
    layer_name=None,
    cam_threshold=GRAD_CAM_THRESHOLD,
):
    image = load_image(image_path, image_size)
    heatmap = compute_grad_cam(model, image, layer_name)

    heatmap_resized = np.array(
        PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(image_size)
    ) / 255.0

    gt_boxes = read_yolo_boxes(label_path) if Path(label_path).exists() else []
    pred_box = cam_bbox_from_heatmap(heatmap, threshold=cam_threshold)
    iou = max_iou_against_gt(pred_box, gt_boxes)

    img_w, img_h = image_size  # (width, height)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ------------------------------------------------------------
    # 7.1 Rita originalbild
    # ------------------------------------------------------------
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # ------------------------------------------------------------
    # 7.2 Rita Grad-CAM-overlay
    # ------------------------------------------------------------
    axes[1].imshow(image)
    axes[1].imshow(heatmap_resized, alpha=0.5, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    # ------------------------------------------------------------
    # 7.3 Rita bounding boxes
    # ------------------------------------------------------------
    axes[2].imshow(image)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1 * img_w, y1 * img_h),
            (x2 - x1) * img_w,
            (y2 - y1) * img_h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            label="GT",
        )
        axes[2].add_patch(rect)

    if pred_box is not None:
        x1, y1, x2, y2 = pred_box
        rect = patches.Rectangle(
            (x1 * img_w, y1 * img_h),
            (x2 - x1) * img_w,
            (y2 - y1) * img_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="CAM",
        )
        axes[2].add_patch(rect)

    axes[2].set_title(f"GT (grön) vs CAM-bbox (röd)  IoU={iou:.3f}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ============================================================
# 8. SPARA EXEMPELBILDER
# ============================================================
#
# Exempelbilder sparas bara för positiva bilder som har YOLO-boxar. Förenklade
# positiva labels utan koordinater kan inte visualisera ground truth-boxar.

def save_grad_cam_samples(
    model,
    split_data,
    output_dir=GRAD_CAM_OUTPUT_DIR,
    sample_count=GRAD_CAM_SAMPLE_COUNT,
    layer_name=None,
    cam_threshold=GRAD_CAM_THRESHOLD,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_indices_with_boxes = [
        i
        for i, label in enumerate(split_data.labels)
        if label == 1 and read_yolo_boxes(_get_label_path_for_image(split_data.image_paths[i]))
    ]
    sample_indices = positive_indices_with_boxes[:sample_count]

    saved_paths = []

    for i, idx in enumerate(tqdm(sample_indices, desc="Sparar exempelbilder", unit="bild")):
        image_path = split_data.image_paths[idx]
        label_path = _get_label_path_for_image(image_path)
        save_path = output_dir / f"grad_cam_sample_{i + 1}.png"

        save_grad_cam_visualization(
            model,
            image_path,
            label_path,
            save_path,
            split_data.image_size,
            layer_name,
            cam_threshold,
        )
        saved_paths.append(save_path)

    return saved_paths
