"""Inspection post-processing, defect geometry, calibration and visual evidence."""
from __future__ import annotations

import base64
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression


def _fit_platt_scaler() -> LogisticRegression:
    # Synthetic CutPaste-style calibration samples: patch area and peak response
    # are features, with known normal/defect labels and deterministic variation.
    rng = np.random.default_rng(42)
    normal = np.column_stack((rng.beta(1.2, 12, 400), rng.beta(1.4, 8, 400)))
    defect = np.column_stack((rng.uniform(0.08, 0.95, 400), rng.uniform(0.35, 1.0, 400)))
    x = np.vstack((normal, defect))
    y = np.r_[np.zeros(len(normal)), np.ones(len(defect))]
    model = LogisticRegression(C=1.2, random_state=42, max_iter=500)
    model.fit(x, y)
    return model


_CALIBRATOR = _fit_platt_scaler()


def _classify(contours: list[np.ndarray], image: np.ndarray, mask: np.ndarray) -> str:
    if not contours:
        return "No Defect"
    contour = max(contours, key=cv2.contourArea)
    area = max(cv2.contourArea(contour), 1.0)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / max(1, min(w, h))
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / max(perimeter * perimeter, 1.0)
    pixels = image[mask > 0]
    color_variance = float(np.mean(np.std(pixels.astype(np.float32), axis=0))) if pixels.size else 0
    if aspect >= 3.2 and circularity < 0.42:
        return "Micro-Scratch" if max(w, h) < 70 else "Structural Crack"
    if circularity > 0.66 and area < image.shape[0] * image.shape[1] * 0.04:
        return "Dent/Puncture"
    if color_variance > 22 or area > image.shape[0] * image.shape[1] * 0.025:
        return "Surface Contamination"
    return "Structural Crack" if aspect > 1.7 else "Dent/Puncture"


def process_result(image: np.ndarray, anomaly_map: np.ndarray, latency_ms: float,
                   shift_warning: bool = False) -> dict:
    h, w = image.shape[:2]
    amap = cv2.resize(anomaly_map.astype(np.float32), (w, h))
    peak = float(np.clip(np.percentile(amap, 99.7), 0, 1))
    threshold = max(float(np.percentile(amap, 95)), 0.35)
    binary = (amap >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    clean = np.zeros_like(binary)
    boxes = []
    contours = []
    for idx in range(1, count):
        x, y, bw, bh, area = stats[idx]
        if area < 25:
            continue
        clean[labels == idx] = 255
        boxes.append({"x": int(x), "y": int(y), "width": int(bw), "height": int(bh), "area": int(area)})
        contour = cv2.findContours((labels == idx).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours.extend(contour)
    area_fraction = float(np.count_nonzero(clean)) / max(h * w, 1)
    defect_type = _classify(contours, image, clean)
    severity = float(np.clip((0.6 * peak + 0.4 * area_fraction) * 100, 0, 100))
    # No localized components means normal unless the shift detector explicitly warned.
    if not boxes or shift_warning:
        severity = min(severity, 18.0)
        defect_type = "Distribution Shift" if shift_warning else "No Defect"
    verdict = "PASS" if severity < 25 else "REWORK" if severity <= 65 else "SCRAP"
    calibrated = float(_CALIBRATOR.predict_proba([[area_fraction, peak]])[0, 1] * 100)
    if not boxes:
        calibrated = max(96.0, calibrated)
    overlay = cv2.applyColorMap(np.uint8(np.clip(amap * 255, 0, 255)), cv2.COLORMAP_TURBO)
    evidence = cv2.addWeighted(image, 0.58, overlay, 0.42, 0)
    for box in boxes:
        cv2.rectangle(evidence, (box["x"], box["y"]),
                      (box["x"] + box["width"], box["y"] + box["height"]), (30, 230, 255), 2)
    ok, encoded = cv2.imencode(".jpg", evidence, [cv2.IMWRITE_JPEG_QUALITY, 82])
    heatmap = base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""
    return {
        "verdict": verdict, "defect_type": defect_type, "severity": round(severity, 2),
        "confidence": round(calibrated, 2), "peak_intensity": round(peak, 4),
        "affected_area_pct": round(area_fraction * 100, 4), "boxes": boxes,
        "heatmap_overlay": heatmap, "latency_ms": round(float(latency_ms), 2),
        "distribution_shift_warning": bool(shift_warning),
    }
