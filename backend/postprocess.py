"""Explainable image post-processing, defect geometry, score and routing."""
import base64
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

# A deterministic synthetic score calibration curve, fit once for a reproducible UI.
_rng = np.random.default_rng(42)
_x = np.concatenate([_rng.normal(0.15, 0.12, 600), _rng.normal(0.72, 0.16, 600)]).clip(0, 1).reshape(-1, 1)
_y = np.r_[np.zeros(600, dtype=int), np.ones(600, dtype=int)]
_CALIBRATOR = LogisticRegression(random_state=7).fit(_x, _y)


def _encode_jpg(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("Could not encode inspection preview")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def process(image, heatmap):
    h, w = heatmap.shape
    threshold = max(0.48, float(np.percentile(heatmap, 98)))
    mask = (heatmap >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, areas = [], []
    annotated = image.copy()
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4*np.pi*area/(perimeter*perimeter+1e-6)
        aspect = max(bw, bh)/(min(bw, bh)+1e-6)
        candidates.append((area, aspect, circularity, x, y, bw, bh, contour))
        boxes.append({"x": int(x), "y": int(y), "width": int(bw), "height": int(bh), "area": float(area)})
        areas.append(area)
    if not candidates:
        defect = "None"
    else:
        dominant = max(candidates, key=lambda item: item[0])
        _, aspect, circularity, x, y, bw, bh, contour = dominant
        roi = image[y:y+bh, x:x+bw]
        color_variance = float(np.std(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1])) if roi.size else 0
        if aspect >= 4.0:
            defect = "Micro-Scratch" if max(bw, bh) < 80 else "Structural Crack"
        elif circularity >= 0.68 and aspect < 2.2:
            defect = "Dent/Puncture"
        elif color_variance > 35:
            defect = "Surface Contamination"
        else:
            defect = "Structural Crack"
    peak = float(np.max(heatmap)) if heatmap.size else 0.0
    affected_pct = min(1.0, sum(areas)/(h*w))
    severity = float(np.clip((0.6*peak + 0.4*affected_pct)*100, 0, 100))
    confidence = float(_CALIBRATOR.predict_proba([[peak]])[0, 1] * 100)
    verdict = "PASS" if severity < 25 else ("REWORK" if severity <= 65 else "SCRAP")
    for box in boxes:
        cv2.rectangle(annotated, (box["x"], box["y"]), (box["x"]+box["width"], box["y"]+box["height"]), (0, 70, 255), 2)
    colored = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1)*255), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image, 0.62, colored, 0.38, 0)
    for box in boxes:
        cv2.rectangle(overlay, (box["x"], box["y"]), (box["x"]+box["width"], box["y"]+box["height"]), (0, 255, 255), 2)
    return {"verdict": verdict, "defect_type": defect, "severity": round(severity, 2),
            "confidence": round(confidence, 2), "peak_intensity": round(peak, 4),
            "affected_area_pct": round(affected_pct*100, 3), "boxes": boxes,
            "heatmap_image": _encode_jpg(overlay), "annotated_image": _encode_jpg(annotated)}
