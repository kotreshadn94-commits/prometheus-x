"""Frame-shift monitoring, dust filtering and memory-bank HITL controls."""
import cv2
import numpy as np

class ShiftGuard:
    def __init__(self):
        self.reference = None
        self.reference_feature = None

    def inspect(self, image, features, heatmap):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness, contrast = float(gray.mean()), float(gray.std())
        if self.reference is None:
            self.reference = (brightness, contrast)
            self.reference_feature = np.asarray(features, dtype=np.float32).copy()
            return {"shift": False, "brightness": round(brightness, 2), "contrast": round(contrast, 2), "stabilized": False}
        rb, rc = self.reference
        feature_drift = float(np.mean(np.abs(np.asarray(features)-self.reference_feature)))
        shifted = abs(brightness-rb) > 38 or abs(contrast-rc) > 30 or feature_drift > 0.20
        localized_peak = float(np.max(heatmap)) if heatmap.size else 0
        # Global illumination/texture change without a spatially localized spike is likely a shift.
        warning = shifted and localized_peak < 0.92
        return {"shift": bool(warning), "brightness": round(brightness, 2),
                "contrast": round(contrast, 2), "feature_drift": round(feature_drift, 4),
                "stabilized": bool(warning)}


def filter_small_dust(heatmap, minimum_pixels=25):
    binary = (np.asarray(heatmap) >= 0.48).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.asarray(heatmap).copy()
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < minimum_pixels:
            cleaned[labels == index] = 0
    return cleaned


guard = ShiftGuard()
