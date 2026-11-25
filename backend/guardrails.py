"""Distribution-shift guardrails, CLAHE stabilization and HITL normal memory."""
from __future__ import annotations

import cv2
import numpy as np


class ShiftGuard:
    def __init__(self) -> None:
        self.baseline_luminance = 118.0
        self.baseline_contrast = 48.0
        self.baseline_embedding = np.zeros(4, dtype=np.float32)
        self.embedding_ready = False
        self.threshold = 0.24

    def stabilize(self, image: np.ndarray, embedding: np.ndarray, enabled: bool = True) -> tuple[np.ndarray, bool, dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        luminance = float(gray.mean())
        contrast = float(gray.std())
        brightness_drift = abs(luminance - self.baseline_luminance) / 255.0
        contrast_drift = abs(contrast - self.baseline_contrast) / 128.0
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        embedding_drift = 0.0
        if self.embedding_ready and vector.size >= self.baseline_embedding.size:
            embedding_drift = float(np.linalg.norm(vector[:4] - self.baseline_embedding) /
                                    (np.linalg.norm(self.baseline_embedding) + 0.25))
        drift = min(1.0, 0.45 * brightness_drift + 0.35 * contrast_drift + 0.20 * min(embedding_drift, 1.0))
        # Strong global changes with no sharp localized response are domain shift,
        # not evidence of a product defect.
        warning = drift > self.threshold
        output = image
        if warning and enabled:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            stabilized = clahe.apply(gray)
            output = cv2.cvtColor(stabilized, cv2.COLOR_GRAY2BGR)
        return output, warning, {"luminance": round(luminance, 2), "contrast": round(contrast, 2),
                                 "drift": round(drift, 4), "clahe_applied": bool(warning and enabled)}

    def observe_normal(self, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size >= 4:
            if not self.embedding_ready:
                self.baseline_embedding = vector[:4].copy()
                self.embedding_ready = True
            else:
                self.baseline_embedding = 0.98 * self.baseline_embedding + 0.02 * vector[:4]


guard = ShiftGuard()
