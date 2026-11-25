"""PatchCore ONNX adapter plus a dependency-light OpenCV feature detector."""
from __future__ import annotations

import os
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

MODEL_PATH = Path(os.getenv("PROMETHEUS_MODEL", Path(__file__).parent / "model.onnx"))


class AnomalyEngine:
    def __init__(self) -> None:
        self.session = None
        self.input_name = None
        self.lock = RLock()
        self.memory_bank: list[np.ndarray] = []
        self.model_mode = "opencv-feature-bank"
        if MODEL_PATH.is_file():
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                self.model_mode = "patchcore-onnx"
            except Exception as exc:
                print(f"ONNX model unavailable ({exc}); using OpenCV feature detector")
        # Reference descriptors represent smooth, evenly illuminated metal at
        # several orientations. Runtime approvals add real production normals.
        for angle in np.linspace(0, np.pi, 12, endpoint=False):
            self.memory_bank.append(np.array([np.cos(angle), np.sin(angle), 0.03, 0.08], dtype=np.float32))

    @staticmethod
    def embedding(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.array([float(gx.mean()), float(gy.mean()), float(gx.std()), float(gy.std())], dtype=np.float32)

    def inject_normal(self, embedding: list[float] | np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size:
            with self.lock:
                self.memory_bank.append(vector[:4])
                self.memory_bank = self.memory_bank[-512:]

    @staticmethod
    def _directional_response(gray: np.ndarray) -> np.ndarray:
        grayf = gray.astype(np.float32) / 255.0
        responses = []
        for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            kernel = cv2.getGaborKernel((15, 15), 2.8, theta, 9.0, 0.65, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(grayf, cv2.CV_32F, kernel)
            responses.append(np.abs(filtered))
        gx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx * gx + gy * gy)
        # Local contrast suppresses broad illumination changes, retaining thin edges.
        local = cv2.absdiff(grayf, cv2.GaussianBlur(grayf, (0, 0), 2.2))
        return np.maximum(np.max(np.stack(responses), axis=0) * 0.35 + gradient * 0.45 + local * 0.2, 0)

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_map = self._directional_response(gray)
        if self.session is not None:
            try:
                resized = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
                tensor = np.transpose(resized[:, :, ::-1], (2, 0, 1))[None].copy()
                outputs = self.session.run(None, {self.input_name: tensor})
                candidates = [np.asarray(value) for value in outputs if np.asarray(value).ndim >= 2]
                if candidates:
                    raw = np.squeeze(candidates[-1]).astype(np.float32)
                    if raw.ndim == 0:
                        raw = np.full((32, 32), float(raw), np.float32)
                    if raw.ndim > 2:
                        raw = np.max(raw, axis=tuple(range(raw.ndim - 2)))
                    neural = cv2.resize(raw, (w, h))
                    lo, hi = np.percentile(neural, 5), np.percentile(neural, 99.5)
                    neural = np.clip((neural - lo) / max(float(hi - lo), 1e-6), 0, 1)
                    edge = edge_map / max(float(np.percentile(edge_map, 99.5)), 1e-6)
                    anomaly = np.clip(0.75 * neural + 0.25 * edge, 0, 1)
                else:
                    anomaly = edge_map
            except Exception as exc:
                print(f"ONNX inference failed ({exc}); using OpenCV features for this frame")
                anomaly = edge_map
        else:
            # Robust per-frame normalization provides useful operation before a
            # trained model is installed; global exposure changes are handled by guardrails.
            q50, q995 = np.percentile(edge_map, [50, 99.5])
            anomaly = np.clip((edge_map - q50) / max(float(q995 - q50), 1e-6), 0, 1)
        anomaly = cv2.GaussianBlur(anomaly.astype(np.float32), (0, 0), 1.0)
        return anomaly, self.embedding(image)


engine = AnomalyEngine()
