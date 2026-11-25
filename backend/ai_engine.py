"""Image inference: ONNX PatchCore when installed, with an operational CV fallback."""
from pathlib import Path
import threading
import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "model.onnx"


class VisionEngine:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.lock = threading.RLock()
        self.normal_bank = []
        self.baseline = None
        if MODEL_PATH.exists():
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
            except Exception as exc:
                print(f"ONNX model unavailable; using CV engine: {exc}")

    @staticmethod
    def features(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return np.concatenate([small.flatten(), np.array([gray.mean()/255, gray.std()/255], dtype=np.float32)])

    def analyze(self, image, stabilize=False):
        source = image.copy()
        if stabilize:
            lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            source = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        response = np.zeros((h, w), np.float32)
        # Directional gradients and oriented Gabor kernels emphasize fine linear marks.
        for angle in (0, 45, 90, 135):
            rad = np.deg2rad(angle)
            dx, dy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            directional = np.abs(dx*np.cos(rad) + dy*np.sin(rad))
            kernel = cv2.getGaborKernel((15, 15), 3.0, rad, 8.0, 0.55, 0, ktype=cv2.CV_32F)
            texture = np.abs(cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel))
            response = np.maximum(response, 0.65*directional + 0.35*texture)
        response = cv2.GaussianBlur(response, (5, 5), 0)
        cv_heat = response / max(float(np.percentile(response, 99.5)), 1.0)
        cv_heat = np.clip(cv_heat, 0, 1)
        neural = None
        if self.session is not None:
            try:
                input_shape = self.session.get_inputs()[0].shape
                ih = int(input_shape[2]) if isinstance(input_shape[2], int) else 256
                iw = int(input_shape[3]) if isinstance(input_shape[3], int) else 256
                rgb = cv2.cvtColor(cv2.resize(source, (iw, ih)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                tensor = np.transpose(rgb, (2, 0, 1))[None]
                out = self.session.run(None, {self.input_name: tensor})[0]
                plane = np.asarray(out).squeeze()
                if plane.ndim == 0:
                    plane = np.full((ih, iw), float(plane), np.float32)
                if plane.ndim == 1:
                    plane = np.full((ih, iw), float(np.max(plane)), np.float32)
                plane = cv2.resize(plane.astype(np.float32), (w, h))
                lo, hi = np.percentile(plane, [5, 99])
                neural = np.clip((plane-lo)/(hi-lo+1e-6), 0, 1)
            except Exception as exc:
                print(f"ONNX inference failed for this frame; using CV response: {exc}")
        heat = cv_heat if neural is None else np.maximum(0.55*cv_heat, 0.45*neural)
        features = self.features(source)
        with self.lock:
            # Approved normal examples influence the baseline conservatively.
            if self.normal_bank:
                distances = [float(np.mean(np.abs(features-x))) for x in self.normal_bank[-64:]]
                if min(distances) < 0.08:
                    heat *= 0.75
        return {"heatmap": heat, "features": features, "image": source,
                "engine": "onnx-patchcore+opencv" if neural is not None else "opencv-directional-fallback"}

    def approve(self, feature_vector):
        vector = np.asarray(feature_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.size < 10 or not np.all(np.isfinite(vector)):
            raise ValueError("Invalid feature vector")
        with self.lock:
            self.normal_bank.append(vector)
            self.normal_bank = self.normal_bank[-128:]
            self.baseline = vector.copy()
        return len(self.normal_bank)


engine = VisionEngine()
