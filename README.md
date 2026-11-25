# PROMETHEUS-X

A runnable industrial image-inspection and quality-assurance starter: FastAPI + SQLite backend, OpenCV anomaly/scratch screening, optional ONNX inference, and a single-page three-portal UI. The built-in OpenCV engine works without downloading a model. It is a useful inspection prototype, not a validated production metrology system; tune thresholds and validate on representative labeled factory data before using it for release decisions.

## Requirements

- Python 3.10+ (tested design target: Python 3.11)
- A modern browser; webcam access requires `localhost` or HTTPS.
- Optional: a trained ONNX model at `backend/model.onnx`.

## Run locally

From the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r backend/requirements.txt
cd backend
python main.py
```

Open http://127.0.0.1:8000. The database is created and seeded on first startup. Example serials: `PX-NUT-000184`, `PX-NUT-000185`, and `PX-HSG-000031`. Upload JPEG/PNG images in the Factory Console or Customer scanner. Camera access uses the browser's camera permission. The API docs are at http://127.0.0.1:8000/docs.

The B2B table uses seeded batch aggregate records. New image inspections are persisted in SQLite. Certificate lookups verify serial-to-batch linkage; a certificate preview appears once an inspection has been linked to that serial.

## Optional PatchCore training/export

The training script expects the MVTec AD dataset under `datasets/MVTec` and an anomalib/PyTorch environment compatible with its API. Install anomalib and its export dependencies separately, then run:

```bash
python -m pip install anomalib torch onnx
python train_colab.py
```

The script selects `metal_nut`, configures PatchCore with `wide_resnet50_2`, `layer2`/`layer3`, and a 0.1 coreset sampling ratio, and exports `backend/model.onnx`. Dataset download/licensing is managed separately. Anomalib's APIs differ across releases; if your installed release changes export/model invocation APIs, pin a compatible anomalib version and adjust the adapter accordingly. The runtime inspects the ONNX input/output and falls back to directional OpenCV filters if the graph is missing or fails to load.

## API overview

- `POST /api/inspect` multipart fields: `file`, optional `batch_id`, `serial_id`, `stabilize`.
- `GET /api/video_feed` streams a local OpenCV webcam when available; the browser camera flow is the more portable option.
- `POST /api/hitl/approve` accepts a feature vector and a 45–65 confidence score.
- `GET /api/batches`, `GET /api/certificate/{serial_id}`, `GET /api/customer/verify?serial_id=...`.
- `GET /api/health` reports the selected inference path.

## Implementation notes

The fallback analyzes oriented Sobel and Gabor responses, small-object filtering removes connected components smaller than 25 pixels, and CLAHE is applied when a global brightness/contrast/feature drift is detected. The classifier uses transparent contour geometry heuristics. Confidence is a reproducible logistic curve fitted on generated score distributions; it is not a substitute for calibration against held-out production labels. The active HITL bank is in process memory, so approved samples are not persisted across server restarts (approval audit records are persisted). Use authenticated access, HTTPS, retention controls, independent model validation, and safety review before production deployment.
