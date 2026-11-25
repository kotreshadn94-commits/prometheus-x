# PROMETHEUS-X

Real-time industrial visual inspection and multi-stakeholder quality assurance. The app includes a FastAPI/SQLite service, ONNX PatchCore inference when a trained model is available, a working OpenCV directional-feature fallback, human review, batch audit records, and a responsive three-portal browser UI.

## Requirements

- Python 3.10 or newer
- A modern browser; camera access requires `localhost` or HTTPS
- Optional: a trained MVTec AD PatchCore ONNX model at `backend/model.onnx`

## Install and run

From the repository root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. The API reference is at <http://localhost:8000/docs>. SQLite creates `backend/prometheus.db` and seeds sample batches and serial IDs on first startup. The database location can be changed with `PROMETHEUS_DB`.

The inference engine uses `backend/model.onnx` when present. Without the model it immediately runs the directional Sobel/Gabor OpenCV detector, so upload inspection, camera scans, customer reports, and review controls work without a training download. The fallback is an operational feature detector, not a trained PatchCore model; install a domain-trained model before relying on it for production acceptance decisions. ONNX location can be changed with `PROMETHEUS_MODEL`.

## Train and export PatchCore

Download/extract the MVTec AD dataset so the `metal_nut` category is available, then install the training dependencies in the training environment. For example:

```bash
python -m pip install anomalib torch torchvision
python train_colab.py --data ./datasets/MVTecAD
```

Training reads the MVTec `metal_nut` training split (normal `train/good` samples) and exports the trained inference graph to `backend/model.onnx`. Training is optional; the application runs in fallback mode until the artifact exists.

## Features and API

- Factory portal: image upload, browser webcam, heatmap evidence, severity/verdict, shift simulation, stabilization, outcomes and HITL queue.
- B2B portal: batch pass-rate and micro-scratch SLA audit view.
- Customer portal: serial certificate verification and post-delivery self-check / report.
- `POST /api/inspect` (multipart `file`, optional `batch_id`, `serial_id`, `auto_stabilize`, `role`)
- `GET /api/video_feed` (MJPEG stream of the latest inspected frame)
- `GET /api/hitl/queue`, `POST /api/hitl/approve` (`{"inspection_id": 1}`)
- `GET /api/batches`, `GET /api/certificate/{serial_id}`, `POST /api/customer/verify`
- `GET /api/health` and `GET /api/docs`

Images and inspection metrics are persisted as inspection records; heatmap evidence is returned by the API in the inspection result. The application is a demonstration-quality inspection workflow and its severity thresholds, model quality, optics, and calibration must be validated against representative production data before being used for safety-critical release decisions.
