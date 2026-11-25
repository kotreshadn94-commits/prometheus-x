"""PROMETHEUS-X HTTP API."""
from pathlib import Path
import time
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import database
from ai_engine import engine
from guardrails import guard, filter_small_dust
from postprocess import process

app = FastAPI(title="PROMETHEUS-X Inspection & Quality Assurance", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@app.on_event("startup")
def startup():
    database.init_db()


@app.get("/")
def home():
    return FileResponse(FRONTEND)


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "onnx-patchcore+opencv" if engine.session else "opencv-directional-fallback"}


async def inspect_bytes(contents, batch_id=None, serial_id=None, stabilize=False):
    if not contents:
        raise HTTPException(400, "An image file is required")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image upload exceeds the 20 MB limit")
    started = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Upload must be a readable image")
    if image.shape[0] * image.shape[1] > 16_000_000:
        raise HTTPException(413, "Image exceeds the 16 megapixel limit")
    if batch_id:
        with database.connect() as db:
            if db.execute("SELECT 1 FROM batches WHERE batch_id=?", (batch_id,)).fetchone() is None:
                raise HTTPException(404, "Batch ID not found")
    if serial_id:
        with database.connect() as db:
            if db.execute("SELECT 1 FROM products WHERE serial_id=?", (serial_id,)).fetchone() is None:
                raise HTTPException(404, "Product serial ID not found")

    # Inspect the unmodified frame first. A global shift is diagnosed independently
    # from localized defects; when enabled, CLAHE is then actually applied and run.
    raw_result = engine.analyze(image, stabilize=False)
    shift = guard.inspect(raw_result["image"], raw_result["features"], raw_result["heatmap"])
    stabilized = bool(shift["shift"] and stabilize)
    result = engine.analyze(image, stabilize=True) if stabilized else raw_result
    heat = filter_small_dust(result["heatmap"])
    vision = process(result["image"], heat)
    if shift["shift"] and not stabilize:
        vision["verdict"], vision["defect_type"], vision["severity"] = "REVIEW", "Distribution Shift", 0.0
    shift["stabilized"] = stabilized
    latency = (time.perf_counter() - started) * 1000
    output = {**vision, "latency_ms": round(latency, 2), "engine": result["engine"],
              "guardrail": shift, "features": result["features"].tolist(),
              "created_at": time.time(), "batch_id": batch_id}
    inspection_id = database.record_inspection({**output, "serial_id": serial_id,
                                                  "image_data": vision["heatmap_image"]})
    if serial_id:
        with database.connect() as db:
            db.execute("UPDATE products SET verdict=?,inspection_id=? WHERE serial_id=?",
                       (vision["verdict"], inspection_id, serial_id))
    output["inspection_id"] = inspection_id
    return output


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...), batch_id: str | None = Form(None),
                  serial_id: str | None = Form(None), stabilize: bool = Form(False)):
    return await inspect_bytes(await file.read(), batch_id, serial_id, stabilize)


@app.get("/api/batches")
def batches():
    return {"batches": database.get_batches(), "inspections": database.recent_inspections(),
            "serials": database.get_products()}


@app.get("/api/batches/{batch_id}/inspections")
def batch_inspections(batch_id: str, limit: int = 20):
    """Return batch-specific inspection evidence for an authorized client portal."""
    if not 1 <= limit <= 100:
        raise HTTPException(400, "limit must be between 1 and 100")
    with database.connect() as db:
        if db.execute("SELECT 1 FROM batches WHERE batch_id=?", (batch_id,)).fetchone() is None:
            raise HTTPException(404, "Batch ID not found")
        rows = db.execute("""SELECT id, serial_id, verdict, defect_type, severity, confidence,
                             latency_ms, created_at, image_data FROM inspections
                             WHERE batch_id=? ORDER BY id DESC LIMIT ?""", (batch_id, limit)).fetchall()
    return {"batch_id": batch_id, "inspections": [dict(row) for row in rows]}


@app.get("/api/certificate/{serial_id}")
def get_certificate(serial_id: str):
    result = database.certificate(serial_id)
    if not result:
        raise HTTPException(404, "Serial number not found")
    return result


@app.get("/api/customer/verify")
def verify(serial_id: str):
    return get_certificate(serial_id)


class Approval(BaseModel):
    features: list[float] = Field(min_length=10, max_length=1100)
    confidence: float = Field(ge=45, le=65)
    note: str = Field(default="Operator verified normal", max_length=300)


@app.post("/api/hitl/approve")
def approve(payload: Approval):
    try:
        count = engine.approve(payload.features)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    database.log_approval(payload.features, payload.note)
    return {"approved": True, "memory_bank_size": count,
            "message": "Normal sample added to the active memory bank."}


_camera = None


@app.get("/api/video_feed")
def video_feed():
    """Stream MJPEG from the server-attached camera, when available."""
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(0)
    if not _camera.isOpened():
        raise HTTPException(503, "No local webcam is available. Use browser camera capture or image upload.")

    def frames():
        while _camera is not None and _camera.isOpened():
            ok, frame = _camera.read()
            if not ok:
                break
            encoded, jpg = cv2.imencode(".jpg", frame)
            if encoded:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
