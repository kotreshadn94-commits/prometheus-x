"""PROMETHEUS-X API: image inspection, live feed, human review and certificates."""
from __future__ import annotations

import asyncio
import time
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from . import database
    from .ai_engine import engine
    from .guardrails import guard
    from .postprocess import process_result
except ImportError:
    import database
    from ai_engine import engine
    from guardrails import guard
    from postprocess import process_result

app = FastAPI(title="PROMETHEUS-X", version="1.0.0", description="Industrial visual quality assurance API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])
latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
latest_lock = asyncio.Lock()


class Approval(BaseModel):
    inspection_id: int = Field(gt=0)


class VerifyRequest(BaseModel):
    serial_id: str


async def decode_upload(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded image is empty")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "File is not a supported image")
    if max(image.shape[:2]) > 2400:
        scale = 2400 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


@app.get("/api/health")
def health():
    return {"status": "ok", "model": engine.model_mode, "memory_bank_entries": len(engine.memory_bank)}


@app.post("/api/inspect")
async def inspect(file: Annotated[UploadFile, File()], batch_id: Annotated[str | None, Form()] = None,
                  serial_id: Annotated[str | None, Form()] = None,
                  auto_stabilize: Annotated[bool, Form()] = True,
                  role: Annotated[str, Form()] = "factory"):
    image = await decode_upload(file)
    started = time.perf_counter()
    raw_map, embedding = engine.infer(image)
    stabilized, warning, shift_metrics = guard.stabilize(image, embedding, auto_stabilize)
    if warning and auto_stabilize:
        # Recompute after CLAHE so localized evidence is not dominated by exposure drift.
        raw_map, embedding = engine.infer(stabilized)
    elapsed = (time.perf_counter() - started) * 1000
    result = process_result(stabilized, raw_map, elapsed, warning)
    result.update({"model": engine.model_mode, "shift_metrics": shift_metrics, "role": role,
                   "embedding": [round(float(v), 6) for v in embedding[:4]]})
    inspection_id = database.record_inspection(result, batch_id, serial_id)
    result["inspection_id"] = inspection_id
    if result["verdict"] == "PASS":
        guard.observe_normal(embedding)
    async with latest_lock:
        global latest_frame
        latest_frame = stabilized.copy()
    return result


@app.get("/api/video_feed")
async def video_feed():
    async def frames():
        while True:
            async with latest_lock:
                frame = latest_frame.copy()
            ok, data = cv2.imencode(".jpg", frame)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data.tobytes() + b"\r\n"
            await asyncio.sleep(0.25)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/hitl/queue")
def review_queue():
    return {"items": database.get_review_queue(), "analytics": database.analytics()}


@app.post("/api/hitl/approve")
def approve(payload: Approval):
    result = database.inspection_by_id(payload.inspection_id)
    if not result:
        raise HTTPException(404, "Inspection not found")
    if result["confidence"] < 45 or result["confidence"] > 65:
        raise HTTPException(422, "Only borderline confidence inspections (45–65%) can be approved")
    if not database.approve_inspection(payload.inspection_id):
        raise HTTPException(404, "Inspection not found")
    embedding = result.get("embedding")
    if embedding:
        engine.inject_normal(embedding)
    return {"approved": True, "inspection_id": payload.inspection_id,
            "memory_bank_entries": len(engine.memory_bank)}


@app.get("/api/batches")
def batches():
    return {"batches": database.get_batches(), "analytics": database.analytics()}


@app.get("/api/certificate/{serial_id}")
def certificate(serial_id: str):
    item = database.get_serial(serial_id)
    if not item:
        raise HTTPException(404, "Serial number was not found in the verified product registry")
    return item


@app.post("/api/customer/verify")
def verify_customer(payload: VerifyRequest):
    item = database.get_serial(payload.serial_id)
    if not item:
        raise HTTPException(404, "This serial number is not present in the product registry")
    return {"verified": True, "serial_id": item["serial_id"], "certificate_id": item["certificate_id"],
            "verdict": item["verdict"], "product": item["product"]}


@app.get("/api/inspections/{inspection_id}")
def inspection_detail(inspection_id: int):
    result = database.inspection_by_id(inspection_id)
    if not result:
        raise HTTPException(404, "Inspection not found")
    return result


_FRONTEND = __import__("pathlib").Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_FRONTEND / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
