"""SignVision FastAPI Web Application.

Provides:
- Web interface (HTML/CSS/JS) with live webcam and video upload
- REST API endpoint for video file processing (/api/upload)
- WebSocket endpoint for low-latency live camera streaming (/ws)
- Health and status endpoint (/api/health)
"""

import os
import sys
import json
import time
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import aiofiles
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp

from src.inference.pipeline import (
    predict_from_frames,
    extract_video,
    extract_frame_landmarks,
    is_checkpoint_available,
    load_model,
    BASELINE_PATH,
    MODEL_PATH,
    NPTS,
    POSE_IDX
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signvision.server")


def to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to JSON-serializable Python types.
    NaN and Infinity floats are converted to None (JSON null) so json.dumps never
    raises 'float values are not JSON serializable'.
    """
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (v != v or v == float('inf') or v == float('-inf')) else v
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


app = FastAPI(
    title="SignVision API",
    description="Real-Time Sign Language Translation System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the SignVision web application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend assets not found.")
    return FileResponse(str(index_file))


@app.get("/api/health")
async def health_check():
    """System health check and model status."""
    has_transformer = MODEL_PATH.exists()
    has_baseline = BASELINE_PATH.exists()
    mode = "transformer" if has_transformer else ("baseline_classifier" if has_baseline else "demo")

    payload = {
        "status": "healthy",
        "model_mode": mode,
        "transformer_available": has_transformer,
        "baseline_available": has_baseline,
        "mediapipe_version": getattr(mp, "__version__", "unknown"),
        "timestamp": time.time()
    }
    return JSONResponse(content=to_jsonable(payload))


@app.post("/api/upload")
async def handle_video_upload(file: UploadFile = File(...), mode: str = "combined"):
    """Upload and translate sign language video file."""
    start_time = time.time()
    filename = file.filename or "video.mp4"
    ext = Path(filename).suffix.lower()

    if ext not in {".mp4", ".webm", ".avi", ".mov", ".mkv"}:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported video format '{ext}'. Allowed: .mp4, .webm, .avi, .mov"}
        )

    temp_path = Path(f"temp_upload_{int(time.time()*1000)}{ext}")

    try:
        async with aiofiles.open(temp_path, "wb") as out_f:
            while chunk := await file.read(1024 * 1024):
                await out_f.write(chunk)

        landmarks = extract_video(str(temp_path))

        if landmarks is None or len(landmarks) == 0:
            return JSONResponse(
                status_code=422,
                content={"error": "No hand or pose landmarks detected in the uploaded video."}
            )

        if len(landmarks) > 32:
            step = len(landmarks) / 32.0
            indices = [int(i * step) for i in range(32)]
            eval_window = landmarks[indices]
        else:
            eval_window = landmarks

        prediction = predict_from_frames(eval_window, mode=mode)
        elapsed = round(time.time() - start_time, 2)

        res = {
            "gloss": prediction["gloss"],
            "text": prediction["text"],
            "digit": prediction.get("digit"),
            "category": prediction.get("category", "word"),
            "confidence": prediction.get("confidence", 0.9),
            "is_fallback": prediction.get("is_fallback", False),
            "total_frames": len(landmarks),
            "processing_time_sec": elapsed,
            "filename": filename
        }
        return JSONResponse(content=to_jsonable(res))

    except Exception as e:
        logger.exception("Error processing video upload: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})

    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ---------- WebSocket Real-Time Camera Stream Handler ----------

class StreamConnectionManager:
    """Manages active WebSocket camera streaming sessions."""

    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections[ws] = {
            "buffer": [],
            "max_buffer": 24,
            "last_prediction_time": 0.0,
            "frame_count": 0
        }
        logger.info("WebSocket client connected. Active: %d", len(self.active_connections))

    def disconnect(self, ws: WebSocket):
        self.active_connections.pop(ws, None)
        logger.info("WebSocket client disconnected. Active: %d", len(self.active_connections))

    async def handle_message(self, ws: WebSocket, message_text: str):
        session = self.active_connections.get(ws)
        if not session:
            return

        data = json.loads(message_text)
        msg_type = data.get("type", "frame")
        buffer = session["buffer"]
        session["frame_count"] += 1
        now = time.time()

        # Extract optional mode ('combined', 'words', or 'numbers')
        mode = data.get("mode", "combined")

        # Case 1: Browser sends pre-extracted client-side keypoints (Ultra-fast & exact!)
        if msg_type == "keypoints":
            raw_kps = data.get("keypoints")
            if raw_kps and len(raw_kps) > 0:
                # JSON null → Python None → np.nan so float32 cast is always safe
                clean = [
                    [(v if v is not None and v == v else float('nan')) for v in row]
                    for row in raw_kps
                ]
                kp_arr = np.array(clean, dtype=np.float32)  # shape (49,3) or (21,3)
                buffer.append(kp_arr)
                if len(buffer) > session["max_buffer"]:
                    buffer.pop(0)

                # Predict every ~150ms
                if len(buffer) >= 2 and (now - session["last_prediction_time"]) > 0.15:
                    session["last_prediction_time"] = now
                    win = np.stack(buffer, axis=0)
                    res = predict_from_frames(win, mode=mode)

                    payload = {
                        "type": "prediction",
                        "gloss": res["gloss"],
                        "text": res["text"],
                        "digit": res.get("digit"),
                        "category": res.get("category", "word"),
                        "confidence": res.get("confidence", 0.0),
                        "is_fallback": res.get("is_fallback", False),
                        "frame_id": session["frame_count"]
                    }
                    await ws.send_text(json.dumps(to_jsonable(payload)))
            return

        # Case 2: Browser sends raw base64 frame (server-side extraction)
        b64_frame = data.get("frame")
        if not b64_frame:
            return

        if "," in b64_frame:
            b64_frame = b64_frame.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_frame)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return

        kp = extract_frame_landmarks(frame_bgr)
        buffer.append(kp)
        if len(buffer) > session["max_buffer"]:
            buffer.pop(0)

        has_detection = bool(np.isfinite(kp).any())

        # Format landmarks for browser visualization
        landmarks_for_viz = {"pose": [], "left_hand": [], "right_hand": []}
        if np.isfinite(kp[0]).all():
            landmarks_for_viz["pose"].append({"x": float(kp[0, 0]), "y": float(kp[0, 1]), "visibility": 1.0})
        for i in range(7, 28):
            if np.isfinite(kp[i]).all():
                landmarks_for_viz["left_hand"].append({"x": float(kp[i, 0]), "y": float(kp[i, 1])})
        for i in range(28, 49):
            if np.isfinite(kp[i]).all():
                landmarks_for_viz["right_hand"].append({"x": float(kp[i, 0]), "y": float(kp[i, 1])})

        prediction_result = None
        if len(buffer) >= 6 and (now - session["last_prediction_time"]) > 0.2:
            session["last_prediction_time"] = now
            window_arr = np.stack(buffer[-16:], axis=0) if len(buffer) >= 16 else np.stack(buffer, axis=0)
            prediction_result = predict_from_frames(window_arr, mode=mode)

        payload = {
            "type": "frame_result",
            "has_landmarks": has_detection,
            "landmarks": landmarks_for_viz,
            "prediction": prediction_result,
            "frame_id": session["frame_count"]
        }
        await ws.send_text(json.dumps(to_jsonable(payload)))


ws_manager = StreamConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            await ws_manager.handle_message(websocket, text)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning("WebSocket session error: %s", e)
        ws_manager.disconnect(websocket)

