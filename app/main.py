import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from .camera import CameraService
from .config import ROOT
from .detector import Detector


detector = Detector()
camera_service = CameraService(detector)
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    await camera_service.start()
    yield
    await camera_service.stop()


app = FastAPI(title="Realtime Vehicle Detection", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": detector.loaded,
        "model_path": str(detector.model_path),
        "classes": detector.classes,
        "camera": camera_service.status(),
    }


@app.post("/source/camera")
async def source_camera():
    await camera_service.use_camera()
    return {"source": "camera"}


@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    content = await video.read()
    try:
        await camera_service.use_video(content, video.filename or "upload.mp4")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"source": "upload", "filename": video.filename}


async def mjpeg_frames():
    last_frame = None
    while True:
        if camera_service.latest_jpeg and camera_service.latest_jpeg != last_frame:
            last_frame = camera_service.latest_jpeg
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + camera_service.latest_jpeg
                + b"\r\n"
            )
        await asyncio.sleep(1 / 10)


@app.get("/stream.mjpg")
async def stream():
    return StreamingResponse(
        mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket(websocket: WebSocket):
    await websocket.accept()
    async for message in camera_service.subscribe():
        await websocket.send_text(message)
