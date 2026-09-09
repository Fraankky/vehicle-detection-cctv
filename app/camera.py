import asyncio
import json
import numpy as np
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path

import cv2

from .config import load_cameras, settings
from .detector import Detector, encode_jpeg


class CameraService:
    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        self.cameras = load_cameras()
        self.active_camera: dict | None = None
        self.latest_jpeg: bytes | None = None
        self.last_frame_at = 0.0
        self.latest_stats = {
            "status": "starting",
            "camera_id": None,
            "camera_name": None,
            "fps": 0,
            "inference_ms": 0,
            "objects": {"total": 0, "classes": {}},
                "error": detector.error,
        }
        self.subscribers: set[asyncio.Queue] = set()
        self.task: asyncio.Task | None = None
        self.process: subprocess.Popen | None = None
        self.video_path: Path | None = None
        self.source = "camera"

    async def use_camera(self) -> None:
        self.source = "camera"
        self._remove_video()
        self._close_process()

    async def use_video(self, content: bytes, filename: str) -> None:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            raise ValueError("Unsupported video format")
        if len(content) > 100 * 1024 * 1024:
            raise ValueError("Video must be smaller than 100 MB")

        self._remove_video()
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
        try:
            handle.write(content)
        finally:
            handle.close()
        self.video_path = Path(handle.name)
        self.source = "upload"
        self._close_process()

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        self._close_process()
        self._remove_video()

    async def subscribe(self):
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self.subscribers.discard(queue)

    def status(self) -> dict:
        return self.latest_stats

    async def _run(self) -> None:
        if not self.cameras:
            self._update(status="no_cameras", error="No cameras configured")
            return

        while True:
            if self.source == "upload" and self.video_path:
                self._update(status="processing", camera_name="Uploaded video", error=None)
                try:
                    await self._read_video(self.video_path)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._update(status="offline", error=str(exc))
                self.source = "camera"
                self._remove_video()
                continue

            camera = self.cameras[0]
            self._update(status="connecting", error=None)
            try:
                await self._read_camera(camera)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._update(status="offline", error=str(exc))
                await asyncio.sleep(2)

    async def _read_video(self, video_path: Path) -> None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("Uploaded video could not be opened")

        try:
            while self.source == "upload":
                success, frame = await asyncio.to_thread(capture.read)
                if not success:
                    self._update(status="completed", camera_name="Uploaded video")
                    return
                annotated, objects, inference_ms = await asyncio.to_thread(
                    self.detector.predict, frame
                )
                self.latest_jpeg = encode_jpeg(annotated)
                self.last_frame_at = asyncio.get_running_loop().time()
                self._update(
                    camera_id="uploaded-video",
                    camera_name="Uploaded video",
                    status="processing",
                    fps=round(settings.target_fps, 2),
                    inference_ms=round(inference_ms, 2),
                    objects=objects,
                    error=self.detector.error,
                )
                await asyncio.sleep(max(0, 1 / settings.target_fps))
        finally:
            capture.release()

    async def _read_camera(self, camera: dict) -> None:
        self.active_camera = camera
        self.process = subprocess.Popen(
            [
                settings.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_on_network_error",
                "1",
                "-i",
                camera["url"],
                "-vf",
                f"scale={settings.image_size}:-2",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        try:
            while True:
                if not self.process.stdout:
                    raise RuntimeError("FFmpeg stdout is unavailable")
                raw = await asyncio.to_thread(self._read_jpeg)
                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("FFmpeg returned an invalid frame")
                annotated, objects, inference_ms = await asyncio.to_thread(
                    self.detector.predict, frame
                )
                self.latest_jpeg = encode_jpeg(annotated)
                self.last_frame_at = asyncio.get_running_loop().time()
                self._update(
                    camera_id=camera["id"],
                    camera_name=camera.get("name"),
                    status="online",
                    fps=round(settings.target_fps, 2),
                    inference_ms=round(inference_ms, 2),
                    objects=objects,
                    error=self.detector.error,
                )
                await asyncio.sleep(max(0, 1 / settings.target_fps))
        finally:
            self._close_process()

    def _read_jpeg(self):
        data = bytearray()
        while True:
            chunk = self.process.stdout.read(4096)
            if not chunk:
                raise RuntimeError("Stream ended while reading a frame")
            data.extend(chunk)
            start = data.find(b"\xff\xd8")
            if start < 0:
                data = data[-1:]
                continue
            end = data.find(b"\xff\xd9", start + 2)
            if end >= 0:
                return np.frombuffer(data[start : end + 2], dtype=np.uint8)
            if start:
                del data[:start]

    def _close_process(self) -> None:
        if self.process:
            self.process.kill()
            self.process.wait()
            self.process = None

    def _remove_video(self) -> None:
        if self.video_path:
            self.video_path.unlink(missing_ok=True)
            self.video_path = None

    def _update(self, **values) -> None:
        self.latest_stats.update(values)
        message = json.dumps({"type": "stats", **self.latest_stats})
        for queue in list(self.subscribers):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with suppress(asyncio.QueueFull):
                queue.put_nowait(message)
