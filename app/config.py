from dataclasses import dataclass
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_path: Path = ROOT / "models" / "best.pt"
    cameras_path: Path = ROOT / "config" / "cameras.json"
    confidence: float = 0.25
    image_size: int = 640
    target_fps: float = 3.0
    ffmpeg_binary: str = "ffmpeg"
    camera_timeout_seconds: int = 10


settings = Settings(
    model_path=Path(os.getenv("MODEL_PATH", str(Settings.model_path))),
    cameras_path=Path(os.getenv("CAMERAS_PATH", str(Settings.cameras_path))),
    confidence=float(os.getenv("CONFIDENCE_THRESHOLD", "0.25")),
    image_size=int(os.getenv("IMAGE_SIZE", "640")),
    target_fps=float(os.getenv("TARGET_FPS", "3")),
    ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg"),
    camera_timeout_seconds=int(os.getenv("CAMERA_TIMEOUT_SECONDS", "10")),
)


def load_cameras() -> list[dict[str, str]]:
    if not settings.cameras_path.exists():
        return []

    cameras = json.loads(settings.cameras_path.read_text())
    return [camera for camera in cameras if camera.get("id") and camera.get("url")]
