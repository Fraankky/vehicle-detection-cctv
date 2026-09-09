from collections import Counter
import time

import cv2

from .config import settings


class Detector:
    def __init__(self) -> None:
        self.model = None
        self.error: str | None = None
        self.model_path = settings.model_path

        if not settings.model_path.exists():
            self.error = f"Model not found: {settings.model_path}"
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(settings.model_path))
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            self.error = f"Model could not be loaded: {exc}"

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def classes(self) -> dict:
        if not self.model:
            return {}
        return dict(self.model.names)

    def predict(self, frame):
        if not self.model:
            return frame, {"total": 0, "classes": {}}, 0.0

        started = time.perf_counter()
        result = self.model.predict(
            source=frame,
            imgsz=settings.image_size,
            conf=settings.confidence,
            verbose=False,
        )[0]
        annotated = result.plot()
        names = result.names
        class_ids = result.boxes.cls.tolist() if result.boxes else []
        classes = Counter(names[int(class_id)] for class_id in class_ids)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return annotated, {
            "total": sum(classes.values()),
            "classes": dict(classes),
        }, elapsed_ms


def encode_jpeg(frame) -> bytes:
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        raise RuntimeError("Could not encode detection frame")
    return encoded.tobytes()
