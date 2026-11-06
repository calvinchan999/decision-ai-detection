import math
import os
from io import BytesIO
from pathlib import Path


class PersonFilterError(Exception):
    pass


class PersonFilter:
    def __init__(self) -> None:
        model_name = os.environ.get("YOLO_MODEL", "yolo26n.pt").strip()
        if not model_name:
            raise PersonFilterError("YOLO_MODEL cannot be empty.")

        model_path = Path(model_name).expanduser()
        if not model_path.is_absolute():
            model_path = Path(__file__).parent / model_path

        try:
            self.confidence = float(
                os.environ.get("YOLO_PERSON_CONFIDENCE", "0.25")
            )
        except ValueError:
            raise PersonFilterError(
                "YOLO_PERSON_CONFIDENCE must be a number from 0 to 1."
            ) from None
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise PersonFilterError(
                "YOLO_PERSON_CONFIDENCE must be a number from 0 to 1."
            )

        try:
            self.image_size = int(os.environ.get("YOLO_IMAGE_SIZE", "960"))
        except ValueError:
            raise PersonFilterError(
                "YOLO_IMAGE_SIZE must be a positive multiple of 32."
            ) from None
        if self.image_size < 32 or self.image_size % 32 != 0:
            raise PersonFilterError(
                "YOLO_IMAGE_SIZE must be a positive multiple of 32."
            )

        self.device = os.environ.get("YOLO_DEVICE", "cpu").strip() or "cpu"
        config_dir = Path(
            os.environ.get(
                "YOLO_CONFIG_DIR", str(Path(__file__).parent / ".ultralytics")
            )
        ).expanduser()
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise PersonFilterError(
                f"Could not create YOLO config directory: {error}"
            ) from None
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

        try:
            from ultralytics import YOLO
        except (ImportError, OSError, RuntimeError) as error:
            raise PersonFilterError(
                "Ultralytics is unavailable; install requirements.txt."
            ) from error

        try:
            self.model = YOLO(str(model_path))
        except (OSError, RuntimeError, ValueError) as error:
            raise PersonFilterError(
                f"Could not load YOLO model {model_path}: {error}"
            ) from None

        names = self.model.names
        if isinstance(names, dict):
            name_items = names.items()
        elif isinstance(names, (list, tuple)):
            name_items = enumerate(names)
        else:
            raise PersonFilterError(
                f"YOLO model {model_path} has no usable class names."
            )
        self.person_classes = [
            int(class_id)
            for class_id, name in name_items
            if str(name).strip().lower() == "person"
        ]
        if not self.person_classes:
            raise PersonFilterError(
                f"YOLO model {model_path} has no 'person' class."
            )

    def detects_person(self, image: bytes) -> bool:
        try:
            from PIL import Image

            with Image.open(BytesIO(image)) as decoded:
                frame = decoded.convert("RGB")
        except (ImportError, OSError) as error:
            raise PersonFilterError(
                f"Could not decode frame for YOLO: {error}"
            ) from None

        try:
            results = self.model.predict(
                source=frame,
                classes=self.person_classes,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise PersonFilterError(f"YOLO person detection failed: {error}") from None

        return any(
            result.boxes is not None and len(result.boxes) > 0
            for result in results
        )
