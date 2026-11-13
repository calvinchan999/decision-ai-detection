import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

from ai_analyzer import AnalyzerError, DetectionScores, FrameAnalyzer, create_analyzer
from azure_blob import AzureBlobUploader, AzureUploadError
from person_filter import PersonFilter, PersonFilterError


POSITIVE_THRESHOLD = 0.5
DEFAULT_AI_IMAGE_MAX_SIZE = (1280, 720)
UPLOAD_COOLDOWN_ENV = "UPLOAD_COOLDOWN_SECONDS"
CAPTURE_FPS_ENV = "CAPTURE_FPS"
AI_REQUEST_INTERVAL_ENV = "AI_REQUEST_INTERVAL_SECONDS"
AI_SAME_CHOICE_COOLDOWN_ENV = "AI_SAME_CHOICE_COOLDOWN_SECONDS"
DEFAULT_CAPTURE_FPS = 2
DEFAULT_AI_REQUEST_INTERVAL_SECONDS = 2.0
DEFAULT_AI_SAME_CHOICE_COOLDOWN_SECONDS = 5.0
stop_event = threading.Event()


def redact(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def load_dotenv() -> None:
    dotenv_path = Path(__file__).with_name(".env")
    if not dotenv_path.exists():
        return

    with dotenv_path.open(encoding="utf-8") as dotenv_file:
        for line_number, line in enumerate(dotenv_file, start=1):
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue

            key, separator, value = entry.partition("=")
            key = key.strip()
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid .env entry on line {line_number}.")

            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


def get_ai_image_max_size() -> tuple[int, int]:
    dimensions = []
    for name, default in zip(
        ("AI_IMAGE_MAX_WIDTH", "AI_IMAGE_MAX_HEIGHT"), DEFAULT_AI_IMAGE_MAX_SIZE
    ):
        try:
            value = int(os.environ.get(name, str(default)))
        except ValueError:
            raise ValueError(f"{name} must be a positive integer.") from None
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        dimensions.append(value)
    return dimensions[0], dimensions[1]


def get_capture_fps() -> int:
    raw = os.environ.get(CAPTURE_FPS_ENV, str(DEFAULT_CAPTURE_FPS)).strip()
    try:
        fps = int(raw or DEFAULT_CAPTURE_FPS)
    except ValueError:
        raise ValueError(f"{CAPTURE_FPS_ENV} must be a positive integer.") from None
    if fps <= 0:
        raise ValueError(f"{CAPTURE_FPS_ENV} must be a positive integer.")
    return fps


def get_ai_request_interval() -> float:
    raw = os.environ.get(
        AI_REQUEST_INTERVAL_ENV, str(DEFAULT_AI_REQUEST_INTERVAL_SECONDS)
    ).strip()
    try:
        seconds = float(raw or DEFAULT_AI_REQUEST_INTERVAL_SECONDS)
    except ValueError:
        raise ValueError(
            f"{AI_REQUEST_INTERVAL_ENV} must be a positive number of seconds."
        ) from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(
            f"{AI_REQUEST_INTERVAL_ENV} must be a positive number of seconds."
        )
    return seconds


def get_upload_cooldown() -> float:
    raw = os.environ.get(UPLOAD_COOLDOWN_ENV, "0").strip() or "0"
    try:
        seconds = float(raw)
    except ValueError:
        raise ValueError(f"{UPLOAD_COOLDOWN_ENV} must be a number of seconds.") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            f"{UPLOAD_COOLDOWN_ENV} must be a non-negative number of seconds."
        )
    return seconds


def get_same_choice_cooldown() -> float:
    raw = os.environ.get(
        AI_SAME_CHOICE_COOLDOWN_ENV, str(DEFAULT_AI_SAME_CHOICE_COOLDOWN_SECONDS)
    ).strip()
    try:
        seconds = float(raw)
    except ValueError:
        raise ValueError(
            f"{AI_SAME_CHOICE_COOLDOWN_ENV} must be a non-negative number of seconds."
        ) from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(
            f"{AI_SAME_CHOICE_COOLDOWN_ENV} must be a non-negative number of seconds."
        )
    return seconds


def iter_jpeg_frames(stream):
    buffer = bytearray()
    while True:
        chunk = stream.read1(65536)
        if not chunk:
            return
        buffer.extend(chunk)

        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                if buffer:
                    del buffer[:-1]
                break
            if start:
                del buffer[:start]

            end = buffer.find(b"\xff\xd9", 2)
            if end < 0:
                break

            yield bytes(buffer[: end + 2])
            del buffer[: end + 2]


def capture_frames(stream, frames: queue.Queue) -> None:
    for frame in iter_jpeg_frames(stream):
        while True:
            try:
                frames.put_nowait(frame)
                break
            except queue.Full:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass


def read_stderr(stream, secrets: tuple[str, ...]) -> None:
    for line in stream:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        print(redact(line.rstrip(), secrets), file=sys.stderr, flush=True)


def analyze_captured_frame(
    analyzer: FrameAnalyzer,
    image: bytes,
    person_present: bool,
    max_size: tuple[int, int] = DEFAULT_AI_IMAGE_MAX_SIZE,
) -> DetectionScores:
    try:
        with Image.open(BytesIO(image)) as frame:
            if frame.width <= max_size[0] and frame.height <= max_size[1]:
                ai_image = image
            else:
                resized = frame.convert("RGB")
                resized.thumbnail(max_size, Image.Resampling.LANCZOS)
                output = BytesIO()
                resized.save(output, format="JPEG", quality=90)
                ai_image = output.getvalue()
    except (OSError, ValueError) as error:
        raise AnalyzerError(f"Could not prepare AI image: {error}") from error
    return analyzer.analyze_frame(ai_image, person_present)


def save_image(image: bytes, capture_dir: Path) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    image_path = capture_dir / f"detection_{timestamp}.jpg"
    image_path.write_bytes(image)
    return image_path


def handle_detections(
    image: bytes,
    scores: DetectionScores,
    capture_dir: Path,
    uploader: AzureBlobUploader,
    cooldown: float = 0.0,
    last_upload_at: Optional[dict[str, float]] = None,
) -> None:
    if scores.drinking is not None:
        print(f"Drinking score: {scores.drinking:.3f}", flush=True)
    print(f"Faucet score: {scores.faucet:.3f}", flush=True)
    print(f"Microwave score: {scores.microwave:.3f}", flush=True)
    categories = []
    if scores.drinking is not None and scores.drinking > POSITIVE_THRESHOLD:
        categories.append("drinking")
    if scores.faucet > POSITIVE_THRESHOLD:
        categories.append("faucet")
    if scores.microwave > POSITIVE_THRESHOLD:
        categories.append("microwave")
    if not categories:
        return

    if cooldown > 0 and last_upload_at is not None:
        now = time.monotonic()
        suppressed = [
            category
            for category in categories
            if now - last_upload_at.get(category, now - cooldown) < cooldown
        ]
        categories = [c for c in categories if c not in suppressed]
        if suppressed:
            print(
                f"Cooldown suppressed {', '.join(suppressed)} "
                f"({cooldown:g}s without a new upload).",
                flush=True,
            )
        if not categories:
            return

    try:
        image_path = save_image(image, capture_dir)
    except OSError as error:
        print(f"Could not save positive frame: {error}", file=sys.stderr, flush=True)
        return
    try:
        uploader.upload_then_delete(image_path, categories)
    except AzureUploadError as error:
        print(
            f"Azure upload or cleanup failed; local image kept at "
            f"{image_path}: {error}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"Uploaded {image_path.name} to Azure categories "
            f"{', '.join(categories)} and deleted the local copy.",
            flush=True,
        )
        if last_upload_at is not None:
            uploaded_at = time.monotonic()
            for category in categories:
                last_upload_at[category] = uploaded_at


def run() -> int:
    try:
        load_dotenv()
    except (OSError, ValueError) as error:
        print(f"Could not read .env: {error}", file=sys.stderr)
        return 2

    try:
        ai_image_max_size = get_ai_image_max_size()
    except ValueError as error:
        print(f"Invalid AI image configuration: {error}", file=sys.stderr)
        return 2

    try:
        upload_cooldown = get_upload_cooldown()
    except ValueError as error:
        print(f"Invalid upload cooldown: {error}", file=sys.stderr)
        return 2

    try:
        capture_fps = get_capture_fps()
        ai_request_interval = get_ai_request_interval()
        same_choice_cooldown = get_same_choice_cooldown()
    except ValueError as error:
        print(f"Invalid capture configuration: {error}", file=sys.stderr)
        return 2

    camera_url = os.environ.get("CAMERA_RTSP_URL", "").strip()
    ffmpeg = os.environ.get("FFMPEG_PATH", "ffmpeg").strip()
    capture_dir = Path(
        os.environ.get("CAPTURE_DIR", "captures").strip() or "captures"
    ).expanduser()
    if not capture_dir.is_absolute():
        capture_dir = Path(__file__).parent / capture_dir

    if not camera_url or camera_url.startswith("rtsp://REPLACE_"):
        print("Set CAMERA_RTSP_URL in .env to the camera's RTSP URL.", file=sys.stderr)
        return 2
    if not camera_url.lower().startswith(("rtsp://", "rtsps://")):
        print("CAMERA_RTSP_URL must use rtsp:// or rtsps://.", file=sys.stderr)
        return 2
    if not shutil.which(ffmpeg):
        print(
            "FFmpeg was not found. Install it or set FFMPEG_PATH to its executable.",
            file=sys.stderr,
        )
        return 2
    try:
        analyzer = create_analyzer()
    except AnalyzerError as error:
        print(f"Could not configure AI analyzer: {error}", file=sys.stderr)
        return 2
    try:
        azure_uploader = AzureBlobUploader()
    except AzureUploadError as error:
        print(f"Could not configure Azure Blob uploader: {error}", file=sys.stderr)
        return 2

    try:
        person_filter = PersonFilter()
    except PersonFilterError as error:
        print(f"Could not configure YOLO person filter: {error}", file=sys.stderr)
        return 2

    def request_stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        f"Frames sampled at {capture_fps} fps; YOLO gates drinking, faucet, "
        f"and microwave detection. AI image maximum is {ai_image_max_size[0]}x"
        f"{ai_image_max_size[1]}. AI requests "
        f"are spaced at least {ai_request_interval:g} seconds apart. "
        f"Repeated choice cooldown: {same_choice_cooldown:g} seconds. "
        + (
            f"Repeat uploads per category are suppressed for "
            f"{upload_cooldown:g} seconds. "
            if upload_cooldown > 0
            else "Every positive frame is uploaded. "
        )
        + "Press Ctrl+C to stop.",
        flush=True,
    )

    upload_state: dict[str, float] = {}
    next_request_at = 0.0
    choice_cooldown_until = 0.0
    previous_choice: Optional[str] = None
    while not stop_event.is_set():
        command = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "warning",
            "-rtsp_transport",
            "tcp",
            "-i",
            camera_url,
            "-an",
            "-vf",
            f"fps={capture_fps}/1",
            "-q:v",
            "2",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            print(f"Could not start FFmpeg: {error}", file=sys.stderr)
            return 1

        frames = queue.Queue(maxsize=1)
        frame_thread = threading.Thread(
            target=capture_frames, args=(process.stdout, frames), daemon=True
        )
        log_thread = threading.Thread(
            target=read_stderr,
            args=(process.stderr, (camera_url,)),
            daemon=True,
        )
        frame_thread.start()
        log_thread.start()
        previous_choice = None
        choice_cooldown_until = 0.0

        while not stop_event.is_set() and process.poll() is None:
            try:
                image = frames.get(timeout=0.5)
            except queue.Empty:
                continue

            while True:
                try:
                    image = frames.get_nowait()
                except queue.Empty:
                    break

            try:
                person_present = person_filter.detects_person(image)
            except PersonFilterError as error:
                print(
                    f"YOLO pre-filter failed; skipping frame: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if not person_present:
                previous_choice = None
                choice_cooldown_until = 0.0
                continue

            now = time.monotonic()
            if now < max(next_request_at, choice_cooldown_until):
                continue
            next_request_at = now + ai_request_interval
            try:
                scores = analyze_captured_frame(
                    analyzer, image, person_present, ai_image_max_size
                )
            except AnalyzerError as error:
                previous_choice = None
                print(f"Frame analysis failed: {error}", file=sys.stderr, flush=True)
            else:
                if (
                    scores.choice is not None
                    and scores.choice == previous_choice
                    and same_choice_cooldown > 0
                ):
                    choice_cooldown_until = time.monotonic() + same_choice_cooldown
                    print(
                        f"Repeated choice '{scores.choice}'; pausing AI requests "
                        f"for at least {same_choice_cooldown:g} seconds.",
                        flush=True,
                    )
                previous_choice = scores.choice
                handle_detections(
                    image,
                    scores,
                    capture_dir,
                    azure_uploader,
                    upload_cooldown,
                    upload_state,
                )

        if stop_event.is_set() and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        else:
            process.wait()

        frame_thread.join(timeout=1)
        log_thread.join(timeout=1)
        if stop_event.is_set():
            break

        print(
            f"FFmpeg exited with code {process.returncode}; retrying in "
            "5 seconds.",
            file=sys.stderr,
            flush=True,
        )
        stop_event.wait(5)

    print("Stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
