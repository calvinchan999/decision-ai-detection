# RTSP beverage drinking detection

This program samples the RTSP camera at 2 frames per second. YOLO26n checks
processed frames for a person. Only when a person is detected does the selected
AI provider check for possible drinking and for a person actively using a
running faucet or tap. Drinking includes any beverage. The tap decision requires
visible water flow and interaction with the same tap or its water; a person
merely nearby or an unattended running tap does not qualify. A still image
cannot reliably establish the exact moment someone turned a tap on. An open or
boiling kettle alone does not qualify.

Each decision has an independent yes-probability; scores strictly greater
than **0.5** trigger an upload. Drinking images go to `drinking/` and tap-use
images to the existing `boilwater/` prefix inside the configured Azure container.
If both are positive, the same JPEG goes to both paths. The local image is deleted only
after all required uploads succeed; otherwise it remains in `captures` and
the error is logged. Failed uploads are not retried automatically.

The camera loop uses the provider-neutral `FrameAnalyzer` interface in
`ai_analyzer.py`. Select `cloudflare` or `openrouter` with `AI_PROVIDER`; the
provider-specific decision requests live in `cloudflare_clef.py` and
`openrouter_decision.py`. Capture, timing, filtering, and image-saving behavior
is the same for either provider.

Before calling either provider, the app creates a JPEG for AI analysis with a
configurable maximum size (1280×720 pixels by default), preserving its aspect
ratio and without enlarging smaller frames. YOLO uses the original frame for
its own preprocessing, and positive images uploaded to Azure retain the
original capture resolution. Larger settings can increase token usage and may
exceed provider image size limits.

## Requirements

- Python 3.9 or newer
- FFmpeg with RTSP and MJPEG/image2pipe support
- Dependencies from `requirements.txt`
- Either a Cloudflare account ID and Workers AI API token or an OpenRouter API
  key

## Configure and run

Replace the placeholders in `.env`. The runner loads it from the same directory
as the script. Values set in the process environment take precedence over
`.env`. Keep `.env` private; it is excluded from Git.

- `CAMERA_RTSP_URL`: full camera RTSP URL
- `AI_PROVIDER`: `cloudflare` (default) or `openrouter`
- `AI_IMAGE_MAX_WIDTH`, `AI_IMAGE_MAX_HEIGHT`: positive integer maximum
  dimensions in pixels for images sent to either AI provider; default to `1280`
  and `720`. For example, set them to `640` and `360` for a 360p limit.
  Restart the application or recreate the Compose container after changing them.
- `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`: required when using
  `AI_PROVIDER=cloudflare`
- `OPENROUTER_API_KEY`: required when using `AI_PROVIDER=openrouter`
- `OPENROUTER_MODEL`: OpenRouter decision model; defaults to `liquid/d1.1`
- `AZURE_BLOB_CONTAINER_URL`: HTTPS URL for the container, for example
  `https://account.blob.core.windows.net/tmp-image`
- `AZURE_BLOB_SAS_TOKEN`: SAS query string with create or write permission;
  keep it private and omit the leading `?`
- `YOLO_PERSON_CONFIDENCE`: person-detection threshold; defaults to `0.20`
- `YOLO_IMAGE_SIZE`: YOLO inference size, a multiple of 32; defaults to `960`
- `YOLO_MODEL`: model path; defaults to `yolo26n.pt`
- `YOLO_DEVICE`: inference device; defaults to `cpu`
- `CAPTURE_DIR` (optional): output directory; defaults to `captures` beside the
  script
- `FFMPEG_PATH` (optional): FFmpeg executable path if it is not on `PATH`

Run from PowerShell:

```powershell
python -m pip install -r requirements.txt
python .\stream_camera.py
```

The first run downloads YOLO26n weights if they are not already cached.
Docker Compose keeps the weights in a named volume. Ultralytics is licensed
under AGPL-3.0; review its terms for your deployment.

## Run with Docker Compose

Replace the placeholders in `.env`, then start the container:

```powershell
New-Item -ItemType Directory -Force .\captures | Out-Null
docker compose up --build -d
docker compose logs -f camera-watcher
```

Positive frames are uploaded to the configured Azure container and deleted
locally after a successful upload. If an upload fails, the image remains in the
host's `captures` folder.
Stop the service with `docker compose down`. On Linux, if the container cannot
write to the captures folder, set `APP_UID` and `APP_GID` in `.env` to the host
user's numeric IDs and rebuild.

The camera is continuously decoded by FFmpeg and sampled at 2 fps without a
software resolution cap; JPEG quality is set to FFmpeg q:v 2. The latest frame
is resized for the AI provider with requests spaced at least one second apart.
Both questions share one request when a person is detected; if YOLO finds no
person or fails, the frame is not sent to the AI. YOLO misses can therefore
cause both kinds of events to be missed. Frames waiting while
analysis runs are discarded in favor of the newest frame, so short events can
still be missed when requests are slow. This increased request rate can increase
API costs. If the camera or FFmpeg disconnects, capture retries after five
seconds. Stop with Ctrl+C.

The API token and camera URL are passed to external processes or services;
run this on a trusted machine and URL-encode reserved characters in camera
usernames and passwords.

# decision-ai-detection
