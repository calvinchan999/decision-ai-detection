# RTSP drinking and faucet detection

This program samples the RTSP camera at 2 frames per second. YOLO26n checks
processed frames for a person. Only when a person is detected does the selected
AI provider check for possible drinking and for a person actively using a
running faucet or tap. Drinking includes any beverage. The faucet decision requires
visible water flow and interaction with the same tap or its water; a person
merely nearby or an unattended running tap does not qualify. A still image
cannot reliably establish the exact moment someone turned a tap on. An open or
boiling kettle alone does not qualify.

In the default `choice` mode the provider answers one exclusive question and
returns a probability for each of `drinking`, `faucet`, and `other`. A probability
strictly greater than **0.5** triggers an upload; because the three options form a
single distribution, at most one of them can clear that bar, so one frame never
lands in both folders. With `QUESTION_MODE=binary` the two questions are
independent and a frame may be uploaded to both paths. Note that 0.5 is a stricter
gate in `choice` mode than on two independent questions: when `drinking` and
`faucet` both sit near 0.45, `other` wins and nothing is uploaded.
Drinking images go to `drinking/` and faucet images to `faucet/` inside the
configured Azure container. The folder for each
category is mapped in `CATEGORY_PREFIXES` in `azure_blob.py`; the faucet
detector was earlier called `boilwater`, so set that value there if the container
folder has to stay continuous with the older data set.
If both are positive, the same JPEG goes to both paths. The local image is deleted only
after all required uploads succeed; otherwise it remains in `captures` and
the error is logged. Failed uploads are not retried automatically.

Because frames are sampled continuously, one event produces several positive
frames: at the default one request per second, a three-second drink yields about
three uploads. That repetition comes from sampling, not from asking two
questions, so it is controlled by `UPLOAD_COOLDOWN_SECONDS` rather than by the
number of questions.

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

Copy `.env.sample` to `.env` and replace the placeholders. The runner loads it
from the same directory as the script. Values set in the process environment take precedence over
`.env`. Keep `.env` private; it is excluded from Git.

- `CAMERA_RTSP_URL`: full camera RTSP URL
- `AI_PROVIDER`: `cloudflare` (default) or `openrouter`
- `QUESTION_MODE`: `choice` (default) or `binary`. `choice` asks one exclusive
  question with three options (`drinking`, `faucet`, `other`) and the answer
  carries a probability per option that sums to one, so one frame can never be
  uploaded to both folders. `binary` asks two independent yes/no questions and
  keeps the older behaviour, including the case where both pass the threshold
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
- `YOLO_PERSON_CONFIDENCE`: person-detection threshold; defaults to `0.25`
- `YOLO_IMAGE_SIZE`: YOLO inference size, a multiple of 32; defaults to `960`
- `YOLO_MODEL`: model path; defaults to `yolo26n.pt`
- `YOLO_DEVICE`: inference device; defaults to `cpu`
- `CAPTURE_FPS`: frames pulled from FFmpeg per second; defaults to `2`. Every
  frame goes through the YOLO person gate
- `AI_REQUEST_INTERVAL_SECONDS`: minimum seconds between AI requests once a person
  is present; defaults to `2`. Lower it to analyse more frames, at higher cost
- `UPLOAD_COOLDOWN_SECONDS` (optional): after a category has been uploaded,
  further uploads for that same category are suppressed for this many seconds;
  defaults to `0`, which uploads every positive frame. The two categories are
  tracked independently, so a cooldown on `drinking` does not block `faucet`
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

Copy `.env.sample` to `.env`, replace the placeholders, then start the
container:

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

The camera is continuously decoded by FFmpeg and sampled at `CAPTURE_FPS` frames
per second (2 by default) without a software resolution cap; JPEG quality is set
to FFmpeg q:v 2. The latest frame is resized for the AI provider with requests
spaced at least `AI_REQUEST_INTERVAL_SECONDS` apart (2 seconds by default). With
2 fps sampling and a 2 second interval, only the newest frame is taken per
interval: while a person is in view the other three frames are discarded before
the YOLO gate runs, so a one-second sip can fall entirely between two analysed
frames and never be seen at all.
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
