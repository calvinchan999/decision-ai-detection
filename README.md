# RTSP drinking, faucet, and microwave detection

This program samples the RTSP camera at 2 frames per second. YOLO26n checks
processed frames for a person. Only when a person is detected does the selected
AI provider check for possible drinking, a person actively using a running
faucet or tap, and a person interacting with an open microwave. Microwave detection
requires a visibly open door and interaction with that microwave; an unattended
open microwave or a person merely nearby does not qualify. A single image
cannot establish the exact moment the door was opened. Drinking includes any
beverage. The faucet decision requires
visible water flow and interaction with the same tap or its water; a person
merely nearby or an unattended running tap does not qualify. A still image
cannot reliably establish the exact moment someone turned a tap on. An open or
boiling kettle alone does not qualify.

In the default `choice` mode the provider answers one exclusive question and
returns a probability for each of `drinking`, `faucet`, `microwave`,
and `other`.
A probability strictly greater than **0.5** triggers an upload; because the
options form a single distribution, at most one category can clear that bar.
With `QUESTION_MODE=binary` the three questions are independent and a frame
may be uploaded to multiple paths. In choice mode, no image is uploaded if all
category probabilities are at or below 0.5. Choice responses missing valid
probabilities are rejected rather than being treated as certain.
Drinking images go to `drinking/`, faucet images to `faucet/`,
and microwave images to `microwave/` inside the configured Azure
container. The folder for each
category is mapped in `CATEGORY_PREFIXES` in `azure_blob.py`; the faucet
detector was earlier called `boilwater`, so set that value there if the container
folder has to stay continuous with the older data set.
If multiple categories are positive, the same JPEG goes to each path. The local image is deleted only
after all required uploads succeed; otherwise it remains in `captures` and
the error is logged. Failed uploads are not retried automatically.

Because frames are sampled continuously, one event can produce several positive
frames. In choice mode, two consecutive identical choice labels (including
`other`) trigger an AI request cooldown, reducing repeat requests. The cooldown
starts when the second response arrives; it applies again after every subsequent
matching response. A different choice returns to the normal request interval.
The first request after a pause uses the latest frame, but any brief event
during the pause can be missed. Failed AI requests break the consecutive-choice
sequence. In binary mode this cooldown does not apply.
YOLO continues checking available frames during the cooldown. A frame with no
person clears the previous choice and cancels the repeated-choice cooldown,
so a later person-positive frame can be sent once the normal request interval
allows it. Camera reconnection also resets the choice and its cooldown. This
does not track identities or detect activity changes while a person stays in view.

`UPLOAD_COOLDOWN_SECONDS` separately suppresses repeated Azure uploads; it
does not reduce the number of AI requests.

The camera loop uses the provider-neutral `FrameAnalyzer` interface in
`ai_analyzer.py`. Select `cloudflare`, `openrouter`, or `llama_cpp` with
`AI_PROVIDER`; the provider-specific requests live in `cloudflare_clef.py`,
`openrouter_decision.py`, and `llama_cpp_decision.py`. Capture, timing,
filtering, and image-saving behavior is the same for all providers.

Before calling a provider, the app creates a JPEG for AI analysis with a
configurable maximum size (1280×720 pixels by default), preserving its aspect
ratio and without enlarging smaller frames. YOLO uses the original frame for
its own preprocessing, and positive images uploaded to Azure retain the
original capture resolution. Larger settings can increase token usage and may
exceed provider image size limits.

## Requirements

- Python 3.9 or newer
- FFmpeg with RTSP and MJPEG/image2pipe support
- Dependencies from `requirements.txt`
- A Cloudflare account ID and Workers AI API token, an OpenRouter API key, or
  access to a llama.cpp server running an image-capable decision model

## Configure and run

Copy `.env.sample` to `.env` and replace the placeholders. The runner loads it
from the same directory as the script. Values set in the process environment take precedence over
`.env`. Keep `.env` private; it is excluded from Git.

- `CAMERA_RTSP_URL`: full camera RTSP URL
- `AI_PROVIDER`: `cloudflare` (default), `openrouter`, or `llama_cpp`
- `MODEL`: model for the selected provider. For Cloudflare use `clef` (default
  when `MODEL` is unset) or `clef-flash`. For OpenRouter use an image-capable
  Decisions model ID such as `liquid/d1.1` (default when `MODEL` is unset).
  When switching between hosted providers, update `MODEL` to a compatible
  value. An unsupported Cloudflare model stops startup with an error.
  `llama_cpp` ignores `MODEL`; the server selects its loaded model.
- `QUESTION_MODE`: `choice` (default) or `binary`. `choice` asks one exclusive
  question with four options (`drinking`, `faucet`, `microwave`,
  `other`) and the answer
  carries a probability per option that sums to one, so one frame can never be
  uploaded to multiple folders. `binary` asks three independent yes/no questions,
  so more than one can pass the threshold
- `AI_IMAGE_MAX_WIDTH`, `AI_IMAGE_MAX_HEIGHT`: positive integer maximum
  dimensions in pixels for images sent to the AI provider; default to `1280`
  and `720`. For example, set them to `640` and `360` for a 360p limit.
  Restart the application or recreate the Compose container after changing them.
- `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`: required when using
  `AI_PROVIDER=cloudflare`
- `OPENROUTER_API_KEY`: required when using `AI_PROVIDER=openrouter`
- `LLAMA_CPP_URL`: required for `AI_PROVIDER=llama_cpp`; full HTTP(S) endpoint
  ending in `/v1/systemone`, for example `http://192.168.1.10:8080/v1/systemone`.
  The address must be reachable from the app; in Docker, `127.0.0.1` refers to
  the container, not the host. Use HTTPS or a trusted VPN for remote traffic.
  Redirects are rejected; configure the final endpoint directly.
- `LLAMA_CPP_API_KEY`: optional bearer token when the server requires one.
  Keep it private in `.env`.
- `LLAMA_CPP_TIMEOUT_SECONDS`: positive request timeout; defaults to `30`.
  Increase this if inference takes longer.
- `AZURE_BLOB_CONTAINER_URL`: HTTPS URL for the container, for example
  `https://account.blob.core.windows.net/tmp-image`
- `AZURE_BLOB_SAS_TOKEN`: SAS query string with create or write permission;
  keep it private and omit the leading `?`
- `YOLO_PERSON_CONFIDENCE`: person-detection threshold; defaults to `0.25`
- `YOLO_IMAGE_SIZE`: YOLO inference size, a multiple of 32; defaults to `960`
- `YOLO_MODEL`: model path; defaults to `yolo26n.pt`
- `YOLO_DEVICE`: inference device; defaults to `cpu`
- `CAPTURE_FPS`: frames pulled from FFmpeg per second; defaults to `2`. The
  latest available frame goes through YOLO even during AI request cooldowns
- `AI_REQUEST_INTERVAL_SECONDS`: minimum seconds between AI requests once a person
  is present; defaults to `2`. Lower it to analyse more frames, at higher cost
- `AI_SAME_CHOICE_COOLDOWN_SECONDS`: after two consecutive matching choice
  labels, wait this many seconds from the second response before the next AI
  request; defaults to `5`. Set to `0` to disable. The normal request interval
  still applies if it is longer. This setting does not affect binary mode.
- `UPLOAD_COOLDOWN_SECONDS` (optional): after a category has been uploaded,
  further uploads for that same category are suppressed for this many seconds;
  defaults to `0`, which uploads every positive frame. Categories are tracked
  independently, so a cooldown on `drinking` does not block the other categories
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

### Self-hosted d1-3B

Run a compatible llama.cpp server with its vision support and d1-3B model
loaded; the [official d1-3B instructions](https://docs.liquid.ai/lfm/models/d1-3b)
require llama.cpp 0.7.0 or later. Set `AI_PROVIDER=llama_cpp` and
`LLAMA_CPP_URL` to that server's `/v1/systemone` endpoint. Set the optional
API key if required. The server must accept a JPEG data URL in `images` and
return typed `choice`/`noul` answers with probabilities. Unsupported or malformed
responses are rejected. Verify detection quality and the 0.5 threshold against
your own camera images; different models can produce different scores. Official
documentation marks d1-3B deprecated in favor of d1-2B.

AI inference stays on your self-hosted server rather than Cloudflare or
OpenRouter, but images travel to that server and positive frames still upload
to Azure. Rebuild the Docker image after adding or changing the adapter.

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
YOLO checks available frames independently of the AI request interval. Only
eligible person-positive frames are sent to AI, so a one-second sip can fall
entirely between two AI requests and never be recognised. Slow YOLO or uploads
can still cause frames to be discarded. The repeated-choice cooldown can extend that
gap further, including when both choices are `other`.
All questions share one request when a person is detected; if YOLO finds no
person or fails, the frame is not sent to the AI. YOLO misses can therefore
cause any of these events to be missed. Frames waiting while
analysis runs are discarded in favor of the newest frame, so short events can
still be missed when requests are slow. This increased request rate can increase
API costs. If the camera or FFmpeg disconnects, capture retries after five
seconds. Stop with Ctrl+C.

The API token and camera URL are passed to external processes or services;
run this on a trusted machine and URL-encode reserved characters in camera
usernames and passwords.

# decision-ai-detection
