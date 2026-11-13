FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG APP_UID=1000
ARG APP_GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=${APP_UID}:${APP_GID} ai_analyzer.py azure_blob.py cloudflare_clef.py openrouter_decision.py llama_cpp_decision.py person_filter.py stream_camera.py ./

RUN mkdir -p /app/captures /app/models/.ultralytics \
    && chown -R "${APP_UID}:${APP_GID}" /app

ENV YOLO_CONFIG_DIR=/app/models/.ultralytics

USER ${APP_UID}:${APP_GID}

CMD ["python", "stream_camera.py"]
