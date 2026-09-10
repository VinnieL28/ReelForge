# syntax=docker/dockerfile:1
#
# ReelForge Studio -- production image.
#
#   docker build -t reelforge .
#   docker run -p 8501:8501 --env-file .env -v reelforge-exports:/app/exports reelforge
#
# Two things are deliberate. ffmpeg comes from Debian rather than from the
# bundled imageio-ffmpeg binary, because burning .ass captions needs libass and
# Debian's build has it. And exports/ plus users.json are volumes: renders and
# accounts must outlive the container, and neither belongs in an image layer.

FROM python:3.11-slim AS base

# IMAGEIO_FFMPEG_EXE points moviepy and imageio at the Debian ffmpeg installed
# below rather than the wheel's bundled copy, which is the build that has
# libass and can therefore burn .ass captions.
#
# REELFORGE_EXPORTS_DIR and REELFORGE_USERS_FILE both point at volume mounts,
# so renders and accounts survive a rebuild instead of dying with the layer.
#
# (Comments cannot sit between continuation lines of an ENV, so they live here.)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    REELFORGE_EXPORTS_DIR=/app/exports \
    REELFORGE_USERS_FILE=/app/userdata/users.json

# ---------------------------------------------------------------------------
# System packages
#
# The font packages matter more than they look: PIL resolves a bare
# "arialbd.ttf" through the Windows font directory and nowhere else, so without
# a real TTF on the image every caption silently renders in an 11px bitmap
# face. dejavu / liberation / freefont are all in Debian stable and cover it.
# Montserrat and Inter are nicer but are not in every mirror, so they are
# best-effort -- the build must not fail over a font.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        fonts-freefont-ttf \
        fonts-dejavu-core \
        fonts-liberation \
        ca-certificates \
        curl \
    && (apt-get install -y --no-install-recommends fonts-montserrat fonts-inter || true) \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies in their own layer so a code change does not reinstall them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---------------------------------------------------------------------------
# Run as a non-root user.
#
# The app writes to exports/ and users.json, so those are chowned; nothing else
# in /app needs to be writable.
# ---------------------------------------------------------------------------
RUN useradd --create-home --uid 10001 reelforge \
    && mkdir -p /app/exports /app/userdata \
    && chown -R reelforge:reelforge /app/exports /app/userdata

USER reelforge

EXPOSE 8501

# Streamlit's own endpoint, so this reports the app being up rather than the
# port being open. start-period covers dependency import, which is a few
# seconds of moviepy and google-genai.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--server.enableXsrfProtection=true", \
            "--server.enableCORS=false", \
            "--server.maxUploadSize=512", \
            "--browser.gatherUsageStats=false"]
