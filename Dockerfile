# syntax=docker/dockerfile:1
#
# CPU-only image. ffmpeg/ffprobe are a hard runtime requirement (see
# whisper_batch_subtitles/cli.py::_validate_tooling) so they're installed at the OS level.
#
# NOT VALIDATED: this image has not been built or run anywhere -- there was no Docker
# daemon available in the environment that wrote it. Treat it as a solid first draft to
# build and shake out, not a proven artifact. Basic things worth checking on first build:
# pip resolving faster-whisper/ctranslate2 cleanly on this base image's Python/glibc, and
# that ffmpeg from Debian's apt repo is recent enough for your media.
#
# For GPU/CUDA support, swap the base image for an nvidia/cuda devel or runtime image
# matching the CUDA version your ctranslate2/faster-whisper build expects, keep the same
# apt-get ffmpeg install, and run the container with `--gpus all` (requires the
# NVIDIA Container Toolkit on the host). CPU inference works fine as shipped below.

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY whisper_batch_subtitles ./whisper_batch_subtitles
COPY whispertranscribetranslate.py ./

RUN pip install --no-cache-dir .

# Mount your media library and persistent state/cache here at container run time, e.g.:
#   docker run --rm -v ./process:/data/process -v ./state:/data/state wbs run
WORKDIR /data
VOLUME ["/data/process", "/data/state"]

ENTRYPOINT ["whisper-batch-subtitles"]
CMD ["run", "--root-dir", "process", "--state-dir", "state"]
