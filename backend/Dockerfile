# syntax=docker/dockerfile:1.7

# ---------- build Go api ----------
FROM golang:1.23-alpine AS build
WORKDIR /src
COPY . .
RUN go mod tidy
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags "-s -w" -o /out/api ./cmd/api

# ---------- runtime: python + numpy/scipy for pipeline worker ----------
# We ship the analyser as a subprocess call. Python 3.12 slim is small
# (~70 MB) and numpy+scipy binary wheels avoid build toolchain.
FROM python:3.12-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy

WORKDIR /app
COPY --from=build /out/api /app/api
COPY --from=build /src/migrations /app/migrations
COPY --from=build /src/web /app/web
# The Python analyser package (synced into backend/ by deploy.sh).
COPY --from=build /src/analysis /app/analysis
# Wave 2: pre-built track templates (.npz). Load'ятся на старте бэкенда.
COPY --from=build /src/tracks /app/tracks
ENV PYTHON_ANALYSER_PATH=/app/analysis/katahub_render.py
ENV TRACKS_DIR=/app/tracks
ENV ACTIVE_TRACK=miks

EXPOSE 8080
ENTRYPOINT ["/app/api"]
