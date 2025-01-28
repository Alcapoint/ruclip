FROM --platform=$BUILDPLATFORM python:3.12-slim-bullseye AS builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM --platform=$TARGETPLATFORM python:3.12-slim-bullseye

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]