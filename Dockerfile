FROM python:3.14-slim

# Tesseract is what pytesseract calls out to for OCR (scanned/photographed
# documents with no text layer): installing it here means the containerized
# deployment can do it out of the box, unlike local dev where it's optional.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runs as a non-root user; .nuru_cache still needs to be writable by it.
RUN useradd --create-home --shell /bin/bash nuru \
    && mkdir -p /app/.nuru_cache /app/archive \
    && chown -R nuru:nuru /app
USER nuru

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/healthz', timeout=3)" || exit 1

# Real config (SMTP, webhook URL, access password, etc.) goes in via
# environment variables at `docker run` time; see nuru.local.env.example.
# The waitress-served app, not Flask's dev server: see README "Running it".
CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=5000", "webapp:app"]
