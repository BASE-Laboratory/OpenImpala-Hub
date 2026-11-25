FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    HUB_MAX_SOLVE_SIDE=128 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        gcc \
        gfortran \
        git \
        libgomp1 \
        libopenmpi-dev \
        make \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 1000 matches Hugging Face Spaces Docker & rootless best practices)
RUN useradd -m -u 1000 hubuser
WORKDIR /app

COPY pyproject.toml README.md LICENSE CITATION.cff AGENTS.md ./
COPY src ./src
COPY data ./data
COPY docs_site/data ./docs_site/data

ARG SMS_GITHUB_TOKEN=""
RUN pip install --upgrade pip \
    && pip install ".[mpi,server]" \
    && if [ -n "$SMS_GITHUB_TOKEN" ]; then \
         git config --global url."https://x-access-token:${SMS_GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/" \
         && pip install --no-deps ".[sms]" \
         && rm -f /root/.gitconfig; \
       else \
         GIT_TERMINAL_PROMPT=0 pip install --no-deps ".[sms]" || echo "Optional private [sms] extra skipped"; \
       fi \
    && chown -R hubuser:hubuser /app

USER hubuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/healthz || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180", "openimpala_hub.app.server:get_wsgi_app()"]
