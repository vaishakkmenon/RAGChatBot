# ============================
# Builder stage
# ============================
FROM cgr.dev/chainguard/python:latest-dev AS builder

USER root
RUN install -d -o nonroot -g nonroot /opt/venv \
    && mkdir -p /workspace && chown -R nonroot:nonroot /workspace
USER nonroot

# Virtualenv for all deps
ENV VENV=/opt/venv PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace
RUN python -m venv "$VENV"

# Install prod + dev deps
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest ruff

# Copy source
COPY app ./app
RUN mkdir -p /workspace/data/chroma /workspace/data/docs

# ============================
# Test stage
# ============================
FROM builder AS test

# Install dev/test tools into venv
RUN /opt/venv/bin/pip install --no-cache-dir pytest ruff

# Copy in tests (not needed in production)
COPY tests ./tests
COPY pytest.ini ./

# Default command for test runs
CMD ["/opt/venv/bin/pytest", "-m", "not integration", "-v"]

# ============================
# Runtime stage
# ============================
FROM cgr.dev/chainguard/python:latest

ENV VENV=/opt/venv PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace

# Bring in the venv from builder (includes pytest + ruff for CI)
COPY --from=builder /opt/venv /opt/venv

# Copy app code only
COPY --chown=nonroot:nonroot app ./app
USER nonroot

# Start the server by default
ENTRYPOINT []
CMD ["/opt/venv/bin/uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--reload"]
