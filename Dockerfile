# ============================
# Builder stage
# ============================
FROM cgr.dev/chainguard/python:latest-dev AS builder

USER root
RUN install -d -o nonroot -g nonroot /opt/venv \
    && mkdir -p /workspace && chown -R nonroot:nonroot /workspace
USER nonroot

ENV VENV=/opt/venv PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace
RUN python -m venv "$VENV"

# Install ONLY production deps here (keep it clean)
COPY requirements.txt .
RUN /opt/venv/bin/python -m pip install --upgrade pip wheel setuptools \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app ./app
RUN mkdir -p /workspace/data/chroma /workspace/data/docs

# ============================
# Test stage
# ============================
FROM builder AS test

# Ensure dev tools are installed into THE SAME venv used to run tests
RUN /opt/venv/bin/python -m pip install --no-cache-dir pytest==8.4.1 ruff==0.12.9

# Bring tests and pytest.ini into the image
COPY tests ./tests
COPY pytest.ini ./

# Run pytest with the venv's interpreter (most reliable)
CMD ["/opt/venv/bin/python", "-m", "pytest", "-m", "not integration", "-v"]

# ============================
# Runtime stage
# ============================
FROM cgr.dev/chainguard/python:latest

ENV VENV=/opt/venv PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace

# Bring in the venv with ONLY prod deps from builder
COPY --from=builder /opt/venv /opt/venv
# Copy app code only
COPY --chown=nonroot:nonroot app ./app

USER nonroot
ENTRYPOINT []
CMD ["/opt/venv/bin/uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--reload"]